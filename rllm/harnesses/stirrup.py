"""Run Stirrup inside an rLLM-managed sandbox, under AA's GDPval-AA contract.

Stirrup normally provisions its own Docker or E2B environment.  This harness
instead starts Stirrup inside the task sandbox, so rLLM keeps ownership of
sandbox creation, teardown, tracing and artifact collection while the agent
code stays stock.

Reproducing Artificial Analysis' GDPval-AA v2 runtime needs three things that
Stirrup's shipped local backend cannot provide:

* **Whole-sandbox filesystem.** ``LocalCodeExecToolProvider`` confines commands
  to a private temp directory and rejects any command mentioning ``/home``,
  ``/tmp`` or ``~`` — the exact paths AA's prompt tells the model to use. The
  driver therefore supplies its own :class:`CodeExecToolProvider` rooted at
  ``/home/user``, which is what an E2B sandbox looks like to the agent.
* **AA's finish contract.** Stirrup's default ``finish`` takes a ``reason`` and
  validates only existence. AA's takes a summary plus *absolute* paths, and has
  a sibling ``abandon_task_finish``. Both are defined here.
* **Non-root identity.** The solver runs as ``user`` (UID 1000), the identity
  AA's prompt promises, rather than as root.

Everything this harness records is provenance for a later grading stage. It
does not score, rank, or compare anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any

from rllm import paths
from rllm.data import gdpval_aa
from rllm.data.gdpval_builder import RUN_METADATA_PATH, SUBMISSION_DIR
from rllm.env import env_int
from rllm.harnesses.cli_harness import BaseCliHarness
from rllm.sandbox.protocol import Sandbox
from rllm.types import AgentConfig, Episode, Task, Trajectory

logger = logging.getLogger(__name__)

#: Pinned so a Stirrup release cannot silently change the runtime contract.
#: Recorded in every submission manifest.
#:
#: 0.2.0 is the floor for any strictly-validating provider: through 0.1.12,
#: ``to_openai_messages`` dumped Stirrup's internal ToolCall model and then
#: layered the OpenAI fields on top, so every tool call also carried
#: ``tool_call_id``/``arguments``/``signature``. OpenAI and OpenRouter ignore
#: the extras; Fireworks rejects the request with "Extra inputs are not
#: permitted". 0.2.0 builds the tool-call payload explicitly.
STIRRUP_VERSION = "0.2.0"

_VENV_DIR = "/opt/stirrup-venv"
_UV_PYTHON_DIR = "/opt/uv-python"
_CONFIG_DIR = "/opt/gdpval-aa"
_DRIVER_PATH = f"{_CONFIG_DIR}/driver.py"
_SYSTEM_PROMPT_PATH = f"{_CONFIG_DIR}/system_prompt.txt"
_INSTRUCTION_PATH = f"{_CONFIG_DIR}/instruction.txt"

_INSTALL_SCRIPT = rf"""
set -e
export DEBIAN_FRONTEND=noninteractive
if [ -f {_VENV_DIR}/.stirrup-ready ]; then
    exit 0
fi
if ! command -v curl >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq && apt-get install -y -qq curl ca-certificates
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache curl ca-certificates bash
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh
fi
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
# The install runs as root but the solver runs as a non-root user, so a
# uv-managed interpreter must not land under root's home (mode 700) — the
# venv's python symlink would be unusable. Accept any 3.12+ interpreter so an
# image that already ships one is reused instead of downloading another.
export UV_PYTHON_INSTALL_DIR={_UV_PYTHON_DIR}
uv venv --python '>=3.12' {_VENV_DIR}
uv pip install --python {_VENV_DIR}/bin/python stirrup=={STIRRUP_VERSION}
chmod -R a+rX {_VENV_DIR} {_UV_PYTHON_DIR} 2>/dev/null || true
touch {_VENV_DIR}/.stirrup-ready
"""


class StirrupHarness(BaseCliHarness):
    """Run the stock Stirrup agent in the task's existing sandbox."""

    name = "stirrup"
    sandbox_backend = "docker"
    stdout_log_path = "/tmp/stirrup.log"
    run_timeout = 14_400

    max_turns: int = gdpval_aa.AA_MAX_TURNS
    shell_timeout: int = gdpval_aa.AA_SHELL_TIMEOUT_SEC
    context_summarization_cutoff: float = gdpval_aa.AA_CONTEXT_SUMMARIZATION_CUTOFF
    # AA publishes no token budgets, so these follow Stirrup's own defaults and
    # are overridable per model. Setting the output cap too low is not a soft
    # truncation: Stirrup raises OutputTokenLimitError and the run dies, which
    # 16k reliably does for a reasoning model mid-tool-call.
    max_output_tokens: int = env_int("RLLM_STIRRUP_MAX_OUTPUT_TOKENS", 64_000)
    # Drives the 70% compaction threshold, so it should match the model's real
    # context window rather than being left at a generic default.
    max_context_tokens: int = env_int("RLLM_STIRRUP_MAX_CONTEXT_TOKENS", 200_000)
    enable_web: bool = True
    # AA exposes View Image only to vision-capable models. rLLM has no vision
    # capability metadata and a model slug does not imply it, so this is an
    # operator switch: set RLLM_STIRRUP_ENABLE_VISION=0 for a text-only model.
    # Offering the tool to one that cannot accept images fails the run — the
    # provider rejects the image content block mid-trajectory.
    enable_vision: bool = env_int("RLLM_STIRRUP_ENABLE_VISION", 1) == 1

    def install_script(self) -> str:
        return _INSTALL_SCRIPT

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        workdir = str(task.metadata.get("workdir") or gdpval_aa.AA_WORKDIR)
        reasoning_effort = config.sampling_params.get("reasoning_effort")

        env = {
            "OPENAI_BASE_URL": config.base_url,
            "OPENAI_API_KEY": self.gateway_api_key(config, "OPENAI_API_KEY"),
            "RLLM_STIRRUP_MODEL": config.model,
            "RLLM_STIRRUP_WORKDIR": workdir,
            "RLLM_STIRRUP_SUBMISSION_DIR": SUBMISSION_DIR,
            "RLLM_STIRRUP_RUN_METADATA_PATH": RUN_METADATA_PATH,
            "RLLM_STIRRUP_SUBMITTABLE_ROOTS": json.dumps(list(gdpval_aa.AA_SUBMITTABLE_ROOTS)),
            "RLLM_STIRRUP_MAX_TURNS": str(self.max_turns),
            "RLLM_STIRRUP_SHELL_TIMEOUT": str(self.shell_timeout),
            "RLLM_STIRRUP_CONTEXT_CUTOFF": str(self.context_summarization_cutoff),
            "RLLM_STIRRUP_MAX_OUTPUT_TOKENS": str(self.max_output_tokens),
            "RLLM_STIRRUP_MAX_CONTEXT_TOKENS": str(self.max_context_tokens),
            "RLLM_STIRRUP_ENABLE_WEB": "1" if self.enable_web else "0",
            "RLLM_STIRRUP_ENABLE_VISION": "1" if self.enable_vision else "0",
            "RLLM_STIRRUP_SYSTEM_PROMPT_PATH": _SYSTEM_PROMPT_PATH,
            "RLLM_STIRRUP_INSTRUCTION_PATH": _INSTRUCTION_PATH,
        }
        if reasoning_effort is not None:
            if not isinstance(reasoning_effort, str):
                raise ValueError("reasoning_effort sampling parameter must be a string")
            env["RLLM_STIRRUP_REASONING_EFFORT"] = reasoning_effort

        # Stirrup consumes this key itself. Agent shell commands do not inherit
        # the parent environment because Agent.share_parent_exec_env defaults
        # to False.
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            env["BRAVE_API_KEY"] = brave_key
        return env

    def build_invocation(self, instruction: str, task: Task, config: AgentConfig) -> str:
        del instruction, task, config
        return f"{_VENV_DIR}/bin/python {_DRIVER_PATH} 2>&1 | tee {shlex.quote(self.stdout_log_path)}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, task: Task, config: AgentConfig, *, env: Sandbox) -> Episode:
        """Run Stirrup as the task's agent user, then preserve its submission.

        ``BaseCliHarness.run`` is not reused: it execs as the class-level
        ``agent_user``, but the solver identity is a per-task property
        (``[agent] user`` in task.toml) and the flow instance is shared across
        concurrent rollouts, so it cannot be stashed on ``self``.
        """
        sandbox = env
        agent_user = task.metadata.get("agent_user") or self.agent_user
        env_vars = self.build_env(task, config)

        # Config files are written as root: /opt is not writable by the
        # solver, and the solver only ever reads them.
        self._exec_agent(sandbox, self._heredoc_write(_DRIVER_PATH, _DRIVER_SCRIPT), env=env_vars)
        self._exec_agent(sandbox, self._heredoc_write(_SYSTEM_PROMPT_PATH, gdpval_aa.AA_GDPVAL_SYSTEM_PROMPT), env=env_vars)
        self._exec_agent(sandbox, self._heredoc_write(_INSTRUCTION_PATH, str(task.instruction)), env=env_vars)
        self._exec_agent(sandbox, f"chmod -R a+rX {shlex.quote(_CONFIG_DIR)}", env=env_vars)

        timeout = float(task.metadata.get("agent_timeout", self.run_timeout))
        try:
            self._exec_agent(sandbox, self.build_invocation("", task, config), timeout=timeout, env=env_vars, user=agent_user)
        except Exception as e:
            # The submission bundle and run metadata may still exist (e.g. the
            # driver finished but the tee pipe failed), so keep collecting.
            logger.warning("%s execution failed: %s", type(self).__name__, e)

        run_data = self._read_run_data(sandbox)
        metrics = _usage_metrics(run_data, config.model)
        artifacts = self._collect_submission(sandbox, task, config, run_data, metrics)
        return Episode(task=task.metadata, trajectories=[Trajectory(name=self.name, steps=[])], metrics=metrics, artifacts=artifacts)

    @staticmethod
    def _read_run_data(sandbox: Sandbox) -> dict[str, Any]:
        try:
            raw = sandbox.exec(f"cat {shlex.quote(RUN_METADATA_PATH)}", user="root")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.debug("No GDPval run metadata at %s", RUN_METADATA_PATH, exc_info=True)
            return {}

    def _collect_submission(
        self,
        sandbox: Sandbox,
        task: Task,
        config: AgentConfig,
        run_data: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Download the submission bundle and write the arena submission record.

        Runs before the sandbox is torn down. The bundle is staged in-sandbox
        by the driver, so this is a single directory download regardless of
        where the solver actually wrote its files.

        """
        safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.session_uid).strip("._") or "run"
        local_dir = Path(paths.rllm_path("agent_outputs", safe_uid))

        downloaded: list[str] = []
        download = getattr(sandbox, "download_dir", None)
        if callable(download):
            try:
                downloaded = [str(path) for path in download(SUBMISSION_DIR, str(local_dir))]
            except Exception:
                logger.warning("Could not download the GDPval submission bundle from %s", SUBMISSION_DIR, exc_info=True)

        manifest = self._write_manifest(local_dir, task, config, run_data, metrics)
        termination = manifest["termination"]
        return {
            "submission_dir": str(local_dir) if downloaded else None,
            "submission_manifest": str(local_dir / "submission_manifest.json"),
            "deliverables": [entry["local_path"] for entry in manifest["artifacts"]],
            "submitted_paths": list(termination.get("submitted_paths") or []),
            "remote_submission_dir": SUBMISSION_DIR,
        }

    def _write_manifest(
        self,
        local_dir: Path,
        task: Task,
        config: AgentConfig,
        run_data: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge sandbox-side and host-side records into one immutable manifest.

        Always written, including when the run produced nothing: a crashed or
        timed-out solver is a fact the corpus needs to record, and a missing
        file is indistinguishable from a task that was never run.
        """
        provenance = _task_provenance(task)
        sandbox_manifest = _read_json(local_dir / "manifest.json")
        termination = run_data.get("termination") if isinstance(run_data.get("termination"), dict) else {}

        artifacts = []
        for entry in sandbox_manifest.get("artifacts") or []:
            local_path = local_dir / "files" / str(entry.get("bundle_path") or "").removeprefix("files/")
            if not local_path.is_file():
                logger.warning("Submitted file %s was not preserved locally", entry.get("submitted_path"))
                continue
            artifacts.append(
                {
                    "submitted_path": entry.get("submitted_path"),
                    "local_path": str(local_path),
                    "sha256": _sha256_file(local_path),
                    "size_bytes": local_path.stat().st_size,
                    "sandbox_sha256": entry.get("sha256"),
                }
            )

        manifest = {
            "schema_version": 1,
            "benchmark": "gdpval",
            "methodology": "GDPval-AA v2",
            "stage": "solver_generation",
            "graded": False,
            "task_id": provenance.get("task_id") or task.id,
            "solver_model": config.model,
            "run_id": config.session_uid,
            "dataset_repo": provenance.get("dataset_repo"),
            "dataset_revision": provenance.get("dataset_revision"),
            "sandbox_image_digest": provenance.get("sandbox_image_digest"),
            "sandbox_platform": provenance.get("sandbox_platform"),
            "stirrup_version": STIRRUP_VERSION,
            "system_prompt_sha256": gdpval_aa.sha256_text(gdpval_aa.AA_GDPVAL_SYSTEM_PROMPT),
            "task_prompt_sha256": gdpval_aa.sha256_text(str(task.instruction)),
            "reference_files": provenance.get("reference_files") or [],
            "termination": termination or {"type": "unknown", "reason": "the solver produced no run metadata"},
            "rejected_paths": sandbox_manifest.get("rejected_paths") or [],
            "artifacts": artifacts,
            "metrics": metrics,
            "sampling_parameters": dict(config.sampling_params or {}),
        }
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _task_provenance(task: Task) -> dict[str, Any]:
    """Read the builder's ``gdpval_aa.json`` for this task, if present."""
    task_dir = getattr(task, "task_dir", None)
    if task_dir is None:
        return {}
    return _read_json(Path(task_dir) / "gdpval_aa.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _usage_metrics(run_data: dict[str, Any], model: str) -> dict[str, Any]:
    metadata = run_data.get("metadata") if isinstance(run_data.get("metadata"), dict) else {}
    raw_usage = metadata.get("token_usage")
    usage_entries = raw_usage if isinstance(raw_usage, list) else [raw_usage]
    usage = [entry for entry in usage_entries if isinstance(entry, dict)]
    input_tokens = sum(int(entry.get("input") or 0) for entry in usage)
    answer_tokens = sum(int(entry.get("answer") or 0) for entry in usage)
    reasoning_tokens = sum(int(entry.get("reasoning") or 0) for entry in usage)
    output_tokens = answer_tokens + reasoning_tokens
    metrics: dict[str, Any] = {
        "turns": int(run_data.get("turns") or 0),
        "input_tokens": input_tokens,
        "answer_tokens": answer_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    pricing_path = os.environ.get("RLLM_PRICING_FILE") or os.environ.get("GDPVAL_PRICING_FILE")
    if pricing_path:
        try:
            pricing = json.loads(Path(pricing_path).expanduser().read_text(encoding="utf-8"))
            models = pricing.get("models") if isinstance(pricing, dict) else {}
            rates = models.get(model) or models.get(model.removeprefix("openrouter/"))
            if isinstance(rates, dict):
                metrics["cost_usd"] = (
                    input_tokens * float(rates.get("input") or 0)
                    + answer_tokens * float(rates.get("answer") or rates.get("output") or 0)
                    + reasoning_tokens * float(rates.get("reasoning") or rates.get("answer") or rates.get("output") or 0)
                ) / 1_000_000
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return metrics


_DRIVER_SCRIPT = r'''"""In-sandbox Stirrup driver for GDPval-AA v2.

Runs as the solver user (UID 1000) inside the task sandbox. Builds the agent
with AA's system prompt, tool set and limits, then stages whatever the model
submitted into a single bundle so the harness can copy it out before teardown.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Annotated, Any

import anyio
from pydantic import BaseModel, Field

from stirrup import Agent, aggregate_metadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.models import AssistantMessage, ImageContentBlock, Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CodeExecToolProvider, CodeExecutionParams, CommandResult
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.tools.web import WebToolProvider

WORKDIR = Path(os.environ.get("RLLM_STIRRUP_WORKDIR", "/home/user"))
SUBMISSION_DIR = Path(os.environ["RLLM_STIRRUP_SUBMISSION_DIR"])
RUN_METADATA_PATH = Path(os.environ["RLLM_STIRRUP_RUN_METADATA_PATH"])
# Roots are resolved once: submitted paths are compared after resolution, so an
# unresolved root (/tmp is a symlink on some systems) would reject every file.
SUBMITTABLE_ROOTS = [Path(p).resolve() for p in json.loads(os.environ.get("RLLM_STIRRUP_SUBMITTABLE_ROOTS", '["/home/user", "/tmp"]'))]
SHELL_TIMEOUT = int(os.environ.get("RLLM_STIRRUP_SHELL_TIMEOUT", "600"))


class SandboxCodeExecToolProvider(CodeExecToolProvider):
    """Whole-sandbox code execution rooted at the AA working directory.

    Stirrup's LocalCodeExecToolProvider is designed for an *unsandboxed* host:
    it confines every command to a private temp directory and rejects commands
    that mention /home, /tmp or ~. rLLM already isolates the task in a
    container, so this provider gives the agent the same view an E2B sandbox
    would — the whole filesystem, with /home/user as the working directory.

    Each call runs a fresh ``bash -c``, so no working directory, environment
    variable or other shell state survives between calls, exactly as AA's task
    prompt tells the model to expect.
    """

    def __init__(self, workdir, *, shell_timeout, env=None):
        # No allowlist: AA's code_exec is an unrestricted shell inside an
        # already-isolated sandbox.
        super().__init__(allowed_commands=None, shell_timeout=shell_timeout)
        self._workdir = Path(workdir)
        self._env = dict(env) if env is not None else None

    @property
    def temp_dir(self):
        return self._workdir

    async def __aenter__(self):
        self._workdir.mkdir(parents=True, exist_ok=True)
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    def _resolve(self, path):
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._workdir / candidate

    async def run_command(self, cmd, *, timeout=None):
        if timeout is None:
            timeout = self._shell_timeout
        process = None
        try:
            with anyio.fail_after(timeout):
                # start_new_session puts bash at the head of its own process
                # group so a timeout can kill the whole tree, not just bash.
                process = await anyio.open_process(
                    ["bash", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self._workdir),
                    env=self._env,
                    start_new_session=True,
                )
                stdout_chunks = []
                stderr_chunks = []

                async def read_stdout():
                    if process.stdout:
                        stdout_chunks.extend([chunk async for chunk in process.stdout])

                async def read_stderr():
                    if process.stderr:
                        stderr_chunks.extend([chunk async for chunk in process.stderr])

                async with anyio.create_task_group() as tg:
                    tg.start_soon(read_stdout)
                    tg.start_soon(read_stderr)
                await process.wait()
                return CommandResult(
                    exit_code=process.returncode or 0,
                    stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                    stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                )
        except TimeoutError:
            if process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with anyio.move_on_after(5):
                    await process.wait()
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        except Exception as exc:
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="execution_error")

    async def read_file_bytes(self, path):
        resolved = self._resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved.read_bytes()

    async def write_file_bytes(self, path, content):
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)

    async def file_exists(self, path):
        return self._resolve(path).is_file()

    async def is_directory(self, path):
        return self._resolve(path).is_dir()

    async def list_files(self, path):
        resolved = self._resolve(path)
        if not resolved.is_dir():
            return []
        return [str(child.relative_to(resolved)) for child in sorted(resolved.rglob("*")) if child.is_file()]

    async def view_image(self, path):
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if resolved.is_dir():
            raise ValueError(f"Path is a directory, not an image: {path}")
        # Downscaling to one megapixel happens in ImageContentBlock, which
        # defaults to stirrup.constants.RESOLUTION_1MP.
        return ImageContentBlock(data=resolved.read_bytes())


class FinishParams(BaseModel):
    """Completed work, submitted as absolute file paths."""

    summary: Annotated[str, Field(description="A brief summary of what you accomplished.")]
    paths: Annotated[
        list[str],
        Field(description="A list of ABSOLUTE file paths for the required output files. Do not submit folders, only files."),
    ]


class AbandonParams(BaseModel):
    """Explanation for why the task cannot be completed."""

    reason: Annotated[str, Field(description="A brief reason why the task cannot be completed.")]


def path_problem(raw):
    """Return why *raw* is not a submittable file, or None if it is valid."""
    if not isinstance(raw, str) or not raw:
        return "is empty"
    if not raw.startswith("/"):
        return "is not an absolute path"
    try:
        resolved = Path(raw).resolve()
    except OSError as exc:
        return f"could not be resolved ({exc})"
    if not any(resolved == root or root in resolved.parents for root in SUBMITTABLE_ROOTS):
        allowed = ", ".join(str(root) for root in SUBMITTABLE_ROOTS)
        return f"resolves outside the writable roots ({allowed})"
    if not resolved.exists():
        return "does not exist"
    if resolved.is_dir():
        return "is a directory, not a file"
    if not resolved.is_file():
        return "is not a regular file"
    return None


def partition_paths(paths):
    accepted, rejected = [], []
    for raw in paths:
        problem = path_problem(raw)
        if problem is None:
            accepted.append(raw)
        else:
            rejected.append({"path": raw, "reason": problem})
    return accepted, rejected


async def finish_executor(params):
    _, rejected = partition_paths(params.paths)
    if rejected:
        details = "; ".join(f"{entry['path']} {entry['reason']}" for entry in rejected)
        return ToolResult(
            content=f"ERROR: these submitted paths are not valid deliverables: {details}. Submit absolute paths to existing files.",
            metadata=ToolUseCountMetadata(),
            success=False,
        )
    if not params.paths:
        return ToolResult(
            content="ERROR: no files submitted. Provide absolute paths to every deliverable, or use abandon_task_finish.",
            metadata=ToolUseCountMetadata(),
            success=False,
        )
    return ToolResult(content=params.summary, metadata=ToolUseCountMetadata(), success=True)


async def abandon_executor(params):
    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


FINISH_TOOL = Tool[FinishParams, ToolUseCountMetadata](
    name="finish",
    description=(
        "Signal task completion and submit your work. Provide a brief summary and the absolute path of every "
        "deliverable file. Note that you will need a separate turn to finish."
    ),
    parameters=FinishParams,
    executor=finish_executor,
)

ABANDON_TOOL = Tool[AbandonParams, ToolUseCountMetadata](
    name="abandon_task_finish",
    description=(
        "Signal that you cannot complete the task, with a brief reason, instead of submitting files. "
        "Use only when required inputs are missing, a hard dependency is unavailable, or the request is incoherent."
    ),
    parameters=AbandonParams,
    executor=abandon_executor,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_submission(paths):
    """Copy submitted files into the bundle, preserving path→file mapping.

    Basenames collide across directories, so each file keeps its full source
    path under the bundle rather than being flattened.
    """
    # Created even when nothing is accepted, so the harness always has a
    # bundle to download and can tell "submitted nothing" from "never ran".
    (SUBMISSION_DIR / "files").mkdir(parents=True, exist_ok=True)
    accepted, rejected = partition_paths(paths)
    artifacts = []
    for raw in accepted:
        source = Path(raw).resolve()
        bundle_relative = Path("files") / source.relative_to("/")
        destination = SUBMISSION_DIR / bundle_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            rejected.append({"path": raw, "reason": f"could not be copied ({exc})"})
            continue
        artifacts.append(
            {
                "submitted_path": raw,
                "bundle_path": str(bundle_relative),
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    return artifacts, rejected


def count_turns(history):
    return sum(1 for messages in history for message in messages if isinstance(message, AssistantMessage))


def write_run_metadata(payload):
    RUN_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_METADATA_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


async def main():
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    client = ChatCompletionsClient(
        model=os.environ["RLLM_STIRRUP_MODEL"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get("OPENAI_API_KEY", "sk-rllm-gateway"),
        # Two distinct budgets: max_tokens caps generation, while
        # context_window_tokens is the capacity the 70% compaction threshold is
        # a fraction of.
        max_tokens=int(os.environ.get("RLLM_STIRRUP_MAX_OUTPUT_TOKENS", "16384")),
        context_window_tokens=int(os.environ.get("RLLM_STIRRUP_MAX_CONTEXT_TOKENS", "200000")),
        reasoning_effort=os.environ.get("RLLM_STIRRUP_REASONING_EFFORT"),
    )

    exec_env = SandboxCodeExecToolProvider(
        WORKDIR,
        shell_timeout=SHELL_TIMEOUT,
        env={**os.environ, "HOME": str(WORKDIR), "PWD": str(WORKDIR)},
    )
    tools = [exec_env]
    if os.environ.get("RLLM_STIRRUP_ENABLE_WEB", "1") == "1":
        tools.append(WebToolProvider())
    if os.environ.get("RLLM_STIRRUP_ENABLE_VISION", "1") == "1":
        tools.append(ViewImageToolProvider(exec_env))

    agent = Agent(
        client=client,
        name="gdpval-aa-solver",
        max_turns=int(os.environ.get("RLLM_STIRRUP_MAX_TURNS", "250")),
        system_prompt=Path(os.environ["RLLM_STIRRUP_SYSTEM_PROMPT_PATH"]).read_text(encoding="utf-8"),
        tools=tools,
        finish_tool=[FINISH_TOOL, ABANDON_TOOL],
        context_summarization_cutoff=float(os.environ.get("RLLM_STIRRUP_CONTEXT_CUTOFF", "0.7")),
    )

    prompt = Path(os.environ["RLLM_STIRRUP_INSTRUCTION_PATH"]).read_text(encoding="utf-8")
    # No output_dir and no input_files: reference files are already staged at
    # the absolute paths the prompt quotes, and submitted files are bundled
    # here rather than flattened into an output directory by basename.
    finish_params, history, metadata, error = None, [], {}, None
    try:
        async with agent.session(cache_on_interrupt=False) as session:
            finish_params, history, metadata = await session.run(prompt)
    except Exception as exc:
        # A run that died still belongs in the corpus; losing the record would
        # make it indistinguishable from a task that was never attempted.
        error = f"{type(exc).__name__}: {exc}"

    artifacts, rejected = [], []
    if error is not None:
        termination = {"type": "error", "reason": error}
    elif isinstance(finish_params, FinishParams):
        artifacts, rejected = stage_submission(finish_params.paths)
        termination = {"type": "finish", "summary": finish_params.summary, "submitted_paths": list(finish_params.paths)}
    elif isinstance(finish_params, AbandonParams):
        termination = {"type": "abandon_task_finish", "reason": finish_params.reason}
    else:
        termination = {"type": "max_turns_exhausted"}

    manifest = {
        "schema_version": 1,
        "termination": termination,
        "artifacts": artifacts,
        "rejected_paths": rejected,
    }
    (SUBMISSION_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_metadata(
        {
            "termination": termination,
            "turns": count_turns(history),
            "metadata": aggregate_metadata(metadata, return_json_serializable=True),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
'''
