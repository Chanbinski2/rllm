from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

from rllm.data import gdpval_aa as aa
from rllm.data import gdpval_builder as gb
from rllm.tasks.loader import BenchmarkLoader


def _rows() -> list[dict]:
    return [
        {
            "task_id": "task-one",
            "sector": "Finance",
            "occupation": "Analyst",
            "prompt": "Review the workbook and produce a report.",
            "reference_files": ["reference_files/source.xlsx"],
            "deliverable_files": ["deliverable_files/expert.docx"],
            "rubric_pretty": "Prefer a correct and polished report.",
            "rubric_json": "[]",
        },
        {
            "task_id": "task-without-gold",
            "sector": "Technology",
            "occupation": "Engineer",
            "prompt": "Create a design.",
            "reference_files": [],
            "deliverable_files": [],
            "rubric_pretty": "",
            "rubric_json": "[]",
        },
    ]


def _build(tmp_path: Path, *, limit: int | None = None) -> Path:
    source = tmp_path / "source.xlsx"
    source.write_bytes(b"source workbook")
    expert = tmp_path / "expert.docx"
    expert.write_bytes(b"expert report")
    files = {
        "reference_files/source.xlsx": str(source),
        "deliverable_files/expert.docx": str(expert),
    }

    with (
        patch("datasets.load_dataset", return_value=_rows()),
        patch("huggingface_hub.hf_hub_download", side_effect=lambda _repo, path, **_kwargs: files[path]),
        patch.object(gb, "_dataset_revision", return_value="abc123"),
    ):
        out = tmp_path / "gdpval"
        gb.build_benchmark(out_dir=out, limit=limit, repair_office_files=False, register=False)
    return out


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_builder_writes_rllm_harbor_layout(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    assert (out / "dataset.toml").exists()
    for name in ["task.toml", "instruction.md", "gdpval_aa.json"]:
        assert (task / name).exists(), name
    assert (task / "environment" / "Dockerfile").exists()
    assert (task / "tests" / "test.sh").exists()
    assert (task / "tests" / "evaluate.py").exists()
    assert (task / "tests" / "test.sh").stat().st_mode & 0o111


def test_expert_deliverables_never_enter_the_solver_environment(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    # Inputs are staged into environment/files (uploaded to the workdir);
    # expert deliverables stay under tests/, which is only uploaded — root
    # owned, mode 700 — after the solver has finished.
    assert (task / "environment" / "files" / "source.xlsx").read_bytes() == b"source workbook"
    assert (task / "tests" / "reference" / "expert.docx").read_bytes() == b"expert report"

    staged = [path.name for path in (task / "environment").rglob("*") if path.is_file()]
    assert "expert.docx" not in staged
    assert not any("expert" in name for name in staged)


def test_solver_facing_files_never_name_the_expert_deliverable(tmp_path):
    out = _build(tmp_path)
    task = out / "task-one"

    for name in ["instruction.md", "task.toml"]:
        text = (task / name).read_text()
        assert "expert.docx" not in text, name
        assert "deliverable_files" not in text, name


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_instruction_is_aa_prompt_with_absolute_reference_paths(tmp_path):
    out = _build(tmp_path)

    instruction = (out / "task-one" / "instruction.md").read_text()

    assert instruction == aa.render_aa_gdpval_task_prompt(
        "Review the workbook and produce a report.",
        ["/home/user/source.xlsx"],
    )
    assert "- /home/user/source.xlsx" in instruction
    # The old harness prompt listed bare basenames and asked for relative paths.
    assert "- source.xlsx" not in instruction
    assert "relative paths" not in instruction
    assert "Expected deliverable filename" not in instruction


def test_instruction_for_a_task_without_reference_files(tmp_path):
    out = _build(tmp_path)

    instruction = (out / "task-without-gold" / "instruction.md").read_text()

    assert instruction == aa.render_aa_gdpval_task_prompt("Create a design.", [])
    assert "## Reference Files Location" not in instruction


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


def test_task_config_runs_the_solver_as_the_non_root_aa_user(tmp_path):
    out = _build(tmp_path, limit=1)

    task_toml = (out / "task-one" / "task.toml").read_text()

    assert '[agent]\nuser = "user"' in task_toml
    assert 'user = "root"' not in task_toml.split("[verifier]")[0]
    assert 'workdir = "/home/user"' in task_toml
    assert 'reference_files = ["/home/user/source.xlsx"]' in task_toml


def test_generated_tasks_carry_no_judge_credentials(tmp_path):
    out = _build(tmp_path, limit=1)

    task_toml = (out / "task-one" / "task.toml").read_text()

    assert "[verifier.env]" not in task_toml
    for name in ["GDPVAL_JUDGE_API_KEY", "GDPVAL_JUDGE_MODEL", "GDPVAL_JUDGE_BASE_URL"]:
        assert name not in task_toml, name


def test_provenance_records_hashes_and_pinned_environment(tmp_path):
    out = _build(tmp_path)

    provenance = json.loads((out / "task-one" / "gdpval_aa.json").read_text())

    assert provenance["dataset_repo"] == "openai/gdpval"
    assert provenance["dataset_revision"] == "abc123"
    assert provenance["sandbox_image_digest"] == aa.AA_BASE_IMAGE_DIGEST
    assert provenance["sandbox_platform"] == "linux/amd64"
    assert provenance["system_prompt_sha256"] == aa.sha256_text(aa.AA_GDPVAL_SYSTEM_PROMPT)
    assert provenance["task_prompt_sha256"] == aa.sha256_text((out / "task-one" / "instruction.md").read_text())
    assert [entry["path"] for entry in provenance["reference_files"]] == ["/home/user/source.xlsx"]


def test_provenance_reference_hash_matches_the_staged_file(tmp_path):
    out = _build(tmp_path)
    provenance = json.loads((out / "task-one" / "gdpval_aa.json").read_text())
    staged = out / "task-one" / "environment" / "files" / "source.xlsx"

    entry = provenance["reference_files"][0]

    assert entry["sha256"] == gb._sha256(staged)
    assert entry["size_bytes"] == staged.stat().st_size


def test_builder_output_round_trips_through_benchmark_loader(tmp_path):
    out = _build(tmp_path)

    result = BenchmarkLoader.load(str(out))

    assert result.name == "gdpval"
    assert result.harness_name == "stirrup"
    assert sorted(task.id for task in result.tasks) == ["task-one", "task-without-gold"]
    by_id = {task.id: task for task in result.tasks}
    assert by_id["task-one"].metadata["reference_files"] == ["/home/user/source.xlsx"]
    assert by_id["task-one"].metadata["agent_user"] == "user"
    assert by_id["task-one"].metadata["workdir"] == "/home/user"
    # The --platform flag must not be mistaken for the image reference.
    assert by_id["task-one"].metadata["environment"]["docker_image"].startswith("debian:")


def test_builder_respects_limit(tmp_path):
    out = _build(tmp_path, limit=1)

    task_dirs = sorted(path.name for path in out.iterdir() if path.is_dir() and not path.name.startswith("."))
    assert task_dirs == ["task-one"]


def test_dataset_catalog_entry_points_to_builder():
    catalog_path = Path(gb.__file__).parents[1] / "registry" / "datasets.json"
    catalog = json.loads(catalog_path.read_text())

    entry = catalog["datasets"]["gdpval"]
    assert entry["builder"] == "rllm.data.gdpval_builder:build_benchmark"
    assert entry["default_agent"] == "stirrup"


# ---------------------------------------------------------------------------
# Office repair
# ---------------------------------------------------------------------------


def _write_repairable_docx(path: Path) -> None:
    relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document_relationships = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        package.writestr("_rels/.rels", relationships)
        package.writestr("word/document.xml", "<document/>")
        package.writestr("word/settings.xml", '<w:settings xmlns:w="w" xmlns:ns1="broken"><ns1:item/></w:settings>')
        package.writestr("word/_rels/document.xml.rels", document_relationships)


def test_office_repair_is_auditable_and_preserves_original(tmp_path):
    source = tmp_path / "broken.docx"
    backup = tmp_path / "backups" / "broken.docx"
    _write_repairable_docx(source)
    original = source.read_bytes()

    record = gb._repair_office_file(source, backup)

    assert record["status"] == "repaired"
    assert backup.read_bytes() == original
    assert record["original_sha256"] != record["repaired_sha256"]
    assert record["original_sha256"] and record["repaired_sha256"]
    with zipfile.ZipFile(source) as package:
        assert "word/settings.xml" not in package.namelist()


def test_office_repair_leaves_intact_files_untouched(tmp_path):
    intact = tmp_path / "fine.xlsx"
    with zipfile.ZipFile(intact, "w") as package:
        package.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        package.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
    before = intact.read_bytes()

    record = gb._repair_office_file(intact, tmp_path / "backups" / "fine.xlsx")

    assert intact.read_bytes() == before
    assert record["original_sha256"] == record["repaired_sha256"]
    assert record["backup"] is None


# ---------------------------------------------------------------------------
# Structural verifier
# ---------------------------------------------------------------------------


def _run_verifier(tmp_path: Path, *, run: dict | None, manifest: dict | None) -> dict:
    """Execute the generated verifier against a fake sandbox layout."""
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    run_path = tmp_path / "run.json"
    reward_path = tmp_path / "reward.json"
    if run is not None:
        run_path.write_text(json.dumps(run))
    if manifest is not None:
        (submission_dir / "manifest.json").write_text(json.dumps(manifest))

    namespace: dict = {"__name__": "gdpval_verifier_under_test"}
    exec(compile(gb._VERIFIER_SOURCE, "evaluate.py", "exec"), namespace)
    namespace["SUBMISSION_DIR"] = submission_dir
    namespace["RUN_PATH"] = run_path
    namespace["MANIFEST_PATH"] = submission_dir / "manifest.json"
    namespace["REWARD_PATH"] = reward_path
    namespace["main"]()
    return json.loads(reward_path.read_text())


def _preserved(tmp_path: Path, submitted: str) -> dict:
    bundle = Path("files") / submitted.lstrip("/")
    target = tmp_path / "submission" / bundle
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("deliverable")
    return {"submitted_path": submitted, "bundle_path": str(bundle), "sha256": "x", "size_bytes": 11}


def test_verifier_cannot_call_a_judge():
    """The verifier is stdlib-only and offline: no client, no credentials."""
    tree = ast.parse(gb._VERIFIER_SOURCE)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "json", "pathlib"}
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for forbidden in ["GDPVAL_JUDGE_API_KEY", "GDPVAL_JUDGE_MODEL", "GDPVAL_JUDGE_BASE_URL"]:
        assert forbidden not in literals


def test_verifier_emits_only_structural_signals(tmp_path):
    artifact = _preserved(tmp_path, "/home/user/report.docx")

    reward = _run_verifier(
        tmp_path,
        run={"termination": {"type": "finish", "summary": "done", "submitted_paths": ["/home/user/report.docx"]}},
        manifest={"artifacts": [artifact], "rejected_paths": []},
    )

    assert set(reward["signals"]) == {"ungraded", "finish_called", "abandoned", "submission_valid", "artifact_count"}
    for forbidden in ["pairwise_win", "win_rate", "quality_score", "elo", "score"]:
        assert forbidden not in reward["signals"]


def test_verifier_accepts_a_well_formed_submission(tmp_path):
    artifact = _preserved(tmp_path, "/home/user/report.docx")

    reward = _run_verifier(
        tmp_path,
        run={"termination": {"type": "finish", "summary": "done", "submitted_paths": ["/home/user/report.docx"]}},
        manifest={"artifacts": [artifact], "rejected_paths": []},
    )

    assert reward["signals"]["finish_called"] == 1.0
    assert reward["signals"]["submission_valid"] == 1.0
    assert reward["signals"]["artifact_count"] == 1.0
    # Structural validity is never reported as a good answer.
    assert reward["reward"] == 0.0
    assert reward["is_correct"] is False
    assert reward["signals"]["ungraded"] == 1.0
    assert reward["metadata"]["graded"] is False


def test_verifier_records_abandonment(tmp_path):
    reward = _run_verifier(
        tmp_path,
        run={"termination": {"type": "abandon_task_finish", "reason": "input missing"}},
        manifest=None,
    )

    assert reward["signals"]["abandoned"] == 1.0
    assert reward["signals"]["finish_called"] == 0.0
    assert reward["signals"]["submission_valid"] == 0.0
    assert reward["metadata"]["abandon_reason"] == "input missing"


def test_verifier_flags_a_run_that_never_finished(tmp_path):
    reward = _run_verifier(tmp_path, run={"termination": {"type": "max_turns_exhausted"}}, manifest=None)

    assert reward["signals"]["finish_called"] == 0.0
    assert reward["metadata"]["reason"] == "no_finish_tool_call"


def test_verifier_flags_missing_run_metadata(tmp_path):
    reward = _run_verifier(tmp_path, run=None, manifest=None)

    assert reward["metadata"]["reason"] == "no_run_metadata"
    assert reward["signals"]["submission_valid"] == 0.0


def test_verifier_flags_artifacts_that_were_not_preserved(tmp_path):
    reward = _run_verifier(
        tmp_path,
        run={"termination": {"type": "finish", "summary": "done", "submitted_paths": ["/home/user/report.docx"]}},
        manifest={
            "artifacts": [{"submitted_path": "/home/user/report.docx", "bundle_path": "files/home/user/report.docx"}],
            "rejected_paths": [],
        },
    )

    assert reward["signals"]["submission_valid"] == 0.0
    assert reward["metadata"]["reason"] == "artifacts_not_preserved"


def test_verifier_flags_rejected_paths(tmp_path):
    artifact = _preserved(tmp_path, "/home/user/report.docx")

    reward = _run_verifier(
        tmp_path,
        run={"termination": {"type": "finish", "summary": "done", "submitted_paths": ["/home/user/report.docx", "relative.docx"]}},
        manifest={"artifacts": [artifact], "rejected_paths": [{"path": "relative.docx", "reason": "is not an absolute path"}]},
    )

    assert reward["signals"]["submission_valid"] == 0.0
    assert reward["metadata"]["reason"] == "invalid_submitted_paths"
