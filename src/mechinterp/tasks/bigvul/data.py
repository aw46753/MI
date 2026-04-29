"""Big-Vul dataset preprocessing and loading helpers."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mechinterp.core.config import ExperimentConfig


SUPPORTED_CWES = ("CWE-119", "CWE-20")
RAW_EXTENSIONS = (".jsonl", ".json", ".csv")


@dataclass(frozen=True)
class BigVulExample:
    """One normalized Big-Vul vulnerable/patched pair."""

    sample_id: str
    split: str
    cwe_id: str
    commit_id: str
    vulnerable_code: str
    patched_code: str
    project: str | None = None
    file_path: str | None = None
    function_name: str | None = None
    cve_id: str | None = None
    commit_message: str | None = None
    vulnerable_line_count: int = 0
    patched_line_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


def normalize_cwe_label(value: Any) -> str | None:
    """Normalize a raw CWE value to the supported `CWE-<id>` form."""

    if value is None:
        return None

    text = str(value).strip().upper()
    if not text:
        return None

    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None

    normalized = f"CWE-{int(digits)}"
    if normalized not in SUPPORTED_CWES:
        return None
    return normalized


def _split_name_for_cwe(cwe_id: str) -> str:
    return cwe_id.lower().replace("-", "_")


def _processed_dir(config: ExperimentConfig) -> Path:
    dataset_path = Path(config.dataset.dataset_path or "data/bigvul/raw")
    base_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent
    if base_dir.name == "raw":
        return base_dir.parent / "processed"
    return base_dir / "processed"


def _iter_raw_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in RAW_EXTENSIONS:
            raise ValueError(f"Unsupported Big-Vul dataset file type: {dataset_path.suffix}")
        return [dataset_path]

    if not dataset_path.exists():
        raise ValueError(f"Big-Vul dataset path does not exist: {dataset_path}")

    files = sorted(
        path for path in dataset_path.rglob("*") if path.is_file() and path.suffix.lower() in RAW_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported Big-Vul dataset files found under {dataset_path}")
    return files


def _load_json_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_csv_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_raw_records(dataset_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_raw_files(dataset_path):
        if path.suffix.lower() == ".jsonl":
            rows.extend(_load_jsonl_file(path))
        elif path.suffix.lower() == ".json":
            rows.extend(_load_json_file(path))
        elif path.suffix.lower() == ".csv":
            rows.extend(_load_csv_file(path))
    return rows


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_code(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _normalize_record(row: dict[str, Any]) -> dict[str, Any] | None:
    cwe_id = normalize_cwe_label(
        _first_value(row, ("cwe_id", "cwe", "CWE ID", "cweid", "vul", "vulnerability_classification"))
    )
    if cwe_id is None:
        return None

    vulnerable_code = _normalize_code(
        _first_value(row, ("func_before", "before", "code_before", "vulnerable_code", "func_before_cleaned"))
    )
    patched_code = _normalize_code(
        _first_value(row, ("func_after", "after", "code_after", "patched_code", "func_after_cleaned"))
    )
    if vulnerable_code is None or patched_code is None:
        return None

    sample_id = _normalize_optional_text(
        _first_value(row, ("sample_id", "id", "item_id", "bigvul_id", "_id", "commit_id"))
    )
    commit_id = _normalize_optional_text(
        _first_value(row, ("commit_id", "commit", "commit_hash", "hash", "sha"))
    )
    stable_id = sample_id or commit_id
    if stable_id is None:
        return None

    return {
        "sample_id": stable_id,
        "commit_id": commit_id or stable_id,
        "cwe_id": cwe_id,
        "vulnerable_code": vulnerable_code,
        "patched_code": patched_code,
        "project": _normalize_optional_text(_first_value(row, ("project", "project_name", "repo_name"))),
        "file_path": _normalize_optional_text(_first_value(row, ("file_path", "path", "files"))),
        "function_name": _normalize_optional_text(_first_value(row, ("function_name", "func_name", "function"))),
        "cve_id": _normalize_optional_text(_first_value(row, ("cve_id", "cve", "CVE ID"))),
        "commit_message": _normalize_optional_text(
            _first_value(row, ("commit_message", "message", "commit_msg"))
        ),
        "vulnerable_line_count": _line_count(vulnerable_code),
        "patched_line_count": _line_count(patched_code),
    }


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["sample_id"])
        current = best_by_id.get(row_id)
        if current is None:
            best_by_id[row_id] = row
            continue

        current_score = len(current["vulnerable_code"]) + len(current["patched_code"])
        row_score = len(row["vulnerable_code"]) + len(row["patched_code"])
        if row_score > current_score:
            best_by_id[row_id] = row

    return [best_by_id[key] for key in sorted(best_by_id)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def preprocess_bigvul_dataset(config: ExperimentConfig) -> dict[str, Any]:
    """Normalize, sample, and materialize Big-Vul paired testcases."""

    dataset_path = Path(config.dataset.dataset_path or "data/bigvul/raw")
    processed_dir = _processed_dir(config)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_raw_records(dataset_path)
    normalized_candidates = [_normalize_record(row) for row in raw_rows]
    filtered_missing_required = sum(1 for row in normalized_candidates if row is None)
    normalized_rows = _deduplicate_rows([row for row in normalized_candidates if row is not None])

    target_cwes = [normalize_cwe_label(cwe) for cwe in config.dataset.target_cwes]
    if any(cwe is None for cwe in target_cwes):
        raise ValueError(f"Unsupported target CWEs: {config.dataset.target_cwes}")
    normalized_target_cwes = [str(cwe) for cwe in target_cwes]

    filtered_rows = [row for row in normalized_rows if row["cwe_id"] in normalized_target_cwes]
    counts_by_cwe = {
        cwe_id: sum(1 for row in filtered_rows if row["cwe_id"] == cwe_id) for cwe_id in normalized_target_cwes
    }
    underfilled = {cwe_id: count for cwe_id, count in counts_by_cwe.items() if count < int(config.dataset.pairs_per_cwe or 0)}
    if underfilled:
        details = ", ".join(f"{cwe_id}={count}" for cwe_id, count in sorted(underfilled.items()))
        raise ValueError(
            "Not enough cleaned Big-Vul pairs for requested sampling counts: "
            f"{details}; requested={config.dataset.pairs_per_cwe}"
        )

    normalized_path = processed_dir / "normalized.jsonl"
    _write_jsonl(normalized_path, filtered_rows)

    pair_files: dict[str, str] = {}
    selected_counts: dict[str, int] = {}
    for cwe_id in normalized_target_cwes:
        rows_for_cwe = [row for row in filtered_rows if row["cwe_id"] == cwe_id]
        sampler = random.Random(f"{config.seed}:{cwe_id}")
        shuffled_rows = list(rows_for_cwe)
        sampler.shuffle(shuffled_rows)
        selected_rows = sorted(shuffled_rows[: int(config.dataset.pairs_per_cwe or 0)], key=lambda row: row["sample_id"])

        split_name = _split_name_for_cwe(cwe_id)
        pair_path = processed_dir / f"{split_name}_pairs.jsonl"
        _write_jsonl(pair_path, selected_rows)
        pair_files[split_name] = str(pair_path)
        selected_counts[cwe_id] = len(selected_rows)

    manifest = {
        "dataset_path": str(dataset_path),
        "processed_dir": str(processed_dir),
        "seed": config.seed,
        "target_cwes": normalized_target_cwes,
        "pairs_per_cwe": config.dataset.pairs_per_cwe,
        "raw_record_count": len(raw_rows),
        "filtered_missing_required_count": filtered_missing_required,
        "normalized_count": len(filtered_rows),
        "counts_by_cwe": counts_by_cwe,
        "selected_counts": selected_counts,
        "pair_files": pair_files,
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_pair_rows(split: str, config: ExperimentConfig) -> list[dict[str, Any]]:
    processed_dir = _processed_dir(config)
    pair_path = processed_dir / f"{split}_pairs.jsonl"
    if not pair_path.exists():
        preprocess_bigvul_dataset(config)
    if not pair_path.exists():
        raise ValueError(f"Processed Big-Vul split file not found: {pair_path}")
    return _read_jsonl(pair_path)


def build_bigvul_dataset(split: str, config: ExperimentConfig) -> list[BigVulExample]:
    """Load one sampled Big-Vul split as paired vulnerable/patched examples."""

    rows = _load_pair_rows(split, config)
    examples: list[BigVulExample] = []
    for row in rows:
        examples.append(
            BigVulExample(
                sample_id=str(row["sample_id"]),
                split=split,
                cwe_id=str(row["cwe_id"]),
                commit_id=str(row["commit_id"]),
                vulnerable_code=str(row["vulnerable_code"]),
                patched_code=str(row["patched_code"]),
                project=_normalize_optional_text(row.get("project")),
                file_path=_normalize_optional_text(row.get("file_path")),
                function_name=_normalize_optional_text(row.get("function_name")),
                cve_id=_normalize_optional_text(row.get("cve_id")),
                commit_message=_normalize_optional_text(row.get("commit_message")),
                vulnerable_line_count=int(row.get("vulnerable_line_count", _line_count(str(row["vulnerable_code"])))),
                patched_line_count=int(row.get("patched_line_count", _line_count(str(row["patched_code"])))),
            )
        )
    return examples
