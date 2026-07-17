from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_clean_training_validation.py"
REPOSITORIES = (
    "legalize-kr/admrule-kr",
    "legalize-kr/legalize-kr",
    "legalize-kr/ordinance-kr",
    "legalize-kr/precedent-kr",
)


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source(root: Path) -> None:
    queries = []
    corpus = []
    qrels = []
    provenance = []
    for repository_index, repository in enumerate(REPOSITORIES):
        for index in range(6):
            query_id = f"q-{repository_index}-{index}"
            corpus_id = f"d-{repository_index}-{index}"
            document_sha = hashlib.sha256(f"doc-{repository_index}-{index}".encode()).hexdigest()
            metadata = {
                "repository": repository,
                "source_document_sha256": document_sha,
                "source_candidate_id": f"candidate-{repository_index}-{index}",
            }
            queries.append({"_id": query_id, "text": f"법률 질의 {repository_index} 제{index}조", "metadata": metadata})
            corpus.append({"_id": corpus_id, "title": f"법률 {repository_index}", "text": f"법률 본문 {repository_index} 제{index}조 고유 내용", "metadata": metadata})
            qrels.append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})
            provenance.append({"query_id": query_id, "source_document_sha256": document_sha})
    files = {
        "queries.jsonl": queries,
        "corpus.jsonl": corpus,
        "qrels.jsonl": qrels,
        "provenance.jsonl": provenance,
    }
    declared = {}
    for name, rows in files.items():
        path = root / name
        path.write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")
        declared[name] = {"rows": len(rows), "bytes": path.stat().st_size, "sha256": file_hash(path)}
    manifest = {
        "artifact_id": "fixture",
        "status": "complete",
        "independence": {"grade": "I", "not_grade": "Z"},
        "assertions": {
            "selected_query_hash_overlap_with_benchmark": 0,
            "selected_positive_hash_overlap_with_benchmark": 0,
            "selected_source_candidate_id_overlap_with_training": 0,
            "selected_source_document_sha256_overlap_with_training": 0,
            "selected_query_hash_overlap_with_training_text": 0,
            "selected_positive_hash_overlap_with_training_text": 0,
        },
        "files": declared,
    }
    (root / "manifest.json").write_text(compact(manifest) + "\n", encoding="utf-8")


def write_training(path: Path) -> None:
    row = {
        "messages": [{"role": "user", "content": "Instruct: x\nQuery: 법률 질의 0 제0조"}],
        "positive_messages": [[{"role": "user", "content": "unrelated positive"}]],
        "negative_messages": [[{"role": "user", "content": "법률 본문 1 제0조 고유 내용"}]],
    }
    path.write_text(compact(row) + "\n", encoding="utf-8")


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def test_build_is_balanced_deterministic_and_verifiable() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        write_source(source)
        training = root / "train.jsonl"
        write_training(training)
        first = root / "first"
        second = root / "second"
        common = (
            "--source-dir", str(source),
            "--training-data", str(training),
            "--target-size", "8",
            "--negative-count", "2",
        )
        run("build", *common, "--output-dir", str(first))
        run("build", *common, "--output-dir", str(second))
        for name in ("validation.jsonl", "provenance.jsonl", "manifest.json"):
            assert file_hash(first / name) == file_hash(second / name)
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["selection"]["selected_repository_counts"] == {
            repository: 2 for repository in REPOSITORIES
        }
        assert manifest["assertions"]["selected_query_training_text_overlap"] == 0
        assert manifest["assertions"]["selected_positive_training_text_overlap"] == 0
        evidence = [json.loads(line) for line in (first / "provenance.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(item["query_id"] != "q-0-0" for item in evidence)
        assert all(item["positive_corpus_id"] != "d-1-0" for item in evidence)
        assert all("d-1-0" not in item["negative_corpus_ids"] for item in evidence)
        run("verify", *common, "--output-dir", str(first))


def test_verify_fails_closed_after_output_corruption() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source"
        source.mkdir()
        write_source(source)
        output = root / "output"
        run("build", "--source-dir", str(source), "--output-dir", str(output), "--target-size", "8", "--negative-count", "2")
        with (output / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        result = run("verify", "--source-dir", str(source), "--output-dir", str(output), "--target-size", "8", "--negative-count", "2", check=False)
        assert result.returncode != 0
        assert "Output hash mismatch" in result.stderr
