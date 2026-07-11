from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "grounded_synthetic_query_factory.py"
CONFIG = ROOT / "configs" / "synthetic_query_factory_v1.json"
FIXTURE = ROOT / "tests" / "fixtures" / "grounded_synthetic_query_factory" / "candidates.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroundedSyntheticFactorySmokeTest(unittest.TestCase):
    maxDiff = None

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--config", str(CONFIG), *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_offline_generation_teacher_selection_and_verification(self) -> None:
        candidates = read_jsonl(FIXTURE)
        by_id = {row["id"]: row for row in candidates}
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            requests = work / "requests.jsonl"
            responses = work / "responses.jsonl"
            validated = work / "validated.jsonl"
            rejected = work / "rejected.jsonl"
            scores = work / "scores.jsonl"
            output = work / "train.jsonl"
            audit = work / "audit.jsonl"
            manifest = work / "manifest.json"

            self.run_cli(
                "prepare",
                "--input",
                str(FIXTURE),
                "--requests",
                str(requests),
                "--max-candidates",
                "2",
                "--style",
                "natural_question",
                "--style",
                "citation_lookup",
            )
            request_rows = read_jsonl(requests)
            self.assertEqual(len(request_rows), 4)
            response_rows = []
            for request in request_rows:
                candidate_id = request["source_candidate_id"]
                if candidate_id == "fixture-law-001":
                    evidence = "담당 기관은 신청서를 접수한 날부터 14일 이내에 처리 결과를 신청인에게 통지하여야 한다."
                    answer = "14일 이내"
                    if request["style"] == "natural_question":
                        query = "민원 신청 결과는 접수 후 며칠 안에 알려줘야 하나요?"
                    else:
                        query = "가상민원처리규정 제3조의 처리 기한 근거를 찾아줘"
                else:
                    evidence = "공개 대상 정보는 전자문서로 제공하며, 신청인이 요청하면 서면으로도 제공할 수 있다."
                    answer = "전자문서"
                    if request["style"] == "natural_question":
                        query = "공개 대상 정보는 기본적으로 어떤 형태로 받을 수 있나요?"
                    else:
                        query = "가상정보공개규정 제5조의 공개 방법 근거를 찾아줘"
                response_rows.append(
                    {
                        "request_id": request["request_id"],
                        "response": {
                            "query": query,
                            "answer": answer,
                            "evidence_quote": evidence,
                            "citation": {
                                "source_candidate_id": candidate_id,
                                "locator": request["allowed_citation_locators"][0],
                            },
                        },
                    }
                )
            write_jsonl(responses, response_rows)
            self.run_cli(
                "generate",
                "--requests",
                str(requests),
                "--mode",
                "offline",
                "--responses",
                str(responses),
                "--validated",
                str(validated),
                "--rejected",
                str(rejected),
            )
            generated_rows = read_jsonl(validated)
            self.assertEqual(len(generated_rows), 4)
            self.assertEqual(read_jsonl(rejected), [])

            score_rows = []
            candidate_ids = sorted(by_id)
            for generated in generated_rows:
                positive_id = generated["source_candidate_id"]
                negatives = [item for item in candidate_ids if item != positive_id]
                documents = [
                    {"candidate_id": positive_id, "reranker_score": 0.99}
                ] + [
                    {"candidate_id": candidate_id, "reranker_score": 0.84 - index * 0.01}
                    for index, candidate_id in enumerate(negatives)
                ]
                score_rows.append(
                    {
                        "generated_id": generated["generated_id"],
                        "score_field": "reranker_score",
                        "scorer": {
                            "model": "fixture-reranker",
                            "revision": "fixture-v1",
                            "score_semantics": "normalized relevance probability",
                        },
                        "documents": documents,
                    }
                )
            write_jsonl(scores, score_rows)
            self.run_cli(
                "compile",
                "--candidates",
                str(FIXTURE),
                "--validated",
                str(validated),
                "--scores",
                str(scores),
                "--output",
                str(output),
                "--audit",
                str(audit),
                "--manifest",
                str(manifest),
                "--work-dir",
                str(work / "index"),
            )
            self.run_cli(
                "verify",
                "--output",
                str(output),
                "--audit",
                str(audit),
                "--manifest",
                str(manifest),
            )
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "validate_embedding_jsonl.py"), str(output)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            train_rows = read_jsonl(output)
            audit_rows = read_jsonl(audit)
            self.assertEqual(len(train_rows), 4)
            self.assertEqual(len(audit_rows), 4)
            self.assertTrue(all(len(row["negative_messages"]) == 7 for row in train_rows))

            second_output = work / "train-second.jsonl"
            second_audit = work / "audit-second.jsonl"
            second_manifest = work / "manifest-second.json"
            self.run_cli(
                "compile",
                "--candidates",
                str(FIXTURE),
                "--validated",
                str(validated),
                "--scores",
                str(scores),
                "--output",
                str(second_output),
                "--audit",
                str(second_audit),
                "--manifest",
                str(second_manifest),
                "--work-dir",
                str(work / "index"),
            )
            self.assertEqual(digest(output), digest(second_output))
            self.assertEqual(digest(audit), digest(second_audit))


if __name__ == "__main__":
    unittest.main()
