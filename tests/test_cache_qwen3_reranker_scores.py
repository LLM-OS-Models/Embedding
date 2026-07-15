from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import cache_qwen3_reranker_scores as scorer


TOKEN_CANARY = "hf_" + "TEST_CANARY_NOT_A_REAL_TOKEN_123456"


def canonical_rows(rows: list[dict]) -> str:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def fixture_rows() -> list[dict]:
    return [
        {
            "generated_id": "gsq-fixture-001",
            "query": f"환불 규정은 무엇인가? {TOKEN_CANARY}",
            "positive": {
                "candidate_id": "doc-positive-001",
                "text": "구매 후 7일 안에는 환불할 수 있다. /private/cache/hidden",
                "retriever_score": 0.91,
            },
            "candidates": [
                {
                    "candidate_id": "doc-negative-001",
                    "text": "배송은 영업일 기준 3일이 걸린다.",
                    "retriever_score": 0.81,
                },
                {
                    "candidate_id": "doc-negative-002",
                    "text": "회원 가입에는 이메일 주소가 필요하다.",
                },
            ],
        },
        {
            "generated_id": "gsq-fixture-002",
            "query": "민원 결과 통지는 며칠 안에 해야 하나?",
            "positive": {
                "candidate_id": "doc-positive-002",
                "text": "접수한 날부터 14일 이내에 결과를 통지한다.",
            },
            "candidates": [
                {
                    "candidate_id": "doc-negative-003",
                    "text": "정보는 전자문서로 공개할 수 있다.",
                }
            ],
        },
        {
            "generated_id": "gsq-fixture-003",
            "query": "중력은 무엇인가?",
            "positive": {
                "candidate_id": "doc-positive-003",
                "text": "중력은 질량을 가진 물체 사이의 인력이다.",
            },
            "candidates": [
                {
                    "candidate_id": "doc-negative-004",
                    "text": "빛은 진공에서 일정한 속도로 이동한다.",
                }
            ],
        },
    ]


def write_input(path: Path, rows: list[dict] | None = None) -> None:
    path.write_text(canonical_rows(rows or fixture_rows()), encoding="utf-8")


def make_options(root: Path, **changes) -> scorer.CacheOptions:
    values = {
        "input_path": root / "requests.jsonl",
        "output_dir": root / "cache",
        "shard_size": 1,
        "model_batch_size": 2,
        "max_documents_per_row": 8,
        "max_text_characters": 10_000,
        "max_length": 512,
    }
    values.update(changes)
    return scorer.CacheOptions(**values)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FailingBackend:
    def __init__(self, delegate: scorer.DeterministicMockBackend, fail_on_call: int):
        self.delegate = delegate
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.score_field = delegate.score_field

    def scorer_provenance(self):
        return self.delegate.scorer_provenance()

    def runtime_provenance(self):
        return self.delegate.runtime_provenance()

    def score(self, instruction, query, documents):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise scorer.ScoreCacheError("fixture_failure", "injected scorer failure")
        return self.delegate.score(instruction, query, documents)


class NonFiniteBackend(FailingBackend):
    def __init__(self, delegate: scorer.DeterministicMockBackend):
        super().__init__(delegate, fail_on_call=10_000)

    def score(self, instruction, query, documents):
        return [scorer.RawLogits(no=math.nan, yes=0.0) for _ in documents]


class Qwen3RerankerScoreCacheTests(unittest.TestCase):
    maxDiff = None

    def test_official_prompt_and_stable_yes_probability(self) -> None:
        formatted = scorer.format_instruction("task", "query", "document")
        self.assertEqual(
            formatted,
            "<Instruct>: task\n<Query>: query\n<Document>: document",
        )
        self.assertAlmostEqual(scorer.normalized_yes_probability(0.0, 0.0), 0.5)
        self.assertAlmostEqual(scorer.normalized_yes_probability(-1000.0, 1000.0), 1.0)
        self.assertAlmostEqual(scorer.normalized_yes_probability(1000.0, -1000.0), 0.0)
        provenance = scorer.build_scorer_provenance(
            backend="fixture",
            instruction=scorer.DEFAULT_INSTRUCTION,
            max_length=8192,
            token_no_id=10,
            token_yes_id=11,
            dtype="bfloat16",
            attention_implementation="sdpa",
        )
        self.assertEqual(provenance["token_no"], "no")
        self.assertEqual(provenance["token_yes"], "yes")
        self.assertIn("next-token logits", provenance["raw_logit_semantics"])
        self.assertEqual(
            provenance["score_semantics"], "normalized yes-token probability in [0,1]"
        )

    def test_offline_environment_removes_inherited_credentials(self) -> None:
        env = {
            "HF_TOKEN": "secret",
            "HUGGINGFACE_HUB_TOKEN": "secret-two",
            "GITHUB": "secret-three",
            "HF_HUB_OFFLINE": "0",
        }
        evidence = scorer.enforce_offline_no_credentials(env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(env["HF_HUB_DISABLE_TELEMETRY"], "1")
        self.assertFalse(
            {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "GITHUB"}.intersection(env)
        )
        self.assertTrue(evidence["offline"])

    def test_production_backend_enforces_model_microbatch_size(self) -> None:
        import torch

        class FakeTokenizer:
            def __init__(self) -> None:
                self.batch_sizes = []

            def __call__(self, formatted, **_kwargs):
                self.batch_sizes.append(len(formatted))
                return {"input_ids": [[index + 2] for index, _ in enumerate(formatted)]}

            def pad(self, inputs, **_kwargs):
                return {
                    "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long)
                }

        class FakeModel:
            device = "cpu"

            def __init__(self) -> None:
                self.batch_sizes = []

            def __call__(self, **inputs):
                batch = inputs["input_ids"].shape[0]
                self.batch_sizes.append(batch)
                logits = torch.zeros((batch, 1, 2), dtype=torch.float32)
                logits[:, :, 0] = -1.0
                logits[:, :, 1] = 1.0
                return type("Output", (), {"logits": logits})()

        options = scorer.CacheOptions(
            input_path=Path("fixture"),
            output_dir=Path("fixture-output"),
            model_batch_size=2,
            max_length=128,
        )
        backend = object.__new__(scorer.Qwen3RerankerBackend)
        backend.options = options
        backend.torch = torch
        backend.tokenizer = FakeTokenizer()
        backend.model = FakeModel()
        backend.token_no_id = 0
        backend.token_yes_id = 1
        backend.prefix_tokens = [10]
        backend.suffix_tokens = [11]
        scores = backend.score("task", "query", [f"doc-{i}" for i in range(5)])
        self.assertEqual(backend.tokenizer.batch_sizes, [2, 2, 1])
        self.assertEqual(backend.model.batch_sizes, [2, 2, 1])
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(score.no == -1.0 and score.yes == 1.0 for score in scores))

    def test_mock_end_to_end_is_deterministic_private_and_non_admissible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = make_options(root)
            write_input(options.input_path)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                state = scorer.run_score_cache(
                    options, scorer.DeterministicMockBackend(options)
                )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["next_row"], 3)
            self.assertEqual(len(state["completed_shards"]), 3)
            manifest = json.loads(
                (options.output_dir / scorer.MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["admissible_for_training"])
            self.assertEqual(manifest["input"]["rows"], 3)
            self.assertEqual(
                manifest["input"]["sha256"],
                hashlib.sha256(options.input_path.read_bytes()).hexdigest(),
            )
            output = options.output_dir / scorer.SCORES_NAME
            score_rows = read_jsonl(output)
            self.assertEqual(
                [row["generated_id"] for row in score_rows],
                [row["generated_id"] for row in fixture_rows()],
            )
            for source, scored in zip(fixture_rows(), score_rows, strict=True):
                self.assertEqual(scored["score_field"], scorer.MOCK_SCORE_FIELD)
                self.assertEqual(scored["scorer"]["revision"], scorer.MODEL_REVISION)
                self.assertTrue(scored["scorer"]["local_files_only"])
                self.assertFalse(scored["scorer"]["trust_remote_code"])
                self.assertEqual(
                    scored["query_sha256"],
                    hashlib.sha256(source["query"].encode()).hexdigest(),
                )
                self.assertNotIn("query", scored)
                expected_documents = [source["positive"], *source["candidates"]]
                for raw_source, document in zip(
                    expected_documents, scored["documents"], strict=True
                ):
                    self.assertNotIn("text", document)
                    self.assertEqual(
                        document["text_sha256"],
                        hashlib.sha256(raw_source["text"].encode()).hexdigest(),
                    )
                    expected_probability = scorer.normalized_yes_probability(
                        document["raw_no_logit"], document["raw_yes_logit"]
                    )
                    self.assertEqual(
                        document[scorer.MOCK_SCORE_FIELD], expected_probability
                    )
            first_sha = hashlib.sha256(output.read_bytes()).hexdigest()
            with contextlib.redirect_stdout(io.StringIO()):
                repeated = scorer.run_score_cache(
                    options, scorer.DeterministicMockBackend(options)
                )
            self.assertEqual(first_sha, repeated["output"]["scores"]["sha256"])
            verified_with_stored_contract = scorer.verify_complete_artifacts(
                scorer.CacheOptions(
                    input_path=options.input_path,
                    output_dir=options.output_dir,
                )
            )
            self.assertEqual(
                verified_with_stored_contract["output"]["scores"]["sha256"],
                first_sha,
            )

            second_options = make_options(root, output_dir=root / "second-cache")
            with contextlib.redirect_stdout(io.StringIO()):
                second = scorer.run_score_cache(
                    second_options, scorer.DeterministicMockBackend(second_options)
                )
            self.assertEqual(first_sha, second["output"]["scores"]["sha256"])

            emitted_artifacts = b"".join(
                path.read_bytes()
                for path in options.output_dir.rglob("*")
                if path.is_file()
            )
            self.assertNotIn(str(root).encode(), emitted_artifacts)
            self.assertNotIn(TOKEN_CANARY.encode(), emitted_artifacts)
            self.assertNotIn(b"/private/cache/hidden", emitted_artifacts)
            self.assertNotIn(str(root), stream.getvalue())

    def test_interrupted_run_resumes_only_after_completed_atomic_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = make_options(root)
            write_input(options.input_path)
            mock_backend = scorer.DeterministicMockBackend(options)
            failing = FailingBackend(mock_backend, fail_on_call=3)
            with self.assertRaisesRegex(scorer.ScoreCacheError, "injected"):
                with contextlib.redirect_stdout(io.StringIO()):
                    scorer.run_score_cache(options, failing)
            state = json.loads((options.output_dir / scorer.STATE_NAME).read_text())
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["next_row"], 1)
            self.assertEqual(len(state["completed_shards"]), 1)
            self.assertEqual(
                list((options.output_dir / scorer.SHARD_DIRECTORY).glob("*.jsonl")),
                [options.output_dir / scorer.SHARD_DIRECTORY / "part-000000.jsonl"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = scorer.run_score_cache(options, mock_backend)
            self.assertEqual(resumed["next_row"], 3)

            fresh_options = make_options(root, output_dir=root / "fresh")
            with contextlib.redirect_stdout(io.StringIO()):
                fresh = scorer.run_score_cache(
                    fresh_options, scorer.DeterministicMockBackend(fresh_options)
                )
            self.assertEqual(
                resumed["output"]["scores"]["sha256"],
                fresh["output"]["scores"]["sha256"],
            )

    def test_atomic_orphan_shard_is_validated_and_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = fixture_rows()[:2]
            options = make_options(root)
            write_input(options.input_path, rows)
            backend = scorer.DeterministicMockBackend(options)
            with contextlib.redirect_stdout(io.StringIO()):
                complete = scorer.run_score_cache(options, backend)
            state = json.loads((options.output_dir / scorer.STATE_NAME).read_text())
            orphan = state["completed_shards"].pop()
            state["next_row"] = orphan["start_row"]
            state["status"] = "running"
            state.pop("output")
            scorer.atomic_write(
                options.output_dir / scorer.STATE_NAME,
                scorer.canonical_json_bytes(state),
            )
            (options.output_dir / scorer.SCORES_NAME).unlink()
            (options.output_dir / scorer.MANIFEST_NAME).unlink()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                recovered = scorer.run_score_cache(options, backend)
            self.assertEqual(recovered["status"], "complete")
            self.assertEqual(
                recovered["output"]["scores"]["sha256"],
                complete["output"]["scores"]["sha256"],
            )
            self.assertIn("orphan_shard_recovered", stream.getvalue())

    def test_tamper_and_nonfinite_scores_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = make_options(root)
            write_input(options.input_path)
            with contextlib.redirect_stdout(io.StringIO()):
                scorer.run_score_cache(
                    options, scorer.DeterministicMockBackend(options)
                )
            shard = options.output_dir / scorer.SHARD_DIRECTORY / "part-000000.jsonl"
            shard.write_bytes(shard.read_bytes() + b"{}\n")
            with self.assertRaises(scorer.ScoreCacheError):
                scorer.verify_complete_artifacts(options)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = make_options(root)
            write_input(options.input_path)
            with self.assertRaisesRegex(scorer.ScoreCacheError, "finite"):
                with contextlib.redirect_stdout(io.StringIO()):
                    scorer.run_score_cache(
                        options,
                        NonFiniteBackend(scorer.DeterministicMockBackend(options)),
                    )
            self.assertFalse((options.output_dir / scorer.SCORES_NAME).exists())
            self.assertEqual(
                list((options.output_dir / scorer.SHARD_DIRECTORY).glob("*.jsonl")),
                [],
            )

    def test_strict_input_schema_duplicate_and_nonfinite_validation(self) -> None:
        mutations: list[tuple[str, list[dict] | str, str]] = []
        unknown = fixture_rows()[:1]
        unknown[0]["unexpected"] = True
        mutations.append(("unknown", unknown, "unknown fields"))
        duplicate_id = fixture_rows()[:2]
        duplicate_id[1]["generated_id"] = duplicate_id[0]["generated_id"]
        mutations.append(("duplicate-generated", duplicate_id, "repeats generated_id"))
        duplicate_candidate = fixture_rows()[:1]
        duplicate_candidate[0]["candidates"][0]["candidate_id"] = duplicate_candidate[
            0
        ]["positive"]["candidate_id"]
        mutations.append(
            ("duplicate-candidate", duplicate_candidate, "repeats a candidate_id")
        )
        nonfinite = fixture_rows()[:1]
        nonfinite[0]["candidates"][0]["retriever_score"] = float("inf")
        mutations.append(("nonfinite", nonfinite, "invalid"))
        mutations.append(
            (
                "duplicate-json-key",
                '{"generated_id":"a","generated_id":"b","query":"q","positive":{"candidate_id":"p","text":"p"},"candidates":[{"candidate_id":"n","text":"n"}]}\n',
                "invalid",
            )
        )
        for label, payload, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                options = make_options(root)
                if isinstance(payload, str):
                    options.input_path.write_text(payload, encoding="utf-8")
                else:
                    # allow_nan=True here intentionally exercises strict reader rejection.
                    options.input_path.write_text(
                        "".join(
                            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                            for row in payload
                        ),
                        encoding="utf-8",
                    )
                with self.assertRaisesRegex(scorer.ScoreCacheError, message):
                    scorer.preflight_input(options.input_path, options)

    def test_dry_run_loads_no_model_writes_nothing_and_emits_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = make_options(root)
            write_input(options.input_path)
            stream = io.StringIO()
            with mock.patch.object(
                scorer.Qwen3RerankerBackend,
                "__init__",
                side_effect=AssertionError("model must not load"),
            ), contextlib.redirect_stdout(stream):
                status = scorer.cli_main(
                    [
                        "--input",
                        str(options.input_path),
                        "--output-dir",
                        str(options.output_dir),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertFalse(options.output_dir.exists())
            self.assertIn('"model_loaded":false', stream.getvalue())
            self.assertNotIn(str(root), stream.getvalue())

    def test_mock_cli_requires_explicit_nonproduction_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = make_options(root)
            write_input(options.input_path)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                status = scorer.cli_main(
                    [
                        "--input",
                        str(options.input_path),
                        "--output-dir",
                        str(options.output_dir),
                        "--backend",
                        "mock",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertFalse(options.output_dir.exists())
            self.assertIn("mock_not_acknowledged", stream.getvalue())
            self.assertNotIn(str(root), stream.getvalue())

    def test_pinned_snapshot_resolution_is_local_only_token_free_and_exact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "hub"
            snapshot = (
                cache
                / "models--Qwen--Qwen3-Reranker-8B"
                / "snapshots"
                / scorer.MODEL_REVISION
            )
            blobs = cache / "models--Qwen--Qwen3-Reranker-8B" / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            (snapshot / "config.json").write_text(
                json.dumps(
                    {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
                ),
                encoding="utf-8",
            )
            (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            blob_names = [
                hashlib.sha256(value).hexdigest() for value in (b"one", b"two")
            ]
            for blob_name, value in zip(blob_names, (b"one", b"two"), strict=True):
                (blobs / blob_name).write_bytes(value)
            for shard_name, blob_name in zip(
                (
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ),
                blob_names,
                strict=True,
            ):
                (snapshot / shard_name).symlink_to(
                    os.path.relpath(blobs / blob_name, snapshot)
                )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_snapshot_download(**kwargs):
                calls.append(kwargs)
                return str(snapshot)

            resolved, evidence = scorer.resolve_pinned_snapshot(
                fake_snapshot_download,
                cache_dir=cache,
                verify_weight_content=True,
            )
            self.assertEqual(resolved, snapshot.resolve())
            self.assertEqual(evidence["weight_shard_count"], 2)
            self.assertTrue(evidence["weight_content_verified"])
            self.assertEqual(
                evidence["weight_shard_bytes"],
                {
                    "model-00001-of-00002.safetensors": 3,
                    "model-00002-of-00002.safetensors": 3,
                },
            )
            self.assertEqual(
                sorted(evidence["weight_shard_sha256"].values()), sorted(blob_names)
            )
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": scorer.MODEL_ID,
                        "revision": scorer.MODEL_REVISION,
                        "cache_dir": cache,
                        "local_files_only": True,
                        "token": False,
                    }
                ],
            )
            self.assertFalse(
                scorer.Qwen3RerankerBackend.tokenizer_load_kwargs()["trust_remote_code"]
            )
            self.assertTrue(
                scorer.Qwen3RerankerBackend.tokenizer_load_kwargs()["local_files_only"]
            )
            model_kwargs = scorer.Qwen3RerankerBackend.model_load_kwargs(
                torch_dtype="fixture", attention_implementation="sdpa"
            )
            self.assertFalse(model_kwargs["trust_remote_code"])
            self.assertTrue(model_kwargs["local_files_only"])
            self.assertFalse(model_kwargs["token"])


if __name__ == "__main__":
    unittest.main()
