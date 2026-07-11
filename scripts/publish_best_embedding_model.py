#!/usr/bin/env python3
"""Validate, card, and resumably publish the selected merged embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from statistics import fmean
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SIONIC_ORDER = [
    "MIRACL",
    "MrTidy",
    "MLDR",
    "AutoRAG",
    "Ko-StrategyQA",
    "PublicHealthQA",
    "Belebele",
    "SQuADKorV1",
    "LawIRKo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sionic-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--clean-summary", type=Path)
    parser.add_argument("--robustness-summary", type=Path)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument(
        "--repo-id", default="LLM-OS-Models/qwen3-embedding-8b-ko-performance-v1"
    )
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_json_tree(source_root: Path, destination_root: Path) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    if not source_root.is_dir():
        return copied
    for source in sorted(source_root.rglob("*.json")):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "path": str(destination.relative_to(destination_root.parent.parent)),
                "sha256": sha256(destination),
            }
        )
    return copied


def model_weights_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    shards = sorted(root.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model safetensors under {root}")
    for shard in shards:
        digest.update(shard.name.encode() + b"\0")
        with shard.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def resolved_local_model(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Evaluation summary has no model path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(f"Evaluation summary model is not a local artifact: {value}")
    return path.resolve()


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    model_dir = args.model_dir.resolve()
    model_evidence_path = model_dir / "merge_report.json"
    if not model_evidence_path.is_file():
        model_evidence_path = model_dir / "full_tuning_report.json"
    required = [
        model_dir / "config.json",
        model_dir / "modules.json",
        model_dir / "1_Pooling/config.json",
        model_dir / "2_Normalize",
        model_evidence_path,
        args.sionic_summary.resolve(),
        args.official_summary.resolve(),
        args.training_manifest.resolve(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing publication evidence: {missing}")
    if not list(model_dir.glob("model*.safetensors")):
        raise FileNotFoundError(f"No merged safetensors weights under {model_dir}")
    model_evidence = read_json(model_evidence_path)
    sionic = read_json(args.sionic_summary.resolve())
    official = read_json(args.official_summary.resolve())
    clean = read_json(args.clean_summary.resolve()) if args.clean_summary else None
    robustness = (
        read_json(args.robustness_summary.resolve())
        if args.robustness_summary
        else None
    )
    training = read_json(args.training_manifest.resolve())
    if model_evidence.get("status") != "pass":
        raise ValueError("Model packaging/parity evidence did not pass")
    contract = model_evidence.get("sentence_transformers_contract", {})
    if contract.get("pooling") != "last_token" or contract.get("normalize") is not True:
        raise ValueError("Merged SentenceTransformers contract drifted")
    if sionic.get("completed_tasks") != 9 or set(sionic.get("scores", {})) != set(
        SIONIC_ORDER
    ):
        raise ValueError("Sionic-9 summary is incomplete")
    if official.get("complete") is not True or official.get("completed_tasks") != 6:
        raise ValueError("Official Korean v1 summary is incomplete")
    if sionic.get("protocol_id") != "sionic9-fixed-prompt-v1":
        raise ValueError("Unexpected Sionic protocol")
    if official.get("protocol_id") != "mteb-korean-v1-mteb-2.18.0":
        raise ValueError("Unexpected official Korean protocol")
    official_environment = official.get("environment", {})
    if (
        official_environment.get("qwen3_instruction_loader") is not True
        or official_environment.get("instruction_contract") != "qwen3-task-instruction"
    ):
        raise ValueError(
            "Official Korean candidate result did not use Qwen3 task instructions"
        )
    if resolved_local_model(sionic.get("model")) != model_dir:
        raise ValueError("Sionic summary belongs to a different model artifact")
    if resolved_local_model(official.get("model")) != model_dir:
        raise ValueError("Official summary belongs to a different model artifact")
    expected_revision = f"model-{model_evidence['model']['weights_sha256'][:12]}"
    for label, summary in (("Sionic", sionic), ("official", official)):
        if summary.get("requested_revision") != expected_revision:
            raise ValueError(f"{label} summary revision does not match model evidence")
    if clean is not None:
        if clean.get("protocol_id") != "legal-source-document-heldout-i-v1":
            raise ValueError("Unexpected clean legal protocol")
        if resolved_local_model(clean.get("model")) != model_dir:
            raise ValueError(
                "Clean legal summary belongs to a different model artifact"
            )
        if clean.get("requested_revision") != expected_revision:
            raise ValueError(
                "Clean legal summary revision does not match model evidence"
            )
        dataset = clean.get("dataset", {})
        if dataset.get("independence_grade") != "I" or dataset.get("not_grade") != "Z":
            raise ValueError("Clean legal independence evidence is invalid")
    if robustness is not None:
        if robustness.get("protocol_id") != "legal-conversational-noise-i-v1":
            raise ValueError("Unexpected conversational noise protocol")
        if resolved_local_model(robustness.get("model")) != model_dir:
            raise ValueError("Robustness summary belongs to a different model artifact")
        if robustness.get("requested_revision") != expected_revision:
            raise ValueError(
                "Robustness summary revision does not match model evidence"
            )
        robustness_dataset = robustness.get("dataset", {})
        if (
            robustness_dataset.get("independence_grade") != "I"
            or robustness_dataset.get("not_grade") != "Z"
        ):
            raise ValueError("Robustness independence evidence is invalid")
        expected_conditions = {
            f"prompt_{state}/noise_{ratio}"
            for state in ("on", "off")
            for ratio in ("0.00", "0.01", "0.05")
        }
        if set(robustness.get("conditions", {})) != expected_conditions:
            raise ValueError("Robustness summary has incomplete conditions")
        if clean is not None:
            clean_ndcg = float(clean["metrics"]["ndcg_at_10"])
            robustness_clean_ndcg = float(
                robustness["conditions"]["prompt_on/noise_0.00"]["ndcg_at_10"]
            )
            if abs(clean_ndcg - robustness_clean_ndcg) > 1e-12:
                raise ValueError("Clean and robustness prompt-on baselines disagree")
    recomputed_sionic = fmean(float(value) for value in sionic["scores"].values())
    if abs(float(sionic["average"]) - recomputed_sionic) > 1e-12:
        raise ValueError("Sionic average is inconsistent with task scores")
    official_task_mean = fmean(
        float(row["score"]) for row in official["scores"].values()
    )
    if (
        abs(float(official["mean_task_leaderboard_points"]) - 100 * official_task_mean)
        > 1e-9
    ):
        raise ValueError("Official Mean(Task) is inconsistent with task scores")
    means_by_type = official.get("means_by_type", {})
    if not means_by_type:
        raise ValueError("Official summary has no task-type means")
    official_type_mean = fmean(float(value) for value in means_by_type.values())
    if (
        abs(
            float(official["mean_task_type_leaderboard_points"])
            - 100 * official_type_mean
        )
        > 1e-9
    ):
        raise ValueError("Official Mean(Type) is inconsistent with type means")
    actual_model_sha = model_weights_sha256(model_dir)
    if model_evidence.get("model", {}).get("weights_sha256") != actual_model_sha:
        raise ValueError("Published model shards do not match model evidence")
    return model_evidence, sionic, official, training, clean, robustness


def is_full_update(evidence: dict[str, Any]) -> bool:
    return str(evidence.get("training_method", "")).startswith("partial-full")


def weights_sha(evidence: dict[str, Any]) -> str:
    if is_full_update(evidence):
        return str(evidence["model"]["weights_sha256"])
    return str(evidence["adapter"]["weights_sha256"])


def score_table(scores: dict[str, float], order: list[str]) -> str:
    lines = ["| Task | Score |", "|---|---:|"]
    lines.extend(f"| {name} | {float(scores[name]):.5f} |" for name in order)
    return "\n".join(lines)


def training_rows(manifest: dict[str, Any]) -> str:
    for key in ("built_rows", "rows", "output_rows", "configured_target_rows"):
        if key in manifest:
            return str(manifest[key])
    files = manifest.get("files", {})
    values = [value.get("rows") for value in files.values() if isinstance(value, dict)]
    return str(
        max((value for value in values if isinstance(value, int)), default="unknown")
    )


def training_dataset_repos(manifest: dict[str, Any]) -> list[str]:
    adaptation = str(manifest.get("benchmark_adaptation", ""))
    if adaptation.startswith("target-adapted") and "legal" in adaptation:
        return [
            "LLM-OS-Models/korean-legal-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-legal-retrieval-source-native-250k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "squad" in adaptation:
        return [
            "LLM-OS-Models/korean-embedding-sionic-squad-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-squad-train-60k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "health" in adaptation:
        return [
            "LLM-OS-Models/korean-embedding-sionic-health-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-health-100k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted") and "autorag" in adaptation:
        return [
            "LLM-OS-Models/korean-embedding-sionic-autorag-quantile-hn7-replay-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-sionic-autorag-100k",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    if adaptation.startswith("target-adapted"):
        return [
            "LLM-OS-Models/korean-embedding-performance-1m-quantile-hn7-v1",
            "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
        ]
    repo = {
        "pilot_50k": "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k",
        "ablation_200k": "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k",
        "performance_1m": "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
    }.get(manifest.get("phase"))
    if repo is None:
        train_path = str(manifest.get("inputs", {}).get("train", {}).get("path", ""))
        if "pilot-50k" in train_path:
            repo = "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k"
        elif "ablation-200k" in train_path:
            repo = "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k"
        elif "performance-1m" in train_path:
            repo = "LLM-OS-Models/korean-embedding-performance-v1-performance-1m"
    return [repo] if repo else []


def build_card(
    repo_id: str,
    evidence: dict[str, Any],
    sionic: dict[str, Any],
    official: dict[str, Any],
    training: dict[str, Any],
    clean: dict[str, Any] | None,
    robustness: dict[str, Any] | None,
) -> str:
    delta = float(sionic["average"]) - 0.793
    full_update = is_full_update(evidence)
    adapter = evidence.get("adapter_config", {})
    merge_dtype = str(evidence.get("merge", {}).get("dtype", "bfloat16"))
    torch_dtype = "torch.float32" if merge_dtype == "float32" else "torch.bfloat16"
    official_order = list(official["scores"])
    dataset_repos = training_dataset_repos(training)
    dataset_yaml = (
        "datasets:\n" + "".join(f"- {repo}\n" for repo in dataset_repos)
        if dataset_repos
        else ""
    )
    dataset_link = (
        ", ".join(f"https://huggingface.co/datasets/{repo}" for repo in dataset_repos)
        if dataset_repos
        else "Training manifest is preserved with the model evaluation artifacts."
    )
    adaptation = str(training.get("benchmark_adaptation", ""))
    target_adapted = adaptation.startswith("target-adapted")
    if target_adapted and "legal" in adaptation:
        adaptation_notice = (
            "**이 모델은 법률/공공 target-adapted 모델이다. LawIRKo와 AutoRAG "
            "legal/public 점수를 clean zero-shot으로 해석하면 안 된다.**"
        )
    elif target_adapted:
        adaptation_notice = (
            "**이 모델은 공개 train/task-family와 current-student hard-negative를 사용한 "
            "performance target-adapted 모델이다. 관련 MTEB/Sionic 점수를 완전한 clean "
            "zero-shot으로 해석하면 안 된다.**"
        )
    else:
        adaptation_notice = "이 모델의 task-family 학습 노출은 아래와 같이 공개한다."
    method_intro = (
        "Qwen3-Embedding-8B의 상위 transformer block을 부분 full-parameter update한 "
        "한국어 retrieval 성능 후보다. optimizer state를 제외한 SentenceTransformers "
        "artifact를 만들고 last-token/L2 계약과 실제 embedding probe를 검증했다."
        if full_update
        else "Qwen3-Embedding-8B를 한국어 retrieval용 contrastive fine-tuning한 연구·비상업 "
        "성능 후보다. PEFT adapter를 base에 safe-merge하고 병합 전후 embedding parity와 "
        "SentenceTransformers last-token/L2/prompt 계약을 검증했다."
    )
    clean_section = ""
    if clean is not None:
        clean_metrics = clean["metrics"]
        clean_section = f"""
### Clean 법률 source-document-held-out 10K

- NDCG@10: **{float(clean_metrics['ndcg_at_10']):.5f}**
- Recall@10: **{float(clean_metrics['recall_at_10']):.5f}**
- MRR@10: **{float(clean_metrics['mrr_at_10']):.5f}**
- Recall@100: **{float(clean_metrics['recall_at_100']):.5f}**
- independence: `I` (same-repository whole-source-document-held-out), **not Z**

각 query에 source-native positive qrel 하나만 있어 relevance judgment는 exhaustive하지 않다.
"""
    robustness_section = ""
    if robustness is not None:
        conditions = robustness["conditions"]
        on_5 = conditions["prompt_on/noise_0.05"]
        off_5 = conditions["prompt_off/noise_0.05"]
        robustness_section = f"""
### 대화형 구조 노이즈 강건성

| Query | Noise ratio | NDCG@10 | Clean 대비 유지율 | Noise intrusion@10 |
|---|---:|---:|---:|---:|
| prompt on | 5% | {float(on_5['ndcg_at_10']):.5f} | {float(on_5['ndcg_retention_vs_same_prompt_clean']):.5f} | {float(on_5['noise_intrusion_at_10']):.5f} |
| prompt off | 5% | {float(off_5['ndcg_at_10']):.5f} | {float(off_5['ndcg_retention_vs_same_prompt_clean']):.5f} | {float(off_5['noise_intrusion_at_10']):.5f} |

고정된 filler/system/assistant artifact를 clean corpus의 5%만큼 추가한 paired test다.
0/1/5% 전체 condition과 per-query rank는 `evaluation/`에 동봉한다.
"""
    if full_update:
        method_rows = f"""- base: `{evidence['base_model']}@{evidence['base_revision']}`
- method: partial full-parameter contrastive fine-tuning, InfoNCE/explicit negatives
- packaged model weight SHA-256: `{weights_sha(evidence)}`
- packaged probe maximum norm error: `{evidence['probe']['metrics']['maximum_norm_error']}`
- packaged probe positive margin: `{evidence['probe']['metrics']['positive_margin']}`"""
    else:
        method_rows = f"""- base: `{evidence['base_model']}@{evidence['base_revision']}`
- method: LoRA continued contrastive fine-tuning, InfoNCE/explicit negatives
- LoRA rank/alpha/dropout: `{adapter.get('r')}` / `{adapter.get('lora_alpha')}` / `{adapter.get('lora_dropout')}`
- target modules: `{', '.join(adapter.get('target_modules') or [])}`
- adapter weight SHA-256: `{weights_sha(evidence)}`
- merge requested/effective dtype: `{evidence.get('merge', {}).get('requested_dtype', merge_dtype)}` / `{merge_dtype}`
- actual trainer rows after tokenization/filtering: `{evidence.get('adapter', {}).get('training', {}).get('actual_train_rows', 'not recorded')}`
- merge minimum probe cosine: `{evidence['probe']['metrics']['minimum_row_cosine']}`
- merge maximum pairwise score delta: `{evidence['probe']['metrics']['maximum_pairwise_score_difference']}`"""
    return f"""---
language:
- ko
- en
license: other
library_name: sentence-transformers
pipeline_tag: feature-extraction
base_model: Qwen/Qwen3-Embedding-8B
{dataset_yaml.rstrip()}
tags:
- sentence-transformers
- text-embeddings-inference
- vllm
- korean
- retrieval
---

# {repo_id.split('/')[-1]}

{method_intro}

{adaptation_notice}

## 결과

### Sionic Korean retrieval 9종

동일한 고정 query prompt, 각 task NDCG@10, 9개 macro average다.

{score_table(sionic['scores'], SIONIC_ORDER)}

- 9-task average: **{float(sionic['average']):.5f}**
- Comsat 카드의 0.7930 대비: **{delta:+.5f}**
- protocol: `{sionic['protocol_id']}`
- model revision evidence SHA: `{evidence['model']['weights_sha256']}`

### 공식 MTEB Korean v1 로컬 재현

{score_table({name: row['score'] for name, row in official['scores'].items()}, official_order)}

- Mean(Task): **{float(official['mean_task_leaderboard_points']):.3f}**
- Mean(Type): **{float(official['mean_task_type_leaderboard_points']):.3f}**
- protocol: `{official['protocol_id']}`
- instruction contract: `qwen3-task-instruction` (pinned MTEB task metadata/fallback,
  query에만 Qwen3 template 적용, passage 무지시문)

이 결과는 pinned MTEB protocol의 로컬 실행이며 MTEB leaderboard 제출 행 자체는
아니다. task별 MTEB raw result JSON은 이 model repository의 `evaluation/raw/`에,
실행 코드는 프로젝트 repository에 보존한다.

{clean_section}
{robustness_section}

## 학습

{method_rows}
- training manifest phase: `{training.get('phase', training.get('purpose', 'documented in manifest'))}`
- manifest rows: `{training_rows(training)}`

학습 데이터에는 official train/task-family source가 포함될 수 있다. Sionic 9에서는
MIRACL, MrTidy, MLDR, Ko-StrategyQA 계열 노출을 명시하며, official Korean v1 결과를
완전한 zero-shot이라고 주장하지 않는다. 데이터 source의 license가 혼재하므로 이
모델 카드의 `other`는 upstream 권리를 재허가하지 않는다.

## SentenceTransformers 사용법

```python
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "{repo_id}",
    model_kwargs={{
        "attn_implementation": "flash_attention_2",
        "torch_dtype": {torch_dtype},
    }},
    tokenizer_kwargs={{"padding_side": "left"}},
)
queries = model.encode(
    ["대한민국의 수도는 어디인가?"],
    prompt_name="query",
    normalize_embeddings=True,
)
documents = model.encode(
    ["대한민국의 수도는 서울특별시이다."],
    normalize_embeddings=True,
)
scores = queries @ documents.T
```

query에는 model의 `query` prompt를 적용하고 document에는 instruction을 붙이지 않는다.
출력은 4,096차원 L2-normalized vector이므로 dot product가 cosine similarity다.

## vLLM API 서빙

```bash
MODEL_ID={repo_id} \\
SERVED_MODEL_NAME=qwen3-embedding-8b-ko \\
MAX_MODEL_LEN=8192 \\
DTYPE={merge_dtype} \\
scripts/serve_vllm_embedding.sh
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
result = client.embeddings.create(
    model="qwen3-embedding-8b-ko",
    input=[
        "Instruct: Given a Korean web search query, retrieve relevant passages that answer the query\\nQuery: 질문",
        "검색할 문서",
    ],
)
```

공개 점수 재현에는 model card의 evaluation dtype
(`{sionic.get('environment', {}).get('torch_dtype', merge_dtype)}`)을 유지한다. 다른
dtype은 별도 parity/회귀 측정 없이 같은 점수라고 간주하지 않는다. vLLM pooling은
동시 요청 API에 적합하지만 offline 고정 대량 corpus에서는 항상
SentenceTransformers+FlashAttention 2보다 빠르지 않다. 실제 traffic으로 두 경로를
benchmark한다.

## 재현

- code: https://github.com/LLM-OS-Models/Embedding
- data: {dataset_link}
- base: https://huggingface.co/Qwen/Qwen3-Embedding-8B
- comparison: https://huggingface.co/sionic-ai/comsat-embed-ko-8b-preview

모델 선택·평가·데이터 노출과 exact command는 repository의 README와 docs에 기록돼
있다. 평가 test row를 학습 또는 hard-negative mining에 되먹이지 않는다.

## 제한

- 한국어 retrieval specialist이며 모든 언어·task에서 base보다 낫다고 보장하지 않는다.
- legal/public target-like 데이터가 포함된 후속 버전은 LawIRKo/AutoRAG에서 반드시
  target-adapted로 별도 표시한다.
- 8B/4096-d vector는 품질은 높지만 serving·storage 비용이 크다. MRL 차원 축소는 해당
  dimension의 회귀 평가 후 사용한다.
"""


def main() -> None:
    args = parse_args()
    evidence, sionic, official, training, clean, robustness = validate(args)
    model_dir = args.model_dir.resolve()
    card = build_card(
        args.repo_id, evidence, sionic, official, training, clean, robustness
    )
    card_path = model_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    evidence_dir = model_dir / "evaluation"
    evidence_dir.mkdir(exist_ok=True)
    evidence_files = {
        "sionic9_summary.json": args.sionic_summary.resolve(),
        "mteb_korean_v1_summary.json": args.official_summary.resolve(),
        "training_manifest.json": args.training_manifest.resolve(),
    }
    if args.clean_summary:
        evidence_files["legal_source_heldout_summary.json"] = (
            args.clean_summary.resolve()
        )
        clean_ranks = args.clean_summary.resolve().parent / "ranks.jsonl"
        if clean_ranks.is_file():
            evidence_files["legal_source_heldout_ranks.jsonl"] = clean_ranks
    if args.robustness_summary:
        evidence_files["conversational_noise_summary.json"] = (
            args.robustness_summary.resolve()
        )
        robustness_ranks = args.robustness_summary.resolve().parent / "ranks.jsonl"
        if robustness_ranks.is_file():
            evidence_files["conversational_noise_ranks.jsonl"] = robustness_ranks
    for name, source in evidence_files.items():
        shutil.copy2(source, evidence_dir / name)
    raw_evidence = {
        "sionic9": copy_json_tree(
            args.sionic_summary.resolve().parent / "mteb_cache",
            evidence_dir / "raw" / "sionic9",
        ),
        "mteb_korean_v1": copy_json_tree(
            args.official_summary.resolve().parent / "mteb_cache",
            evidence_dir / "raw" / "mteb_korean_v1",
        ),
    }
    evidence_name = (
        "full_tuning_report.json" if is_full_update(evidence) else "merge_report.json"
    )
    publication_manifest = {
        "schema_version": 1,
        "repo_id": args.repo_id,
        "model_dir": str(model_dir),
        "model_evidence": {
            "file": evidence_name,
            "sha256": sha256(model_dir / evidence_name),
        },
        "card_sha256": sha256(card_path),
        "evidence": {
            name: {"sha256": sha256(evidence_dir / name)} for name in evidence_files
        },
        "raw_evaluation_json": raw_evidence,
        "model_weights_evidence_sha256": evidence["model"]["weights_sha256"],
    }
    (model_dir / "publication_manifest.json").write_text(
        json.dumps(publication_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "repo_id": args.repo_id,
        "model_dir": str(model_dir),
        "card": str(card_path),
        "publication_manifest": str(model_dir / "publication_manifest.json"),
        "sionic9_average": sionic["average"],
        "official_mean_task": official["mean_task_leaderboard_points"],
        "visibility": "public" if args.public else "private",
        "upload_requested": args.upload,
        "validated": True,
    }
    if args.upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be exported")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=not args.public,
            exist_ok=True,
        )
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=model_dir,
            private=not args.public,
            num_workers=4,
            print_report_every=60,
        )
        report["url"] = f"https://huggingface.co/{args.repo_id}"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
