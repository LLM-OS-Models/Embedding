#!/usr/bin/env python3
"""Validate, card, and resumably publish the selected merged embedding model."""

from __future__ import annotations

import argparse
import json
import os
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


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    model_dir = args.model_dir.resolve()
    required = [
        model_dir / "config.json",
        model_dir / "modules.json",
        model_dir / "1_Pooling/config.json",
        model_dir / "2_Normalize",
        model_dir / "merge_report.json",
        args.sionic_summary.resolve(),
        args.official_summary.resolve(),
        args.training_manifest.resolve(),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing publication evidence: {missing}")
    if not list(model_dir.glob("model*.safetensors")):
        raise FileNotFoundError(f"No merged safetensors weights under {model_dir}")
    merge = read_json(model_dir / "merge_report.json")
    sionic = read_json(args.sionic_summary.resolve())
    official = read_json(args.official_summary.resolve())
    training = read_json(args.training_manifest.resolve())
    if merge.get("status") != "pass":
        raise ValueError("Merge parity did not pass")
    contract = merge.get("sentence_transformers_contract", {})
    if contract.get("pooling") != "last_token" or contract.get("normalize") is not True:
        raise ValueError("Merged SentenceTransformers contract drifted")
    if sionic.get("completed_tasks") != 9 or set(sionic.get("scores", {})) != set(
        SIONIC_ORDER
    ):
        raise ValueError("Sionic-9 summary is incomplete")
    if official.get("complete") is not True or official.get("completed_tasks") != 6:
        raise ValueError("Official Korean v1 summary is incomplete")
    return merge, sionic, official, training


def score_table(scores: dict[str, float], order: list[str]) -> str:
    lines = ["| Task | Score |", "|---|---:|"]
    lines.extend(f"| {name} | {float(scores[name]):.5f} |" for name in order)
    return "\n".join(lines)


def training_rows(manifest: dict[str, Any]) -> str:
    for key in ("built_rows", "rows", "configured_target_rows"):
        if key in manifest:
            return str(manifest[key])
    files = manifest.get("files", {})
    values = [value.get("rows") for value in files.values() if isinstance(value, dict)]
    return str(max((value for value in values if isinstance(value, int)), default="unknown"))


def training_dataset_repo(manifest: dict[str, Any]) -> str | None:
    if str(manifest.get("benchmark_adaptation", "")).startswith("target-adapted"):
        return "LLM-OS-Models/korean-legal-retrieval-source-native-250k"
    return {
        "pilot_50k": "LLM-OS-Models/korean-embedding-performance-v1-pilot-50k",
        "ablation_200k": "LLM-OS-Models/korean-embedding-performance-v1-ablation-200k",
        "performance_1m": "LLM-OS-Models/korean-embedding-performance-v1-performance-1m",
    }.get(manifest.get("phase"))


def build_card(
    repo_id: str,
    merge: dict[str, Any],
    sionic: dict[str, Any],
    official: dict[str, Any],
    training: dict[str, Any],
) -> str:
    delta = float(sionic["average"]) - 0.793
    adapter = merge["adapter_config"]
    official_order = list(official["scores"])
    dataset_repo = training_dataset_repo(training)
    dataset_yaml = f"datasets:\n- {dataset_repo}\n" if dataset_repo else ""
    dataset_link = (
        f"https://huggingface.co/datasets/{dataset_repo}"
        if dataset_repo
        else "Training manifest is preserved with the model evaluation artifacts."
    )
    target_adapted = str(training.get("benchmark_adaptation", "")).startswith(
        "target-adapted"
    )
    adaptation_notice = (
        "**이 모델은 법률/공공 target-adapted 모델이다. LawIRKo와 AutoRAG legal/public "
        "점수를 clean zero-shot으로 해석하면 안 된다.**"
        if target_adapted
        else "이 모델의 task-family 학습 노출은 아래와 같이 공개한다."
    )
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

Qwen3-Embedding-8B를 한국어 retrieval용 contrastive fine-tuning한 연구·비상업
성능 후보다. PEFT adapter를 base에 safe-merge하고 병합 전후 embedding parity와
SentenceTransformers last-token/L2/prompt 계약을 검증했다.

{adaptation_notice}

## 결과

### Sionic Korean retrieval 9종

동일한 고정 query prompt, 각 task NDCG@10, 9개 macro average다.

{score_table(sionic['scores'], SIONIC_ORDER)}

- 9-task average: **{float(sionic['average']):.5f}**
- Comsat 카드의 0.7930 대비: **{delta:+.5f}**
- protocol: `{sionic['protocol_id']}`
- model revision evidence: adapter SHA `{merge['adapter']['weights_sha256']}`

### 공식 MTEB Korean v1 로컬 재현

{score_table({name: row['score'] for name, row in official['scores'].items()}, official_order)}

- Mean(Task): **{float(official['mean_task_leaderboard_points']):.3f}**
- Mean(Type): **{float(official['mean_task_type_leaderboard_points']):.3f}**
- protocol: `{official['protocol_id']}`

이 결과는 pinned MTEB protocol의 로컬 실행이며 MTEB leaderboard 제출 행 자체는
아니다. raw JSON과 실행 코드는 프로젝트 repository에 보존한다.

## 학습

- base: `{merge['base_model']}@{merge['base_revision']}`
- method: LoRA continued contrastive fine-tuning, InfoNCE/explicit negatives
- LoRA rank/alpha/dropout: `{adapter.get('r')}` / `{adapter.get('lora_alpha')}` / `{adapter.get('lora_dropout')}`
- target modules: `{', '.join(adapter.get('target_modules') or [])}`
- training manifest phase: `{training.get('phase', training.get('purpose', 'documented in manifest'))}`
- manifest rows: `{training_rows(training)}`
- adapter weight SHA-256: `{merge['adapter']['weights_sha256']}`
- merge minimum probe cosine: `{merge['probe']['metrics']['minimum_row_cosine']}`
- merge maximum pairwise score delta: `{merge['probe']['metrics']['maximum_pairwise_score_difference']}`

학습 데이터에는 official train/task-family source가 포함될 수 있다. Sionic 9에서는
MIRACL, MrTidy, MLDR, Ko-StrategyQA 계열 노출을 명시하며, official Korean v1 결과를
완전한 zero-shot이라고 주장하지 않는다. 데이터 source의 license가 혼재하므로 이
모델 카드의 `other`는 upstream 권리를 재허가하지 않는다.

## SentenceTransformers 사용법

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "{repo_id}",
    model_kwargs={{"attn_implementation": "flash_attention_2"}},
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

vLLM pooling은 동시 요청 API에 적합하지만 offline 고정 대량 corpus에서는 항상
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
    merge, sionic, official, training = validate(args)
    model_dir = args.model_dir.resolve()
    card = build_card(args.repo_id, merge, sionic, official, training)
    card_path = model_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    report = {
        "repo_id": args.repo_id,
        "model_dir": str(model_dir),
        "card": str(card_path),
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
