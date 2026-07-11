---
pretty_name: Korean Embedding Benchmark Blocklist v1
language:
- ko
license: other
task_categories:
- text-retrieval
tags:
- decontamination
- mteb
- korean
---

# Korean Embedding Benchmark Blocklist v1

Sionic Korean retrieval 9종과 공식 `MTEB(kor, v1)` 6종의 평가 입력을 학습 데이터에서
제외하기 위한 **평가 전용 SHA-256 blocklist**다. 원문, 원본 query, qrel label, raw
source ID를 포함하지 않는다. 각 gzip 파일에는 canonicalized evaluation field의
SHA-256 digest만 한 줄에 하나씩 정렬해 저장한다.

## 범위

- protocol 1: `sionic9-fixed-prompt-v1`, 9 tasks
- protocol 2: `mteb-korean-v1-mteb-2.18.0`, 6 tasks
- complete task artifacts: 15/15
- field families: query/document/evaluation text, example/query/corpus IDs, relevance 또는
  candidate relation
- dataset revision, split, subset, field별 occurrence/unique count는 task manifest에 고정

MIRACL과 Ko-StrategyQA처럼 두 protocol에서 공유되는 dataset도 protocol별 manifest를
별도로 보존한다. 이 dataset 자체를 학습, synthetic query 생성, hard-negative mining,
distillation, checkpoint selection에 사용하면 안 된다.

## 사용

학습 후보 text를 builder와 같은 canonicalization 규칙으로 normalize한 뒤 SHA-256을
계산하고, 대응하는 `*.sha256.gz`의 digest set과 exact 비교한다. ID/relation hash는
평가 source lineage 차단과 audit에 사용한다. near-duplicate 검사는 이 release의 exact
hash와 별도로 수행해야 한다.

```bash
PYTHONPATH=third_party/mteb .venv-mteb/bin/python \
  scripts/build_benchmark_blocklist.py \
  --output-dir outputs/decontamination/benchmark_blocklist \
  --minhash off
```

재현 코드와 고정 protocol은
[LLM-OS-Models/Embedding](https://github.com/LLM-OS-Models/Embedding)에 있다.

## 제한과 권리

digest는 원문을 대신하는 학습 데이터가 아니며 평가 dataset의 license를 재허가하지
않는다. 각 task manifest에 upstream dataset/license metadata를 기록한다. hash exact
match가 없다는 사실은 의미상 오염이나 번역·요약·부분문자열 overlap 부재를 보장하지
않는다.
