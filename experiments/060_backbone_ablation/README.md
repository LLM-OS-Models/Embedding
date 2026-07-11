# 060 — Backbone ablation

후보:

- Qwen3-Embedding 0.6B / 4B / 8B
- NVIDIA llama-embed-nemotron-8b
- Tencent KaLM Gemma3 12B
- Microsoft Harrier 0.6B; 27B는 inference/LoRA feasibility만 우선 측정

동일 데이터 budget과 동일 evaluator로 backbone 효과와 data recipe 효과를 분리합니다.
