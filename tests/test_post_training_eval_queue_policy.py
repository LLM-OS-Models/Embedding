from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts/run_post_training_eval_queue.sh"
SCALE_QUEUE = ROOT / "scripts/run_scale_1m_queue.sh"
FRONTIER_QUEUE = ROOT / "scripts/run_frontier_200k_pair_queue.sh"
SOUP_QUEUE = ROOT / "scripts/run_model_soup_queue.sh"
CAPACITY_QUEUE = ROOT / "scripts/run_capacity_ablation_queue.sh"
CAPACITY_TRAIN = ROOT / "experiments/070_tuning_strategy/train_quality.sh"
CAPACITY_PROBE = ROOT / "experiments/070_tuning_strategy/probe_memory.sh"


def test_post_training_queue_selects_clean_before_public_benchmarks() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    selector_call = source.index('"$ROOT/scripts/select_best_clean_model.py"')
    final_sionic_call = source.index(
        'run_sionic_with_fallback "final-selected"', selector_call
    )
    final_official_call = source.index(
        'run_official_with_fallback "v1-final-selected"', selector_call
    )
    comprehensive_call = source.index("run_comprehensive_with_fallback", selector_call)
    assert selector_call < final_sionic_call < final_official_call < comprehensive_call
    assert "select_best_sionic_model.py" not in source
    assert "--candidate-model" in source
    assert source.count('run_sionic_with_fallback "') == 1
    assert source.count('run_official_with_fallback "') == 1
    assert source.count("run_comprehensive_with_fallback \\") == 1
    selection_only_gate = source.index('if [[ "$SELECTION_ONLY" == 1 ]]', selector_call)
    assert selector_call < selection_only_gate < final_sionic_call
    assert "public evaluation and publication skipped" in source


def test_queue_uses_safe_batches_and_token_free_offline_evaluation() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert "CAMPAIGN_EVAL_BATCH_SIZES:-192 128 64 32 16 8 4 2" in source
    assert "CAMPAIGN_EVAL_BATCH_SIZE:-192" not in source
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in source
    assert 'PUBLISH_HF_TOKEN_FILE="$ROOT/.env"' in source
    assert "PUBLISH_HF_TOKEN=" not in source
    assert 'env HF_TOKEN="$PUBLISH_HF_TOKEN"' not in source
    assert '--hf-token-file "$PUBLISH_HF_TOKEN_FILE"' in source
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1" in source
    for evaluator in (
        "evaluate_sionic9.py",
        "evaluate_mteb_korean_v1.py",
        "evaluate_legal_source_holdout.py",
        "evaluate_conversational_noise_robustness.py",
        "evaluate_comprehensive_text_v1.py",
    ):
        index = source.index(evaluator)
        invocation = source[max(0, index - 180) : index + 250]
        assert '"${OFFLINE_ENV[@]}"' in invocation


def test_queue_compares_qwen_and_comsat_under_the_same_200k_contract() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert "qwen3-embedding-8b-ko-performance200k-lora-r64" in source
    assert "comsat-embed-ko-8b-performance200k-lora-r64" in source
    assert "--base-model sionic-ai/comsat-embed-ko-8b-preview" in source
    assert "--base-revision a5cc22b651c1b2e51cdd8bf671774ae93584f0ab" in source
    assert source.count('"${merge_base_args[@]}"') == 2
    assert "clean-first-selection.json" in source
    assert "average_lora_checkpoints.py" in source
    assert "--last-n 5 --minimum-checkpoints 2" in source
    assert "last-available5-fp32-average-merged" in source
    for later_run in (
        "performance1m-lora-r64",
        "reranker-listwise-kl07-queue4096-lora-r64",
        "sionic-retrieval-family50-replay50-lora-r64",
        "sionic-squad50-replay50-lora-r64",
        "sionic-health50-replay50-lora-r64",
        "sionic-autorag50-replay50-lora-r64",
        "legal25-replay75-lora-r64",
        "sionic-combined-target-lora-r64",
    ):
        assert later_run in source
    assert "resolve_training_manifest" in source
    assert "train.reranker-quantile-kd15.manifest.json" in source
    assert "SOUP_MODELS" in source
    assert "soup_report.json" in source
    assert "qwen3-embedding-8b-ko-performance200k-last4" in source
    assert "comsat-embed-ko-8b-performance200k-last4" in source
    assert "capacity_run_manifest.json" in source


def test_frontier_queue_chains_selection_scale_and_target_adaptation() -> None:
    frontier = FRONTIER_QUEUE.read_text(encoding="utf-8")
    post_eval = frontier.index("run_post_training_eval_queue.sh")
    selection_gate = frontier.index('[[ ! -s "$POST_EVAL_SELECTION" ]]', post_eval)
    capacity = frontier.index("run_capacity_ablation_queue.sh", selection_gate)
    capacity_eval = frontier.index("run_post_training_eval_queue.sh", capacity)
    capacity_gate = frontier.index(
        '[[ ! -s "$CAPACITY_EVAL_SELECTION" ]]', capacity_eval
    )
    scale = frontier.index("run_scale_1m_queue.sh", capacity_gate)
    legal = frontier.index("run_legal_adaptation_queue.sh", scale)
    soup = frontier.index("run_model_soup_queue.sh", legal)
    final_eval = frontier.index("run_post_training_eval_queue.sh", soup)
    final_gate = frontier.index('[[ ! -s "$FINAL_EVAL_SELECTION" ]]', final_eval)
    assert (
        post_eval
        < selection_gate
        < capacity
        < capacity_eval
        < capacity_gate
        < scale
        < legal
        < soup
        < final_eval
        < final_gate
    )
    assert 'SELECTION_ONLY=1 LOG_DIR="$POST_EVAL_LOG"' in frontier
    assert 'POSTTRAIN_SELECTION="$CAPACITY_EVAL_SELECTION"' in frontier
    assert frontier.count("embedding_require_storage_headroom") >= 8

    scale_source = SCALE_QUEUE.read_text(encoding="utf-8")
    assert "POSTTRAIN_SELECTION:-$ROOT/outputs/post-capacity-eval-20260717-frontier/clean-first-selection.json" in scale_source
    assert "SAVE_TOTAL_LIMIT=5" in frontier
    assert "SAVE_TOTAL_LIMIT=5" in scale_source
    assert '"$ROOT/.venv-train-fa2/bin/python"' in frontier
    assert '"$ROOT/.venv-hf-tools/bin/python"' not in frontier


def test_frontier_queues_keep_hf_token_out_of_training_and_evaluation() -> None:
    queues = (
        SCALE_QUEUE,
        ROOT / "scripts/run_legal_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_squad_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_combined_adaptation_queue.sh",
    )
    for queue in queues:
        source = queue.read_text(encoding="utf-8")
        assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in source, queue
        assert "HF_TOKEN=\"$(sed" not in source, queue
        assert 'PUBLISH_HF_TOKEN_FILE="$ROOT/.env"' in source, queue
        assert '--hf-token-file "$PUBLISH_HF_TOKEN_FILE"' in source, queue
        # Dataset publishers still require an environment token, but source the
        # ignored credential file only after the training checkpoint is chosen.
        source_token = source.index('source "$PUBLISH_HF_TOKEN_FILE"')
        selected_checkpoint = source.index("select_best_checkpoint.py")
        assert selected_checkpoint < source_token, queue


def test_model_soup_coefficients_are_fixed_before_clean_evaluation() -> None:
    source = SOUP_QUEUE.read_text(encoding="utf-8")
    for label in (
        "soup-general50-combined50",
        "soup-general50-specialists10x5",
        "soup-general25-combined25-specialists10x5",
        "soup-combined50-specialists10x5",
    ):
        assert label in source
    assert "merge_full_model_soup.py" in source
    assert "evaluate_sionic9.py" not in source
    assert "evaluate_mteb_korean_v1.py" not in source
    assert "HF_TOKEN" in source and "unset HF_TOKEN" in source


def test_capacity_queue_runs_one_selected_lineage_last4_challenger() -> None:
    source = CAPACITY_QUEUE.read_text(encoding="utf-8")
    assert "LINEAGE_SELECTION" in source
    assert "merge_report.json" in source
    assert "Qwen/Qwen3-Embedding-8B@1d8ad4ca" in source
    assert "sionic-ai/comsat-embed-ko-8b-preview@a5cc22b" in source
    assert "probe_memory.sh\" last4" in source
    assert "train_quality.sh\" last4" in source
    assert "TRAIN_BATCH_SIZE=8 GRAD_ACCUM_STEPS=8" in source
    assert "MAX_STEPS=3123 SAVE_TOTAL_LIMIT=5" in source
    assert "INFONCE_HARD_NEGATIVES=4" in source
    assert "EMBEDDING_OFFLINE=1" in source
    assert "unset HF_TOKEN HUGGINGFACE_HUB_TOKEN" in source
    assert "3123/3123" in source
    assert source.count("train_quality.sh") == 1

    train = CAPACITY_TRAIN.read_text(encoding="utf-8")
    probe = CAPACITY_PROBE.read_text(encoding="utf-8")
    for script in (train, probe):
        assert 'BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-8B}"' in script
        assert "BASE_REVISION" in script
    assert "capacity_run_manifest.json" in train
    assert "save_total_limit \"$SAVE_TOTAL_LIMIT\"" in train
    assert "dataset_shuffle false" in probe
    assert "train_dataloader_shuffle false" in probe


def test_campaign_queues_resolve_an_available_training_runtime() -> None:
    queues = (
        QUEUE,
        SCALE_QUEUE,
        ROOT / "scripts/run_legal_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_squad_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_combined_adaptation_queue.sh",
        ROOT / "scripts/run_night_gpu_queue.sh",
        CAPACITY_QUEUE,
    )
    for queue in queues:
        source = queue.read_text(encoding="utf-8")
        assert "embedding_resolve_train_runtime" in source, queue
        assert ".venv-train/bin/python" not in source, queue


def test_queue_can_only_publish_the_clean_selected_private_candidate() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert source.count("publish_best_embedding_model.py") == 1
    assert "performance-v1-private-candidate" in source
    assert "--comprehensive-summary" in source
    assert "--upload --public" not in source
    assert "public_benchmark_used_for_selection" not in source
