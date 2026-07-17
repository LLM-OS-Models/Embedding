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
COMMON_RUNTIME = ROOT / "scripts/common_runtime.sh"
PILOT_TRAIN = ROOT / "experiments/020_hard_negative/train_pilot_lora_r64.sh"


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
    assert "public evaluation and public-score publication skipped" in source
    private_backup = source.index("publish_private_clean_candidate.py", selection_only_gate)
    selection_only_exit = source.index("exit 0", private_backup)
    assert selection_only_gate < private_backup < selection_only_exit < final_sionic_call


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
    assert "list_validated_adapter_checkpoints.py" in source
    assert "contaminated_validation=1" in source
    assert '"artifacts/models/${run_name}-${checkpoint_label}-clean-candidate-merged"' in source


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
    assert 'SELECTION_ONLY=1 LOG_DIR="$CAPACITY_EVAL_LOG"' in frontier
    assert "SELECTION_ONLY=0" not in frontier
    assert 'POSTTRAIN_SELECTION="$CAPACITY_EVAL_SELECTION"' in frontier
    assert frontier.count("ENABLE_PUBLIC_INTERMEDIATE_EVAL=0") == 2
    assert frontier.count("SELECTION_PRIVATE_REPO_ID=LLM-OS-Models2/") == 2
    assert frontier.count("embedding_require_storage_headroom") >= 8

    scale_source = SCALE_QUEUE.read_text(encoding="utf-8")
    assert "POSTTRAIN_SELECTION:-$ROOT/outputs/post-capacity-eval-20260717-frontier/clean-first-selection.json" in scale_source
    assert "SAVE_TOTAL_LIMIT=5" in frontier
    assert "SAVE_TOTAL_LIMIT=5" in scale_source
    assert '"$ROOT/.venv-train-fa2/bin/python"' in frontier
    assert '"$ROOT/.venv-hf-tools/bin/python"' not in frontier


def test_frontier_waits_for_qwen_wrapper_exit_before_comsat_probe() -> None:
    frontier = FRONTIER_QUEUE.read_text(encoding="utf-8")
    step_gate = frontier.index('rg -q \'"3123/3123"\'')
    wrapper_wait = frontier.index("while qwen_wrapper_alive", step_gate)
    comsat_probe = frontier.index("starting Comsat exact probe", wrapper_wait)
    assert step_gate < wrapper_wait < comsat_probe
    assert 'QWEN_TRAIN_PID="${QWEN_TRAIN_PID:-}"' in frontier
    assert '"/proc/$QWEN_TRAIN_PID/cmdline"' in frontier
    assert "train_pilot_lora_r64.sh*" in frontier


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
        assert "EMBEDDING_OFFLINE=1" in source, queue
        assert "HF_TOKEN=\"$(sed" not in source, queue
        assert 'PUBLISH_HF_TOKEN_FILE="$ROOT/.env"' in source, queue
        assert '--hf-token-file "$PUBLISH_HF_TOKEN_FILE"' in source, queue
        # Dataset publishers still require an environment token, but source the
        # ignored credential file only after the training checkpoint is chosen.
        source_token = source.index('source "$PUBLISH_HF_TOKEN_FILE"')
        selected_checkpoint = source.index("select_best_checkpoint.py")
        assert selected_checkpoint < source_token, queue


def test_intermediate_queues_disable_public_evaluation_by_default() -> None:
    queues = (
        SCALE_QUEUE,
        ROOT / "scripts/run_legal_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_squad_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_combined_adaptation_queue.sh",
    )
    for queue in queues:
        source = queue.read_text(encoding="utf-8")
        assert (
            'ENABLE_PUBLIC_INTERMEDIATE_EVAL="${ENABLE_PUBLIC_INTERMEDIATE_EVAL:-0}"'
            in source
        ), queue
        gate = source.index('if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1 ]]')
        public_sionic = source.index("run_sionic", gate)
        public_official = source.index("run_official", public_sionic)
        gate_end = source.index("\nfi", public_official)
        assert gate < public_sionic < public_official < gate_end, queue
        publish_gate = source.index(
            'if [[ "$ENABLE_PUBLIC_INTERMEDIATE_EVAL" == 1', gate_end
        )
        publisher = source.index("publish_best_embedding_model.py", publish_gate)
        assert publish_gate < publisher, queue


def test_future_lora_runs_upload_reconstructable_private_checkpoints() -> None:
    queues = (
        SCALE_QUEUE,
        ROOT / "scripts/run_reranker_kd_ablation_queue.sh",
        ROOT / "scripts/run_legal_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_squad_adaptation_queue.sh",
        ROOT / "scripts/run_sionic_combined_adaptation_queue.sh",
    )
    for queue in queues:
        source = queue.read_text(encoding="utf-8")
        assert "ENABLE_PRIVATE_CHECKPOINT_WATCHER=1" in source, queue
        assert "CHECKPOINT_TRAINING_MANIFEST=" in source, queue
        assert "CHECKPOINT_BASE_UPLOAD_REPORT=" in source, queue
        assert "PRIVATE_CHECKPOINT_REPO_ID=" in source, queue
        assert "EMBEDDING_OFFLINE=1" in source, queue

    train = PILOT_TRAIN.read_text(encoding="utf-8")
    watcher = train.index("watch_private_adapter_checkpoints.py")
    swift = train.index('"$TRAIN_ENV/bin/swift" sft')
    reconciliation = train.index("--once --settle-seconds 0", swift)
    assert watcher < swift < reconciliation
    assert "local continual base requires a verified private upload report" in train
    assert '"$report_weights_sha" != "$expected_base_sha"' in train
    assert '"$watcher_base_revision" =~ ^[0-9a-f]{40}$' in train
    assert train.count("-u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u HF_DATASETS_OFFLINE") == 2


def test_model_soup_coefficients_are_fixed_before_clean_evaluation() -> None:
    source = SOUP_QUEUE.read_text(encoding="utf-8")
    for label in (
        "soup-general75-parent25",
        "soup-general50-parent50",
        "soup-general50-combined50",
        "soup-general25-combined75",
        "soup-general50-specialists10x5",
        "soup-general25-combined25-specialists10x5",
        "soup-combined50-specialists10x5",
    ):
        assert label in source
    assert "merge_full_model_soup.py" in source
    assert "resolve_local_parent_model" in source
    assert '"$ROOT"/artifacts/models/*' in source
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


def test_all_training_entrypoints_fail_closed_on_text_strict_validation() -> None:
    runtime = COMMON_RUNTIME.read_text(encoding="utf-8")
    assert "embedding_require_clean_validation()" in runtime
    assert "legal-source-heldout-i-v2-text-strict-training-validation" in runtime
    assert 'actual_sha="$(sha256sum "$validation"' in runtime
    for path in (CAPACITY_TRAIN, ROOT / "experiments/020_hard_negative/train_pilot_lora_r64.sh", ROOT / "experiments/080_f2_recipe/train_pilot_f2_dual_lora_r64.sh"):
        source = path.read_text(encoding="utf-8")
        assert 'embedding_require_clean_validation "$VAL_FILE"' in source, path
    pilot = (ROOT / "experiments/020_hard_negative/train_pilot_lora_r64.sh").read_text(encoding="utf-8")
    assert "legacy eval-loss continual promotion is disabled" in pilot


def test_queue_can_only_publish_the_clean_selected_private_candidate() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert source.count("publish_best_embedding_model.py") == 1
    assert "performance-v1-private-candidate" in source
    assert "--comprehensive-summary" in source
    assert "--upload --public" not in source
    assert "public_benchmark_used_for_selection" not in source
