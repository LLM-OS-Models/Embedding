from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_training_wrapper_uses_external_plugin_only_for_explicit_kd_runs() -> None:
    trainer = source("experiments/020_hard_negative/train_pilot_lora_r64.sh")
    assert '"${ENABLE_LISTWISE_KD:-0}" == 1' in trainer
    assert "listwise_kd_plugin.py" in trainer
    assert "--remove_unused_columns false" in trainer
    assert "--loss_type listwise_embedding_kd" in trainer
    assert "--require-teacher-scores" in trainer
    assert "LOSS_ARGS=(--loss_type infonce)" in trainer


def test_kd_queue_runs_filter_kl_and_queue_ablation_before_target_adaptation() -> None:
    kd = source("scripts/run_reranker_kd_ablation_queue.sh")
    for variant in ("filter-only", "listwise-kl07", "listwise-kl07-queue4096"):
        assert variant in kd
    assert "Qwen3-Reranker-8B" not in kd  # scorer owns the pinned identity
    assert "cache_qwen3_reranker_scores.py" in kd
    assert "KD_RERANKER_BATCH_SIZE:-8" in kd
    assert "compile_reranker_kd_dataset.py" in kd
    assert "select_best_clean_model.py" in kd
    assert "publish_private_clean_candidate.py" in kd
    assert "qwen3-embedding-8b-ko-reranker-kd-clean-winner-v1-private" in kd
    assert '--hf-token-file "$ROOT/.env"' in kd
    assert "GENERAL_TRAINING_MANIFEST" in kd
    assert "evaluate_sionic9.py" not in kd
    assert "evaluate_mteb_korean_v1.py" not in kd

    scale = source("scripts/run_scale_1m_queue.sh")
    kd_index = scale.index("run_reranker_kd_ablation_queue.sh")
    target_index = scale.index("sionic-retrieval-train-family-adaptation", kd_index)
    assert kd_index < target_index
    assert 'GENERAL_BASE_MODEL="$MODEL_DIR"' in scale
    assert 'GENERAL_TRAINING_MANIFEST="$TRAINING_MANIFEST"' in scale


def test_every_target_queue_resolves_the_clean_selected_general_base() -> None:
    common = source("scripts/common_runtime.sh")
    assert "embedding_resolve_general_base" in common
    assert "reranker-kd-20260717-frontier/clean-first-selection.json" in common
    for relative in (
        "scripts/run_sionic_squad_adaptation_queue.sh",
        "scripts/run_legal_adaptation_queue.sh",
        "scripts/run_sionic_combined_adaptation_queue.sh",
    ):
        assert "embedding_resolve_general_base" in source(relative), relative


def test_faiss_miner_can_emit_wide_teacher_requests_with_manifest_evidence() -> None:
    miner = source("scripts/mine_faiss_hard_negatives.py")
    assert "--teacher-request-output" in miner
    assert "--teacher-request-limit" in miner
    assert "--teacher-candidate-count" in miner
    assert "seeded_without_replacement_over_input_rows" in miner
    assert '"documents_per_row": args.teacher_candidate_count + 1' in miner
