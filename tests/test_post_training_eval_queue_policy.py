from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts/run_post_training_eval_queue.sh"


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


def test_queue_uses_safe_batches_and_token_free_offline_evaluation() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert "CAMPAIGN_EVAL_BATCH_SIZES:-8 4 2" in source
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


def test_queue_can_only_publish_the_clean_selected_private_candidate() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert source.count("publish_best_embedding_model.py") == 1
    assert "performance-v1-private-candidate" in source
    assert "--comprehensive-summary" in source
    assert "--upload --public" not in source
    assert "public_benchmark_used_for_selection" not in source
