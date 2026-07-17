from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.merge_nemotron3_adapter import QUERY_PROMPT, validate_embedding_contract


def write_contract(root: Path, *, pooling_mean: bool = True) -> None:
    (root / "1_Pooling").mkdir(parents=True)
    (root / "2_Normalize").mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ministral3",
                "architectures": ["Ministral3Model"],
                "hidden_size": 4096,
            }
        )
    )
    (root / "modules.json").write_text(
        json.dumps(
            [
                {"type": "sentence_transformers.models.Transformer"},
                {"type": "sentence_transformers.models.Pooling"},
                {"type": "sentence_transformers.models.Normalize"},
            ]
        )
    )
    (root / "1_Pooling/config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": 4096,
                "pooling_mode_cls_token": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_tokens": pooling_mean,
                "pooling_mode_mean_sqrt_len_tokens": False,
                "pooling_mode_weightedmean_tokens": False,
                "pooling_mode_lasttoken": False,
                "include_prompt": True,
            }
        )
    )
    (root / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "prompts": {"query": QUERY_PROMPT, "document": ""},
                "default_prompt_name": None,
                "similarity_fn_name": "cosine",
            }
        )
    )


def test_accepts_masked_mean_nemotron_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_contract(root)
        result = validate_embedding_contract(root)
        assert result["pooling"] == "masked_mean"
        assert result["normalize"] is True


def test_rejects_last_token_or_disabled_mean_pooling() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_contract(root, pooling_mean=False)
        with pytest.raises(ValueError, match="masked-mean"):
            validate_embedding_contract(root)
