#!/usr/bin/env python3
"""Check an OpenAI-compatible Qwen embedding endpoint and retrieval contract."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3-embedding-8b")
    parser.add_argument(
        "--instruction",
        default="Given a Korean web search query, retrieve relevant passages that answer the query",
    )
    return parser.parse_args()


def embed(base_url: str, model: str, inputs: list[str]) -> list[list[float]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=json.dumps(
            {"model": model, "input": inputs, "encoding_format": "float"}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return [row["embedding"] for row in sorted(payload["data"], key=lambda x: x["index"])]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: list[float]) -> float:
    return math.sqrt(dot(vector, vector))


def main() -> None:
    args = parse_args()
    query = f"Instruct: {args.instruction}\nQuery: 대한민국의 수도는 어디인가?"
    documents = [
        "대한민국의 수도는 서울특별시이다.",
        "목성은 태양계에서 가장 큰 행성이다.",
    ]
    vectors = embed(args.base_url, args.model, [query, *documents])
    if not vectors or any(len(vector) != len(vectors[0]) for vector in vectors):
        raise RuntimeError("Endpoint returned missing or inconsistent vectors")
    norms = [norm(vector) for vector in vectors]
    scores = [dot(vectors[0], vector) for vector in vectors[1:]]
    if any(abs(value - 1.0) > 5e-3 for value in norms):
        raise RuntimeError(f"Expected L2-normalized embeddings, got norms={norms}")
    if scores[0] <= scores[1]:
        raise RuntimeError(f"Retrieval order is wrong: scores={scores}")
    print(
        json.dumps(
            {
                "model": args.model,
                "dimension": len(vectors[0]),
                "norms": norms,
                "scores": scores,
                "retrieval_order_ok": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
