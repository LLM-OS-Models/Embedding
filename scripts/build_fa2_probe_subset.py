#!/usr/bin/env python3
"""Build a small, auditable workload subset for the 8B FA2 admission probe.

The input is expected to be the source-homogeneous, length-bucketed JSONL plus
its aligned provenance sidecar produced by ``build_homogeneous_batches.py``.
Selection happens at complete microbatch granularity.  It preserves an exact
Hamilton apportionment of the input source distribution and assigns one global
length-quantile stratum to every gradient-accumulation slot in every optimizer
step.

Only the selected rows are later tokenized by ms-swift.  Training JSONL bytes
are copied unchanged; the projected provenance records the original batch and
the probe execution plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable


SELECTION_CONTRACT = "fa2-probe-source-length-stratified-v1"


@dataclass(frozen=True)
class Batch:
    original_index: int
    source_id: str
    row_count: int
    train_start: int
    train_end: int
    provenance_start: int
    provenance_end: int
    length_proxy_min: int
    length_proxy_max: int


@dataclass(frozen=True)
class ScanResult:
    batches: tuple[Batch, ...]
    rows: int
    train_sha256: str
    provenance_sha256: str


@dataclass(frozen=True)
class PlannedBatch:
    batch: Batch
    length_stratum: int
    optimizer_step: int
    accumulation_slot: int
    subset_batch_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a representative complete-microbatch FA2 probe subset"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--training-max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_digest(seed: int, namespace: str, value: object) -> bytes:
    payload = f"{seed}\0{namespace}\0{value}".encode()
    return hashlib.sha256(payload).digest()


def _as_int(value: Any, name: str, row_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Provenance row {row_number}: {name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Provenance row {row_number}: {name} must be an integer"
        ) from error
    if result < 0:
        raise ValueError(f"Provenance row {row_number}: {name} must be non-negative")
    return result


def _finish_batch(
    batches: list[Batch],
    *,
    index: int,
    source_id: str,
    row_count: int,
    batch_size: int,
    train_start: int,
    train_end: int,
    provenance_start: int,
    provenance_end: int,
    observed_lengths: list[int],
    declared_min: int,
    declared_max: int,
) -> None:
    if row_count != batch_size:
        raise ValueError(
            f"Original batch {index} has {row_count} rows, expected {batch_size}"
        )
    actual_min = min(observed_lengths)
    actual_max = max(observed_lengths)
    if (actual_min, actual_max) != (declared_min, declared_max):
        raise ValueError(
            f"Original batch {index} length metadata drift: "
            f"declared=({declared_min}, {declared_max}) "
            f"observed=({actual_min}, {actual_max})"
        )
    batches.append(
        Batch(
            original_index=index,
            source_id=source_id,
            row_count=row_count,
            train_start=train_start,
            train_end=train_end,
            provenance_start=provenance_start,
            provenance_end=provenance_end,
            length_proxy_min=actual_min,
            length_proxy_max=actual_max,
        )
    )


def scan_aligned(train: Path, provenance: Path, batch_size: int) -> ScanResult:
    """Validate alignment and record byte ranges without parsing the 800MB train file."""

    batches: list[Batch] = []
    train_digest = hashlib.sha256()
    provenance_digest = hashlib.sha256()
    rows = 0

    current_index: int | None = None
    current_source = ""
    current_rows = 0
    current_train_start = 0
    current_provenance_start = 0
    current_lengths: list[int] = []
    current_declared_min = 0
    current_declared_max = 0

    with train.open("rb") as train_handle, provenance.open("rb") as provenance_handle:
        while True:
            train_start = train_handle.tell()
            provenance_start = provenance_handle.tell()
            train_line = train_handle.readline()
            provenance_line = provenance_handle.readline()
            if not train_line and not provenance_line:
                break
            if not train_line or not provenance_line:
                raise ValueError(f"Train/provenance length mismatch after row {rows}")
            rows += 1
            train_digest.update(train_line)
            provenance_digest.update(provenance_line)
            try:
                record = json.loads(provenance_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid provenance JSON at row {rows}") from error
            metadata = record.get("homogeneous_batch")
            if not isinstance(metadata, dict):
                raise ValueError(f"Provenance row {rows} has no homogeneous_batch object")
            index = _as_int(metadata.get("batch_index"), "batch_index", rows)
            declared_size = _as_int(metadata.get("batch_size"), "batch_size", rows)
            output_row_index = _as_int(
                metadata.get("output_row_index"), "output_row_index", rows
            )
            length_proxy = _as_int(metadata.get("length_proxy"), "length_proxy", rows)
            declared_min = _as_int(
                metadata.get("batch_length_proxy_min"),
                "batch_length_proxy_min",
                rows,
            )
            declared_max = _as_int(
                metadata.get("batch_length_proxy_max"),
                "batch_length_proxy_max",
                rows,
            )
            source_id = metadata.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"Provenance row {rows}: source_id must be non-empty")
            if declared_size != batch_size:
                raise ValueError(
                    f"Provenance row {rows}: batch_size={declared_size}, expected {batch_size}"
                )
            if output_row_index != rows - 1:
                raise ValueError(
                    f"Provenance row {rows}: output_row_index={output_row_index}, "
                    f"expected {rows - 1}"
                )

            if current_index is None:
                if index != 0:
                    raise ValueError(f"First homogeneous batch index is {index}, expected 0")
                current_index = index
                current_source = source_id
                current_train_start = train_start
                current_provenance_start = provenance_start
                current_declared_min = declared_min
                current_declared_max = declared_max
            elif index != current_index:
                if index != current_index + 1:
                    raise ValueError(
                        f"Homogeneous batch index jumped from {current_index} to {index}"
                    )
                _finish_batch(
                    batches,
                    index=current_index,
                    source_id=current_source,
                    row_count=current_rows,
                    batch_size=batch_size,
                    train_start=current_train_start,
                    train_end=train_start,
                    provenance_start=current_provenance_start,
                    provenance_end=provenance_start,
                    observed_lengths=current_lengths,
                    declared_min=current_declared_min,
                    declared_max=current_declared_max,
                )
                current_index = index
                current_source = source_id
                current_rows = 0
                current_train_start = train_start
                current_provenance_start = provenance_start
                current_lengths = []
                current_declared_min = declared_min
                current_declared_max = declared_max

            if source_id != current_source:
                raise ValueError(
                    f"Original batch {current_index} mixes {current_source!r} and {source_id!r}"
                )
            if (declared_min, declared_max) != (
                current_declared_min,
                current_declared_max,
            ):
                raise ValueError(
                    f"Original batch {current_index} repeats inconsistent length metadata"
                )
            current_rows += 1
            current_lengths.append(length_proxy)

        if current_index is not None:
            _finish_batch(
                batches,
                index=current_index,
                source_id=current_source,
                row_count=current_rows,
                batch_size=batch_size,
                train_start=current_train_start,
                train_end=train_handle.tell(),
                provenance_start=current_provenance_start,
                provenance_end=provenance_handle.tell(),
                observed_lengths=current_lengths,
                declared_min=current_declared_min,
                declared_max=current_declared_max,
            )

    if not batches:
        raise ValueError("No complete homogeneous batches found")
    return ScanResult(
        batches=tuple(batches),
        rows=rows,
        train_sha256=train_digest.hexdigest(),
        provenance_sha256=provenance_digest.hexdigest(),
    )


def apportion(
    counts: dict[Any, int], slots: int, *, seed: int, namespace: str
) -> dict[Any, int]:
    """Hamilton/largest-remainder apportionment with a stable seeded tie-break."""

    total = sum(counts.values())
    if total < 1 or slots < 1:
        raise ValueError("Apportionment requires positive counts and slots")
    if slots > total:
        raise ValueError(f"Cannot select {slots} unique items from {total}")
    quotas = {key: count * slots // total for key, count in counts.items()}
    remaining = slots - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            -(counts[key] * slots % total),
            stable_digest(seed, namespace, key),
        ),
    )
    for key in ranked[:remaining]:
        quotas[key] += 1
    return quotas


def assign_length_strata(
    batches: Iterable[Batch], strata: int, seed: int
) -> dict[int, int]:
    ranked = sorted(
        batches,
        key=lambda batch: (
            batch.length_proxy_max,
            batch.length_proxy_min,
            stable_digest(seed, "length-tie", batch.original_index),
        ),
    )
    total = len(ranked)
    return {
        batch.original_index: rank * strata // total
        for rank, batch in enumerate(ranked)
    }


class _FlowNetwork:
    def __init__(self) -> None:
        self.capacity: dict[tuple[str, str], int] = {}
        self.adjacency: dict[str, list[str]] = defaultdict(list)

    def add_edge(self, source: str, target: str, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Flow capacity cannot be negative")
        if (source, target) in self.capacity:
            raise ValueError(f"Duplicate flow edge: {source} -> {target}")
        self.capacity[source, target] = capacity
        self.capacity[target, source] = 0
        self.adjacency[source].append(target)
        self.adjacency[target].append(source)

    def max_flow(self, source: str, sink: str) -> int:
        result = 0
        while True:
            parent: dict[str, str | None] = {source: None}
            queue = deque([source])
            while queue and sink not in parent:
                node = queue.popleft()
                for neighbour in self.adjacency[node]:
                    if neighbour in parent or self.capacity[node, neighbour] <= 0:
                        continue
                    parent[neighbour] = node
                    queue.append(neighbour)
            if sink not in parent:
                return result
            increment = 1 << 60
            node = sink
            while parent[node] is not None:
                previous = parent[node]
                increment = min(increment, self.capacity[previous, node])
                node = previous
            node = sink
            while parent[node] is not None:
                previous = parent[node]
                self.capacity[previous, node] -= increment
                self.capacity[node, previous] += increment
                node = previous
            result += increment


def joint_allocation(
    batches: Iterable[Batch],
    source_quotas: dict[str, int],
    stratum_quotas: dict[int, int],
    stratum_by_batch: dict[int, int],
) -> dict[tuple[str, int], int]:
    availability: Counter[tuple[str, int]] = Counter(
        (batch.source_id, stratum_by_batch[batch.original_index]) for batch in batches
    )
    network = _FlowNetwork()
    network.add_edge("START", "SOURCE-GUARD", sum(source_quotas.values()))
    # The guard makes source node names unambiguous even if a source is named START.
    for source_id in sorted(source_quotas):
        network.add_edge(
            "SOURCE-GUARD", f"source:{source_id}", source_quotas[source_id]
        )
        for stratum in sorted(stratum_quotas):
            network.add_edge(
                f"source:{source_id}",
                f"stratum:{stratum}",
                availability[source_id, stratum],
            )
    for stratum in sorted(stratum_quotas):
        network.add_edge(
            f"stratum:{stratum}", "END", stratum_quotas[stratum]
        )
    required = sum(source_quotas.values())
    actual = network.max_flow("START", "END")
    if actual != required:
        raise ValueError(
            "Could not satisfy source and length-stratum quotas jointly: "
            f"required={required}, feasible={actual}"
        )
    allocation = {}
    for source_id in sorted(source_quotas):
        for stratum in sorted(stratum_quotas):
            # Reverse residual capacity is the selected flow.
            allocation[source_id, stratum] = network.capacity[
                f"stratum:{stratum}", f"source:{source_id}"
            ]
    return allocation


def evenly_spaced(items: list[Batch], count: int) -> list[Batch]:
    if count == 0:
        return []
    if count > len(items):
        raise ValueError(f"Cannot take {count} items from a cell containing {len(items)}")
    # Midpoints of equal-width partitions.  For count <= len(items), positions
    # are unique and do not over-emphasize either end of a quantile cell.
    positions = [(2 * index + 1) * len(items) // (2 * count) for index in range(count)]
    if len(set(positions)) != count:
        raise AssertionError("Internal evenly-spaced selection collision")
    return [items[position] for position in positions]


def plan_subset(
    batches: tuple[Batch, ...],
    *,
    probe_steps: int,
    gradient_accumulation_steps: int,
    seed: int,
) -> tuple[list[PlannedBatch], dict[str, Any]]:
    selected_batches = probe_steps * gradient_accumulation_steps
    if selected_batches > len(batches):
        raise ValueError(
            f"Probe needs {selected_batches} microbatches but input has {len(batches)}"
        )
    source_counts = Counter(batch.source_id for batch in batches)
    source_quotas = apportion(
        dict(source_counts), selected_batches, seed=seed, namespace="source-quota"
    )
    strata = gradient_accumulation_steps
    stratum_by_batch = assign_length_strata(batches, strata, seed)
    stratum_counts = Counter(stratum_by_batch.values())
    stratum_quotas = apportion(
        dict(stratum_counts), selected_batches, seed=seed, namespace="stratum-quota"
    )
    expected_per_stratum = probe_steps
    if set(stratum_quotas.values()) != {expected_per_stratum}:
        raise AssertionError(
            "Equal-frequency length strata did not allocate one batch per optimizer step"
        )
    allocation = joint_allocation(
        batches, source_quotas, stratum_quotas, stratum_by_batch
    )

    cells: dict[tuple[str, int], list[Batch]] = defaultdict(list)
    for batch in batches:
        cells[batch.source_id, stratum_by_batch[batch.original_index]].append(batch)
    selected_by_stratum: dict[int, list[Batch]] = defaultdict(list)
    for cell, candidates in sorted(cells.items()):
        source_id, stratum = cell
        candidates.sort(
            key=lambda batch: (
                batch.length_proxy_max,
                batch.length_proxy_min,
                stable_digest(seed, f"cell:{source_id}:{stratum}", batch.original_index),
            )
        )
        selected_by_stratum[stratum].extend(
            evenly_spaced(candidates, allocation.get(cell, 0))
        )
    for stratum, selected in selected_by_stratum.items():
        selected.sort(
            key=lambda batch: stable_digest(
                seed, f"execution-stratum:{stratum}", batch.original_index
            )
        )

    plan: list[PlannedBatch] = []
    cursors = Counter()
    for step in range(probe_steps):
        # Rotate the quartile-to-slot mapping so a systematic warm-up effect is
        # not always assigned to the same length stratum.
        for slot in range(gradient_accumulation_steps):
            stratum = (slot + step) % gradient_accumulation_steps
            batch = selected_by_stratum[stratum][cursors[stratum]]
            cursors[stratum] += 1
            plan.append(
                PlannedBatch(
                    batch=batch,
                    length_stratum=stratum,
                    optimizer_step=step + 1,
                    accumulation_slot=slot + 1,
                    subset_batch_index=len(plan),
                )
            )
    if len({item.batch.original_index for item in plan}) != selected_batches:
        raise AssertionError("Subset plan contains a duplicate original batch")
    selected_source_counts = Counter(item.batch.source_id for item in plan)
    selected_stratum_counts = Counter(item.length_stratum for item in plan)
    if dict(selected_source_counts) != {
        key: value for key, value in source_quotas.items() if value
    }:
        raise AssertionError("Selected source counts do not match apportioned quotas")
    if dict(selected_stratum_counts) != dict(stratum_quotas):
        raise AssertionError("Selected length counts do not match apportioned quotas")
    details = {
        "source_input_batches": dict(sorted(source_counts.items())),
        "source_selected_batches": dict(sorted(selected_source_counts.items())),
        "source_quotas": dict(sorted(source_quotas.items())),
        "length_stratum_input_batches": {
            str(key): value for key, value in sorted(stratum_counts.items())
        },
        "length_stratum_selected_batches": {
            str(key): value for key, value in sorted(selected_stratum_counts.items())
        },
    }
    return plan, details


def _canonical_row_sha(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_binary_writer(path: Path) -> tuple[BinaryIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    return os.fdopen(descriptor, "wb"), Path(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(sorted_values: list[int], numerator: int, denominator: int) -> int:
    if not sorted_values:
        raise ValueError("Cannot describe an empty distribution")
    # Nearest-rank quantile, including both endpoints.
    index = ((len(sorted_values) - 1) * numerator + denominator // 2) // denominator
    return sorted_values[index]


def describe_lengths(values: Iterable[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 1, 4),
        "p50": _quantile(ordered, 1, 2),
        "p75": _quantile(ordered, 3, 4),
        "p90": _quantile(ordered, 9, 10),
        "p95": _quantile(ordered, 19, 20),
        "p99": _quantile(ordered, 99, 100),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def total_variation_distance(
    input_counts: Counter[str], selected_counts: Counter[str]
) -> float:
    input_total = sum(input_counts.values())
    selected_total = sum(selected_counts.values())
    return 0.5 * sum(
        abs(input_counts[source] / input_total - selected_counts[source] / selected_total)
        for source in input_counts.keys() | selected_counts.keys()
    )


def write_subset(
    train: Path,
    provenance: Path,
    output_dir: Path,
    plan: list[PlannedBatch],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_train = output_dir / "train.jsonl"
    output_provenance = output_dir / "provenance.jsonl"
    train_writer, train_temporary = _atomic_binary_writer(output_train)
    provenance_writer, provenance_temporary = _atomic_binary_writer(output_provenance)
    subset_row_index = 0
    try:
        with train.open("rb") as train_handle, provenance.open("rb") as provenance_handle:
            for planned in plan:
                batch = planned.batch
                train_handle.seek(batch.train_start)
                provenance_handle.seek(batch.provenance_start)
                for _ in range(batch.row_count):
                    train_line = train_handle.readline()
                    provenance_line = provenance_handle.readline()
                    if (
                        train_handle.tell() > batch.train_end
                        or provenance_handle.tell() > batch.provenance_end
                    ):
                        raise AssertionError("Recorded batch byte range was exceeded")
                    try:
                        train_row = json.loads(train_line)
                        record = json.loads(provenance_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"Invalid selected JSON in original batch {batch.original_index}"
                        ) from error
                    expected = {"messages", "positive_messages", "negative_messages"}
                    if not isinstance(train_row, dict) or set(train_row) != expected:
                        raise ValueError(
                            f"Invalid strict train row in original batch {batch.original_index}"
                        )
                    declared_sha = record.get("row_sha256")
                    if declared_sha and declared_sha != _canonical_row_sha(train_row):
                        raise ValueError(
                            f"row_sha256 mismatch in original batch {batch.original_index}"
                        )
                    train_writer.write(train_line)
                    if not train_line.endswith(b"\n"):
                        train_writer.write(b"\n")
                    projected = dict(record)
                    projected["fa2_probe_subset"] = {
                        "selection_contract": SELECTION_CONTRACT,
                        "subset_row_index": subset_row_index,
                        "subset_batch_index": planned.subset_batch_index,
                        "optimizer_step": planned.optimizer_step,
                        "gradient_accumulation_slot": planned.accumulation_slot,
                        "length_stratum": planned.length_stratum,
                        "original_batch_index": batch.original_index,
                    }
                    provenance_writer.write(
                        json.dumps(
                            projected,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode()
                        + b"\n"
                    )
                    subset_row_index += 1
                if train_handle.tell() != batch.train_end:
                    raise AssertionError("Recorded train batch byte range did not close")
                if provenance_handle.tell() != batch.provenance_end:
                    raise AssertionError("Recorded provenance batch byte range did not close")
        for writer in (train_writer, provenance_writer):
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        train_writer.close()
        provenance_writer.close()
        train_temporary.unlink(missing_ok=True)
        provenance_temporary.unlink(missing_ok=True)
        raise
    else:
        train_writer.close()
        provenance_writer.close()
        os.replace(train_temporary, output_train)
        os.replace(provenance_temporary, output_provenance)
    return output_train, output_provenance


def build(args: argparse.Namespace) -> dict[str, Any]:
    train = args.train.resolve()
    provenance = args.provenance.resolve()
    output_dir = args.output_dir.resolve()
    for name in (
        "batch_size",
        "probe_steps",
        "gradient_accumulation_steps",
        "training_max_length",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not train.is_file() or not provenance.is_file():
        raise FileNotFoundError("Both --train and --provenance must be existing files")
    output_paths = {
        output_dir / "train.jsonl",
        output_dir / "provenance.jsonl",
        output_dir / "manifest.json",
    }
    if train in output_paths or provenance in output_paths:
        raise ValueError("Output directory must not overwrite either input")

    scanned = scan_aligned(train, provenance, args.batch_size)
    plan, selection = plan_subset(
        scanned.batches,
        probe_steps=args.probe_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )
    output_train, output_provenance = write_subset(
        train, provenance, output_dir, plan
    )

    input_source_counts = Counter(batch.source_id for batch in scanned.batches)
    selected_source_counts = Counter(item.batch.source_id for item in plan)
    input_lengths = [batch.length_proxy_max for batch in scanned.batches]
    selected_lengths = [item.batch.length_proxy_max for item in plan]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_contract": SELECTION_CONTRACT,
        "purpose": "single-GPU 8B attention-backend admission workload only",
        "release_eligible": False,
        "parameters": {
            "batch_size": args.batch_size,
            "probe_steps": args.probe_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "selected_batches": len(plan),
            "selected_rows": len(plan) * args.batch_size,
            "training_max_length_tokens": args.training_max_length,
            "selection_seed": args.seed,
            "length_strata": args.gradient_accumulation_steps,
        },
        "semantics": {
            "complete_source_homogeneous_batches": True,
            "train_rows_copied_byte_for_byte": True,
            "source_allocation": "Hamilton apportionment over complete input batches",
            "length_allocation": (
                "equal-frequency global strata over batch max text-character length proxy; "
                "one stratum per accumulation slot per optimizer step"
            ),
            "length_proxy_note": (
                "The proxy is the maximum Unicode-character length emitted by the upstream "
                "homogeneous-batch builder; ms-swift computes exact tokenizer lengths only "
                "for this subset and applies the recorded training max_length token cap."
            ),
            "backend_comparison_requirement": (
                "SDPA and FlashAttention 2 must be measured on this exact train SHA256, "
                "with identical model, batch, accumulation, seed, and warm-up settings."
            ),
        },
        "inputs": {
            "train": {
                "path": str(train),
                "sha256": scanned.train_sha256,
                "rows": scanned.rows,
            },
            "provenance": {
                "path": str(provenance),
                "sha256": scanned.provenance_sha256,
                "rows": scanned.rows,
            },
            "complete_batches": len(scanned.batches),
        },
        "representativeness": {
            **selection,
            "source_total_variation_distance": total_variation_distance(
                input_source_counts, selected_source_counts
            ),
            "input_batch_length_proxy": describe_lengths(input_lengths),
            "selected_batch_length_proxy": describe_lengths(selected_lengths),
        },
        "optimizer_step_plan": [
            {
                "optimizer_step": step,
                "original_batch_indices": [
                    item.batch.original_index
                    for item in plan
                    if item.optimizer_step == step
                ],
                "sources": [
                    item.batch.source_id for item in plan if item.optimizer_step == step
                ],
                "length_strata": [
                    item.length_stratum for item in plan if item.optimizer_step == step
                ],
                "batch_length_proxy_max": [
                    item.batch.length_proxy_max
                    for item in plan
                    if item.optimizer_step == step
                ],
            }
            for step in range(1, args.probe_steps + 1)
        ],
        "files": {
            "train.jsonl": {
                "path": str(output_train),
                "rows": len(plan) * args.batch_size,
                "sha256": _sha256(output_train),
            },
            "provenance.jsonl": {
                "path": str(output_provenance),
                "rows": len(plan) * args.batch_size,
                "sha256": _sha256(output_provenance),
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".manifest.json.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    manifest = build(parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
