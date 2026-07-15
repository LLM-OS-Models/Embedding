# Private checkpoint candidate watcher

`scripts/watch_private_adapter_checkpoints.py` watches a 200K LoRA training
output for completed `checkpoint-N` directories and incrementally commits only
verified adapter candidates to the private model repository:

`LLM-OS-Models/qwen3-embedding-8b-ko-performance200k-lora-r64-candidates`

The watcher is deliberately separate from the training process. It does not
start or modify training and uploading is disabled unless `--upload` is given.

## Upload contract

Each remote checkpoint prefix contains exactly three files:

```text
checkpoints/checkpoint-250/
  adapter_model.safetensors
  adapter_config.json
  candidate_manifest.json
```

No directory upload is used. One atomic `create_commit` is assembled from an
in-code allowlist of those three paths. The following local files are never
staged, uploaded, or copied into the manifest:

- `optimizer.pt`, scheduler/scaler state, and RNG state
- `trainer_state.json` and `training_args.bin`
- `args.json`, logs, raw data, or processed data
- local absolute paths or credentials

The source `adapter_config.json` is reconstructed from pinned PEFT fields. Its
base model and revision are replaced with the public model ID and pinned Git
SHA, so a local model-cache path cannot leak.

## Completion and validation gates

A candidate is eligible only when all gates pass:

1. The directory name is exactly `checkpoint-[1-9][0-9]*`, is not a symlink,
   resolves inside the watch root, and is unique for that step.
2. `adapter_model.safetensors`, `adapter_config.json`, and the local-only
   `trainer_state.json` completion sentinel are non-empty regular files whose
   inode/size/mtime fingerprints remain unchanged for `--settle-seconds`.
3. `trainer_state.json.global_step` matches the directory suffix and contains a
   finite same-step `eval_loss`. Thus a checkpoint saved before validation does
   not become a candidate.
4. Weights are copied to a mode-`0700` private staging directory. Source
   fingerprints are checked again after the copy.
5. Every staged safetensors payload is materialized on CPU; only F16/BF16/F32
   tensors are accepted, LoRA A/B tensors must exist, and every value must be
   finite. The full staged file SHA-256 is recomputed before upload.
6. The Hub repository is created with `private=True` and its visibility is
   checked both before and after the commit. A public or unknown-visibility
   repository is rejected.

The generated manifest contains only checksums, tensor summary, same-step
validation evidence, pinned base lineage, run ID, and optional input-manifest
and FA2-admission SHA-256 values. It contains no local path.

## Token handling

Do not put a token on the command line and do not run `huggingface-cli login`.
The watcher checks `HF_TOKEN`, then `HUGGINGFACE_HUB_TOKEN`, then parses only
those keys from the ignored repository `.env`. The `.env` must be a regular
mode-`0600` file. The token is passed directly to `HfApi` in process memory; it
is never exported, printed, stored in state, or included in an exception.

## Dry validation

First validate existing checkpoints without any Hub request:

```bash
.venv-train-fa2/bin/python scripts/watch_private_adapter_checkpoints.py \
  --watch-dir outputs/qwen3-embedding-8b-ko-performance200k-lora-r64 \
  --once
```

Without `--upload`, no token is read, no repository is created, and no state is
marked uploaded. Output is restricted to sanitized JSON events containing a
checkpoint label, step, and checksum; it never includes the watch path.

## Continuous private upload

After recording the exact training-manifest and admission-report checksums,
start the watcher alongside training:

```bash
.venv-train-fa2/bin/python scripts/watch_private_adapter_checkpoints.py \
  --watch-dir outputs/qwen3-embedding-8b-ko-performance200k-lora-r64 \
  --training-manifest-sha256 <64-hex-sha256> \
  --admission-report-sha256 <64-hex-sha256> \
  --poll-seconds 5 \
  --settle-seconds 10 \
  --upload
```

The watch root may contain ms-swift's `v0-...` version directory; discovery is
recursive but does not follow symlinks. If two version directories contain the
same `checkpoint-N`, the watcher fails closed instead of choosing one.

Use `Ctrl-C` to stop. `--once --upload` processes all currently completed,
previously unseen checkpoints and exits.

## Idempotency and recovery

The local state is
`.hf-candidate-upload-state.json` beside the watch root and is atomically
written mode `0600` only after a successful remote commit. An exclusive process
lock prevents two local watchers from racing. An uploaded step in state is
never uploaded again.

There is also a remote recovery gate for the crash window between a successful
commit and the local state write. Before committing, the watcher checks the
deterministic remote `candidate_manifest.json`:

- matching adapter and config checksums: recover local state without a commit;
- same step with different checksums: hard failure, never overwrite;
- no remote manifest: commit with the observed repository HEAD as
  `parent_commit` (compare-and-swap behavior).

## Offline verification

The test suite uses real safetensors fixtures and a mocked Hub API. It makes no
network request and creates no repository:

```bash
python -m pytest -q tests/test_watch_private_adapter_checkpoints.py
```

It covers allowlist-only commits, private visibility checks, state and remote
idempotency, incomplete/changing checkpoints, corrupt and non-finite tensors,
symlink rejection, source races, token/path redaction, dotenv permissions, and
public-repository refusal.
