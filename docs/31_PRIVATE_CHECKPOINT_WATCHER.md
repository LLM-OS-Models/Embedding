# Private checkpoint candidate watcher

`scripts/watch_private_adapter_checkpoints.py` watches a LoRA training output
for completed `checkpoint-N` directories and incrementally commits only
verified adapter candidates to a private model repository. The active Qwen
200K repository is:

`LLM-OS-Models2/qwen3-embedding-8b-ko-performance200k-lora-r64-candidates`

The watcher does not start or modify training and uploading is disabled unless
`--upload` is given. The active Qwen process uses a separately supervised
watcher. Future 1M/KD/target/legal LoRA entrypoints opt in through
`ENABLE_PRIVATE_CHECKPOINT_WATCHER=1`; the trainer wrapper starts the same
watcher, stops it after training, and runs a final `--once --settle-seconds 0`
reconciliation so the last completed checkpoint cannot be missed.

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
base model and revision are replaced with a Hub model ID and pinned Git SHA, so
a local model-cache path cannot leak. For continual training from a local
winner, the wrapper requires the preceding private full-model upload report,
checks its model path and 64-hex weight identity against local merge/full/soup
evidence, and only then records that private repository's exact 40-hex commit.

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

Visibility is not trusted from an earlier cached check. It is re-read before
and after both a new commit and crash-recovery acceptance of an existing remote
manifest, so a repository changed to public mid-run is never recorded as a
successful private backup.

The generated manifest contains only checksums, tensor summary, same-step
validation evidence, pinned base lineage, run ID, and optional input-manifest
and FA2-admission SHA-256 values. It contains no local path.

## Local averaging archive

Upload mode also retains the already validated adapter snapshot under the
ignored watch-root directory `.adapter-checkpoint-archive/<training-version>/`.
Each archived checkpoint contains only the adapter safetensors, the sanitized
adapter config, and a checksum-bound `archive_manifest.json`. Trainer state,
optimizer state, logs, data, and credentials are never copied there. The
snapshot is atomically renamed only after the remote state record is durable.

This archive is deliberately separate from Trainer retention. Even when an
active run keeps only three full resume checkpoints, the final FP32 averaging
stage can use the latest five validated adapters from the same exact training
version. On watcher restart, an uploaded checkpoint that is still local but
missing from the archive is revalidated and backfilled without another Hub
commit. `--no-local-archive` is available for storage-constrained runs, but is
not used by this performance-first campaign.

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

BF16 finite 검증은 기본적으로 PyTorch CPU thread 1개만 사용한다. r64 adapter 전체 payload를
검사하면서 학습 dataloader의 CPU를 빼앗지 않기 위한 production 기본값이다. 학습이 없는
maintenance window에서만 `EMBEDDING_WATCHER_TORCH_THREADS=2`처럼 최대 8까지 명시적으로
올릴 수 있다. watcher는 `torch`와 `safetensors`가 모두 있는 `.venv-train-fa2`로 실행한다.

Use `Ctrl-C` to stop. `--once --upload` processes all currently completed,
previously unseen checkpoints and exits.

Transient Hub repository checks, manifest recovery reads, and uploads are
retried up to three times with a 15-second delay. The retry set is deliberately
narrow: public visibility, checksum conflict, unsafe local files, invalid
state, and credential/config failures remain immediate hard stops. The bounds
can be tightened with `--remote-attempts` and `--remote-retry-seconds`.

### 2026-07-17 Qwen 복구 실측

첫 새 production checkpoint인 step 250은 same-step `eval_loss=0.003494835924357176`,
698,419,728-byte adapter 전체 finite 검증을 통과했다. private repo
`LLM-OS-Models2/qwen3-embedding-8b-ko-performance200k-lora-r64-candidates`의 commit
`7da3a5737e332f0a85981a5fcaa02aecfe7df6c7`에 allowlist 3파일이 올라갔다. 원격
`candidate_manifest.json` SHA `4a9bc1e73b6aeab2c253ccfd23ee1f77b543a0287f59723fe8dbc12b4b3f6270`와
adapter SHA `f004b4fb012eb69807147f4b24b40cddb6ab4a00380397a1c45a74a3c3b68ac2`는 local state와
재조회 결과가 exact match였고 repo visibility는 private였다.

Step 500도 `eval_loss=0.00343669`로 step 250보다 개선됐으며 commit
`ea613d324cbccdacb1385bb9e437fa877df35af3`에 업로드됐다. 원격 prefix는 allowlist 3파일만
포함했고, 698,419,728-byte LFS object SHA
`36268988d5b529dc364ad242ef98900d0f80d126d490b40a0bbbe2d32cc386f7`와 manifest SHA
`f77f0ac6788adda45a48eb60d5d088f625d5f111c8ef099753d2af5beb9ca249`가 local state와
일치했으며 repo visibility도 private로 재확인했다.

Step 750은 `eval_loss=0.00342328`, commit
`adf013c90eeb6e93fe7de905f854870bc707c645`로 보존됐다. Step 1000은
`eval_loss=0.0034344`, commit `363fede65381bc597904a2f49f4514ced318080b`로 보존됐다.
Step 1000의 698,419,728-byte remote LFS SHA
`b5a65f907b63079476f2c80d9ebc4df4cde2c5b0488737a544a2fff36499fe57`와 manifest SHA
`8222ebb346f1995f0e981f5a2888dbebfda1bd3f41efdd5323b6b296fcf1cb2a`는 local state/archive와
exact match였고, 해당 commit의 prefix는 allowlist 3파일뿐이며 repo는 private였다.

## Continual-base 복구 사슬

selection-only 200K lineage winner와 capacity 포함 winner는 public benchmark를 호출하지 않은
채 Grade-I clean/robustness evidence로 고른 전체 모델을 각각 private upload한다. uploader는
merge LoRA뿐 아니라 partial-full 및 fixed soup evidence도 같은 weight SHA gate로 검증하고,
remote manifest 재다운로드가 일치한 뒤 `repo_id`, `commit_sha`, `weights_sha256`를 atomic report로
남긴다. 1M은 capacity report, KD는 1M report, specialist/legal/combined는 KD winner report를
사용한다. 어느 local base든 report의 path·weight SHA·private visibility·remote manifest·commit이
하나라도 맞지 않으면 다음 watcher를 시작하지 않는다. 학습 프로세스는 Hub token-free offline이고
watcher child에서만 inherited token과 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`/
`HF_DATASETS_OFFLINE`을 제거한 뒤 mode-0600 `.env`를 process memory에서 읽는다.

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
public-repository refusal. It also checks the sanitized local averaging archive
and archive-aware five-checkpoint selection.
