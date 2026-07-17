# Private checkpoint candidate watcher

`scripts/watch_private_adapter_checkpoints.py` watches a LoRA training output
for completed `checkpoint-N` directories and incrementally commits only
verified adapter candidates to a private model repository. The active Qwen
200K repository is:

`LLM-OS-Models2/qwen3-embedding-8b-ko-performance200k-lora-r64-candidates-v2`

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

새 watcher process는 commit 응답만으로 state를 전진시키지 않는다. returned immutable
revision을 `files_metadata=true`로 다시 조회해 checkpoint prefix가 exact 3-file allowlist인지,
adapter LFS SHA-256/size가 local finite-inspected payload와 같은지, config와 manifest를 같은
revision에서 재다운로드한 SHA가 같은지 확인한 뒤에만 `uploaded`를 atomic state에 쓴다.
commit 응답 유실 뒤 existing manifest로 복구할 때도 `created_at_utc`만 제외한 전체 manifest
의미 구조를 비교하므로 adapter SHA가 같아도 base/train/admission lineage가 다르면 거부한다.
LFS 또는 metadata mismatch는 retryable upload 응답 오류로 취급하지 않고 fail closed한다.

### 2026-07-17 Qwen 복구 실측

첫 watcher command가 실제 train SHA `8e2731ab25299ff558af675f067b253a6ce4375a850aa925acfe3b3117505e3c`
대신 `8e2731a93fb6...`를 manifest lineage에 잘못 넣은 사실을 step 1000 원격 재감사에서
발견했다. adapter payload와 finite 검증은 맞았지만 provenance가 틀렸으므로 기존 private repo는
성능 선택/복구에서 제외했고 root commit `9ec6b86bf0d1f13d645824179e7989c3b3fff2d9`에
superseded card를 올렸다. 기존 artifact는 감사 추적을 위해 덮어쓰거나 삭제하지 않았다.

새 private `...-candidates-v2`는 실제 파일에서 계산한 SHA를 사용한다. 회전 전에 full checkpoint가
남은 step 500/750/1000은 각각 commits `ade73a6a554ce715edd9aeed253b55c850290537`,
`fecf0c730552c8b1cef038aac1f2e322c092ab2e`,
`684a680765e58123f6b48f8d4aab859c2813974a`로 watcher가 다시 전체 finite 검증했다.
manifest SHA는 각각 `a434f20a17c6defc2edd3ed15355de659f2872e509ed849abc672e3414a232c7`,
`2ba942388f8aeb603353bd9228047fe2c34f5f3151d5fe6aea485be98224b054`,
`e058c0160a8199310d4c481229ebabffe04a35eb82eca6f72fef66ccc775f8e8`다.

Step 250은 Trainer 회전으로 full checkpoint가 사라지고 sanitized archive만 남아 있었다.
`correct_private_candidate_lineage.py`가 기존 private commit
`7da3a5737e332f0a85981a5fcaa02aecfe7df6c7`의 same-step eval/full-payload manifest,
로컬 archive SHA와 698,419,728-byte tensor 전체 finite를 함께 재검증했다. adapter/config bytes를
바꾸지 않고 lineage correction provenance를 추가해 v2 commit
`7d3b3ed1103fe50ed94d7b305c5b1461cd487d3e`에 이관했다. corrected manifest SHA는
`98bc5a8dfc53088e8b02c238b3ed005a23344bdd34499c26683348bb9fc1a744`다.
네 step 모두 remote prefix 3파일, LFS SHA/size, corrected training SHA, manifest download exact와
private visibility를 최신 v2 head에서 다시 확인했다. eval loss는 step 250/500/750/1000 순서로
`0.003494835924357176`/`0.00343669`/`0.00342328`/`0.0034344`이며 성능 선택에는 쓰지 않는다.

같은 watcher는 2026-07-17 16:37 KST에 새 step 1250도 자동 보존했다. private immutable commit은
`38ae9b1dabb022b36ce296036e66ae1a6b8343c1`, adapter SHA는
`86846cb39594687ee3301db6164375752fd0b4632866dc6db35741fc10f08d8e`, config SHA는
`eac78f0773789a361251829e3cde41561a94873bf0d5fece6cd94d22661b4d3d`, manifest SHA는
`1887f0fe3f26b858239ead6034ef919c3a19f62113141649f149a075257ce6ef`다. 독립 remote
재조회에서 private visibility, exact 3-file prefix, 698,419,728-byte adapter LFS SHA/size,
config/manifest download SHA, base revision, actual train SHA `8e2731ab…`, training manifest
`eeed4fcd…`, admission report `c409291a…`를 모두 local archive/state와 대조했다. 같은 step의
legacy eval loss `0.00344981`은 completion/finite 증거일 뿐 checkpoint 선택 점수가 아니다.
full local checkpoint에는 adapter와 optimizer/scheduler/RNG/trainer state가 모두 남아 exact
resume 가능하고, sanitized archive는 adapter/config/archive manifest 세 파일만 갖는다.

2026-07-17 17:34 KST에는 step 1500도 같은 watcher가 자동 보존했다. private immutable commit은
`82f3fd1188e91829795e0f72664a322bbc8fabbc`, adapter SHA는
`86032a3f5073e49601471c8effcdd40020149e2e861dab20d416fdaca58eb55a`, 698,419,728 bytes,
config SHA는 `eac78f0773789a361251829e3cde41561a94873bf0d5fece6cd94d22661b4d3d`, manifest
SHA는 `6fab14dc9b60ad5249a85550b317956926fe9019b993e4b52ee5454f3df50141`다. watcher state와
별개인 one-shot remote 감사에서 해당 immutable revision의 private visibility, exact 3-file
prefix, adapter LFS SHA/size, config/manifest download SHA, base/data/training-manifest/admission
lineage를 local archive와 모두 대조했다. full local checkpoint에는 optimizer
1,397,128,954-byte payload와 scheduler/RNG/trainer state가 있고 `global_step=1500` 및 same-step
finite eval이 일치한다. legacy eval loss `0.0034647216089069843`는 completion 증거일 뿐
성능 선택에는 사용하지 않는다.

2026-07-17 18:35 KST의 step 1750 immutable commit은
`1b84bea2cad8b9ac31180a7da77f57cd96475bc6`다. adapter SHA는
`b835258d526a7f857940b71c9fdfa8511b226f895ec25f216a45994b158eae62`, size는
698,419,728 bytes, config SHA는
`eac78f0773789a361251829e3cde41561a94873bf0d5fece6cd94d22661b4d3d`, candidate manifest
SHA는 `e448695686c5d99cf095375821db842f348004efc5ba21955db9fdfc9e0a3461`다. watcher와
독립된 one-shot 감사가 immutable revision의 private visibility, exact 3-file prefix,
adapter LFS identity, metadata download SHA, Qwen base revision과 data/training/admission
lineage를 재검증했다. local sanitized archive의 504개 tensor도 전부 다시 읽어 finite임을
확인했다. same-step legacy eval loss `0.0035100767854601145`는 completion 증거일 뿐 모델
선택 점수가 아니다.

## Continual-base 복구 사슬

selection-only 200K lineage winner와 capacity 포함 winner는 public benchmark를 호출하지 않은
채 Grade-I legal/multidomain/robustness evidence로 고른 전체 모델을 각각 private upload한다. uploader는
merge LoRA뿐 아니라 partial-full 및 fixed soup evidence도 같은 weight SHA gate로 검증하고,
remote manifest 재다운로드가 일치한 뒤 `repo_id`, `commit_sha`, `weights_sha256`를 atomic report로
남긴다. 1M은 capacity report, KD는 1M report, specialist/legal/combined는 KD winner report를
사용한다. 어느 local base든 report의 path·weight SHA·private visibility·remote manifest·commit이
하나라도 맞지 않으면 다음 watcher를 시작하지 않는다. 학습 프로세스는 Hub token-free offline이고
watcher child에서만 inherited token과 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`/
`HF_DATASETS_OFFLINE`을 제거한 뒤 mode-0600 `.env`를 process memory에서 읽는다.

## Exact Trainer resume

`AUTO_RESUME_FROM_LATEST_CHECKPOINT=1`이 LoRA entrypoint 기본값이다. 기존 checkpoint가 하나라도
있으면 `select_best_checkpoint.py --latest-resume`가 최신 step 하나만 고른다. adapter와
`trainer_state.json`, optimizer/scheduler/RNG/training args가 모두 non-empty regular file이고,
directory step과 `global_step`, 같은 step finite `eval_loss`가 일치해야 한다. 서로 다른 training
version에 같은 step이 두 개면 임의로 고르지 않고 실패한다.

이어 `validate_resume_checkpoint.py`가 train/validation exact path, base model/revision,
max steps, microbatch/accumulation, max length, LoRA rank/alpha/dropout, learning rate, loss,
두 shuffle flag와 seed를 현재 invocation과 비교한다. 전부 맞을 때만 Swift에
`--resume_from_checkpoint`를 전달하고 atomic `resume-validation.json`을 남긴다. existing history가
있는데 valid resume point가 없으면 새 v1 run을 조용히 시작하지 않는다. 현재 Qwen step 1000
checkpoint도 원래 legacy validation path를 명시한 exact contract로 이 validator를 통과했다.
legacy eval loss는 여전히 성능 선택에 사용하지 않고 독립 Grade-I 10K로 재선택한다.

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
