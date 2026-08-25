# Pretrained-representation implementation

This document describes the first executable layer behind the prospective Cosmos, V-JEPA, and
DINOv3 study. It is an engineering implementation, not foundation-model evidence.

## Implemented now

The new `latent_safety.foundation` namespace is deliberately separate from the legacy AE/beta-VAE
trainer. The latter assumes reconstruction, scalar actions, and controlled synthetic domains; it
is useful plumbing but is the wrong abstraction for heterogeneous pretrained encoders.

The initial vertical slice provides:

- a canonical visual input contract: `[B,T,C,H,W]`, float RGB in `[0,1]`, plus an explicit valid
  frame mask and optional time-aligned proprioception;
- a common representation boundary with dense tokens `[B,N,D]`, a separately specified pooled
  decision code, token mask, latent grid, typed discrete IDs and class/register tokens, diagnostic
  native outputs, and declared artifact/config provenance;
- concrete family/backend contracts for Cosmos Tokenize1 CV/DV, V-JEPA 2/2.1, and DINOv3 ViT
  without importing or downloading heavyweight upstream code;
- a viable-only buffered common-action audit with traceable sample witnesses, deterministic action
  ties, trajectory-balanced tail summaries, and fail-closed empty-trajectory coverage;
- a dependency-free reference implementation of the hard Common-Action Geometry objective;
- a lazy PyTorch implementation of the normalized-softmin repair surrogate, boundary/tail profile
  loss, feature retention, pairwise-geometry retention, neighborhood-coverage diagnostics, and the
  combined named objective;
- an identity-initialized residual bottleneck intervention, an explicitly action-conditioned
  training head, and trainable-parameter, gradient-route, and frozen-parameter mutation guards; and
- a whitelist-based encoder checkpoint helper that removes the declared training head and rejects
  every undeclared controller, side channel, or other non-encoder subtree.

The legacy exact-fiber, radius, and matched-radius metrics were also corrected to filter the viable
reference population *before* forming fibers or neighborhoods. A nearby physically infeasible state
can no longer hide a conflict between viable states.

## Executable audit

Once an extractor has emitted portable held-out `AuditRecord` JSONL rows, run:

```bash
python3 scripts/run_foundation_audit.py \
  --records /path/to/validation_records.jsonl \
  --delta 0.05 \
  --gamma 0.0 \
  --output /path/to/foundation_audit.json
```

Validation is the default declared input split, so the ordinary validation path needs no access
flags. The command refuses overwrite, pins the input and implementation hashes, emits sample-level
witnesses, and labels its result diagnostic. Its self-hashed envelope also records the producing
command, UTC creation time, Git revision and dirty state, Python and installed-distribution
environment, and the implementation of the shared manifest builder.

Final-test access is fail-closed. Before the command reads or hashes the test record file, it
requires all three of a checksum-pinned pre-access freeze, an explicit unblinding flag, and a
non-empty immutable approval or handoff identifier:

```bash
python3 scripts/run_foundation_audit.py \
  --records /path/to/test_records.jsonl \
  --input-split test \
  --delta 0.05 \
  --gamma 0.0 \
  --test-freeze /path/to/foundation_test_freeze.json \
  --test-freeze-sha256 <sha256-pinned-in-the-pre-access-handoff> \
  --unblind-test \
  --access-provenance final-test-handoff-2026-08-25 \
  --output /path/to/foundation_test_audit.json
```

`--test-freeze-sha256` is the digest pinned by the pre-access handoff, not a digest discovered after
unblinding. The freeze uses protocol `foundation_test_audit_freeze_v1`. It must contain exactly
`schema_version`, `protocol`, `status`, `analysis`, `input`, `audit_parameters`, `code`, and
`manifest_sha256`; pin split `test`, the exact record-file SHA-256, every audit parameter, and a
clean Git object ID; and set `status` to `frozen_before_test_access`. `manifest_sha256` is the
canonical JSON SHA-256 of the other seven fields. The audit runs only from the same clean revision.
Calibration remains available through `--input-split calibration` without test-unblinding flags.

Promotion additionally requires checkpoint, preprocessing, normalization, action-library,
simulator, split, and environment manifests.

## Repair objective

For fixed physical profile targets `q` and calibration-normalized decision codes `z`, the hard
reference implementation computes

```text
C[i,a] = sum over viable j of
         relu(gamma - q[j,a]) * relu(delta_train - distance(z[i],z[j]))^2
L_CA    = mean over viable i of min_a C[i,a].
```

The action minimum is taken only after aggregating a whole centered set. This is what preserves the
higher-order conflict signal. Profile targets are detached in the differentiable path. The
optimization surrogate uses a normalized log-mean-exp soft minimum, which stays nonnegative; the
hard minimum is always logged separately because only its zero set has the finite-reference
interpretation in the paper.

`delta_train` must be strictly greater than `delta_audit`. The audit ball is closed, while the
compact-support training hinge is zero at its boundary. The implementation does not fit a batch
normalizer: callers must provide features transformed by the frozen calibration normalizer.

An exact duplicate has zero radial Euclidean gradient under the CA term alone. The joint action
profile loss can supply asymmetric encoder gradients when the colliding inputs have different
upstream Jacobians; no deterministic encoder can split truly identical inputs. The experimental
protocol therefore retains both cases as documented optimization and observation-floor controls
rather than claiming that CA-only always splits exact aliases.

## Family adapter facts and boundaries

No real foundation-model loader exists in this vertical slice. `DeclaredCallableAdapter` validates
shapes, family/backend semantics, required artifact roles, feature views, repair scopes, and named
repair parameter groups around caller-supplied functions. It cannot inspect those functions for
downloads, network traffic, checkpoint hashing, or code execution. Its manifest therefore records
loader properties under `caller_declaration_not_observed_by_adapter` and separately records that the
scaffold did not read artifacts, verify hashes, inspect a checkout, or observe callable behavior.
`PinnedCallableAdapter` remains only as a backward-compatible import alias; it makes no stronger
claim.

Production loaders must resolve local artifacts, verify every declared SHA-256 before
deserialization, check out a full source commit, disable network access during load, and record any
installed-package or pinned-local upstream code that executes. Local `torch.hub` is still code
execution, a local DINO weight argument may still be a URL, and PyTorch/TorchScript files are not made
safe merely by residing on disk. These are future loader requirements, not properties implemented by
the callable scaffold.

The registry uses concrete keys rather than family-wide aliases:

- `cosmos_tokenize1_{cv,dv}_{jit,native_training}` separates continuous/discrete outputs and frozen
  TorchScript from the training checkpoint needed for internal repair;
- `vjepa2_{native,hf}` separates native BCHTW from Transformers BTCHW conventions;
- `vjepa21_native_video_384` names the supported official 2.1 video path; and
- `dinov3_vit_{native,hf}` excludes incompatible DINOv3 ConvNeXt backbones and makes patch extraction
  backend-specific.

Each registry entry separates `facts`, sourced from upstream APIs, from the study's `policy`, such as
canonical conversion, pooling view, allowed control arms, and proposed intervention location.

`AdapterCapabilities.accepts_video_batches` describes the full adapter. `native_temporal_mode`
separately records `causal_video`, `bidirectional_video`, or `framewise_image`. Thus a framewise DINO
adapter truthfully accepts `[B,T,C,H,W]` while the upstream encoder remains image-only. The repair
scope is one of `none`, `post_representation_control`, or `encoder_internal`; reconstruction and
prediction booleans were removed because no decoder or predictor protocol is implemented.

### Cosmos Tokenizer

The maintained Tokenize1 path is in
[Cosmos Predict1](https://github.com/nvidia-cosmos/cosmos-predict1), with the official
[tokenizer inference guide](https://docs.nvidia.com/cosmos/latest/predict1/tokenizer/inference_guide.html).
Cosmos consumes `[B,C,T,H,W]` scaled to `[-1,1]`. Continuous codes are flattened from
`[B,Cz,Tz,Hz,Wz]`. Discrete `indices [B,Tz,Hz,Wz]` become the typed integer `discrete_ids [B,N]`,
while `codes [B,Cz,Tz,Hz,Wz]` are the pre-quantization continuous geometry. The registered contracts
cover video CV/DV checkpoints only; image CI/DI checkpoints must not be passed under these keys.

The official frozen interface consumes `encoder.jit`. The official native inference path can
instantiate PyTorch modules while extracting state dictionaries from JIT files, whereas
`config.json` plus `model.pt` are the full training/post-training checkpoint path. The scaffold
therefore permits `encoder_internal` only for the explicit `native_training` variants. An internal
hook before final compression or token assignment remains a revision-specific study intervention to
validate, not an upstream stable API. JIT variants support frozen extraction and a
post-representation negative control only.

### V-JEPA 2 and 2.1

The official source is [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
Native video encoders consume `[B,C,T,H,W]` and return dense `[B,N,D]` tokens; V-JEPA 2.1 also has a
separate `[B,C,H,W]` image branch, so media mode cannot be inferred silently from `T=1`. Official Hub
factories return `(encoder, predictor)`, so the frozen representation path explicitly discards the
predictor. The official Transformers V-JEPA 2 path instead consumes
`pixel_values_videos [B,T,C,H,W]` and must use `skip_predictor=True` or the encoder feature method.
V-JEPA 2.1 uses the native official checkpoints here; community HF ports are not relabeled as
official baselines.

The current upstream native `main` has a localhost pretrained base URL, which reinforces the rule:
instantiate from a pinned local checkout with implicit pretraining disabled, then load an explicitly
verified local checkpoint. A proposed repair hook must name a module before final encoder blocks;
an MLP on returned tokens is only the post-representation control.

### DINOv3

The official source is [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3). This
study contract covers ViT backbones only. DINO is applied framewise to the same source timestamps and
history window as the video encoders, though family-specific crops and resolutions mean the
post-transform tensors need not be identical. The complete adapter accepts video batches while its
native temporal mode remains `framewise_image`.

Native dense features come from `forward_features["x_norm_patchtokens"]`, not default `forward`,
which applies the head to the class token. Native class and storage/register tokens are mapped to
separate typed fields. The HF `last_hidden_state` contains CLS, register tokens, then patches, so the
patch slice begins at `1 + config.num_register_tokens`. The transform must assert height and width
are multiples of patch size 16 rather than relying on DINO's silent crop to a lower multiple.

A future shared, mask-aware temporal reducer will supply the standardized pooled comparison and be
identified by both an ID and a config/state hash. The current protocol records those declarations
but does not implement or verify the reducer. A named hook before final ViT blocks is the internal
repair arm; an MLP after returned tokens is the negative control.

## Environment isolation

The upstream stacks are not one clean environment: Cosmos Predict1 targets a Linux/Python 3.10
CUDA stack with compiled dependencies, V-JEPA targets Python 3.11+, and full DINOv3 training has a
newer PyTorch/Linux requirement. The implementation therefore expects separate pinned GPU
environments or containers:

- `cosmos-py310` for native tokenizer extraction and repair;
- `vjepa-py311` for V-JEPA extraction and internal adapters; and
- `dinov3-pt271` for DINOv3 extraction and internal adapters.

Each feature artifact crosses environments only through the common manifest and array schema.
Access and licenses are family-specific: Cosmos weights are gated under the NVIDIA Open Model
License, DINOv3 weights are gated under the custom DINOv3 License, and the official V-JEPA 2 HF
checkpoints are public and marked MIT. Redistribution of every original or adapted checkpoint must
follow its actual selected model-card revision and institutional ownership policy; the scaffold's
`license_id` is a declaration, not a legal compliance check.

## Next implementation milestone

The next vertical slice is intentionally narrow:

1. add a rich versioned foundation record and feature-array manifest;
2. implement a no-download mock adapter and saved-state branching mock domain end to end;
3. implement real loaders that hash explicit local DINOv3 and V-JEPA 2 artifacts before loading and
   emit observed loader evidence rather than callable declarations;
4. add deterministic chunked neighborhood construction with exact small-fixture parity;
5. wire the existing dependency-lazy intervention builders into model-specific internal hooks, then
   add gradient-route and encoder-only export/reload tests; and
6. only then add ManiSkill3 and Habitat saved-state branching and launch GPU smoke runs.

No Cosmos, V-JEPA, DINOv3, ManiSkill3, or Habitat result has been run or promoted yet.
