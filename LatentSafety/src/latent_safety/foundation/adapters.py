"""Dependency-light family/backend contracts and a declared callable scaffold."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from latent_safety.foundation.protocols import (
    AdapterCapabilities,
    AdapterProvenance,
    LoaderDeclaration,
    RepairParameterGroup,
    RepresentationBatch,
    VisualBatch,
)


@dataclass(frozen=True)
class UpstreamAdapterFacts:
    """Facts verified from a named upstream family and backend API."""

    family: str
    model_scope: str
    backend: str
    upstream_repository: str
    native_input_layout: str
    native_feature_view: str
    native_temporal_mode: str
    representation_kind: str
    required_artifact_roles: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class AdapterStudyPolicy:
    """Study decisions layered on top of upstream behavior; these are not upstream facts."""

    canonical_input_conversion: str
    adapter_feature_layout: str
    accepts_video_batches: bool
    required_feature_view: str
    allowed_repair_scopes: tuple[str, ...]
    repair_note: str
    loader_policy: str


@dataclass(frozen=True)
class FamilyAdapterContract:
    """Concrete upstream-family/backend facts plus an explicitly separate study policy."""

    key: str
    facts: UpstreamAdapterFacts
    policy: AdapterStudyPolicy


_COSMOS_REPOSITORY = "https://github.com/nvidia-cosmos/cosmos-predict1"
_COSMOS_INFERENCE = (
    "https://docs.nvidia.com/cosmos/latest/predict1/tokenizer/inference_guide.html"
)
_COSMOS_POSTTRAINING = (
    "https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/"
    "examples/post-training_tokenizer.md"
)
_VJEPA_REPOSITORY = "https://github.com/facebookresearch/vjepa2"
_VJEPA_BACKBONES = "https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py"
_VJEPA_HF = "https://huggingface.co/docs/transformers/model_doc/vjepa2"
_VJEPA21_MODEL = (
    "https://github.com/facebookresearch/vjepa2/blob/main/"
    "app/vjepa_2_1/models/vision_transformer.py"
)
_DINOV3_REPOSITORY = "https://github.com/facebookresearch/dinov3"
_DINOV3_MODEL = (
    "https://github.com/facebookresearch/dinov3/blob/main/"
    "dinov3/models/vision_transformer.py"
)
_DINOV3_MODEL_CARD = "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m"


def _cosmos_contract(*, discrete: bool, native_training: bool) -> FamilyAdapterContract:
    latent = "dv" if discrete else "cv"
    backend = "native_training_checkpoint" if native_training else "torchscript_encoder"
    key = f"cosmos_tokenize1_{latent}_{'native_training' if native_training else 'jit'}"
    output = (
        "encode -> indices [B,Tz,Hz,Wz] and pre-quantization codes [B,Cz,Tz,Hz,Wz]"
        if discrete
        else "encode -> continuous latent [B,Cz,Tz,Hz,Wz]"
    )
    return FamilyAdapterContract(
        key=key,
        facts=UpstreamAdapterFacts(
            family="cosmos_tokenize1",
            model_scope=f"Cosmos-Tokenize1-{latent.upper()} video tokenizers",
            backend=backend,
            upstream_repository=_COSMOS_REPOSITORY,
            native_input_layout="[B,C,T,H,W] normalized to [-1,1]",
            native_feature_view=output,
            native_temporal_mode="causal_video",
            representation_kind=(
                "discrete_ids_with_continuous_codes" if discrete else "continuous_dense"
            ),
            required_artifact_roles=("config", "model") if native_training else ("encoder",),
            sources=(_COSMOS_INFERENCE, _COSMOS_POSTTRAINING),
        ),
        policy=AdapterStudyPolicy(
            canonical_input_conversion="[B,T,C,H,W]_[0,1]_to_[B,C,T,H,W]_[-1,1]",
            adapter_feature_layout=(
                "codes_[B,Cz,Tz,Hz,Wz]_to_tokens_and_indices_to_discrete_ids"
                if discrete
                else "latent_[B,Cz,Tz,Hz,Wz]_to_[B,Tz*Hz*Wz,Cz]"
            ),
            accepts_video_batches=True,
            required_feature_view=(
                "prequantization_codes_plus_integer_indices"
                if discrete
                else "continuous_encoder_latent"
            ),
            allowed_repair_scopes=(
                ("none", "post_representation_control", "encoder_internal")
                if native_training
                else ("none", "post_representation_control")
            ),
            repair_note=(
                "A revision-specific native module hook must prove that encoder_internal is before "
                "final compression or token assignment."
                if native_training
                else "TorchScript output supports frozen extraction or a post-code control only."
            ),
            loader_policy="verified local artifacts; no network during load; pinned local source",
        ),
    )


FAMILY_ADAPTER_CONTRACTS = {
    contract.key: contract
    for contract in (
        _cosmos_contract(discrete=False, native_training=False),
        _cosmos_contract(discrete=False, native_training=True),
        _cosmos_contract(discrete=True, native_training=False),
        _cosmos_contract(discrete=True, native_training=True),
        FamilyAdapterContract(
            key="vjepa2_native",
            facts=UpstreamAdapterFacts(
                family="vjepa2",
                model_scope="V-JEPA 2 native ViT video encoders",
                backend="pinned_local_pytorch_repository",
                upstream_repository=_VJEPA_REPOSITORY,
                native_input_layout="[B,C,T,H,W]",
                native_feature_view=(
                    "encoder dense tokens [B,N,D]; Hub factory returns encoder,predictor"
                ),
                native_temporal_mode="bidirectional_video",
                representation_kind="continuous_dense",
                required_artifact_roles=("checkpoint",),
                sources=(_VJEPA_BACKBONES,),
            ),
            policy=AdapterStudyPolicy(
                canonical_input_conversion="[B,T,C,H,W]_to_[B,C,T,H,W]_pinned_eval_transform",
                adapter_feature_layout="encoder_dense_tokens_[B,N,D]",
                accepts_video_batches=True,
                required_feature_view="encoder_dense_tokens_predictor_discarded",
                allowed_repair_scopes=(
                    "none",
                    "post_representation_control",
                    "encoder_internal",
                ),
                repair_note="Register a revision-specific hook before named final encoder blocks.",
                loader_policy=(
                    "pretrained=False in a pinned local checkout; verified local checkpoint"
                ),
            ),
        ),
        FamilyAdapterContract(
            key="vjepa2_hf",
            facts=UpstreamAdapterFacts(
                family="vjepa2",
                model_scope="official Meta V-JEPA 2 Transformers checkpoints",
                backend="transformers_safetensors",
                upstream_repository=_VJEPA_REPOSITORY,
                native_input_layout="pixel_values_videos [B,T,C,H,W]",
                native_feature_view="last_hidden_state [B,N,D]; predictor skipped explicitly",
                native_temporal_mode="bidirectional_video",
                representation_kind="continuous_dense",
                required_artifact_roles=("config", "weights", "preprocessor"),
                sources=(_VJEPA_HF,),
            ),
            policy=AdapterStudyPolicy(
                canonical_input_conversion="[B,T,C,H,W]_through_pinned_AutoVideoProcessor",
                adapter_feature_layout="last_hidden_state_[B,N,D]",
                accepts_video_batches=True,
                required_feature_view="last_hidden_state_with_predictor_skipped",
                allowed_repair_scopes=(
                    "none",
                    "post_representation_control",
                    "encoder_internal",
                ),
                repair_note=(
                    "Use a named encoder-layer hook; do not count a returned-token MLP as internal."
                ),
                loader_policy="pinned HF revision, local_files_only, trust_remote_code disabled",
            ),
        ),
        FamilyAdapterContract(
            key="vjepa21_native_video_384",
            facts=UpstreamAdapterFacts(
                family="vjepa21",
                model_scope="V-JEPA 2.1 native 384-resolution video branch",
                backend="pinned_local_pytorch_repository",
                upstream_repository=_VJEPA_REPOSITORY,
                native_input_layout="video [B,C,T,H,W]; separate image [B,C,H,W] branch exists",
                native_feature_view=(
                    "encoder dense tokens [B,N,D]; Hub factory returns encoder,predictor"
                ),
                native_temporal_mode="bidirectional_video",
                representation_kind="continuous_dense",
                required_artifact_roles=("checkpoint",),
                sources=(_VJEPA_BACKBONES, _VJEPA21_MODEL),
            ),
            policy=AdapterStudyPolicy(
                canonical_input_conversion="[B,T,C,H,W]_to_[B,C,T,H,W]_384_crop_video_mode",
                adapter_feature_layout="encoder_dense_tokens_[B,N,D]",
                accepts_video_batches=True,
                required_feature_view="video_encoder_dense_tokens_predictor_discarded",
                allowed_repair_scopes=(
                    "none",
                    "post_representation_control",
                    "encoder_internal",
                ),
                repair_note=(
                    "Register a revision-specific hook before named final video-encoder blocks."
                ),
                loader_policy=(
                    "pretrained=False in a pinned local checkout; verified local checkpoint"
                ),
            ),
        ),
        FamilyAdapterContract(
            key="dinov3_vit_native",
            facts=UpstreamAdapterFacts(
                family="dinov3",
                model_scope="DINOv3 ViT backbones only; ConvNeXt is excluded",
                backend="pinned_local_pytorch_repository",
                upstream_repository=_DINOV3_REPOSITORY,
                native_input_layout="image [B,C,H,W]",
                native_feature_view=(
                    "forward_features x_norm_patchtokens plus x_norm_clstoken and x_storage_tokens"
                ),
                native_temporal_mode="framewise_image",
                representation_kind="continuous_dense",
                required_artifact_roles=("weights",),
                sources=(_DINOV3_MODEL, _DINOV3_MODEL_CARD),
            ),
            policy=AdapterStudyPolicy(
                canonical_input_conversion="framewise_[B*T,C,H,W]_pinned_transform_patch16_aligned",
                adapter_feature_layout="patches_[B*T,Np,D]_to_[B,T*Np,D]",
                accepts_video_batches=True,
                required_feature_view="forward_features_x_norm_patchtokens",
                allowed_repair_scopes=(
                    "none",
                    "post_representation_control",
                    "encoder_internal",
                ),
                repair_note="Register a revision-specific hook before named final ViT blocks.",
                loader_policy=(
                    "source=local in pinned checkout; weights must resolve to a verified local file"
                ),
            ),
        ),
        FamilyAdapterContract(
            key="dinov3_vit_hf",
            facts=UpstreamAdapterFacts(
                family="dinov3",
                model_scope="official Meta DINOv3 ViT Transformers checkpoints",
                backend="transformers_safetensors",
                upstream_repository=_DINOV3_REPOSITORY,
                native_input_layout="image pixel_values [B,C,H,W]",
                native_feature_view="last_hidden_state contains CLS, register tokens, then patches",
                native_temporal_mode="framewise_image",
                representation_kind="continuous_dense",
                required_artifact_roles=("config", "weights", "preprocessor"),
                sources=(_DINOV3_MODEL_CARD,),
            ),
            policy=AdapterStudyPolicy(
                canonical_input_conversion="framewise_[B*T,C,H,W]_pinned_transform_patch16_aligned",
                adapter_feature_layout="slice_cls_and_registers_then_[B*T,Np,D]_to_[B,T*Np,D]",
                accepts_video_batches=True,
                required_feature_view="last_hidden_state_patch_slice_after_cls_and_registers",
                allowed_repair_scopes=(
                    "none",
                    "post_representation_control",
                    "encoder_internal",
                ),
                repair_note="Register a revision-specific hook before named final ViT blocks.",
                loader_policy="pinned HF revision, local_files_only, trust_remote_code disabled",
            ),
        ),
    )
}


def get_family_adapter_contract(contract_key: str) -> FamilyAdapterContract:
    """Return one concrete family/backend contract without loading code or weights."""

    try:
        return FAMILY_ADAPTER_CONTRACTS[contract_key]
    except KeyError as error:
        choices = ", ".join(sorted(FAMILY_ADAPTER_CONTRACTS))
        raise ValueError(
            f"unknown pretrained adapter contract {contract_key!r}; expected one of {choices}"
        ) from error


def _declared_loader_policy_violations(declaration: LoaderDeclaration) -> tuple[str, ...]:
    violations: list[str] = []
    if declaration.artifact_source != "local_files":
        violations.append("artifact_source_is_not_local_files")
    if declaration.network_access != "disabled":
        violations.append("network_access_is_not_declared_disabled")
    if declaration.artifact_hash_check != "before_load":
        violations.append("artifact_hash_not_declared_checked_before_load")
    if declaration.source_revision_check != "full_commit_checked_out":
        violations.append("source_revision_not_declared_checked_out_at_full_commit")
    if declaration.upstream_code_execution in {
        "floating_or_remote_repository",
        "unknown",
    }:
        violations.append("upstream_code_execution_is_floating_or_unknown")
    if declaration.hub_remote_code_trust not in {"disabled", "not_applicable"}:
        violations.append("hub_remote_code_trust_is_not_declared_disabled")
    return tuple(violations)


class DeclaredCallableAdapter:
    """Validate caller-supplied functions without claiming to verify how they loaded a model.

    The wrapper imports no upstream model package and downloads no artifact itself. It cannot,
    however, inspect arbitrary preprocessing/encoding callables. Loader behavior is therefore
    recorded as a caller declaration and separately marked as unobserved by this scaffold.
    """

    def __init__(
        self,
        *,
        provenance: AdapterProvenance,
        capabilities: AdapterCapabilities,
        loader_declaration: LoaderDeclaration,
        preprocess_fn: Callable[[VisualBatch], Any],
        encode_fn: Callable[[Any, bool], RepresentationBatch],
        repair_parameters_fn: Callable[[], tuple[RepairParameterGroup, ...]] | None = None,
        extra_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        provenance.validate()
        capabilities.validate()
        loader_declaration.validate()
        contract = get_family_adapter_contract(provenance.contract_key)
        expected_repository = contract.facts.upstream_repository.rstrip("/")
        if provenance.source_repository.rstrip("/") != expected_repository:
            raise ValueError("provenance source_repository disagrees with the adapter contract")
        if provenance.feature_view != contract.policy.required_feature_view:
            raise ValueError("provenance feature_view disagrees with the adapter contract")
        artifact_roles = {artifact.role for artifact in provenance.artifacts}
        missing_roles = set(contract.facts.required_artifact_roles) - artifact_roles
        if missing_roles:
            missing = ", ".join(sorted(missing_roles))
            raise ValueError(f"provenance is missing required artifact roles: {missing}")
        if capabilities.accepts_video_batches != contract.policy.accepts_video_batches:
            raise ValueError("adapter video-batch capability disagrees with the study policy")
        if capabilities.native_temporal_mode != contract.facts.native_temporal_mode:
            raise ValueError("adapter temporal mode disagrees with upstream facts")
        if capabilities.representation_kind != contract.facts.representation_kind:
            raise ValueError("adapter representation kind disagrees with upstream facts")
        if capabilities.repair_scope not in contract.policy.allowed_repair_scopes:
            raise ValueError("adapter repair scope is not allowed for this backend")
        if capabilities.repair_scope == "none" and repair_parameters_fn is not None:
            raise ValueError("a frozen adapter cannot expose repair parameters")
        if capabilities.repair_scope != "none" and repair_parameters_fn is None:
            raise ValueError("a repairable adapter must expose named repair parameter groups")
        if capabilities.repair_scope == "none" and provenance.intervention_point != "none":
            raise ValueError("a frozen adapter must declare intervention_point='none'")
        if (
            capabilities.repair_scope == "post_representation_control"
            and provenance.intervention_point != "post_representation"
        ):
            raise ValueError(
                "a post-representation control must declare its exact intervention point"
            )
        if capabilities.repair_scope == "encoder_internal" and provenance.intervention_point in {
            "none",
            "post_representation",
        }:
            raise ValueError("encoder-internal repair needs a named pre-output intervention point")
        self._provenance = provenance
        self._capabilities = capabilities
        self._loader_declaration = loader_declaration
        self._preprocess_fn = preprocess_fn
        self._encode_fn = encode_fn
        self._repair_parameters_fn = repair_parameters_fn
        self._extra_manifest = dict(extra_manifest or {})

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def preprocess(self, batch: VisualBatch) -> Any:
        batch.validate()
        if int(batch.frames.shape[1]) > 1 and not self._capabilities.accepts_video_batches:
            raise ValueError("this adapter does not accept multi-frame canonical batches")
        return self._preprocess_fn(batch)

    def encode(
        self,
        batch: VisualBatch,
        *,
        return_intermediates: bool = False,
    ) -> RepresentationBatch:
        if return_intermediates and not self._capabilities.returns_intermediates:
            raise ValueError("this adapter does not declare intermediate-feature support")
        preprocessed = self.preprocess(batch)
        result = self._encode_fn(preprocessed, return_intermediates)
        if not isinstance(result, RepresentationBatch):
            raise TypeError("encode_fn must return RepresentationBatch")
        result.validate()
        if result.provenance != self._provenance:
            raise ValueError("representation provenance does not match its adapter")
        expects_discrete = (
            self._capabilities.representation_kind == "discrete_ids_with_continuous_codes"
        )
        if expects_discrete != (result.discrete_ids is not None):
            raise ValueError("discrete_ids presence disagrees with the representation kind")
        if return_intermediates and not result.intermediates:
            raise ValueError("encode_fn declared intermediate support but returned none")
        return result

    def repair_parameters(self) -> tuple[RepairParameterGroup, ...]:
        if self._repair_parameters_fn is None:
            return ()
        groups = tuple(self._repair_parameters_fn())
        if not groups:
            raise ValueError("repair_parameters_fn returned no parameter groups")
        names: list[str] = []
        for group in groups:
            if not isinstance(group, RepairParameterGroup):
                raise TypeError("repair_parameters_fn must return RepairParameterGroup values")
            group.validate()
            if group.intervention_point != self._provenance.intervention_point:
                raise ValueError("repair parameter group disagrees with the declared intervention")
            names.append(group.name)
        if len(set(names)) != len(names):
            raise ValueError("repair parameter group names must be unique")
        return groups

    def checkpoint_manifest(self) -> Mapping[str, Any]:
        """Return declarations and explicit non-observations; this method performs no I/O."""

        return {
            "schema_version": 2,
            "adapter": "declared_callable_adapter",
            "provenance": dataclasses.asdict(self._provenance),
            "capabilities": dataclasses.asdict(self._capabilities),
            "family_contract": dataclasses.asdict(
                get_family_adapter_contract(self._provenance.contract_key)
            ),
            "loader_declaration": {
                "evidence_origin": "caller_declaration_not_observed_by_adapter",
                **dataclasses.asdict(self._loader_declaration),
                "study_policy_violations": list(
                    _declared_loader_policy_violations(self._loader_declaration)
                ),
            },
            "adapter_observations": {
                "artifact_bytes_read": False,
                "artifact_hashes_verified": False,
                "source_checkout_inspected": False,
                "callable_network_activity_observed": False,
                "callable_code_execution_inspected": False,
            },
            "extra": dict(self._extra_manifest),
        }


# Backward-compatible import name. Its manifest and documentation deliberately avoid claiming
# that this callable scaffold itself pins or verifies loader behavior.
PinnedCallableAdapter = DeclaredCallableAdapter
