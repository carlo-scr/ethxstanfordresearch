from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_safety.foundation.adapters import (  # noqa: E402
    FAMILY_ADAPTER_CONTRACTS,
    DeclaredCallableAdapter,
    get_family_adapter_contract,
)
from latent_safety.foundation.protocols import (  # noqa: E402
    ActionProfileBatch,
    AdapterCapabilities,
    AdapterProvenance,
    ArtifactProvenance,
    LoaderDeclaration,
    RepairParameterGroup,
    RepresentationBatch,
    VisualBatch,
)


class Shaped:
    def __init__(self, *shape: int, dtype: str | None = None) -> None:
        self.shape = shape
        self.dtype = dtype


class Inspectable(Shaped):
    def __init__(self, data: object, *shape: int, dtype: str) -> None:
        super().__init__(*shape, dtype=dtype)
        self._data = data

    def tolist(self) -> object:
        return self._data


def _dino_provenance(*, intervention_point: str = "none") -> AdapterProvenance:
    return AdapterProvenance(
        contract_key="dinov3_vit_native",
        checkpoint_id="dinov3-vits16-local",
        artifacts=(
            ArtifactProvenance(
                role="weights",
                artifact_id="dinov3_vits16.pth",
                sha256="a" * 64,
                size_bytes=123,
            ),
        ),
        source_repository="https://github.com/facebookresearch/dinov3",
        source_revision="c" * 40,
        preprocessing_id="lvd-eval-patch16-v1",
        preprocessing_sha256="d" * 64,
        feature_view="forward_features_x_norm_patchtokens",
        pooling_id="masked-spatiotemporal-mean-v1",
        pooling_sha256="e" * 64,
        normalization_id="calibration-v1",
        normalization_sha256="f" * 64,
        intervention_point=intervention_point,
        license_id="dinov3-license",
    )


def _local_loader_declaration() -> LoaderDeclaration:
    return LoaderDeclaration(
        artifact_source="local_files",
        network_access="disabled",
        artifact_hash_check="before_load",
        source_revision_check="full_commit_checked_out",
        upstream_code_execution="pinned_local_repository",
        hub_remote_code_trust="not_applicable",
    )


class FoundationProtocolTests(unittest.TestCase):
    def test_canonical_visual_and_representation_shapes(self) -> None:
        visual = VisualBatch(
            frames=Shaped(2, 4, 3, 224, 224),
            valid_frames=Shaped(2, 4),
            proprioception=Shaped(2, 4, 7),
        )
        visual.validate()
        provenance = _dino_provenance()
        representation = RepresentationBatch(
            tokens=Shaped(2, 16, 384),
            pooled=Shaped(2, 384),
            token_mask=Shaped(2, 16),
            grid=(4, 2, 2),
            provenance=provenance,
            class_tokens=Shaped(2, 384),
            register_tokens=Shaped(2, 4, 384),
        )
        representation.validate()

    def test_shape_contract_rejects_hidden_information_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid_frames"):
            VisualBatch(
                frames=Shaped(2, 4, 3, 224, 224),
                valid_frames=Shaped(2, 3),
            ).validate()

    def test_strict_visual_validation_checks_dtype_range_and_prefix_masks(self) -> None:
        valid = VisualBatch(
            frames=Inspectable(
                [[[[[0.0]], [[0.5]], [[1.0]]], [[[0.1]], [[0.2]], [[0.3]]]]],
                1,
                2,
                3,
                1,
                1,
                dtype="torch.float32",
            ),
            valid_frames=Inspectable([[True, False]], 1, 2, dtype="torch.bool"),
        )
        valid.validate(inspect_values=True)
        with self.assertRaisesRegex(ValueError, "frames must use a float dtype"):
            VisualBatch(
                frames=Shaped(1, 1, 3, 2, 2, dtype="torch.uint8"),
                valid_frames=Shaped(1, 1, dtype="torch.bool"),
            ).validate()
        with self.assertRaisesRegex(ValueError, "frames values must lie"):
            VisualBatch(
                frames=Inspectable(
                    [[[[[1.1]], [[0.0]], [[0.0]]]]],
                    1,
                    1,
                    3,
                    1,
                    1,
                    dtype="torch.float32",
                ),
                valid_frames=Inspectable([[True]], 1, 1, dtype="torch.bool"),
            ).validate(inspect_values=True)
        with self.assertRaisesRegex(ValueError, "true-prefix"):
            VisualBatch(
                frames=Inspectable(
                    [
                        [
                            [[[0.0]], [[0.0]], [[0.0]]],
                            [[[0.0]], [[0.0]], [[0.0]]],
                            [[[0.0]], [[0.0]], [[0.0]]],
                        ]
                    ],
                    1,
                    3,
                    3,
                    1,
                    1,
                    dtype="torch.float32",
                ),
                valid_frames=Inspectable(
                    [[True, False, True]],
                    1,
                    3,
                    dtype="torch.bool",
                ),
            ).validate(inspect_values=True)
        with self.assertRaisesRegex(ValueError, "grid product"):
            RepresentationBatch(
                tokens=Shaped(2, 16, 32),
                pooled=Shaped(2, 32),
                token_mask=Shaped(2, 16),
                grid=(1, 3, 5),
                provenance=_dino_provenance(),
            ).validate()

    def test_discrete_ids_are_typed_and_match_dense_codes(self) -> None:
        provenance = AdapterProvenance(
            contract_key="cosmos_tokenize1_dv_jit",
            checkpoint_id="Cosmos-Tokenize1-DV4x8x8-360p",
            artifacts=(
                ArtifactProvenance(
                    role="encoder",
                    artifact_id="encoder.jit",
                    sha256="1" * 64,
                ),
            ),
            source_repository="https://github.com/nvidia-cosmos/cosmos-predict1",
            source_revision="2" * 40,
            preprocessing_id="cosmos-minus-one-one-v1",
            preprocessing_sha256="3" * 64,
            feature_view="prequantization_codes_plus_integer_indices",
            pooling_id="masked-latent-mean-v1",
            pooling_sha256="4" * 64,
            normalization_id="calibration-v1",
            normalization_sha256="5" * 64,
            intervention_point="none",
            license_id="nvidia-open-model-license",
        )
        representation = RepresentationBatch(
            tokens=Shaped(1, 12, 6),
            pooled=Shaped(1, 6),
            token_mask=Shaped(1, 12),
            grid=(3, 2, 2),
            provenance=provenance,
            discrete_ids=Shaped(1, 12, dtype="torch.int64"),
        )
        representation.validate()
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            RepresentationBatch(
                tokens=Shaped(1, 12, 6),
                pooled=Shaped(1, 6),
                token_mask=Shaped(1, 12),
                grid=(3, 2, 2),
                provenance=provenance,
                discrete_ids=Shaped(1, 12, dtype="torch.float32"),
            ).validate()

    def test_action_profile_contract_keeps_raw_constraint_channels(self) -> None:
        profiles = ActionProfileBatch(
            joint_margins=Shaped(3, 5),
            normalized_constraint_margins=Shaped(3, 5, 2),
            skill_ids=("left", "right", "forward", "back", "stop"),
            constraint_names=("clearance", "stopping"),
            constraint_scales=(0.5, 1.0),
            profile_source="saved_state_simulator_branching",
        )
        profiles.validate()

    def test_action_profile_checks_joint_minimum_when_values_are_inspectable(self) -> None:
        profiles = ActionProfileBatch(
            joint_margins=Inspectable([[0.2, -0.1]], 1, 2, dtype="torch.float32"),
            normalized_constraint_margins=Inspectable(
                [[[0.2, 0.5], [-0.1, 0.4]]],
                1,
                2,
                2,
                dtype="torch.float32",
            ),
            skill_ids=("left", "right"),
            constraint_names=("clearance", "stopping"),
            constraint_scales=(0.5, 1.0),
            profile_source="saved_state_simulator_branching",
        )
        profiles.validate()
        with self.assertRaisesRegex(ValueError, "minimum normalized constraint"):
            ActionProfileBatch(
                joint_margins=Inspectable([[0.3]], 1, 1, dtype="torch.float32"),
                normalized_constraint_margins=Inspectable(
                    [[[0.2, 0.5]]],
                    1,
                    1,
                    2,
                    dtype="torch.float32",
                ),
                skill_ids=("left",),
                constraint_names=("clearance", "stopping"),
                constraint_scales=(0.5, 1.0),
                profile_source="saved_state_simulator_branching",
            ).validate()

    def test_registry_uses_concrete_family_backend_variants(self) -> None:
        self.assertEqual(
            set(FAMILY_ADAPTER_CONTRACTS),
            {
                "cosmos_tokenize1_cv_jit",
                "cosmos_tokenize1_cv_native_training",
                "cosmos_tokenize1_dv_jit",
                "cosmos_tokenize1_dv_native_training",
                "vjepa2_native",
                "vjepa2_hf",
                "vjepa21_native_video_384",
                "dinov3_vit_native",
                "dinov3_vit_hf",
            },
        )
        cosmos_jit = get_family_adapter_contract("cosmos_tokenize1_cv_jit")
        self.assertNotIn("encoder_internal", cosmos_jit.policy.allowed_repair_scopes)
        cosmos_native = get_family_adapter_contract("cosmos_tokenize1_cv_native_training")
        self.assertIn("encoder_internal", cosmos_native.policy.allowed_repair_scopes)
        dino = get_family_adapter_contract("dinov3_vit_native")
        self.assertTrue(dino.policy.accepts_video_batches)
        self.assertEqual(dino.facts.native_temporal_mode, "framewise_image")
        self.assertIn("forward_features", dino.policy.required_feature_view)
        with self.assertRaisesRegex(ValueError, "unknown pretrained adapter contract"):
            get_family_adapter_contract("floating-main-checkpoint")

    def test_callable_manifest_separates_declarations_from_observations(self) -> None:
        provenance = _dino_provenance(intervention_point="encoder.blocks.10.input")
        visual = VisualBatch(
            frames=Shaped(2, 4, 3, 224, 224),
            valid_frames=Shaped(2, 4),
        )
        calls: list[object] = []

        def preprocess(batch: VisualBatch) -> object:
            calls.append(batch)
            return object()

        def encode(preprocessed: object, return_intermediates: bool) -> RepresentationBatch:
            calls.extend((preprocessed, return_intermediates))
            return RepresentationBatch(
                tokens=Shaped(2, 16, 384),
                pooled=Shaped(2, 384),
                token_mask=Shaped(2, 16),
                grid=(4, 2, 2),
                provenance=provenance,
                intermediates={"encoder.blocks.10.input": Shaped(2, 16, 384)},
            )

        parameter_group = RepairParameterGroup(
            name="last-block-adapter",
            intervention_point="encoder.blocks.10.input",
            parameters=("adapter-weight",),
        )
        adapter = DeclaredCallableAdapter(
            provenance=provenance,
            capabilities=AdapterCapabilities(
                accepts_video_batches=True,
                native_temporal_mode="framewise_image",
                representation_kind="continuous_dense",
                repair_scope="encoder_internal",
                returns_intermediates=True,
            ),
            loader_declaration=_local_loader_declaration(),
            preprocess_fn=preprocess,
            encode_fn=encode,
            repair_parameters_fn=lambda: (parameter_group,),
        )
        result = adapter.encode(visual, return_intermediates=True)
        self.assertEqual(result.provenance, provenance)
        self.assertEqual(adapter.repair_parameters(), (parameter_group,))
        self.assertEqual(len(calls), 3)
        manifest = adapter.checkpoint_manifest()
        self.assertEqual(
            manifest["loader_declaration"]["evidence_origin"],
            "caller_declaration_not_observed_by_adapter",
        )
        self.assertEqual(manifest["loader_declaration"]["study_policy_violations"], [])
        self.assertFalse(manifest["adapter_observations"]["artifact_hashes_verified"])
        self.assertFalse(manifest["adapter_observations"]["callable_network_activity_observed"])
        self.assertNotIn("auto_download", manifest)
        self.assertNotIn("remote_code_execution", manifest)

    def test_discrete_adapter_rejects_missing_discrete_ids(self) -> None:
        contract = get_family_adapter_contract("cosmos_tokenize1_dv_jit")
        provenance = AdapterProvenance(
            contract_key=contract.key,
            checkpoint_id="Cosmos-Tokenize1-DV4x8x8-360p",
            artifacts=(
                ArtifactProvenance("encoder", "encoder.jit", "6" * 64),
            ),
            source_repository=contract.facts.upstream_repository,
            source_revision="7" * 40,
            preprocessing_id="cosmos-minus-one-one-v1",
            preprocessing_sha256="8" * 64,
            feature_view=contract.policy.required_feature_view,
            pooling_id="masked-latent-mean-v1",
            pooling_sha256="9" * 64,
            normalization_id="calibration-v1",
            normalization_sha256="a" * 64,
            intervention_point="none",
            license_id="nvidia-open-model-license",
        )

        def encode(_preprocessed: object, _return_intermediates: bool) -> RepresentationBatch:
            return RepresentationBatch(
                tokens=Shaped(1, 4, 6),
                pooled=Shaped(1, 6),
                token_mask=Shaped(1, 4),
                grid=(1, 2, 2),
                provenance=provenance,
            )

        adapter = DeclaredCallableAdapter(
            provenance=provenance,
            capabilities=AdapterCapabilities(
                accepts_video_batches=True,
                native_temporal_mode="causal_video",
                representation_kind="discrete_ids_with_continuous_codes",
                repair_scope="none",
                returns_intermediates=False,
            ),
            loader_declaration=_local_loader_declaration(),
            preprocess_fn=lambda batch: batch.frames,
            encode_fn=encode,
        )
        with self.assertRaisesRegex(ValueError, "discrete_ids presence"):
            adapter.encode(
                VisualBatch(frames=Shaped(1, 1, 3, 64, 64), valid_frames=Shaped(1, 1))
            )

    def test_jit_contract_refuses_encoder_internal_repair(self) -> None:
        contract = get_family_adapter_contract("cosmos_tokenize1_cv_jit")
        provenance = AdapterProvenance(
            contract_key=contract.key,
            checkpoint_id="Cosmos-Tokenize1-CV4x8x8-360p",
            artifacts=(ArtifactProvenance("encoder", "encoder.jit", "b" * 64),),
            source_repository=contract.facts.upstream_repository,
            source_revision="c" * 40,
            preprocessing_id="cosmos-minus-one-one-v1",
            preprocessing_sha256="d" * 64,
            feature_view=contract.policy.required_feature_view,
            pooling_id="masked-latent-mean-v1",
            pooling_sha256="e" * 64,
            normalization_id="calibration-v1",
            normalization_sha256="f" * 64,
            intervention_point="encoder.pre_quantizer",
            license_id="nvidia-open-model-license",
        )
        with self.assertRaisesRegex(ValueError, "repair scope"):
            DeclaredCallableAdapter(
                provenance=provenance,
                capabilities=AdapterCapabilities(
                    accepts_video_batches=True,
                    native_temporal_mode="causal_video",
                    representation_kind="continuous_dense",
                    repair_scope="encoder_internal",
                    returns_intermediates=True,
                ),
                loader_declaration=_local_loader_declaration(),
                preprocess_fn=lambda batch: batch.frames,
                encode_fn=lambda _batch, _flag: object(),  # type: ignore[arg-type]
                repair_parameters_fn=lambda: (
                    RepairParameterGroup(
                        "invalid",
                        "encoder.pre_quantizer",
                        ("weight",),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
