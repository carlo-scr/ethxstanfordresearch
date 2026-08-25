"""Pretrained-representation audit and Common-Action Geometry Repair primitives.

The namespace is deliberately independent of the legacy AE/beta-VAE trainer.  Base imports remain
dependency-free; checkpoint-specific packages and PyTorch are loaded only by the execution layer.
"""

from latent_safety.foundation.adapters import (
    FAMILY_ADAPTER_CONTRACTS,
    AdapterStudyPolicy,
    DeclaredCallableAdapter,
    FamilyAdapterContract,
    PinnedCallableAdapter,
    UpstreamAdapterFacts,
    get_family_adapter_contract,
)
from latent_safety.foundation.audit import (
    CommonActionNeighborhoodAudit,
    CommonActionWitness,
    audit_common_action_neighborhoods,
    audit_common_action_records,
)
from latent_safety.foundation.interventions import (
    assert_frozen_parameters_unchanged,
    build_action_conditioned_profile_head,
    build_residual_bottleneck_intervention,
    configure_trainable_parameter_policy,
    snapshot_frozen_parameters,
    validate_gradient_route,
    validate_trainable_parameter_policy,
)
from latent_safety.foundation.protocols import (
    ActionProfileBatch,
    AdapterCapabilities,
    AdapterProvenance,
    ArtifactProvenance,
    BranchableDomainAdapter,
    LoaderDeclaration,
    RepairParameterGroup,
    RepresentationAdapter,
    RepresentationBatch,
    VisualBatch,
)
from latent_safety.foundation.repair import (
    CommonActionRepairConfig,
    ReferenceCommonActionLoss,
    common_action_geometry_reference_loss,
    encoder_only_state_dict,
)

__all__ = [
    "ActionProfileBatch",
    "AdapterCapabilities",
    "AdapterProvenance",
    "AdapterStudyPolicy",
    "ArtifactProvenance",
    "BranchableDomainAdapter",
    "CommonActionNeighborhoodAudit",
    "CommonActionRepairConfig",
    "CommonActionWitness",
    "DeclaredCallableAdapter",
    "FAMILY_ADAPTER_CONTRACTS",
    "FamilyAdapterContract",
    "LoaderDeclaration",
    "PinnedCallableAdapter",
    "ReferenceCommonActionLoss",
    "RepairParameterGroup",
    "RepresentationAdapter",
    "RepresentationBatch",
    "VisualBatch",
    "UpstreamAdapterFacts",
    "assert_frozen_parameters_unchanged",
    "audit_common_action_neighborhoods",
    "audit_common_action_records",
    "build_action_conditioned_profile_head",
    "build_residual_bottleneck_intervention",
    "common_action_geometry_reference_loss",
    "configure_trainable_parameter_policy",
    "encoder_only_state_dict",
    "get_family_adapter_contract",
    "snapshot_frozen_parameters",
    "validate_gradient_route",
    "validate_trainable_parameter_policy",
]
