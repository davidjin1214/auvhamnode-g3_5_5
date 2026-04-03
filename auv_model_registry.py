"""Central registry for AUV dynamics model naming, metadata, and builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from AUVHamNODE import AUVHamNODE
from auv_baselines import BASELINE_MODELS


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a trainable model family."""

    name: str
    display_name: str
    family: str
    group: str
    description: str
    builder_kind: str
    energy_semantics: str = "not_comparable"
    hidden_dim_scale: float = 1.0
    use_mass_init: bool = True
    overrides: Mapping[str, Any] = field(default_factory=dict)
    cli_visible: bool = True


MODEL_SPECS: Dict[str, ModelSpec] = {
    "phnode_full": ModelSpec(
        name="phnode_full",
        display_name="pH NODE Full",
        family="main",
        group="core",
        description="Exact SE(3) + mass + scalar potential + split D/J/B.",
        builder_kind="hamnode",
        energy_semantics="mechanical_energy",
    ),
    "phnode_merged_force": ModelSpec(
        name="phnode_merged_force",
        display_name="pH NODE Merged Force",
        family="baseline",
        group="core",
        description="pH core with a single merged non-conservative force branch.",
        builder_kind="merged_force",
        energy_semantics="mechanical_energy",
    ),
    "phnode_qforce": ModelSpec(
        name="phnode_qforce",
        display_name="pH NODE q-Force",
        family="baseline",
        group="core",
        description="Structured pH model with a generic configuration-dependent force.",
        builder_kind="qforce",
    ),
    "se3_momentum_blackbox": ModelSpec(
        name="se3_momentum_blackbox",
        display_name="SE(3) Momentum Black-Box",
        family="baseline",
        group="core",
        description="Exact SE(3) + constant mass matrix, with black-box momentum dynamics.",
        builder_kind="se3_momentum",
    ),
    "se3_accel_blackbox": ModelSpec(
        name="se3_accel_blackbox",
        display_name="SE(3) Acceleration Black-Box",
        family="baseline",
        group="core",
        description="Exact SE(3) kinematics with black-box acceleration dynamics.",
        builder_kind="se3_accel",
        hidden_dim_scale=1.88,
    ),
    "blackbox_fullstate": ModelSpec(
        name="blackbox_fullstate",
        display_name="Full-State Black-Box",
        family="baseline",
        group="core",
        description="Fully unstructured state-derivative model.",
        builder_kind="fullstate",
        hidden_dim_scale=1.78,
    ),
    "ablate_no_mass_prior": ModelSpec(
        name="ablate_no_mass_prior",
        display_name="Ablation: No Mass Prior",
        family="ablation",
        group="ablation",
        description="AUVHamNODE without physics-based mass initialization.",
        builder_kind="hamnode",
        energy_semantics="mechanical_energy",
        use_mass_init=False,
    ),
    "ablate_diag_damping": ModelSpec(
        name="ablate_diag_damping",
        display_name="Ablation: Diagonal Damping",
        family="ablation",
        group="ablation",
        description="AUVHamNODE with diagonal damping only.",
        builder_kind="hamnode",
        energy_semantics="mechanical_energy",
        overrides={"coupled_damping": False},
    ),
    "ablate_no_lift": ModelSpec(
        name="ablate_no_lift",
        display_name="Ablation: No Lift",
        family="ablation",
        group="ablation",
        description="AUVHamNODE without the learned skew-symmetric lift term.",
        builder_kind="hamnode",
        energy_semantics="mechanical_energy",
        overrides={"learn_lift": False},
    ),
    "ablate_bu_only": ModelSpec(
        name="ablate_bu_only",
        display_name="Ablation: B(u) Only",
        family="ablation",
        group="ablation",
        description="AUVHamNODE with actuation conditioned only on actuator states.",
        builder_kind="hamnode",
        energy_semantics="mechanical_energy",
        overrides={
            "actuation_condition_on_velocity": False,
            "actuation_current_feature": "none",
        },
    ),
}


CANONICAL_MODEL_TYPES = tuple(MODEL_SPECS.keys())
MODEL_TYPE_CHOICES = tuple(
    name for name, spec in MODEL_SPECS.items() if spec.cli_visible
)


def canonicalize_model_type(model_type: str) -> str:
    """Validate that the provided model type is already canonical."""
    key = str(model_type).strip()
    if key not in MODEL_SPECS:
        valid = ", ".join(MODEL_TYPE_CHOICES)
        raise ValueError(
            f"Unknown model_type {model_type!r}. Canonical choices: {valid}"
        )
    return key


def get_model_spec(model_type: str) -> ModelSpec:
    """Return the registry entry for a canonical model name."""
    return MODEL_SPECS[canonicalize_model_type(model_type)]


def format_model_type_help() -> str:
    """Readable CLI help string for visible model types."""
    group_titles = {
        "core": "Core models",
        "ablation": "Ablations",
        "legacy": "Legacy",
    }
    lines = []
    current_group = None
    for name in MODEL_TYPE_CHOICES:
        spec = MODEL_SPECS[name]
        if spec.group != current_group:
            current_group = spec.group
            lines.append(f"{group_titles.get(spec.group, spec.group.title())}:")
        lines.append(f"  {name:<22} -- {spec.description}")
    return "\n".join(lines)


def _attach_metadata(model, spec: ModelSpec):
    model.model_type = spec.name
    model.model_display_name = spec.display_name
    model.model_family = spec.family
    model.model_group = spec.group
    model.energy_semantics = spec.energy_semantics
    return model


def instantiate_model(model_type, device, hidden_dim=128, M_init=None, **kwargs):
    """Instantiate a model from the registry."""
    spec = get_model_spec(model_type)
    effective_hidden_dim = int(hidden_dim * spec.hidden_dim_scale)

    if spec.builder_kind == "hamnode":
        init_kwargs = {
            "device": device,
            "hidden_dim": effective_hidden_dim,
            "coupled_damping": kwargs.get("coupled_damping", True),
            "include_depth_in_potential": kwargs.get("include_depth_in_potential", False),
            "M_init": M_init if spec.use_mass_init else None,
            "ocean_current": kwargs.get("ocean_current", False),
            "learn_lift": True,
            "actuation_condition_on_velocity": True,
            "actuation_current_feature": kwargs.get("actuation_current_feature", "current_body"),
            "dj_current_feature": kwargs.get("dj_current_feature", "none"),
            "T_actuator_init": kwargs.get("t_actuator_init"),
            "u_act_scale": kwargs.get("u_act_scale"),
            "u_dim": kwargs.get("u_dim", 3),
            "absolute_depth_context": kwargs.get("absolute_depth_context", False),
        }
        init_kwargs.update(spec.overrides)
        return _attach_metadata(AUVHamNODE(**init_kwargs).to(device), spec)

    if spec.builder_kind == "merged_force":
        model = BASELINE_MODELS[spec.name](
            device=device,
            hidden_dim=effective_hidden_dim,
            M_init=M_init,
            include_depth_in_potential=kwargs.get("include_depth_in_potential", False),
            ocean_current=kwargs.get("ocean_current", False),
            T_actuator_init=kwargs.get("t_actuator_init"),
            u_act_scale=kwargs.get("u_act_scale"),
            u_dim=kwargs.get("u_dim", 3),
            absolute_depth_context=kwargs.get("absolute_depth_context", False),
        ).to(device)
        return _attach_metadata(model, spec)

    if spec.builder_kind == "qforce":
        model = BASELINE_MODELS[spec.name](
            device=device,
            hidden_dim=effective_hidden_dim,
            M_init=M_init,
            include_depth=kwargs.get("include_depth_in_potential", False),
            ocean_current=kwargs.get("ocean_current", False),
            coupled_damping=kwargs.get("coupled_damping", True),
            learn_lift=True,
            actuation_condition_on_velocity=True,
            actuation_current_feature=kwargs.get("actuation_current_feature", "current_body"),
            dj_current_feature=kwargs.get("dj_current_feature", "none"),
            T_actuator_init=kwargs.get("t_actuator_init"),
            u_act_scale=kwargs.get("u_act_scale"),
            u_dim=kwargs.get("u_dim", 3),
            absolute_depth_context=kwargs.get("absolute_depth_context", False),
        ).to(device)
        return _attach_metadata(model, spec)

    if spec.builder_kind == "se3_momentum":
        model = BASELINE_MODELS[spec.name](
            device=device,
            hidden_dim=effective_hidden_dim,
            M_init=M_init,
            include_depth=kwargs.get("include_depth_in_potential", False),
            ocean_current=kwargs.get("ocean_current", False),
            T_actuator_init=kwargs.get("t_actuator_init"),
            u_act_scale=kwargs.get("u_act_scale"),
            u_dim=kwargs.get("u_dim", 3),
            absolute_depth_context=kwargs.get("absolute_depth_context", False),
        ).to(device)
        return _attach_metadata(model, spec)

    if spec.builder_kind == "se3_accel":
        model = BASELINE_MODELS[spec.name](
            device=device,
            hidden_dim=effective_hidden_dim,
            include_depth=kwargs.get("include_depth_in_potential", False),
            ocean_current=kwargs.get("ocean_current", False),
            T_actuator_init=kwargs.get("t_actuator_init"),
            u_act_scale=kwargs.get("u_act_scale"),
            u_dim=kwargs.get("u_dim", 3),
            absolute_depth_context=kwargs.get("absolute_depth_context", False),
        ).to(device)
        return _attach_metadata(model, spec)

    if spec.builder_kind == "fullstate":
        model = BASELINE_MODELS[spec.name](
            device=device,
            hidden_dim=effective_hidden_dim,
            ocean_current=kwargs.get("ocean_current", False),
            T_actuator_init=kwargs.get("t_actuator_init"),
            u_act_scale=kwargs.get("u_act_scale"),
            u_dim=kwargs.get("u_dim", 3),
            absolute_depth_context=kwargs.get("absolute_depth_context", False),
        ).to(device)
        return _attach_metadata(model, spec)

    raise ValueError(f"Unsupported builder kind {spec.builder_kind!r} for {spec.name!r}.")
