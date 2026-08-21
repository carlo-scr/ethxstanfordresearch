"""Lazily constructed world models for AE, beta-VAE, and history-encoder arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from latent_safety.learning.config import DataConfig, LearningConfig, ModelConfig, parse_config
from latent_safety.learning.runtime import TorchModules


def model_metadata(
    model: Any, model_config: ModelConfig, data_config: DataConfig
) -> dict[str, Any]:
    """Return architecture metadata without serializing implementation objects."""

    return {
        "family": model_config.family,
        "history_mode": f"{model_config.history_encoder}_h{data_config.history_length}",
        "history_encoder": model_config.history_encoder,
        "stack_summary": (
            "last_frame_and_last_minus_first_feature"
            if model_config.history_encoder == "stack"
            else None
        ),
        "history_length": data_config.history_length,
        "latent_dim": model_config.latent_dim,
        "hidden_dim": model_config.hidden_dim,
        "transition_hidden_dim": model_config.transition_hidden_dim,
        "channels": data_config.channels,
        "image_size": data_config.image_size,
        "action_dim": 1,
        "action_profile_dim": len(data_config.actions),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "stochastic_training_latent": model_config.family == "beta_vae",
        "evaluation_latent": "posterior_mean",
    }


def build_world_model(
    modules: TorchModules,
    model_config: ModelConfig,
    data_config: DataConfig,
) -> Any:
    """Build the requested model after PyTorch has been explicitly loaded."""

    torch = modules.torch
    nn = modules.nn
    functional = modules.functional

    class FrameEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convolutions = nn.Sequential(
                nn.Conv2d(data_config.channels, 32, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 96, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(96 * 2 * 2, model_config.hidden_dim),
                nn.LayerNorm(model_config.hidden_dim),
                nn.SiLU(),
            )

        def forward(self, frame: Any) -> Any:
            return self.projection(self.convolutions(frame))

    class ObservationDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(model_config.latent_dim, 96 * 4 * 4),
                nn.SiLU(),
            )
            self.deconvolutions = nn.Sequential(
                nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(
                    32, data_config.channels, kernel_size=4, stride=2, padding=1
                ),
                nn.Sigmoid(),
            )

        def forward(self, latent: Any) -> Any:
            hidden = self.projection(latent).view(latent.shape[0], 96, 4, 4)
            image = self.deconvolutions(hidden)
            if image.shape[-2:] != (data_config.image_size, data_config.image_size):
                image = functional.interpolate(
                    image,
                    size=(data_config.image_size, data_config.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return image

    class LatentWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frame_encoder = FrameEncoder()
            if model_config.history_encoder == "gru":
                self.history_encoder = nn.GRU(
                    input_size=model_config.hidden_dim,
                    hidden_size=model_config.hidden_dim,
                    batch_first=True,
                )
                posterior_input_dim = model_config.hidden_dim
            else:
                self.history_encoder = nn.Sequential(
                    nn.Linear(
                        2 * model_config.hidden_dim,
                        model_config.hidden_dim,
                    ),
                    nn.LayerNorm(model_config.hidden_dim),
                    nn.SiLU(),
                )
                posterior_input_dim = model_config.hidden_dim
            self.posterior_mean = nn.Linear(posterior_input_dim, model_config.latent_dim)
            self.posterior_logvar = nn.Linear(posterior_input_dim, model_config.latent_dim)
            self.decoder = ObservationDecoder()
            transition_hidden = model_config.transition_hidden_dim
            self.transition = nn.Sequential(
                nn.Linear(model_config.latent_dim + 1, transition_hidden),
                nn.LayerNorm(transition_hidden),
                nn.SiLU(),
                nn.Linear(transition_hidden, transition_hidden),
                nn.SiLU(),
                nn.Linear(transition_hidden, model_config.latent_dim),
            )
            safety_hidden = max(8, model_config.hidden_dim // 2)
            self.margin_head = nn.Sequential(
                nn.Linear(model_config.latent_dim, safety_hidden),
                nn.SiLU(),
                nn.Linear(safety_hidden, 1),
            )
            self.action_profile_head = nn.Sequential(
                nn.Linear(model_config.latent_dim, safety_hidden),
                nn.SiLU(),
                nn.Linear(safety_hidden, len(data_config.actions)),
            )

        def encode(self, history: Any, *, sample: bool | None = None) -> dict[str, Any]:
            if history.ndim != 5:
                raise ValueError("history must have shape [batch, history, channel, height, width]")
            batch_size, history_length = history.shape[:2]
            if history_length != data_config.history_length:
                raise ValueError(
                    f"expected history length {data_config.history_length}, got {history_length}"
                )
            flat_frames = history.reshape(
                batch_size * history_length,
                data_config.channels,
                data_config.image_size,
                data_config.image_size,
            )
            features = self.frame_encoder(flat_frames).view(
                batch_size, history_length, model_config.hidden_dim
            )
            if model_config.history_encoder == "gru":
                encoded_sequence, _ = self.history_encoder(features)
                context = encoded_sequence[:, -1]
            else:
                # Fixed width removes the parameter-count confound between h1 and h4.  The first
                # block carries the current visual feature; the second exposes finite-difference
                # motion information when a history is available and is exactly zero for h1.
                endpoint_summary = torch.cat(
                    (features[:, -1], features[:, -1] - features[:, 0]), dim=-1
                )
                context = self.history_encoder(endpoint_summary)
            mean = self.posterior_mean(context)
            if model_config.family == "beta_vae":
                logvar = self.posterior_logvar(context).clamp(
                    model_config.posterior_logvar_min,
                    model_config.posterior_logvar_max,
                )
                should_sample = self.training if sample is None else sample
                if should_sample:
                    latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                else:
                    latent = mean
            else:
                logvar = torch.zeros_like(mean)
                latent = mean
            return {"z": latent, "mean": mean, "logvar": logvar}

        def predict_next(self, latent: Any, action: Any) -> Any:
            if action.ndim == 1:
                action = action.unsqueeze(-1)
            return self.transition(torch.cat((latent, action), dim=-1))

        def forward(self, history: Any, action: Any) -> dict[str, Any]:
            encoded = self.encode(history)
            latent = encoded["z"]
            return {
                **encoded,
                "is_variational": model_config.family == "beta_vae",
                "reconstruction": self.decoder(latent),
                "predicted_next_z": self.predict_next(latent, action),
                "predicted_margin": self.margin_head(latent).squeeze(-1),
                "predicted_action_profile": self.action_profile_head(latent),
            }

    return LatentWorldModel()


def load_world_model_checkpoint(
    modules: TorchModules,
    checkpoint_path: Path,
    *,
    device: Any,
) -> tuple[Any, LearningConfig, dict[str, Any]]:
    """Restore a trusted pipeline checkpoint and its exact resolved configuration."""

    checkpoint = modules.torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"invalid learning checkpoint: {checkpoint_path}")
    config = parse_config(checkpoint["config"])
    model = build_world_model(modules, config.model, config.data).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint
