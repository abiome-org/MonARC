"""Frozen DINOv2-B/14 RGB backbone.

Default construction is a frozen patch-14 768-d stub so CPU tests and offline
CLI paths never call ``torch.hub`` or Hugging Face. Official ViT-B/14 weights
load only when ``mode="vitb14"`` (or ``mode="auto"`` on CUDA with a cache or
``allow_download=True``).

Load order for real weights:

1. Local ``weights_path`` (``.pth`` state dict or a Hugging Face model dir).
2. Torch Hub ``facebookresearch/dinov2`` entry ``dinov2_vitb14``.
3. Hugging Face ``facebook/dinov2-base`` when ``transformers`` is installed.

DINOv2 stays frozen. RGB is 3-channel only.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_B_EMBED = 768
DINOV2_B_PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HUB_REPO = "facebookresearch/dinov2"
HUB_ENTRY = "dinov2_vitb14"
HF_MODEL_ID = "facebook/dinov2-base"
HUB_WEIGHT_NAME = "dinov2_vitb14_pretrain.pth"

BACKBONE_MODES = frozenset({"stub", "vitb14", "auto"})


def freeze_module(module: nn.Module) -> nn.Module:
    """Disable gradients and set eval mode on every parameter."""
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()
    return module


def hub_checkpoint_path() -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / HUB_WEIGHT_NAME


def hub_local_repo() -> Path | None:
    hub_dir = Path(torch.hub.get_dir())
    if not hub_dir.is_dir():
        return None
    matches = sorted(p for p in hub_dir.glob("facebookresearch_dinov2*") if p.is_dir())
    return matches[-1] if matches else None


def official_weights_cached() -> bool:
    if hub_checkpoint_path().is_file():
        return True
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_cache = hf_home / "hub"
    if hub_cache.is_dir():
        for path in hub_cache.glob("models--facebook--dinov2-base/**"):
            if path.is_file() and path.suffix in {".bin", ".safetensors"}:
                return True
    return False


def resolve_backbone_mode(
    mode: str = "auto",
    *,
    device: str | torch.device | None = None,
    weights_path: str | Path | None = None,
    allow_download: bool = False,
) -> str:
    """Map ``auto`` to stub or vitb14 without touching the network."""
    if mode not in BACKBONE_MODES:
        raise ValueError(f"unsupported backbone mode: {mode}")
    if mode == "stub":
        return "stub"
    if mode == "vitb14":
        return "vitb14"
    if weights_path is not None:
        return "vitb14"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return "stub"
    if allow_download or official_weights_cached():
        return "vitb14"
    return "stub"


def tokens_to_nchw(
    patch_tokens: torch.Tensor,
    height: int,
    width: int,
    patch_size: int = DINOV2_B_PATCH,
    embed_dim: int = DINOV2_B_EMBED,
) -> torch.Tensor:
    """Reshape ``[B, N, D]`` patch tokens to ``[B, D, H/P, W/P]``."""
    if patch_tokens.ndim != 3:
        raise ValueError(f"patch tokens must be [B, N, D], got {tuple(patch_tokens.shape)}")
    batch, count, dim = patch_tokens.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    expected = grid_h * grid_w
    if dim != embed_dim:
        raise ValueError(f"expected embed dim {embed_dim}, got {dim}")
    if count != expected:
        raise ValueError(
            f"patch token count {count} != {expected} for spatial {(height, width)} "
            f"and patch {patch_size}"
        )
    return patch_tokens.reshape(batch, grid_h, grid_w, embed_dim).permute(0, 3, 1, 2).contiguous()


def _hub_load(source_repo: str, source: str, pretrained: bool) -> nn.Module:
    kwargs = {
        "source": source,
        "pretrained": pretrained,
        "trust_repo": True,
        "verbose": False,
    }
    try:
        return torch.hub.load(source_repo, HUB_ENTRY, **kwargs)
    except TypeError:
        kwargs.pop("trust_repo", None)
        kwargs.pop("verbose", None)
        return torch.hub.load(source_repo, HUB_ENTRY, **kwargs)


def load_official_vitb14(
    *,
    weights_path: Path | None = None,
    allow_download: bool = False,
) -> tuple[nn.Module, str]:
    """Return ``(module, source_tag)`` for frozen DINOv2-B/14.

    Never contacts the network unless ``allow_download`` is true.
    """
    if weights_path is not None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")
        if weights_path.is_dir():
            model = _load_hf(weights_path, allow_download=False)
            return model, f"hf-dir:{weights_path}"
        inner, tag = _load_architecture(allow_download=allow_download, pretrained=False)
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        inner.load_state_dict(payload, strict=False)
        return inner, f"local:{weights_path.name}"

    if not allow_download and not official_weights_cached() and hub_local_repo() is None:
        raise FileNotFoundError(
            "DINOv2-B/14 weights are not cached. Pass weights_path, "
            "allow_download=True (torch hub "
            f"{HUB_REPO}:{HUB_ENTRY} or Hugging Face {HF_MODEL_ID}), "
            "or use mode='stub'."
        )

    local_repo = hub_local_repo()
    ckpt = hub_checkpoint_path()
    if local_repo is not None and (ckpt.is_file() or allow_download):
        inner = _hub_load(str(local_repo), source="local", pretrained=True)
        return inner, f"torch-hub-local:{HUB_ENTRY}"
    if allow_download:
        inner = _hub_load(HUB_REPO, source="github", pretrained=True)
        return inner, f"torch-hub:{HUB_ENTRY}"
    inner = _load_hf(HF_MODEL_ID, allow_download=False)
    return inner, f"hf:{HF_MODEL_ID}"


def _load_architecture(*, allow_download: bool, pretrained: bool) -> tuple[nn.Module, str]:
    local_repo = hub_local_repo()
    if local_repo is not None:
        inner = _hub_load(str(local_repo), source="local", pretrained=pretrained)
        return inner, f"torch-hub-local:{HUB_ENTRY}"
    if allow_download:
        inner = _hub_load(HUB_REPO, source="github", pretrained=pretrained)
        return inner, f"torch-hub:{HUB_ENTRY}"
    try:
        inner = _load_hf(HF_MODEL_ID, allow_download=False)
        return inner, f"hf:{HF_MODEL_ID}"
    except Exception as exc:
        raise FileNotFoundError(
            "DINOv2-B/14 architecture is not cached. allow_download=True to fetch "
            f"{HUB_REPO}:{HUB_ENTRY}."
        ) from exc


def _load_hf(name: str | Path, *, allow_download: bool) -> nn.Module:
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise FileNotFoundError(
            "transformers is not installed; cannot load Hugging Face DINOv2"
        ) from exc
    local_only = not allow_download
    return AutoModel.from_pretrained(str(name), local_files_only=local_only)


class OfficialDinov2Grid(nn.Module):
    """Wrap a hub/HF DINOv2-B so forward returns ``[B, 768, H/14, W/14]``."""

    def __init__(self, inner: nn.Module, embed_dim: int = DINOV2_B_EMBED, patch_size: int = DINOV2_B_PATCH) -> None:
        super().__init__()
        self.inner = inner
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        height, width = normalized_rgb.shape[-2], normalized_rgb.shape[-1]
        inner = self.inner
        if hasattr(inner, "forward_features"):
            out = inner.forward_features(normalized_rgb)
            if isinstance(out, dict) and "x_norm_patchtokens" in out:
                return tokens_to_nchw(out["x_norm_patchtokens"], height, width, self.patch_size, self.embed_dim)
        if hasattr(inner, "get_intermediate_layers"):
            layers = inner.get_intermediate_layers(normalized_rgb, n=1, reshape=True)
            grid = layers[0]
            if grid.shape[1] != self.embed_dim:
                raise ValueError(f"expected embed dim {self.embed_dim}, got {grid.shape[1]}")
            return grid
        hf_out = inner(pixel_values=normalized_rgb)
        hidden = hf_out.last_hidden_state
        patch = hidden[:, 1:, :]
        return tokens_to_nchw(patch, height, width, self.patch_size, self.embed_dim)


class FrozenDinoBackbone(nn.Module):
    """Frozen RGB encoder with the DINOv2-B/14 spatial contract.

    ``mode="stub"`` is a single stride-14 convolution (no network, CPU tests).
    ``mode="vitb14"`` loads official ViT-B/14 weights (hub, HF, or local file).
    ``mode="auto"`` is stub on CPU / when weights are absent; vitb14 when a CUDA
    device is selected and weights are cached or download is allowed.
    """

    def __init__(
        self,
        mode: str = "stub",
        embed_dim: int = DINOV2_B_EMBED,
        patch_size: int = DINOV2_B_PATCH,
        weights_path: str | Path | None = None,
        allow_download: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        resolved = resolve_backbone_mode(
            mode,
            device=device,
            weights_path=weights_path,
            allow_download=allow_download,
        )
        self.requested_mode = mode
        self.mode = resolved
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.source = "stub"
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        if resolved == "stub":
            self.encoder: nn.Module = nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            if weights_path is not None:
                self._load_local_patch(Path(weights_path))
        else:
            if (
                weights_path is None
                and not allow_download
                and not official_weights_cached()
                and hub_local_repo() is None
            ):
                raise ValueError(
                    "vitb14 requires a local DINOv2-B weights file, a torch-hub/"
                    "Hugging Face cache, or allow_download=True; "
                    "this package does not download checkpoints by default"
                )
            inner, source = load_official_vitb14(
                weights_path=Path(weights_path) if weights_path is not None else None,
                allow_download=allow_download,
            )
            self.encoder = OfficialDinov2Grid(inner, embed_dim=embed_dim, patch_size=patch_size)
            self.source = source
        freeze_module(self)

    def _load_local_patch(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "patch_embed.proj.weight" in payload:
            conv = self.encoder
            if not isinstance(conv, nn.Conv2d):
                raise ValueError("patch-embed weights require stub encoder")
            conv.weight.copy_(payload["patch_embed.proj.weight"])
            if "patch_embed.proj.bias" in payload and conv.bias is not None:
                conv.bias.copy_(payload["patch_embed.proj.bias"])
        elif isinstance(payload, dict) and "weight" in payload:
            self.encoder.load_state_dict(payload)
        else:
            raise ValueError(f"unrecognized local DINO weight format: {path}")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"RGB tensor must be [B, 3, H, W], got {tuple(rgb.shape)}")
        _, _, height, width = rgb.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"spatial size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        x = (rgb - self.mean) / self.std
        return self.encoder(x)


def load_frozen_dino(
    mode: str = "auto",
    *,
    weights_path: str | Path | None = None,
    allow_download: bool = False,
    device: str | torch.device | None = None,
) -> FrozenDinoBackbone:
    """Construct the frozen backbone; default is stub when weights are absent."""
    net = FrozenDinoBackbone(
        mode=mode,
        weights_path=weights_path,
        allow_download=allow_download,
        device=device,
    )
    if device is not None:
        net = net.to(device)
    return net


def interpolate_pos_grid(feat: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear resize of a feature grid to ``(height, width)``."""
    return F.interpolate(feat, size=(height, width), mode="bilinear", align_corners=False)
