"""DINOv2-based change detector.

Compares rendered and observed images using DINOv2 feature similarity
to predict a per-pixel change mask.
"""

import torch
import torchvision

import clsplats.change_detection.base_detector as base_detector
from clsplats.config import ChangeDetectionConfig
from clsplats.utils.custom_types import Image


class DinoV2Detector(base_detector.BaseDetector):
    """Detect scene changes by comparing DINOv2 feature cosine similarity."""

    def __init__(self, cfg: ChangeDetectionConfig):
        super().__init__(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True
        )
        self.model.eval()  # type: ignore
        self.model.to(self.device)  # type: ignore
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _preprocess_image(self, image: Image) -> torch.Tensor:
        """Resize to DINO-compatible dimensions and apply ImageNet normalisation."""
        DINO_PATCH_SIZE = 14
        h, w, c = image.shape
        aligned_h = h - (h % DINO_PATCH_SIZE)
        aligned_w = w - (w % DINO_PATCH_SIZE)
        # Permute to (C, H, W) for torch transforms
        image_t = image.permute(2, 0, 1)
        resized = torch.nn.functional.interpolate(
            image_t.unsqueeze(0),
            size=(aligned_h, aligned_w),
            mode="bilinear",
            align_corners=False,
        )
        # Apply ImageNet normalisation (Bug 3 fix)
        normalised = self.normalize(resized.squeeze(0))
        return normalised

    def _dilate_mask(self, mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Morphologically dilate a binary mask."""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), dtype=torch.float32, device=mask.device
        )
        mask = mask.float().unsqueeze(0)  # (1, 1, H, W)
        dilated = torch.nn.functional.conv2d(mask, kernel, padding=kernel_size // 2)
        dilated = (dilated > 0).float()
        return dilated.squeeze(0).squeeze(0)

    def predict_change_mask(self, rendered_image: Image, observation: Image) -> torch.Tensor:
        """Return a boolean change mask comparing *rendered_image* to *observation*.

        Args:
            rendered_image: Rendered image ``[H, W, 3]``.
            observation: Ground-truth observation ``[H, W, 3]``.

        Returns:
            Boolean mask ``[H, W]`` where ``True`` indicates a changed pixel.
        """
        rendered = self._preprocess_image(rendered_image)
        observed = self._preprocess_image(observation)
        # DINO expects (B, C, H, W)
        rendered = rendered.unsqueeze(0)
        observed = observed.unsqueeze(0)
        with torch.no_grad():
            (rendered_feats,) = self.model.get_intermediate_layers(rendered, reshape=True)  # type: ignore
            (observed_feats,) = self.model.get_intermediate_layers(observed, reshape=True)  # type: ignore
            # DINOv2 returns (B, C, H, W) — use the batch directly, not [0]
            # (Bug 6 fix: removed double-indexing)
            cos_sim = self.cos(rendered_feats, observed_feats)  # (1, H, W) — batch dim preserved
            cos_sim = cos_sim.squeeze(0)  # (H, W)
            mask = cos_sim < self.cfg.threshold


        if self.cfg.dilate_mask:
            mask = self._dilate_mask(mask, kernel_size=self.cfg.dilate_kernel_size)
        # Optionally upsample back to original size
        if self.cfg.upsample:
            orig_h, orig_w, _ = rendered_image.shape
            mask = mask.float().unsqueeze(0).unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask, size=(orig_h, orig_w), mode="nearest")
            mask = mask.squeeze(0).squeeze(0) > 0.5
        return mask
