from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    input_dim: int = 30
    hidden_dim: int = 128
    embed_dim: int = 64
    gamma: float = 10.0
    lambda_open: float = 0.1
    beta_entropy: float = 0.2
    alpha1: float = 1.0
    alpha2: float = 0.6
    alpha3: float = 0.3
    latent_dim: int = 32


class FeatureBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VectorEmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        hidden = max(embed_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReciprocalPointClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, gamma: float, lambda_open: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambda_open = lambda_open
        self.reciprocal_points = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.radii = nn.Parameter(torch.ones(num_classes) * 0.6)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        points = self.reciprocal_points
        d_e = ((embedding[:, None, :] - points[None, :, :]) ** 2).mean(dim=-1)
        d_d = (embedding[:, None, :] * points[None, :, :]).sum(dim=-1) / (embedding.shape[-1] ** 0.5)
        distance = d_e - d_d
        logits = self.gamma * distance
        probs = F.softmax(logits, dim=-1)
        output = {
            "distance": distance,
            "logits": logits,
            "probs": probs,
            "d_e": d_e,
        }
        if labels is not None:
            classification = F.cross_entropy(logits, labels)
            radii = F.softplus(self.radii)
            open_risk = F.relu(radii[labels] - d_e.gather(1, labels.unsqueeze(1)).squeeze(1)).mean()
            total = classification + self.lambda_open * open_risk
            output.update(
                {
                    "classification_loss": classification,
                    "open_loss": open_risk,
                    "total_loss": total,
                }
            )
        return output


class FeaturePyramidBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.stage1 = FeatureBlock(input_dim, hidden_dim)
        self.stage2 = FeatureBlock(hidden_dim, max(64, hidden_dim // 2))
        self.stage3 = FeatureBlock(max(64, hidden_dim // 2), max(48, hidden_dim // 3))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        a3 = self.stage3(f2)
        return {"f1": f1, "f2": f2, "a3": a3}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = in_channels
        for _ in range(layers):
            modules.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            current = out_channels
        self.block = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LaplacianPyramidReconstructionBackbone(nn.Module):
    """Paper-style 2-D PDG backbone with three score-map reconstruction scales."""

    def __init__(self, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        low_channels = max(32, hidden_dim // 2)
        mid_channels = max(64, hidden_dim)
        high_channels = 512
        bottleneck_channels = 4096
        self.stem = ConvBlock(1, low_channels, layers=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.block16 = ConvBlock(low_channels, 128, layers=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.block8 = nn.Sequential(
            ConvBlock(128, mid_channels, layers=1),
            ConvBlock(mid_channels, high_channels, layers=1),
        )
        self.to_bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(high_channels, bottleneck_channels, kernel_size=1),
            nn.GELU(),
        )
        self.reconstruct3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, 512, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )
        self.reconstruct2 = nn.Sequential(
            nn.Conv2d(high_channels, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.reconstruct1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _fuse(self, high_resolution: torch.Tensor, low_resolution: torch.Tensor) -> torch.Tensor:
        upsampled_low = self._resize(low_resolution, high_resolution.shape[-2:])
        low_mask = F.softmax(upsampled_low, dim=1)
        detail_mask = high_resolution - low_mask
        detail = detail_mask * high_resolution
        return upsampled_low + detail

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != 65 * 65:
            raise ValueError(f"Paper PR-RPAD expects flattened 65x65 PDG input, got shape {tuple(x.shape)}.")
        image = x.view(x.shape[0], 1, 65, 65)
        stem = self.stem(image)
        f16 = self.block16(self.pool1(stem))
        f8 = self.block8(self.pool2(f16))
        f6 = self.to_bottleneck(f8)
        a3 = self._resize(self.reconstruct3(f6), (16, 16))
        a2 = self._resize(self.reconstruct2(f8), (32, 32))
        a1 = self._resize(self.reconstruct1(f16), (65, 65))
        f2 = self._fuse(a2, a3)
        f1 = self._fuse(a1, f2)
        return {
            "f1": f1,
            "f2": f2,
            "a3": a3,
            "feature16": f16,
            "feature8": f8,
            "feature6": f6,
        }


class ScoreMapEmbeddingHead(nn.Module):
    def __init__(self, num_classes: int, height: int, width: int, embed_dim: int) -> None:
        super().__init__()
        in_dim = num_classes * height * width
        hidden = min(512, max(embed_dim, in_dim // 4))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConfusingSampleGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PRRPADModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = FeaturePyramidBackbone(config.input_dim, config.hidden_dim)
        stage2_dim = max(64, config.hidden_dim // 2)
        stage3_dim = max(48, config.hidden_dim // 3)
        self.scale1 = VectorEmbeddingHead(config.hidden_dim, config.embed_dim)
        self.scale2 = VectorEmbeddingHead(stage2_dim, config.embed_dim)
        self.scale3 = VectorEmbeddingHead(stage3_dim, config.embed_dim)
        self.head1 = ReciprocalPointClassifier(config.num_classes, config.embed_dim, config.gamma, config.lambda_open)
        self.head2 = ReciprocalPointClassifier(config.num_classes, config.embed_dim, config.gamma, config.lambda_open)
        self.head3 = ReciprocalPointClassifier(config.num_classes, config.embed_dim, config.gamma, config.lambda_open)
        self.generator = ConfusingSampleGenerator(config.latent_dim, config.input_dim)
        self.discriminator = Discriminator(config.input_dim)

    def _scale_outputs(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> list[dict[str, torch.Tensor]]:
        features = self.backbone(x)
        embeddings = [
            self.scale1(features["f1"]),
            self.scale2(features["f2"]),
            self.scale3(features["a3"]),
        ]
        heads = [self.head1, self.head2, self.head3]
        return [head(embed, labels) for head, embed in zip(heads, embeddings, strict=True)]

    def forward_known(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._scale_outputs(x, labels)
        total = (
            self.config.alpha1 * outputs[0]["total_loss"]
            + self.config.alpha2 * outputs[1]["total_loss"]
            + self.config.alpha3 * outputs[2]["total_loss"]
        )
        combined_probs = self.combine_probs(outputs)
        return {"loss": total, "outputs": outputs, "probs": combined_probs, "open_score": self.open_score(outputs)}

    def forward_unknown(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._scale_outputs(x, None)
        combined_probs = self.combine_probs(outputs)
        return {"outputs": outputs, "probs": combined_probs, "open_score": self.open_score(outputs)}

    def combine_probs(self, outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.stack([item["probs"] for item in outputs], dim=0).mean(dim=0)

    def reciprocal_point_tensors(self) -> list[torch.Tensor]:
        return [
            self.head1.reciprocal_points,
            self.head2.reciprocal_points,
            self.head3.reciprocal_points,
        ]

    def confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_unknown(x)
        probs = outputs["probs"]
        confidence, labels = probs.max(dim=1)
        return labels, confidence

    def open_score(self, outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        scores = []
        for output in outputs:
            probs = output["probs"]
            preds = probs.argmax(dim=1)
            scores.append(output["distance"].gather(1, preds.unsqueeze(1)).squeeze(1))
        return torch.stack(scores, dim=0).mean(dim=0)

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1).mean()


class PaperPRRPADModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.input_dim != 65 * 65:
            raise ValueError("PaperPRRPADModel requires input_dim=4225 from 65x65 PDG images.")
        self.config = config
        self.backbone = LaplacianPyramidReconstructionBackbone(config.num_classes, config.hidden_dim)
        self.scale1 = ScoreMapEmbeddingHead(config.num_classes, 65, 65, config.embed_dim)
        self.scale2 = ScoreMapEmbeddingHead(config.num_classes, 32, 32, config.embed_dim)
        self.scale3 = ScoreMapEmbeddingHead(config.num_classes, 16, 16, config.embed_dim)
        self.head1 = ReciprocalPointClassifier(config.num_classes, config.embed_dim, config.gamma, config.lambda_open)
        self.head2 = ReciprocalPointClassifier(config.num_classes, config.embed_dim, config.gamma, config.lambda_open)
        self.head3 = ReciprocalPointClassifier(config.num_classes, config.embed_dim, config.gamma, config.lambda_open)
        self.generator = ConfusingSampleGenerator(config.latent_dim, config.input_dim)
        self.discriminator = Discriminator(config.input_dim)

    def _scale_outputs(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> list[dict[str, torch.Tensor]]:
        features = self.backbone(x)
        score_maps = [features["f1"], features["f2"], features["a3"]]
        embeddings = [
            self.scale1(score_maps[0]),
            self.scale2(score_maps[1]),
            self.scale3(score_maps[2]),
        ]
        heads = [self.head1, self.head2, self.head3]
        outputs = []
        for score_map, head, embedding in zip(score_maps, heads, embeddings, strict=True):
            output = head(embedding, labels)
            output["score_map"] = score_map
            output["map_probs"] = F.softmax(score_map, dim=1)
            outputs.append(output)
        return outputs

    def forward_known(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._scale_outputs(x, labels)
        total = (
            self.config.alpha1 * outputs[0]["total_loss"]
            + self.config.alpha2 * outputs[1]["total_loss"]
            + self.config.alpha3 * outputs[2]["total_loss"]
        )
        combined_probs = self.combine_probs(outputs)
        return {"loss": total, "outputs": outputs, "probs": combined_probs, "open_score": self.open_score(outputs)}

    def forward_unknown(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._scale_outputs(x, None)
        combined_probs = self.combine_probs(outputs)
        return {"outputs": outputs, "probs": combined_probs, "open_score": self.open_score(outputs)}

    def combine_probs(self, outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        return torch.stack([item["probs"] for item in outputs], dim=0).mean(dim=0)

    def reciprocal_point_tensors(self) -> list[torch.Tensor]:
        return [
            self.head1.reciprocal_points,
            self.head2.reciprocal_points,
            self.head3.reciprocal_points,
        ]

    def confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.forward_unknown(x)
        probs = outputs["probs"]
        confidence, labels = probs.max(dim=1)
        return labels, confidence

    def open_score(self, outputs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        scores = []
        for output in outputs:
            probs = output["probs"]
            preds = probs.argmax(dim=1)
            scores.append(output["distance"].gather(1, preds.unsqueeze(1)).squeeze(1))
        return torch.stack(scores, dim=0).mean(dim=0)

    def entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
