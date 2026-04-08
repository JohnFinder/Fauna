from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _NonNegativeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.weight_unconstrained = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.trunc_normal_(self.weight_unconstrained, std=0.02)
        nn.init.constant_(self.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.weight_unconstrained)
        return F.linear(x, w, self.bias)


class PrototypicalPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tags: int,
        num_prototypes_per_tag: int = 5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_tags = num_tags
        self.num_prototypes_per_tag = num_prototypes_per_tag
        self.prototypes = nn.Parameter(
            torch.empty(num_tags, num_prototypes_per_tag, dim),
        )
        nn.init.trunc_normal_(self.prototypes, std=0.02)
        self.class_heads = nn.ModuleList(
            _NonNegativeLinear(num_prototypes_per_tag, 1) for _ in range(num_tags)
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, p, d = patch_embeddings.shape
        patch_n = F.normalize(patch_embeddings, dim=-1)
        proto_n = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bpd,tkd->btkp", patch_n, proto_n)
        pooled = sim.max(dim=-1).values
        max_similarities = pooled.max(dim=-1).values
        logits_list = [
            self.class_heads[t](pooled[:, t, :]).squeeze(-1) for t in range(self.num_tags)
        ]
        logits = torch.stack(logits_list, dim=1)
        return logits, max_similarities


class ContextMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        num_context_features: int,
        num_context_tags: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim + num_context_features, dim)
        self.act = Swish()
        self.fc2 = nn.Linear(dim, num_context_tags)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([cls_embedding, context_features], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)


class TagInferenceHead(nn.Module):
    def __init__(
        self,
        dim: int,
        acoustic_functional_tags: List[str],
        context_tags: List[str],
        num_context_features: int,
        num_prototypes: int = 5,
        novelty_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.acoustic_functional_tag_names = list(acoustic_functional_tags)
        self.context_tag_names = list(context_tags)
        self.num_context_features = num_context_features
        self.novelty_threshold = novelty_threshold
        self.proto_pool = PrototypicalPooling(
            dim=dim,
            num_tags=len(acoustic_functional_tags),
            num_prototypes_per_tag=num_prototypes,
        )
        self.context_mlp = ContextMLP(
            dim=dim,
            num_context_features=num_context_features,
            num_context_tags=len(context_tags),
        )

    def forward(
        self,
        patch_embeddings: torch.Tensor,
        cls_embedding: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = patch_embeddings.size(0)
        if context_features is None:
            ctx = torch.zeros(
                b,
                self.num_context_features,
                device=cls_embedding.device,
                dtype=cls_embedding.dtype,
            )
        else:
            ctx = context_features
        af_logits, max_sim = self.proto_pool(patch_embeddings)
        ctx_logits = self.context_mlp(cls_embedding, ctx)
        novelty_flag = max_sim.max(dim=1).values < self.novelty_threshold
        return {
            "acoustic_functional_logits": af_logits,
            "context_logits": ctx_logits,
            "max_similarities": max_sim,
            "novelty_flag": novelty_flag,
        }

    def predict(
        self,
        patch_embeddings: torch.Tensor,
        cls_embedding: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(patch_embeddings, cls_embedding, context_features)
        t = max(temperature, 1e-8)
        return {
            "acoustic_functional_probs": torch.sigmoid(out["acoustic_functional_logits"] / t),
            "context_probs": torch.sigmoid(out["context_logits"] / t),
            "max_similarities": out["max_similarities"],
            "novelty_flag": out["novelty_flag"],
        }


class TemperatureScaling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-8)
