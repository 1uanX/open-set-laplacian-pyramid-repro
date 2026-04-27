from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


ROUND4_CONTRACT_VERSION = "round4_v1"
ROUND4_FIXED_SEEDS = (20260414, 20260417, 20260418)
ROUND4_EXPERIMENTS = ("exp4", "exp5", "exp6")
ROUND4_DEFAULT_TARGET_ROOTS = ("backbone", "scale1", "scale2", "scale3")
ROUND4_ALLOWED_VARIANTS = {
    "p003_baseline",
    "full_lora",
    "frozen_no_adapter",
    "full_finetune",
    "rank_sweep",
    "no_preservation_lock",
    "merged_lora",
    "unmerged_lora",
}
ROUND4_LORA_VARIANTS = {
    "full_lora",
    "rank_sweep",
    "no_preservation_lock",
    "merged_lora",
    "unmerged_lora",
}


def _tuple_of_str(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"{field_name} must be a list of strings.")


def _tuple_of_int(value: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(value, tuple) and all(isinstance(item, int) for item in value):
        return value
    if isinstance(value, list):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a list of integers.") from exc
    raise ValueError(f"{field_name} must be a list of integers.")


@dataclass(frozen=True)
class Round4Settings:
    contract_version: str = ROUND4_CONTRACT_VERSION
    enabled: bool = False
    method_variant: str = "p003_baseline"
    seed: int | None = None
    fixed_seeds: tuple[int, ...] = ROUND4_FIXED_SEEDS
    run_id: str | None = None
    output_dir: str | None = None
    checkpoint_path: str | None = None
    adapter_rank: int = 0
    adapter_alpha: float = 16.0
    adapter_dropout: float = 0.0
    target_module_roots: tuple[str, ...] = ROUND4_DEFAULT_TARGET_ROOTS
    preservation_lock: bool = True
    merge_adapters_for_inference: bool = False
    resource_logging: bool = True
    latency_warmup_steps: int = 1
    latency_repeats: int = 3
    expected_experiments: tuple[str, ...] = ROUND4_EXPERIMENTS

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Round4Settings":
        payload = dict(payload or {})
        if "target_modules" in payload:
            payload["target_module_roots"] = payload.pop("target_modules")
        valid_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown Round4 config keys: {unknown}")
        if "target_module_roots" in payload:
            payload["target_module_roots"] = _tuple_of_str(payload["target_module_roots"], "target_module_roots")
        if "fixed_seeds" in payload:
            payload["fixed_seeds"] = _tuple_of_int(payload["fixed_seeds"], "fixed_seeds")
        if "expected_experiments" in payload:
            payload["expected_experiments"] = _tuple_of_str(payload["expected_experiments"], "expected_experiments")
        settings = cls(**payload)
        settings.validate()
        return settings

    @classmethod
    def from_json(cls, path: Path | str | None) -> "Round4Settings":
        if path is None:
            return cls()
        config_path = Path(path)
        return cls.from_dict(json.loads(config_path.read_text(encoding="utf-8")))

    def with_overrides(self, **overrides: Any) -> "Round4Settings":
        payload = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                payload[key] = value
        return Round4Settings.from_dict(payload)

    def validate(self) -> None:
        if self.contract_version != ROUND4_CONTRACT_VERSION:
            raise ValueError("Round4 contract_version must be round4_v1.")
        if self.method_variant not in ROUND4_ALLOWED_VARIANTS:
            raise ValueError(f"Unsupported Round4 method_variant: {self.method_variant}")
        if tuple(self.fixed_seeds) != ROUND4_FIXED_SEEDS:
            raise ValueError("Round4 fixed_seeds must be exactly 20260414, 20260417, 20260418.")
        if self.seed is not None and int(self.seed) not in ROUND4_FIXED_SEEDS:
            raise ValueError("Round4 seed must be one of the fixed contract seeds.")
        if tuple(self.expected_experiments) != ROUND4_EXPERIMENTS:
            raise ValueError("Round4 expected_experiments must be exactly exp4, exp5, exp6.")
        if self.adapter_rank < 0:
            raise ValueError("Round4 adapter_rank must be non-negative.")
        if self.adapter_alpha <= 0:
            raise ValueError("Round4 adapter_alpha must be positive.")
        if not 0.0 <= self.adapter_dropout < 1.0:
            raise ValueError("Round4 adapter_dropout must be in [0, 1).")
        if not self.target_module_roots:
            raise ValueError("Round4 target_module_roots cannot be empty.")
        if self.enabled and self.method_variant in ROUND4_LORA_VARIANTS and self.adapter_rank <= 0:
            raise ValueError("Round4 LoRA variants require adapter_rank > 0.")

    def validate_selected_experiments(self, selected: Iterable[str]) -> None:
        unsupported = sorted(set(selected) - set(self.expected_experiments))
        if self.enabled and unsupported:
            raise ValueError(f"Round4 selected experiments must stay within exp4, exp5, exp6: {unsupported}")

    def uses_lora(self) -> bool:
        return bool(self.enabled and self.method_variant in ROUND4_LORA_VARIANTS and self.adapter_rank > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "enabled": self.enabled,
            "method_variant": self.method_variant,
            "seed": self.seed,
            "fixed_seeds": list(self.fixed_seeds),
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "checkpoint_path": self.checkpoint_path,
            "adapter_rank": self.adapter_rank,
            "adapter_alpha": self.adapter_alpha,
            "adapter_dropout": self.adapter_dropout,
            "target_module_roots": list(self.target_module_roots),
            "preservation_lock": self.preservation_lock,
            "merge_adapters_for_inference": self.merge_adapters_for_inference,
            "resource_logging": self.resource_logging,
            "latency_warmup_steps": self.latency_warmup_steps,
            "latency_repeats": self.latency_repeats,
            "expected_experiments": list(self.expected_experiments),
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRALinear rank must be positive.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.merged = False
        self.reset_lora_parameters()
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        if self.merged:
            return
        with torch.no_grad():
            self.base.weight.add_(self.delta_weight().to(dtype=self.base.weight.dtype))
        self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            return
        with torch.no_grad():
            self.base.weight.sub_(self.delta_weight().to(dtype=self.base.weight.dtype))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        if self.merged:
            return result
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + update * self.scaling


def _named_parent_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = dict(model.named_modules())
    parents: dict[str, nn.Module] = {}
    for name in modules:
        if "." not in name:
            parents[name] = model
        else:
            parent_name = name.rsplit(".", 1)[0]
            parents[name] = modules[parent_name]
    return parents


def _is_round4_target(name: str, target_roots: tuple[str, ...]) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in target_roots)


def inject_lora_adapters(model: nn.Module, settings: Round4Settings) -> dict[str, Any]:
    if not settings.uses_lora():
        return {"inserted_modules": [], "inserted_count": 0}
    parents = _named_parent_modules(model)
    inserted: list[str] = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _is_round4_target(name, settings.target_module_roots):
            continue
        parent = parents[name]
        child_name = name.rsplit(".", 1)[-1]
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=settings.adapter_rank,
                alpha=settings.adapter_alpha,
                dropout=settings.adapter_dropout,
            ),
        )
        inserted.append(name)
    if not inserted:
        raise ValueError("Round4 LoRA target selection did not match any Linear modules.")
    return {"inserted_modules": inserted, "inserted_count": len(inserted)}


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def merge_lora_adapters(model: nn.Module) -> list[str]:
    merged = []
    for name, module in iter_lora_modules(model):
        module.merge()
        merged.append(name)
    return merged


def unmerge_lora_adapters(model: nn.Module) -> list[str]:
    unmerged = []
    for name, module in iter_lora_modules(model):
        module.unmerge()
        unmerged.append(name)
    return unmerged


def _set_all_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(value)


def _set_lora_requires_grad(model: nn.Module, value: bool = True) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(bool(value and (".lora_A" in name or ".lora_B" in name)))


def _set_boundary_heads_requires_grad(model: nn.Module, value: bool) -> None:
    for head_name in ("head1", "head2", "head3"):
        head = getattr(model, head_name, None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad_(value)


def configure_round4_trainable_parameters(model: nn.Module, settings: Round4Settings) -> dict[str, Any]:
    if not settings.enabled or settings.method_variant == "p003_baseline":
        _set_all_requires_grad(model, True)
    elif settings.method_variant == "full_finetune":
        _set_all_requires_grad(model, True)
    elif settings.method_variant == "frozen_no_adapter":
        _set_all_requires_grad(model, False)
    elif settings.uses_lora():
        _set_lora_requires_grad(model, True)
        if not settings.preservation_lock:
            _set_boundary_heads_requires_grad(model, True)
    else:
        raise ValueError(f"Unsupported Round4 trainable setup: {settings.method_variant}")
    return parameter_report(model)


def apply_round4_scaffolding(model: nn.Module, settings: Round4Settings) -> dict[str, Any]:
    adapter_report = inject_lora_adapters(model, settings)
    parameter_payload = configure_round4_trainable_parameters(model, settings)
    merged_modules: list[str] = []
    if settings.uses_lora() and settings.merge_adapters_for_inference:
        merged_modules = merge_lora_adapters(model)
    return {
        "settings": settings.to_dict(),
        "adapter_report": adapter_report,
        "parameter_report": parameter_payload,
        "merged_modules": merged_modules,
        "merge_supported": True,
    }


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    adapter = 0
    trainable_names: list[str] = []
    adapter_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if ".lora_A" in name or ".lora_B" in name:
            adapter += count
            adapter_names.append(name)
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "adapter_parameters": adapter,
        "trainable_parameter_names": trainable_names,
        "adapter_parameter_names": adapter_names,
    }


def measure_inference_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    *,
    warmup_steps: int,
    repeats: int,
) -> float:
    if repeats <= 0:
        raise ValueError("latency repeats must be positive.")
    model.eval()
    sample = sample.to(device)
    model.to(device)
    with torch.no_grad():
        for _ in range(max(0, warmup_steps)):
            model.forward_unknown(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model.forward_unknown(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    return float((elapsed / repeats) * 1000.0)
