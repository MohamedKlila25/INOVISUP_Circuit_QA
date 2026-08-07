# -*- coding: utf-8 -*-
"""
Common.Config — configuration typée, chargée depuis Configs/*.yaml.

Usage
-----
    from Common.Config import HybridConfig
    cfg = HybridConfig.from_yaml("Configs/hybrid.yaml")
    cfg.yolo.epochs
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def _project_root() -> Path:
    """Racine du dépôt : dossier contenant pyproject.toml.

    Un chemin relatif dans une config doit toujours être résolu depuis
    la racine du dépôt, jamais depuis le dossier courant — qui change
    selon qu'on lance un script, un notebook Jupyter ou un job SLURM.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _project_root()


class _YamlLoadable(BaseModel):
    @classmethod
    def from_yaml(cls, path: str | Path) -> "_YamlLoadable":
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / p


# ═══════════════════════════════════════════════════════════════════
#  Configs/hybrid.yaml
# ═══════════════════════════════════════════════════════════════════

class YoloConfig(BaseModel):
    model: str = "yolo11n.pt"
    epochs: int = Field(60, gt=0)
    imgsz: int = Field(800, gt=0)
    batch: int = Field(32, gt=0)
    conf: float = Field(0.35, ge=0.0, le=1.0)
    iou: float = Field(0.50, ge=0.0, le=1.0)
    project: str = "Runs/hybrid"
    name: str = "yolo11n_circuits"


class WireTracerConfig(BaseModel):
    bbox_margin: int = Field(6, ge=0)
    touch_dist: int = Field(10, gt=0)
    min_net_pixels: int = Field(30, gt=0)


class OcrConfig(BaseModel):
    enabled: bool = True
    gpu: bool = True
    search_margin: int = Field(55, gt=0)


class HybridDataConfig(BaseModel):
    # Le générateur écrit directement dans Data/ (images/, labels/,
    # annotations/, circuit.yaml) — pas de sous-dossier versionné.
    dataset_dir: str = "Data"
    train_json: str = "Data/finetune/train_conversations.json"
    test_json: str = "Data/finetune/test_conversations.json"

    @field_validator("dataset_dir", "train_json", "test_json")
    @classmethod
    def _reject_legacy_paths(cls, v: str) -> str:
        if v.lower().startswith("notebooks/"):
            raise ValueError(
                f"chemin '{v}' préfixé par 'notebooks/' — obsolète, "
                f"les données vivent sous Data/ à la racine du dépôt.")
        return v

    @property
    def circuit_yaml(self) -> str:
        """Chemin du circuit.yaml généré par Data_Generation, prêt pour YOLO."""
        return f"{self.dataset_dir}/circuit.yaml"


class EvaluationConfig(BaseModel):
    n_samples: int = Field(500, gt=0)
    wandb_project: str = "circuitvqa"
    wandb_dir: str = "/tmp"


class HybridConfig(_YamlLoadable):
    yolo: YoloConfig
    wire_tracer: WireTracerConfig
    ocr: OcrConfig
    data: HybridDataConfig
    evaluation: EvaluationConfig


# ═══════════════════════════════════════════════════════════════════
#  Configs/train_vlm.yaml
# ═══════════════════════════════════════════════════════════════════

class VLMModelConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    hf_cache: str | None = None
    quantization: str | None = None

    @field_validator("quantization")
    @classmethod
    def _valid_quantization(cls, v: str | None) -> str | None:
        if v is not None and v not in ("4bit", "8bit"):
            raise ValueError(f"quantization doit être '4bit', '8bit' ou null, reçu {v!r}")
        return v


class VLMDataConfig(BaseModel):
    train_json: str
    test_json: str
    max_length: int = Field(1024, gt=0)
    min_pixels: int = Field(gt=0)
    max_pixels: int = Field(gt=0)

    @model_validator(mode="after")
    def _pixel_range_valid(self) -> "VLMDataConfig":
        if self.min_pixels >= self.max_pixels:
            raise ValueError(
                f"min_pixels ({self.min_pixels}) doit être < "
                f"max_pixels ({self.max_pixels})")
        return self


class LoraConfig(BaseModel):
    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    target_modules: list[str] = Field(default_factory=list)
    bias: str = "none"


class TrainingConfig(BaseModel):
    epochs: int = Field(3, gt=0)
    batch_size: int = Field(4, gt=0)
    grad_accum: int = Field(8, gt=0)
    lr: float = Field(1e-5, gt=0)
    warmup_ratio: float = Field(0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(0.05, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0)
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, gt=0)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


class VLMLoggingConfig(BaseModel):
    log_steps: int = Field(10, gt=0)
    eval_steps: int = Field(200, gt=0)
    save_steps: int = Field(1000, gt=0)
    n_metric_samples: int = Field(10, gt=0)
    wandb_project: str = "circuitvqa"
    wandb_dir: str = "/tmp"
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    output_dir: str


class VLMTrainConfig(_YamlLoadable):
    model: VLMModelConfig
    data: VLMDataConfig
    lora: LoraConfig
    training: TrainingConfig
    logging: VLMLoggingConfig
    output: OutputConfig
