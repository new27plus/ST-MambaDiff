"""Core training engine for UCLA ablation experiments."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def find_project_root(start_dir: Path) -> Path:
    """Find the project root by locating feeders and models directories."""
    current = start_dir.resolve()
    while True:
        if (current / "feeders").is_dir() and (current / "models").is_dir():
            return current
        parent = current.parent
        if parent == current:
            return start_dir.resolve()
        current = parent


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feeders.ucla_feeder import Feeder
from experiments.UCLA.code.models.embeddings import TimestepEmbedding

DEFAULT_DATA_ROOT = (
    "/home/hp/code/Spatio-Temporal_U-Net_Denoising/"
    "Spatio-Temporal_U-Net_Denoising/datasets/NW-UCLA/"
)
DEFAULT_SEED = 42


BONE_PAIRS_0_BASED = [
    (2 - 1, 1 - 1),
    (3 - 1, 2 - 1),
    (4 - 1, 3 - 1),
    (5 - 1, 3 - 1),
    (6 - 1, 5 - 1),
    (7 - 1, 6 - 1),
    (8 - 1, 7 - 1),
    (9 - 1, 3 - 1),
    (10 - 1, 9 - 1),
    (11 - 1, 10 - 1),
    (12 - 1, 11 - 1),
    (13 - 1, 1 - 1),
    (14 - 1, 13 - 1),
    (15 - 1, 14 - 1),
    (16 - 1, 15 - 1),
    (17 - 1, 1 - 1),
    (18 - 1, 17 - 1),
    (19 - 1, 18 - 1),
    (20 - 1, 19 - 1),
]


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Set worker seed for deterministic DataLoader behavior."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_scalar(value: str) -> Any:
    """Parse scalar values from YAML-like text."""
    raw = value.strip()
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.lower() in {"null", "none"}:
        return None
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')):
        return raw[1:-1]
    try:
        if "." in raw or "e" in raw.lower():
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_simple_yaml(config_path: Path) -> Dict[str, Any]:
    """Load a simple key-value YAML file without external dependencies."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data: Dict[str, Any] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = parse_scalar(value)
    return data


def ensure_data_paths(data_root: Path) -> Tuple[Path, Path, Path]:
    """Validate dataset files and return canonical paths."""
    skeleton_root = data_root / "all_sqe"
    train_label_path = data_root / "ucla_train_data.json"
    val_label_path = data_root / "ucla_val_data.json"
    missing = [
        str(path)
        for path in [skeleton_root, train_label_path, val_label_path]
        if not path.exists()
    ]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Dataset paths are missing. Check --data_root.\n"
            f"Missing paths:\n{joined}"
        )
    return skeleton_root, train_label_path, val_label_path


class RMSNorm(nn.Module):
    """RMS normalization used by temporal Mamba."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x**2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class Graph:
    """Build graph adjacency matrix for skeleton joints."""

    def __init__(self, feeder_bone_pairs: List[Tuple[int, int]], num_joints: int = 20):
        self.num_joints = num_joints
        self.bone_pairs = feeder_bone_pairs
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self) -> torch.Tensor:
        A = np.zeros((3, self.num_joints, self.num_joints), dtype=np.float32)
        A[0] = np.eye(self.num_joints, dtype=np.float32)
        for child, parent in self.bone_pairs:
            A[1, child, parent] = 1
            A[2, parent, child] = 1
        for i in range(3):
            D = np.sum(A[i], axis=1)
            D[D == 0] = 1
            D_inv = np.diag(D**-0.5)
            A[i] = D_inv @ A[i] @ D_inv
        return torch.from_numpy(A)


class AdvancedShiftGCN(nn.Module):
    """Spatial graph block with shift operation."""

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.num_partitions = A.shape[0]
        self.num_joints = A.shape[1]
        c_per_part = in_channels // self.num_partitions
        c_per_neighbor = max(1, c_per_part // 2)

        shift_index = torch.arange(self.num_joints).unsqueeze(0).repeat(in_channels, 1)
        A_binary = (A > 0.001).float()
        for p in range(self.num_partitions):
            part_start = p * c_per_part
            part_end = (p + 1) * c_per_part if p < self.num_partitions - 1 else in_channels
            if p == 0:
                continue
            for v in range(self.num_joints):
                neighbors = A_binary[p, v].nonzero(as_tuple=False).flatten()
                if len(neighbors) == 0:
                    targets = [v, v]
                else:
                    values = neighbors.tolist()
                    while len(values) < 2:
                        values.append(values[0])
                    targets = values[:2]
                for k, target in enumerate(targets):
                    sub_start = part_start + k * c_per_neighbor
                    sub_end = part_end if k == 1 else min(part_end, part_start + (k + 1) * c_per_neighbor)
                    if sub_start < sub_end:
                        shift_index[sub_start:sub_end, v] = target
        self.register_buffer("shift_index", shift_index.long())
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, _A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        B, C, T, V = x.shape
        index = self.shift_index.view(1, C, 1, V).expand(B, C, T, V)
        shifted = torch.gather(x, dim=3, index=index)
        out = self.conv(shifted)
        return out + res


class TemporalMambaBlock(nn.Module):
    """Temporal module based on Mamba."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for temporal_operator=mamba.")
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=1)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)
        seq = self.norm(seq)
        seq = self.mamba(seq)
        return seq.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()


class TemporalConvBlock(nn.Module):
    """Temporal module based on lightweight 1D convolution."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        seq = x.permute(0, 3, 1, 2).contiguous().view(B * V, C, T)
        seq = self.conv(seq)
        seq = seq.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()
        return self.act(self.norm(seq))


class TemporalTransformerBlock(nn.Module):
    """Temporal module based on Transformer encoder."""

    def __init__(self, d_model: int, nhead: int, ff_ratio: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_ratio,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        seq = x.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)
        seq = self.encoder(self.norm(seq))
        return seq.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()


class ResTemporalDownsample(nn.Module):
    """Temporal downsampling block with residual path."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
        )
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.proj(self.pool(x))


class STUnetBlock(nn.Module):
    """Unified U-Net block with configurable temporal operator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        A: torch.Tensor,
        temporal_operator: str,
        mamba_d_state: int,
        transformer_nhead: int,
        dropout: float,
    ):
        super().__init__()
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.conv1 = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        self.spatial_gcn = AdvancedShiftGCN(out_channels, out_channels, A)

        if temporal_operator == "mamba":
            self.temporal_block = TemporalMambaBlock(out_channels, mamba_d_state)
        elif temporal_operator == "conv":
            self.temporal_block = TemporalConvBlock(out_channels)
        elif temporal_operator == "transformer":
            self.temporal_block = TemporalTransformerBlock(out_channels, transformer_nhead)
        else:
            raise ValueError(f"Unsupported temporal operator: {temporal_operator}")

        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None, A: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if t_emb is not None:
            x = x + self.time_proj(t_emb)[:, :, None, None]
        x_s = self.spatial_gcn(x, A)
        x_t = self.temporal_block(x_s)
        return self.dropout(self.act(self.norm(x_t + x)))


class STUnetEncoder(nn.Module):
    """Encoder used by denoising and classifier models."""

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        time_embed_dim: int,
        A: torch.Tensor,
        temporal_operator: str,
        mamba_d_state: int,
        transformer_nhead: int,
        dropout: float,
    ):
        super().__init__()
        C1, C2, C3 = model_channels, model_channels * 2, model_channels * 4
        kwargs = dict(
            time_embed_dim=time_embed_dim,
            A=A,
            temporal_operator=temporal_operator,
            mamba_d_state=mamba_d_state,
            transformer_nhead=transformer_nhead,
            dropout=dropout,
        )
        self.enc_in = STUnetBlock(in_channels, C1, **kwargs)
        self.enc_block1 = STUnetBlock(C1, C1, **kwargs)
        self.down1 = ResTemporalDownsample(C1, C1)
        self.enc_block2 = STUnetBlock(C1, C2, **kwargs)
        self.enc_block3 = STUnetBlock(C2, C2, **kwargs)
        self.down2 = ResTemporalDownsample(C2, C2)
        self.mid_block1 = STUnetBlock(C2, C3, **kwargs)
        self.mid_block2 = STUnetBlock(C3, C3, **kwargs)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None, A: torch.Tensor):
        skips = []
        h1 = self.enc_block1(self.enc_in(x, t_emb, A), t_emb, A)
        skips.append(h1)
        h2 = self.enc_block3(self.enc_block2(self.down1(h1), t_emb, A), t_emb, A)
        skips.append(h2)
        h_mid = self.mid_block2(self.mid_block1(self.down2(h2), t_emb, A), t_emb, A)
        return h_mid, skips


class STUnetDecoder(nn.Module):
    """Decoder used for denoising pretraining."""

    def __init__(
        self,
        out_channels: int,
        model_channels: int,
        time_embed_dim: int,
        A: torch.Tensor,
        temporal_operator: str,
        mamba_d_state: int,
        transformer_nhead: int,
        dropout: float,
    ):
        super().__init__()
        C1, C2, C3 = model_channels, model_channels * 2, model_channels * 4
        kwargs = dict(
            time_embed_dim=time_embed_dim,
            A=A,
            temporal_operator=temporal_operator,
            mamba_d_state=mamba_d_state,
            transformer_nhead=transformer_nhead,
            dropout=dropout,
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode="nearest"),
            nn.Conv2d(C3, C2, (3, 1), padding=(1, 0)),
        )
        self.dec_block1 = STUnetBlock(C2 * 2, C2, **kwargs)
        self.dec_block2 = STUnetBlock(C2, C2, **kwargs)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1), mode="nearest"),
            nn.Conv2d(C2, C1, (3, 1), padding=(1, 0)),
        )
        self.dec_block3 = STUnetBlock(C1 * 2, C1, **kwargs)
        self.dec_block4 = STUnetBlock(C1, C1, **kwargs)
        self.out_head = nn.Conv2d(C1, out_channels, kernel_size=1)

    def forward(
        self,
        h_mid: torch.Tensor,
        skip_connections: List[torch.Tensor],
        t_emb: torch.Tensor | None,
        A: torch.Tensor,
    ) -> torch.Tensor:
        skips = list(skip_connections)
        h1 = skips.pop()
        h2 = skips.pop()
        d1 = self.dec_block2(self.dec_block1(torch.cat([self.up1(h_mid), h1], dim=1), t_emb, A), t_emb, A)
        d2 = self.dec_block4(self.dec_block3(torch.cat([self.up2(d1), h2], dim=1), t_emb, A), t_emb, A)
        return self.out_head(d2)


class DenoisingModel(nn.Module):
    """Denoising pretraining model with configurable decoder time conditioning."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.graph = Graph(BONE_PAIRS_0_BASED, num_joints=cfg["num_joints"])
        self.register_buffer("A", self.graph.A)
        self.time_embed_dim = cfg["model_channels"] * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(cfg["model_channels"]),
            nn.Linear(cfg["model_channels"], self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        common_kwargs = dict(
            in_channels=cfg["in_channels"],
            model_channels=cfg["model_channels"],
            time_embed_dim=self.time_embed_dim,
            A=self.A,
            temporal_operator=cfg["temporal_operator"],
            mamba_d_state=cfg["mamba_d_state"],
            transformer_nhead=cfg["transformer_nhead"],
            dropout=cfg["dropout"],
        )
        self.encoder = STUnetEncoder(**common_kwargs)
        self.decoder = STUnetDecoder(
            out_channels=cfg["in_channels"],
            model_channels=cfg["model_channels"],
            time_embed_dim=self.time_embed_dim,
            A=self.A,
            temporal_operator=cfg["temporal_operator"],
            mamba_d_state=cfg["mamba_d_state"],
            transformer_nhead=cfg["transformer_nhead"],
            dropout=cfg["dropout"],
        )
        self.decoder_time_condition = cfg["decoder_time_condition"]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h_mid, skips = self.encoder(x, t_emb, self.A)
        decoder_t_emb = t_emb if self.decoder_time_condition else None
        return self.decoder(h_mid, skips, decoder_t_emb, self.A)


class ClassificationHead(nn.Module):
    """Classifier head for HAR."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=3)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = torch.cat([x_max, x_avg], dim=1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fc2(x)


class ClassifierModel(nn.Module):
    """Classifier with shared encoder architecture."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.graph = Graph(BONE_PAIRS_0_BASED, num_joints=cfg["num_joints"])
        self.register_buffer("A", self.graph.A)
        self.time_embed_dim = cfg["model_channels"] * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(cfg["model_channels"]),
            nn.Linear(cfg["model_channels"], self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.encoder = STUnetEncoder(
            in_channels=cfg["in_channels"],
            model_channels=cfg["model_channels"],
            time_embed_dim=self.time_embed_dim,
            A=self.A,
            temporal_operator=cfg["temporal_operator"],
            mamba_d_state=cfg["mamba_d_state"],
            transformer_nhead=cfg["transformer_nhead"],
            dropout=cfg["dropout"],
        )
        self.classifier = ClassificationHead(cfg["model_channels"] * 4, cfg["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_mid, _ = self.encoder(x, None, self.A)
        return self.classifier(h_mid)


@dataclass
class ExperimentContext:
    """Runtime context for a single ablation experiment."""

    cfg: Dict[str, Any]
    device: torch.device
    log_dir: Path
    metrics_path: Path
    cmd_args_path: Path
    best_ckpt_path: Path
    pretrain_ckpt_path: Path


def build_context(
    experiment_name: str,
    cli_args: argparse.Namespace,
    base_config: Dict[str, Any],
) -> ExperimentContext:
    """Build runtime context with merged config and output paths."""
    cfg = copy.deepcopy(base_config)
    cfg["experiment_name"] = experiment_name
    for key in [
        "subset_ratio",
        "modality",
        "batch_size",
        "num_workers",
        "pretrain_epochs",
        "finetune_epochs",
        "pretrain_lr",
        "finetune_lr",
        "model_channels",
        "mamba_d_state",
        "dropout",
        "transformer_nhead",
    ]:
        value = getattr(cli_args, key, None)
        if value is not None:
            cfg[key] = value
    if cli_args.log_dir is None:
        log_dir = Path.cwd() / "logs" / f"{experiment_name}_{cli_args.seed}"
    else:
        log_dir = Path(cli_args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(f"cuda:{cli_args.gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return ExperimentContext(
        cfg=cfg,
        device=device,
        log_dir=log_dir,
        metrics_path=log_dir / "metrics.json",
        cmd_args_path=log_dir / "cmd_args.json",
        best_ckpt_path=log_dir / "best.ckpt",
        pretrain_ckpt_path=log_dir / "pretrain_best.ckpt",
    )


def maybe_subset(dataset: Feeder, ratio: float, seed: int) -> Subset | Feeder:
    """Apply deterministic subset sampling for fast sanity checks."""
    if ratio >= 1.0:
        return dataset
    subset_size = max(1, int(len(dataset) * ratio))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False).tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    cfg: Dict[str, Any],
    data_root: Path,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create deterministic train, val, and test loaders."""
    data_path, train_label_path, val_label_path = ensure_data_paths(data_root)

    train_set = Feeder(
        data_path=str(data_path),
        label_path=str(train_label_path),
        modality=cfg["modality"],
        repeat=1,
    )
    val_set = Feeder(
        data_path=str(data_path),
        label_path=str(val_label_path),
        modality=cfg["modality"],
        repeat=1,
    )
    train_set = maybe_subset(train_set, cfg["subset_ratio"], seed)
    val_set = maybe_subset(val_set, cfg["subset_ratio"], seed + 1)
    test_set = val_set

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


def evaluate_denoising(
    model: DenoisingModel,
    loader: DataLoader,
    device: torch.device,
    num_train_timesteps: int,
) -> float:
    """Evaluate denoising loss on validation or test split."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_data, _labels, _indices in loader:
            clean_data = batch_data.squeeze(-1).to(device)
            noise = torch.randn_like(clean_data)
            timesteps = torch.randint(
                0, num_train_timesteps, (clean_data.shape[0],), device=device
            ).long()
            noisy_data = clean_data + noise
            preds = model(noisy_data, timesteps)
            # Compute denoising objective for representation pretraining.
            loss = F.mse_loss(preds, noise)
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)


def train_pretrain(
    ctx: ExperimentContext,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    metrics: Dict[str, Any],
) -> None:
    """Run denoising pretraining stage."""
    model = DenoisingModel(ctx.cfg).to(ctx.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctx.cfg["pretrain_lr"])
    best_val = float("inf")

    for epoch in range(1, int(ctx.cfg["pretrain_epochs"]) + 1):
        model.train()
        epoch_train_loss = 0.0
        batches = 0
        train_loop = tqdm(train_loader, desc=f"[Pretrain] Epoch {epoch}", leave=False)
        for batch_data, _labels, _indices in train_loop:
            clean_data = batch_data.squeeze(-1).to(ctx.device)
            noise = torch.randn_like(clean_data)
            timesteps = torch.randint(
                0, int(ctx.cfg["num_train_timesteps"]), (clean_data.shape[0],), device=ctx.device
            ).long()
            noisy_data = clean_data + noise
            optimizer.zero_grad()
            preds = model(noisy_data, timesteps)
            # Denoising loss drives representation learning in the encoder.
            loss = F.mse_loss(preds, noise)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            batches += 1

        train_loss = epoch_train_loss / max(1, batches)
        val_loss = evaluate_denoising(model, val_loader, ctx.device, int(ctx.cfg["num_train_timesteps"]))
        test_loss = evaluate_denoising(model, test_loader, ctx.device, int(ctx.cfg["num_train_timesteps"]))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
        }
        metrics["pretrain_epochs"].append(record)
        print(
            f"[Pretrain][Epoch {epoch}/{int(ctx.cfg['pretrain_epochs'])}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} test_loss={test_loss:.6f}"
        )

        # Save the best pretraining checkpoint by validation denoising loss.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ctx.pretrain_ckpt_path)


def evaluate_classifier(
    model: ClassifierModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate classifier loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_data, labels, _indices in loader:
            data = batch_data.squeeze(-1).to(device)
            labels = labels.long().to(device)
            logits = model(data)
            # Compute supervised classification loss.
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            # Compute accuracy for model selection and reporting.
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
    mean_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_samples)
    return mean_loss, acc


def train_finetune(
    ctx: ExperimentContext,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    metrics: Dict[str, Any],
) -> None:
    """Run classification finetuning stage."""
    model = ClassifierModel(ctx.cfg).to(ctx.device)
    if ctx.pretrain_ckpt_path.exists():
        pretrain_state = torch.load(ctx.pretrain_ckpt_path, map_location=ctx.device)
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in pretrain_state.items() if k in model_state and k.startswith(("encoder.", "time_embed."))
        }
        model_state.update(filtered)
        model.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(ctx.cfg["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ctx.cfg["finetune_lr"]),
        weight_decay=float(ctx.cfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(ctx.cfg["finetune_epochs"]))
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(1, int(ctx.cfg["finetune_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        train_loop = tqdm(train_loader, desc=f"[Finetune] Epoch {epoch}", leave=False)
        for batch_data, labels, _indices in train_loop:
            data = batch_data.squeeze(-1).to(ctx.device)
            labels = labels.long().to(ctx.device)
            optimizer.zero_grad()
            logits = model(data)
            # Compute task loss for supervised HAR objective.
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            preds = torch.argmax(logits, dim=1)
            # Track training accuracy to monitor optimization stability.
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader))
        train_acc = total_correct / max(1, total_samples)
        val_loss, val_acc = evaluate_classifier(model, val_loader, criterion, ctx.device)
        test_loss, test_acc = evaluate_classifier(model, test_loader, criterion, ctx.device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        metrics["finetune_epochs"].append(record)
        print(
            f"[Finetune][Epoch {epoch}/{int(ctx.cfg['finetune_epochs'])}] "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.8f}"
        )

        # Save the best model checkpoint according to validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), ctx.best_ckpt_path)

    metrics["summary"] = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
    }
    print(
        f"[Finetune][Best] epoch={best_epoch} "
        f"best_val_acc={best_val_acc:.4f} best_test_acc={best_test_acc:.4f}"
    )


def parse_cli_args() -> argparse.Namespace:
    """Parse common CLI arguments for all train scripts."""
    parser = argparse.ArgumentParser(description="UCLA ablation experiment runner")
    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root path",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--log_dir", type=str, default=None, help="Output log directory")
    parser.add_argument("--subset_ratio", type=float, default=None, help="Subset ratio")
    parser.add_argument("--modality", type=str, default=None, help="joint/bone/motion/bone-motion")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--pretrain_epochs", type=int, default=None, help="Pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=None, help="Finetuning epochs")
    parser.add_argument("--pretrain_lr", type=float, default=None, help="Pretraining learning rate")
    parser.add_argument("--finetune_lr", type=float, default=None, help="Finetuning learning rate")
    parser.add_argument("--model_channels", type=int, default=None, help="Base model channels")
    parser.add_argument("--mamba_d_state", type=int, default=None, help="Mamba state size")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout value")
    parser.add_argument("--transformer_nhead", type=int, default=None, help="Transformer heads")
    return parser.parse_args()


def run_experiment(experiment_name: str, config_path: Path) -> None:
    """Entry point used by each ablation train.py."""
    cli_args = parse_cli_args()
    base_config = load_simple_yaml(config_path)
    ctx = build_context(experiment_name, cli_args, base_config)
    set_seed(cli_args.seed)

    data_root = Path(cli_args.data_root).resolve()
    ensure_data_paths(data_root)
    train_loader, val_loader, test_loader = build_dataloaders(ctx.cfg, data_root, cli_args.seed)

    metrics: Dict[str, Any] = {
        "metadata": {
            "experiment_name": experiment_name,
            "seed": cli_args.seed,
            "data_root": str(data_root),
            "device": str(ctx.device),
            "config": ctx.cfg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "pretrain_epochs": [],
        "finetune_epochs": [],
        "summary": {},
    }

    train_pretrain(ctx, train_loader, val_loader, test_loader, metrics)
    train_finetune(ctx, train_loader, val_loader, test_loader, metrics)

    with ctx.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cmd_payload = {
        "args": vars(cli_args),
        "resolved_config": ctx.cfg,
        "experiment_name": experiment_name,
    }
    with ctx.cmd_args_path.open("w", encoding="utf-8") as f:
        json.dump(cmd_payload, f, indent=2)

    print(f"Finished experiment: {experiment_name}")
    print(f"Metrics saved to: {ctx.metrics_path}")
    print(f"Best checkpoint saved to: {ctx.best_ckpt_path}")
