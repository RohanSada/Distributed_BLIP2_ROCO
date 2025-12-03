#!/usr/bin/env python

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image

from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)

from transformers import Blip2Processor, Blip2ForConditionalGeneration


# ============================================================
# Config
# ============================================================

@dataclass
class FSDPConfig:
    model_name_or_path: str
    data_root: str

    train_jsonl: str = "train.jsonl"
    val_jsonl: str = "val.jsonl"
    output_dir: str = "./checkpoints"

    num_epochs: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1
    learning_rate: float = 1e-5   # keep small for stability
    weight_decay: float = 0.01
    warmup_steps: int = 0

    max_text_len: int = 64
    num_workers: int = 4
    seed: int = 42
    log_every: int = 10
    save_every: int = 0           # 0 = only save best
    grad_clip: float = 1.0

    sharding_strategy: str = "full"   # "full" or "shard_grad"
    use_mixed_precision: bool = True  # bfloat16
    cpu_offload: bool = False


# ============================================================
# Dataset
# ============================================================

class RocoCaptionDataset(Dataset):
    """ROCO captioning dataset (jsonl).

    Each line:
      {"image": "train/img_0000001.png", "text": "a caption ..."}
    """

    def __init__(self, jsonl_path: str, data_root: str):
        self.samples = []
        self.data_root = data_root

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_rel = obj.get("image") or obj.get("image_path")
                txt = obj.get("text") or obj.get("caption") or ""
                if img_rel is None:
                    continue
                self.samples.append(
                    {
                        "image_path": os.path.join(data_root, img_rel),
                        "text": txt,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {jsonl_path}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        text = sample["text"]

        image = Image.open(img_path).convert("RGB")
        return {"image": image, "text": text}


# ============================================================
# Collate
# ============================================================

def build_collate_fn(processor: Blip2Processor, max_text_len: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]

        inputs = processor(
            images=images,
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )

        # Causal LM: labels = input_ids with pad tokens masked to -100
        labels = inputs["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        inputs["labels"] = labels

        # All tensors are CPU here; DataLoader will pin, and we move to GPU in the loop.
        return inputs

    return collate


# ============================================================
# Distributed setup / teardown
# ============================================================

def setup_distributed():
    """
    Works with either:
      - torchrun (RANK, WORLD_SIZE, LOCAL_RANK)
      - srun on Perlmutter (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID)
    """

    # Rank / world size
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
    else:
        rank = 0
        world_size = 1
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    # Local rank
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        os.environ["LOCAL_RANK"] = str(local_rank)
    else:
        local_rank = 0
        os.environ["LOCAL_RANK"] = "0"

    # ------- FIXED RENDEZVOUS PART -------
    # Use Slurm's launch node IP as master if available (multi-node safe)
    if "MASTER_ADDR" not in os.environ:
        master_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")
        if master_addr is None:
            # Fallback: first hostname from SLURM_NODELIST
            nodelist = os.environ.get("SLURM_NODELIST")
            if nodelist:
                # e.g., "nid[001000-001001]" -> "nid001000"
                base = nodelist.split(",")[0]
                if "[" in base:
                    prefix = base.split("[")[0]
                    inside = base.split("[")[1].split("]")[0]
                    first_idx = inside.split("-")[0]
                    master_addr = f"{prefix}{first_idx}"
                else:
                    master_addr = base
            else:
                master_addr = "127.0.0.1"  # last-resort single-node fallback
        os.environ["MASTER_ADDR"] = master_addr

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        host = os.uname().nodename
        print(
            f"[FSDP] init on host={host} | rank={rank}/{world_size} | "
            f"local_rank={local_rank} | device_index={torch.cuda.current_device()} | "
            f"MASTER_ADDR={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} | "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}",
            flush=True,
        )

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(rank: int) -> bool:
    return rank == 0


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Metrics logging helpers
# ============================================================

def save_run_metadata(cfg: FSDPConfig, world_size: int, rank: int):
    """Save a JSON with run + config info (rank 0 only)."""
    if not is_main_process(rank):
        return

    ensure_dir(cfg.output_dir)
    meta_path = os.path.join(cfg.output_dir, "run_meta.json")
    meta = {
        "config": cfg.__dict__,
        "world_size": int(world_size),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", None),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[FSDP] Saved run metadata to {meta_path}", flush=True)


def log_epoch_metrics(
    cfg: FSDPConfig,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    time_per_step: float,
    local_throughput: float,
    global_throughput: float,
    rank: int,
):
    """
    Append one JSON line per epoch to <output_dir>/metrics_epoch.jsonl (rank 0 only).
    """
    if not is_main_process(rank):
        return

    ensure_dir(cfg.output_dir)
    log_path = os.path.join(cfg.output_dir, "metrics_epoch.jsonl")

    record = {
        "epoch": int(epoch),
        "train_loss": float(train_metrics["train_loss"]),
        "val_loss": float(val_metrics["val_loss"]),
        "epoch_time": float(train_metrics["epoch_time"]),
        "eval_time": float(val_metrics["eval_time"]),
        "time_per_step": float(time_per_step),
        "train_steps": float(train_metrics["steps"]),
        "val_steps": float(val_metrics["steps"]),
        "samples_per_sec_per_gpu": float(local_throughput),
        "samples_per_sec_global": float(global_throughput),
        "peak_mem_gb_rank0": float(train_metrics["peak_mem_gb"]),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[FSDP] Logged epoch metrics to {log_path}", flush=True)


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(
    model: FSDP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: FSDPConfig,
    epoch: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch and return metrics:
      - train_loss (global avg)
      - epoch_time (seconds)
      - steps (number of optimizer steps)
      - peak_mem_gb (this rank; rank 0 used for logging)
    """
    model.train()
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    use_amp = cfg.use_mixed_precision
    autocast_dtype = torch.bfloat16

    total_loss = 0.0
    n_steps = 0

    # reset + measure peak memory over this epoch
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        # Protect against NaNs
        if torch.isnan(loss):
            if is_main_process(rank):
                print(f"[Epoch {epoch} Step {step}] NaN loss, skipping step", flush=True)
            continue

        loss.backward()

        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        # Global average loss
        with torch.no_grad():
            loss_detached = loss.detach()
            if world_size > 1:
                dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
                loss_detached /= world_size
            total_loss += loss_detached.item()
            n_steps += 1

        if is_main_process(rank) and step % cfg.log_every == 0:
            avg_loss = total_loss / max(1, n_steps)
            print(
                f"[Epoch {epoch} Step {step}/{len(train_loader)}] "
                f"Loss (global avg): {avg_loss:.4f}",
                flush=True,
            )

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    epoch_time = end_time - start_time
    avg_loss = total_loss / max(1, n_steps)

    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)

    return {
        "train_loss": float(avg_loss),
        "epoch_time": float(epoch_time),
        "steps": float(n_steps),
        "peak_mem_gb": float(peak_mem_gb),
    }


@torch.no_grad()
def evaluate(
    model: FSDP,
    val_loader: DataLoader,
    cfg: FSDPConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate for one epoch and return metrics:
      - val_loss (global avg)
      - eval_time (seconds)
      - steps (number of eval batches)
    """
    model.eval()
    use_amp = cfg.use_mixed_precision
    autocast_dtype = torch.bfloat16

    total_loss = 0.0
    n_steps = 0

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if use_amp:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss

        loss_detached = loss.detach()
        if world_size > 1:
            dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
            loss_detached /= world_size

        total_loss += loss_detached.item()
        n_steps += 1

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    eval_time = end_time - start_time

    avg_loss = total_loss / max(1, n_steps)
    if is_main_process(rank):
        print(f"[Eval] Loss (global avg): {avg_loss:.4f}", flush=True)

    return {
        "val_loss": float(avg_loss),
        "eval_time": float(eval_time),
        "steps": float(n_steps),
    }


def save_checkpoint(model: FSDP, optimizer, epoch: int, cfg: FSDPConfig, rank: int):
    if not is_main_process(rank):
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.output_dir, f"fsdp_epoch{epoch:02d}.pt")

    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        state_dict = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"[Checkpoint] Saved to {ckpt_path}", flush=True)


# ============================================================
# Config loading
# ============================================================

def load_config(path: str) -> FSDPConfig:
    with open(path, "r") as f:
        raw = json.load(f)

    fsdp_section = raw.get("fsdp", {})

    cfg = FSDPConfig(
        model_name_or_path=raw["model_name_or_path"],
        data_root=raw["data_root"],
        train_jsonl=raw.get("train_jsonl", "train.jsonl"),
        val_jsonl=raw.get("val_jsonl", "val.jsonl"),
        output_dir=raw.get("output_dir", "./checkpoints"),
        num_epochs=raw.get("num_epochs", 1),
        train_batch_size=raw.get("train_batch_size", 1),
        eval_batch_size=raw.get("eval_batch_size", 1),
        learning_rate=raw.get("learning_rate", 1e-5),
        weight_decay=raw.get("weight_decay", 0.01),
        warmup_steps=raw.get("warmup_steps", 0),
        max_text_len=raw.get("max_text_len", 64),
        num_workers=raw.get("num_workers", 4),
        seed=raw.get("seed", 42),
        log_every=raw.get("log_every", 10),
        save_every=raw.get("save_every", 0),
        grad_clip=raw.get("grad_clip", 1.0),
        sharding_strategy=fsdp_section.get("sharding_strategy", "full"),
        use_mixed_precision=fsdp_section.get("use_mixed_precision", True),
        cpu_offload=fsdp_section.get("cpu_offload", False),
    )
    return cfg


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    rank, world_size, local_rank, device = setup_distributed()
    set_seed(cfg.seed + rank)

    if is_main_process(rank):
        print(
            f"[FSDP] world_size={world_size}, rank={rank}, "
            f"local_rank={local_rank}, device_index={torch.cuda.current_device()}",
            flush=True,
        )
        print(f"[FSDP] Config: {cfg}", flush=True)

    # Save static run metadata
    save_run_metadata(cfg, world_size, rank)

    # Dataset / DataLoader
    train_path = os.path.join(cfg.data_root, cfg.train_jsonl)
    val_path = os.path.join(cfg.data_root, cfg.val_jsonl)

    train_dataset = RocoCaptionDataset(train_path, cfg.data_root)
    val_dataset = RocoCaptionDataset(val_path, cfg.data_root)

    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )

    processor = Blip2Processor.from_pretrained(cfg.model_name_or_path)
    collate_fn = build_collate_fn(processor, cfg.max_text_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if is_main_process(rank):
        if world_size > 1:
            print(
                f"[FSDP] num_train_batches_per_rank={len(train_loader)}, "
                f"num_training_steps={len(train_loader)}",
                flush=True,
            )
        else:
            print(
                f"[FSDP] num_train_batches={len(train_loader)}, "
                f"num_training_steps={len(train_loader)}",
                flush=True,
            )

    # Model
    if is_main_process(rank):
        print(
            f"[rank{rank}] Loading BLIP-2 model from {cfg.model_name_or_path}",
            flush=True,
        )

    # Load in full precision; FSDP + MixedPrecision will handle bf16
    model = Blip2ForConditionalGeneration.from_pretrained(
        cfg.model_name_or_path, torch_dtype=torch.float32
    )
    model.config.use_cache = False  # important for training

    model.to(device)

    # FSDP setup
    if cfg.sharding_strategy == "full":
        sharding = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "shard_grad":
        sharding = ShardingStrategy.SHARD_GRAD_OP
    else:
        raise ValueError(f"Unknown sharding_strategy: {cfg.sharding_strategy}")

    mp_policy = None
    if cfg.use_mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    cpu_offload = CPUOffload(offload_params=True) if cfg.cpu_offload else None
    device_index = torch.cuda.current_device()

    model = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload,
        device_id=device_index,
        use_orig_params=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(cfg.num_epochs):
        if is_main_process(rank):
            print(f"========== Epoch {epoch} ==========", flush=True)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, cfg, epoch, rank, world_size, device
        )
        val_metrics = evaluate(model, val_loader, cfg, rank, world_size, device)

        train_loss = train_metrics["train_loss"]
        val_loss = val_metrics["val_loss"]

        # Throughput calculations
        steps = max(1.0, train_metrics["steps"])
        time_per_step = train_metrics["epoch_time"] / steps
        local_throughput = cfg.train_batch_size / time_per_step
        global_throughput = (cfg.train_batch_size * world_size) / time_per_step

        if is_main_process(rank):
            print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"epoch_time={train_metrics['epoch_time']:.1f}s, "
                f"time/step={time_per_step:.4f}s, "
                f"samples/s_per_gpu={local_throughput:.1f}, "
                f"samples/s_global={global_throughput:.1f}, "
                f"peak_mem_gb_rank0={train_metrics['peak_mem_gb']:.2f}",
                flush=True,
            )

        # Log metrics to JSONL
        log_epoch_metrics(
            cfg=cfg,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            time_per_step=time_per_step,
            local_throughput=local_throughput,
            global_throughput=global_throughput,
            rank=rank,
        )

        '''
        # Checkpointing logic unchanged
        if cfg.save_every and (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(model, optimizer, epoch, cfg, rank)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, cfg, rank)
        '''
        
    cleanup_distributed()


if __name__ == "__main__":
    main()