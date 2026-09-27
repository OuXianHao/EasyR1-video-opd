"""Read-only tensor comparison of the completed 16-rank OPD checkpoints."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

BASE = Path("/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct")
ROOT = Path("/home/xianhao/projects/EasyR1/outputs/video_opd_1k_natural")
STEPS = (8, 16, 24, 31)
SELECTED = (
    "model.visual.patch_embed.proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.blocks.26.mlp.linear_fc2.weight",
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.18.mlp.down_proj.weight",
    "model.language_model.layers.35.self_attn.o_proj.weight",
)


def local(value):
    return value._local_tensor if hasattr(value, "_local_tensor") else value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    ckpts = {step: torch.load(
        ROOT / f"global_step_{step}" / "actor" / "model_world_size_16_rank_0.pt",
        map_location="cpu", weights_only=False, mmap=True,
    ) for step in STEPS}
    keys = list(ckpts[8])
    assert all(list(ckpts[step]) == keys for step in STEPS)
    counters = {step: {"changed": 0, "unchanged": 0, "base_delta_sq": 0.0, "base_norm_sq": 0.0}
                for step in STEPS}
    between = {f"{a}_to_{b}": {"changed": 0, "unchanged": 0, "delta_sq": 0.0}
               for a, b in zip(STEPS, STEPS[1:])}
    selected = {}
    unindexed = []
    for key in keys:
        if key not in index:
            unindexed.append(key)
            continue
        values = {step: local(ckpts[step][key]) for step in STEPS}
        n = values[8].shape[0]
        with safe_open(str(BASE / index[key]), framework="pt", device="cpu") as f:
            base = f.get_slice(key)[:n]
        assert all(value.shape == base.shape for value in values.values()), key
        base_float = base.float()
        norm_sq = float((base_float * base_float).sum())
        info = {"base_norm": math.sqrt(norm_sq), "shape": list(base.shape)}
        for step, value in values.items():
            delta = value.float() - base_float
            delta_sq = float((delta * delta).sum())
            changed = not torch.equal(value, base)
            counter = counters[step]
            counter["changed" if changed else "unchanged"] += 1
            counter["base_delta_sq"] += delta_sq
            counter["base_norm_sq"] += norm_sq
            if key in SELECTED:
                info[f"step_{step}_delta_norm"] = math.sqrt(delta_sq)
                info[f"step_{step}_relative_delta"] = math.sqrt(delta_sq / norm_sq) if norm_sq else None
        for a, b in zip(STEPS, STEPS[1:]):
            delta = values[b].float() - values[a].float()
            delta_sq = float((delta * delta).sum())
            target = between[f"{a}_to_{b}"]
            target["changed" if not torch.equal(values[a], values[b]) else "unchanged"] += 1
            target["delta_sq"] += delta_sq
            if key in SELECTED:
                info[f"{a}_to_{b}_delta_norm"] = math.sqrt(delta_sq)
        if key in SELECTED:
            selected[key] = info
    for step, item in counters.items():
        item["base_delta_norm"] = math.sqrt(item.pop("base_delta_sq"))
        item["base_norm"] = math.sqrt(item.pop("base_norm_sq"))
        item["relative_delta_norm"] = item["base_delta_norm"] / item["base_norm"]
    for item in between.values():
        item["delta_norm"] = math.sqrt(item.pop("delta_sq"))
    metadata = {}
    for step in STEPS:
        folder = ROOT / f"global_step_{step}"
        snapshot = torch.load(folder / "dataloader.pt", map_location="cpu", weights_only=False)
        extra = torch.load(folder / "actor" / "extra_state_world_size_16_rank_0.pt",
                           map_location="cpu", weights_only=False)
        optim = torch.load(folder / "actor" / "optim_world_size_16_rank_0.pt",
                           map_location="cpu", weights_only=False, mmap=True)
        first = optim["state"]["model.visual.patch_embed.proj.weight"]
        metadata[step] = {
            "dataloader_snapshot_step": snapshot["_snapshot"]["_snapshot_step"],
            "dataloader_samples_yielded": snapshot["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]["samples_yielded"],
            "dataloader_batches_yielded": snapshot["_snapshot"]["_main_snapshot"]["_sampler_iter_yielded"],
            "optimizer_parameter_states": len(optim["state"]),
            "optimizer_first_param_step": int(local(first["step"]).item()),
            "optimizer_first_param_exp_avg_norm": float(local(first["exp_avg"]).float().norm()),
            "lr_scheduler": extra["lr_scheduler"],
            "model_rank0_sha256": sha256(folder / "actor" / "model_world_size_16_rank_0.pt"),
            "model_shard_count": len(list((folder / "actor").glob("model_world_size_16_rank_*.pt"))),
            "optimizer_shard_count": len(list((folder / "actor").glob("optim_world_size_16_rank_*.pt"))),
        }
    result = {
        "base_student": str(BASE), "checkpoint_root": str(ROOT),
        "rank_compared": 0, "total_model_keys": len(keys), "unindexed_base_keys": unindexed,
        "base_comparison": counters, "between_checkpoints": between,
        "selected_parameters": selected, "checkpoint_metadata": metadata,
    }
    out = Path(__file__).with_name("training_integrity_checkpoint_evidence.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"base_comparison": counters, "between_checkpoints": between,
                      "selected_parameters": selected, "checkpoint_metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
