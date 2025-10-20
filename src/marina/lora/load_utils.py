import re
import torch

def _maybe_remap_key_for_lora(key: str, lora_sd_keys) -> str | None:
    # Direct Q/K/V/O (e.g., cross-attn) → add ".base."
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        if key.endswith(f"{proj}.weight"):
            cand = key.replace(f"{proj}.weight", f"{proj}.base.weight")
            return cand if cand in lora_sd_keys else None
        if key.endswith(f"{proj}.bias"):
            cand = key.replace(f"{proj}.bias", f"{proj}.base.bias")
            return cand if cand in lora_sd_keys else None
    # Final head fc → fc.base
    if key.endswith("fc.weight"):
        cand = key.replace("fc.weight", "fc.base.weight")
        return cand if cand in lora_sd_keys else None
    if key.endswith("fc.bias"):
        cand = key.replace("fc.bias", "fc.base.bias")
        return cand if cand in lora_sd_keys else None
    # Otherwise, keep verbatim if present
    return key if key in lora_sd_keys else None


def load_base_ckpt_into_lora_model(lora_model, ckpt_path: str, strict_shapes: bool = True):
    """
    Load a *base* SPECTRE checkpoint into a *LoRA* SPECTRE model.
    Handles:
      - direct Q/K/V/O → *.base.* mapping (cross-attn)
      - torch MHA self-attn: in_proj_weight/bias → split into q/k/v
      - fc → fc.base
    Ignores LoRA A/B tensors (A,B missing is expected).
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base_sd = raw.get("state_dict", raw)

    lora_sd = lora_model.state_dict()
    lora_keys = set(lora_sd.keys())
    mapped = {}
    skipped_shape = []

    for k, v in base_sd.items():
        # 1) Handle self-attn from torch MHA: in_proj_weight/bias → split
        if k.endswith("self_attn.in_proj_weight"):
            # v shape: (3*D, D) → q,k,v chunks
            Wq, Wk, Wv = torch.chunk(v, 3, dim=0)
            prefix = k[:-len("in_proj_weight")]  # ends with "...self_attn."
            kq = prefix + "q_proj.base.weight"
            kk = prefix + "k_proj.base.weight"
            kv = prefix + "v_proj.base.weight"
            if kq in lora_keys: mapped[kq] = Wq
            if kk in lora_keys: mapped[kk] = Wk
            if kv in lora_keys: mapped[kv] = Wv
            continue

        if k.endswith("self_attn.in_proj_bias"):
            # v shape: (3*D,) → q,k,v chunks
            bq, bk, bv = torch.chunk(v, 3, dim=0)
            prefix = k[:-len("in_proj_bias")]
            kq = prefix + "q_proj.base.bias"
            kk = prefix + "k_proj.base.bias"
            kv = prefix + "v_proj.base.bias"
            if kq in lora_keys: mapped[kq] = bq
            if kk in lora_keys: mapped[kk] = bk
            if kv in lora_keys: mapped[kv] = bv
            continue

        # 2) Direct out_proj from torch MHA self-attn (same name)
        if k.endswith("self_attn.out_proj.weight") or k.endswith("self_attn.out_proj.bias"):
            # Replace "...out_proj.{w/b}" → "...out_proj.base.{w/b}" if LoRA exists
            base_key = k.replace("out_proj.weight", "out_proj.base.weight").replace(
                "out_proj.bias", "out_proj.base.bias"
            )
            if base_key in lora_keys:
                mapped[base_key] = v
                continue

        # 3) All other params: try direct or Q/K/V/O/fc remap
        mk = _maybe_remap_key_for_lora(k, lora_keys)
        if mk is None:
            continue
        if strict_shapes and mk in lora_sd and lora_sd[mk].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(lora_sd[mk].shape)))
            continue
        mapped[mk] = v

    if not mapped:
        raise RuntimeError("No parameters mapped from base ckpt to LoRA model (check architecture/paths).")

    missing, unexpected = lora_model.load_state_dict(mapped, strict=False)
    return {
        "loaded_count": len(mapped),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "skipped_shape_mismatches": skipped_shape[:5],
        "some_missing": missing[:5],
        "some_unexpected": unexpected[:5],
    }