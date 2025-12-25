import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Allow running from repo root OR from within `tools/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _unwrap_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _best_effort_align_to_keys(sd: Dict[str, torch.Tensor], target_keys) -> Dict[str, torch.Tensor]:
    candidates = [
        sd,
        _strip_prefix(sd, "module."),
        _strip_prefix(sd, "backbone."),
        _strip_prefix(_strip_prefix(sd, "module."), "backbone."),
        _strip_prefix(_strip_prefix(sd, "backbone."), "module."),
    ]
    best = sd
    best_hits = -1
    for cand in candidates:
        hits = sum(1 for k in cand.keys() if k in target_keys)
        if hits > best_hits:
            best_hits = hits
            best = cand
    return best


def convert_cmnext_noquant_to_quantizable(
    ref_sd: Dict[str, torch.Tensor], new_model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """Convert `semseg.models.backbones.cmnext_noquant.CMNeXt` weights -> `semseg.models.backbones.cmnext.CMNeXt`."""
    new_sd = new_model.state_dict()
    out: Dict[str, torch.Tensor] = {}

    # 1) Direct copies (same key + same shape).
    for k, v in ref_sd.items():
        if k in new_sd and tuple(v.shape) == tuple(new_sd[k].shape):
            out[k] = v

    # 2) Linear([O, I]) -> Conv1x1([O, I, 1, 1]) for matching keys.
    for k, v in ref_sd.items():
        if k not in new_sd:
            continue
        exp = new_sd[k]
        if v.ndim == 2 and exp.ndim == 4 and tuple(exp.shape[-2:]) == (1, 1) and tuple(v.shape) == tuple(exp.shape[:2]):
            out[k] = v.unsqueeze(-1).unsqueeze(-1)

    # 3) Attention kv -> k/v (and reshape to Conv1x1).
    for k, v in ref_sd.items():
        if k.endswith("attn.kv.weight") and v.ndim == 2:
            dim = v.shape[1]
            k_key = k.replace("attn.kv.weight", "attn.k.weight")
            v_key = k.replace("attn.kv.weight", "attn.v.weight")
            if k_key in new_sd:
                out[k_key] = v[:dim].unsqueeze(-1).unsqueeze(-1)
            if v_key in new_sd:
                out[v_key] = v[dim:].unsqueeze(-1).unsqueeze(-1)
        elif k.endswith("attn.kv.bias") and v.ndim == 1:
            dim = v.shape[0] // 2
            k_key = k.replace("attn.kv.bias", "attn.k.bias")
            v_key = k.replace("attn.kv.bias", "attn.v.bias")
            if k_key in new_sd:
                out[k_key] = v[:dim]
            if v_key in new_sd:
                out[v_key] = v[dim:]

    return out


def _tensor_to_uint8_gray(x: torch.Tensor, vmin: Optional[torch.Tensor] = None, vmax: Optional[torch.Tensor] = None) -> torch.Tensor:
    # x: [C,H,W] or [H,W], normalize with an optional shared range
    if x.ndim == 3:
        x = x.mean(dim=0)
    x = x.float()
    if vmin is None:
        vmin = x.min()
    if vmax is None:
        vmax = x.max()
    denom = (vmax - vmin).clamp_min(1e-12)
    x = (x - vmin) / denom
    return (x * 255.0).round().clamp(0, 255).to(torch.uint8)


def _maybe_save_png(t: torch.Tensor, path: Path, vmin: Optional[torch.Tensor] = None, vmax: Optional[torch.Tensor] = None) -> None:
    try:
        from PIL import Image
    except Exception:
        return

    img = _tensor_to_uint8_gray(t, vmin=vmin, vmax=vmax).cpu().numpy()
    Image.fromarray(img, mode="L").save(str(path))


def compare(
    model_name: str,
    height: int,
    width: int,
    batch: int,
    device: str,
    seed: int,
    tolerance: float,
    ckpt_path: Optional[str],
    save_dir: Optional[str],
    use_dummy_extras: bool,
) -> Tuple[float, bool]:
    from semseg.models.backbones.cmnext_noquant import CMNeXt as CMNeXtNoQuant
    from semseg.models.backbones.cmnext import CMNeXt as CMNeXtQuant

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 用单模态，确保两边都不会走 event/extra 分支（只比较 backbone 主路径）
    modals = ["img"]
    ref = CMNeXtNoQuant(model_name=model_name, modals=modals, with_events=False).to(dev)
    new = CMNeXtQuant(model_name=model_name, modals=modals, with_events=False, quantize=False).to(dev)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = _unwrap_checkpoint(ckpt)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")
        ckpt = _best_effort_align_to_keys(ckpt, set(ref.state_dict().keys()))
        msg = ref.load_state_dict(ckpt, strict=False)
        print("[ref] load_state_dict:", msg)

    # 把 ref 的参数映射到 new（kv->k/v, Linear->Conv1x1）
    mapped = convert_cmnext_noquant_to_quantizable(ref.state_dict(), new)
    msg = new.load_state_dict(mapped, strict=False)
    print("[new] load_state_dict:", msg)
    if msg.missing_keys:
        raise RuntimeError(f"[new] missing keys after conversion: {msg.missing_keys[:20]}")

    ref.eval()
    new.eval()

    rgb = torch.randn(batch, 3, height, width, device=dev)
    if use_dummy_extras:
        x = [rgb, torch.zeros_like(rgb), torch.zeros_like(rgb)]
    else:
        x = [rgb]

    with torch.no_grad():
        out_ref = ref(x)
        out_new = new(x)

    if isinstance(out_ref, tuple):
        # 保险处理：某些分支可能返回 (outs, outs_event)
        out_ref = out_ref[0]
    if isinstance(out_new, tuple):
        out_new = out_new[0]

    if len(out_ref) != len(out_new):
        raise RuntimeError(f"Output list length mismatch: ref={len(out_ref)} new={len(out_new)}")

    max_abs = 0.0
    ok = True
    for i, (a, b) in enumerate(zip(out_ref, out_new)):
        if a.shape != b.shape:
            raise RuntimeError(f"Stage {i} shape mismatch: ref={tuple(a.shape)} new={tuple(b.shape)}")
        diff = (a - b).abs()
        stage_max = diff.max().item()
        stage_mean = diff.mean().item()
        max_abs = max(max_abs, stage_max)
        stage_ok = stage_max <= tolerance
        ok = ok and stage_ok
        print(f"[stage {i}] max_abs={stage_max:.6e} mean_abs={stage_mean:.6e} ok={stage_ok}")

        if save_dir:
            save_root = Path(save_dir)
            save_root.mkdir(parents=True, exist_ok=True)
            a0 = a[0]
            b0 = b[0]
            vmin = torch.minimum(a0.min(), b0.min())
            vmax = torch.maximum(a0.max(), b0.max())
            _maybe_save_png(a0, save_root / f"ref_stage{i}.png", vmin=vmin, vmax=vmax)
            _maybe_save_png(b0, save_root / f"new_stage{i}.png", vmin=vmin, vmax=vmax)
            _maybe_save_png((a0 - b0).abs(), save_root / f"diff_stage{i}.png", vmin=torch.tensor(0.0, device=a0.device), vmax=None)

    return max_abs, ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CMNeXt backbones: cmnext_noquant vs cmnext(quantize=False).")
    parser.add_argument("--model-name", default="B2", choices=["B2", "B4", "B5"])
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="", help="e.g. cpu / cuda / cuda:0 (default: auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint (trained on cmnext_noquant) to load into ref.")
    parser.add_argument("--save-dir", default=None, help="Optional directory to save PNG visualizations.")
    parser.add_argument("--use-dummy-extras", action="store_true", help="Pass extra dummy tensors in x list (ignored).")
    args = parser.parse_args()

    max_abs, ok = compare(
        model_name=args.model_name,
        height=args.height,
        width=args.width,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        tolerance=args.tolerance,
        ckpt_path=args.ckpt,
        save_dir=args.save_dir,
        use_dummy_extras=args.use_dummy_extras,
    )
    print(f"[result] max_abs={max_abs:.6e} ok={ok}")


if __name__ == "__main__":
    main()
