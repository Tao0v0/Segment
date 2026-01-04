import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    import torch
except Exception as e:  # pragma: no cover
    raise SystemExit(f"This script requires PyTorch. Import error: {e}")


@dataclass(frozen=True)
class PWLParams:
    name: str
    seg_point: torch.Tensor  # [7]
    coeff: torch.Tensor      # [8]
    intercept: torch.Tensor  # [8]
    clamp: Optional[Tuple[float, float]] = None


def _round_bits(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    if frac_bits <= 0:
        return torch.round(x)
    scale = 2 ** frac_bits
    return torch.round(x * scale) / scale


def _pwl_eval(x: torch.Tensor, seg_point: torch.Tensor, coeff: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
    # seg_point: strictly increasing length 7; bucketize gives idx in [0..7]
    idx = torch.bucketize(x, seg_point)
    return coeff[idx] * x + intercept[idx]


def _gelu_tanh_ref(x: torch.Tensor) -> torch.Tensor:
    k = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(k * (x + 0.044715 * x.pow(3))))


def _report_one(
    params: PWLParams,
    ref_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    seg_point_used: torch.Tensor,
    coeff_used: torch.Tensor,
    intercept_used: torch.Tensor,
) -> Dict[str, float]:
    y_ref = ref_fn(x)
    y_pwl = _pwl_eval(x, seg_point_used, coeff_used, intercept_used)
    if params.clamp is not None:
        y_pwl = torch.clamp(y_pwl, params.clamp[0], params.clamp[1])

    abs_err = (y_pwl - y_ref).abs()
    max_abs = abs_err.max()
    mean_abs = abs_err.mean()
    rmse = torch.sqrt((abs_err * abs_err).mean())
    idx = int(abs_err.argmax().item())
    x_at = float(x[idx].item())
    return {
        "max_abs": float(max_abs.item()),
        "mean_abs": float(mean_abs.item()),
        "rmse": float(rmse.item()),
        "x_at_max": x_at,
        "y_ref_at_max": float(y_ref[idx].item()),
        "y_pwl_at_max": float(y_pwl[idx].item()),
    }


def _print_stats(prefix: str, stats: Dict[str, float]) -> None:
    print(
        f"{prefix} max_abs={stats['max_abs']:.6e} mean_abs={stats['mean_abs']:.6e} rmse={stats['rmse']:.6e} "
        f"at x={stats['x_at_max']:.6f} (ref={stats['y_ref_at_max']:.6f}, pwl={stats['y_pwl_at_max']:.6f})"
    )


def _parse_scales(scales: str) -> List[float]:
    out: List[float] = []
    for part in scales.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("2**"):
            out.append(2 ** float(part[3:]))
        else:
            out.append(float(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PWL approximation error for GELU/tanh/sigmoid.")
    parser.add_argument("--xmin", type=float, default=-8.0)
    parser.add_argument("--xmax", type=float, default=8.0)
    parser.add_argument("--num", type=int, default=200001, help="Number of points for continuous scan.")
    parser.add_argument(
        "--scales",
        default="2**-3,2**-4,2**-5,2**-6",
        help="Comma list of quant steps for grid test (supports '2**-k').",
    )
    parser.add_argument("--device", default="", help="cpu / cuda / cuda:0 (default: auto)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    def _fallback_params() -> Dict[str, PWLParams]:
        return {
            "gelu": PWLParams(
                name="GELU(tanh-approx)",
                seg_point=torch.tensor([-3.015625, -2.203125, -0.890625, -0.421875, 0.03125, 0.625, 2.796875]),
                coeff=torch.tensor([-0.0, -0.03125, -0.109375, 0.046875, 0.359375, 0.75, 1.078125, 1.0]),
                intercept=torch.tensor([-0.0, -0.09375, -0.265625, -0.125, 0.0, -0.015625, -0.21875, -0.0]),
                clamp=None,
            ),
            "tanh": PWLParams(
                name="tanh",
                seg_point=torch.tensor([-2.125, -1.25, -0.625, 0.0, 0.625, 1.25, 2.125]),
                coeff=torch.tensor([0.0, 0.140625, 0.46875, 0.890625, 0.890625, 0.46875, 0.140625, 0.0]),
                intercept=torch.tensor([-1.0, -0.671875, -0.265625, 0.0, 0.0, 0.265625, 0.671875, 1.0]),
                clamp=(-1.0, 1.0),
            ),
            "sigmoid": PWLParams(
                name="sigmoid",
                seg_point=torch.tensor([-4.375, -2.0, -1.0, 0.0, 1.0, 2.0, 4.375]),
                coeff=torch.tensor([0.0, 0.046875, 0.15625, 0.234375, 0.234375, 0.15625, 0.046875, 0.0]),
                intercept=torch.tensor([0.0, 0.203125, 0.421875, 0.5, 0.5, 0.578125, 0.796875, 1.0]),
                clamp=(0.0, 1.0),
            ),
        }

    fallback = _fallback_params()

    def _from_module_or_fallback(name: str, module_obj) -> PWLParams:
        if all(hasattr(module_obj, k) for k in ("seg_point", "coeff", "intercept")):
            return PWLParams(
                name=fallback[name].name,
                seg_point=getattr(module_obj, "seg_point").detach().cpu(),
                coeff=getattr(module_obj, "coeff").detach().cpu(),
                intercept=getattr(module_obj, "intercept").detach().cpu(),
                clamp=fallback[name].clamp,
            )
        print(f"[warn] {type(module_obj).__name__} missing PWL params; using fallback constants for {fallback[name].name}.")
        return fallback[name]

    # Read current params from code (so you can tune seg_point/coeff/intercept and re-run).
    try:
        from diffusers_dpm.models.quantizer.quan_layer_annan_2 import QuanGELU, QuanSigmoid, QuanTanh

        gelu = QuanGELU(quan_input=False)
        tanh = QuanTanh(quan_input=False)
        sigmoid = QuanSigmoid(quan_input=False)

        gelu_params = _from_module_or_fallback("gelu", gelu)
        tanh_params = _from_module_or_fallback("tanh", tanh)
        sigmoid_params = _from_module_or_fallback("sigmoid", sigmoid)
    except Exception as e:
        print(f"[warn] Failed to import PWL modules from repo; using fallback constants. Error: {e}")
        gelu_params = fallback["gelu"]
        tanh_params = fallback["tanh"]
        sigmoid_params = fallback["sigmoid"]

    pwl_list: List[Tuple[PWLParams, Callable[[torch.Tensor], torch.Tensor]]] = [
        (gelu_params, _gelu_tanh_ref),
        (tanh_params, torch.tanh),
        (sigmoid_params, torch.sigmoid),
    ]

    x = torch.linspace(args.xmin, args.xmax, steps=args.num, device=device, dtype=dtype)
    scales = _parse_scales(args.scales)

    for params, ref_fn in pwl_list:
        print(f"\n=== {params.name} ===")

        # "Continuous" scan: use exact seg points (coeff/intercept are still rounded to 6 bits to match runtime).
        seg_point_used = params.seg_point.to(device=device, dtype=dtype)
        coeff_used = _round_bits(params.coeff.to(device=device, dtype=dtype), 6)
        intercept_used = _round_bits(params.intercept.to(device=device, dtype=dtype), 6)

        stats = _report_one(params, ref_fn, x, seg_point_used, coeff_used, intercept_used)
        _print_stats("[continuous]", stats)

        # "Quantized grid" scan: only evaluate x on multiples of scale, and apply seg-point rounding like runtime.
        for s in scales:
            if s <= 0:
                continue
            k_min = int(math.floor(args.xmin / s))
            k_max = int(math.ceil(args.xmax / s))
            xq = (torch.arange(k_min, k_max + 1, device=device, dtype=dtype) * s).clamp(args.xmin, args.xmax)

            # Runtime does: decimal_bit = int(-log2(scale)) (trunc towards zero).
            decimal_bit = int(-math.log2(s))
            seg_point_q = _round_bits(params.seg_point.to(device=device, dtype=dtype), decimal_bit)
            stats_q = _report_one(params, ref_fn, xq, seg_point_q, coeff_used, intercept_used)
            _print_stats(f"[grid step={s:g} bits={decimal_bit}]", stats_q)


if __name__ == "__main__":
    main()
