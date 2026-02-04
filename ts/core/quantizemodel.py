# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

import itertools
import math
import time
from typing import Any, Dict, Optional, OrderedDict, Sequence, Tuple

import torch
from ts.core.misc import exp_golomb_nbins
from ts.core.misc import (
    MAX_AC_MAX_VAL,
    POSSIBLE_EXP_GOL_COUNT,
    POSSIBLE_Q_STEP,
    get_q_step_from_parameter_name,
)
from ts.core.quantizer import round_dgm, round_ste, ste_round_to_bitdepth
from torch import Tensor


_POOLKEY_TO_MODULE_NAME = {
    "ap": "arm",
    "up": "upsampling",
    "sp": "synthesis",
}


def _is_weight_tensor(x: Tensor) -> bool:
    return len(x.shape) >= 2


def _to_float(v: Any) -> float:
    if isinstance(v, (float, int)):
        return float(v)
    if torch.is_tensor(v):
        return float(v.item())
    return float(v)


def _to_int(v: Any) -> int:
    if isinstance(v, int):
        return int(v)
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def _as_step_tensor(step: float, like: Tensor) -> Tensor:
    return torch.as_tensor(step, device=like.device, dtype=like.dtype)


def _as_py_list(v: Any) -> list[Any]:
    """Convert optional iterables / torch tensors to a plain Python list.

    Note: do NOT use `v or []` when `v` can be a multi-element Tensor.
    """
    if v is None:
        return []
    if torch.is_tensor(v):
        return v.detach().flatten().tolist()
    return list(v)


def _iter_param_tensors(v: Any) -> list[torch.nn.Parameter]:
    """Best-effort: normalize various pool entries to a flat list of Parameters."""
    if v is None:
        return []
    if isinstance(v, torch.nn.ParameterList):
        return list(v)
    if isinstance(v, (list, tuple)):
        return [p for p in v if isinstance(p, torch.nn.Parameter)]
    if isinstance(v, torch.nn.Parameter):
        return [v]
    if torch.is_tensor(v):
        return []
    try:
        return [p for p in list(v) if isinstance(p, torch.nn.Parameter)]
    except Exception:
        return []


def _grad_energy_stats(params: Sequence[torch.nn.Parameter]) -> Dict[str, float]:
    """Return gradient stats aggregated over a param sequence.

    Uses sum(||g||_2^2) as energy (stable for ratios).
    """
    grad_sq = 0.0
    grad_abs = 0.0
    n_none = 0.0
    n_zero = 0.0
    n_tensors = float(len(params))
    n_elems = 0.0
    n_requires_grad = 0.0
    n_leaf = 0.0
    n_grad_present = 0.0
    for p in params:
        try:
            if bool(getattr(p, "requires_grad", False)):
                n_requires_grad += 1.0
        except Exception:
            pass
        try:
            if bool(getattr(p, "is_leaf", False)):
                n_leaf += 1.0
        except Exception:
            pass
        g = getattr(p, "grad", None)
        if g is None:
            n_none += 1.0
            continue
        if not torch.is_tensor(g):
            n_none += 1.0
            continue
        n_grad_present += 1.0
        if g.numel() == 0:
            n_zero += 1.0
            continue
        # cast to float for stable accumulation
        gf = g.detach().float()
        n_elems += float(gf.numel())
        grad_sq += float((gf * gf).sum().item())
        grad_abs += float(gf.abs().sum().item())
        try:
            if float(gf.abs().max().item()) == 0.0:
                n_zero += 1.0
        except Exception:
            pass
    return {
        "grad_sq": float(grad_sq),
        "grad_abs": float(grad_abs),
        "n_none": float(n_none),
        "n_zero": float(n_zero),
        "n_tensors": float(n_tensors),
        "n_elems": float(n_elems),
        "n_requires_grad": float(n_requires_grad),
        "n_leaf": float(n_leaf),
        "n_grad_present": float(n_grad_present),
    }


def _param_update_stats(
    params: Sequence[torch.nn.Parameter],
    before: Sequence[Tensor],
) -> Dict[str, float]:
    """Compare current params to a saved snapshot (detached clones)."""
    if len(params) != len(before):
        return {
            "base_l2": float("nan"),
            "delta_l2": float("nan"),
            "delta_mean_abs": float("nan"),
            "delta_max_abs": float("nan"),
            "delta_rel": float("nan"),
        }

    base_sq = 0.0
    delta_sq = 0.0
    delta_abs_sum = 0.0
    delta_abs_max = 0.0
    total_elems = 0.0

    for p, b in zip(params, before):
        cur = p.detach()
        bf = b.detach()
        # float for stable norms
        cur_f = cur.float()
        bf_f = bf.float()
        d = (cur_f - bf_f).abs()
        total_elems += float(d.numel())
        base_sq += float((bf_f * bf_f).sum().item())
        delta_sq += float((d * d).sum().item())
        delta_abs_sum += float(d.sum().item())
        if d.numel() > 0:
            delta_abs_max = max(delta_abs_max, float(d.max().item()))

    base_l2 = float(base_sq ** 0.5)
    delta_l2 = float(delta_sq ** 0.5)
    delta_mean_abs = float(delta_abs_sum / max(1.0, total_elems))
    delta_rel = float(delta_l2 / max(1e-12, base_l2))
    return {
        "base_l2": base_l2,
        "delta_l2": delta_l2,
        "delta_mean_abs": delta_mean_abs,
        "delta_max_abs": float(delta_abs_max),
        "delta_rel": delta_rel,
    }


def _fake_quant_param_list_ste(
    fp_param: torch.nn.ParameterList,
    q_step: Dict[str, float],
) -> Tuple[list[Tensor], list[Tensor]]:
    """Return (quantized_param_list, integer_symbol_list) using STE rounding."""
    q_list: list[Tensor] = []
    int_list: list[Tensor] = []
    for p in fp_param:
        is_w = _is_weight_tensor(p)
        step = _as_step_tensor(q_step["weight" if is_w else "bias"], p)
        scaled = p / step
        # w_int = round_ste(scaled)
        #使用更加高级的round_dgm
        w_int=round_dgm(scaled)
        w_q = w_int * step
        q_list.append(w_q)
        int_list.append(w_int.reshape(-1))
    return q_list, int_list


def _hard_quant_param_list(
    fp_param: torch.nn.ParameterList,
    q_step: Dict[str, float],
) -> Tuple[list[Tensor], Dict[str, list[Tensor]]]:
    """Hard quantize (no grad). Returns (quantized_param_list, int_by_type)."""
    q_list: list[Tensor] = []
    int_by_type: Dict[str, list[Tensor]] = {"weight": [], "bias": []}
    for p in fp_param:
        is_w = _is_weight_tensor(p)
        step = _as_step_tensor(q_step["weight" if is_w else "bias"], p)
        w_int = torch.round(p / step)
        if w_int.abs().max() > MAX_AC_MAX_VAL:
            return [], {"weight": [], "bias": []}
        q_list.append(w_int * step)
        int_by_type["weight" if is_w else "bias"].append(w_int.reshape(-1))
    return q_list, int_by_type


def _estimate_nn_bits_expgolomb(
    fp_param: torch.nn.ParameterList,
    q_step: Dict[str, float],
    expgol_cnt: Dict[str, Optional[int]],
) -> float:
    """Estimate NN parameter bits using exp_golomb_nbins with fixed counts."""
    _q_list, int_by_type = _hard_quant_param_list(fp_param, q_step)
    total_bits = 0.0
    for worb in ("weight", "bias"):
        if len(int_by_type[worb]) == 0:
            continue
        cnt = expgol_cnt.get(worb)
        if cnt is None:
            continue
        v = torch.cat(int_by_type[worb])
        total_bits += float(exp_golomb_nbins(v, count=int(cnt)).item())
    return total_bits


@torch.no_grad()
def _forward_for_test_with_params(op, grids, ap_q, up_q, sp_q) -> Tuple[Tensor, Tensor]:
    """Hardround forward with override params (keeps dp.pool untouched)."""
    op.set_to_eval()
    output, rate = op.gen.forward(
        grids,
        ap_q,
        up_q,
        sp_q,
        quantizer_noise_type="none",
        quantizer_type="hardround",
    )
    max_dynamic = 2 ** (op.param.bitdepth) - 1
    decoded_image = torch.round(output * max_dynamic) / max_dynamic
    return decoded_image, rate


def stageB_qat_fixed_quant(
    op,
    param,
    quant_config: Dict[str, Dict[str, Dict[str, Any]]],
    lmbda: float,
    steps: int = 200,
    lr: float = 1e-4,
    soft_round_temperature: float = 0.3,
    verbose_every: int = 50,
    ref: Optional[Tensor] = None,
    *,
    latent_quantizer_type: str = "ste",
    latent_quantizer_noise_type: str = "none",
    latent_noise_parameter: float = 1.0,
    train_grids: bool = True,
    lr_grids: Optional[float] = None,
    lr_nn: Optional[float] = None,
    debug_monitor: bool = False,
    debug_steps: Optional[Sequence[int]] = None,
    debug_every: int = 0,
    debug_prefix: str = "",
) -> Tuple[Any, Dict[str, Any]]:
    """Stage B: fixed-quant-params QAT (no learning of q_step / expgol_cnt).

    - NN params fake-quantized with STE using fixed q_step (from Stage A)
    - Distortion: MSE(output_q, ref)
    - Rate term (in loss): latent_bits_bpp (differentiable) + NN L1 proxy
      where NN L1 proxy = mean(|round(w_fp/q_step)|) with STE.

    For reporting RD curve, use:
      rate_est_bpp = latent_bits_bpp(hard) + nn_bits_expgolomb_est/npixels
    where nn_bits_expgolomb_est uses fixed exp_golomb_count (Stage A).

    Returns:
        param (updated in-place), history dict
    """
    if ref is None:
        raise ValueError(
            "stageB_qat_fixed_quant requires `ref` to be the original target image. "
            "(Previously this defaulted to a decoded reference; pass ref explicitly.)"
        )

    op.set_param(param)
    op.to_run()
    op.set_to_train()

    # Optimize latent grids + NN parameters (ap/up/sp). Do NOT replace Parameter objects.
    grids_params = _iter_param_tensors(param.pool.get("grids")) if bool(train_grids) else []
    nn_params = [
        *param.pool.get("ap", []),
        *param.pool.get("up", []),
        *param.pool.get("sp", []),
    ]
    train_params = [*grids_params, *nn_params]
    for p in train_params:
        p.requires_grad_(True)
    if not bool(train_grids):
        for p in _iter_param_tensors(param.pool.get("grids")):
            try:
                p.requires_grad_(False)
            except Exception:
                pass

    # Allow different learning rates for latent grids vs NN params.
    # Defaults keep legacy behavior (single lr).
    lr_grids_eff = float(lr if lr_grids is None else lr_grids)
    lr_nn_eff = float(lr if lr_nn is None else lr_nn)
    param_groups = []
    if len(grids_params) > 0:
        param_groups.append({"params": grids_params, "lr": lr_grids_eff})
    if len(nn_params) > 0:
        param_groups.append({"params": nn_params, "lr": lr_nn_eff})
    opt = torch.optim.Adam(param_groups if len(param_groups) > 0 else train_params, lr=lr)

    npixels = ref.numel() / ref.shape[1]
    grids = param.pool["grids"]

    # Auto-scale NN proxy to bpp scale (no new hyper-parameters).
    # We estimate an initial nn_bpp (expgolomb, using fixed StageA counts) and
    # match it to the initial nn_proxy magnitude: alpha = nn_bpp0 / nn_proxy0.
    nn_bits0 = 0.0
    if "ap" in quant_config:
        nn_bits0 += _estimate_nn_bits_expgolomb(param.pool["ap"], quant_config["ap"]["q_step"], quant_config["ap"]["expgol_cnt"])
    if "up" in quant_config:
        nn_bits0 += _estimate_nn_bits_expgolomb(param.pool["up"], quant_config["up"]["q_step"], quant_config["up"]["expgol_cnt"])
    if "sp" in quant_config:
        nn_bits0 += _estimate_nn_bits_expgolomb(param.pool["sp"], quant_config["sp"]["q_step"], quant_config["sp"]["expgol_cnt"])
    nn_bpp0 = float(nn_bits0 / npixels) if npixels else 0.0
    alpha = None  # lazily initialized at it=0 from nn_proxy

    hist: Dict[str, Any] = {
        "loss": [],
        "mse": [],
        "latent_bpp": [],
        "nn_proxy": [],
        "nn_proxy_scaled": [],
        "rate_term": [],
        "alpha": [],
        "debug": [],
    }

    # Debug sampling schedule (only used when debug_monitor=True)
    debug_step_set: Optional[set[int]] = None
    if debug_monitor:
        if debug_steps is not None:
            try:
                debug_step_set = {int(x) for x in debug_steps}
            except Exception:
                debug_step_set = None
        if debug_step_set is None and int(debug_every) <= 0:
            # default sparse sampling
            cand = [0, 1, 2, 5, 10, 20, 50, 100, int(steps) - 1]
            debug_step_set = {int(x) for x in cand if 0 <= int(x) < int(steps)}

    def _do_debug(it: int) -> bool:
        if not debug_monitor:
            return False
        if debug_step_set is not None:
            return int(it) in debug_step_set
        if int(debug_every) > 0:
            return (int(it) % int(debug_every) == 0) or (int(it) == int(steps) - 1)
        return False

    def _do_verbose(it: int) -> bool:
        if not verbose_every:
            return False
        return (int(it) % int(verbose_every) == 0) or (int(it) == int(steps) - 1)

    for it in range(int(steps)):
        do_debug = _do_debug(it)
        do_verbose = _do_verbose(it)
        do_snap = bool(do_debug or do_verbose)

        opt.zero_grad(set_to_none=True)

        ap_q, ap_int = _fake_quant_param_list_ste(param.pool["ap"], quant_config["ap"]["q_step"]) if "ap" in quant_config else (param.pool["ap"], [])
        up_q, up_int = _fake_quant_param_list_ste(param.pool["up"], quant_config["up"]["q_step"]) if "up" in quant_config else (param.pool["up"], [])
        sp_q, sp_int = _fake_quant_param_list_ste(param.pool["sp"], quant_config["sp"]["q_step"]) if "sp" in quant_config else (param.pool["sp"], [])

        out, rate = op.gen.forward(
            grids,
            ap_q,
            up_q,
            sp_q,
            quantizer_noise_type=str(latent_quantizer_noise_type),
            quantizer_type=str(latent_quantizer_type),
            soft_round_temperature=float(soft_round_temperature),
            noise_parameter=float(latent_noise_parameter),
        )

        out_q = ste_round_to_bitdepth(out, op.param.bitdepth)
        mse = torch.mean((out_q - ref) ** 2)
        # mse=torch.mean((out - ref) ** 2)
        latent_bpp = rate.sum() / npixels

        all_int = []
        if len(ap_int) > 0:
            all_int.append(torch.cat(ap_int))
        if len(up_int) > 0:
            all_int.append(torch.cat(up_int))
        if len(sp_int) > 0:
            all_int.append(torch.cat(sp_int))
        if len(all_int) == 0:
            nn_proxy = torch.zeros((), device=out.device, dtype=out.dtype)#l1
        else:
            # nn_proxy = torch.mean(torch.abs(torch.cat(all_int)))#l1
            nn_proxy = torch.mean(torch.log1p(torch.abs(torch.cat(all_int))))  # natural log


        if alpha is None:
            nn_proxy0 = float(nn_proxy.detach().item())
            alpha = float(nn_bpp0 / max(1e-12, nn_proxy0))
            print(f"[StageB] auto_scale: nn_bpp0={nn_bpp0:.6g} nn_proxy0={nn_proxy0:.6g} alpha={alpha:.6g}")

        nn_proxy_scaled = nn_proxy * float(alpha)
        rate_term = latent_bpp + nn_proxy_scaled
        # if it % 100 == 0 or it == steps - 1:
        #     print(
        #         f"it={it} mse={mse.item():.6g} latent_bpp={latent_bpp.item():.6g} "
        #         f"nn_proxy={nn_proxy.item():.6g} nn_proxy_scaled={nn_proxy_scaled.item():.6g} "
        #         f"alpha={float(alpha):.6g} rate_term={rate_term.item():.6g}"
        #     )
        loss = mse*10 + float(lmbda) * rate_term
        loss.backward()

        groups: Dict[str, list[torch.nn.Parameter]] | None = None
        grad_stats: Dict[str, Dict[str, float]] | None = None
        grad_frac: Dict[str, float] | None = None
        before_snap: Dict[str, list[Tensor]] | None = None
        upd_stats: Dict[str, Dict[str, float]] | None = None

        if do_snap:
            groups = {
                "grids": _iter_param_tensors(param.pool.get("grids")),
                "ap": _iter_param_tensors(param.pool.get("ap")),
                "up": _iter_param_tensors(param.pool.get("up")),
                "sp": _iter_param_tensors(param.pool.get("sp")),
            }
            before_snap = {k: [p.detach().clone() for p in v] for k, v in groups.items()}

        opt.step()

        if do_snap and groups is not None and before_snap is not None:
            upd_stats = {k: _param_update_stats(groups[k], before_snap[k]) for k in groups.keys()}

        if do_debug and groups is not None:
            grad_stats = {k: _grad_energy_stats(v) for k, v in groups.items()}
            total_grad_sq = sum(v.get("grad_sq", 0.0) for v in grad_stats.values())
            grad_frac = {
                k: (float(v.get("grad_sq", 0.0)) / max(1e-12, float(total_grad_sq)))
                for k, v in grad_stats.items()
            }

            rec = {
                "it": int(it),
                "graph": {
                    "out_requires_grad": bool(getattr(out, "requires_grad", False)),
                    "rate_requires_grad": bool(getattr(rate, "requires_grad", False)),
                    "mse_requires_grad": bool(getattr(mse, "requires_grad", False)),
                    "latent_bpp_requires_grad": bool(getattr(latent_bpp, "requires_grad", False)),
                    "nn_proxy_requires_grad": bool(getattr(nn_proxy, "requires_grad", False)),
                    "rate_term_requires_grad": bool(getattr(rate_term, "requires_grad", False)),
                },
                "grad_frac": grad_frac,
                "grad_stats": grad_stats,
                "update_stats": upd_stats,
            }
            hist["debug"].append(rec)

            tag = (debug_prefix + " ") if debug_prefix else ""
            print(
                "[QAT-MON] "
                + tag
                + (
                    f"it={int(it):05d} "
                    f"grad_frac(grids={grad_frac['grids']:.3f},ap={grad_frac['ap']:.3f},up={grad_frac['up']:.3f},sp={grad_frac['sp']:.3f}) "
                    f"grids_grad(n={int(grad_stats['grids'].get('n_tensors',0))},leaf={int(grad_stats['grids'].get('n_leaf',0))},req={int(grad_stats['grids'].get('n_requires_grad',0))},hasg={int(grad_stats['grids'].get('n_grad_present',0))}) "
                    f"graph(out_grad={int(bool(getattr(out,'requires_grad',False)))},rate_grad={int(bool(getattr(rate,'requires_grad',False)))},mse_grad={int(bool(getattr(mse,'requires_grad',False)))}) "
                    + (
                        ""
                        if upd_stats is None
                        else (
                            f"drel(grids={upd_stats['grids']['delta_rel']:.3g},ap={upd_stats['ap']['delta_rel']:.3g},up={upd_stats['up']['delta_rel']:.3g},sp={upd_stats['sp']['delta_rel']:.3g}) "
                            f"dl2(grids={upd_stats['grids']['delta_l2']:.3g},ap={upd_stats['ap']['delta_l2']:.3g},up={upd_stats['up']['delta_l2']:.3g},sp={upd_stats['sp']['delta_l2']:.3g})"
                        )
                    )
                )
            )

        hist["loss"].append(float(loss.detach().item()))
        hist["mse"].append(float(mse.detach().item()))
        hist["latent_bpp"].append(float(latent_bpp.detach().item()))
        hist["nn_proxy"].append(float(nn_proxy.detach().item()))
        hist["nn_proxy_scaled"].append(float(nn_proxy_scaled.detach().item()))
        hist["rate_term"].append(float(rate_term.detach().item()))
        hist["alpha"].append(float(alpha) if alpha is not None else 0.0)

        if do_verbose:
            upd = upd_stats
            if upd is None:
                upd = {
                    "grids": {"delta_l2": float("nan"), "delta_rel": float("nan")},
                    "ap": {"delta_l2": float("nan"), "delta_rel": float("nan")},
                    "up": {"delta_l2": float("nan"), "delta_rel": float("nan")},
                    "sp": {"delta_l2": float("nan"), "delta_rel": float("nan")},
                }

            d_grids = float(upd.get("grids", {}).get("delta_l2", float("nan")))
            d_ap = float(upd.get("ap", {}).get("delta_l2", float("nan")))
            d_up = float(upd.get("up", {}).get("delta_l2", float("nan")))
            d_sp = float(upd.get("sp", {}).get("delta_l2", float("nan")))
            f_grids = d_grids if math.isfinite(d_grids) else 0.0
            f_ap = d_ap if math.isfinite(d_ap) else 0.0
            f_up = d_up if math.isfinite(d_up) else 0.0
            f_sp = d_sp if math.isfinite(d_sp) else 0.0
            e_sum = float(f_grids * f_grids + f_ap * f_ap + f_up * f_up + f_sp * f_sp)
            denom = max(1e-12, e_sum)
            p_grids = 100.0 * (f_grids * f_grids) / denom if math.isfinite(d_grids) else float("nan")
            p_ap = 100.0 * (f_ap * f_ap) / denom if math.isfinite(d_ap) else float("nan")
            p_up = 100.0 * (f_up * f_up) / denom if math.isfinite(d_up) else float("nan")
            p_sp = 100.0 * (f_sp * f_sp) / denom if math.isfinite(d_sp) else float("nan")

            print(
                f"[StageB] it={it:05d} loss={hist['loss'][-1]:.6g} mse={hist['mse'][-1]:.6g} "
                f"latent_bpp={hist['latent_bpp'][-1]:.6g} nn_proxy={hist['nn_proxy'][-1]:.6g} "
                f"nn_proxy_scaled={hist['nn_proxy_scaled'][-1]:.6g} alpha={hist['alpha'][-1]:.6g} "
                f"upd_l2(g={d_grids:.3g},ap={d_ap:.3g},up={d_up:.3g},sp={d_sp:.3g}) "
                f"upd_pct(g={p_grids:.1f}%,ap={p_ap:.1f}%,up={p_up:.1f}%,sp={p_sp:.1f}%) "
                f"upd_rel(g={float(upd['grids'].get('delta_rel',float('nan'))):.3g},ap={float(upd['ap'].get('delta_rel',float('nan'))):.3g},up={float(upd['up'].get('delta_rel',float('nan'))):.3g},sp={float(upd['sp'].get('delta_rel',float('nan'))):.3g})"
            )

    # Final RD evaluation (hard quant + fixed expgolomb count)
    with torch.no_grad():
        ap_q_h, _ap_int = _hard_quant_param_list(param.pool["ap"], quant_config["ap"]["q_step"]) if "ap" in quant_config else (param.pool["ap"], {"weight": [], "bias": []})
        up_q_h, _up_int = _hard_quant_param_list(param.pool["up"], quant_config["up"]["q_step"]) if "up" in quant_config else (param.pool["up"], {"weight": [], "bias": []})
        sp_q_h, _sp_int = _hard_quant_param_list(param.pool["sp"], quant_config["sp"]["q_step"]) if "sp" in quant_config else (param.pool["sp"], {"weight": [], "bias": []})

        y_hat, rate_hat = _forward_for_test_with_params(op, grids, ap_q_h, up_q_h, sp_q_h)
        mse_eval = float(torch.mean((y_hat - ref) ** 2).item())
        latent_bpp_eval = float(rate_hat.sum().item() / npixels)
        nn_bits = 0.0
        if "ap" in quant_config:
            nn_bits += _estimate_nn_bits_expgolomb(param.pool["ap"], quant_config["ap"]["q_step"], quant_config["ap"]["expgol_cnt"])
        if "up" in quant_config:
            nn_bits += _estimate_nn_bits_expgolomb(param.pool["up"], quant_config["up"]["q_step"], quant_config["up"]["expgol_cnt"])
        if "sp" in quant_config:
            nn_bits += _estimate_nn_bits_expgolomb(param.pool["sp"], quant_config["sp"]["q_step"], quant_config["sp"]["expgol_cnt"])

        nn_bpp_eval = float(nn_bits / npixels)
        total_bpp_eval = float(latent_bpp_eval + nn_bpp_eval)

    hist["eval"] = {
        "mse": mse_eval,
        "latent_bpp": latent_bpp_eval,
        "nn_bpp_expgolomb": nn_bpp_eval,
        "total_bpp_est": total_bpp_eval,
        "lambda": float(lmbda),
    }
    return param, hist


@torch.no_grad()
def stageA_calibrate_quant_config(
    op,
    param,
    mse_err: float = 0.00005,
    pool_keys: Tuple[str, ...] = ("up", "ap", "sp"),
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Tensor]:
    """Stage A (PTQ / calibration): search and FIX quantization hyper-params.

        For each module (ap/up/sp), perform a grid-search over q_step (w/b) and
        exp-golomb count (w/b) under MSE constraint.

        IMPORTANT: quantizing NN parameters can change the latent rate returned by
        op.forward_for_test(). Therefore, the selection objective is the *total*
        bit-length (non-proxy):
            total_bits = latent_bits + nn_param_bits_expgolomb

    Returns:
        quant_config: dict keyed by pool key (ap/up/sp):
            {
              'q_step': {'weight': float, 'bias': float},
              'expgol_cnt': {'weight': int|None, 'bias': int|None},
            }
        ref: reference decoded image (y_fp) used for the MSE constraint.
    """
    op.set_param(param)
    op.to_run()
    y_ref, _rate_ref = op.forward_for_test()
    ref = y_ref.clone().detach()
    _latent_bits_ref = float(_rate_ref.sum().item())

    quant_config: Dict[str, Dict[str, Dict[str, Any]]] = {}
    npixels = ref.numel() / ref.shape[1]

    # Match quantize_model_no_ref_v2 ordering semantics:
    # iterate modules sorted by module_name (arm/synthesis/upsampling), not by (up/ap/sp).
    ordered_pool_keys = [pk for pk in pool_keys if pk in _POOLKEY_TO_MODULE_NAME]
    # ordered_pool_keys.sort(key=lambda pk: _POOLKEY_TO_MODULE_NAME[pk])

    for pool_key in ordered_pool_keys:
        print(f"=== StageA: quantizing model part={pool_key} ===")
        module_name = _POOLKEY_TO_MODULE_NAME.get(pool_key)
        if module_name is None:
            raise KeyError(f"Unknown pool key {pool_key}. Expected one of {list(_POOLKEY_TO_MODULE_NAME.keys())}.")

        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        if all_q_step is None or all_expgol_cnt is None:
            raise KeyError(f"Missing POSSIBLE_Q_STEP / POSSIBLE_EXP_GOL_COUNT for module {module_name}.")
        print(
            f"pool_key={pool_key} module={module_name} \n all_q_step:={all_q_step}\n all_expgol_cnt:={all_expgol_cnt} \n searching q_step and expgol_cnt..."
        )

        # Keep a reference to the current (FP) parameters for this module.
        # Previously selected modules are kept quantized in param.pool.
        fp_param = param.pool[pool_key]

        # Fixed NN bits from previously-quantized modules (do not change during this module search).
        fixed_nn_bits = 0.0
        for fixed_pk, fixed_cfg in quant_config.items():
            if fixed_pk == pool_key:
                continue
            fixed_nn_bits += _estimate_nn_bits_expgolomb(
                param.pool[fixed_pk],
                fixed_cfg["q_step"],
                fixed_cfg["expgol_cnt"],
            )

        best_total_bits = float("inf")
        best_q_step: Dict[str, float] = {"weight": 0.0, "bias": 0.0}
        best_cnt: Dict[str, Optional[int]] = {"weight": None, "bias": None}
        best_q_param: Optional[torch.nn.ParameterList] = None

        # Keep MSE constraint strict, but expand q_step candidates (smaller steps)
        # if nothing satisfies mse_err. This increases coding cost but keeps distortion bounded.
        weight_candidates = _as_py_list(all_q_step.get("weight"))
        bias_candidates = _as_py_list(all_q_step.get("bias"))
        base_w_steps = sorted({_to_float(v) for v in weight_candidates if _to_float(v) > 0.0})
        base_b_steps = sorted({_to_float(v) for v in bias_candidates if _to_float(v) > 0.0})
        if len(base_w_steps) == 0 or len(base_b_steps) == 0:
            raise RuntimeError(f"[StageA] Empty q_step candidate list for module={module_name} pool_key={pool_key}.")

        max_expand_pow = 12
        expand_pow = 0
        found_any = False
        for expand_pow in range(max_expand_pow + 1):
            # Add finer steps by halving the smallest known step.
            if expand_pow == 0:
                w_steps = base_w_steps
                b_steps = base_b_steps
            else:
                w_steps = sorted(set(base_w_steps + [base_w_steps[0] * (0.5**k) for k in range(1, expand_pow + 1)]))
                b_steps = sorted(set(base_b_steps + [base_b_steps[0] * (0.5**k) for k in range(1, expand_pow + 1)]))

            for q_step_w, q_step_b in itertools.product(w_steps, b_steps):
                current_q_step = {"weight": q_step_w, "bias": q_step_b}
                q_param, param_int = _quantize_parameters(fp_param, current_q_step)
                if q_param is None or param_int is None:
                    continue

                # Check output MSE constraint
                param.pool[pool_key] = q_param
                op.set_param(param)
                y_hat, _rate_hat = op.forward_for_test()
                mse = torch.mean((y_hat - ref) ** 2)
                if float(mse.item()) > 0.00005:
                    continue
                print(f"Testing[expand_pow={expand_pow}] q_step_w={q_step_w} q_step_b={q_step_b} => mse={float(mse.item()):.6g} (thr={mse_err:.6g})")
                if float(mse.item()) > float(mse_err):
                    continue

                found_any = True

                # NOTE: quantizing NN params may change latent rate.
                latent_bits_hat = float(_rate_hat.sum().item())

                # Compute true exp-golomb cost for parameters; choose best count per (w/b)
                nn_bits_cur = 0.0
                chosen_cnt: Dict[str, Optional[int]] = {"weight": None, "bias": None}
                for worb in ("weight", "bias"):
                    if len(param_int[worb]) == 0:
                        chosen_cnt[worb] = None
                        continue

                    v = torch.cat(param_int[worb])
                    cur_best_rate = float("inf")
                    cur_best_cnt: Optional[int] = None
                    for expgol_cnt in _as_py_list(all_expgol_cnt.get(worb)):
                        cur_rate = float(exp_golomb_nbins(v, count=int(_to_int(expgol_cnt))).item())
                        if cur_rate < cur_best_rate:
                            cur_best_rate = cur_rate
                            cur_best_cnt = _to_int(expgol_cnt)
                    nn_bits_cur += cur_best_rate
                    chosen_cnt[worb] = cur_best_cnt

                total_bits = float(latent_bits_hat + fixed_nn_bits + nn_bits_cur)

                if verbose:
                    print(
                        f"[StageA] {pool_key} q_step_w={_to_float(q_step_w):.6g} q_step_b={_to_float(q_step_b):.6g} "
                        f"mse={float(mse.item()):.6g} thr={float(mse_err):.6g} latent_bits={latent_bits_hat:.0f} "
                        f"nn_bits_fixed={fixed_nn_bits:.0f} nn_bits_cur={nn_bits_cur:.0f} total_bits={total_bits:.0f} "
                        f"cnt={chosen_cnt}"
                    )

                if total_bits < best_total_bits:
                    best_total_bits = total_bits
                    best_q_step = {"weight": _to_float(q_step_w), "bias": _to_float(q_step_b)}
                    best_cnt = chosen_cnt
                    best_q_param = q_param

            if found_any:
                break

        if best_q_param is None:
            raise RuntimeError(
                f"[StageA] No feasible quantization found for pool_key={pool_key} under mse_err={mse_err}. "
                f"Tried expanding q_step by halving min step up to 2^{max_expand_pow}. "
                f"Consider increasing mse_err, extending POSSIBLE_Q_STEP, or checking MAX_AC_MAX_VAL constraints."
            )

        # Keep (do not restore) the selected quantized parameters for this module.
        param.pool[pool_key] = best_q_param
        op.set_param(param)

        quant_config[pool_key] = {
            "q_step": best_q_step,
            "expgol_cnt": best_cnt,
        }

        if verbose:
            expand_msg = f" (expand step /2^{expand_pow})" if expand_pow else ""
            print(
                f"[StageA] Selected {pool_key}: q_step={best_q_step}, expgol_cnt={best_cnt}, "
                f"total_bits={best_total_bits:.0f}{expand_msg}"
            )

    return quant_config, ref


def _quantize_parameters(
    fp_param: torch.nn.ParameterList,
    q_step,
) -> Tuple[Optional[torch.nn.ParameterList], Optional[Dict[str, list[Tensor]]]]:

    q_param = torch.nn.ParameterList()
    param_int: Dict[str, list[Tensor]] = {"weight": [], "bias": []}
    for pid in range(len(fp_param)):
        is_weight = len(fp_param[pid].shape)>=2
        current_q_step = q_step["weight"] if is_weight else q_step["bias"]
        if _to_float(current_q_step) <= 0.0:
            return None, None
        sent_param = torch.round(fp_param[pid] / current_q_step)

        if sent_param.abs().max() > MAX_AC_MAX_VAL:
            #print( f"Sent param {pid} exceed MAX_AC_MAX_VAL! Q step {current_q_step} too small."  )
            return None, None

        # Keep quantized tensors as leaf Parameters so later optimizers/QAT can work.
        q_tensor = (sent_param * current_q_step).detach()
        q_param.append(torch.nn.Parameter(q_tensor))
        if is_weight:
            param_int["weight"].append(sent_param.view(-1))
        else:
            param_int["bias"].append(sent_param.view(-1))
        

    return q_param, param_int
'''这里就是码流的上界'''
def cal_nparam(params:torch.nn.ParameterList):
    cur_total = 0
    for pid in range(len(params)):
        param = params[pid]
        cur_size = param.numel()
        cur_total += cur_size
    return cur_total*32

@torch.no_grad()
def quantize_model_img(op, param, loss_func) :
    start_time = time.time()
    op.set_param(param)
    op.to_run()
    module_to_quantize = {'arm':param.pool['ap'],'upsampling':param.pool['up'],'synthesis':param.pool['sp']}
    bits_per_module = {'arm':cal_nparam(param.pool['ap']),'upsampling':cal_nparam(param.pool['up']),'synthesis':cal_nparam(param.pool['sp'])}
    quant_param: Dict[str, Any] = {'arm':None,'upsampling':None,'synthesis':None}
    y,bits = op.forward_for_test()
    print(bits_per_module)
    loss,mse,bpp = loss_func.cal(y,bits,0)
    print(loss.item(),mse.item(),bpp.item())
    for module_name, cur_module in sorted(module_to_quantize.items()):
        best_loss = 1e6
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        assert all_q_step is not None and all_expgol_cnt is not None
        fp_param = cur_module
        bits_base = 0.
        for pk in bits_per_module: 
            if not pk == module_name: 
                bits_base += bits_per_module[pk]
        best_q_step = {}
        final_best_expgol_cnt = {}
        print(module_name)
        for q_step_w, q_step_b in itertools.product(_as_py_list(all_q_step.get("weight")), _as_py_list(all_q_step.get("bias"))):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            #print(f'qstep:{q_step_w},{q_step_b}')
            current_q_step = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param, param_int = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None or param_int is None:   continue

            param.pool[module_name] = q_param

            y,bits = op.forward_for_test()

            best_expgol_cnt = {}
            net_bits = 0
            for worb in param_int.keys():
                if len(param_int[worb])==0: continue
                v =  torch.cat(param_int[worb])
                cur_best_rate = 1e8
                for expgol_cnt in _as_py_list(all_expgol_cnt.get(worb)):
                    cur_rate = exp_golomb_nbins(v, count=int(_to_int(expgol_cnt))).item()
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt
                        best_expgol_cnt[worb] = int(cur_best_expgol_cnt)
                net_bits += cur_best_rate
       

            loss,mse,bpp = loss_func.cal(y,bits,net_bits+bits_base)
            print(q_step_w, q_step_b,loss.item(),mse.item(),bpp.item(),net_bits,bits_base)

            # Store best quantization steps
            if loss < best_loss:
                best_loss = loss
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
                bits_per_module[module_name] = net_bits
                print(bits_per_module)

        quant_param[module_name] = {'best_q_step':best_q_step,'final_best_expgol_cnt':final_best_expgol_cnt}
       
    print(bits_per_module)
    time_nn_quantization = time.time() - start_time

    return param
def compare_two_nets(current_param,q_param,module_name):
            # 新增：比较q_param和当前pool中的参数
            total_diff = 0.0
            for q_p, c_p in zip(q_param, current_param):
                    diff = torch.sum(torch.abs(q_p - c_p)).item()
                    total_diff += diff
                    print(f"shape:{q_p.shape},Parameter diff: {diff:.6f}")
                
            print(f"Total absolute difference for {module_name}: {total_diff:.6f}")
@torch.no_grad()
def quantize_model_no_ref(op, param,mse_err=0.00005) :
    start_time = time.time()
    op.set_param(param)
    op.to_run()
    module_to_quantize = {'upsampling':param.pool['up'],'arm':param.pool['ap'],'synthesis':param.pool['sp']}
    bits_per_module = {'arm':cal_nparam(param.pool['ap']),'upsampling':cal_nparam(param.pool['up']),'synthesis':cal_nparam(param.pool['sp'])}
    quant_param: Dict[str, Any] = {'arm':None,'upsampling':None,'synthesis':None}
    name_translate={"arm":"ap","upsampling":"up","synthesis":"sp"}
    y,bits = op.forward_for_test()

    ref = y.clone().detach()
    npixels = ref.numel() / ref.shape[1]
    simple_latents=torch.sum(bits).item()/npixels
    print(f"before:{torch.sum(bits).item()/npixels},npixels:{npixels},bits_per_module:{bits_per_module}")
    final_mse=0.0
    before_bits_per_module=bits_per_module.copy()
    best_loss=1e6
    for module_name, cur_module in sorted(module_to_quantize.items()):
        #best_loss = Max_best_loss
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        assert all_q_step is not None and all_expgol_cnt is not None
        fp_param = cur_module
        bits_base = 0.
        for pk in bits_per_module: 
            if not pk == module_name: 
                bits_base += bits_per_module[pk]
        best_q_step = {}
        final_best_expgol_cnt = {}
        print(module_name)
        best_q_param=param.pool[name_translate[module_name]]
        for q_step_w, q_step_b in itertools.product(_as_py_list(all_q_step.get("weight")), _as_py_list(all_q_step.get("bias"))):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            #print(f'qstep:{q_step_w},{q_step_b}')
            current_q_step = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param, param_int = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None or param_int is None: continue
            #compare_two_nets(fp_param,q_param,module_name)
            param.pool[name_translate[module_name]] = q_param
            op.set_param(param)
            y,bits = op.forward_for_test()
            
            mse = torch.mean((y-ref)**2)
            print(f"module_name{module_name},q_step_W:{q_step_w},q_step_b:{q_step_b},mse:{mse}")
            if mse > mse_err: continue
            best_expgol_cnt = {}
            net_bits = 0
            for worb in param_int.keys():
                if len(param_int[worb])==0: continue
                v =  torch.cat(param_int[worb])
                cur_best_rate = 1e8
                for expgol_cnt in _as_py_list(all_expgol_cnt.get(worb)):
                    cur_rate = exp_golomb_nbins(v, count=int(_to_int(expgol_cnt))).item()
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt
                        best_expgol_cnt[worb] = int(cur_best_expgol_cnt)
                net_bits += cur_best_rate
       

            bpp = (net_bits+bits_base+torch.sum(bits).item())/npixels
            #print(f"q_step_w,q_step_b,bpp,net_bits,bits_base")
            print(q_step_w, q_step_b,bpp,net_bits,bits_base,torch.sum(bits).item())
            
            # Store best quantization steps
            if bpp < best_loss:
                best_loss = bpp
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
                bits_per_module[module_name] = net_bits
                print(f"new bpp {bpp},and {bits_per_module},mse:{mse},and ratio is {(torch.sum(bits).item()*100/npixels)/bpp}%")
                simple_latents=torch.sum(bits).item()/npixels
                final_mse=mse
                print(q_step_w, q_step_b,bpp,net_bits,bits_base)
                best_q_param=q_param
                if module_name=="synthesis":
                    print(fp_param[0][0],fp_param[0].shape)
                    print(q_param[0][0])
                    print(param_int['weight'][0][:6])
                    print(f"q_step_w,q_step_b:{q_step_w},{q_step_b}")
        param.pool[name_translate[module_name]]=best_q_param
        compare_two_nets(fp_param,param.pool[name_translate[module_name]],module_name)
        compare_two_nets(best_q_param,param.pool[name_translate[module_name]],f"{module_name}_best_q_param")
        quant_param[module_name] = {'best_q_step':best_q_step,'final_best_expgol_cnt':final_best_expgol_cnt}
    print(f"before_bpp{simple_latents},final bpp{best_loss};ratio:{(simple_latents/best_loss)*100}")   
    print("quant_param:",quant_param)
    print(f"before:{before_bits_per_module}")
    #print(f"after:{bits_per_module}")
    result_distribution={}
    for module_name, bits in bits_per_module.items():
        result_distribution[module_name]=bits/npixels
    result_distribution["grids"]=simple_latents
    print(f"after:{result_distribution}")

    print(f"param.pool.keys:{param.pool.keys()}")
    time_nn_quantization = time.time() - start_time
    op.set_param(param)
    y,bits = op.forward_for_test()
            
    mse = torch.mean((y-ref)**2)
    print(time_nn_quantization)
    final_mse_val = float(final_mse.item()) if torch.is_tensor(final_mse) else float(final_mse)
    return param,best_loss,simple_latents,time_nn_quantization,final_mse_val,mse.item(),ref,y,result_distribution

@torch.no_grad()
def quantize_model_no_ref_v2(op, param,mse_err=0.00005) :
    start_time = time.time()
    op.set_param(param)
    op.to_run()
    module_to_quantize = {'upsampling':param.pool['up'],'arm':param.pool['ap'],'synthesis':param.pool['sp']}
    bits_per_module = {'arm':cal_nparam(param.pool['ap']),'upsampling':cal_nparam(param.pool['up']),'synthesis':cal_nparam(param.pool['sp'])}
    quant_param: Dict[str, Any] = {'arm':None,'upsampling':None,'synthesis':None}
    name_translate={"arm":"ap","upsampling":"up","synthesis":"sp"}
    y,bits = op.forward_for_test()

    ref = y.clone().detach()
    npixels = ref.numel() / ref.shape[1]
    simple_latents=torch.sum(bits).item()/npixels
    print(f"before:{torch.sum(bits).item()/npixels},npixels:{npixels},bits_per_module:{bits_per_module}")
    final_mse=0.0
    before_bits_per_module=bits_per_module.copy()
    best_loss=1e6
    for module_name, cur_module in sorted(module_to_quantize.items()):
        #best_loss = Max_best_loss
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)
        assert all_q_step is not None and all_expgol_cnt is not None
        fp_param = cur_module
        bits_base = 0.
        for pk in bits_per_module: 
            if not pk == module_name: 
                bits_base += bits_per_module[pk]
        best_q_step = {}
        final_best_expgol_cnt = {}
        print(module_name)
        best_q_param=param.pool[name_translate[module_name]]
        for q_step_w, q_step_b in itertools.product(_as_py_list(all_q_step.get("weight")), _as_py_list(all_q_step.get("bias"))):
            # Reset full precision parameters, set the quantization step
            # and quantize the model.
            #print(f'qstep:{q_step_w},{q_step_b}')
            current_q_step = {"weight": q_step_w, "bias": q_step_b}

            # Reset full precision parameter before quantizing
            q_param, param_int = _quantize_parameters(fp_param, current_q_step)

            # Quantization has failed
            if q_param is None or param_int is None: continue
            #compare_two_nets(fp_param,q_param,module_name)
            param.pool[name_translate[module_name]] = q_param
            op.set_param(param)
            y,bits = op.forward_for_test()
            
            mse = torch.mean((y-ref)**2)
            print(f"module_name{module_name},q_step_W:{q_step_w},q_step_b:{q_step_b},mse:{mse}")
            if mse > mse_err: continue
            best_expgol_cnt = {}
            net_bits = 0
            for worb in param_int.keys():
                if len(param_int[worb])==0: continue
                v =  torch.cat(param_int[worb])
                cur_best_rate = 1e8
                for expgol_cnt in _as_py_list(all_expgol_cnt.get(worb)):
                    cur_rate = exp_golomb_nbins(v, count=int(_to_int(expgol_cnt))).item()
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt
                        best_expgol_cnt[worb] = int(cur_best_expgol_cnt)
                net_bits += cur_best_rate
       

            bpp = (net_bits+bits_base+torch.sum(bits).item())/npixels
            #print(f"q_step_w,q_step_b,bpp,net_bits,bits_base")
            print(q_step_w, q_step_b,bpp,net_bits,bits_base,torch.sum(bits).item())
            
            # Store best quantization steps
            if bpp < best_loss:
                best_loss = bpp
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
                bits_per_module[module_name] = net_bits
                print(f"new bpp {bpp},and {bits_per_module},mse:{mse},and ratio is {(torch.sum(bits).item()*100/npixels)/bpp}%")
                simple_latents=torch.sum(bits).item()/npixels
                final_mse=mse
                print(q_step_w, q_step_b,bpp,net_bits,bits_base)
                best_q_param=q_param
                if module_name=="synthesis":
                    print(fp_param[0][0],fp_param[0].shape)
                    print(q_param[0][0])
                    print(param_int['weight'][0][:6])
                    print(f"q_step_w,q_step_b:{q_step_w},{q_step_b}")
        param.pool[name_translate[module_name]]=best_q_param
        compare_two_nets(fp_param,param.pool[name_translate[module_name]],module_name)
        compare_two_nets(best_q_param,param.pool[name_translate[module_name]],f"{module_name}_best_q_param")
        quant_param[module_name] = {'best_q_step':best_q_step,'final_best_expgol_cnt':final_best_expgol_cnt}
    print(f"before_bpp{simple_latents},final bpp{best_loss};ratio:{(simple_latents/best_loss)*100}")   
    print("quant_param:",quant_param)
    print(f"before:{before_bits_per_module}")
    #print(f"after:{bits_per_module}")
    result_distribution={}
    for module_name, bits in bits_per_module.items():
        result_distribution[module_name]=bits/npixels
    result_distribution["grids"]=simple_latents
    print(f"after:{result_distribution}")

    print(f"param.pool.keys:{param.pool.keys()}")
    '''
    time_nn_quantization = time.time() - start_time
    op.set_param(param)
    y,bits = op.forward_for_test()
    mse = torch.mean((y-ref)**2)
    '''
    return best_loss,result_distribution