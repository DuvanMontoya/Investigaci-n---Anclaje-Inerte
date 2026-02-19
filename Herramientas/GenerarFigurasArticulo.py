#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenerarFigurasArticulo.py  ·  v2.0
================================

Genera figuras "paper-ready" y tablas LaTeX coherentes con el manuscrito
«Anclaje inerte y constantes de criba en familias cuadráticas de normas».

Lee CSVs tipo: csigma_inerte_D*.csv

Columnas esperadas (acepta sinónimos):
  D, R, H_R, C_sigma_inerte  [+ opcionalmente N_delta]

Salida (acorde a \\graphicspath en LaTeX):
  Figuras/Articulo/
    D{D}_stabilizacion.pdf
    D{D}_raw_vs_norm.pdf
    D{D}_Craw_vs_H.pdf
    D{D}_tail_residual.pdf
    Heegner_Ssing.pdf
    Heegner_Cinf_vs_Cbase.pdf
    Heegner_partial_products.pdf
    Heegner_summary_grid.pdf
    summary_tail_fits.csv
    summary_tail_fits_star.csv
    summary_tail_sd_comparison.csv
    summary_tail_grid_fits.csv
    summary_tail_uncertainty.csv
    table1_heegner.tex          ← Tabla 1 del paper (LaTeX)
    table2_uncertainty.tex      ← Tabla 2 del paper (LaTeX)
  Figuras/Articulo/Extra/
    Heegner_pred_small_inerts.pdf
  Figuras/Articulo/Diagnosticos/
    (diagnósticos adicionales si los activas)

Uso:
  python Herramientas/GenerarFigurasArticulo.py \\
      --input_dir Datos/Canonicos \\
      --bundle_dir Figuras/Articulo \\
      --R_tail_min 1000000 \\
      --jobs 4          # procesamiento paralelo por discriminante
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cycler import cycler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("paper_figures")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    bundle: Path
    extra: Path
    diagnostics: Path


@dataclass(frozen=True)
class PlotConfig:
    """Centraliza todas las constantes visuales."""
    # Colores principales (paleta sobria, print-friendly)
    color_main:    str = "#1E3A5F"
    color_accent:  str = "#8A4B08"
    color_band:    str = "#9BB9D6"
    color_zero:    str = "#6B7280"
    color_fit:     str = "#C0392B"
    # Tamaños de figura en pulgadas
    fig_w_single:  float = 3.46   # ≈ una columna en doble-columna (86 mm)
    fig_w_full:    float = 6.80   # columna completa / figura ancha
    fig_h_std:     float = 4.20
    fig_h_tall:    float = 5.80
    # Exportación
    dpi_screen:    int = 120
    dpi_save:      int = 300
    export_png:    bool = False
    # Ajuste de cola
    r_tail_min:    int = 1_000_000
    n_bins:        int = 200
    # Scatter sample size
    n_sample_scatter: int = 70_000


PCFG = PlotConfig()   # instancia global; puede ser reemplazada en main()

ANNOT_BOX = dict(
    boxstyle="round,pad=0.30",
    facecolor="white",
    alpha=0.92,
    linewidth=0.5,
    edgecolor="#D1D5DB",
)


@dataclass
class TailFit:
    """Resultado completo del ajuste de cola por WLS binneado."""
    D:            int
    model:        str          # "linear" | "quadratic"
    C_inf:        float
    a:            float
    b:            float        # NaN para modelo lineal
    se_C_inf:     float
    se_a:         float
    se_b:         float        # NaN para modelo lineal
    r2:           float        # coeficiente de determinación (bins)
    n_tail_pts:   int
    n_bins_eff:   int          # bins efectivos (≥ mín. puntos por bin)
    R_tail_min:   int
    R_min:        int
    R_max:        int

    @property
    def label(self) -> str:
        return f"Δ={self.D}, {self.model}"

    def predict(self, R: np.ndarray) -> np.ndarray:
        """Evalúa el ansatz de relajación sobre un array de R."""
        invlog = 1.0 / np.log(np.asarray(R, dtype=float))
        if self.model == "quadratic":
            return self.C_inf + self.a * invlog + self.b * invlog * invlog
        return self.C_inf + self.a * invlog


# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------

def set_mpl_defaults(cfg: PlotConfig = PCFG) -> None:
    plt.rcParams.update({
        "figure.dpi":         cfg.dpi_screen,
        "savefig.dpi":        cfg.dpi_save,
        "font.family":        "serif",
        "font.serif":         ["Latin Modern Roman", "CMU Serif",
                               "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "mathtext.rm":        "serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "legend.fontsize":    8.5,
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "axes.linewidth":     0.75,
        "lines.linewidth":    1.20,
        "axes.prop_cycle": cycler(color=[
            "#1E3A5F", "#8A4B08", "#2F6F4E",
            "#7A3B69", "#3E4A61", "#A3592A",
        ]),
        "axes.grid":          True,
        "grid.alpha":         0.22,
        "grid.color":         "#C4CDD6",
        "grid.linestyle":     "-",
        "grid.linewidth":     0.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "savefig.transparent": False,
        "legend.frameon":     True,
        "legend.framealpha":  0.90,
        "legend.edgecolor":   "#D1D5DB",
        "axes.unicode_minus": False,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(direction="out", length=3, width=0.75)


def save_fig(
    fig: plt.Figure,
    path_pdf: Path,
    cfg: PlotConfig = PCFG,
) -> None:
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(path_pdf, bbox_inches="tight")
    if cfg.export_png:
        fig.savefig(path_pdf.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    log.debug("  → %s", path_pdf.name)


# ---------------------------------------------------------------------------
# Aritmética de primos (cacheada)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def primes_upto(n: int) -> np.ndarray:
    """Criba de Eratóstenes cacheada."""
    if n < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = False
    return np.flatnonzero(sieve).astype(np.int64)


@lru_cache(maxsize=64)
def odd_ramified_primes(D: int) -> Tuple[int, ...]:
    """Primos impares ramificados del discriminante fundamental D."""
    t = abs(int(D))
    out: List[int] = []
    p = 3
    while p * p <= t:
        if t % p == 0:
            out.append(p)
            while t % p == 0:
                t //= p
        p += 2
    if t > 2 and (t % 2 == 1):
        out.append(int(t))
    return tuple(sorted(set(out)))


@lru_cache(maxsize=512)
def chi_legendre(D: int, p: int) -> int:
    """Símbolo de Legendre / Kronecker χ_D(p) para p primo impar."""
    if p == 2:
        raise ValueError("Usa chi_legendre solo para p impar.")
    if D % p == 0:
        return 0
    a = D % p
    t = pow(a, (p - 1) // 2, p)
    if t == 1:
        return 1
    if t == p - 1:
        return -1
    return 0


def vp_vector(R: np.ndarray, p: int) -> np.ndarray:
    """v_p(R[i]) para cada elemento del array (vectorizado)."""
    R = np.asarray(R, dtype=np.int64).copy()
    vp = np.zeros(len(R), dtype=np.int16)
    while True:
        mask = R % p == 0
        if not mask.any():
            break
        vp[mask] += 1
        R[mask] //= p
    return vp


def ramified_odd_multiplier(D: int, R: np.ndarray) -> np.ndarray:
    """∏_{p | Δ, p≥3} p^{v_p(R)}."""
    mult = np.ones(len(R), dtype=np.float64)
    for p in odd_ramified_primes(D):
        mult *= np.power(float(p), vp_vector(R, p))
    return mult


# ---------------------------------------------------------------------------
# Carga de CSV (robusta, con sinónimos y pre-cómputo)
# ---------------------------------------------------------------------------

_COL_SYNONYMS: Dict[str, List[str]] = {
    "D":              ["D", "Delta", "disc", "discriminant"],
    "R":              ["R", "Radius", "r"],
    "H_R":            ["H_R", "H", "HDelta", "H_delta", "H_Delta",
                       "HΔ", "H_inercial"],
    "C_sigma_inerte": ["C_sigma_inerte", "C_sigma", "C_sigma_delta",
                       "Cobs", "C_sigma_Delta", "C_sigma_D"],
    "N_delta":        ["N_delta", "N_Delta", "N", "Ncount", "N_norms"],
}


def load_csigma_csv(path: Path) -> pd.DataFrame:
    """
    Carga un CSV de C_{σ,Δ}(R) con normalización de columnas y
    pre-cómputo de todas las columnas derivadas necesarias.
    """
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Resolución de sinónimos
    resolved: Dict[str, str] = {}
    for canonical, variants in _COL_SYNONYMS.items():
        for v in variants:
            if v in df.columns:
                resolved[canonical] = v
                break

    missing = [c for c in ["D", "R", "H_R", "C_sigma_inerte"]
               if c not in resolved]
    if missing:
        raise ValueError(
            f"CSV {path.name}: columnas requeridas no encontradas {missing}. "
            f"Disponibles: {list(df.columns)}"
        )

    df = df.rename(columns={resolved[k]: k for k in resolved})
    df = df.dropna(subset=["D", "R", "H_R", "C_sigma_inerte"]).copy()
    df["D"] = df["D"].astype(int)
    df["R"] = df["R"].astype(np.int64)
    df = df[(df["R"] > 1) & (df["H_R"] > 0)].copy()

    # Columnas derivadas — pre-computadas una sola vez
    R_arr  = df["R"].to_numpy(dtype=np.float64)
    logR   = np.log(R_arr)
    D0     = int(df["D"].iloc[0])

    df["logR"]    = logR
    df["log10R"]  = logR / math.log(10.0)
    df["invlogR"] = 1.0 / logR
    df["C_raw"]   = df["C_sigma_inerte"].to_numpy() * df["H_R"].to_numpy()

    # Corrector ramificado impar: ∏ p^{v_p(R)}
    R_int = df["R"].to_numpy(dtype=np.int64)
    ram   = ramified_odd_multiplier(D0, R_int)
    df["ramified_odd_mult"] = ram
    df["H_R_star"]          = df["H_R"].to_numpy() / ram
    df["C_sigma_star"]      = df["C_sigma_inerte"].to_numpy() * ram

    # Valuaciones de primos inertes relevantes (útiles para fig_vp_effect)
    for p in (3, 5, 7):
        df[f"vp{p}"] = vp_vector(R_int, p)

    df = df.sort_values("R").reset_index(drop=True)

    elapsed = time.perf_counter() - t0
    log.info("  Cargado Δ=%4d  n=%7d  (%.2f s)", D0, len(df), elapsed)
    return df


# ---------------------------------------------------------------------------
# binned_band: 100% NumPy (mucho más rápido que pandas groupby)
# ---------------------------------------------------------------------------

def binned_band(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 350,
    qlo: float = 0.10,
    qhi: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve (x_mid, y_mean, y_lo, y_hi) en n_bins bins uniformes en x.
    Implementación 100% NumPy — sin pandas, ~10-50× más rápido para n > 500k.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) == 0:
        return (np.array([]), np.array([]),
                np.array([]), np.array([]))

    edges    = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_ids  = np.searchsorted(edges[1:-1], x, side="right")  # 0 … n_bins-1

    # Acumuladores usando np.bincount
    counts  = np.bincount(bin_ids, minlength=n_bins).astype(float)
    x_sum   = np.bincount(bin_ids, weights=x, minlength=n_bins)
    y_sum   = np.bincount(bin_ids, weights=y, minlength=n_bins)

    mask = counts > 0
    x_mid = np.where(mask, x_sum / np.where(mask, counts, 1), 0.0)
    y_mean = np.where(mask, y_sum / np.where(mask, counts, 1), np.nan)

    # Cuantiles por bin (sólo los bins no vacíos)
    y_lo  = np.full(n_bins, np.nan)
    y_hi  = np.full(n_bins, np.nan)
    sort_idx = np.argsort(bin_ids, kind="stable")
    sorted_bins = bin_ids[sort_idx]
    sorted_y    = y[sort_idx]
    boundaries  = np.searchsorted(sorted_bins,
                                  np.arange(n_bins + 1), side="left")
    for b in np.where(mask)[0]:
        sl = sorted_y[boundaries[b]: boundaries[b + 1]]
        if len(sl) >= 2:
            y_lo[b] = np.quantile(sl, qlo)
            y_hi[b] = np.quantile(sl, qhi)

    valid = mask & np.isfinite(y_lo) & np.isfinite(y_hi)
    return x_mid[valid], y_mean[valid], y_lo[valid], y_hi[valid]


# ---------------------------------------------------------------------------
# Ajuste de cola WLS binneado
# ---------------------------------------------------------------------------

def tail_fit_binned_wls(
    df: pd.DataFrame,
    R_tail_min: int,
    n_bins: int,
    y_col: str = "C_sigma_inerte",
    model: str = "linear",
) -> TailFit:
    """
    Ajuste de cola WLS (ponderado por tamaño de bin) con bins en cuantiles
    de log R.  Devuelve TailFit con C_inf, coeficientes, EE y R².
    """
    tail = df[df["R"] >= R_tail_min]
    if len(tail) < 500:
        raise ValueError(
            f"Δ={df['D'].iloc[0]}: cola demasiado corta "
            f"(n={len(tail)}) para R_tail_min={R_tail_min:,}."
        )
    if y_col not in tail.columns:
        raise ValueError(
            f"Δ={df['D'].iloc[0]}: columna '{y_col}' no encontrada."
        )

    logR = tail["logR"].to_numpy(dtype=np.float64)
    y    = tail[y_col].to_numpy(dtype=np.float64)

    # Bins por cuantiles de log R (robusto ante heterocedasticidad)
    qs    = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(logR, qs))
    if len(edges) < 6:
        edges = np.linspace(logR.min(), logR.max(), min(n_bins + 1, 60))

    bin_ids   = np.digitize(logR, edges[1:-1], right=True)
    uniq_bins = np.unique(bin_ids)

    R_arr = tail["R"].to_numpy(dtype=np.float64)
    R_mean  = np.array([R_arr[bin_ids == b].mean()  for b in uniq_bins])
    y_mean  = np.array([y[bin_ids == b].mean()       for b in uniq_bins])
    weights = np.array([(bin_ids == b).sum()          for b in uniq_bins],
                       dtype=np.float64)

    x1 = 1.0 / np.log(R_mean)
    model_key = str(model).strip().lower()
    if model_key in ("linear", "lin"):
        X = np.column_stack([np.ones_like(x1), x1])
        model_name = "linear"
    elif model_key in ("quadratic", "quad"):
        X = np.column_stack([np.ones_like(x1), x1, x1 * x1])
        model_name = "quadratic"
    else:
        raise ValueError(f"Modelo no soportado: {model!r}")

    p = X.shape[1]
    n = len(y_mean)
    if n <= p:
        raise ValueError(
            f"Δ={df['D'].iloc[0]}: bins insuficientes ({n}) "
            f"para modelo {model_name}. Aumenta cola o reduce bins."
        )

    # WLS
    W     = np.diag(weights)
    XtWX  = X.T @ W @ X
    XtWy  = X.T @ (weights * y_mean)
    beta  = np.linalg.solve(XtWX, XtWy)
    yhat  = X @ beta
    resid = y_mean - yhat

    # Varianza residual (grados de libertad = n_bins - p)
    dof   = max(1.0, n - p)
    s2    = float((weights * resid * resid).sum() / dof)
    cov   = s2 * np.linalg.inv(XtWX)
    se    = np.sqrt(np.maximum(0.0, np.diag(cov)))

    # R² ponderado
    y_wbar = float((weights * y_mean).sum() / weights.sum())
    ss_tot = float((weights * (y_mean - y_wbar) ** 2).sum())
    ss_res = float((weights * resid * resid).sum())
    r2     = 1.0 - ss_res / max(ss_tot, 1e-300)

    D = int(df["D"].iloc[0])
    a, b     = float(beta[1]), float(beta[2]) if model_name == "quadratic" else float("nan")
    se_a     = float(se[1])
    se_b     = float(se[2]) if model_name == "quadratic" else float("nan")

    return TailFit(
        D=D, model=model_name,
        C_inf=float(beta[0]), a=a, b=b,
        se_C_inf=float(se[0]), se_a=se_a, se_b=se_b,
        r2=float(r2),
        n_tail_pts=int(len(tail)), n_bins_eff=int(n),
        R_tail_min=int(R_tail_min),
        R_min=int(tail["R"].min()), R_max=int(tail["R"].max()),
    )


def parse_tail_grid(grid_arg: str, r_ref: int) -> List[int]:
    vals: List[int] = [r_ref]
    for tok in str(grid_arg).replace(";", ",").split(","):
        t = tok.strip()
        if t:
            vals.append(int(t))
    return sorted(set(v for v in vals if v > 0))


# ---------------------------------------------------------------------------
# Baseline de Heegner
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015328606


def C_base(D: int) -> float:
    """C_base(Δ) = 4π e^{-γ} / (w(Δ) √|Δ|)  para discriminantes de Heegner."""
    w = 6 if D == -3 else (4 if D == -4 else 2)
    return float(4.0 * math.pi * math.exp(-_EULER_GAMMA) / (w * math.sqrt(abs(D))))


# ---------------------------------------------------------------------------
# Figura 1: estabilización de C_{σ,Δ}(R)
# ---------------------------------------------------------------------------

def fig_stabilization(
    df: pd.DataFrame,
    fit: TailFit,
    out_pdf: Path,
    cfg: PlotConfig = PCFG,
) -> None:
    x = df["log10R"].to_numpy()
    y = df["C_sigma_inerte"].to_numpy()
    xmid, ymean, ylo, yhi = binned_band(x, y, n_bins=380)

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    ax.plot(xmid, ymean, color=cfg.color_main, lw=1.2)
    ax.fill_between(xmid, ylo, yhi, color=cfg.color_band, alpha=0.22)

    R_line = np.logspace(float(x.min()), float(x.max()), 400)
    ax.plot(np.log10(R_line), fit.predict(R_line),
            ls="--", color=cfg.color_fit, lw=1.1, label="Ajuste de cola")
    ax.axhline(fit.C_inf, ls=":", lw=0.9, color=cfg.color_zero,
               label=rf"$C_\infty={fit.C_inf:.5f}$")

    ax.set_xlabel(r"$\log_{10} R$")
    ax.set_ylabel(r"$C_{\sigma,\Delta}(R)$")
    ax.set_title(rf"$\Delta={fit.D}$: estabilización de $C_{{\sigma,\Delta}}(R)$")
    ax.legend(frameon=True, fontsize=8)
    style_axis(ax)

    ax.text(
        0.02, 0.98,
        rf"Cola: $R \geq {fit.R_tail_min:,}$" + "\n"
        + rf"$C_\infty = {fit.C_inf:.5f}$, $a = {fit.a:.3f}$" + "\n"
        + rf"$R^2 = {fit.r2:.5f}$",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5, bbox=ANNOT_BOX,
    )
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Figura 2: raw vs normalizado (doble panel — Fig. 3 del paper)
# ---------------------------------------------------------------------------

def fig_raw_vs_norm(
    df: pd.DataFrame,
    fit: TailFit,
    out_pdf: Path,
    cfg: PlotConfig = PCFG,
) -> None:
    x     = df["log10R"].to_numpy()
    Craw  = df["C_raw"].to_numpy()
    Csig  = df["C_sigma_inerte"].to_numpy()

    xr, yr, ylor, yhir = binned_band(x, Craw, n_bins=360)
    xs, ys, ylos, yhis = binned_band(x, Csig, n_bins=360)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(cfg.fig_w_full, cfg.fig_h_tall),
                                   sharex=True)
    ax1.plot(xr, yr, color=cfg.color_main, lw=1.2)
    ax1.fill_between(xr, ylor, yhir, color=cfg.color_band, alpha=0.20)
    ax1.set_ylabel(r"$C_{\rm raw}(R)=N_\Delta(R)\log R/R$")
    ax1.set_title(
        rf"$\Delta={fit.D}$: anclaje inerte (crudo vs normalizado)")
    style_axis(ax1)

    R_line = np.logspace(float(x.min()), float(x.max()), 400)
    ax2.plot(xs, ys, color=cfg.color_main, lw=1.2)
    ax2.fill_between(xs, ylos, yhis, color=cfg.color_band, alpha=0.20)
    ax2.plot(np.log10(R_line), fit.predict(R_line),
             ls="--", color=cfg.color_fit, lw=1.1)
    ax2.axhline(fit.C_inf, ls=":", lw=0.9, color=cfg.color_zero)
    ax2.set_xlabel(r"$\log_{10} R$")
    ax2.set_ylabel(r"$C_{\sigma,\Delta}(R)$")
    style_axis(ax2)

    ax2.text(
        0.02, 0.98,
        rf"$C_\infty = {fit.C_inf:.5f}$, $a = {fit.a:.3f}$",
        transform=ax2.transAxes, va="top", ha="left",
        fontsize=8.5, bbox=ANNOT_BOX,
    )
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Figura 3: colapso lineal C_raw vs H_Δ(R)  — Fig. 7 del paper
# ---------------------------------------------------------------------------

def fig_Craw_vs_H(
    df: pd.DataFrame,
    fit: TailFit,
    out_pdf: Path,
    R_tail_min: int,
    n_sample: int = 70_000,
    seed: int = 0,
    cfg: PlotConfig = PCFG,
) -> None:
    tail = df[df["R"] >= R_tail_min]
    if len(tail) == 0:
        log.warning("Δ=%d: sin datos en cola para fig_Craw_vs_H.", fit.D)
        return
    samp = tail.sample(n=min(n_sample, len(tail)), random_state=seed)
    H    = samp["H_R"].to_numpy(dtype=np.float64)
    Craw = samp["C_raw"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    ax.scatter(H, Craw, s=2, alpha=0.12, color=cfg.color_main, rasterized=True)

    H_line = np.linspace(H.min(), H.max(), 300)
    ax.plot(H_line, fit.C_inf * H_line, ls="--", color=cfg.color_fit, lw=1.2,
            label=rf"Pendiente $C_\infty={fit.C_inf:.4f}$")

    ax.set_xlabel(r"$H_\Delta(R)$")
    ax.set_ylabel(r"$C_{\rm raw}(R)=N_\Delta(R)\log R/R$")
    ax.set_title(
        rf"$\Delta={fit.D}$: colapso lineal en cola ($R\geq {R_tail_min:,}$)")
    ax.legend(frameon=True, fontsize=8)
    style_axis(ax)
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Figura 4: residuales del ajuste de cola  — Fig. 11 del paper
# ---------------------------------------------------------------------------

def fig_tail_residual(
    df: pd.DataFrame,
    fit: TailFit,
    out_pdf: Path,
    R_tail_min: int,
    cfg: PlotConfig = PCFG,
) -> None:
    tail = df[df["R"] >= R_tail_min].copy()
    R    = tail["R"].to_numpy(dtype=np.float64)
    y    = tail["C_sigma_inerte"].to_numpy(dtype=np.float64)
    resid = y - fit.predict(R)
    x = np.log10(R)

    xmid, rmean, rlo, rhi = binned_band(x, resid, n_bins=320)

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    ax.plot(xmid, rmean, color=cfg.color_main, lw=1.2)
    ax.fill_between(xmid, rlo, rhi, color=cfg.color_band, alpha=0.20)
    ax.axhline(0.0, ls="--", lw=1.0, color=cfg.color_zero)

    ax.set_xlabel(r"$\log_{10} R$ (cola)")
    ax.set_ylabel(r"Residuo $C_{\sigma,\Delta}(R)-(C_\infty+a/\log R)$")
    ax.set_title(
        rf"$\Delta={fit.D}$: diagnóstico de residuales (cola, $R^2={fit.r2:.5f}$)")
    style_axis(ax)
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Grid resumen: todos los Δ en una sola figura (9 paneles) — Fig. 5 del paper
# ---------------------------------------------------------------------------

def fig_summary_grid(
    sumdfs: Dict[int, pd.DataFrame],
    fits_by_D: Dict[int, TailFit],
    out_pdf: Path,
    cfg: PlotConfig = PCFG,
) -> None:
    Ds_sorted = sorted(sumdfs.keys(), key=abs)
    ncols = 3
    nrows = math.ceil(len(Ds_sorted) / ncols)
    fig   = plt.figure(figsize=(cfg.fig_w_full * 1.05, nrows * 2.50))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig,
                              hspace=0.38, wspace=0.32)

    for idx, D in enumerate(Ds_sorted):
        r, c = divmod(idx, ncols)
        ax   = fig.add_subplot(gs[r, c])
        df   = sumdfs[D]
        fit  = fits_by_D[D]

        x = df["log10R"].to_numpy()
        y = df["C_sigma_inerte"].to_numpy()
        xmid, ymean, ylo, yhi = binned_band(x, y, n_bins=250)

        ax.plot(xmid, ymean, color=cfg.color_main, lw=0.9)
        ax.fill_between(xmid, ylo, yhi, color=cfg.color_band, alpha=0.20)

        R_line = np.logspace(float(x.min()), float(x.max()), 250)
        ax.plot(np.log10(R_line), fit.predict(R_line),
                ls="--", color=cfg.color_fit, lw=0.85)
        ax.axhline(fit.C_inf, ls=":", lw=0.7, color=cfg.color_zero)

        ax.set_title(rf"$\Delta={D}$", fontsize=9)
        ax.text(
            0.96, 0.96,
            rf"$C_\infty\!=\!{fit.C_inf:.4f}$" + "\n"
            + rf"$a\!=\!{fit.a:.3f}$",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=6.5, bbox={**ANNOT_BOX, "pad": 0.20},
        )
        if c == 0:
            ax.set_ylabel(r"$C_{\sigma,\Delta}(R)$", fontsize=8)
        if r == nrows - 1:
            ax.set_xlabel(r"$\log_{10} R$", fontsize=8)
        ax.tick_params(labelsize=7)
        style_axis(ax)

    fig.suptitle(
        r"Estabilización de $C_{\sigma,\Delta}(R)$ — discriminantes de Heegner"
        + f"\n(Cola: $R\\geq {list(fits_by_D.values())[0].R_tail_min:,}$,"
          f"  ajuste lineal $C_\\infty + a/\\log R$)",
        fontsize=10, y=1.01,
    )
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Figuras comparativas Heegner
# ---------------------------------------------------------------------------

def _summary_df_from_fits(fits: List[TailFit]) -> pd.DataFrame:
    rows = []
    for f in fits:
        Cb = C_base(f.D)
        rows.append({
            "D":        f.D,
            "C_base":   Cb,
            "C_inf":    f.C_inf,
            "se_C_inf": f.se_C_inf,
            "a":        f.a,
            "se_a":     f.se_a,
            "S_Delta":  f.C_inf / Cb,
            "R2":       f.r2,
        })
    return pd.DataFrame(rows).sort_values("D")


def fig_heegner_Ssing(
    fits: List[TailFit],
    out_pdf: Path,
    out_csv: Path,
    cfg: PlotConfig = PCFG,
) -> pd.DataFrame:
    """Bar-plot de S_Δ = C_∞ / C_base para todos los discriminantes."""
    sumdf = _summary_df_from_fits(fits)
    sumdf.to_csv(out_csv, index=False)

    D = sumdf["D"].to_numpy(dtype=int)
    S = sumdf["S_Delta"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    colors = [cfg.color_accent if s == S.min() else cfg.color_main for s in S]
    bars   = ax.bar(range(len(D)), S, color=colors,
                    edgecolor="white", linewidth=0.7, width=0.65)

    ax.set_xticks(range(len(D)))
    ax.set_xticklabels([str(int(d)) for d in D], fontsize=9)
    ax.set_xlabel(r"$\Delta$ (Heegner)")
    ax.set_ylabel(r"$\mathfrak{S}_\Delta = C_\infty / C_{\rm base}$")
    ax.set_title(r"Heegner: tamaño del factor singular $\mathfrak{S}_\Delta$")
    style_axis(ax)

    j = int(np.argmin(S))
    ax.text(j, S[j] + 0.005, f"{S[j]:.4f}",
            ha="center", va="bottom", fontsize=8.5, color=cfg.color_accent)
    save_fig(fig, out_pdf, cfg)
    return sumdf


def fig_heegner_Cinf_vs_Cbase(
    sumdf: pd.DataFrame,
    out_pdf: Path,
    cfg: PlotConfig = PCFG,
) -> None:
    x = sumdf["C_base"].to_numpy(dtype=float)
    y = sumdf["C_inf"].to_numpy(dtype=float)
    k = float((x @ y) / (x @ x))   # OLS a través del origen

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    ax.scatter(x, y, s=45, alpha=0.88, color=cfg.color_main, zorder=3)

    xl = np.linspace(x.min() * 0.95, x.max() * 1.02, 200)
    ax.plot(xl, k * xl, ls="--", color=cfg.color_accent, lw=1.1,
            label=rf"Ajuste por origen: $k={k:.4f}$")

    for _, row in sumdf.iterrows():
        ax.text(float(row["C_base"]) + 0.004, float(row["C_inf"]),
                f"${int(row['D'])}$", fontsize=8.5, va="center")

    ax.set_xlabel(r"$C_{\rm base}(\Delta)$")
    ax.set_ylabel(r"$C_\infty(\Delta)$")
    ax.set_title(r"Heegner: $C_\infty$ vs $C_{\rm base}$ (recta $y=kx$)")
    ax.legend(frameon=True, fontsize=8)
    style_axis(ax)
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Producto parcial sobre primos inertes  — Fig. 8 del paper
# ---------------------------------------------------------------------------

def fig_heegner_partial_products(
    Ds: List[int],
    out_pdf: Path,
    pmax: int = 2000,
    cfg: PlotConfig = PCFG,
) -> None:
    ps = primes_upto(pmax)
    ps = ps[ps >= 3]

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std + 0.4))
    for D in sorted(Ds):
        prod = 1.0
        xs, ys = [], []
        for p in ps:
            chi = chi_legendre(D, int(p))
            if chi == -1:
                prod *= (p - 1.0) / (p + 1.0)
            xs.append(int(p))
            ys.append(prod)
        ax.plot(xs, ys, lw=0.9, label=str(D))

    ax.set_xlabel(r"$x$ (cota en primos)")
    ax.set_ylabel(
        r"$\prod_{p\leq x,\;\chi_\Delta(p)=-1}\dfrac{p-1}{p+1}$")
    ax.set_title(r"Producto parcial de primer orden sobre primos inertes")
    ax.legend(ncol=3, frameon=True, fontsize=8)
    style_axis(ax)
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Efecto de v_p(R) en C_{σ,Δ}  — Fig. 9 del paper
# ---------------------------------------------------------------------------

def fig_vp_effect(
    df: pd.DataFrame,
    D: int,
    p: int,
    out_pdf: Path,
    m_max: int = 4,
    cfg: PlotConfig = PCFG,
) -> None:
    """
    Estratificación de C_{σ,Δ}(R) por v_p(R).
    Recibe df YA filtrado a D (o completo); si contiene varios D los filtra.
    """
    if df["D"].nunique() > 1:
        df = df[df["D"] == D]
    if len(df) == 0:
        log.warning("fig_vp_effect: sin datos para Δ=%d.", D)
        return

    vp_col = f"vp{p}"
    if vp_col not in df.columns:
        log.warning("Columna %s no encontrada; computando.", vp_col)
        vp_vals = vp_vector(df["R"].to_numpy(dtype=np.int64), p)
    else:
        vp_vals = df[vp_col].to_numpy()

    vp_clip = np.minimum(vp_vals, m_max)

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    x_all = df["log10R"].to_numpy()
    y_all = df["C_sigma_inerte"].to_numpy()

    plotted = False
    for m in range(m_max + 1):
        mask = vp_clip == m
        if mask.sum() < 200:
            continue
        xmid, ymean, ylo, yhi = binned_band(x_all[mask], y_all[mask],
                                             n_bins=220)
        label = (rf"$v_{p}(R)={m}$" if m < m_max
                 else rf"$v_{p}(R)\geq {m_max}$")
        ax.plot(xmid, ymean, label=label, lw=1.0)
        ax.fill_between(xmid, ylo, yhi, alpha=0.10)
        plotted = True

    if not plotted:
        log.warning("fig_vp_effect: todos los estratos tienen <200 puntos "
                    "para Δ=%d, p=%d.", D, p)
        plt.close(fig)
        return

    ax.set_xlabel(r"$\log_{10} R$")
    ax.set_ylabel(r"$C_{\sigma,\Delta}(R)$")
    ax.set_title(
        rf"$\Delta={D}$: estratificación por $v_{p}(R)$")
    ax.legend(frameon=True, ncol=2, fontsize=8)
    style_axis(ax)
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Extra: predicción por primos inertes pequeños  — Fig. 10 del paper
# ---------------------------------------------------------------------------

def _predict_small_inerts(D: int, cutoff: int = 50) -> float:
    Cb = C_base(D)
    prod = 1.0
    for p in primes_upto(cutoff):
        if p < 3:
            continue
        chi = chi_legendre(D, int(p))
        if chi == -1:
            prod *= (p - 1.0) / (p + 1.0)
    return Cb * prod


def fig_heegner_pred_small_inerts(
    sumdf: pd.DataFrame,
    out_pdf: Path,
    cutoff: int = 50,
    cfg: PlotConfig = PCFG,
) -> None:
    Ds   = [int(d) for d in sumdf["D"]]
    pred = np.array([_predict_small_inerts(D, cutoff) for D in Ds])
    obs  = sumdf["C_inf"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(cfg.fig_w_full, cfg.fig_h_std))
    ax.scatter(pred, obs, s=50, alpha=0.90, color=cfg.color_main, zorder=3)

    mn = min(pred.min(), obs.min()) * 0.95
    mx = max(pred.max(), obs.max()) * 1.03
    ax.plot([mn, mx], [mn, mx], ls="--", color=cfg.color_accent, lw=1.0,
            label="$y=x$")

    for D, xv, yv in zip(Ds, pred, obs):
        ax.text(float(xv) + 0.003, float(yv), f"${D}$",
                fontsize=8.5, va="center")

    ax.set_xlabel(r"Predicción 1er orden (inertes $\leq$ cutoff)")
    ax.set_ylabel(r"$C_\infty(\Delta)$ observado")
    ax.set_title(
        rf"Heegner: $C_\infty$ vs predicción por inertes pequeños (cutoff={cutoff})")
    ax.legend(frameon=True, fontsize=8)
    style_axis(ax)
    save_fig(fig, out_pdf, cfg)


# ---------------------------------------------------------------------------
# Tablas LaTeX  (Tabla 1 y Tabla 2 del paper)
# ---------------------------------------------------------------------------

_LATEX_HEADER = r"""\documentclass[10pt]{article}
\usepackage{booktabs,siunitx,amsmath}
\begin{document}
"""
_LATEX_FOOTER = r"\end{document}" + "\n"


def _fmt_pm(val: float, err: float, decimals: int = 6) -> str:
    """Formatea val ± err en LaTeX con notación siunitx."""
    fmt = f"{{:.{decimals}f}}"
    return rf"\num{{{fmt.format(val)}}} \pm \num{{{fmt.format(err)}}}"


def export_latex_table1(
    sumdf: pd.DataFrame,
    out_tex: Path,
    fits_by_D: Dict[int, TailFit],
) -> None:
    """
    Tabla 1 del paper: C_base, C_∞, EE_WLS, a, EE_WLS(a), S_Δ.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Heegner: baseline canónico $C_{\rm base}(\Delta)$ y ajuste "
        r"de cola $C_\infty(\Delta)$ usando $C_{\sigma,\Delta}(R)\approx "
        r"C_\infty + a/\log R$ en $R\geq 10^6$ (200~bins por cuantiles, WLS).}",
        r"\label{tab:heegner_Cinf}",
        r"\small",
        r"\begin{tabular}{r S[table-format=1.6] S[table-format=1.6] "
        r"S[table-format=1.6] S[table-format=1.3] S[table-format=1.3] S[table-format=1.6]}",
        r"\toprule",
        r"$\Delta$ & {$C_{\rm base}$} & {$C_\infty$} & "
        r"{$\mathrm{EE}_{\rm WLS}(C_\infty)$} & "
        r"{$a$} & {$\mathrm{EE}_{\rm WLS}(a)$} & {$\mathfrak{S}_\Delta$} \\",
        r"\midrule",
    ]
    for _, row in sumdf.sort_values("D", key=lambda s: s.abs()).iterrows():
        D    = int(row["D"])
        fit  = fits_by_D[D]
        lines.append(
            rf"  ${D}$ & {row['C_base']:.6f} & {row['C_inf']:.6f} & "
            rf"{fit.se_C_inf:.6f} & {fit.a:.3f} & {fit.se_a:.3f} & "
            rf"{row['S_Delta']:.6f} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("  → %s", out_tex.name)


def export_latex_table2(
    uncertainty_df: pd.DataFrame,
    out_tex: Path,
) -> None:
    """
    Tabla 2 del paper: C_∞, EE_WLS, EE_sist, |Δ_model|, EE_full.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Descomposición de incertidumbre de cola para $C_\infty(\Delta)$. "
        r"Modelo de referencia: $C_\infty+a/\log R$ en $R_{\rm tail}=10^6$. "
        r"Se reporta $\mathrm{EE}_{\rm WLS}$, la sensibilidad $\mathrm{EE}_{\rm sist}$ "
        r"en $R_{\rm tail}\in\{10^6,2\cdot10^6,3\cdot10^6\}$, y el desplazamiento de modelo "
        r"$|\Delta_{\rm model}|:=|C^{(2)}_\infty-C^{(1)}_\infty|$. "
        r"Además, $\mathrm{EE}_{\rm full}:=\sqrt{\mathrm{EE}_{\rm WLS}^2+\mathrm{EE}_{\rm sist}^2+|\Delta_{\rm model}|^2}$.}",
        r"\label{tab:uncertainty}",
        r"\small",
        r"\begin{tabular}{r S[table-format=1.6] S[table-format=1.6] "
        r"S[table-format=1.6] S[table-format=1.6] S[table-format=1.6]}",
        r"\toprule",
        r"$\Delta$ & {$C^{(1)}_\infty$} & {$\mathrm{EE}_{\rm WLS}$} & "
        r"{$\mathrm{EE}_{\rm sist}$} & {$|\Delta_{\rm model}|$} & {$\mathrm{EE}_{\rm full}$} \\",
        r"\midrule",
    ]
    for _, row in uncertainty_df.sort_values("D", key=lambda s: s.abs()).iterrows():
        delta_abs = row["Delta_model_abs_C_inf"] if "Delta_model_abs_C_inf" in row else abs(row["Delta_model_C_inf"])
        ee_full = row["EE_full_linear"] if "EE_full_linear" in row else math.sqrt(
            row["EE_WLS_linear"] ** 2 + row["EE_systematic_linear"] ** 2 + delta_abs ** 2
        )
        lines.append(
            rf"  ${int(row['D'])}$ & {row['C_inf_linear']:.6f} & "
            rf"{row['EE_WLS_linear']:.6f} & {row['EE_systematic_linear']:.6f} & "
            rf"{delta_abs:.6f} & {ee_full:.6f} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("  → %s", out_tex.name)


# ---------------------------------------------------------------------------
# Summaries: star (refinamiento ramificado) + incertidumbre de cola
# ---------------------------------------------------------------------------

def write_star_summaries(
    sumdfs: Dict[int, pd.DataFrame],
    out_fits_csv: Path,
    out_sd_csv: Path,
    R_tail_min: int,
    n_bins: int,
) -> List[TailFit]:
    star_fits: List[TailFit] = []
    sd_rows: List[dict]      = []

    for D in sorted(sumdfs):
        df   = sumdfs[D]
        fit  = tail_fit_binned_wls(df, R_tail_min, n_bins,
                                   y_col="C_sigma_star")
        star_fits.append(fit)

        tail   = df[df["R"] >= R_tail_min]
        sd_b   = float(np.std(tail["C_sigma_inerte"].to_numpy(), ddof=1))
        sd_s   = float(np.std(tail["C_sigma_star"].to_numpy(), ddof=1))
        ratio  = sd_s / sd_b if sd_b > 0 else float("nan")
        sd_rows.append({
            "D":                    D,
            "SD_C_sigma_inerte":    sd_b,
            "SD_C_sigma_star":      sd_s,
            "ratio_star_over_inerte": ratio,
            "R_tail_min":           R_tail_min,
        })

    _summary_df_from_fits(star_fits).to_csv(out_fits_csv, index=False)
    pd.DataFrame(sd_rows).sort_values("D").to_csv(out_sd_csv, index=False)
    log.info("  star_fits → %s", out_fits_csv.name)
    log.info("  sd_comparison → %s", out_sd_csv.name)
    return star_fits


def write_tail_uncertainty(
    sumdfs: Dict[int, pd.DataFrame],
    out_grid_csv: Path,
    out_summary_csv: Path,
    R_ref: int,
    R_grid: List[int],
    n_bins: int,
) -> pd.DataFrame:
    grid_rows:    List[dict] = []
    summary_rows: List[dict] = []

    for D in sorted(sumdfs):
        df = sumdfs[D]
        by_model: Dict[str, List[TailFit]] = {"linear": [], "quadratic": []}

        for model in ("linear", "quadratic"):
            for rmin in R_grid:
                try:
                    fit = tail_fit_binned_wls(df, rmin, n_bins,
                                              y_col="C_sigma_inerte",
                                              model=model)
                except ValueError as e:
                    log.warning("Saltando ajuste Δ=%d, R_tail=%d, %s: %s",
                                D, rmin, model, e)
                    continue
                by_model[model].append(fit)
                grid_rows.append({
                    "D":           D, "model": fit.model,
                    "R_tail_min":  fit.R_tail_min,
                    "C_inf":       fit.C_inf, "se_C_inf": fit.se_C_inf,
                    "a":           fit.a,      "se_a":     fit.se_a,
                    "b":           fit.b,      "se_b":     fit.se_b,
                    "r2":          fit.r2,
                    "n_tail_pts":  fit.n_tail_pts, "n_bins":  fit.n_bins_eff,
                    "R_min":       fit.R_min,       "R_max":   fit.R_max,
                })

        lin_ref  = next((f for f in by_model["linear"]
                         if f.R_tail_min == R_ref), None)
        quad_ref = next((f for f in by_model["quadratic"]
                         if f.R_tail_min == R_ref), None)
        if lin_ref is None or quad_ref is None:
            log.warning("Δ=%d: R_ref=%d no en rejilla; saltando summary.", D, R_ref)
            continue

        ee_sys_l = max(abs(f.C_inf - lin_ref.C_inf)
                       for f in by_model["linear"])   if by_model["linear"]  else 0.0
        ee_sys_q = max(abs(f.C_inf - quad_ref.C_inf)
                       for f in by_model["quadratic"]) if by_model["quadratic"] else 0.0
        delta_model = quad_ref.C_inf - lin_ref.C_inf
        delta_model_abs = abs(delta_model)
        ee_linear_nomodel = math.hypot(lin_ref.se_C_inf, ee_sys_l)
        ee_linear_full = math.sqrt(
            lin_ref.se_C_inf ** 2 + ee_sys_l ** 2 + delta_model_abs ** 2
        )

        summary_rows.append({
            "D":                    D,
            "R_tail_ref":           R_ref,
            "R_tail_grid":          ";".join(str(v) for v in R_grid),
            "C_inf_linear":         lin_ref.C_inf,
            "EE_WLS_linear":        lin_ref.se_C_inf,
            "EE_systematic_linear": ee_sys_l,
            "C_inf_quadratic":      quad_ref.C_inf,
            "EE_WLS_quadratic":     quad_ref.se_C_inf,
            "EE_systematic_quadratic": ee_sys_q,
            "Delta_model_C_inf":    delta_model,
            "Delta_model_abs_C_inf": delta_model_abs,
            "EE_total_linear":      ee_linear_nomodel,
            "EE_full_linear":       ee_linear_full,
            "EE_total_quadratic":   math.hypot(quad_ref.se_C_inf, ee_sys_q),
            "R2_linear":            lin_ref.r2,
            "R2_quadratic":         quad_ref.r2,
        })

    grid_df    = pd.DataFrame(grid_rows).sort_values(["D", "model", "R_tail_min"])
    summary_df = pd.DataFrame(summary_rows).sort_values("D")
    grid_df.to_csv(out_grid_csv,    index=False)
    summary_df.to_csv(out_summary_csv, index=False)
    log.info("  grid_fits → %s", out_grid_csv.name)
    log.info("  uncertainty → %s", out_summary_csv.name)
    return summary_df


# ---------------------------------------------------------------------------
# Procesamiento de un discriminante (apto para multiprocessing)
# ---------------------------------------------------------------------------

def _process_one_D(
    csv_path: Path,
    bundle:   Path,
    R_tail:   int,
    n_bins:   int,
    mech_Ds:  List[int],
    cfg:      PlotConfig,
) -> Tuple[int, TailFit, pd.DataFrame]:
    """
    Carga, ajusta y genera todas las figuras por discriminante.
    Diseñada para ejecutarse en un proceso separado (serializable).
    """
    set_mpl_defaults(cfg)
    df  = load_csigma_csv(csv_path)
    D   = int(df["D"].iloc[0])
    fit = tail_fit_binned_wls(df, R_tail, n_bins, model="linear")
    pfx = f"D{D}"

    fig_stabilization(df, fit, bundle / f"{pfx}_stabilizacion.pdf", cfg)

    if D in mech_Ds:
        fig_raw_vs_norm(df, fit,   bundle / f"{pfx}_raw_vs_norm.pdf",   cfg)
        fig_Craw_vs_H(df, fit,     bundle / f"{pfx}_Craw_vs_H.pdf",
                      R_tail, cfg=cfg)
        fig_tail_residual(df, fit, bundle / f"{pfx}_tail_residual.pdf",
                          R_tail, cfg)

    if D == -7:
        extra = bundle / "Extra"
        extra.mkdir(parents=True, exist_ok=True)
        fig_vp_effect(df, D, 3, bundle / f"{pfx}_vp_effect_p3.pdf",
                      cfg=cfg)
        fig_vp_effect(df, D, 5, bundle / f"{pfx}_vp_effect_p5.pdf",
                      cfg=cfg)

    return D, fit, df


# ---------------------------------------------------------------------------
# Argumentos de línea de comandos
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Genera figuras paper-ready para el manuscrito de anclaje inerte.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input_dir",    default="Datos/Canonicos",
                    help="Directorio con csigma_inerte_D*.csv")
    ap.add_argument("--bundle_dir",   default="Figuras/Articulo",
                    help="Directorio de salida (coincide con LaTeX \\graphicspath)")
    ap.add_argument("--R_tail_min",   type=int, default=1_000_000,
                    help="Corte mínimo de R para ajuste de cola")
    ap.add_argument("--tail_grid",    default="1000000,2000000,3000000",
                    help="Rejilla de cortes para análisis de sensibilidad")
    ap.add_argument("--bins_tail",    type=int, default=200,
                    help="Número de bins (cuantiles) en ajuste WLS de cola")
    ap.add_argument("--pmax_partial", type=int, default=2000,
                    help="Cota de primos para producto parcial")
    ap.add_argument("--pred_cutoff",  type=int, default=50,
                    help="Cutoff de inertes para figura de predicción")
    ap.add_argument("--jobs",         type=int, default=1,
                    help="Número de procesos paralelos (1 = secuencial)")
    ap.add_argument("--mech_Ds",      default="-43,-7,-4",
                    help="Discriminantes para figuras de mecanismo (raw/norm/colapso/residuos)")
    ap.add_argument("--png",          action="store_true",
                    help="Exportar también PNG (además de PDF)")
    ap.add_argument("--figw_full",    type=float, default=6.80,
                    help="Ancho de figura completa (pulgadas)")
    ap.add_argument("--figw_single",  type=float, default=3.46,
                    help="Ancho de figura de una columna (pulgadas)")
    ap.add_argument("--verbose",      action="store_true",
                    help="Activa logging DEBUG")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # PlotConfig actualizado con los argumentos del usuario
    cfg = PlotConfig(
        fig_w_full=args.figw_full,
        fig_w_single=args.figw_single,
        export_png=args.png,
        r_tail_min=args.R_tail_min,
        n_bins=args.bins_tail,
    )
    set_mpl_defaults(cfg)

    R_grid  = parse_tail_grid(args.tail_grid, args.R_tail_min)
    mech_Ds = [int(s) for s in args.mech_Ds.replace(" ", "").split(",")]

    input_dir = Path(args.input_dir)
    bundle    = Path(args.bundle_dir)
    paths     = Paths(
        bundle      = bundle,
        extra       = bundle / "Extra",
        diagnostics = bundle / "Diagnosticos",
    )
    for p in (paths.bundle, paths.extra, paths.diagnostics):
        p.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        p for p in input_dir.iterdir()
        if p.name.startswith("csigma_inerte_D") and p.suffix == ".csv"
    )
    if not csv_files:
        raise SystemExit(
            f"No se encontraron csigma_inerte_D*.csv en {input_dir.resolve()}"
        )
    log.info("Encontrados %d archivos CSV en %s.", len(csv_files), input_dir)

    # ------------------------------------------------------------------
    # 1) Por discriminante: carga + ajuste + figuras individuales
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    fits_by_D: Dict[int, TailFit]      = {}
    sumdfs:    Dict[int, pd.DataFrame] = {}

    if args.jobs > 1:
        log.info("Procesamiento paralelo con %d workers.", args.jobs)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    _process_one_D, csv, paths.bundle,
                    args.R_tail_min, args.bins_tail, mech_Ds, cfg
                ): csv
                for csv in csv_files
            }
            for future in as_completed(futures):
                csv = futures[future]
                try:
                    D, fit, df = future.result()
                    fits_by_D[D] = fit
                    sumdfs[D]    = df
                except Exception as exc:
                    log.error("Error en %s: %s", csv.name, exc)
    else:
        for csv in csv_files:
            try:
                D, fit, df = _process_one_D(
                    csv, paths.bundle,
                    args.R_tail_min, args.bins_tail, mech_Ds, cfg
                )
                fits_by_D[D] = fit
                sumdfs[D]    = df
            except Exception as exc:
                log.error("Error procesando %s: %s", csv.name, exc)

    log.info("Figuras por discriminante: %.1f s total.",
             time.perf_counter() - t0)

    if not fits_by_D:
        raise SystemExit("No se generaron ajustes. Revisa los CSV de entrada.")

    fits = list(fits_by_D.values())

    # ------------------------------------------------------------------
    # 2) Grid resumen (Fig. 5 del paper)
    # ------------------------------------------------------------------
    log.info("Generando grid resumen...")
    fig_summary_grid(sumdfs, fits_by_D,
                     paths.bundle / "Heegner_summary_grid.pdf", cfg)

    # ------------------------------------------------------------------
    # 3) Figuras comparativas Heegner
    # ------------------------------------------------------------------
    log.info("Generando figuras comparativas Heegner...")
    out_csv = paths.bundle / "summary_tail_fits.csv"
    sumdf   = fig_heegner_Ssing(
        fits,
        out_pdf=paths.bundle / "Heegner_Ssing.pdf",
        out_csv=out_csv,
        cfg=cfg,
    )
    fig_heegner_Cinf_vs_Cbase(
        sumdf, paths.bundle / "Heegner_Cinf_vs_Cbase.pdf", cfg)
    fig_heegner_partial_products(
        [int(d) for d in sumdf["D"]],
        paths.bundle / "Heegner_partial_products.pdf",
        pmax=args.pmax_partial,
        cfg=cfg,
    )
    fig_heegner_pred_small_inerts(
        sumdf, paths.extra / "Heegner_pred_small_inerts.pdf",
        cutoff=args.pred_cutoff, cfg=cfg,
    )

    # ------------------------------------------------------------------
    # 4) Tablas LaTeX (Tabla 1 y Tabla 2 del paper)
    # ------------------------------------------------------------------
    log.info("Generando tablas LaTeX...")
    export_latex_table1(sumdf, paths.bundle / "table1_heegner.tex", fits_by_D)

    out_star_csv = paths.bundle / "summary_tail_fits_star.csv"
    out_sd_csv   = paths.bundle / "summary_tail_sd_comparison.csv"
    write_star_summaries(sumdfs, out_star_csv, out_sd_csv,
                         args.R_tail_min, args.bins_tail)

    out_grid_csv = paths.bundle / "summary_tail_grid_fits.csv"
    out_unc_csv  = paths.bundle / "summary_tail_uncertainty.csv"
    uncertainty_df = write_tail_uncertainty(
        sumdfs, out_grid_csv, out_unc_csv,
        R_ref=args.R_tail_min, R_grid=R_grid, n_bins=args.bins_tail,
    )
    if not uncertainty_df.empty:
        export_latex_table2(uncertainty_df, paths.bundle / "table2_uncertainty.tex")

    # ------------------------------------------------------------------
    # Reporte final
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info("Completado en %.1f s.", elapsed)
    log.info("Salida principal:     %s", paths.bundle.resolve())
    log.info("Discriminantes:       %s",
             sorted(fits_by_D.keys(), key=abs))
    log.info("Figuras generadas:    %d PDF",
             len(list(paths.bundle.rglob("*.pdf"))))
    log.info("CSV de resumen:       %s", out_csv.name)
    log.info("Tablas LaTeX:         table1_heegner.tex, table2_uncertainty.tex")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
