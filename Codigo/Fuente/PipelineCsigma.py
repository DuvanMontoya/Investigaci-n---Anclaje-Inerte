#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  C_SIGMA^INERTE — PIPELINE CANÓNICO v2.1                                      ║
║  Auditoría · Inferencia asintótica · Figuras                                 ║
║                                                                              ║
║  Implementación orientada a reproducibilidad, rendimiento y robustez,        ║
║  sin cambiar ninguna fórmula matemática del manuscrito.                      ║
║                                                                              ║
║  Basado en:                                                                  ║
║    "De la esfera x²+y²+z²=R² al anclaje inerte: densidades p-ádicas          ║
║    explícitas para la familia R²−c²"                                         ║
║                                                                              ║
║  Fórmulas implementadas (exactas):                                           ║
║    · Teorema 5.3   — σ_p^(m) para p≥3 inerte                                 ║
║    · Teorema 6.4   — σ_2^(m) para 2 inerte (Δ≡5 mod 8)                       ║
║    · Corolario 6.5 — h_p(m) = σ_p^(m)/σ_p^(0)                                ║
║    · Teorema B.1   — σ_2^(m) para Δ=−8 (2 ramificado)                        ║
║    · Teorema B.4   — σ_2^(m) para Δ=−4 (2 ramificado, gaussiano)             ║
║    · Teorema B.9   — σ_3^(m) para Δ=−3 (3 ramificado, eisensteiniano)        ║
║    · Teorema 7.3   — C_base(Δ) para Heegner                                  ║
║    · Def. HΔ(R) y Cσ,Δ(R) (Sección 1.7)                                      ║
║    · Refinamiento H⋆Δ(R) y C⋆σ,Δ(R) (Sección 9.4)                             ║
║                                                                              ║
║  Requisitos: Python 3.10+ | numpy · pandas · scipy · matplotlib              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Matplotlib en modo no-interactivo (CLI/headless seguro)
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 0. LOGGING
# ──────────────────────────────────────────────────────────────────────────────

LOG = logging.getLogger("csigma_inerte")


def _setup_logging(verbosity: int) -> None:
    """
    verbosity:
      0 -> WARNING
      1 -> INFO
      2+-> DEBUG
    """
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTES UNIVERSALES Y TABLA HEEGNER
# ──────────────────────────────────────────────────────────────────────────────

# Euler–Mascheroni γ (constante universal)
EULER_GAMMA: float = 0.5772156649015328606065120900824024310
EXP_MINUS_GAMMA: float = math.exp(-EULER_GAMMA)

# C_univ := C_base(−4) = π·e^{−γ}/2  (gaussiano: w=4, |Δ|=4)
C_UNIV: float = math.pi * EXP_MINUS_GAMMA / 2.0

# Discriminantes de Heegner (h(Δ)=1) y w(Δ) (unidades)
HEEGNER: Dict[int, Dict[str, int]] = {
    -3:   {"w": 6, "h": 1},
    -4:   {"w": 4, "h": 1},
    -7:   {"w": 2, "h": 1},
    -8:   {"w": 2, "h": 1},
    -11:  {"w": 2, "h": 1},
    -19:  {"w": 2, "h": 1},
    -43:  {"w": 2, "h": 1},
    -67:  {"w": 2, "h": 1},
    -163: {"w": 2, "h": 1},
}

# Referencias empíricas (del manuscrito; usadas solo para tabla/validación contextual)
HEEGNER_C_INF_REF: Dict[int, float] = {
    -3:   0.566005,
    -4:   0.735032,
    -7:   0.431485,
    -8:   0.692068,
    -11:  0.650708,
    -19:  0.473185,
    -43:  0.303469,
    -67:  0.241297,
    -163: 0.156052,
}

HEEGNER_A_REF: Dict[int, float] = {
    -3:   0.563,
    -4:   0.775,
    -7:   0.513,
    -8:   0.819,
    -11:  0.666,
    -19:  0.590,
    -43:  0.400,
    -67:  0.272,
    -163: -0.010,
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. KRONECKER (D/p) + CLASIFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def kronecker_symbol(D: int, p: int) -> int:
    """
    Símbolo de Kronecker (D/p).

    Para p=2 y D impar:
      - D≡1,7 (mod 8) → +1  (2 split)
      - D≡3,5 (mod 8) → −1  (2 inerte)
    Para p=2 y D par: devuelve 0 (2 ramificado en discriminantes pares).
    Para p impar: símbolo de Legendre (D/p) mediante exponenciación modular.
    """
    if p == 2:
        if (D & 1) == 0:
            return 0
        r = D % 8
        return 1 if r in (1, 7) else -1
    a = D % p
    if a == 0:
        return 0
    t = pow(int(a), (p - 1) // 2, p)
    return 1 if t == 1 else -1


def prime_type(D: int, p: int) -> str:
    """Clasifica p en Q(√D): 'split' | 'inert' | 'ramified'."""
    chi = kronecker_symbol(D, p)
    if chi == 1:
        return "split"
    if chi == -1:
        return "inert"
    return "ramified"


# ──────────────────────────────────────────────────────────────────────────────
# 3. C_BASE(Δ) (TEOREMA 7.3 PARA HEEGNER)
# ──────────────────────────────────────────────────────────────────────────────

def cbase_heegner(D: int) -> float:
    """
    Heegner: C_base(Δ) = 4π·e^{−γ} / (w(Δ)·√|Δ|).
    """
    if D not in HEEGNER:
        raise ValueError(f"Δ={D} no es Heegner. Valores válidos: {sorted(HEEGNER)}")
    w = HEEGNER[D]["w"]
    return 4.0 * math.pi * EXP_MINUS_GAMMA / (w * math.sqrt(abs(D)))


def cbase_heegner_table() -> Dict[int, float]:
    return {D: cbase_heegner(D) for D in HEEGNER}


# ──────────────────────────────────────────────────────────────────────────────
# 4. DENSIDADES LOCALES σ y FACTORES h (COR. 6.5)
# ──────────────────────────────────────────────────────────────────────────────

def sigma_p_odd_inert(p: int, m: int) -> float:
    """
    Teorema 5.3 (p≥3 impar):
      σ_p^(m) = 1 − 2 / (p^m·(p+1))
    """
    pm = float(pow(p, m))
    return 1.0 - 2.0 / (pm * (p + 1.0))


def sigma_2_inert(m: int) -> float:
    """
    Teorema 6.4 (p=2 inerte, Δ≡5 mod 8):
      σ_2^(m) = 1 − 1/(3·2^m)
    """
    pm = float(1 << m)  # 2^m exacto en int
    return 1.0 - 1.0 / (3.0 * pm)


def sigma_2_ramified_d8(m: int) -> float:
    """
    Teorema B.1 (Δ=−8):
      σ_2^(0) = 1/2
      σ_2^(m) = 3/2^{m+1}  para m≥1
    """
    if m == 0:
        return 0.5
    return 3.0 / float(1 << (m + 1))


def sigma_2_ramified_d4(m: int) -> float:
    """
    Teorema B.4 (Δ=−4):
      σ_2^(m) = 3/2^{m+2}
      σ_2^(0) = 3/4
    """
    return 3.0 / float(1 << (m + 2))


def sigma_3_ramified_d3(m: int) -> float:
    """
    Teorema B.9 (Δ=−3):
      σ_3^(m) = 2/3^{m+1}
      σ_3^(0) = 2/3
    """
    return 2.0 / float(pow(3, m + 1))


def h_inert(p: int, m: int) -> float:
    """
    Corolario 6.5, forma cerrada del multiplicador h_p(m)=σ_p^(m)/σ_p^(0),
    válido cuando p es inerte.

    - p≥3 inerte:  h_p(m) = (p^m(p+1) − 2) / (p^m(p−1))
    - p=2  inerte: h_2(m) = (3·2^m − 1) / 2^{m+1}
    """
    if m == 0:
        return 1.0
    if p == 2:
        pm = float(1 << m)
        return (3.0 * pm - 1.0) / (2.0 * pm)  # (3·2^m−1)/2^{m+1}
    pm = float(pow(p, m))
    return (pm * (p + 1.0) - 2.0) / (pm * (p - 1.0))


def ramified_correction_factor(D: int, p: int, m: int) -> float:
    """
    Corrección ramificada para H⋆Δ (Sección 9.4):
      p impar ramificado → multiplicar por p^{−v_p(R)}.
    """
    if m == 0 or p == 2:
        return 1.0
    if kronecker_symbol(D, p) == 0:
        return float(p) ** (-m)
    return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 5. CRIBA SPF (LINEAL) + FACTORIZACIÓN ITERATIVA (SIN DICCIONARIOS)
# ──────────────────────────────────────────────────────────────────────────────

def build_spf(n: int) -> np.ndarray:
    """
    Criba lineal O(n): spf[k] = menor primo que divide k (k≥2).
    dtype int32 para eficiencia de memoria.
    """
    if n < 1:
        return np.zeros(1, dtype=np.int32)
    spf = np.zeros(n + 1, dtype=np.int32)
    primes: List[int] = []
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            ip = i * p
            if ip > n:
                break
            if spf[ip] == 0:
                spf[ip] = p
            if p == spf[i]:
                break
    return spf


def iter_factorization(n: int, spf: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Itera (p, e) con n = ∏ p^e usando tabla SPF.
    Más rápido y con menos presión de memoria que construir dicts por factor.
    """
    n = int(n)
    while n > 1:
        p = int(spf[n])
        e = 1
        n //= p
        while n > 1 and int(spf[n]) == p:
            e += 1
            n //= p
        yield p, e


def factorize_to_dict(n: int, spf: np.ndarray) -> Dict[int, int]:
    """Conveniencia: dict {p:e} a partir de iter_factorization."""
    return {p: e for p, e in iter_factorization(n, spf)}


# ──────────────────────────────────────────────────────────────────────────────
# 6. HΔ(R) y H⋆Δ(R)
# ──────────────────────────────────────────────────────────────────────────────

def compute_H_single(R: int, D: int, spf: np.ndarray, star: bool = False) -> float:
    """
    HΔ(R)  = ∏_{p|R, p inerte} h_p(v_p(R))   (Def. HΔ, Sección 1.7)
    H⋆Δ(R) = HΔ(R) · ∏_{p|Δ, p impar ramificado} p^{−v_p(R)} (Sección 9.4)

    Implementación con factorización iterativa para reducir overhead.
    """
    H = 1.0
    for p, m in iter_factorization(R, spf):
        chi = kronecker_symbol(D, p)
        if chi == -1:
            H *= h_inert(p, m)
        elif star and chi == 0 and p > 2:
            H *= float(p) ** (-m)
    return H


def compute_H_array(R_arr: np.ndarray, D: int, spf: np.ndarray, star: bool = False) -> np.ndarray:
    """
    Calcula HΔ(R) o H⋆Δ(R) para un array de R.
    """
    R_arr = np.asarray(R_arr, dtype=np.int64)
    out = np.empty(R_arr.size, dtype=np.float64)
    for i, R in enumerate(R_arr):
        out[i] = compute_H_single(int(R), D, spf, star=star)
    return out


def recompute_H_and_audit(
    df: pd.DataFrame,
    spf: np.ndarray,
    D: int,
    star: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Recalcula HΔ(R) o H⋆Δ(R) y, si existe columna correspondiente en el CSV,
    devuelve max|H_recomputed − H_csv|.

    Convención de columnas aceptadas:
      - HΔ:  H_R, H, HR
      - H⋆:  H_star_R, Hstar_R, H_star, Hstar, H*_R, H*_star (insensible a mayúsculas)
    """
    R_vals = df["R"].to_numpy(dtype=np.int64)
    H_re = compute_H_array(R_vals, D, spf, star=star)

    if not star:
        candidates = ("H_R", "H", "HR", "h_r", "h")
    else:
        candidates = ("H_star_R", "Hstar_R", "H_star", "Hstar", "H*_R", "H*",
                      "H_star", "H⋆", "H⋆_R", "H_star_R")

    col = _find_col(df, *candidates)
    if col is None:
        return H_re, float("nan")

    H_csv = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    err = float(np.nanmax(np.abs(H_re - H_csv)))
    return H_re, err


# ──────────────────────────────────────────────────────────────────────────────
# 7. CSV: CARGA + NORMALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    colmap = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in colmap:
            return colmap[key]
    return None


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Normaliza columnas a:
      R, D (opcional), N_R (opcional), H_R (opcional), X (opcional).
    Si X no existe y (N_R,H_R) sí, reconstruye X = N_R·log R /(H_R·R).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV vacío.")

    colR = _find_col(df, "R", "r")
    if colR is None:
        raise ValueError(f"Columna 'R' no encontrada. Columnas: {list(df.columns)}")

    colD = _find_col(df, "D", "d", "discriminant", "delta", "Δ")
    colN = _find_col(df, "N_R", "N", "NR", "n_r", "n")
    colH = _find_col(df, "H_R", "H", "HR", "h_r", "h")
    colX = _find_col(df, "C_sigma_inerte", "csigma_inerte", "C_sigma", "X", "ratio", "x")

    keep = [c for c in (colR, colD, colN, colH, colX) if c is not None]
    df2 = df[keep].copy()

    rename = {colR: "R"}  # type: ignore[index]
    if colD:
        rename[colD] = "D"
    if colN:
        rename[colN] = "N_R"
    if colH:
        rename[colH] = "H_R"
    if colX:
        rename[colX] = "X"
    df2.rename(columns=rename, inplace=True)

    # Tipos
    df2["R"] = pd.to_numeric(df2["R"], errors="raise").astype(np.int64)
    if "D" in df2.columns:
        df2["D"] = pd.to_numeric(df2["D"], errors="raise").astype(np.int64)
    if "N_R" in df2.columns:
        df2["N_R"] = pd.to_numeric(df2["N_R"], errors="raise").astype(np.int64)
    if "H_R" in df2.columns:
        df2["H_R"] = pd.to_numeric(df2["H_R"], errors="coerce").astype(float)
    if "X" in df2.columns:
        df2["X"] = pd.to_numeric(df2["X"], errors="coerce").astype(float)

    df2 = (
        df2.sort_values("R")
           .drop_duplicates(subset=["R"])
           .reset_index(drop=True)
    )

    # Reconstruir X si no existe
    if "X" not in df2.columns:
        if {"N_R", "H_R"}.issubset(df2.columns):
            R = df2["R"].to_numpy(dtype=float)
            N = df2["N_R"].to_numpy(dtype=float)
            H = df2["H_R"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                df2["X"] = N * np.log(R) / (H * R)
        else:
            raise ValueError("Falta X y también falta (N_R, H_R) para reconstruirla.")
    return df2


# ──────────────────────────────────────────────────────────────────────────────
# 8. AUDITORÍAS
# ──────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def audit_R_structure(df: pd.DataFrame, require_odd_step2: bool = True) -> None:
    """
    Verifica:
      1) R>0 y estrictamente creciente.
      2) Si require_odd_step2: R[0] impar y paso constante par.
    """
    R = df["R"].to_numpy(np.int64)
    if R.size == 0:
        raise AssertionError("CSV sin filas.")
    if np.any(R <= 0):
        raise AssertionError("Existen valores R ≤ 0.")
    d = np.diff(R)
    if d.size > 0 and np.any(d <= 0):
        raise AssertionError("R no es estrictamente creciente.")
    if require_odd_step2:
        if (R[0] % 2) == 0:
            raise AssertionError("El primer R es par (se esperaba impar).")
        if d.size > 0:
            step = int(d[0])
            if (step % 2) != 0:
                raise AssertionError(f"Paso={step}: debe ser par para preservar paridad.")
            if not np.all(d == step):
                raise AssertionError("R no tiene paso constante.")


def audit_identity_X(df: pd.DataFrame) -> Optional[float]:
    """
    Verifica X = N_R·log(R)/(H_R·R). Devuelve max|X_recon − X|.
    """
    if not {"N_R", "H_R", "X"}.issubset(df.columns):
        return None
    R = df["R"].to_numpy(float)
    N = df["N_R"].to_numpy(float)
    H = df["H_R"].to_numpy(float)
    X = df["X"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        X_re = N * np.log(R) / (H * R)
    return float(np.nanmax(np.abs(X_re - X)))


def audit_Cstar(
    df: pd.DataFrame,
    D: int,
    spf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sección 9.4:
      C⋆σ,Δ(R) = Cσ,Δ(R) · ∏_{p|Δ, p impar} p^{v_p(R)}.
    Retorna (C_sigma, C_sigma_star).
    """
    C_sigma = df["X"].to_numpy(float).copy()

    # primes ramificados impares: p|Δ y (D/p)=0 con p>2
    factors_D = factorize_to_dict(abs(D), spf)
    ram_odd = {p for p in factors_D if p > 2 and kronecker_symbol(D, p) == 0}

    if not ram_odd:
        return C_sigma, C_sigma.copy()

    R_arr = df["R"].to_numpy(np.int64)
    ram_factor = np.ones(R_arr.size, dtype=float)

    # Optimización: factorizar cada R una sola vez
    for i, R in enumerate(R_arr):
        for p, e in iter_factorization(int(R), spf):
            if p in ram_odd:
                ram_factor[i] *= float(p) ** e

    return C_sigma, C_sigma * ram_factor


# ──────────────────────────────────────────────────────────────────────────────
# 9. ESTADÍSTICA GLOBAL ROBUSTA
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GlobalStats:
    n: int
    mean: float
    median: float
    std: float
    mad: float
    q05: float
    q25: float
    q75: float
    q95: float
    min_val: float
    max_val: float
    skew: float
    kurtosis: float


def compute_global_stats(x: np.ndarray) -> GlobalStats:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return GlobalStats(0, float("nan"), float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), float("nan"))
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    std = float(np.std(x, ddof=1)) if n >= 2 else 0.0
    skew = float(stats.skew(x, bias=False)) if n >= 3 else float("nan")
    kurt = float(stats.kurtosis(x, fisher=True, bias=False)) if n >= 4 else float("nan")
    return GlobalStats(
        n=n,
        mean=float(np.mean(x)),
        median=med,
        std=std,
        mad=mad,
        q05=float(np.quantile(x, 0.05)),
        q25=float(np.quantile(x, 0.25)),
        q75=float(np.quantile(x, 0.75)),
        q95=float(np.quantile(x, 0.95)),
        min_val=float(np.min(x)),
        max_val=float(np.max(x)),
        skew=skew,
        kurtosis=kurt,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 10. BLOQUES + WLS (MODELO 1er ORDEN) + ORDEN 2
# ──────────────────────────────────────────────────────────────────────────────

def make_blocks(df: pd.DataFrame, n_blocks: int, x_col: str = "X") -> pd.DataFrame:
    tmp = df[["R", x_col]].copy().rename(columns={x_col: "_X"})
    tmp["block"] = pd.qcut(tmp["R"], q=int(n_blocks), labels=False, duplicates="drop")
    agg = (
        tmp.groupby("block")
        .agg(R_center=("R", "mean"), X_bar=("_X", "mean"), std=("_X", "std"), n=("_X", "count"))
        .reset_index(drop=True)
        .sort_values("R_center")
        .reset_index(drop=True)
    )
    agg["var"] = agg["std"] ** 2
    return agg


@dataclass(frozen=True)
class WLSResult:
    weight_mode: str
    n_blocks_total: int
    n_blocks_used: int
    minR_center_used: float
    maxR_center_used: float
    C_inf: float
    C_inf_CI95: Tuple[float, float]
    a: float
    a_CI95: Tuple[float, float]
    R2_w: float
    dof: int
    sigma2: float
    cov_beta: List[List[float]]


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve A x = b con fallback a least-squares si A es mal condicionada.
    """
    try:
        cond = np.linalg.cond(A)
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError("Matriz mal condicionada.")
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x


def fit_wls(
    blocks: pd.DataFrame,
    minR_center: float = 1e4,
    weight_mode: str = "n_over_var",
) -> Tuple[WLSResult, pd.DataFrame]:
    """
    Modelo (Sección 9.3):
      X_bar ≈ C_∞ + a/log(R_center)
    WLS con pesos: 'n' | 'inv_var' | 'n_over_var'.
    """
    b = blocks.copy()
    b = b[np.isfinite(b["R_center"]) & np.isfinite(b["X_bar"])].copy()
    b = b[b["R_center"] >= float(minR_center)].copy()
    if len(b) < 3:
        raise ValueError(
            f"Solo {len(b)} bloques disponibles (minR_center={minR_center:g}). "
            "Reduce minR o aumenta el número de bloques."
        )

    R = b["R_center"].to_numpy(float)
    y = b["X_bar"].to_numpy(float)
    x = 1.0 / np.log(R)

    n_col = b["n"].to_numpy(float)
    var_col = b["var"].to_numpy(float)

    # Sustitución robusta para varianzas nulas/NaN
    med_var = float(np.nanmedian(var_col[var_col > 0])) if np.any(var_col > 0) else 1.0
    var_col = np.where((~np.isfinite(var_col)) | (var_col <= 0), med_var, var_col)

    if weight_mode == "n":
        w = n_col
    elif weight_mode == "inv_var":
        w = 1.0 / var_col
    elif weight_mode == "n_over_var":
        w = n_col / var_col
    else:
        raise ValueError(f"weight_mode inválido: '{weight_mode}'")

    w = np.where(np.isfinite(w) & (w > 0), w, 1e-18)
    w = w / np.mean(w)

    Xmat = np.column_stack([np.ones_like(x), x])  # beta=[C_inf, a]
    XtW = Xmat.T * w
    XtWX = XtW @ Xmat
    XtWy = XtW @ y

    beta = _safe_solve(XtWX, XtWy)

    yhat = Xmat @ beta
    resid = y - yhat
    dof = int(len(y) - 2)
    if dof <= 0:
        raise ValueError("DOF ≤ 0. Necesitas al menos 3 bloques para el ajuste.")

    sigma2 = float(np.sum(w * resid**2) / dof)
    cov_beta = sigma2 * np.linalg.pinv(XtWX)  # pinv por estabilidad
    se = np.sqrt(np.diag(cov_beta))
    tcrit = float(stats.t.ppf(0.975, dof))

    C_inf = float(beta[0])
    a = float(beta[1])
    CI_C = (C_inf - tcrit * se[0], C_inf + tcrit * se[0])
    CI_a = (a - tcrit * se[1], a + tcrit * se[1])

    ybar = float(np.average(y, weights=w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    R2_w = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    res = WLSResult(
        weight_mode=weight_mode,
        n_blocks_total=int(len(blocks)),
        n_blocks_used=int(len(b)),
        minR_center_used=float(np.min(R)),
        maxR_center_used=float(np.max(R)),
        C_inf=C_inf,
        C_inf_CI95=CI_C,
        a=a,
        a_CI95=CI_a,
        R2_w=float(R2_w),
        dof=dof,
        sigma2=sigma2,
        cov_beta=cov_beta.tolist(),
    )

    b = b.copy()
    b["x"] = x
    b["y"] = y
    b["yhat"] = yhat
    b["resid"] = resid
    b["w"] = w
    return res, b


def fit_wls_order2(
    blocks: pd.DataFrame,
    minR_center: float = 1e4,
    weight_mode: str = "n_over_var",
) -> Tuple[float, float, float]:
    """
    Orden 2 (Sección 9.3, Tabla 2):
      X_bar ≈ C_∞ + a/log R + b/log² R
    Retorna (C_inf_2, a_2, b_2). Si no hay suficientes bloques, retorna NaN.
    """
    b = blocks.copy()
    b = b[np.isfinite(b["R_center"]) & np.isfinite(b["X_bar"])].copy()
    b = b[b["R_center"] >= float(minR_center)].copy()
    if len(b) < 4:
        return (float("nan"), float("nan"), float("nan"))

    R = b["R_center"].to_numpy(float)
    y = b["X_bar"].to_numpy(float)
    logR = np.log(R)

    n_col = b["n"].to_numpy(float)
    var_col = b["var"].to_numpy(float)
    med_var = float(np.nanmedian(var_col[var_col > 0])) if np.any(var_col > 0) else 1.0
    var_col = np.where((~np.isfinite(var_col)) | (var_col <= 0), med_var, var_col)

    w = (n_col / var_col) if weight_mode == "n_over_var" else n_col
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-18)
    w = w / np.mean(w)

    Xmat = np.column_stack([np.ones_like(logR), 1.0 / logR, 1.0 / logR**2])
    XtW = Xmat.T * w
    beta = _safe_solve(XtW @ Xmat, XtW @ y)
    return float(beta[0]), float(beta[1]), float(beta[2])


# ──────────────────────────────────────────────────────────────────────────────
# 11. BOOTSTRAP BLOQUE-A-BLOQUE
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_Cinf(block_fit: pd.DataFrame, n_boot: int = 2000, seed: int = 123) -> Dict[str, float]:
    """
    Remuestreo de bloques (filas) con reemplazo; re-ajuste WLS en (C_inf, a).
    Cuantifica sensibilidad de C_inf respecto a muestra de bloques.
    """
    x = block_fit["x"].to_numpy(float)
    y = block_fit["y"].to_numpy(float)
    w = block_fit["w"].to_numpy(float)
    m = int(len(x))
    if m < 3 or n_boot <= 0:
        return {"boot_n": 0, "boot_sd": float("nan"), "boot_q025": float("nan"), "boot_q975": float("nan")}

    rng = np.random.default_rng(int(seed))
    C_samples = np.empty(int(n_boot), dtype=float)

    for i in range(int(n_boot)):
        idx = rng.integers(0, m, size=m)
        xb = x[idx]
        yb = y[idx]
        wb = w[idx]
        wb = wb / np.mean(wb)

        Xmat = np.column_stack([np.ones_like(xb), xb])
        XtW = Xmat.T * wb
        beta = _safe_solve(XtW @ Xmat, XtW @ yb)
        C_samples[i] = float(beta[0])

    finite = C_samples[np.isfinite(C_samples)]
    if finite.size == 0:
        return {"boot_n": float(n_boot), "boot_sd": float("nan"), "boot_q025": float("nan"), "boot_q975": float("nan")}

    q025, q975 = np.quantile(finite, [0.025, 0.975])
    return {
        "boot_n": float(n_boot),
        "boot_sd": float(np.std(finite, ddof=1)),
        "boot_q025": float(q025),
        "boot_q975": float(q975),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 12. PREDICCIÓN PRIMER ORDEN (SECCIÓN 10.1) + PRIMOS
# ──────────────────────────────────────────────────────────────────────────────

def primes_upto(n: int) -> np.ndarray:
    if n < 2:
        return np.array([], dtype=np.int32)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = False
    return np.flatnonzero(sieve).astype(np.int32)


def partial_product_inert(D: int, cutoff: int = 50) -> float:
    """
    Sección 10.1:
      ∏_{p≤cutoff, χΔ(p)=-1} (p−1)/(p+1)
    """
    primes = primes_upto(int(cutoff))
    prod = 1.0
    for p in primes:
        if kronecker_symbol(D, int(p)) == -1:
            prod *= (float(p) - 1.0) / (float(p) + 1.0)
    return prod


# ──────────────────────────────────────────────────────────────────────────────
# 13. INCERTIDUMBRE SISTEMÁTICA (TABLA 2)
# ──────────────────────────────────────────────────────────────────────────────

def compute_EEsist(
    blocks: pd.DataFrame,
    R_tails: Tuple[float, ...] = (1e6, 2e6, 3e6),
    weight_mode: str = "n_over_var",
) -> Dict[str, float]:
    """
    Tabla 2:
      EE_sist = max_{R_tail} |C_∞(R_tail) − C_∞(R_tail=10^6)|
      |Δmodel| = |C_∞^(2) − C_∞^(1)| en la cola de referencia.
    """
    C_vals: Dict[float, float] = {}
    for R_tail in R_tails:
        try:
            res, _ = fit_wls(blocks, minR_center=float(R_tail), weight_mode=weight_mode)
            C_vals[float(R_tail)] = res.C_inf
        except ValueError:
            C_vals[float(R_tail)] = float("nan")

    ref = C_vals.get(1e6, float("nan"))
    others = [v for k, v in C_vals.items() if k != 1e6 and np.isfinite(v)]
    EEsist = max((abs(v - ref) for v in others), default=float("nan")) if np.isfinite(ref) else float("nan")

    # Desplazamiento de modelo (orden 2 vs orden 1)
    C2, _, _ = fit_wls_order2(blocks, minR_center=1e6, weight_mode=weight_mode)
    delta_model = abs(C2 - ref) if (np.isfinite(C2) and np.isfinite(ref)) else float("nan")

    out: Dict[str, float] = {"EE_sist": float(EEsist), "delta_model": float(delta_model)}
    # también serializamos C_inf por cola (como strings)
    out.update({f"C_inf_tail_{int(k):d}": float(v) for k, v in C_vals.items()})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 14. FIGURAS (PDF)
# ──────────────────────────────────────────────────────────────────────────────

def _setup_mpl() -> None:
    matplotlib.rcParams.update({
        "figure.figsize": (9.0, 5.2),
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.22,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "lines.linewidth": 1.6,
    })


def plot_X_vs_R(df: pd.DataFrame, outdir: Path, C_inf: float, D: Optional[int] = None) -> None:
    R = df["R"].to_numpy(float)
    X = df["X"].to_numpy(float)

    fig, ax = plt.subplots()
    ax.scatter(R, X, s=1, alpha=0.3, rasterized=True, label="X(R) crudo")
    ax.set_xscale("log")
    ax.axhline(C_inf, ls="--", lw=1.8, label=f"$C_\\infty \\approx {C_inf:.6f}$")
    ax.axhline(C_UNIV, ls=":", lw=1.5, label=f"$C_{{\\rm univ}} = C_{{\\rm base}}(-4) \\approx {C_UNIV:.6f}$")
    if D is not None and D in HEEGNER:
        Cb = cbase_heegner(D)
        ax.axhline(Cb, ls="-.", lw=1.5, label=f"$C_{{\\rm base}}(\\Delta={D}) \\approx {Cb:.6f}$")

    title_disc = f"  ($\\Delta = {D}$)" if D is not None else ""
    ax.set_xlabel("$R$  (escala log)")
    ax.set_ylabel("$X(R) = C_{{\\sigma,\\Delta}}(R)$")
    ax.set_title(f"Evolución de $X(R)${title_disc}")
    ax.legend(fontsize=9)
    fig.savefig(outdir / "fig_X_vs_R.pdf")
    plt.close(fig)


def plot_blocks_regression(block_fit: pd.DataFrame, wls: WLSResult, outdir: Path, D: Optional[int] = None) -> None:
    x = block_fit["x"].to_numpy(float)
    y = block_fit["y"].to_numpy(float)
    yhat = block_fit["yhat"].to_numpy(float)
    resid = block_fit["resid"].to_numpy(float)

    cov = np.array(wls.cov_beta, float)
    tcrit = float(stats.t.ppf(0.975, wls.dof))

    xg = np.linspace(x.min(), x.max(), 400)
    Xg = np.column_stack([np.ones_like(xg), xg])
    yg = Xg @ np.array([wls.C_inf, wls.a])
    var_m = np.einsum("ij,jk,ik->i", Xg, cov, Xg)
    se_m = np.sqrt(np.maximum(var_m, 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    disc_str = f" ($\\Delta = {D}$)" if D is not None else ""
    ax1.scatter(x, y, s=40, zorder=3, label="Media por bloque")
    ax1.plot(x, yhat, lw=2, label=f"WLS  $R^2_w = {wls.R2_w:.5f}$")
    ax1.fill_between(xg, yg - tcrit * se_m, yg + tcrit * se_m, alpha=0.15, label="IC 95% (media WLS)")
    ax1.axhline(wls.C_inf, ls="--", lw=1.5, label=f"$C_\\infty \\approx {wls.C_inf:.8f}$")
    ax1.set_ylabel("$\\bar{{X}}$ por bloque")
    ax1.set_title(f"Régimen asintótico: $\\bar{{X}} \\approx C_\\infty + a/\\log R${disc_str}")
    ax1.legend(fontsize=9)

    ax2.scatter(x, resid, s=20, alpha=0.7, rasterized=True)
    ax2.axhline(0.0, lw=1)
    ax2.set_xlabel("$x = 1/\\log(R_{\\rm center})$  ($R$ crece hacia la izquierda)")
    ax2.set_ylabel("Residuo")

    if x.max() > x.min():
        ax2.set_xlim(x.max() * 1.02, x.min() * 0.98)  # R crece cuando x→0
    fig.tight_layout()
    fig.savefig(outdir / "fig_regresion_bloques.pdf")
    plt.close(fig)


def plot_hist_kde(df: pd.DataFrame, outdir: Path, C_inf: float, g: GlobalStats, D: Optional[int] = None) -> None:
    X = df["X"].to_numpy(float)
    X = X[np.isfinite(X)]
    if X.size == 0:
        return

    fig, ax = plt.subplots()
    ax.hist(X, bins=90, density=True, alpha=0.65, edgecolor="white", lw=0.4, label="Histograma X(R)")
    try:
        kde = stats.gaussian_kde(X)
        xs = np.linspace(g.min_val, g.max_val, 600)
        ax.plot(xs, kde(xs), lw=2, label="KDE")
    except Exception:
        pass

    ax.axvline(g.median, ls="--", lw=1.5, label=f"Mediana = {g.median:.6f}")
    ax.axvline(C_inf, ls=":", lw=1.8, label=f"$C_\\infty$ = {C_inf:.6f}")
    ax.axvline(C_UNIV, ls="-.", lw=1.4, label=f"$C_{{\\rm univ}}$ = {C_UNIV:.6f}")
    if D is not None and D in HEEGNER:
        Cb = cbase_heegner(D)
        ax.axvline(Cb, ls="-", lw=1.2, alpha=0.7, label=f"$C_{{\\rm base}}$ = {Cb:.6f}")

    disc_str = f" ($\\Delta = {D}$)" if D is not None else ""
    ax.set_xlabel("$X(R)$")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución empírica de $X(R)${disc_str}")
    ax.legend(fontsize=9)
    fig.savefig(outdir / "fig_hist_kde.pdf")
    plt.close(fig)


def plot_Cstar_vs_C(C_sigma: np.ndarray, C_sigma_star: np.ndarray, R_arr: np.ndarray, outdir: Path, D: int) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    logR = np.log10(np.asarray(R_arr, dtype=float))

    ax1.scatter(logR, C_sigma, s=1, alpha=0.3, rasterized=True, label="$C_{\\sigma,\\Delta}(R)$ (inerte)")
    ax1.set_ylabel("$C_{\\sigma,\\Delta}(R)$")
    ax1.set_title(f"Anclaje inerte: crudo vs refinado ($\\Delta = {D}$)")
    ax1.legend(fontsize=9)

    ax2.scatter(logR, C_sigma_star, s=1, alpha=0.3, rasterized=True, label="$C^*_{\\sigma,\\Delta}(R)$ (+ ramificado)")
    ax2.set_xlabel("$\\log_{10}(R)$")
    ax2.set_ylabel("$C^*_{\\sigma,\\Delta}(R)$")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / "fig_Cstar_refinado.pdf")
    plt.close(fig)


def plot_partial_products(D: int, outdir: Path, cutoff: int = 500) -> None:
    primes = primes_upto(int(cutoff))
    xs = primes.astype(int)
    ys = np.empty(xs.size, dtype=float)

    prod = 1.0
    for i, p in enumerate(xs):
        if kronecker_symbol(D, int(p)) == -1:
            prod *= (p - 1.0) / (p + 1.0)
        ys[i] = prod

    fig, ax = plt.subplots()
    ax.plot(xs, ys, lw=1.5, label=f"$\\prod_{{p \\leq x,\\, \\chi_\\Delta(p)=-1}} (p-1)/(p+1)$  ($\\Delta={D}$)")
    ax.set_xlabel("$x$ (cota en primos)")
    ax.set_ylabel("Producto parcial")
    ax.set_title(f"Producto de primer orden sobre primos inertes  ($\\Delta = {D}$)")
    ax.legend(fontsize=9)
    fig.savefig(outdir / "fig_producto_parcial_inertes.pdf")
    plt.close(fig)


def plot_linear_collapse(df: pd.DataFrame, D: int, spf: np.ndarray, outdir: Path, min_R: float = 1e6) -> None:
    """
    Sección 9.7: colapso lineal C_raw(R) = NΔ(R) log R / R vs HΔ(R).
    Ajuste por origen: C_raw ≈ k·H con k = argmin ||kH - C_raw||²
                      = (H·C_raw) / (H·H).
    """
    if "N_R" not in df.columns:
        return

    mask = df["R"].to_numpy(float) >= float(min_R)
    sub = df[mask].copy()
    if len(sub) < 10:
        return

    R_arr = sub["R"].to_numpy(float)
    N_arr = sub["N_R"].to_numpy(float)
    H_arr = compute_H_array(sub["R"].to_numpy(np.int64), D, spf, star=False)

    Craw = N_arr * np.log(R_arr) / R_arr
    H_arr = np.asarray(H_arr, dtype=float)
    Craw = np.asarray(Craw, dtype=float)

    denom = float(np.dot(H_arr, H_arr))
    k = float(np.dot(H_arr, Craw) / denom) if denom > 0 else float("nan")

    fig, ax = plt.subplots()
    ax.scatter(H_arr, Craw, s=2, alpha=0.35, rasterized=True, label=f"$R \\geq 10^{{{int(np.log10(min_R))}}}$")

    xg = np.linspace(float(np.nanmin(H_arr)), float(np.nanmax(H_arr)), 200)
    ax.plot(xg, k * xg, lw=2, ls="--", label=f"Ajuste por origen: $k = {k:.6f}$")

    ax.set_xlabel("$H_\\Delta(R)$")
    ax.set_ylabel("$C_{\\rm raw}(R) = N_\\Delta(R)\\,\\log R / R$")
    ax.set_title(f"Colapso lineal ($\\Delta = {D}$, cola $R \\geq {min_R:.0e}$)")
    ax.legend(fontsize=9)
    fig.savefig(outdir / "fig_colapso_lineal.pdf")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 15. SUMA DE DOS CUADRADOS (SOLO Δ=−4)
# ──────────────────────────────────────────────────────────────────────────────

def is_sum_two_squares(n: int, primes: np.ndarray) -> bool:
    """Fermat–Euler: n es suma de dos cuadrados ⇔ todo q≡3 (mod 4) tiene exponente par."""
    if n < 0:
        return False
    if n == 0:
        return True
    tmp = int(n)
    for p in primes:
        pp = int(p) * int(p)
        if pp > tmp:
            break
        if tmp % p == 0:
            e = 0
            while tmp % p == 0:
                tmp //= p
                e += 1
            if (p % 4 == 3) and (e % 2 == 1):
                return False
    if tmp > 1 and (tmp % 4 == 3):
        return False
    return True


def compute_N_family(R: int, family: str, primes: np.ndarray) -> int:
    """
    Cuenta N(R) para familia R²−c² (family='R') o (2R)²−c² (family='2R').
    Solo para K=Q(i), Δ=−4.
    """
    R = int(R)
    A = R if family == "R" else 2 * R
    A2 = A * A
    cnt = 0
    for c in range(1, A):
        M = A2 - c * c
        if M > 0 and is_sum_two_squares(M, primes):
            cnt += 1
    return cnt


def infer_family(df: pd.DataFrame) -> Optional[str]:
    """Infiere si N_R corresponde a familia R o 2R (usa un R pequeño para no costar)."""
    if "N_R" not in df.columns or df.empty:
        return None

    R_candidates = df["R"].to_numpy(int)
    # Tomamos un R moderado para test puntual
    R0 = next((int(r) for r in R_candidates if r >= 101), int(R_candidates[0]))
    N_csv = int(df.loc[df["R"] == R0, "N_R"].iloc[0])

    pA = primes_upto(2 * R0 + 10)
    NR = compute_N_family(R0, "R", pA)
    N2R = compute_N_family(R0, "2R", pA)
    return "R" if abs(NR - N_csv) <= abs(N2R - N_csv) else "2R"


# ──────────────────────────────────────────────────────────────────────────────
# 16. REPORTES
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_ci(ci: Tuple[float, float]) -> str:
    return f"[{ci[0]:.10f}, {ci[1]:.10f}]"


def write_summary_txt(
    outdir: Path,
    csv_path: Path,
    sha: str,
    g: GlobalStats,
    wls: WLSResult,
    boot: Dict[str, float],
    extra: Dict[str, object],
    D: Optional[int] = None,
) -> None:
    Cb = cbase_heegner(D) if (D is not None and D in HEEGNER) else None
    K = wls.C_inf / C_UNIV
    S_D = (wls.C_inf / Cb) if Cb is not None else None

    lines: List[str] = [
        "=" * 82,
        "RESUMEN — C_SIGMA^INERTE  (versión 2.1)",
        "=" * 82,
        f"Archivo : {csv_path.name}",
        f"SHA-256 : {sha}",
        f"Δ       : {D}" if D is not None else "Δ : no especificado",
        "",
    ]

    if "max_abs_error_identity_X" in extra:
        lines.append(f"[AUDIT] max|X_reconstruido − X_csv| = {extra['max_abs_error_identity_X']:.4e}")
    if "max_abs_error_H_recompute" in extra:
        lines.append(f"[AUDIT] max|H_recomputed − H_csv|   = {extra['max_abs_error_H_recompute']:.4e}")
    if "max_abs_error_H_star_recompute" in extra:
        lines.append(f"[AUDIT] max|H⋆_recomputed − H⋆_csv| = {extra['max_abs_error_H_star_recompute']:.4e}")
    if extra.get("inferred_family"):
        lines.append(f"[AUDIT] Familia inferida para N(R) = {extra['inferred_family']}")
    lines.append("")

    lines += [
        "── Estadísticas globales X(R) ──────────────────────────────────────────",
        f"  n           = {g.n}",
        f"  media       = {g.mean:.12f}",
        f"  mediana     = {g.median:.12f}",
        f"  std         = {g.std:.12f}",
        f"  MAD         = {g.mad:.12f}",
        f"  q05 / q95   = {g.q05:.12f}  /  {g.q95:.12f}",
        f"  min / max   = {g.min_val:.12f}  /  {g.max_val:.12f}",
        f"  skew / kurt = {g.skew:.6f}  /  {g.kurtosis:.6f}",
        "",
        "── WLS principal (modelo 1er orden) ────────────────────────────────────",
        f"  weight_mode = {wls.weight_mode}",
        f"  bloques     = {wls.n_blocks_used} / {wls.n_blocks_total}",
        f"  R_center ∈  [{wls.minR_center_used:.1f}, {wls.maxR_center_used:.1f}]",
        f"  C_inf       = {wls.C_inf:.12f}",
        f"  IC 95%      = {_fmt_ci(wls.C_inf_CI95)}",
        f"  a           = {wls.a:.12f}",
        f"  IC 95%(a)   = {_fmt_ci(wls.a_CI95)}",
        f"  R²_w        = {wls.R2_w:.8f}",
        f"  DOF         = {wls.dof}",
        "",
        "── Constantes teóricas ─────────────────────────────────────────────────",
        f"  C_univ = C_base(−4) = {C_UNIV:.12f}",
    ]

    if Cb is not None and S_D is not None:
        lines += [
            f"  C_base(Δ={D})    = {Cb:.12f}",
            f"  S_Δ = C_∞/C_base = {S_D:.12f}",
        ]
    lines.append(f"  K_inerte = C_∞/C_univ = {K:.12f}")

    # Incertidumbre sistemática
    if "EE_sist" in extra and extra["EE_sist"] is not None:
        EE = float(extra["EE_sist"])
        dm = float(extra.get("delta_model", float("nan")))
        lines += [
            "",
            "── Incertidumbre sistemática (Tabla 2) ────────────────────────────────",
            f"  EE_WLS   = {math.sqrt(wls.sigma2):.6e}  (raíz σ² residual WLS)",
            f"  EE_sist  = {EE:.6e}  (sensibilidad de truncación)",
            f"  |Δmodel| = {dm:.6e}  (1er vs 2do orden en cola)",
        ]

    if boot.get("boot_n", 0) and np.isfinite(boot.get("boot_sd", float("nan"))):
        lines += [
            "",
            f"── Bootstrap C_∞ (n_boot={int(boot['boot_n'])}) ────────────────────────",
            f"  sd(C_∞) ≈ {boot['boot_sd']:.6e}",
            f"  IC 95%  ≈ [{boot['boot_q025']:.12f}, {boot['boot_q975']:.12f}]",
        ]

    if D is not None and D in HEEGNER:
        pp = partial_product_inert(D, cutoff=50)
        C_pred = cbase_heegner(D) * pp
        lines += [
            "",
            "── Predicción primer orden inertes (Sección 10.1) ─────────────────────",
            f"  ∏_{{p≤50, inerte}} (p−1)/(p+1) = {pp:.10f}",
            f"  C_∞^pred ≈ C_base × ∏         = {C_pred:.10f}",
            f"  C_∞ observado                 = {wls.C_inf:.10f}",
            f"  Diferencia relativa           = {abs(wls.C_inf - C_pred)/abs(wls.C_inf):.4%}",
        ]

    lines.append("")
    (outdir / "Csigma_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(
    outdir: Path,
    csv_path: Path,
    sha: str,
    g: GlobalStats,
    wls: WLSResult,
    boot: Dict[str, float],
    extra: Dict[str, object],
    D: Optional[int] = None,
) -> None:
    Cb = cbase_heegner(D) if (D is not None and D in HEEGNER) else None
    S_D = (wls.C_inf / Cb) if Cb is not None else None

    pred_info: Dict[str, float] = {}
    if D is not None and D in HEEGNER:
        pp = partial_product_inert(D, 50)
        pred_info = {
            "partial_product_inert_p50": float(pp),
            "C_inf_pred_order1": float(cbase_heegner(D) * pp),
            "C_inf_observed": float(wls.C_inf),
        }

    payload = {
        "file": {"name": csv_path.name, "sha256": sha},
        "discriminant": D,
        "constants": {"C_univ": C_UNIV, "C_base_D": Cb, "EulerGamma": EULER_GAMMA, "exp_minus_gamma": EXP_MINUS_GAMMA},
        "global_stats": asdict(g),
        "wls": asdict(wls),
        "bootstrap": boot,
        "S_Delta": S_D,
        "K_inerte": float(wls.C_inf / C_UNIV),
        "prediction_order1": pred_info,
        "extra": extra,
    }
    with (outdir / "Csigma_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# 17. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> None:
    _setup_logging(int(args.verbose))
    _setup_mpl()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    sha = sha256_file(csv_path)

    # Discriminante
    D: Optional[int] = None
    if args.discriminant is not None:
        D = int(args.discriminant)
    elif "D" in df.columns:
        D_vals = df["D"].dropna().astype(int).unique()
        if len(D_vals) == 1:
            D = int(D_vals[0])
        elif len(D_vals) > 1:
            raise ValueError("El CSV contiene múltiples discriminantes. Usa --discriminant.")

    # Auditoría estructura de R
    audit_R_structure(df, require_odd_step2=bool(args.require_odd_step2))

    extra: Dict[str, object] = {}
    id_err = audit_identity_X(df)
    if id_err is not None:
        extra["max_abs_error_identity_X"] = float(id_err)

    # SPF (aseguramos cobertura también para |D| por si supera R_max)
    R_max = int(df["R"].max())
    n_spf = max(R_max, abs(int(D)) if D is not None else 0)
    LOG.info("Construyendo SPF hasta n=%d ...", n_spf)
    spf = build_spf(n_spf)

    # Auditoría H y H⋆
    if args.audit_H and D is not None:
        _, H_err = recompute_H_and_audit(df, spf, D, star=False)
        extra["max_abs_error_H_recompute"] = float(H_err)
        if np.isfinite(H_err) and H_err > 1e-8:
            warnings.warn(f"Error de recomputo HΔ(R) > 1e-8: {H_err:.4e}. Revisa el CSV.")
        _, H_err_star = recompute_H_and_audit(df, spf, D, star=True)
        extra["max_abs_error_H_star_recompute"] = float(H_err_star)

    # Inferencia de familia (solo Δ=-4; también permitimos D None cuando CSV no define)
    if args.audit_N and (D is None or D == -4):
        extra["inferred_family"] = infer_family(df)

    # Estadísticas globales
    g = compute_global_stats(df["X"].to_numpy(float))

    # Observable refinado C⋆ (si aplica)
    C_sigma_star: Optional[np.ndarray] = None
    if D is not None:
        C_sigma, C_sigma_star = audit_Cstar(df, D, spf)

    # Bloques y WLS
    blocks = make_blocks(df, n_blocks=int(args.blocks))
    wls, block_fit = fit_wls(blocks, minR_center=float(args.minR), weight_mode=str(args.weights))

    blocks_star: Optional[pd.DataFrame] = None
    wls_star: Optional[WLSResult] = None
    if C_sigma_star is not None:
        df_star = df.copy()
        df_star["X"] = C_sigma_star
        blocks_star = make_blocks(df_star, n_blocks=int(args.blocks), x_col="X")
        try:
            wls_star, _ = fit_wls(blocks_star, minR_center=float(args.minR), weight_mode=str(args.weights))
        except ValueError:
            wls_star = None

    # Incertidumbre sistemática (Tabla 2)
    sist = compute_EEsist(blocks, weight_mode=str(args.weights))
    extra.update({"EE_sist": sist.get("EE_sist", float("nan")), "delta_model": sist.get("delta_model", float("nan"))})
    extra.update({k: v for k, v in sist.items() if k.startswith("C_inf_tail_")})

    # Bootstrap
    boot: Dict[str, float] = {"boot_n": 0.0}
    if int(args.bootstrap) > 0:
        boot = bootstrap_Cinf(block_fit, n_boot=int(args.bootstrap), seed=int(args.seed))

    # Exportar tablas
    blocks.to_csv(outdir / "blocks_table.csv", index=False)
    block_fit.to_csv(outdir / "blocks_fit_table.csv", index=False)
    if blocks_star is not None:
        blocks_star.to_csv(outdir / "blocks_star_table.csv", index=False)

    # Figuras
    if not bool(args.no_plots):
        plot_X_vs_R(df, outdir, wls.C_inf, D=D)
        plot_blocks_regression(block_fit, wls, outdir, D=D)
        plot_hist_kde(df, outdir, wls.C_inf, g, D=D)
        if D is not None and C_sigma_star is not None:
            plot_Cstar_vs_C(C_sigma, C_sigma_star, df["R"].to_numpy(float), outdir, D)
            plot_partial_products(D, outdir, cutoff=int(args.prime_cutoff))
            plot_linear_collapse(df, D, spf, outdir, min_R=float(args.minR_collapse))

    # Reportes
    write_summary_txt(outdir, csv_path, sha, g, wls, boot, extra, D=D)
    write_summary_json(outdir, csv_path, sha, g, wls, boot, extra, D=D)

    # Consola (paper-ready)
    Cb = cbase_heegner(D) if (D is not None and D in HEEGNER) else None
    S_D = (wls.C_inf / Cb) if Cb is not None else None

    print("=" * 82)
    print("  C_SIGMA^INERTE  —  PIPELINE v2.1")
    print("=" * 82)
    print(f"  Archivo : {csv_path.name}")
    print(f"  SHA-256 : {sha}")
    if D is not None:
        print(f"  Δ       : {D}")
    print()

    if id_err is not None:
        print(f"  [AUDIT] max|X_recon − X_csv|  = {id_err:.4e}")
    if "max_abs_error_H_recompute" in extra:
        print(f"  [AUDIT] max|H_recom − H_csv|  = {extra['max_abs_error_H_recompute']:.4e}")
    if extra.get("inferred_family"):
        print(f"  [AUDIT] Familia N(R)          = {extra['inferred_family']}")

    print()
    print("  Estadísticas globales X(R):")
    print(f"    n     = {g.n}")
    print(f"    media = {g.mean:.12f}")
    print(f"    med.  = {g.median:.12f}")
    print(f"    std   = {g.std:.12f}")
    print(f"    MAD   = {g.mad:.12f}")
    print()

    print("  WLS  (X ≈ C_∞ + a/log R):")
    print(f"    C_∞      = {wls.C_inf:.12f}")
    print(f"    IC 95%   = {_fmt_ci(wls.C_inf_CI95)}")
    print(f"    a        = {wls.a:.12f}")
    print(f"    R²_w     = {wls.R2_w:.8f}")
    print(f"    bloques  = {wls.n_blocks_used}/{wls.n_blocks_total}")
    print()

    print("  Constantes teóricas:")
    print(f"    C_univ = C_base(−4)  = {C_UNIV:.12f}")
    if Cb is not None and S_D is not None:
        print(f"    C_base(Δ={D:5d})     = {Cb:.12f}")
        print(f"    S_Δ = C_∞/C_base     = {S_D:.12f}")
    print(f"    K = C_∞/C_univ       = {wls.C_inf/C_UNIV:.12f}")
    print()

    EE_sist = extra.get("EE_sist", float("nan"))
    if np.isfinite(float(EE_sist)):
        print("  Incertidumbre sistemática:")
        print(f"    EE_sist  = {float(EE_sist):.4e}")
        print(f"    |Δmodel| = {float(extra.get('delta_model', float('nan'))):.4e}")
        print()

    if boot.get("boot_n", 0) > 0:
        print(f"  Bootstrap C_∞ (n={int(boot['boot_n'])}):")
        print(f"    sd  ≈ {boot['boot_sd']:.4e}")
        print(f"    IC  ≈ [{boot['boot_q025']:.10f}, {boot['boot_q975']:.10f}]")
        print()

    if D is not None and D in HEEGNER:
        pp = partial_product_inert(D, 50)
        C_pred = cbase_heegner(D) * pp
        print("  Predicción 1er orden (Sección 10.1):")
        print(f"    C_∞^pred ≈ {C_pred:.10f}")
        print(f"    C_∞ obs  ≈ {wls.C_inf:.10f}")
        print(f"    error    ≈ {abs(wls.C_inf - C_pred)/abs(wls.C_inf):.4%}")
        print()

    if wls_star is not None:
        print("  WLS sobre C⋆_σ,Δ (refinamiento ramificado):")
        print(f"    C⋆_∞     = {wls_star.C_inf:.12f}")
        print(f"    IC 95%   = {_fmt_ci(wls_star.C_inf_CI95)}")
        print(f"    R²_w     = {wls_star.R2_w:.8f}")
        print()

    print(f"  [OK] Resultados guardados en: {outdir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# 18. SUBCOMANDO: TABLA HEEGNER
# ──────────────────────────────────────────────────────────────────────────────

def print_heegner_table() -> None:
    print("=" * 90)
    print("  TABLA HEEGNER — C_base(Δ) y predicciones de primer orden (Teo. 7.3 + §10.1)")
    print("=" * 90)
    print(f"  {'Δ':>6}  {'w(Δ)':>6}  {'C_base':>14}  {'∏ p≤50':>12}  {'C_pred':>12}  {'C_inf(ref)':>12}  {'S_Δ(ref)':>10}")
    print("  " + "-" * 86)
    for D, meta in sorted(HEEGNER.items(), reverse=True):
        Cb = cbase_heegner(D)
        pp = partial_product_inert(D, 50)
        Cp = Cb * pp
        Cref = HEEGNER_C_INF_REF.get(D, float("nan"))
        S = Cref / Cb if np.isfinite(Cref) else float("nan")
        print(f"  {D:>6}  {meta['w']:>6}  {Cb:>14.8f}  {pp:>12.8f}  {Cp:>12.8f}  {Cref:>12.6f}  {S:>10.6f}")
    print()
    print(f"  C_univ = C_base(−4) = {C_UNIV:.12f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 19. CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="csigma_inerte",
        description=(
            "Pipeline canonico C_sigma_inerte v2.1\n"
            "Auditoria · WLS asintotico · Bootstrap · Figuras PDF\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Aumenta verbosidad (-v, -vv).")
    sub = ap.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Ejecuta el pipeline completo sobre un CSV.")
    run_p.add_argument("csv", type=str, help="Ruta al CSV (fuente de verdad).")
    run_p.add_argument("--outdir", type=str, default="Analisis/Csigma", help="Directorio de salida.")
    run_p.add_argument("--discriminant", type=int, default=None, help="Discriminante D<0 (anula CSV si se pasa).")
    run_p.add_argument("--blocks", type=int, default=50, help="Número de bloques WLS [default: 50].")
    run_p.add_argument("--minR", type=float, default=1e4, help="R_center mínimo para WLS [default: 1e4].")
    run_p.add_argument("--weights", type=str, default="n_over_var",
                       choices=["n", "inv_var", "n_over_var"], help="Modo de pesos WLS.")
    run_p.add_argument("--bootstrap", type=int, default=2000, help="Iteraciones bootstrap (0 desactiva).")
    run_p.add_argument("--seed", type=int, default=123, help="Semilla RNG.")
    run_p.add_argument("--audit_H", action="store_true", help="Recomputa H y H_estrella y compara contra CSV si hay columnas.")
    run_p.add_argument("--audit_N", action="store_true", help="Infiere familia R/2R (solo D=-4).")
    run_p.add_argument("--require_odd_step2", action="store_true", default=True,
                       help="Exige R inicial impar y paso constante par (default: True).")
    run_p.add_argument("--no_plots", action="store_true", help="No generar figuras (solo tablas y reportes).")
    run_p.add_argument("--prime_cutoff", type=int, default=500, help="Corte de primos para figura de producto parcial.")
    run_p.add_argument("--minR_collapse", type=float, default=1e6, help="R mínimo para figura de colapso lineal.")

    sub.add_parser("heegner", help="Imprime tabla de constantes teóricas para Heegner.")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.command == "heegner":
        _setup_logging(int(args.verbose))
        print_heegner_table()
        return
    if args.command == "run":
        run_pipeline(args)
        return

    # Compatibilidad: invocación sin subcomando (primer argumento posicional = CSV)
    ap2 = argparse.ArgumentParser(add_help=False)
    ap2.add_argument("csv")
    ap2.add_argument("--outdir", type=str, default="Analisis/Csigma")
    ap2.add_argument("--discriminant", type=int, default=None)
    ap2.add_argument("--blocks", type=int, default=50)
    ap2.add_argument("--minR", type=float, default=1e4)
    ap2.add_argument("--weights", type=str, default="n_over_var", choices=["n", "inv_var", "n_over_var"])
    ap2.add_argument("--bootstrap", type=int, default=2000)
    ap2.add_argument("--seed", type=int, default=123)
    ap2.add_argument("--audit_H", action="store_true")
    ap2.add_argument("--audit_N", action="store_true")
    ap2.add_argument("--require_odd_step2", action="store_true", default=True)
    ap2.add_argument("--no_plots", action="store_true")
    ap2.add_argument("--prime_cutoff", type=int, default=500)
    ap2.add_argument("--minR_collapse", type=float, default=1e6)
    ap2.add_argument("-v", "--verbose", action="count", default=0)
    args2, _ = ap2.parse_known_args()
    args2.command = "run"
    run_pipeline(args2)


if __name__ == "__main__":
    main()
