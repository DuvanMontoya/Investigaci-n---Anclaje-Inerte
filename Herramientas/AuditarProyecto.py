#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoria reproducible del manuscrito.

Verifica:
1) Integridad de CSVs de datos.
2) Consistencia exacta C_sigma_inerte = N_R*log(R)/(H_R*R).
3) Coherencia numerica entre tabla LaTeX principal y summary_tail_fits.csv.
4) Checks numericos de formulas locales (muestreo Monte Carlo).
5) Duplicados de labels en LaTeX.
6) Limpieza del log de compilacion (sin warnings/overfull/etc).
7) Consistencia del post-procesado ramificado (CSVs *_star).
8) Consistencia de sensibilidad de cola (modelo lineal vs cuadratico).

Uso:
  python Herramientas/AuditarProyecto.py
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Datos" / "Canonicos"
TEX_PATH = ROOT / "Manuscrito" / "Fuente" / "InvestigacionCompleta.tex"
LOG_PATH = ROOT / "Manuscrito" / "Temporal" / "InvestigacionCompleta.log"
SUMMARY_CSV = ROOT / "Figuras" / "Articulo" / "summary_tail_fits.csv"
STAR_SUMMARY_CSV = ROOT / "Figuras" / "Articulo" / "summary_tail_fits_star.csv"
STAR_SD_CSV = ROOT / "Figuras" / "Articulo" / "summary_tail_sd_comparison.csv"
TAIL_GRID_CSV = ROOT / "Figuras" / "Articulo" / "summary_tail_grid_fits.csv"
TAIL_UNCERTAINTY_CSV = ROOT / "Figuras" / "Articulo" / "summary_tail_uncertainty.csv"
EXPECTED_DS = [-163, -67, -43, -19, -11, -8, -7, -4, -3]


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def primes_upto(n: int) -> List[int]:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = False
    return [int(p) for p in np.flatnonzero(sieve)]


PRIMES_10000 = primes_upto(10000)


def factorize(n: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    t = n
    for p in PRIMES_10000:
        if p * p > t:
            break
        if t % p == 0:
            m = 0
            while t % p == 0:
                t //= p
                m += 1
            out.append((p, m))
    if t > 1:
        out.append((t, 1))
    return out


def chi_delta_p(delta: int, p: int) -> int:
    if p == 2:
        # Kronecker symbol for odd fundamental discriminants.
        if delta % 2 == 0:
            return 0
        return -1 if delta % 8 == 5 else 1
    if delta % p == 0:
        return 0
    t = pow(delta % p, (p - 1) // 2, p)
    return 1 if t == 1 else -1


def sigma_ratio_odd_inert(p: int, m: int) -> float:
    sigma_m = 1.0 - 2.0 / (p**m * (p + 1.0))
    sigma_0 = (p - 1.0) / (p + 1.0)
    return sigma_m / sigma_0


def predicted_H(delta: int, R: int) -> float:
    # Modelo usado en los CSV: producto sobre primos inertes impares que dividen R.
    val = 1.0
    for p, m in factorize(R):
        if p >= 3 and chi_delta_p(delta, p) == -1:
            val *= sigma_ratio_odd_inert(p, m)
    return val


def odd_ramified_primes(delta: int) -> List[int]:
    return [p for p, _ in factorize(abs(delta)) if p % 2 == 1]


def vp_vector(R: np.ndarray, p: int) -> np.ndarray:
    work = R.copy()
    vp = np.zeros(len(work), dtype=np.int16)
    while True:
        mask = (work % p == 0)
        if not mask.any():
            break
        vp[mask] += 1
        work[mask] //= p
    return vp


def ramified_odd_multiplier_vector(delta: int, R: np.ndarray) -> np.ndarray:
    mult = np.ones(len(R), dtype=float)
    for p in odd_ramified_primes(delta):
        mult *= np.power(float(p), vp_vector(R, p))
    return mult


def vp_abs(n: int, p: int) -> int:
    t = abs(n)
    if t == 0:
        return 0
    v = 0
    while t % p == 0:
        t //= p
        v += 1
    return v


def is_norm_minus8_local(n: int) -> bool:
    # Appendix criterion used in manuscript:
    # n = 2^v * u with odd u, and u ≡ 1,3 mod 8 (plus n=0).
    if n == 0:
        return True
    v = vp_abs(n, 2)
    u = n // (2**v)
    return (u % 8) in (1, 3)


def load_data_files() -> List[Path]:
    files = sorted(DATA_DIR.glob("csigma_inerte_D-*_Rmax10000001_step10_Rmin101.csv"))
    if len(files) != 9:
        fail(f"Se esperaban 9 CSV en Datos/Canonicos, encontrados {len(files)}.")
    return files


def check_data_integrity(files: Sequence[Path]) -> Dict[int, pd.DataFrame]:
    by_d: Dict[int, pd.DataFrame] = {}
    r_grid_ref: pd.Series | None = None
    for fp in files:
        df = pd.read_csv(fp)
        required = {"D", "R", "N_R", "H_R", "C_sigma_inerte"}
        missing = required - set(df.columns)
        if missing:
            fail(f"{fp.name}: faltan columnas {sorted(missing)}")
        if df.isna().any().any():
            fail(f"{fp.name}: contiene NaN")
        if df["D"].nunique() != 1:
            fail(f"{fp.name}: columna D no es constante")
        d = int(df["D"].iloc[0])
        by_d[d] = df
        if not df["R"].is_monotonic_increasing:
            fail(f"{fp.name}: R no es creciente")
        if df["R"].duplicated().any():
            fail(f"{fp.name}: hay R duplicados")
        calc = df["N_R"] * np.log(df["R"]) / (df["H_R"] * df["R"])
        err = (calc - df["C_sigma_inerte"]).abs().max()
        if err > 1e-9:
            fail(f"{fp.name}: mismatch C_sigma_inerte (max err={err:.3e})")
        if r_grid_ref is None:
            r_grid_ref = df["R"]
        else:
            if not df["R"].equals(r_grid_ref):
                fail(f"{fp.name}: grilla R no coincide con los demas CSV")
    if sorted(by_d) != EXPECTED_DS:
        fail(f"Discriminantes inesperados en Datos/Canonicos: {sorted(by_d)}")
    ok("Integridad de CSVs y formula C_sigma_inerte validada")
    return by_d


def check_H_consistency(by_d: Dict[int, pd.DataFrame], sample_per_d: int = 1200) -> None:
    rng = random.Random(20260216)
    for d, df in by_d.items():
        idxs = [rng.randrange(len(df)) for _ in range(sample_per_d)]
        sub = df.iloc[idxs]
        diffs = []
        for row in sub.itertuples(index=False):
            h = predicted_H(d, int(row.R))
            diffs.append(abs(h - float(row.H_R)))
        mx = max(diffs)
        if mx > 1e-9:
            fail(f"D={d}: H_R no coincide con producto local (max err={mx:.3e})")
    ok("Consistencia H_R con producto teorico local (muestreo) validada")


def parse_main_table_from_tex(tex_text: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for line in tex_text.splitlines():
        s = line.strip()
        if s.startswith("$-") and s.count("&") == 6:
            p = [x.strip() for x in s.rstrip("\\").split("&")]
            rows.append(
                {
                    "D": int(p[0].strip("$")),
                    "C_base_tex": float(p[1]),
                    "C_inf_tex": float(p[2]),
                    "se_C_inf_tex": float(p[3]),
                    "a_tex": float(p[4]),
                    "se_a_tex": float(p[5]),
                    "S_tex": float(p[6]),
                }
            )
    if len(rows) < 9:
        fail("No se pudo parsear la tabla principal en el .tex")
    df = pd.DataFrame(rows).drop_duplicates("D").sort_values("D").reset_index(drop=True)
    if sorted(df["D"].tolist()) != EXPECTED_DS:
        fail("La tabla principal en .tex no tiene los 9 discriminantes esperados")
    return df


def check_summary_vs_tex() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")
    tab = parse_main_table_from_tex(tex)
    summ = pd.read_csv(SUMMARY_CSV).sort_values("D").reset_index(drop=True)
    if sorted(summ["D"].tolist()) != EXPECTED_DS:
        fail("summary_tail_fits.csv no contiene los 9 discriminantes esperados")
    m = tab.merge(
        summ[["D", "C_base", "C_inf", "se_C_inf", "a", "se_a", "S_sing"]],
        on="D",
        how="inner",
    )
    # Tolerancias acordes al redondeo de la tabla en el manuscrito.
    checks = [
        ("C_base_tex", "C_base", 1e-6),
        ("C_inf_tex", "C_inf", 1e-6),
        ("se_C_inf_tex", "se_C_inf", 1e-6),
        ("a_tex", "a", 1e-3),
        ("se_a_tex", "se_a", 1e-3),
        ("S_tex", "S_sing", 1e-6),
    ]
    for a, b, tol in checks:
        mx = (m[a] - m[b]).abs().max()
        if mx > tol:
            fail(f"Diferencia {a} vs {b} excede tolerancia: max={mx:.3e}, tol={tol:.1e}")
    ok("Tabla principal del .tex consistente con summary_tail_fits.csv")


def check_star_outputs(by_d: Dict[int, pd.DataFrame]) -> None:
    if not STAR_SUMMARY_CSV.exists():
        fail("No existe summary_tail_fits_star.csv; ejecuta GenerarFigurasArticulo.py.")
    if not STAR_SD_CSV.exists():
        fail("No existe summary_tail_sd_comparison.csv; ejecuta GenerarFigurasArticulo.py.")

    summ_star = pd.read_csv(STAR_SUMMARY_CSV).sort_values("D").reset_index(drop=True)
    if sorted(summ_star["D"].tolist()) != EXPECTED_DS:
        fail("summary_tail_fits_star.csv no contiene los 9 discriminantes esperados")
    req_star = {"D", "C_base", "C_inf", "se_C_inf", "a", "se_a", "S_sing"}
    if not req_star.issubset(set(summ_star.columns)):
        fail("summary_tail_fits_star.csv no tiene todas las columnas requeridas")

    sd_csv = pd.read_csv(STAR_SD_CSV).sort_values("D").reset_index(drop=True)
    req_sd = {"D", "SD_C_sigma_inerte", "SD_C_sigma_star", "ratio_star_over_inerte", "R_tail_min"}
    if not req_sd.issubset(set(sd_csv.columns)):
        fail("summary_tail_sd_comparison.csv no tiene columnas requeridas")
    if sorted(sd_csv["D"].tolist()) != EXPECTED_DS:
        fail("summary_tail_sd_comparison.csv no contiene los 9 discriminantes esperados")

    for row in sd_csv.itertuples(index=False):
        d = int(row.D)
        r_tail_min = int(row.R_tail_min)
        df = by_d[d]
        tail = df[df["R"] >= r_tail_min]
        base = tail["C_sigma_inerte"].to_numpy(dtype=float)
        mult = ramified_odd_multiplier_vector(d, tail["R"].to_numpy(dtype=np.int64))
        star = base * mult

        sd_base = float(np.std(base, ddof=0))
        sd_star = float(np.std(star, ddof=0))
        ratio = float(sd_star / sd_base) if sd_base > 0 else float("nan")

        if abs(sd_base - float(row.SD_C_sigma_inerte)) > 2e-6:
            fail(f"D={d}: SD base inconsistente en summary_tail_sd_comparison.csv")
        if abs(sd_star - float(row.SD_C_sigma_star)) > 2e-6:
            fail(f"D={d}: SD star inconsistente en summary_tail_sd_comparison.csv")
        if abs(ratio - float(row.ratio_star_over_inerte)) > 2e-6:
            fail(f"D={d}: ratio star/base inconsistente en summary_tail_sd_comparison.csv")

    ok("Post-procesado ramificado (H_star y SD de cola) consistente")


def check_tail_uncertainty_outputs() -> None:
    if not TAIL_GRID_CSV.exists():
        fail("No existe summary_tail_grid_fits.csv; ejecuta GenerarFigurasArticulo.py.")
    if not TAIL_UNCERTAINTY_CSV.exists():
        fail("No existe summary_tail_uncertainty.csv; ejecuta GenerarFigurasArticulo.py.")

    grid = pd.read_csv(TAIL_GRID_CSV).sort_values(["D", "model", "R_tail_min"]).reset_index(drop=True)
    unc = pd.read_csv(TAIL_UNCERTAINTY_CSV).sort_values("D").reset_index(drop=True)
    summ = pd.read_csv(SUMMARY_CSV).sort_values("D").reset_index(drop=True)

    req_grid = {
        "D", "model", "R_tail_min", "C_inf", "se_C_inf", "a", "se_a", "b", "se_b",
        "n_tail_points", "n_bins", "R_min", "R_max",
    }
    if not req_grid.issubset(set(grid.columns)):
        fail("summary_tail_grid_fits.csv no tiene columnas requeridas")

    req_unc = {
        "D", "R_tail_ref", "R_tail_grid", "C_inf_linear", "EE_WLS_linear", "EE_systematic_linear",
        "C_inf_quadratic", "EE_WLS_quadratic", "EE_systematic_quadratic", "Delta_model_C_inf",
        "EE_total_linear", "EE_total_quadratic",
    }
    if not req_unc.issubset(set(unc.columns)):
        fail("summary_tail_uncertainty.csv no tiene columnas requeridas")

    if sorted(grid["D"].unique().tolist()) != EXPECTED_DS:
        fail("summary_tail_grid_fits.csv no contiene los 9 discriminantes esperados")
    if sorted(unc["D"].tolist()) != EXPECTED_DS:
        fail("summary_tail_uncertainty.csv no contiene los 9 discriminantes esperados")

    models = set(str(m) for m in grid["model"].unique().tolist())
    if models != {"linear", "quadratic"}:
        fail(f"Modelos inesperados en summary_tail_grid_fits.csv: {sorted(models)}")

    for row in unc.itertuples(index=False):
        d = int(row.D)
        r_ref = int(row.R_tail_ref)
        sub = grid[grid["D"] == d]
        sub_lin = sub[sub["model"] == "linear"].sort_values("R_tail_min")
        sub_quad = sub[sub["model"] == "quadratic"].sort_values("R_tail_min")

        if sub_lin.empty or sub_quad.empty:
            fail(f"D={d}: faltan filas de modelo lineal/cuadrático en summary_tail_grid_fits.csv")

        if r_ref not in set(int(x) for x in sub_lin["R_tail_min"].tolist()):
            fail(f"D={d}: R_tail_ref={r_ref} no aparece en filas lineales")
        if r_ref not in set(int(x) for x in sub_quad["R_tail_min"].tolist()):
            fail(f"D={d}: R_tail_ref={r_ref} no aparece en filas cuadráticas")

        lin_ref = sub_lin[sub_lin["R_tail_min"] == r_ref].iloc[0]
        quad_ref = sub_quad[sub_quad["R_tail_min"] == r_ref].iloc[0]

        ee_sys_lin = float(np.max(np.abs(sub_lin["C_inf"].to_numpy(dtype=float) - float(lin_ref["C_inf"]))))
        ee_sys_quad = float(np.max(np.abs(sub_quad["C_inf"].to_numpy(dtype=float) - float(quad_ref["C_inf"]))))

        if abs(float(row.C_inf_linear) - float(lin_ref["C_inf"])) > 2e-9:
            fail(f"D={d}: C_inf_linear inconsistente con la grilla")
        if abs(float(row.EE_WLS_linear) - float(lin_ref["se_C_inf"])) > 2e-9:
            fail(f"D={d}: EE_WLS_linear inconsistente con la grilla")
        if abs(float(row.EE_systematic_linear) - ee_sys_lin) > 2e-9:
            fail(f"D={d}: EE_systematic_linear inconsistente con la grilla")

        if abs(float(row.C_inf_quadratic) - float(quad_ref["C_inf"])) > 2e-9:
            fail(f"D={d}: C_inf_quadratic inconsistente con la grilla")
        if abs(float(row.EE_WLS_quadratic) - float(quad_ref["se_C_inf"])) > 2e-9:
            fail(f"D={d}: EE_WLS_quadratic inconsistente con la grilla")
        if abs(float(row.EE_systematic_quadratic) - ee_sys_quad) > 2e-9:
            fail(f"D={d}: EE_systematic_quadratic inconsistente con la grilla")

        delta_model = float(quad_ref["C_inf"]) - float(lin_ref["C_inf"])
        if abs(float(row.Delta_model_C_inf) - delta_model) > 2e-9:
            fail(f"D={d}: Delta_model_C_inf inconsistente con la grilla")

    m = unc.merge(summ[["D", "C_inf", "se_C_inf"]], on="D", how="inner")
    if (m["C_inf_linear"] - m["C_inf"]).abs().max() > 1e-10:
        fail("C_inf_linear en summary_tail_uncertainty.csv no coincide con summary_tail_fits.csv")
    if (m["EE_WLS_linear"] - m["se_C_inf"]).abs().max() > 1e-10:
        fail("EE_WLS_linear no coincide con se_C_inf de summary_tail_fits.csv")

    ok("Sensibilidad de cola (lineal/cuadrática y grilla de R_tail) consistente")


def check_local_formula_montecarlo() -> None:
    rng = random.Random(20260216)

    # 1) Formula impar inerte: P(vp impar)=2/(p^m(p+1))
    max_err_odd = 0.0
    for p in [3, 5, 7, 11, 13]:
        for m in [0, 1, 2, 3]:
            R = (p**m) * 17
            n_samp = 140_000
            odd = 0
            mod = p**8
            for _ in range(n_samp):
                c = rng.randrange(mod)
                v = vp_abs(R * R - c * c, p)
                odd += int(v % 2 == 1)
            est = odd / n_samp
            theo = 2.0 / (p**m * (p + 1.0))
            max_err_odd = max(max_err_odd, abs(est - theo))
    if max_err_odd > 0.003:
        fail(f"Check Monte Carlo impar-inerte falla (max err={max_err_odd:.4f})")

    # 2) Formula 2-inerte: P(v2 impar)=1/(3*2^m)
    max_err_2 = 0.0
    for m in [0, 1, 2, 3, 4, 5]:
        R = (2**m) * 3
        n_samp = 180_000
        odd = 0
        for _ in range(n_samp):
            c = rng.randrange(2**20)
            v = vp_abs(R * R - c * c, 2)
            odd += int(v % 2 == 1)
        est = odd / n_samp
        theo = 1.0 / (3.0 * (2**m))
        max_err_2 = max(max_err_2, abs(est - theo))
    if max_err_2 > 0.003:
        fail(f"Check Monte Carlo 2-inerte falla (max err={max_err_2:.4f})")

    # 3) Formula appendix Delta=-8
    max_err_m8 = 0.0
    for m in [0, 1, 2, 3, 4]:
        R = (2**m) * 3
        n_samp = 180_000
        ok_count = 0
        for _ in range(n_samp):
            c = rng.randrange(2**20)
            n = R * R - c * c
            ok_count += int(is_norm_minus8_local(n))
        est = ok_count / n_samp
        theo = 0.5 if m == 0 else 3.0 / (2 ** (m + 1))
        max_err_m8 = max(max_err_m8, abs(est - theo))
    if max_err_m8 > 0.004:
        fail(f"Check Monte Carlo Delta=-8 falla (max err={max_err_m8:.4f})")

    ok(
        "Checks Monte Carlo locales superados "
        f"(odd={max_err_odd:.4f}, two-inert={max_err_2:.4f}, delta-8={max_err_m8:.4f})"
    )


def check_labels_unique() -> None:
    text = TEX_PATH.read_text(encoding="utf-8")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    counts: Dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        fail(f"Labels duplicados en .tex: {dups}")
    ok("No hay labels duplicados en el .tex")


def check_log_clean() -> None:
    if not LOG_PATH.exists():
        ok("No existe InvestigacionCompleta.log; se omite chequeo de log (compila sin limpieza para validarlo).")
        return
    log = LOG_PATH.read_text(encoding="utf-8", errors="replace")
    bad_patterns = [
        r"Warning:",
        r"Overfull",
        r"Underfull",
        r"Runaway argument",
        r"Fatal error",
        r"Undefined",
        r"duplicate",
    ]
    for pat in bad_patterns:
        if re.search(pat, log):
            fail(f"Log no limpio: detectado patron '{pat}'")
    ok("Log de LaTeX limpio (sin warnings/overfull/errores)")


def main() -> None:
    files = load_data_files()
    by_d = check_data_integrity(files)
    check_H_consistency(by_d)
    check_summary_vs_tex()
    check_star_outputs(by_d)
    check_tail_uncertainty_outputs()
    check_local_formula_montecarlo()
    check_labels_unique()
    check_log_clean()
    print("[OK] Auditoria completa superada")


if __name__ == "__main__":
    main()
