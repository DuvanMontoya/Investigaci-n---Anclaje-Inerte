#!/usr/bin/env python3
"""
Validacion profunda de CSVs canonicos de C_sigma inerte (Heegner).

Checks:
1) Estructura de malla R (impares, paso constante par, estrictamente creciente).
2) Identidad exacta: C_sigma_inerte = N_R * log(R) / (H_R * R).
3) Recomputo de H_R por factorizacion y caracter cuadratico chi_D.
4) Recomputo bruto de N_R en radios pequenos (R <= max_R_check).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


HEEGNER_D = [-3, -4, -7, -8, -11, -19, -43, -67, -163]


def chi_D_prime(D: int, p: int) -> int:
    if p == 2:
        if (D & 1) == 0:
            return 0
        r = D % 8
        return 1 if r in (1, 7) else -1
    if D % p == 0:
        return 0
    t = pow(D % p, (p - 1) // 2, p)
    return 1 if t == 1 else -1


def factorize(n: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    t = int(n)
    p = 2
    while p * p <= t:
        if t % p == 0:
            e = 0
            while t % p == 0:
                t //= p
                e += 1
            out.append((p, e))
        p = 3 if p == 2 else p + 2
    if t > 1:
        out.append((t, 1))
    return out


def H_from_R(D: int, R: int) -> float:
    h = 1.0
    for p, e in factorize(R):
        if chi_D_prime(D, p) != -1:
            continue
        if p == 2:
            sigma_m = 1.0 - 1.0 / (3.0 * (2.0 ** e))
            sigma_0 = 2.0 / 3.0
        else:
            sigma_m = 1.0 - 2.0 / ((p ** e) * (p + 1.0))
            sigma_0 = (p - 1.0) / (p + 1.0)
        h *= sigma_m / sigma_0
    return h


def is_norm_inerte_only(D: int, n: int) -> bool:
    if n <= 0:
        return False
    for p, e in factorize(n):
        if chi_D_prime(D, p) == -1 and (e % 2 == 1):
            return False
    return True


def N_bruteforce(D: int, R: int) -> int:
    rr = R * R
    cnt = 0
    for c in range(1, R):
        n = rr - c * c
        if is_norm_inerte_only(D, n):
            cnt += 1
    return cnt


@dataclass
class ValidationResult:
    D: int
    n_rows: int
    max_abs_identity: float
    max_abs_H: float
    max_abs_N_small: int


def validate_csv(path: Path, max_R_check: int) -> ValidationResult:
    df = pd.read_csv(path)
    req = {"D", "R", "N_R", "H_R"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")

    c_col = "C_sigma_inerte"
    if c_col not in df.columns:
        if "C_sigma" in df.columns:
            c_col = "C_sigma"
        else:
            raise ValueError(f"{path.name}: falta columna C_sigma_inerte/C_sigma")

    Dvals = df["D"].astype(int).unique().tolist()
    if len(Dvals) != 1:
        raise ValueError(f"{path.name}: D no es constante")
    D = int(Dvals[0])

    R = df["R"].to_numpy(dtype=np.int64)
    if np.any(R <= 0):
        raise ValueError(f"{path.name}: hay R no positivos")
    if np.any((R % 2) == 0):
        raise ValueError(f"{path.name}: hay R pares")
    dR = np.diff(R)
    if dR.size > 0:
        if np.any(dR <= 0):
            raise ValueError(f"{path.name}: R no es estrictamente creciente")
        step = int(dR[0])
        if (step % 2) != 0:
            raise ValueError(f"{path.name}: paso impar en R ({step})")
        if not np.all(dR == step):
            raise ValueError(f"{path.name}: paso no constante en R")

    N = df["N_R"].to_numpy(dtype=np.int64)
    H = df["H_R"].to_numpy(dtype=float)
    C = df[c_col].to_numpy(dtype=float)

    C_re = N.astype(float) * np.log(R.astype(float)) / (H * R.astype(float))
    max_abs_identity = float(np.max(np.abs(C_re - C)))

    H_re = np.array([H_from_R(D, int(r)) for r in R], dtype=float)
    max_abs_H = float(np.max(np.abs(H_re - H)))

    max_abs_N_small = 0
    small = df[df["R"] <= int(max_R_check)]
    for row in small.itertuples(index=False):
        n_brute = N_bruteforce(D, int(row.R))
        diff = abs(int(row.N_R) - n_brute)
        if diff > max_abs_N_small:
            max_abs_N_small = diff

    return ValidationResult(
        D=D,
        n_rows=int(len(df)),
        max_abs_identity=max_abs_identity,
        max_abs_H=max_abs_H,
        max_abs_N_small=max_abs_N_small,
    )


def discover_csvs(csv_dir: Path) -> List[Path]:
    files = sorted(csv_dir.glob("csigma_inerte_D*_Rmax*_step*_Rmin*.csv"))
    if not files:
        files = sorted(csv_dir.glob("csigma_inerte_D*_Rmax*_step*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron CSVs canonicos en {csv_dir}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=str, default="Datos/Canonicos")
    ap.add_argument("--max-R-check", type=int, default=401)
    ap.add_argument("--tol-identity", type=float, default=1e-9)
    ap.add_argument("--tol-H", type=float, default=1e-9)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    files = discover_csvs(csv_dir)

    results: List[ValidationResult] = []
    for fp in files:
        res = validate_csv(fp, max_R_check=int(args.max_R_check))
        results.append(res)

    Ds = sorted(r.D for r in results)
    if Ds != sorted(HEEGNER_D):
        raise ValueError(f"Conjunto D inesperado: {Ds}")

    for r in results:
        if r.max_abs_identity > float(args.tol_identity):
            raise ValueError(
                f"D={r.D}: max_abs_identity={r.max_abs_identity:.3e} > tol={args.tol_identity:.1e}"
            )
        if r.max_abs_H > float(args.tol_H):
            raise ValueError(f"D={r.D}: max_abs_H={r.max_abs_H:.3e} > tol={args.tol_H:.1e}")
        if r.max_abs_N_small != 0:
            raise ValueError(f"D={r.D}: mismatch en N_R bruto (max diff={r.max_abs_N_small})")

    print("OK validate_heegner")
    for r in sorted(results, key=lambda x: x.D):
        print(
            f"D={r.D:4d} rows={r.n_rows:7d} "
            f"max|C-id|={r.max_abs_identity:.3e} "
            f"max|H|={r.max_abs_H:.3e} "
            f"max|N_small|={r.max_abs_N_small}"
        )


if __name__ == "__main__":
    main()
