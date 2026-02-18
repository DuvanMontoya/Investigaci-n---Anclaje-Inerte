#!/usr/bin/env python3
"""
GenerarConteosPrimarios.py
==========================

Primary data generator for:
  N_R, H_R, C_sigma_inerte
used in the inert-normalized pipeline.

Design goals:
1) Reproducible CLI execution (seeded when stochastic sampling is used).
2) Support the legacy principal grid and a complementary grid that
   explicitly explores 2-adic strata and v5(R) > 0 strata.

Notes:
- The generated observable follows the same inert-only convention used in
  current CSV artifacts.
- Counting can be exact (slow), Monte Carlo (seeded), or hybrid.
- By default output goes to Datos/Generados to avoid mixing files with the
  legacy principal CSV set consumed by other scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


HEEGNER_DS_DEFAULT = [-3, -4, -7, -8, -11, -19, -43, -67, -163]


@dataclass(frozen=True)
class CountResult:
    n_est: float
    n_hits: int
    n_samples: int
    se_n: float
    mode_used: str


def parse_d_list(text: str) -> List[int]:
    vals: List[int] = []
    for token in str(text).replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("D_list is empty.")
    return sorted(set(vals))


def parse_strata(text: str) -> List[Tuple[int, int]]:
    """
    Parse e.g. "1:0,2:0,0:1,1:1,2:1,0:2" -> [(0,1), (0,2), (1,0), ...]
    """
    out: List[Tuple[int, int]] = []
    for token in str(text).replace(";", ",").split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f"Invalid stratum token: {t!r}. Expected 'v2:v5'.")
        a, b = t.split(":", 1)
        v2 = int(a.strip())
        v5 = int(b.strip())
        if v2 < 0 or v5 < 0:
            raise ValueError(f"Invalid negative valuation in stratum token: {t!r}")
        out.append((v2, v5))
    if not out:
        raise ValueError("No strata parsed for complementary grid.")
    return sorted(set(out))


def vp_int(n: int, p: int) -> int:
    t = abs(int(n))
    if t == 0:
        return 0
    v = 0
    while t % p == 0:
        t //= p
        v += 1
    return v


def primes_upto(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    lim = int(n**0.5)
    for p in range(2, lim + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = False
    return [int(p) for p in np.flatnonzero(sieve)]


def factorize_abs(n: int, primes: Sequence[int]) -> List[Tuple[int, int]]:
    t = abs(int(n))
    if t <= 1:
        return []
    out: List[Tuple[int, int]] = []
    for p in primes:
        if p * p > t:
            break
        if t % p == 0:
            e = 0
            while t % p == 0:
                t //= p
                e += 1
            out.append((p, e))
    if t > 1:
        out.append((int(t), 1))
    return out


def chi_delta_p(delta: int, p: int) -> int:
    """
    Kronecker/Legaendre-like character used in the current pipeline.
    """
    if p == 2:
        if delta % 2 == 0:
            return 0
        return -1 if delta % 8 == 5 else 1
    if delta % p == 0:
        return 0
    t = pow(delta % p, (p - 1) // 2, p)
    return 1 if t == 1 else -1


def sigma_ratio_odd_inert(p: int, m: int) -> float:
    # sigma_m = 1 - 2/(p^m (p+1)), sigma_0 = (p-1)/(p+1)
    sigma_m = 1.0 - 2.0 / (float(p) ** m * (p + 1.0))
    sigma_0 = (p - 1.0) / (p + 1.0)
    return sigma_m / sigma_0


def sigma_ratio_two_inert(m: int) -> float:
    # 2-inert analogue used in manuscript checks:
    # P(v2 odd | m) = 1/(3*2^m) => sigma_2^(m) = 1 - 1/(3*2^m), sigma_2^(0)=2/3.
    sigma_m = 1.0 - 1.0 / (3.0 * (2.0**m))
    sigma_0 = 2.0 / 3.0
    return sigma_m / sigma_0


def h_delta_inerte_only(delta: int, R: int, primes: Sequence[int], include_two_inert: bool) -> float:
    val = 1.0
    for p, m in factorize_abs(R, primes):
        if p == 2:
            if include_two_inert and chi_delta_p(delta, 2) == -1:
                val *= sigma_ratio_two_inert(m)
        elif chi_delta_p(delta, p) == -1:
            val *= sigma_ratio_odd_inert(p, m)
    return float(val)


def is_norm_inerte_only(delta: int, n: int, primes: Sequence[int], include_two_inert: bool) -> bool:
    """
    Inert-only local filter:
    - odd inert primes require even valuation in n;
    - p=2 handled only when 2 is inert and include_two_inert=True.
    """
    if n <= 0:
        return False
    for p, e in factorize_abs(n, primes):
        if p == 2:
            if include_two_inert and chi_delta_p(delta, 2) == -1 and (e % 2 == 1):
                return False
        elif chi_delta_p(delta, p) == -1 and (e % 2 == 1):
            return False
    return True


def count_exact(delta: int, R: int, primes: Sequence[int], include_two_inert: bool) -> CountResult:
    rr = R * R
    hits = 0
    for c in range(1, R):
        if is_norm_inerte_only(delta, rr - c * c, primes, include_two_inert):
            hits += 1
    return CountResult(
        n_est=float(hits),
        n_hits=int(hits),
        n_samples=int(max(0, R - 1)),
        se_n=0.0,
        mode_used="exact",
    )


def count_mc(
    delta: int,
    R: int,
    primes: Sequence[int],
    include_two_inert: bool,
    seed: int,
    samples_per_R: int,
    replace: bool,
) -> CountResult:
    pop = max(0, R - 1)
    if pop == 0:
        return CountResult(n_est=0.0, n_hits=0, n_samples=0, se_n=0.0, mode_used="mc")

    if replace:
        n_samp = int(max(1, samples_per_R))
    else:
        n_samp = int(max(1, min(samples_per_R, pop)))

    seq = np.random.SeedSequence([int(seed), int(delta), int(R)])
    rng = np.random.default_rng(seq)
    if replace:
        cs = rng.integers(low=1, high=R, size=n_samp, endpoint=False, dtype=np.int64)
    else:
        cs = rng.choice(np.arange(1, R, dtype=np.int64), size=n_samp, replace=False)

    rr = R * R
    hits = 0
    for c in cs:
        if is_norm_inerte_only(delta, rr - int(c) * int(c), primes, include_two_inert):
            hits += 1

    p_hat = float(hits) / float(n_samp)
    n_est = p_hat * float(pop)
    se_n = math.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / float(n_samp)) * float(pop)
    return CountResult(
        n_est=float(n_est),
        n_hits=int(hits),
        n_samples=int(n_samp),
        se_n=float(se_n),
        mode_used="mc",
    )


def build_principal_grid(r_min: int, r_max: int, step: int, residue_mod: int, residue: int) -> List[int]:
    if step <= 0:
        raise ValueError("principal_step must be > 0.")
    if r_min > r_max:
        return []

    start = r_min
    if residue_mod > 0:
        start = r_min + ((residue - r_min) % residue_mod)
    grid = list(range(int(start), int(r_max) + 1, int(step)))
    return [int(r) for r in grid if r >= r_min and r <= r_max]


def build_complement_grid(
    r_min: int,
    r_max: int,
    strata: Sequence[Tuple[int, int]],
    q_step: int,
    q_residue: int,
) -> List[int]:
    if r_min > r_max:
        return []
    if q_step <= 0:
        raise ValueError("complement_q_step must be > 0.")

    out: set[int] = set()
    for v2, v5 in strata:
        base = (2**int(v2)) * (5**int(v5))
        q_min = (r_min + base - 1) // base
        q_max = r_max // base
        q_start = q_min + ((q_residue - q_min) % q_step)
        for q in range(int(q_start), int(q_max) + 1, int(q_step)):
            if q % 2 == 0 or q % 5 == 0:
                continue
            R = int(base * q)
            if R < r_min or R > r_max:
                continue
            # exact valuation check for safety
            if vp_int(R, 2) == v2 and vp_int(R, 5) == v5:
                out.add(R)
    return sorted(out)


def maybe_subsample_grid(grid: Sequence[int], max_points: int, seed: int, label: str) -> List[int]:
    vals = sorted(int(x) for x in grid)
    if max_points <= 0 or len(vals) <= max_points:
        return vals
    seq = np.random.SeedSequence([int(seed), len(vals), hash(label) & 0xFFFFFFFF])
    rng = np.random.default_rng(seq)
    idx = rng.choice(len(vals), size=int(max_points), replace=False)
    return sorted(vals[int(i)] for i in idx)


def estimate_checks(
    Rs: Sequence[int],
    method: str,
    samples_per_R: int,
    exact_up_to_R: int,
) -> int:
    if method == "exact":
        return int(sum(max(0, int(R) - 1) for R in Rs))
    if method == "mc":
        return int(len(Rs) * max(1, int(samples_per_R)))
    # hybrid
    total = 0
    for R in Rs:
        if int(R) <= exact_up_to_R:
            total += max(0, int(R) - 1)
        else:
            total += max(1, int(samples_per_R))
    return int(total)


def summarize_strata_counts(Rs: Sequence[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for R in Rs:
        key = f"v2={vp_int(int(R), 2)};v5={vp_int(int(R), 5)}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def sanitize_mode_token(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def build_output_name(
    D: int,
    grid_name: str,
    method: str,
    r_min: int,
    r_max: int,
    seed: int,
) -> str:
    g = sanitize_mode_token(grid_name)
    m = sanitize_mode_token(method)
    return f"csigma_inerte_D{D}_Rmin{r_min}_Rmax{r_max}_grid-{g}_method-{m}_seed{seed}.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="Datos/Generados", help="Output directory for generated CSV files.")
    ap.add_argument("--D_list", default=",".join(str(d) for d in HEEGNER_DS_DEFAULT), help="Comma-separated discriminants.")
    ap.add_argument("--D", dest="D_values", action="append", type=int, default=None, help="Repeatable discriminant flag (can be used multiple times).")
    ap.add_argument("--R_min", type=int, default=101, help="Minimum R.")
    ap.add_argument("--R_max", type=int, default=20_001, help="Maximum R.")
    ap.add_argument("--grid", choices=["principal", "complement", "both"], default="principal", help="Grid mode.")
    ap.add_argument("--principal_step", type=int, default=20, help="Step for principal grid progression.")
    ap.add_argument("--principal_residue_mod", type=int, default=20, help="Modulus for principal residue filter.")
    ap.add_argument("--principal_residue", type=int, default=1, help="Residue for principal grid (mod principal_residue_mod).")
    ap.add_argument(
        "--complement_strata",
        default="1:0,2:0,3:0,0:1,1:1,2:1,0:2",
        help="Complement strata as v2:v5 pairs. Example: 1:0,2:0,0:1,1:1.",
    )
    ap.add_argument("--complement_q_step", type=int, default=10, help="Step in q for complement grid R=2^v2*5^v5*q.")
    ap.add_argument("--complement_q_residue", type=int, default=1, help="Residue in q progression for complement grid.")
    ap.add_argument("--max_R_points_per_grid", type=int, default=0, help="Optional random subsample cap per grid (0 means no cap).")
    ap.add_argument("--method", choices=["exact", "mc", "hybrid"], default="hybrid", help="Counting method.")
    ap.add_argument("--exact_up_to_R", type=int, default=5000, help="For hybrid mode: exact counting up to this R.")
    ap.add_argument("--samples_per_R", type=int, default=1200, help="For mc/hybrid: samples per R.")
    ap.add_argument("--sampling_without_replacement", action="store_true", help="Sample c without replacement in MC.")
    ap.add_argument("--seed", type=int, default=20260217, help="Global seed for reproducibility.")
    ap.add_argument("--include_two_inert_in_H", action="store_true", help="Include p=2 inert ratio in H_R when applicable.")
    ap.add_argument("--max_checks", type=int, default=80_000_000, help="Safety cap for estimated total checks per D and grid.")
    ap.add_argument("--force", action="store_true", help="Run even if estimated checks exceed max_checks.")
    ap.add_argument("--progress_every", type=int, default=50, help="Progress print cadence in rows.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.D_values:
        Ds = sorted(set(int(d) for d in args.D_values))
    else:
        Ds = parse_d_list(args.D_list)
    strata = parse_strata(args.complement_strata)

    grid_map: Dict[str, List[int]] = {}
    if args.grid in ("principal", "both"):
        grid_map["principal"] = build_principal_grid(
            r_min=int(args.R_min),
            r_max=int(args.R_max),
            step=int(args.principal_step),
            residue_mod=int(args.principal_residue_mod),
            residue=int(args.principal_residue),
        )
    if args.grid in ("complement", "both"):
        grid_map["complement"] = build_complement_grid(
            r_min=int(args.R_min),
            r_max=int(args.R_max),
            strata=strata,
            q_step=int(args.complement_q_step),
            q_residue=int(args.complement_q_residue),
        )

    for gname, grid in list(grid_map.items()):
        grid_map[gname] = maybe_subsample_grid(
            grid=grid,
            max_points=int(args.max_R_points_per_grid),
            seed=int(args.seed),
            label=gname,
        )
        if not grid_map[gname]:
            raise SystemExit(f"Grid {gname!r} is empty with current parameters.")

    r_max_eff = int(max(max(Rs) for Rs in grid_map.values()))
    prime_bound = int(max(3, r_max_eff))
    t0 = time.time()
    primes = primes_upto(prime_bound)
    t_pr = time.time() - t0
    print(f"Prime table ready up to {prime_bound} ({len(primes)} primes) in {t_pr:.2f}s.")

    manifest: Dict[str, object] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": vars(args),
        "D_list": Ds,
        "grid_sizes": {k: len(v) for k, v in grid_map.items()},
        "grid_strata_counts": {k: summarize_strata_counts(v) for k, v in grid_map.items()},
        "files": [],
    }

    include_two_inert = bool(args.include_two_inert_in_H)
    replace = not bool(args.sampling_without_replacement)

    for grid_name, Rs in grid_map.items():
        est = estimate_checks(
            Rs=Rs,
            method=str(args.method),
            samples_per_R=int(args.samples_per_R),
            exact_up_to_R=int(args.exact_up_to_R),
        )
        print(f"[grid={grid_name}] points={len(Rs)} estimated_checks_per_D={est:,}")
        if est > int(args.max_checks) and not args.force:
            raise SystemExit(
                f"Estimated checks {est:,} exceeds max_checks={int(args.max_checks):,}. "
                f"Use --force, lower R range, fewer points, or switch method."
            )

        for D in Ds:
            print(f"[D={D}] grid={grid_name} rows={len(Rs)}")
            rows: List[Dict[str, float]] = []
            t_d0 = time.time()
            for i, R in enumerate(Rs, start=1):
                if str(args.method) == "exact":
                    cnt = count_exact(D, int(R), primes, include_two_inert)
                elif str(args.method) == "mc":
                    cnt = count_mc(
                        D,
                        int(R),
                        primes,
                        include_two_inert,
                        seed=int(args.seed),
                        samples_per_R=int(args.samples_per_R),
                        replace=replace,
                    )
                else:
                    if int(R) <= int(args.exact_up_to_R):
                        cnt = count_exact(D, int(R), primes, include_two_inert)
                    else:
                        cnt = count_mc(
                            D,
                            int(R),
                            primes,
                            include_two_inert,
                            seed=int(args.seed),
                            samples_per_R=int(args.samples_per_R),
                            replace=replace,
                        )

                h_r = h_delta_inerte_only(D, int(R), primes, include_two_inert)
                c_sigma = float("nan")
                if R > 1 and h_r > 0:
                    c_sigma = float(cnt.n_est * math.log(float(R)) / (h_r * float(R)))

                rows.append(
                    {
                        "D": int(D),
                        "R": int(R),
                        "N_R": float(cnt.n_est),
                        "H_R": float(h_r),
                        "C_sigma_inerte": float(c_sigma),
                        "v2_R": int(vp_int(int(R), 2)),
                        "v5_R": int(vp_int(int(R), 5)),
                        "method_row": cnt.mode_used,
                        "n_samples": int(cnt.n_samples),
                        "n_hits": int(cnt.n_hits),
                        "se_N_R": float(cnt.se_n),
                        "seed": int(args.seed),
                        "grid_name": grid_name,
                    }
                )

                if args.progress_every > 0 and (i % int(args.progress_every) == 0 or i == len(Rs)):
                    print(f"  progress {i}/{len(Rs)}")

            df = pd.DataFrame(rows).sort_values("R").reset_index(drop=True)
            out_name = build_output_name(
                D=int(D),
                grid_name=grid_name,
                method=str(args.method),
                r_min=int(min(Rs)),
                r_max=int(max(Rs)),
                seed=int(args.seed),
            )
            out_path = out_dir / out_name
            df.to_csv(out_path, index=False)
            elapsed = time.time() - t_d0
            print(f"  wrote {out_path.name} ({len(df)} rows) in {elapsed:.2f}s")
            manifest["files"].append(str(out_path))

    manifest_path = out_dir / f"manifest_generate_primary_counts_seed{int(args.seed)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done.")
    print("Manifest:", manifest_path.resolve())


if __name__ == "__main__":
    main()
