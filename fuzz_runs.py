"""Fuzz harness for Neuro Societies (Mesa).

Ejemplos:
  python fuzz_runs.py --runs 30 --steps_min 60 --steps_max 160
  python fuzz_runs.py --runs 100 --steps_min 30 --steps_max 80 --out results_fuzz_fast
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

import model
from model import SocietyModel


def strict_load_profiles(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_expected_sets(data: dict) -> Dict[str, Set[str]]:
    profiles = data.get("profiles", [])
    rare = data.get("rare_variants", [])
    latents = set()
    bio = set()
    spec = set()
    rare_mods = set()
    config_keys = set()
    for p in profiles:
        latents |= set((p.get("latents") or {}).keys())
        bio |= set((p.get("biological_bias") or {}).keys())
        spec |= set((p.get("spectrum_ranges") or {}).keys())
        if "gauss_std" in p:
            config_keys.add("gauss_std")
    for v in rare:
        rare_mods |= set((v.get("trait_mods") or {}).keys())
    expected_all = latents | bio | spec | rare_mods
    return {
        "latents": latents,
        "bio": bio,
        "spec": spec,
        "rare": rare_mods,
        "config": config_keys,
        "all": expected_all,
    }


def random_params(profile_ids: List[str]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    climates = ["scarce", "stable", "abundant"]
    externals = ["none", "disaster", "epidemic", "technological"]
    scales_pool = ["tiny", "tiny", "tribe", "tribe", "city"]
    params["climate"] = random.choice(climates)
    params["external_factor"] = random.choice(externals)
    params["population_scale"] = random.choice(scales_pool)
    picks = random.sample(profile_ids, k=min(3, len(profile_ids)))
    while len(picks) < 3:
        picks.append(random.choice(profile_ids))
    weights_raw = [random.random() for _ in range(3)]
    total = sum(weights_raw)
    weights = [w / total for w in weights_raw]
    params["profile1"], params["profile2"], params["profile3"] = picks
    params["weight1"], params["weight2"], params["weight3"] = weights
    return params


def filter_params_for_model(params: Dict[str, object]) -> Dict[str, object]:
    sig = inspect.signature(SocietyModel.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in params.items() if k in allowed}


def run_fuzz(runs: int, steps_min: int, steps_max: int, out_dir: Path, save_evolution: bool, expected: Dict[str, Set[str]], profile_ids: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_report = {
        "expected_latents": sorted(expected["latents"]),
        "expected_biological_bias": sorted(expected["bio"]),
        "expected_spectrum_ranges": sorted(expected["spec"]),
        "expected_rare_mods": sorted(expected["rare"]),
        "profile_config_keys": sorted(expected["config"]),
        "runs": [],
    }
    run_rows = []
    used_union: Set[str] = set()

    for i in range(1, runs + 1):
        seed = random.randint(1, 10_000_000)
        params = random_params(profile_ids)
        steps = random.randint(steps_min, steps_max)
        os.environ["TRACE_LATENTS"] = "1"
        model.USED_LATENT_KEYS.clear()
        filtered = filter_params_for_model(params)
        filtered["seed"] = seed
        m = SocietyModel(**filtered)
        for _ in range(steps):
            if not m.running:
                break
            m.step()
        metrics = {
            "regime": getattr(m, "regime", ""),
            "coop_rate": m.last_metrics.get("coop_rate", 0.0),
            "violence_rate": m.last_metrics.get("violence_rate", 0.0),
            "gini_wealth": getattr(m, "gini_wealth", 0.0),
            "population": len(m.agents_alive()) * getattr(m, "scale_factor", 1.0),
        }
        used_keys_run = set(model.USED_LATENT_KEYS)
        used_union |= used_keys_run
        missing_run = expected["all"] - used_keys_run
        coverage_report["runs"].append(
            {
                "run": i,
                "seed": seed,
                "params": params,
                "steps": steps,
                "metrics": metrics,
                "used_keys": sorted(used_keys_run),
                "missing_keys": sorted(missing_run),
            }
        )
        run_rows.append(
            {
                "run": i,
                "seed": seed,
                "steps": steps,
                "regime": metrics["regime"],
                "coop_rate": metrics["coop_rate"],
                "violence_rate": metrics["violence_rate"],
                "gini_wealth": metrics["gini_wealth"],
                "population": metrics["population"],
                "used_count": len(used_keys_run),
                "missing_count": len(missing_run),
            }
        )
        if save_evolution:
            df = m.datacollector.get_model_vars_dataframe()
            df.to_csv(out_dir / f"summary_evolution_run_{i}.csv")
        model.USED_LATENT_KEYS.clear()

    missing_union = expected["all"] - used_union
    coverage_report["used_union"] = sorted(used_union)
    coverage_report["missing_union"] = sorted(missing_union)
    with (out_dir / "coverage_report.json").open("w", encoding="utf-8") as f:
        json.dump(coverage_report, f, ensure_ascii=False, indent=2)

    with (out_dir / "run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run", "seed", "steps", "regime", "coop_rate", "violence_rate", "gini_wealth", "population", "used_count", "missing_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    print(f"Expected keys: {len(expected['all'])} | Used union: {len(used_union)} | Missing: {len(missing_union)}")
    if missing_union:
        print("Missing keys:", sorted(missing_union))
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--steps_min", type=int, default=80)
    parser.add_argument("--steps_max", type=int, default=250)
    parser.add_argument("--out", type=str, default="results_fuzz")
    parser.add_argument("--save_evolution", action="store_true")
    args = parser.parse_args()

    profiles_path = Path("profiles.json")
    if not profiles_path.exists():
        print("profiles.json no encontrado")
        raise SystemExit(1)
    data = strict_load_profiles(profiles_path)
    expected = build_expected_sets(data)
    profile_ids = [str(p.get("id")) for p in data.get("profiles", []) if p.get("id") is not None]
    if not profile_ids:
        print("No hay perfiles definidos en profiles.json")
        raise SystemExit(1)

    out_dir = Path(args.out)
    run_fuzz(args.runs, args.steps_min, args.steps_max, out_dir, args.save_evolution, expected, profile_ids)


if __name__ == "__main__":
    main()
