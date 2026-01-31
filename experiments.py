"""Experimentos controlados para aislar perfiles cognitivos en Neuro Societies."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import pandas as pd

from model import SocietyModel


DEFAULT_PROFILES = ["1", "2", "3", "4"]


def run_pure_profile_experiment(profile_id: str, n_runs: int = 50, steps: int = 500) -> pd.DataFrame:
    results = []
    for seed in range(n_runs):
        model = SocietyModel(
            seed=seed,
            profile1=str(profile_id),
            weight1=1.0,
            profile2="",
            weight2=0.0,
            profile3="",
            weight3=0.0,
            climate="stable",
            enable_reproduction=False,
            enable_sexual_selection=False,
        )
        for _ in range(steps):
            if not model.running:
                break
            model.step()
        results.append(
            {
                "seed": seed,
                "profile_id": str(profile_id),
                "regime": model.regime,
                "gini": model.gini_wealth,
                "legal_formalism": model.legal_formalism,
                "liberty_index": model.liberty_index,
                "violence_rate": model.last_metrics.get("violence_rate", 0.0),
                "coop_rate": model.last_metrics.get("coop_rate", 0.0),
                "avg_justice_score": model.last_metrics.get("avg_justice_score", 0.0),
                "emergent_norms_count": len(getattr(model, "emergent_norms", [])),
            }
        )
    return pd.DataFrame(results)


def run_profiles(profiles: Iterable[str], n_runs: int, steps: int, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for profile_id in profiles:
        df = run_pure_profile_experiment(profile_id, n_runs=n_runs, steps=steps)
        all_results.append(df)
        df.to_csv(os.path.join(output_dir, f"profile_{profile_id}_pure_experiment.csv"), index=False)
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled Neuro Societies experiments.")
    parser.add_argument("--profiles", type=str, default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if not profiles:
        profiles = DEFAULT_PROFILES

    df = run_profiles(profiles, n_runs=args.runs, steps=args.steps, output_dir=args.output_dir)
    if df.empty:
        print("No profiles executed.")
        return
    df.to_csv(os.path.join(args.output_dir, "pure_profile_experiments_summary.csv"), index=False)
    print("Resultados guardados en", args.output_dir)


if __name__ == "__main__":
    main()
