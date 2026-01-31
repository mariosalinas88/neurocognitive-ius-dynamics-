import csv
import os
import numpy as np

from model import SocietyModel, Citizen


SCENARIOS = [
    {"name": "all_nt", "profiles": [("1", 1.0)], "notes": "Todos neurotípicos"},
    {"name": "mixed_nt_aut_adhd", "profiles": [("1", 0.4), ("3", 0.3), ("2", 0.3)], "notes": "NT + Autismo + ADHD"},
    {"name": "high_reasoning_emp", "profiles": [("3", 0.5), ("11", 0.5)], "initial_moral_bias": "high_prosocial", "notes": "Alto razonamiento empático"},
    {"name": "high_reasoning_dark", "profiles": [("3", 0.4), ("10", 0.6)], "initial_moral_bias": "high_dark", "notes": "Alto razonamiento oscuro"},
    {"name": "low_reasoning", "profiles": [("4", 0.5), ("12", 0.5)], "notes": "Dislexia + Depresión"},
    {"name": "inclusive_resilient", "profiles": [("1", 0.3), ("6", 0.3), ("11", 0.4)], "resilience_bias": "high", "notes": "Mixto inclusivo alta resiliencia"},
    {"name": "competitive_low_res", "profiles": [("7", 0.4), ("10", 0.3), ("12", 0.3)], "resilience_bias": "low", "notes": "Competitivo baja resiliencia"},
    {"name": "high_nd_mix", "profiles": [("3", 0.3), ("2", 0.3), ("6", 0.4)], "notes": "Autismo + ADHD + HSP"},
    {"name": "high_reasoning_supported", "profiles": [("3", 0.5), ("11", 0.3), ("9", 0.2)], "resilience_bias": "high", "initial_moral_bias": "high_prosocial", "notes": "Alto razonamiento apoyado"},
    {"name": "nd_low_support", "profiles": [("3", 0.3), ("2", 0.3), ("12", 0.4)], "resilience_bias": "low", "notes": "ND mixto bajo apoyo"},
]


def build_model(cfg, seed):
    weights = cfg["profiles"]
    pids = [weights[0][0] if len(weights) > 0 else "", weights[1][0] if len(weights) > 1 else "", weights[2][0] if len(weights) > 2 else ""]
    ws = [weights[0][1] if len(weights) > 0 else 0.0, weights[1][1] if len(weights) > 1 else 0.0, weights[2][1] if len(weights) > 2 else 0.0]
    return SocietyModel(
        seed=seed,
        climate="stable",
        external_factor="none",
        population_scale="tiny",
        profile1=pids[0],
        weight1=ws[0],
        profile2=pids[1],
        weight2=ws[1],
        profile3=pids[2],
        weight3=ws[2],
        spectrum_level=None,
        initial_moral_bias=cfg.get("initial_moral_bias"),
        resilience_bias=cfg.get("resilience_bias"),
        emotional_bias=cfg.get("emotional_bias"),
    )


def run_scenario(cfg, seed):
    model = build_model(cfg, seed)
    for _ in range(200):
        if not model.running:
            break
        model.step()
    alive = [a for a in model.agents if isinstance(a, Citizen) and a.alive]
    total_wealth = model.total_wealth
    gini = model.gini_wealth
    regime = model.regime
    top5_share, _, _ = model._top5_power()
    victims = sum(1 for a in alive if a.last_action == "violence")
    rep_avg = np.mean([a.reputation_coop - a.reputation_fear for a in alive]) if alive else 0.0
    # per-profile dominance
    prof_stats = {}
    for a in alive:
        pid = getattr(a, "profile_id", "n/a")
        prof_stats.setdefault(pid, {"wealth": [], "lead": [], "rep": [], "victims": 0, "nd_contrib": []})
        prof_stats[pid]["wealth"].append(a.wealth)
        prof_stats[pid]["rep"].append(a.reputation_coop - a.reputation_fear)
        prof_stats[pid]["nd_contrib"].append(getattr(a, "nd_contribution", 0.0))
        if a.last_action == "violence":
            prof_stats[pid]["victims"] += 1
        if a.reputation_fear > 0.6 or a.wealth > np.percentile([x.wealth for x in alive], 80) if alive else False:
            prof_stats[pid]["lead"].append(1)
    dominant_profiles = ""
    if prof_stats:
        wealth_means = [np.mean(v["wealth"]) for v in prof_stats.values()]
        threshold = np.percentile(wealth_means, 80) if wealth_means else 0
        dominant_profiles = ",".join(sorted(k for k, v in prof_stats.items() if np.mean(v["wealth"]) >= threshold))
    contrib_mean = float(np.mean([c for lst in [v["nd_contrib"] for v in prof_stats.values()] for c in lst])) if prof_stats else 0.0
    return {
        "scenario": cfg["name"],
        "notes": cfg.get("notes", ""),
        "regime": regime,
        "total_wealth": round(total_wealth, 3),
        "gini": round(gini, 3),
        "avg_rep": round(rep_avg, 3),
        "top5_share": round(top5_share, 3),
        "victims_rate": round(victims / max(len(alive), 1), 3),
        "nd_contrib": round(contrib_mean, 3),
        "dominant_profiles": dominant_profiles,
        "input_profiles": cfg.get("profiles"),
        "initial_moral_bias": cfg.get("initial_moral_bias", ""),
        "resilience_bias": cfg.get("resilience_bias", ""),
    }


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for i, cfg in enumerate(SCENARIOS):
        rows.append(run_scenario(cfg, seed=100 + i))
    out_path = "results/test_runs.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Guardado {out_path}")


if __name__ == "__main__":
    main()
