import csv
import os
import numpy as np
from model import SocietyModel, Citizen

out_dir = "results_extreme_tests"
os.makedirs(out_dir, exist_ok=True)

tests = [
    {"name": "JC_99_lev_01", "weights": {"profile1": "1", "weight1": 0.99, "profile2": "6", "weight2": 0.01, "profile3": "", "weight3": 0.0}},
    {"name": "JC_50_lev_50", "weights": {"profile1": "1", "weight1": 0.50, "profile2": "6", "weight2": 0.50, "profile3": "", "weight3": 0.0}},
    {"name": "JC_01_lev_99", "weights": {"profile1": "1", "weight1": 0.01, "profile2": "6", "weight2": 0.99, "profile3": "", "weight3": 0.0}},
]

over_rows = []
prof_rows = []

for t in tests:
    params = dict(
        seed=42,
        climate="stable",
        external_factor="none",
        population_scale="tribe",
        **t["weights"],
    )
    m = SocietyModel(**params)
    for _ in range(200):
        if not m.running:
            break
        m.step()
    alive = [a for a in m.agents if isinstance(a, Citizen) and a.alive]
    metrics = m.last_metrics or {}
    over_rows.append(
        {
            "test": t["name"],
            "profiles": t["weights"],
            "regime": getattr(m, "regime", ""),
            "alive": len(alive),
            "pop_scaled": len(alive) * m.scale_factor,
            "coop_rate": metrics.get("coop_rate", 0.0),
            "violence_rate": metrics.get("violence_rate", 0.0),
            "gini": getattr(m, "gini_wealth", 0.0),
            "avg_reasoning": metrics.get("avg_reasoning", 0.0),
            "avg_empathy": metrics.get("avg_empathy", 0.0),
            "avg_dominance": metrics.get("avg_dominance", 0.0),
            "alliances": len(getattr(m, "alliances", {})),
        }
    )
    if alive:
        per_profile = {}
        for a in alive:
            pid = getattr(a, "profile_id", "unknown")
            entry = per_profile.setdefault(pid, {"wealth": [], "rep_coop": [], "rep_fear": [], "violence": 0, "count": 0})
            entry["wealth"].append(a.wealth)
            entry["rep_coop"].append(a.reputation_coop)
            entry["rep_fear"].append(a.reputation_fear)
            entry["violence"] += 1 if a.last_action == "violence" else 0
            entry["count"] += 1
        for pid, data in per_profile.items():
            prof_rows.append(
                {
                    "test": t["name"],
                    "profile": pid,
                    "count": data["count"],
                    "wealth_mean": np.mean(data["wealth"]),
                    "rep_coop_mean": np.mean(data["rep_coop"]),
                    "rep_fear_mean": np.mean(data["rep_fear"]),
                    "violence_rate": data["violence"] / data["count"] if data["count"] else 0.0,
                    "coop_rate_run": metrics.get("coop_rate", 0.0),
                    "violence_rate_run": metrics.get("violence_rate", 0.0),
                }
            )

with open(os.path.join(out_dir, "overview_extremes.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=over_rows[0].keys())
    writer.writeheader()
    writer.writerows(over_rows)

if prof_rows:
    with open(os.path.join(out_dir, "profile_extremes.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prof_rows[0].keys())
        writer.writeheader()
        writer.writerows(prof_rows)

print("Escrito en", out_dir)
