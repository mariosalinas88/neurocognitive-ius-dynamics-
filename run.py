import argparse
import csv
import os

from deps import ensure_dependencies

ensure_dependencies(["numpy", "pandas", "mesa"], context="run.py")

import numpy as np
import pandas as pd

from model import Citizen, SocietyModel


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--spectrum_level", type=int, choices=[1, 2, 3], default=None)
parser.add_argument("--initial_moral_bias", type=str, choices=["high_dark", "low_dark", "high_prosocial"], default=None)
parser.add_argument("--resilience_bias", type=str, choices=["high", "low"], default=None)
parser.add_argument("--emotional_bias", type=str, choices=["high", "low"], default=None)
parser.add_argument("--enable_reproduction", action="store_true", default=False)
parser.add_argument("--enable_sexual_selection", action="store_true", default=False)
parser.add_argument("--male_violence_multiplier", type=float, default=1.2)
parser.add_argument("--coalition_enabled", action="store_true", default=False)
args, unknown = parser.parse_known_args()

model = SocietyModel(
    width=30,
    height=30,
    seed=42,
    climate="stable",
    spectrum_level=args.spectrum_level,
    initial_moral_bias=args.initial_moral_bias,
    resilience_bias=args.resilience_bias,
    emotional_bias=args.emotional_bias,
    enable_reproduction=args.enable_reproduction,
    enable_sexual_selection=args.enable_sexual_selection,
    male_violence_multiplier=args.male_violence_multiplier,
    coalition_enabled=args.coalition_enabled,
)

print("Iniciando simulación evolutiva...")
if getattr(model, "weight_warning", False):
    print(f"Pesos normalizados (suma original={model.weight_sum_original:.3f})")

for step in range(args.steps):
    model.step()
    if step % 50 == 0:
        print(f"Step {step}: Régimen -> {model.regime} | Formalismo: {model.legal_formalism:.2f} | Libertad: {model.liberty_index:.2f}")

print("\n" + "=" * 30 + " REPORTE DE NEUROPLASTICIDAD " + "=" * 30)
agents = [a for a in model.agents if isinstance(a, Citizen) and a.alive]

neutral_keys = {"attn_selective", "attn_flex", "hyperfocus", "impulsivity", "risk_aversion", "sociality", "language", "reasoning", "emotional_impulsivity", "resilience", "sexual_impulsivity"}
moral_keys = {
    "empathy",
    "dominance",
    "affect_reg",
    "aggression",
    "moral_prosocial",
    "moral_common_good",
    "moral_honesty",
    "moral_spite",
    "dark_narc",
    "dark_mach",
    "dark_psycho",
}


def print_block(block_keys, title):
    print(f"\n-- {title} --")
    print(f"{'Rasgo':<20} | {'Inicial':<10} | {'Final':<10} | {'Cambio %':<10}")
    print("-" * 65)
    for trait in sorted(block_keys):
        avg_init = np.mean([a.original_latent.get(trait, 0.5) for a in agents])
        avg_curr = np.mean([a.latent.get(trait, 0.5) for a in agents])
        delta_pct = ((avg_curr - avg_init) / avg_init) * 100 if avg_init > 0 else 0
        print(f"{trait:<20} | {avg_init:.3f}      | {avg_curr:.3f}      | {delta_pct:+.2f}%")


if agents:
    print_block(neutral_keys, "Cambios Neutrales/Biológicos")
    print_block(moral_keys, "Cambios Morales/Emocionales")

    avg_happy = np.mean([a.happiness for a in agents])
    avg_wealth = np.mean([a.wealth for a in agents])
    avg_dark_core = np.mean([a.dark_core for a in agents])
    nd_contrib = np.mean([a.nd_contribution for a in agents]) if agents else 0.0
    nd_costs = np.mean([a.nd_cost for a in agents]) if agents else 0.0
    consciousness = np.mean([a.conscious_core.get("awareness", 0.0) for a in agents]) if agents else 0.0
    alliances = getattr(model, "alliances", {})

    profile_stats = {}
    for a in agents:
        pid = getattr(a, "profile_id", "unknown")
        entry = profile_stats.setdefault(pid, {"wealth": [], "reputation": [], "victim": 0, "leader": 0, "nd_contrib": [], "nd_cost": []})
        entry["wealth"].append(a.wealth)
        entry["reputation"].append(a.reputation_coop - a.reputation_fear)
        entry["nd_contrib"].append(a.nd_contribution)
        entry["nd_cost"].append(a.nd_cost)
        if a.last_action == "violence":
            entry["leader"] += 1
            entry["victim"] += 1
    print("\n-- Métricas por perfil (wealth, reputación, liderazgo, víctimas, ND contrib/costo) --")
    for pid, data in profile_stats.items():
        w_mean = np.mean(data["wealth"]) if data["wealth"] else 0.0
        r_mean = np.mean(data["reputation"]) if data["reputation"] else 0.0
        leaders = data["leader"]
        victims = data["victim"]
        ndc = np.mean(data["nd_contrib"]) if data["nd_contrib"] else 0.0
        ndcost = np.mean(data["nd_cost"]) if data["nd_cost"] else 0.0
        print(f"Perfil {pid}: wealth={w_mean:.2f}, rep={r_mean:.2f}, lider/violencia={leaders}, victimas={victims}, nd_contrib={ndc:.3f}, nd_cost={ndcost:.3f}")

    print(f"\nFelicidad Promedio Final: {avg_happy:.3f}")
    print(f"Wealth Promedio Final: {avg_wealth:.3f}")
    print(f"Dark core medio: {avg_dark_core:.3f}")
    print(f"ND contribución media: {nd_contrib:.3f} | ND costo: {nd_costs:.3f}")
    print(f"Conciencia media: {consciousness:.3f}")
    print(f"Alianzas activas: {len(alliances)}")
    for aid, data in alliances.items():
        members = data.get("members", [])
        goal = data.get("goal", "")
        rule = data.get("rule", "")
        print(f" - {aid}: miembros={len(members)} objetivo={goal} norma={rule}")
    print(f"Régimen Final Establecido: {model.regime}")
else:
    print("¡La sociedad se ha extinguido!")

df = model.datacollector.get_model_vars_dataframe()
os.makedirs("results", exist_ok=True)
try:
    df.to_csv("results/summary_evolution.csv")
except PermissionError:
    alt = f"results/summary_evolution_{int(np.random.randint(1e9))}.csv"
    df.to_csv(alt)

# Export per-profile stats CSV
if agents:
    with open("results/per_profile_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["profile", "wealth_avg", "leadership_avg", "rep_avg", "victims_avg", "nd_contrib_avg", "nd_cost_avg"])
        writer.writeheader()
        for pid, data in profile_stats.items():
            writer.writerow(
                {
                    "profile": pid,
                    "wealth_avg": np.mean(data["wealth"]) if data["wealth"] else 0.0,
                    "leadership_avg": data["leader"],
                    "rep_avg": np.mean(data["reputation"]) if data["reputation"] else 0.0,
                    "victims_avg": data["victim"],
                    "nd_contrib_avg": np.mean(data["nd_contrib"]) if data["nd_contrib"] else 0.0,
                    "nd_cost_avg": np.mean(data["nd_cost"]) if data["nd_cost"] else 0.0,
                }
            )
print("Datos guardados en results/summary_evolution.csv y per_profile_stats.csv")
