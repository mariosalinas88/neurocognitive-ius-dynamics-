# batch_run.py — Versión 100% compatible con tu model.py actual
# Ejecuta con: python batch_run.py

import random
import pandas as pd
import numpy as np
from model import SocietyModel
import os
from datetime import datetime

# ================================================
# ESCENARIOS REALISTAS (ajustados a tu SocietyModel actual)
# ================================================

SCENARIOS = [
    ("Población General",      ["2", "1", "4", "3", "29"], [0.78, 0.08, 0.07, 0.03, 0.04], "stable",   "none"),
    ("Alta Psicopatía Corp.",  ["2", "5", "30"],           [0.85, 0.10, 0.05],           "stable",   "technological"),
    ("Sociedad Violenta",      ["2", "6"],                 [0.90, 0.10],                 "scarce",   "disaster"),
    ("Neurodivergente Pura",   ["1", "3", "4", "29"],      [0.25, 0.25, 0.25, 0.25],     "abundant", "none"),
    ("Empáticos Dominantes",   ["2", "29", "27"],          [0.70, 0.20, 0.10],           "abundant", "none"),
    ("Crisis Epidémica",       ["2", "5", "6", "29"],      [0.80, 0.08, 0.07, 0.05],     "scarce",   "epidemic"),
    ("Tecnocracia Fría",       ["2", "3", "5", "30"],      [0.70, 0.10, 0.10, 0.10],     "stable",   "technological"),
    ("Comunidad Intencional",  ["29", "27", "1"],          [0.60, 0.30, 0.10],           "abundant", "none"),
    ("Colapso Tribal",         ["6"],                      [1.00],                       "scarce",   "disaster"),
    ("Utopía Empática",        ["29"],                     [1.00],                       "abundant", "none"),
]

STEPS_PER_RUN = 400
SEED_BASE = 42
RESULTS_DIR = "batch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []

print(f"Iniciando {len(SCENARIOS)} escenarios × 10 repeticiones = {len(SCENARIOS)*10} simulaciones...\n")

for idx, (name, profiles, weights, climate, external) in enumerate(SCENARIOS):
    print(f"[{idx+1}/{len(SCENARIOS)}] {name}")
    
    for rep in range(10):
        seed = SEED_BASE + rep + idx * 100
        
        # Ajuste para 1, 2 o 3 perfiles
        p1 = profiles[0] if len(profiles) >= 1 else "2"
        w1 = weights[0] if len(weights) >= 1 else 1.0
        p2 = profiles[1] if len(profiles) >= 2 else ""
        w2 = weights[1] if len(weights) >= 2 else 0.0
        p3 = profiles[2] if len(profiles) >= 3 else ""
        w3 = weights[2] if len(profiles) >= 3 else 0.0
        
        model = SocietyModel(
            seed=seed,
            profile1=p1,
            weight1=w1,
            profile2=p2,
            weight2=w2,
            profile3=p3,
            weight3=w3,
            climate=climate,
            external_factor=external
        )
        
        for step in range(STEPS_PER_RUN):
            model.step()
            if not model.running:
                break
        
        alive = len([a for a in model.schedule.agents if a.alive]) if hasattr(model, "schedule") else 0
        initial = 120  # tu modelo usa población fija ~120 en tribe
        
        results.append({
            "run": len(results) + 1,
            "scenario": name,
            "replica": rep + 1,
            "profiles": "/".join(profiles),
            "weights": "/".join(f"{w:.2f}" for w in weights),
            "climate": climate,
            "external": external,
            "steps_run": model.current_step if hasattr(model, "current_step") else STEPS_PER_RUN,
            "population_final": alive,
            "survival_rate": alive / initial if initial > 0 else 0,
            "regime_final": getattr(model, "regime", "Desconocido"),
            "violence_final": model.last_metrics.get("violence_rate", 0.0) if hasattr(model, "last_metrics") else 0.0,
            "coop_final": model.last_metrics.get("coop_rate", 0.0) if hasattr(model, "last_metrics") else 0.0,
            "gini_wealth": getattr(model, "gini_wealth", 0.0),
            "legal_formalism": getattr(model, "legal_formalism", 0.0),
            "liberty_index": getattr(model, "liberty_index", 0.0),
        })
        
