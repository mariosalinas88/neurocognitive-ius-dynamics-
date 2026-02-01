"""Dashboard Solara para Neuro Societies (sin mesa.visualization)."""

from __future__ import annotations

import threading
import time
from typing import Dict, List

from deps import ensure_dependencies

ensure_dependencies(["matplotlib", "numpy", "pandas", "solara", "mesa"], context="server.py")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import solara

from model import Citizen, SocietyModel, load_profiles


PROFILE_MAP = load_profiles()


def profile_label(pid: str) -> str:
    meta = PROFILE_MAP.get(pid, {})
    name = meta.get("name") or ""
    return f"{pid}: {name}" if name else pid


def profile_options() -> List[str]:
    opts = [""]
    for pid in PROFILE_MAP.keys():
        opts.append(profile_label(pid))
    return opts


def profile_id_from_value(value: str) -> str:
    if value in PROFILE_MAP:
        return value
    if ":" in value:
        return value.split(":", 1)[0].strip()
    return value.strip()


def profile_description(value: str) -> str:
    pid = profile_id_from_value(value)
    return PROFILE_MAP.get(pid, {}).get("description", "") or ""


PROFILE_OPTIONS = profile_options()

DEFAULT_PARAMS: Dict[str, object] = {
    "seed": 42,
    "profile1": PROFILE_OPTIONS[1] if len(PROFILE_OPTIONS) > 1 else "",
    "weight1": 0.6,
    "profile2": PROFILE_OPTIONS[2] if len(PROFILE_OPTIONS) > 2 else "",
    "weight2": 0.3,
    "profile3": PROFILE_OPTIONS[3] if len(PROFILE_OPTIONS) > 3 else "",
    "weight3": 0.1,
    "climate": "stable",
    "external_factor": "none",
    "population_scale": "tribe",
    "enable_reproduction": False,
    "enable_sexual_selection": False,
    "male_violence_multiplier": 1.2,
    "coalition_enabled": False,
    "mate_weight_wealth": 0.4,
    "mate_weight_dom": 0.3,
    "mate_weight_health": 0.2,
    "mate_weight_age": 0.1,
    "mate_choice_beta": 1.0,
    "female_repro_cooldown": 10,
    "male_repro_cooldown": 2,
    "repro_base_offset": 0.2,
    "repro_desire_scale": 0.3,
    "male_initiation_base": 0.05,
    "male_desire_scale": 0.3,
    "neuro_decay_k": 0.1,
    "bonding_steps": 5,
    "bonding_delta": 0.02,
    "enable_coercion": False,
}


def make_grid_figure(model: SocietyModel):
    fig = Figure(figsize=(5.5, 5.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    xs, ys, colors, sizes = [], [], [], []
    for agent in model.agents_alive():
        if not isinstance(agent, Citizen) or agent.pos is None:
            continue
        x, y = agent.pos
        xs.append(x)
        ys.append(y)
        if getattr(agent, "alliance_id", None):
            seed = abs(hash(agent.alliance_id)) % (2**32)
            rng = np.random.default_rng(seed)
            base = rng.random(3) * 0.5 + 0.4
            colors.append(tuple(base.tolist()))
        else:
            r = float(np.clip(agent.latent.get("dominance", 0.5), 0, 1))
            g = float(np.clip(agent.latent.get("empathy", 0.5), 0, 1))
            b = float(np.clip(agent.latent.get("language", 0.5), 0, 1))
            colors.append((r, g, b))
        sizes.append(40 * float(np.clip(agent.wealth, 0.2, 3.0)))

    ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.set_xlim(-0.5, model.grid.width - 0.5)
    ax.set_ylim(-0.5, model.grid.height - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Grid: color=dom/empa/idioma, tamano=riqueza")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.invert_yaxis()
    return fig


def make_line_figure(history: pd.DataFrame, column: str, title: str, color: str = "#2563eb"):
    fig = Figure(figsize=(4.5, 3))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    if column in history:
        ax.plot(history.index, history[column], color=color, linewidth=2)
        ax.set_ylabel(column)
    else:
        ax.text(0.5, 0.5, "sin datos", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


@solara.component
def InfoPanel(model: SocietyModel):
    m = model.last_metrics or {}
    return solara.Card(
        title="Metricas",
        children=[
            solara.Markdown(f"**Regimen**: {model.regime}"),
            solara.Markdown(f"**Formalismo legal**: {model.legal_formalism:.3f} | **Libertad**: {model.liberty_index:.3f}"),
            solara.Markdown(f"**Cooperacion**: {m.get('coop_rate', 0):.2%} | **Violencia**: {m.get('violence_rate', 0):.2%}"),
            solara.Markdown(f"**Violencia M/F**: {m.get('male_violence_rate', 0):.2%} / {m.get('female_violence_rate', 0):.2%}"),
            solara.Markdown(f"**Poblacion (escala)**: {len(model.agents_alive()) * model.scale_factor:,.0f}"),
            solara.Markdown(f"**Gini**: {getattr(model, 'gini_wealth', 0.0):.3f} | **Vida**: {m.get('life_expectancy', 0):.1f}"),
            solara.Markdown(f"**Alianzas**: {m.get('alliances_count', len(getattr(model, 'alliances', {})))} | **Miembros**: {m.get('allied_share', 0):.2%}"),
            solara.Markdown(f"**Conciencia media**: {m.get('conscious_awareness', 0):.3f}"),
            solara.Markdown(f"**Suma de pesos perfiles**: {getattr(model, 'weight_sum_original', 1.0):.3f} (normalizados auto)"),
            solara.Markdown(f"**Nacimientos(step)**: {m.get('births', 0):.1f} | **Sex ratio**: {m.get('sex_ratio', 0):.2f}"),
            solara.Markdown(f"**Repro Gini M**: {m.get('repro_gini_males', 0):.3f} | **Childless M**: {m.get('male_childless_share', 0):.2%}"),
        ],
    )


def summary_text(model: SocietyModel, history: pd.DataFrame, window: int = 30) -> str:
    if history is None or history.empty:
        return "Sin datos."
    tail = history.tail(window)
    last = tail.iloc[-1]
    def pct(val): return f"{val:.2%}"
    coop_mean = pct(tail["coop_rate"].mean()) if "coop_rate" in tail else "n/d"
    viol_mean = pct(tail["violence_rate"].mean()) if "violence_rate" in tail else "n/d"
    gini_last = f"{last.get('gini_wealth', 0):.3f}"
    regime = last.get("regime", model.regime)
    alliances = last.get("alliances_count", 0)
    awareness = last.get("conscious_awareness_mean", last.get("conscious_awareness", 0))
    return "\n".join(
        [
            f"- Regimen: **{regime}**",
            f"- Coop media (ult. {len(tail)}): {coop_mean} | Violencia: {viol_mean}",
            f"- Gini: {gini_last}",
            f"- Alianzas: {alliances} | Conciencia: {awareness:.3f}",
        ]
    )


def profile_metrics(model: SocietyModel) -> pd.DataFrame:
    rows = []
    for a in model.agents_alive():
        pid = getattr(a, "profile_id", "n/a")
        rows.append(
            {
                "profile": pid,
                "wealth": a.wealth,
                "rep_net": a.reputation_coop - a.reputation_fear,
                "fear": a.reputation_fear,
                "coop": a.reputation_coop,
                "dark_core": a.dark_core,
                "violence": 1 if a.last_action == "violence" else 0,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("profile").agg(
        wealth_mean=("wealth", "mean"),
        wealth_gini=("wealth", lambda s: float(np.abs(np.subtract.outer(s, s)).sum() / (2 * len(s) ** 2 * (s.mean() or 1e-6)))),
        rep_mean=("rep_net", "mean"),
        fear_mean=("fear", "mean"),
        coop_mean=("coop", "mean"),
        dark_core_mean=("dark_core", "mean"),
        violence_rate=("violence", "mean"),
        count=("wealth", "count"),
    )
    return agg.reset_index()


@solara.component
def SummaryPanel(model: SocietyModel, history: pd.DataFrame):
    with solara.Card(title="Resumen"):
        solara.Markdown(summary_text(model, history))
        cols = [
            "population",
            "coop_rate",
            "violence_rate",
            "male_violence_rate",
            "female_violence_rate",
            "gini_wealth",
            "life_expectancy",
            "legal_formalism",
            "liberty_index",
            "total_wealth",
            "nd_contribution_mean",
            "alliances_count",
            "allied_share",
            "conscious_awareness_mean",
            "births",
            "sex_ratio",
            "coalition_count",
            "sneaky_success_rate",
            "mating_inequality",
            "repro_gini_males",
            "male_childless_share",
        ]
        cols = [c for c in cols if c in history.columns]
        if cols:
            solara.DataFrame(history[cols].tail(8).reset_index(drop=True))
        prof_df = profile_metrics(model)
        if not prof_df.empty:
            solara.Markdown("**Perfiles (wealth/rep/fear/coop/dark/violencia)**")
            solara.DataFrame(prof_df)


@solara.component
def Controls(params_state, reset_model):
    p = params_state.value

    def set_param(key, value):
        params_state.value = {**params_state.value, key: value}

    solara.Markdown("### Parametros")
    solara.InputInt("Semilla", value=int(p["seed"]), on_value=lambda v: set_param("seed", int(v)))
    solara.Select("Clima", value=p["climate"], values=["scarce", "stable", "abundant"], on_value=lambda v: set_param("climate", v))
    solara.Select(
        "Factor externo",
        value=p["external_factor"],
        values=["none", "disaster", "epidemic", "technological"],
        on_value=lambda v: set_param("external_factor", v),
    )
    solara.Select(
        "Escala",
        value=p["population_scale"],
        values=["tiny", "tribe", "city", "nation"],
        on_value=lambda v: set_param("population_scale", v),
    )
    solara.Checkbox(label="Reproduccion", value=bool(p["enable_reproduction"]), on_value=lambda v: set_param("enable_reproduction", bool(v)))
    solara.Checkbox(label="Seleccion sexual", value=bool(p["enable_sexual_selection"]), on_value=lambda v: set_param("enable_sexual_selection", bool(v)))
    solara.SliderFloat("Violencia masculina x", value=float(p["male_violence_multiplier"]), min=0.5, max=2.0, step=0.05, on_value=lambda v: set_param("male_violence_multiplier", float(v)))
    solara.Checkbox(label="Coaliciones", value=bool(p["coalition_enabled"]), on_value=lambda v: set_param("coalition_enabled", bool(v)))
    solara.Markdown("### Preferencia femenina (pesos softmax)")
    solara.SliderFloat("w_wealth", value=float(p["mate_weight_wealth"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("mate_weight_wealth", float(v)))
    solara.SliderFloat("w_dom", value=float(p["mate_weight_dom"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("mate_weight_dom", float(v)))
    solara.SliderFloat("w_health", value=float(p["mate_weight_health"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("mate_weight_health", float(v)))
    solara.SliderFloat("w_age", value=float(p["mate_weight_age"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("mate_weight_age", float(v)))
    solara.SliderFloat("beta", value=float(p["mate_choice_beta"]), min=0.0, max=3.0, step=0.1, on_value=lambda v: set_param("mate_choice_beta", float(v)))
    solara.Markdown("### Reproduccion / deseo")
    solara.SliderInt("Cooldown F", value=int(p["female_repro_cooldown"]), min=1, max=30, step=1, on_value=lambda v: set_param("female_repro_cooldown", int(v)))
    solara.SliderInt("Cooldown M", value=int(p["male_repro_cooldown"]), min=1, max=20, step=1, on_value=lambda v: set_param("male_repro_cooldown", int(v)))
    solara.SliderFloat("Base offset", value=float(p["repro_base_offset"]), min=-0.5, max=0.5, step=0.05, on_value=lambda v: set_param("repro_base_offset", float(v)))
    solara.SliderFloat("Desire scale", value=float(p["repro_desire_scale"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("repro_desire_scale", float(v)))
    solara.Markdown("### Iniciativa masculina")
    solara.SliderFloat("init_base", value=float(p["male_initiation_base"]), min=0.0, max=0.5, step=0.02, on_value=lambda v: set_param("male_initiation_base", float(v)))
    solara.SliderFloat("init_desire_scale", value=float(p["male_desire_scale"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("male_desire_scale", float(v)))
    solara.Markdown("### Neuro/bonding")
    solara.SliderFloat("neuro_decay_k", value=float(p["neuro_decay_k"]), min=0.01, max=0.5, step=0.01, on_value=lambda v: set_param("neuro_decay_k", float(v)))
    solara.SliderInt("bonding_steps", value=int(p["bonding_steps"]), min=0, max=20, step=1, on_value=lambda v: set_param("bonding_steps", int(v)))
    solara.SliderFloat("bonding_delta", value=float(p["bonding_delta"]), min=0.0, max=0.2, step=0.01, on_value=lambda v: set_param("bonding_delta", float(v)))
    solara.Checkbox(label="Enable coercion (rare)", value=bool(p["enable_coercion"]), on_value=lambda v: set_param("enable_coercion", bool(v)))

    solara.Markdown("### Perfiles")
    solara.Select("Perfil 1", value=p["profile1"], values=PROFILE_OPTIONS, on_value=lambda v: set_param("profile1", v))
    desc1 = profile_description(p["profile1"])
    if desc1:
        solara.Markdown(f"*{desc1}*")
    solara.SliderFloat("Peso 1", value=float(p["weight1"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("weight1", float(v)))
    solara.Select("Perfil 2", value=p["profile2"], values=PROFILE_OPTIONS, on_value=lambda v: set_param("profile2", v))
    desc2 = profile_description(p["profile2"])
    if desc2:
        solara.Markdown(f"*{desc2}*")
    solara.SliderFloat("Peso 2", value=float(p["weight2"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("weight2", float(v)))
    solara.Select("Perfil 3", value=p["profile3"], values=PROFILE_OPTIONS, on_value=lambda v: set_param("profile3", v))
    desc3 = profile_description(p["profile3"])
    if desc3:
        solara.Markdown(f"*{desc3}*")
    solara.SliderFloat("Peso 3", value=float(p["weight3"]), min=0.0, max=1.0, step=0.05, on_value=lambda v: set_param("weight3", float(v)))

    solara.Button("Aplicar y reiniciar", icon_name="refresh", on_click=reset_model, color="primary", text=True)


@solara.component
def Page():
    params_state = solara.use_reactive(dict(DEFAULT_PARAMS))
    sim_state = solara.use_reactive({"history": None, "steps": 0})
    model_ref = solara.use_ref(None)

    def build_model(params: Dict[str, object]):
        clean = dict(params)
        clean["profile1"] = profile_id_from_value(str(clean.get("profile1", "")))
        clean["profile2"] = profile_id_from_value(str(clean.get("profile2", "")))
        clean["profile3"] = profile_id_from_value(str(clean.get("profile3", "")))
        model = SocietyModel(**clean)
        history = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        model_ref.current = model
        sim_state.value = {"history": history, "steps": 0}

    def ensure_model():
        if model_ref.current is None:
            build_model(params_state.value)

    solara.use_effect(ensure_model, [])

    def reset_model():
        build_model(params_state.value)

    def step_model(n: int = 1):
        model = model_ref.current
        if model is None:
            return
        steps_done = 0
        for _ in range(n):
            if not model.running:
                break
            model.step()
            steps_done += 1
        history = model.datacollector.get_model_vars_dataframe().reset_index(drop=True)
        sim_state.value = {"history": history, "steps": sim_state.value["steps"] + steps_done}

    model = model_ref.current
    history = sim_state.value["history"]
    steps = sim_state.value["steps"]

    if model is None or history is None:
        solara.Text("Inicializando modelo...")
        return

    with solara.Column(gap="1.25rem"):
        solara.Markdown("# Neuro Societies - Simulacion Solara")
        with solara.Row(gap="1rem"):
            with solara.Column(gap="0.8rem", style={"minWidth": "320px"}):
                Controls(params_state=params_state, reset_model=reset_model)
                solara.Button("Step", on_click=lambda: step_model(1))
                solara.Button("Step x50", on_click=lambda: step_model(50), text=True, color="primary")
                solara.Button("Run 1000 steps", on_click=lambda: step_model(1000), text=True, color="primary")
                solara.Button("Reset modelo", on_click=reset_model, icon_name="refresh", color="warning", text=True)
                solara.Markdown(f"**Steps ejecutados**: {steps}")
            with solara.Column(gap="1rem", style={"alignItems": "stretch"}):
                InfoPanel(model)
                solara.FigureMatplotlib(make_grid_figure(model))
                SummaryPanel(model, history)
                with solara.Tabs():
                    with solara.Tab("Desarrollo ND"):
                        solara.FigureMatplotlib(make_line_figure(history, "total_wealth", "Total Wealth / Desarrollo", "#059669"))
                        solara.FigureMatplotlib(make_line_figure(history, "nd_contribution_mean", "ND Contribución media", "#22c55e"))
                    with solara.Tab("Dinámica Social"):
                        solara.FigureMatplotlib(make_line_figure(history, "coop_rate", "Cooperacion", "#16a34a"))
                        solara.FigureMatplotlib(make_line_figure(history, "violence_rate", "Violencia", "#dc2626"))
                    with solara.Tab("Economía / Poder"):
                        solara.FigureMatplotlib(make_line_figure(history, "gini_wealth", "Gini riqueza", "#9333ea"))
                        solara.FigureMatplotlib(make_line_figure(history, "top5_wealth_share", "Top5 share", "#f59e0b"))
                    with solara.Tab("Reproducción"):
                        solara.FigureMatplotlib(make_line_figure(history, "repro_gini_males", "Repro Gini (machos)", "#f97316"))
                        solara.FigureMatplotlib(make_line_figure(history, "male_childless_share", "Machos sin hijos", "#0ea5e9"))
                with solara.Row():
                    solara.FigureMatplotlib(make_line_figure(history, "population", "Poblacion", "#2563eb"))
                    solara.FigureMatplotlib(make_line_figure(history, "coop_rate", "Cooperacion", "#16a34a"))
                with solara.Row():
                    solara.FigureMatplotlib(make_line_figure(history, "violence_rate", "Violencia", "#dc2626"))
                    solara.FigureMatplotlib(make_line_figure(history, "gini_wealth", "Gini riqueza", "#9333ea"))
                with solara.Row():
                    solara.FigureMatplotlib(make_line_figure(history, "legal_formalism", "Formalismo legal", "#f59e0b"))
                    solara.FigureMatplotlib(make_line_figure(history, "life_expectancy", "Esperanza de vida", "#0ea5e9"))


if __name__ == "__main__":
    print("Ejecuta: python -m solara run server.py")
