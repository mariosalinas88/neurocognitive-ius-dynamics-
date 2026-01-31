"""Modelo Neuro Societies con reputación dual y regímenes emergentes (Mesa 3.3+)."""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from mesa import Agent, DataCollector, Model
from mesa.space import MultiGrid


def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def gini(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = arr.mean()
    if mean == 0:
        return 0.0
    diff_sum = np.abs(np.subtract.outer(arr, arr)).sum()
    return float(diff_sum / (2 * arr.size**2 * mean))


def load_profiles(path: str = "profiles.json") -> Dict[str, Dict[str, object]]:
    if not os.path.exists(path):
        return {}
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        result = {}
        for p in data.get("profiles", []):
            pid = str(p.get("id", "")).strip()
            raw_traits = p.get("traits") or p.get("latents") or {}
            traits = {k: clamp01(float(v)) for k, v in raw_traits.items()}
            result[pid] = {
                "traits": traits,
                "name": p.get("name", ""),
                "description": p.get("description", ""),
                "biological_bias": p.get("biological_bias", {}),
                "spectrum_ranges": p.get("spectrum_ranges", {}),
            }
        return result
    except Exception:
        return {}


PROFILE_MAP = load_profiles()
TRACE_LATENTS = os.getenv("TRACE_LATENTS", "0") == "1"
USED_LATENT_KEYS: set[str] = set()


class TraceDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _touch(self, key):
        try:
            USED_LATENT_KEYS.add(str(key))
        except Exception:
            pass

    def get(self, key, default=None):
        self._touch(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self._touch(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self._touch(key)
        return super().__contains__(key)

    def setdefault(self, key, default=None):
        self._touch(key)
        return super().setdefault(key, default)


def load_rare_variants(path: str = "profiles.json") -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        variants = []
        for v in data.get("rare_variants", []):
            vid = str(v.get("id", "")).strip()
            if not vid:
                continue
            variants.append(
                {
                    "id": vid,
                    "name": v.get("name", ""),
                    "probability": clamp01(float(v.get("probability", 0.0))),
                    "trait_mods": {k: float(vv) for k, vv in (v.get("trait_mods", {}) or {}).items()},
                    "imagination_boost": float(v.get("imagination_boost", 0.0) or 0.0),
                    "chaos_innovation": float(v.get("chaos_innovation", 0.0) or 0.0),
                    "narrative": v.get("narrative", "") or "",
                }
            )
        return variants
    except Exception:
        return []


RARE_VARIANTS = load_rare_variants()


def gauss_clip(rng: np.random.Generator, mean: float, std: float = 0.15, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(rng.normal(mean, std), lo, hi))


class Citizen(Agent):
    def __init__(
        self,
        model: "SocietyModel",
        latent: Dict[str, float],
        bias_ranges: Dict[str, List[float]] | None = None,
        spectrum_ranges: Dict[str, List[float]] | None = None,
        spectrum_level: int | None = None,
    ):
        super().__init__(model)
        base_latent = {k: clamp01(v) for k, v in latent.items()}
        self.latent = TraceDict(base_latent) if TRACE_LATENTS else dict(base_latent)
        self.bias_ranges = bias_ranges or {}
        self.spectrum_ranges = spectrum_ranges or {}
        self.spectrum_level = spectrum_level if spectrum_level else int(self.model.rng.integers(1, 4))
        self.last_p_coop = 0.0
        self.last_p_violence = 0.0
        self.last_p_support = 0.0
        self.last_delta = 0.0
        self.rare_variant: Dict[str, object] | None = None
        self.gender = "Female" if self.model.rng.random() < 0.5 else "Male"
        # ciclo vital y reproducción
        self.age = 0
        self.max_age = max(90, int(abs(self.model.rng.normal(260, 45))))
        self.fertile_prob = clamp01(gauss_clip(self.model.rng, 0.35, 0.12) + 0.1 * self.latent.get("empathy", 0.5))
        self.gestation_steps = max(10, int(abs(self.model.rng.normal(24, 6))))
        self.gestation_timer = 0
        self.fertility_cooldown = 0
        self.mate_history: List[int] = []
        self.offspring_count = 0
        self.mating_success = 0.0
        self.dominance_rank = 0.5
        self.status_score = 0.0
        self.current_partner: int | None = None
        # gender biases
        if self.gender == "Female":
            self.latent["language"] = clamp01(self.latent.get("language", 0.5) + gauss_clip(self.model.rng, 0.05, 0.05))
            self.latent["sociality"] = clamp01(self.latent.get("sociality", 0.5) + gauss_clip(self.model.rng, 0.05, 0.05))
            self.latent["risk_aversion"] = clamp01(self.latent.get("risk_aversion", 0.5) - gauss_clip(self.model.rng, 0.05, 0.05))
            self.latent["emotional_impulsivity"] = clamp01(self.latent.get("emotional_impulsivity", 0.5) - gauss_clip(self.model.rng, 0.05, 0.05))
        else:
            self.latent["language"] = clamp01(self.latent.get("language", 0.5) - gauss_clip(self.model.rng, 0.05, 0.05))
            self.latent["sociality"] = clamp01(self.latent.get("sociality", 0.5) - gauss_clip(self.model.rng, 0.05, 0.05))
            self.latent["risk_aversion"] = clamp01(self.latent.get("risk_aversion", 0.5) + gauss_clip(self.model.rng, 0.05, 0.05))
            self.latent["emotional_impulsivity"] = clamp01(self.latent.get("emotional_impulsivity", 0.5) + gauss_clip(self.model.rng, 0.05, 0.05))

        def level_scaled_range(base_range: List[float] | tuple[float, float]) -> tuple[float, float]:
            if not isinstance(base_range, (list, tuple)) or len(base_range) != 2:
                return (0.4, 0.6)
            low, high = float(base_range[0]), float(base_range[1])
            level = max(1, min(3, int(self.spectrum_level)))
            scale = level / 3.0
            adjusted_high = low + (high - low) * scale
            return clamp01(low), clamp01(max(low, adjusted_high))

        def sample_trait(key: str, default_range: tuple[float, float] = (0.4, 0.6)) -> float:
            if key in self.latent:
                return clamp01(self.latent[key])
            base_range = self.bias_ranges.get(key, default_range)
            if isinstance(base_range, (list, tuple)) and len(base_range) == 2:
                low, high = float(base_range[0]), float(base_range[1])
            else:
                low, high = default_range
            if key in self.spectrum_ranges:
                low, high = level_scaled_range(self.spectrum_ranges[key])
            if high < low:
                high = low
            noise = float(self.model.rng.normal(0, 0.1))
            return clamp01(float(self.model.rng.uniform(low, high) + noise))

        moral_emotional_keys = (
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
            "emotional_impulsivity",
            "resilience",
        )
        for key in moral_emotional_keys:
            self.latent[key] = sample_trait(key)
        # baseline benevolence shift
        self.latent["moral_prosocial"] = clamp01(self.latent.get("moral_prosocial", 0.5) + gauss_clip(self.model.rng, 0.05, 0.03))
        self.latent["moral_spite"] = clamp01(self.latent.get("moral_spite", 0.5) - gauss_clip(self.model.rng, 0.02, 0.02))
        if "sexual_impulsivity" not in self.latent:
            base_imp = 0.5 * self.latent.get("impulsivity", 0.5) + 0.5 * self.latent.get("emotional_impulsivity", 0.5)
            self.latent["sexual_impulsivity"] = clamp01(base_imp)

        self.rare_variant = self._maybe_apply_rare_variant()
        if self.rare_variant:
            for k, delta in self.rare_variant.get("trait_mods", {}).items():
                self.latent[k] = clamp01(self.latent.get(k, 0.5) + float(delta))

        self.original_latent = dict(self.latent)
        e = self.latent.get("empathy", 0.5)
        d = self.latent.get("dominance", 0.5)
        imp = self.latent.get("impulsivity", 0.5)
        reg = self.latent.get("affect_reg", 0.5)
        narc = self.latent.get("dark_narc", 0.0)
        mach = self.latent.get("dark_mach", 0.0)
        psy = self.latent.get("dark_psycho", 0.0)
        prosocial = self.latent.get("moral_prosocial", 0.5)
        dark_tri = clamp01(0.35 * narc + 0.35 * mach + 0.30 * psy)
        self.dark_core = clamp01(
            0.5 * dark_tri
            + 0.3 * (1.0 - e)
            + 0.2 * d
            + 0.2 * imp
            - 0.3 * reg
            - 0.2 * prosocial
        )
        reasoning = self.latent.get("reasoning", 0.5)
        self.conscious_core = self._init_conscious_core()
        self.conscious_core["self_model"]["agency"] = clamp01(0.5 * reasoning + 0.3 * self.latent.get("language", 0.5) + 0.2 * self.latent.get("sociality", 0.5))
        self.resource_generation = reasoning * (e if prosocial > 0.5 else d)
        self.nd_contribution = reasoning * e * 0.3
        self.nd_cost = 0.0
        self.violence_cap = clamp01(gauss_clip(self.model.rng, 0.4 + 0.4 * dark_tri, 0.2))
        self.wealth = 1.0
        self.reproduction_cooldown = 0
        self.bonding_timer = 0
        self.mates_lifetime: set[int] = set()
        self.children_ids: List[int] = []
        # neuromoduladores
        self.dopamine = 0.5
        self.oxytocin = 0.5
        self.serotonin = 0.5
        self.endorphin = 0.5
        self.happiness = 0.5
        self.health = 1.0
        self.alive = True
        # reputaciones duales
        self.reputation_coop = 0.5
        self.reputation_fear = 0.2
        self.memory: Dict[int, Dict[str, object]] = {}
        self.last_action: str | None = None
        self.alliance_id: str | None = None

    def reputation_total(self) -> float:
        high = max(self.reputation_coop, self.reputation_fear)
        low = min(self.reputation_coop, self.reputation_fear)
        return clamp01(high + 0.2 * low)

    def get_status_score(self) -> float:
        wealth_term = math.log1p(max(self.wealth, 0.0) + 1.0)
        fear_term = 4.0 * self.reputation_fear
        coop_term = 2.0 * self.reputation_coop
        dom_term = 2.0 * self.latent.get("dominance", 0.5)
        reason_term = self.latent.get("reasoning", 0.5)
        noise = gauss_clip(self.model.rng, 0.0, 0.2, -0.4, 0.4)
        score = wealth_term + fear_term + coop_term + dom_term + 0.8 * reason_term + noise
        return max(0.0, score)

    def _update_status(self):
        self.status_score = self.get_status_score()
        self.dominance_rank = clamp01(0.5 * self.latent.get("dominance", 0.5) + 0.5 * math.tanh(self.status_score / 6.0))

    def _decay_neurochem(self):
        k = clamp01(self.model.neuro_decay_k)
        for attr in ("dopamine", "oxytocin", "serotonin", "endorphin"):
            val = getattr(self, attr, 0.5)
            val = clamp01(val + k * (0.5 - val))
            setattr(self, attr, val)

    def reward(self, event_type: str, intensity: float = 1.0):
        intensity = clamp01(intensity)
        if event_type == "reproduction":
            self.dopamine = clamp01(self.dopamine + 0.2 * intensity)
            self.oxytocin = clamp01(self.oxytocin + 0.3 * intensity)
            self.endorphin = clamp01(self.endorphin + 0.2 * intensity)
        elif event_type == "reproduction_partner":
            self.dopamine = clamp01(self.dopamine + 0.2 * intensity)
            self.oxytocin = clamp01(self.oxytocin + 0.1 * intensity)
        elif event_type == "alliance":
            self.dopamine = clamp01(self.dopamine + 0.05 * intensity)
            self.oxytocin = clamp01(self.oxytocin + 0.05 * intensity)
        elif event_type == "conflict_win":
            delta = 0.08 * intensity
            self.dopamine = clamp01(self.dopamine + delta)
            self.serotonin = clamp01(self.serotonin + delta)
        elif event_type == "conflict_lose":
            delta = 0.06 * intensity
            self.dopamine = clamp01(self.dopamine - delta)
            self.serotonin = clamp01(self.serotonin - delta)

    def is_fertile(self) -> bool:
        if not self.model.enable_reproduction or self.gender != "Female":
            return False
        if self.gestation_timer > 0 or self.fertility_cooldown > 0:
            return False
        roll = self.model.rng.random()
        target = clamp01(gauss_clip(self.model.rng, self.fertile_prob, 0.1))
        return roll < target

    def female_preference_score(self, male: "Citizen") -> float:
        wealth_score = self.model.normalized_wealth(male.wealth)
        dom = male.latent.get("dominance", 0.5)
        health_signal = clamp01(gauss_clip(self.model.rng, male.health, 0.1))
        # edad: pico en adultez
        adult_center = male.max_age * 0.35
        width = max(1.0, male.max_age * 0.25)
        age_factor = math.exp(-((male.age - adult_center) / width) ** 2)

        # pesos base globales
        w_wealth = self.model.mate_weight_wealth
        w_dom = self.model.mate_weight_dom
        w_health = self.model.mate_weight_health
        w_age = self.model.mate_weight_age

        # variaciones individuales
        w_dom *= clamp01(0.8 + 0.4 * self.latent.get("dominance", 0.5))
        w_wealth *= clamp01(0.9 + 0.2 * self.latent.get("risk_aversion", 0.5))
        w_health *= clamp01(0.9 + 0.2 * self.latent.get("empathy", 0.5))

        score = w_wealth * wealth_score + w_dom * dom + w_health * health_signal + w_age * age_factor
        risk_penalty = clamp01(gauss_clip(self.model.rng, male.latent.get("aggression", 0.5), 0.1)) * 0.2
        score -= risk_penalty
        score += gauss_clip(self.model.rng, 0.0, 0.05, -0.1, 0.1)
        return score

    def _start_gestation(self, father: "Citizen"):
        self.gestation_timer = max(8, int(abs(self.model.rng.normal(self.gestation_steps, 4))))
        self.fertility_cooldown = max(1, int(abs(self.model.rng.normal(self.model.female_repro_cooldown, 3)))))
        self.current_partner = father.unique_id
        self.mate_history.append(father.unique_id)
        father.mating_success += 1.0
        father.reproduction_cooldown = max(1, int(abs(self.model.rng.normal(self.model.male_repro_cooldown, 2)))))
        if self.model.rng.random() < clamp01(self.dopamine):
            father.reproduction_cooldown = max(1, father.reproduction_cooldown - 1)
        cooldown_f = max(1, int(abs(self.model.rng.normal(self.model.female_repro_cooldown, 3)))))
        self.reproduction_cooldown = max(self.reproduction_cooldown, cooldown_f)
        self.model.step_events["mating_attempts"] += 1

    def _apply_reproduction_costs(self):
        drain = gauss_clip(self.model.rng, self.model.reproduction_costs, 0.05, 0.0, 1.0)
        self.wealth -= drain
        self.health = clamp01(self.health - 0.05 * drain)
        self.happiness = clamp01(self.happiness - 0.02 * drain)

    def _give_birth(self):
        father = None
        for a in self.model.agents_alive():
            if a.unique_id == self.current_partner:
                father = a
                break
        if father is None:
            return
        child_traits: Dict[str, float] = {}
        for k in self.latent.keys():
            avg = 0.5 * (self.latent.get(k, 0.5) + father.latent.get(k, 0.5))
            child_traits[k] = clamp01(avg + self.model.rng.normal(0, 0.05))
        child = Citizen(self.model, child_traits, bias_ranges={}, spectrum_ranges={}, spectrum_level=None)
        child.profile_id = getattr(self, "profile_id", "")
        self.model.agents.add(child)
        self.model.grid.place_agent(child, self.pos)
        self.offspring_count += 1
        father.offspring_count += 1
        self.children_ids.append(child.unique_id)
        father.children_ids.append(child.unique_id)
        self.mating_success += 1.0
        self.mates_lifetime.add(father.unique_id)
        father.mates_lifetime.add(self.unique_id)
        self.model.births_total += 1
        self.model.step_events["births"] += 1
        self.reward("reproduction", intensity=1.0)
        father.reward("reproduction_partner", intensity=1.0)
        self._apply_reproduction_costs()
        self.gestation_timer = 0
        self.current_partner = None
        self.bonding_timer = self.model.bonding_steps

    def _bonding_tick(self):
        if self.bonding_timer <= 0 or self.current_partner is None:
            return
        partner = None
        for a in self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=1):
            if isinstance(a, Citizen) and a.unique_id == self.current_partner:
                partner = a
                break
        if partner:
            self.oxytocin = clamp01(self.oxytocin + self.model.bonding_delta)
            partner.oxytocin = clamp01(partner.oxytocin + 0.5 * self.model.bonding_delta)
        self.bonding_timer -= 1

    def male_initiation(self):
        if not self.model.enable_reproduction or self.gender != "Male" or self.reproduction_cooldown > 0:
            return
        p = clamp01(self.model.male_initiation_base + self.model.male_desire_scale * self.latent.get("sexual_impulsivity", 0.5))
        p = clamp01(gauss_clip(self.model.rng, p, 0.05))
        if self.model.rng.random() > p:
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=2)
        females = [
            f
            for f in neighbors
            if isinstance(f, Citizen) and f.gender == "Female" and f.alive and f.gestation_timer == 0 and f.reproduction_cooldown == 0
        ]
        if not females:
            return
        target = self.model.rng.choice(females)
        score = target.female_preference_score(self)
        beta = self.model.mate_choice_beta
        exp_accept = math.exp(beta * score)
        exp_reject = math.exp(0.0)
        accept_p = clamp01(exp_accept / (exp_accept + exp_reject))
        success_base = self.model.last_metrics.get("coop_rate", 0.0) - self.model.last_metrics.get("violence_rate", 0.0) + self.model.repro_base_offset
        desire = self.model.repro_desire_scale * target.latent.get("sexual_impulsivity", 0.5)
        success_p = clamp01(success_base + desire)
        final_p = clamp01(0.5 * accept_p + 0.5 * success_p)
        if self.model.rng.random() < final_p:
            target._start_gestation(self)
            target._apply_reproduction_costs()

    def attempt_mating(self):
        if not self.is_fertile():
            return
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=2)
        candidates = [
            m
            for m in neighbors
            if isinstance(m, Citizen)
            and m.gender == "Male"
            and m.alive
            and m.age > 18
            and m.reproduction_cooldown == 0
        ]
        if not candidates:
            return
        scored = []
        for m in candidates:
            score = self.female_preference_score(m)
            resource_gate = clamp01(1.0 - self.model.resource_constraint * max(0.0, 0.6 - self.wealth))
            score *= resource_gate
            scored.append((score, m))
        if not scored:
            return
        beta = self.model.mate_choice_beta
        max_score = max(s for s, _ in scored)
        exps = [math.exp(beta * (s - max_score)) for s, _ in scored]
        total = sum(exps) or 1e-6
        pick_r = self.model.rng.random() * total
        accum = 0.0
        chosen = scored[0][1]
        for (expv, (_, cand)) in zip(exps, scored):
            accum += expv
            if pick_r <= accum:
                chosen = cand
                break
        success_base = self.model.last_metrics.get("coop_rate", 0.0) - self.model.last_metrics.get("violence_rate", 0.0) + self.model.repro_base_offset
        desire = self.model.repro_desire_scale * self.latent.get("sexual_impulsivity", 0.5)
        success_p = clamp01(success_base + desire)
        if self.model.rng.random() < success_p:
            self._start_gestation(chosen)
            self._apply_reproduction_costs()
        else:
            self.model.step_events["female_indirect_competition"] += 1

    def _mem_entry(self, other: "Citizen") -> Dict[str, object]:
        return self.memory.setdefault(other.unique_id, {"trust": 0.5, "last_outcome": None, "interactions": 0})

    def predicted_coop(self, other: "Citizen") -> float:
        entry = self._mem_entry(other)
        trust = float(entry["trust"])
        empathy = self.latent.get("empathy", 0.5)
        # percepción social del otro
        coop_other = other.reputation_coop
        fear_other = other.reputation_fear
        status_boost = clamp01(other.get_perceived_status() / 15.0)
        perceived = clamp01(0.6 * empathy + 0.25 * coop_other + 0.1 * (1 - fear_other) + 0.05 * status_boost)
        base = 0.5 * trust + 0.5 * perceived
        return clamp01(base)

    def decide_action(self, other: "Citizen") -> str:
        entry = self._mem_entry(other)
        trust = float(entry["trust"])
        empathy = self.latent.get("empathy", 0.5)
        dominance = self.latent.get("dominance", 0.5)
        impulsivity = self.latent.get("impulsivity", 0.5)
        risk_aversion = self.latent.get("risk_aversion", 0.5)
        reasoning = self.latent.get("reasoning", 0.5)
        aggression = self.latent.get("aggression", 0.5)
        emo_imp = self.latent.get("emotional_impulsivity", 0.5)
        moral_prosocial = self.latent.get("moral_prosocial", 0.5)
        moral_common_good = self.latent.get("moral_common_good", 0.5)
        moral_honesty = self.latent.get("moral_honesty", 0.5)
        moral_spite = self.latent.get("moral_spite", 0.5)
        resilience = self.latent.get("resilience", 0.5)
        attn_flex = self.latent.get("attn_flex", 0.5)
        dark = self.dark_core

        predicted_coop = self.predicted_coop(other)
        reputation_other_coop = other.reputation_coop
        reputation_other_fear = other.reputation_fear
        # perception bias modulation
        threat_sens = self.latent.get("perception_threat", self.latent.get("perception_bias", {}).get("threat_sensitivity", 0.5))
        detail_orient = self.latent.get("perception_detail", self.latent.get("perception_bias", {}).get("detail_orientation", 0.5))
        social_cue = self.latent.get("perception_social", self.latent.get("perception_bias", {}).get("social_cue_weight", 0.5))
        perceived_threat = threat_sens * (1.0 - reputation_other_coop) + (1 - social_cue) * reputation_other_fear + detail_orient * (1 - predicted_coop)
        predicted_coop = clamp01(predicted_coop - perceived_threat * 0.2)
        self._update_conscious_perception(other, perceived_threat)

        base_coop = empathy + trust + gauss_clip(self.model.rng, 0.3, 0.15) * reputation_other_coop - risk_aversion
        base_coop += gauss_clip(self.model.rng, 0.4, 0.15) * moral_prosocial + gauss_clip(self.model.rng, 0.2, 0.15) * moral_common_good + gauss_clip(self.model.rng, 0.1, 0.15) * moral_honesty - gauss_clip(self.model.rng, 0.1, 0.15) * moral_spite
        base_coop *= (1.0 - gauss_clip(self.model.rng, 0.7, 0.15) * dark)
        base_coop += gauss_clip(self.model.rng, 0.2 * resilience * empathy, 0.1)
        base_coop += 0.05 * attn_flex
        base_coop += 0.15 * reasoning * max(0.0, 1.0 - self.model.last_metrics.get("violence_rate", 0.0))
        if self.alliance_id and self.alliance_id == getattr(other, "alliance_id", None):
            base_coop *= 1.15
            base_support_bias = 0.08
        else:
            base_support_bias = 0.0
        base_coop *= gauss_clip(self.model.rng, 1.7, 0.2)
        base_coop += self._imagine_outcome(other)
        prob_coop = clamp01(gauss_clip(self.model.rng, base_coop, 0.15))

        base_support = gauss_clip(self.model.rng, 0.3, 0.1) * (empathy * 0.7 + reasoning * 0.3)
        if moral_prosocial > 0.6:
            base_support += gauss_clip(self.model.rng, 0.2, 0.1)
        base_support += base_support_bias
        base_support += 0.05 * attn_flex
        base_support *= 1.0 + 0.1 * self.oxytocin
        prob_support = clamp01(gauss_clip(self.model.rng, base_support, 0.15))

        base_violence = dominance * (1.0 - empathy) * (impulsivity + 0.1) * (1.0 - predicted_coop)
        base_violence *= (0.3 + 0.7 * dark)
        strategic = reasoning
        base_violence *= (0.5 + 0.5 * strategic * (1.0 - reputation_other_coop))
        base_violence *= 1.0 + gauss_clip(self.model.rng, 0.5, 0.15) * aggression + gauss_clip(self.model.rng, 0.4, 0.15) * emo_imp
        moral_brake = clamp01(gauss_clip(self.model.rng, 0.4, 0.15) * moral_prosocial + gauss_clip(self.model.rng, 0.2, 0.15) * moral_common_good + gauss_clip(self.model.rng, 0.15, 0.1) * moral_honesty)
        base_violence *= max(0.0, 1.0 - gauss_clip(self.model.rng, 0.6, 0.15) * moral_brake)
        low_v = gauss_clip(self.model.rng, 0.2, 0.05)
        high_v = gauss_clip(self.model.rng, 0.5, 0.1)
        if high_v < low_v:
            high_v = low_v
        base_violence += self.model.rng.uniform(low_v, high_v) * emo_imp
        base_violence *= max(gauss_clip(self.model.rng, 0.5, 0.1), 1.0 - gauss_clip(self.model.rng, 0.4, 0.15) * resilience)
        base_violence += gauss_clip(self.model.rng, 0.2 * dark * reasoning, 0.1)
        if self.alliance_id and self.alliance_id != getattr(other, "alliance_id", None):
            base_violence *= 0.9
        if self._calculate_malice(other):
            base_violence += 0.5
        base_violence *= max(0.5, 1.0 - 0.2 * attn_flex)
        base_violence *= max(0.1, 1.0 - 0.4 * reasoning)
        base_violence *= max(0.5, 1.0 - 0.4 * self.oxytocin)
        base_violence *= 1.0 + 0.2 * max(0.0, 0.6 - self.serotonin)
        prob_violence = clamp01(gauss_clip(self.model.rng, base_violence, 0.15))
        self.last_p_coop = prob_coop
        self.last_p_violence = prob_violence
        self.last_p_support = prob_support

        prob_defect = max(0.0, 1.0 - min(1.0, prob_coop + prob_violence + prob_support))

        total = prob_coop + prob_defect + prob_violence + prob_support
        if total <= 0:
            return "defect"
        r = self.model.rng.random() * total
        if r < prob_coop:
            return "coop"
        if r < prob_coop + prob_defect:
            return "defect"
        if r < prob_coop + prob_defect + prob_violence:
            return "violence"
        return "support"

    def neuroplasticity(self, context: str):
        affect_reg = self.latent.get("affect_reg", 0.5)
        dark = self.dark_core
        emo_imp = self.latent.get("emotional_impulsivity", 0.5)
        resilience = self.latent.get("resilience", 0.5)

        def rand_delta(base_min: float = 0.2, base_max: float = 1.0) -> float:
            bias_scale = 0.5 + emo_imp
            lo = base_min * bias_scale
            hi = base_max * bias_scale
            mid = (lo + hi) / 2
            return float(np.clip(self.model.rng.normal(mid, 0.3), lo, hi))

        def should_change() -> bool:
            base_p = 0.2 + 0.6 * (1.0 - resilience)
            return self.model.rng.random() < clamp01(base_p)

        def apply_delta(key: str, sign: float, magnitude: float):
            self.latent[key] = clamp01(self.latent.get(key, 0.5) + sign * magnitude)

        traumatic_loss = self.wealth < -0.2
        trauma_boost = 2.0 if traumatic_loss else 1.0

        if context == "coop_success":
            if should_change():
                delta = rand_delta(0.2, 1.0) * trauma_boost
                apply_delta("empathy", +1, delta)
                apply_delta("moral_prosocial", +1, delta)
                apply_delta("moral_common_good", +1, delta * 0.8)
                apply_delta("aggression", -1, delta * 0.5)
                apply_delta("moral_spite", -1, delta * 0.5)
                apply_delta("dark_mach", -1, delta * 0.4)
                apply_delta("dark_psycho", -1, delta * 0.4)
                apply_delta("affect_reg", +1, delta * 0.5)
            if self.model.rng.random() < 0.3:
                apply_delta("trust", +1, rand_delta(0.1, 0.4))

        elif context == "betrayed":
            if should_change():
                delta = rand_delta(0.2, 1.0) * trauma_boost
                apply_delta("empathy", -1, delta)
                apply_delta("moral_prosocial", -1, delta)
                apply_delta("moral_honesty", -1, delta * 0.8)
                apply_delta("aggression", +1, delta)
                apply_delta("moral_spite", +1, delta)
                apply_delta("dark_mach", +1, delta * 0.6)
                apply_delta("emotional_impulsivity", +1, delta * 0.5)
                apply_delta("risk_aversion", +1, delta * 0.5)
            if dark > 0.6:
                apply_delta("dominance", +1, rand_delta(0.1, 0.5))

        elif context == "violence_success":
            if should_change():
                delta = rand_delta(0.2, 1.0) * trauma_boost
                apply_delta("aggression", +1, delta)
                apply_delta("moral_spite", +1, delta * 0.8)
                apply_delta("dark_psycho", +1, delta * 0.7)
                apply_delta("dark_narc", +1, delta * 0.5)
                apply_delta("moral_prosocial", -1, delta * 0.8)
                apply_delta("empathy", -1, delta * 0.5)
            if self.latent.get("empathy", 0.5) > 0.6 and self.model.rng.random() < 0.5:
                apply_delta("moral_spite", -1, rand_delta(0.1, 0.4))
            self.reputation_fear = clamp01(self.reputation_fear + 0.05 - 0.08 * self.model.institution_pressure)

        # fluid reversion toward original if new rentable outcome arises
        if self.model.rng.random() < self.model.rng.uniform(0.1, 0.3):
            for key, orig in self.original_latent.items():
                self.latent[key] = clamp01(self.latent[key] + 0.1 * (orig - self.latent[key]))

        noise_sigma = 0.005 * (1.0 + (1.0 - affect_reg))
        for key in (
            "empathy",
            "dominance",
            "reasoning",
            "risk_aversion",
            "affect_reg",
            "impulsivity",
            "language",
            "aggression",
            "emotional_impulsivity",
            "resilience",
        ):
            self.latent[key] = clamp01(self.latent.get(key, 0.5) + self.model.rng.normal(0, noise_sigma))
        self.last_delta = delta if 'delta' in locals() else 0.0
        reasoning = self.latent.get("reasoning", 0.5)
        if self.conscious_core["self_model"]["metacognition"] > 0.6 and reasoning > 0.6 and context == "violence_success" and self.wealth < 0.5:
            self.latent["aggression"] = clamp01(self.latent.get("aggression", 0.5) - 0.02)

        e = self.latent.get("empathy", 0.5)
        d = self.latent.get("dominance", 0.5)
        imp = self.latent.get("impulsivity", 0.5)
        reg = self.latent.get("affect_reg", 0.5)
        narc = self.latent.get("dark_narc", 0.0)
        mach = self.latent.get("dark_mach", 0.0)
        psy = self.latent.get("dark_psycho", 0.0)
        prosocial = self.latent.get("moral_prosocial", 0.5)
        dark_tri = clamp01(0.35 * narc + 0.35 * mach + 0.30 * psy)
        self.dark_core = clamp01(
            0.5 * dark_tri
            + 0.3 * (1.0 - e)
            + 0.2 * d
            + 0.2 * imp
            - 0.3 * reg
            - 0.2 * prosocial
        )

    def update_memory(self, other: "Citizen", outcome: str, trust_delta: float):
        entry = self._mem_entry(other)
        entry["interactions"] += 1
        entry["last_outcome"] = outcome
        entry["trust"] = clamp01(entry["trust"] + trust_delta)

    def interact(self):
        if not self.alive:
            return
        raw_neighbors = [
            a for a in self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=1)
            if isinstance(a, Citizen) and a.alive
        ]
        if not raw_neighbors:
            return
        reasoning = self.latent.get("reasoning", 0.5)
        filtered = []
        for n in raw_neighbors:
            risk = 0.6 * n.reputation_fear + 0.4 * (1.0 - n.reputation_coop)
            dominance = self.latent.get("dominance", 0.5)
            impulsivity = self.latent.get("impulsivity", 0.5)
            risk_aversion = self.latent.get("risk_aversion", 0.5)
            bravery = clamp01(0.5 * dominance + 0.3 * impulsivity + 0.2 * (1 - risk_aversion))
            avoid_p = clamp01(reasoning * risk * (1 - bravery))
            seek_p = clamp01(bravery * (0.2 + 0.5 * (1 - risk_aversion)))
            if reasoning > 0.7 and self.model.rng.random() < gauss_clip(self.model.rng, 0.4 + 0.4 * max(0, reasoning - 0.5), 0.15) and risk > 0.5:
                continue  # predicted negative interaction, skip
            if self.model.rng.random() < avoid_p and self.model.rng.random() > seek_p:
                continue
            filtered.append(n)
        neighbors = filtered or raw_neighbors
        interactions = min(len(neighbors), 2)
        for other in self.model.rng.choice(neighbors, interactions, replace=False):
            my_action = self.decide_action(other)
            other_action = other.decide_action(self)
            self.last_action = my_action
            other.last_action = other_action

            if my_action == "violence" or other_action == "violence":
                attacker, victim = (self, other) if my_action == "violence" else (other, self)
                special = attacker._attack(victim)
                if special == "SUCCESS":
                    self._log_episode(attacker, victim, "violence")
                    drag = 0.05 * (1.0 - self.model.last_metrics.get("coop_rate", 0.0))
                    attacker.wealth -= drag
                    victim.wealth -= drag
                    self.model.total_wealth -= drag * 2
                    self.model.register_violence(attacker, victim)
                    continue
                if special == "FAILED":
                    self._log_episode(attacker, victim, "violence_fail")
                    drag = 0.03
                    attacker.wealth -= drag
                    self.model.total_wealth -= drag
                    self.model.register_violence(attacker, victim)
                    continue
                gain = gauss_clip(self.model.rng, 0.8, 0.15) * (attacker.latent.get("dominance", 0.5) + gauss_clip(self.model.rng, 0.2, 0.05))
                attacker.wealth += gain
                victim.wealth -= gain * 1.2
                cost = self.model.legal_formalism * gauss_clip(self.model.rng, 0.3, 0.1)
                attacker.wealth -= cost
                self.model.total_wealth -= cost
                attacker.happiness = clamp01(attacker.happiness + 0.05)
                victim.happiness = clamp01(victim.happiness - 0.2)
                mortality = 0.03 * self.model.mortality_multiplier * max(0.5, 1.0 - 0.3 * victim.endorphin)
                victim.alive = victim.wealth > -2.0 and victim.model.rng.random() > mortality
                attacker.neuroplasticity("violence_success")
                victim.neuroplasticity("betrayed")
                attacker.reward("conflict_win", intensity=1.0)
                victim.reward("conflict_lose", intensity=1.0)
                attacker.update_memory(victim, "violence", trust_delta=-0.12)
                victim.update_memory(attacker, "violence", trust_delta=-0.12)
                attacker.reputation_fear = clamp01(attacker.reputation_fear + 0.12)
                attacker.reputation_coop = clamp01(attacker.reputation_coop - 0.15)
                victim.nd_cost += 0.1
                self.model.total_wealth -= 0.1
                self._log_episode(attacker, victim, "violence")
                continue

            if my_action == "coop" and other_action == "coop":
                bonus_raw = gauss_clip(self.model.rng, 0.6, 0.15) * (self.latent.get("empathy", 0.5) + other.latent.get("empathy", 0.5))
                efficiency = self._calculate_trade_efficiency(other)
                bonus = bonus_raw * efficiency
                self.wealth += bonus
                other.wealth += bonus
                self.happiness = clamp01(self.happiness + 0.1)
                other.happiness = clamp01(other.happiness + 0.1)
                # development: high reasoning + empathetic resource generation boosts total wealth
                for actor in (self, other):
                    actor.resource_generation = actor.latent.get("reasoning", 0.5) * (actor.latent.get("empathy", 0.5) if actor.latent.get("moral_prosocial", 0.5) > 0.5 else actor.latent.get("dominance", 0.5))
                    if actor.latent.get("moral_prosocial", 0.5) > 0.5 and actor.resource_generation > 0.4:
                        low_dev = gauss_clip(self.model.rng, 0.2, 0.05)
                        high_dev = gauss_clip(self.model.rng, 0.5, 0.1)
                        if high_dev < low_dev:
                            high_dev = low_dev
                        dev = float(self.model.rng.uniform(low_dev, high_dev) * actor.resource_generation)
                        self.model.total_wealth += dev
                        actor.wealth += 0.1 * dev
                        other.wealth += 0.1 * dev
                        actor.nd_contribution += dev
                self.neuroplasticity("coop_success")
                other.neuroplasticity("coop_success")
                self.update_memory(other, "coop", trust_delta=0.05)
                other.update_memory(self, "coop", trust_delta=0.05)
                self.reputation_coop = clamp01(self.reputation_coop + 0.06)
                other.reputation_coop = clamp01(other.reputation_coop + 0.06)
                self._log_episode(self, other, "coop")
            elif my_action == "support" or other_action == "support":
                supporter = self if my_action == "support" else other
                receiver = other if supporter is self else self
                low_s = gauss_clip(self.model.rng, 0.2, 0.05)
                high_s = gauss_clip(self.model.rng, 0.6, 0.1)
                if high_s < low_s:
                    high_s = low_s
                dev = float(self.model.rng.uniform(low_s, high_s) * supporter.latent.get("reasoning", 0.5))
                self.model.total_wealth += dev
                supporter.wealth += 0.05 * dev
                receiver.wealth += 0.1 * dev
                supporter.reputation_coop = clamp01(supporter.reputation_coop + 0.05)
                supporter.latent["empathy"] = clamp01(supporter.latent.get("empathy", 0.5) + 0.02)
                supporter.nd_contribution += dev
                self._log_episode(supporter, receiver, "support")
            elif my_action == "coop" and other_action == "defect":
                steal = 0.9
                other.wealth += steal
                self.wealth -= steal
                other.happiness = clamp01(other.happiness + 0.05)
                self.happiness = clamp01(self.happiness - 0.1)
                self.neuroplasticity("betrayed")
                other.neuroplasticity("violence_success")
                self.update_memory(other, "defect", trust_delta=-0.12)
                other.update_memory(self, "defect", trust_delta=0.0)
                other.reputation_fear = clamp01(other.reputation_fear + 0.08)
                other.reputation_coop = clamp01(other.reputation_coop - 0.10)
                self._log_episode(other, self, "defect")
            elif my_action == "defect" and other_action == "coop":
                steal = 0.9
                self.wealth += steal
                other.wealth -= steal
                self.happiness = clamp01(self.happiness + 0.05)
                other.happiness = clamp01(other.happiness - 0.1)
                self.neuroplasticity("violence_success")
                other.neuroplasticity("betrayed")
                self.update_memory(other, "defect", trust_delta=0.0)
                other.update_memory(self, "defect", trust_delta=-0.12)
                self.reputation_fear = clamp01(self.reputation_fear + 0.08)
                self.reputation_coop = clamp01(self.reputation_coop - 0.10)
                self._log_episode(self, other, "defect")
            else:
                self.happiness = clamp01(self.happiness - 0.02)
                other.happiness = clamp01(other.happiness - 0.02)
                self.update_memory(other, "defect", trust_delta=-0.02)
                other.update_memory(self, "defect", trust_delta=-0.02)
                self._log_episode(self, other, "standoff")

            # psicópata estratégico: simulación de empatía para reputación cooperativa
            if self.latent.get("reasoning", 0.5) > 0.8 and self.latent.get("empathy", 0.5) < 0.3:
                self.reputation_coop = clamp01(self.reputation_coop + 0.02)
            if other.latent.get("reasoning", 0.5) > 0.8 and other.latent.get("empathy", 0.5) < 0.3:
                other.reputation_coop = clamp01(other.reputation_coop + 0.02)
        if raw_neighbors:
            self._perform_community_service(raw_neighbors)

    def move(self):
        if not self.alive:
            return
        impulsivity = self.latent.get("impulsivity", 0.5)
        attn_flex = self.latent.get("attn_flex", 0.5)
        desire = 0.2 + impulsivity * 0.5 + max(0.0, 0.4 - self.happiness) + 0.1 * attn_flex
        desire -= 0.3 * min(1.0, max(0.0, self.wealth))
        if self.model.rng.random() > desire:
            return
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        empties = [pos for pos in neighbors if self.model.grid.is_cell_empty(pos)]
        if empties:
            idx = int(self.model.rng.integers(len(empties)))
            self.model.grid.move_agent(self, tuple(empties[idx]))

    def production_and_consumption(self):
        reasoning = self.latent.get("reasoning", 0.5)
        hyperfocus = self.latent.get("hyperfocus", 0.5)
        prod = 1.0 + 0.5 * reasoning + 0.3 * hyperfocus
        prod *= self.model.production_multiplier
        self.wealth += prod * self.model.scale_factor
        consumption = 0.8 + 0.2 * (1 - reasoning)
        self.wealth -= consumption
        if self.wealth < -1.5:
            self.alive = False

    def epidemic_effect(self):
        if self.model.external_factor != "epidemic":
            return
        affect_reg = self.latent.get("affect_reg", 0.5)
        risk = 0.02 * (1 - affect_reg)
        if self.model.rng.random() < risk:
            self.alive = False

    def step(self):
        if not self.alive:
            return
        self._decay_neurochem()
        self.age += 1
        self.production_and_consumption()
        # reproduction cooldown
        if self.reproduction_cooldown > 0:
            self.reproduction_cooldown -= 1
        if self.gestation_timer > 0 and self.gender == "Female":
            self.gestation_timer -= 1
            if self.gestation_timer == 0:
                self._give_birth()
        if self.gender == "Female":
            if self.fertility_cooldown > 0:
                self.fertility_cooldown -= 1
            self.attempt_mating()
        else:
            self.male_initiation()
            self._male_competition()
        if self.bonding_timer > 0:
            self._bonding_tick()
        if not self.alive:
            return
        self.epidemic_effect()
        if not self.alive:
            return
        if self.model.rng.random() < 0.03 * self.model.mortality_multiplier or self.age > 320:
            self.alive = False
            return
        self.move()
        self.interact()
        if self.model.innovation_boost > 0:
            self.latent["reasoning"] = clamp01(self.latent.get("reasoning", 0.5) + self.model.innovation_boost)
        self._reflective_cycle()
        if self.health <= 0:
            self.alive = False

    def _maybe_apply_rare_variant(self) -> Dict[str, object] | None:
        if not RARE_VARIANTS:
            return None
        pick = None
        for variant in RARE_VARIANTS:
            if self.model.rng.random() < variant.get("probability", 0.0):
                pick = variant
                break
        return pick

    def _init_conscious_core(self) -> Dict[str, object]:
        awareness = clamp01(0.4 + 0.2 * self.latent.get("language", 0.5) + 0.2 * self.latent.get("reasoning", 0.5) + 0.1 * self.latent.get("sociality", 0.5))
        if self.rare_variant:
            awareness = clamp01(awareness + 0.05)
        goals = []
        if self.latent.get("reasoning", 0.5) > 0.65:
            goals.append("orden_formal")
        if self.latent.get("empathy", 0.5) > 0.6:
            goals.append("cuidado_comunidad")
        if self.latent.get("impulsivity", 0.5) > 0.6:
            goals.append("innovacion_caotica")
        if self.latent.get("hyperfocus", 0.5) > 0.6:
            goals.append("precision_leibniziana")
        narrative = f"{getattr(self, 'profile_id', 'anon')}|{self.rare_variant['name'] if self.rare_variant else 'base'}"
        return {
            "awareness": awareness,
            "perception_map": {"bias": {}, "last": None},
            "self_model": {"agency": awareness, "goals": goals, "metacognition": clamp01(self.latent.get("reasoning", 0.5))},
            "memory_episodic": [],
            "imagination_buffer": [],
            "identity_narrative": narrative,
        }

    def _update_conscious_perception(self, other: "Citizen", perceived_threat: float):
        bias = {
            "threat": perceived_threat,
            "detail": self.latent.get("attn_selective", 0.5),
            "social": self.latent.get("sociality", 0.5),
        }
        collapse_p = clamp01(0.02 + 0.05 * self.latent.get("hyperfocus", 0.5) - 0.02 * self.latent.get("resilience", 0.5))
        if self.rare_variant:
            collapse_p += 0.03
        if self.model.rng.random() < collapse_p:
            bias["collapse"] = clamp01(self.model.rng.random())
        self.conscious_core["perception_map"] = {"bias": bias, "last": getattr(other, "unique_id", None)}

    def _interpretation_style(self) -> str:
        if self.latent.get("reasoning", 0.5) >= self.latent.get("emotional_impulsivity", 0.5):
            return "logico"
        return "afectivo"

    def _log_episode(self, actor: "Citizen", target: "Citizen", label: str):
        memory = self.conscious_core["memory_episodic"]
        if len(memory) > 60:
            del memory[:10]
        interp = "rigido" if self.latent.get("reasoning", 0.5) > 0.65 else ("visionario" if self.rare_variant else "contextual")
        memory.append(
            {
                "actor": actor.unique_id,
                "target": target.unique_id,
                "event": label,
                "interpretation": f"{self._interpretation_style()}_{interp}",
            }
        )

    def _imagine_outcome(self, other: "Citizen") -> float:
        buf = self.conscious_core["imagination_buffer"]
        if len(buf) > 30:
            del buf[:10]
        vision = self.model.rng.normal(0, 0.05)
        if self.rare_variant and self.rare_variant.get("imagination_boost", 0.0) > 0 and self.model.rng.random() < 0.1:
            vision += 0.15
            buf.append({"type": "vision", "target": other.unique_id, "delta": vision})
        else:
            buf.append({"type": "project", "target": other.unique_id, "delta": vision})
        return vision

    def _reflective_cycle(self):
        awareness = self.conscious_core["awareness"]
        awareness = clamp01(awareness + 0.05 * (self.latent.get("language", 0.5) + self.latent.get("affect_reg", 0.5) - 0.5))
        if self.rare_variant and self.rare_variant.get("chaos_innovation", 0.0) > 0:
            awareness = clamp01(awareness + 0.03)
        self.conscious_core["awareness"] = awareness
        if self.latent.get("reasoning", 0.5) > 0.7 and self.last_action == "violence":
            self.latent["impulsivity"] = clamp01(self.latent.get("impulsivity", 0.5) - 0.01)
        if self.latent.get("empathy", 0.5) > 0.7 and self.last_action == "violence":
            self.latent["affect_reg"] = clamp01(self.latent.get("affect_reg", 0.5) + 0.01)
        if self.last_action == "violence":
            self.health = clamp01(self.health - 0.02)
        if self.latent.get("hyperfocus", 0.5) > 0.8 and self.latent.get("reasoning", 0.5) > 0.8:
            if self.model.rng.random() < 0.01 * self.latent.get("reasoning", 0.5):
                invention_value = self.latent.get("reasoning", 0.5) * 5.0
                self.model.total_wealth += invention_value
                self.reputation_coop = clamp01(self.reputation_coop + 0.1 * self.latent.get("language", 0.5))

    def _calculate_malice(self, other: "Citizen") -> bool:
        spite = self.latent.get("moral_spite", 0.0)
        envy_trigger = 1.0 if other.wealth > self.wealth else 0.0
        prob_sabotage = spite * envy_trigger * self.latent.get("impulsivity", 0.5)
        return self.model.rng.random() < prob_sabotage

    def _perform_community_service(self, neighbors: List["Citizen"]):
        if self.wealth <= 0 or not neighbors:
            return
        altruism_score = clamp01(self.latent.get("moral_common_good", 0.0) * (1.0 - self.latent.get("dark_mach", 0.0)))
        if self.model.rng.random() < altruism_score:
            donation = self.wealth * 0.05 * altruism_score
            poorest = min(neighbors, key=lambda a: a.wealth)
            self.wealth -= donation
            poorest.wealth += donation
            self.happiness = clamp01(self.happiness + 0.02 * altruism_score)

    def get_perceived_status(self) -> float:
        base_status = self.wealth + 10.0 * self.reputation_total()
        inflation = 1.0 + 0.5 * self.latent.get("dark_narc", 0.0)
        return base_status * inflation

    def _calculate_trade_efficiency(self, other: "Citizen") -> float:
        avg_language = (self.latent.get("language", 0.5) + other.latent.get("language", 0.5)) / 2.0
        return 0.5 + 0.5 * avg_language

    def _attack(self, target: "Citizen"):
        wealth_drive = clamp01(self.wealth / 2.0)
        expected_gain = target.wealth * 0.4
        risk_factor = clamp01((1.0 - self.latent.get("risk_aversion", 0.5)) + self.latent.get("dark_psycho", 0.0) * 0.5 + wealth_drive * 0.2)
        spite_attack = self.latent.get("moral_spite", 0.0) > 0.8 and target.wealth > self.wealth
        if self.model.rng.random() < clamp01(risk_factor) or spite_attack:
            attack_success_chance = clamp01(self.latent.get("aggression", 0.5) * 0.7 - target.latent.get("resilience", 0.5) * 0.3)
            if self.model.rng.random() < attack_success_chance:
                stolen_wealth = target.wealth * 0.5
                self.wealth += stolen_wealth
                target.wealth -= stolen_wealth
                self.reputation_fear = clamp01(self.reputation_fear + 0.1)
                self.latent["risk_aversion"] = clamp01(self.latent.get("risk_aversion", 0.5) - 0.05)
                self._log_episode(self, target, "attack_success")
                return "SUCCESS"
            else:
                self.health = clamp01(self.health - 0.1 * self.latent.get("aggression", 0.5))
                self.reputation_fear = clamp01(self.reputation_fear - 0.05)
                self._log_episode(self, target, "attack_fail")
                return "FAILED"
        return "NO_ATTACK"

    def _form_punishment_alliance(self, target: "Citizen"):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False, radius=2)
        potential_allies = [
            n
            for n in neighbors
            if isinstance(n, Citizen)
            and n.alive
            and n is not self
            and self.model.rng.random() < clamp01(n.latent.get("empathy", 0.5) * (1.0 - n.latent.get("dark_psycho", 0.0)))
        ]
        if not potential_allies:
            return
        size = max(1, int(len(potential_allies) * 0.5))
        alliance_members = list(self.model.rng.choice(potential_allies, size, replace=False))
        alliance_members.append(self)
        total_coop_power = sum(a.latent.get("empathy", 0.5) * a.latent.get("language", 0.5) for a in alliance_members)
        target_violence_power = target.latent.get("dominance", 0.5) * target.latent.get("aggression", 0.5) * 1.5
        success_p = total_coop_power / (total_coop_power + target_violence_power + 1e-6)
        if self.model.rng.random() < success_p:
            stolen_wealth = max(0.0, target.wealth * 0.6)
            target.wealth -= stolen_wealth
            self.model.redistribute_wealth_to_allies(stolen_wealth, alliance_members)
            target.reputation_fear = clamp01(target.reputation_fear - 0.2)
            for ally in alliance_members:
                ally.reputation_coop = clamp01(ally.reputation_coop + 0.1)
            self.model.log_event("CASTIGO_EXITOSO", getattr(target, "profile_id", "n/a"))
        else:
            self.model.log_event("CASTIGO_FALLIDO", getattr(target, "profile_id", "n/a"))
            for ally in alliance_members:
                ally.health = clamp01(ally.health - 0.3)
                ally.wealth -= ally.wealth * 0.3
                ally.reputation_coop = clamp01(ally.reputation_coop - 0.1)


class SocietyModel(Model):
    POP_SCALES = {"tiny": 10, "tribe": 1000, "city": 100000, "nation": 10000000}

    def __init__(
        self,
        seed: int | None = None,
        climate: str = "stable",
        external_factor: str = "none",
        population_scale: str = "tribe",
        profile1: str | None = None,
        profile2: str | None = None,
        profile3: str | None = None,
        weight1: float = 0.6,
        weight2: float = 0.3,
        weight3: float = 0.1,
        jitter: float = 0.05,
        spectrum_level: int | None = None,
        initial_moral_bias: str | None = None,
        resilience_bias: str | None = None,
        emotional_bias: str | None = None,
        enable_reproduction: bool = True,
        enable_sexual_selection: bool = True,
        male_violence_multiplier: float = 1.2,
        female_violence_multiplier: float = 0.35,
        female_target_protection: float = 0.6,
        coalition_enabled: bool = True,
        coalition_power_weight: float = 0.6,
        coalition_strategy_weight: float = 0.4,
        sneaky_strategy_enabled: bool = True,
        sneaky_success_weight: float = 0.6,
        reproduction_costs: float = 0.3,
        resource_constraint: float = 0.4,
        mate_weight_wealth: float = 0.4,
        mate_weight_dom: float = 0.3,
        mate_weight_health: float = 0.2,
        mate_weight_age: float = 0.1,
        mate_choice_beta: float = 1.0,
        female_repro_cooldown: int = 10,
        male_repro_cooldown: int = 2,
        repro_base_offset: float = 0.2,
        repro_desire_scale: float = 0.3,
        male_initiation_base: float = 0.05,
        male_desire_scale: float = 0.3,
        neuro_decay_k: float = 0.1,
        bonding_steps: int = 5,
        bonding_delta: float = 0.02,
        enable_coercion: bool = False,
        **kwargs,
    ):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.climate = climate
        self.external_factor = external_factor
        self.spectrum_level = spectrum_level
        self.initial_moral_bias = (initial_moral_bias or "").strip().lower() or None
        self.resilience_bias = (resilience_bias or "").strip().lower() or None
        self.emotional_bias = (emotional_bias or "").strip().lower() or None
        self.total_wealth = 0.0
        self.nd_contribution_log: Dict[str, float] = {}
        self.population_scale_key = population_scale
        self.profile_weights = {
            "profile1": (profile1 or "").strip(),
            "profile2": (profile2 or "").strip(),
            "profile3": (profile3 or "").strip(),
            "weight1": float(weight1),
            "weight2": float(weight2),
            "weight3": float(weight3),
        }
        w_sum = self.profile_weights["weight1"] + self.profile_weights["weight2"] + self.profile_weights["weight3"]
        self.weight_warning = False
        self.weight_sum_original = w_sum
        if w_sum <= 0:
            self.profile_weights["weight1"] = 1.0
            self.profile_weights["weight2"] = 0.0
            self.profile_weights["weight3"] = 0.0
            self.weight_warning = True
        elif abs(w_sum - 1.0) > 1e-6:
            self.profile_weights["weight1"] /= w_sum
            self.profile_weights["weight2"] /= w_sum
            self.profile_weights["weight3"] /= w_sum
            self.weight_warning = True

        target = self.POP_SCALES.get(population_scale, 1000)
        self.max_agents = 5000
        self.actual_agents = min(target, self.max_agents)
        self.scale_factor = target / self.actual_agents if self.actual_agents else 1.0
        self.institution_pressure = min(1.0, math.log10(target) / 7.0)
        self.initial_population_scaled = self.actual_agents * self.scale_factor

        side = max(8, int(math.sqrt(self.actual_agents) * 1.4))
        self.grid = MultiGrid(side, side, torus=True)
        self.alliances: Dict[str, Dict[str, object]] = {}

        self.production_multiplier = self._compute_production_multiplier()
        self.mortality_multiplier = self._compute_mortality_multiplier()
        self.innovation_boost = 0.001 if self.external_factor == "technological" else 0.0
        self.recent_violence_events: List[Tuple[int, int]] = []
        self.enable_reproduction = bool(enable_reproduction)
        self.enable_sexual_selection = bool(enable_sexual_selection)
        self.male_violence_multiplier = float(male_violence_multiplier)
        self.female_violence_multiplier = float(female_violence_multiplier)
        self.female_target_protection = float(female_target_protection)
        self.coalition_enabled = bool(coalition_enabled)
        self.coalition_power_weight = float(coalition_power_weight)
        self.coalition_strategy_weight = float(coalition_strategy_weight)
        self.sneaky_strategy_enabled = bool(sneaky_strategy_enabled)
        self.sneaky_success_weight = float(sneaky_success_weight)
        self.reproduction_costs = float(reproduction_costs)
        self.resource_constraint = clamp01(float(resource_constraint))
        self.mate_weight_wealth = float(mate_weight_wealth)
        self.mate_weight_dom = float(mate_weight_dom)
        self.mate_weight_health = float(mate_weight_health)
        self.mate_weight_age = float(mate_weight_age)
        self.mate_choice_beta = float(mate_choice_beta)
        self.female_repro_cooldown = max(1, int(female_repro_cooldown))
        self.male_repro_cooldown = max(1, int(male_repro_cooldown))
        self.repro_base_offset = float(repro_base_offset)
        self.repro_desire_scale = float(repro_desire_scale)
        self.male_initiation_base = float(male_initiation_base)
        self.male_desire_scale = float(male_desire_scale)
        self.neuro_decay_k = clamp01(float(neuro_decay_k))
        self.bonding_steps = max(0, int(bonding_steps))
        self.bonding_delta = float(bonding_delta)
        self.enable_coercion = bool(enable_coercion)
        self.births_total = 0
        self.step_events: Dict[str, float] = {}

        for _ in range(self.actual_agents):
            latent, bias_ranges, spectrum_ranges, profile_id = self._compose_latent(jitter=jitter)
            agent = Citizen(
                self,
                latent,
                bias_ranges=bias_ranges,
                spectrum_ranges=spectrum_ranges,
                spectrum_level=self.spectrum_level,
            )
            agent.profile_id = profile_id
            agent.conscious_core["identity_narrative"] = f"{profile_id or 'anon'}|{agent.rare_variant['name'] if agent.rare_variant else 'base'}"
            self.agents.add(agent)
            self.grid.place_agent(agent, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))

        self.running = True
        self.total_wealth = sum(a.wealth for a in self.agents if isinstance(a, Citizen))
        self.regime = "Inicial"
        self.legal_formalism = 0.5
        self.liberty_index = 0.5
        self.gini_wealth = 0.0
        self.last_metrics: Dict[str, float] = {}
        self.datacollector = DataCollector(
            model_reporters={
                "population": lambda m: len(m.agents_alive()) * m.scale_factor,
                "coop_rate": lambda m: m.last_metrics.get("coop_rate", 0.0),
                "violence_rate": lambda m: m.last_metrics.get("violence_rate", 0.0),
                "male_violence_rate": lambda m: m.last_metrics.get("male_violence_rate", 0.0),
                "female_violence_rate": lambda m: m.last_metrics.get("female_violence_rate", 0.0),
                "gini_wealth": lambda m: m.gini_wealth,
                "life_expectancy": lambda m: m.last_metrics.get("life_expectancy", 0.0),
                "legal_formalism": lambda m: m.legal_formalism,
                "liberty_index": lambda m: m.liberty_index,
                "regime": lambda m: m.regime,
                "avg_reasoning": lambda m: m.last_metrics.get("avg_reasoning", 0.0),
                "avg_empathy": lambda m: m.last_metrics.get("avg_empathy", 0.0),
                "avg_dominance": lambda m: m.last_metrics.get("avg_dominance", 0.0),
                "reputation_coop_mean": lambda m: float(np.mean([a.reputation_coop for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "reputation_fear_mean": lambda m: float(np.mean([a.reputation_fear for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "top5_wealth_share": lambda m: m._top5_power()[0],
                "total_wealth": lambda m: m.total_wealth,
                "nd_contribution_mean": lambda m: float(np.mean([a.nd_contribution for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "alliances_count": lambda m: len(m.alliances),
                "allied_share": lambda m: (sum(1 for a in m.agents_alive() if getattr(a, "alliance_id", None)) / len(m.agents_alive())) if m.agents_alive() else 0.0,
                "conscious_awareness_mean": lambda m: float(np.mean([a.conscious_core.get("awareness", 0.0) for a in m.agents_alive()])) if m.agents_alive() else 0.0,
                "births": lambda m: m.last_metrics.get("births", 0.0),
                "sex_ratio": lambda m: m.last_metrics.get("sex_ratio", 0.0),
                "coalition_count": lambda m: m.last_metrics.get("coalition_count", 0.0),
                "coalition_wins": lambda m: m.last_metrics.get("coalition_wins", 0.0),
                "sneaky_success_rate": lambda m: m.last_metrics.get("sneaky_success_rate", 0.0),
                "male_male_conflicts": lambda m: m.last_metrics.get("male_male_conflicts", 0.0),
                "female_indirect_competition": lambda m: m.last_metrics.get("female_indirect_competition", 0.0),
                "mating_inequality": lambda m: m.last_metrics.get("mating_inequality", 0.0),
                "mean_harem_size": lambda m: m.last_metrics.get("mean_harem_size", 0.0),
                "repro_gini_males": lambda m: m.last_metrics.get("repro_gini_males", 0.0),
                "male_childless_share": lambda m: m.last_metrics.get("male_childless_share", 0.0),
                "mean_partners_male": lambda m: m.last_metrics.get("mean_partners_male", 0.0),
                "mean_partners_female": lambda m: m.last_metrics.get("mean_partners_female", 0.0),
            }
        )
        self._update_metrics()
        self.datacollector.collect(self)

    def _reset_step_events(self):
        self.step_events = {
            "male_violence": 0.0,
            "female_violence": 0.0,
            "male_male_conflicts": 0.0,
            "female_indirect_competition": 0.0,
            "coalition_count": 0.0,
            "coalition_wins": 0.0,
            "sneaky_attempts": 0.0,
            "sneaky_success": 0.0,
            "mating_attempts": 0.0,
            "births": 0.0,
        }

    def _compute_production_multiplier(self) -> float:
        base = 1.0
        if self.climate == "scarce":
            base *= 0.4
        elif self.climate == "abundant":
            base *= 1.2
        if self.external_factor == "disaster":
            base *= 0.4
        return base

    def _compute_mortality_multiplier(self) -> float:
        mult = 1.0
        if self.climate == "scarce":
            mult *= 3.0
        if self.external_factor in {"disaster"}:
            mult *= 3.0
        return mult

    def _goal_tag(self, agent: Citizen) -> str:
        goals = agent.conscious_core.get("self_model", {}).get("goals") or []
        if goals:
            return goals[0]
        reasoning = agent.latent.get("reasoning", 0.5)
        empathy = agent.latent.get("empathy", 0.5)
        if reasoning > 0.65 and empathy < 0.5:
            return "orden_formal"
        if empathy > 0.65:
            return "cuidado_comunidad"
        if agent.latent.get("impulsivity", 0.5) > 0.6:
            return "innovacion_caotica"
        return "neutro"

    def _latent_vector(self, agent: Citizen, keys: Tuple[str, ...]) -> np.ndarray:
        return np.array([agent.latent.get(k, 0.5) for k in keys], dtype=float)

    def _cosine_similarity(self, a_vec: np.ndarray, b_vec: np.ndarray) -> float:
        denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) or 1e-6
        return float(np.dot(a_vec, b_vec) / denom)

    def _update_alliances(self):
        alive = self.agents_alive()
        for a in alive:
            a.alliance_id = None
        alliances: Dict[str, Dict[str, object]] = {}
        if not alive:
            self.alliances = alliances
            return
        keys = ("reasoning", "empathy", "dominance", "impulsivity")
        aid_counter = 0
        for a in alive:
            affinity = clamp01(0.4 * a.latent.get("empathy", 0.5) + 0.3 * a.latent.get("sociality", 0.5) + 0.3 * a.latent.get("language", 0.5))
            if self.rng.random() > affinity:
                continue
            partners = [n for n in self.grid.get_neighbors(a.pos, moore=True, include_center=True, radius=2) if isinstance(n, Citizen) and n.alive]
            members = [a]
            for n in partners:
                sim = self._cosine_similarity(self._latent_vector(a, keys), self._latent_vector(n, keys))
                join_p = clamp01(0.5 * sim + 0.5 * n.latent.get("empathy", 0.5))
                if self.rng.random() < join_p:
                    members.append(n)
            if len(members) < 2:
                continue
            aid = f"ally_{aid_counter}"
            aid_counter += 1
            alliances[aid] = {
                "goal": self._goal_tag(a),
                "members": members,
                "rule": "prosocial" if np.mean([m.latent.get("empathy", 0.5) for m in members]) >= np.mean([m.latent.get("dominance", 0.5) for m in members]) else "dominance",
            }
            for m in members:
                m.alliance_id = aid
                if alliances[aid]["rule"] == "prosocial":
                    m.reputation_coop = clamp01(m.reputation_coop + 0.01 * affinity)
                else:
                    m.reputation_fear = clamp01(m.reputation_fear + 0.01 * (1.0 - affinity))
                m.reward("alliance", intensity=1.0)
        self.alliances = alliances
        self._resolve_alliance_conflict()

    def _resolve_alliance_conflict(self):
        if len(self.alliances) < 2:
            return
        ids = list(self.alliances.keys())
        for aid in ids:
            ally = self.alliances[aid]
            members: List[Citizen] = ally["members"]  # type: ignore
            if not members:
                continue
            agg = np.mean([m.latent.get("aggression", 0.5) for m in members])
            reason = np.mean([m.latent.get("reasoning", 0.5) for m in members])
            target_id = None
            for other_id in ids:
                if other_id == aid:
                    continue
                target_id = other_id
                break
            if not target_id:
                continue
            target = self.alliances[target_id]
            target_members: List[Citizen] = target["members"]  # type: ignore
            if not target_members:
                continue
            clash_p = clamp01(0.4 * agg + 0.2 * (1.0 - reason))
            if self.rng.random() < clash_p:
                victim = self.rng.choice(target_members)
                victim.wealth -= 0.2
                victim.happiness = clamp01(victim.happiness - 0.05)
                victim.reputation_fear = clamp01(victim.reputation_fear + 0.05)
            else:
                for m in members:
                    boost = m.latent.get("language", 0.5) * m.latent.get("reasoning", 0.5) * 0.02
                    m.reputation_coop = clamp01(m.reputation_coop + boost)
                # Prosocial coalitions reduce violence pressure proportionally
                self.legal_formalism = clamp01(self.legal_formalism + 0.01 * reason)

    def _compose_latent(self, jitter: float = 0.05):
        traits = {
            "attn_selective": 0.5,
            "attn_flex": 0.5,
            "hyperfocus": 0.5,
            "impulsivity": 0.5,
            "risk_aversion": 0.5,
            "sociality": 0.5,
            "language": 0.5,
            "reasoning": 0.5,
            "trust": 0.5,
            "emotional_impulsivity": 0.5,
            "resilience": 0.5,
        }
        bias_min: Dict[str, float] = {}
        bias_max: Dict[str, float] = {}
        bias_w: Dict[str, float] = {}
        spec_min: Dict[str, float] = {}
        spec_max: Dict[str, float] = {}
        spec_w: Dict[str, float] = {}
        ids = [self.profile_weights["profile1"], self.profile_weights["profile2"], self.profile_weights["profile3"]]
        weights = [self.profile_weights["weight1"], self.profile_weights["weight2"], self.profile_weights["weight3"]]
        total_w = sum(w for w in weights if w > 0)
        chosen_id = None
        for pid, w in zip(ids, weights):
            profile_def = PROFILE_MAP.get(pid)
            if not pid or profile_def is None or w <= 0:
                continue
            if chosen_id is None:
                chosen_id = pid
            for k, v in profile_def.get("traits", {}).items():
                traits[k] = traits.get(k, 0.5) + v * w / max(total_w, 1e-6)
            bio_bias = profile_def.get("biological_bias", {}) or {}
            for trait, rng in bio_bias.items():
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    bias_min[trait] = bias_min.get(trait, 0.0) + float(rng[0]) * w
                    bias_max[trait] = bias_max.get(trait, 0.0) + float(rng[1]) * w
                    bias_w[trait] = bias_w.get(trait, 0.0) + w
            spec_ranges = profile_def.get("spectrum_ranges", {}) or {}
            for trait, rng in spec_ranges.items():
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    spec_min[trait] = spec_min.get(trait, 0.0) + float(rng[0]) * w
                    spec_max[trait] = spec_max.get(trait, 0.0) + float(rng[1]) * w
                    spec_w[trait] = spec_w.get(trait, 0.0) + w
        for k in traits:
            traits[k] = clamp01(traits[k] + float(self.rng.normal(0, jitter)))
        default_range = (0.4, 0.6)
        moral_emotional_keys = {
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
            "emotional_impulsivity",
            "resilience",
        }
        combined_bias: Dict[str, List[float]] = {}
        for trait in moral_emotional_keys | set(bias_min.keys()):
            weight = bias_w.get(trait, 0.0)
            if weight > 0:
                low = bias_min.get(trait, 0.0) / weight
                high = bias_max.get(trait, 0.0) / weight
            else:
                low, high = default_range
            combined_bias[trait] = [clamp01(low), clamp01(max(low, high))]

        if self.initial_moral_bias:
            dark_related = {"dark_narc", "dark_mach", "dark_psycho", "aggression", "moral_spite"}
            prosocial_related = {"moral_prosocial", "moral_common_good", "moral_honesty", "empathy"}

            for trait, rng in combined_bias.items():
                low, high = rng
                if self.initial_moral_bias == "high_dark" and trait in dark_related:
                    low = clamp01(low + 0.1)
                    high = clamp01(high + 0.1)
                elif self.initial_moral_bias == "high_prosocial" and trait in prosocial_related:
                    low = clamp01(low + 0.1)
                    high = clamp01(high + 0.1)
                elif self.initial_moral_bias == "low_dark" and trait in dark_related:
                    low = clamp01(low - 0.1)
                    high = clamp01(high - 0.1)
                combined_bias[trait] = [min(low, high), max(low, high)]

        if self.resilience_bias:
            for trait in ["resilience", "affect_reg"]:
                if trait in combined_bias:
                    low, high = combined_bias[trait]
                    shift = 0.1 if self.resilience_bias == "high" else -0.1
                    combined_bias[trait] = [clamp01(low + shift), clamp01(high + shift)]

        if self.emotional_bias:
            for trait in ["emotional_impulsivity", "aggression"]:
                if trait in combined_bias:
                    low, high = combined_bias[trait]
                    shift = 0.1 if self.emotional_bias == "high" else -0.1
                    combined_bias[trait] = [clamp01(low + shift), clamp01(high + shift)]

        combined_spec: Dict[str, List[float]] = {}
        for trait, weight in spec_w.items():
            low = spec_min.get(trait, 0.0) / max(weight, 1e-6)
            high = spec_max.get(trait, 0.0) / max(weight, 1e-6)
            combined_spec[trait] = [clamp01(low), clamp01(max(low, high))]

        return traits, combined_bias, combined_spec, chosen_id or ""

    def agents_alive(self) -> List[Citizen]:
        return [a for a in self.agents if isinstance(a, Citizen) and a.alive]

    def _top5_power(self):
        alive = self.agents_alive()
        if not alive:
            return 0.0, 0.0, 0.0
        sorted_agents = sorted(alive, key=lambda a: a.wealth, reverse=True)
        top5 = sorted_agents[:5]
        total_wealth = sum(a.wealth for a in alive) or 1e-6
        top5_share = sum(a.wealth for a in top5) / total_wealth
        fear_avg = np.mean([a.reputation_fear for a in top5])
        coop_avg = np.mean([a.reputation_coop for a in top5])
        return top5_share, fear_avg, coop_avg

    def normalized_wealth(self, wealth: float) -> float:
        alive = self.agents_alive()
        if not alive:
            return 0.5
        vals = np.array([a.wealth for a in alive], dtype=float)
        mean = vals.mean() if vals.size else 1.0
        std = vals.std() if vals.size else 1.0
        if std < 1e-6:
            return clamp01(wealth / (mean + 1e-6))
        z = (wealth - mean) / std
        return clamp01(0.5 + 0.2 * z)

    def _update_regime(self, avg_reasoning: float, avg_empathy: float, avg_dom: float):
        g = self.gini_wealth
        pop_scaled = len(self.agents_alive()) * self.scale_factor
        violence = self.last_metrics.get("violence_rate", 0.0)
        avg_language = self.latent_mean("language")
        avg_affect_reg = self.latent_mean("affect_reg")
        top_share, top_fear, top_coop = self._top5_power()

        if pop_scaled < 0.2 * self.initial_population_scaled:
            self.regime = "Colapso Social"
        elif top_share > 0.65 and top_fear > 0.75:
            self.regime = "Tirania Psicopatica"
        elif top_share > 0.60 and top_coop > 0.80:
            self.regime = "Oligarquia Carismatica / Liderazgo Empatico"
        elif g > 0.58 and avg_dom > 0.70:
            self.regime = "Oligarquia Predatoria"
        elif avg_reasoning > 0.78 and g > 0.55 and avg_empathy < 0.55:
            self.regime = "Tecnocracia Autoritaria"
        elif avg_empathy > 0.75 and g < 0.38 and violence < 0.08:
            self.regime = "Democracia Ilustrada / Comunidad Empatica"
        elif violence > 0.30:
            self.regime = "Anarquia Violenta"
        elif avg_reasoning < 0.4 and violence > 0.2:
            self.regime = "Anarquia Tribal"
        elif g > 0.55 and avg_dom > 0.6 and avg_language > 0.6:
            self.regime = "Oligarquia Carismatica"
        elif self.legal_formalism > 0.7 and self.liberty_index < 0.3:
            self.regime = "Dictadura Burocratica"
        elif avg_dom > 0.6 and avg_affect_reg > 0.6:
            self.regime = "Teocracia Moralista"
        elif g > 0.6 and avg_empathy < 0.4:
            self.regime = "Plutocracia"
        else:
            self.regime = "Regimen Transicional"

    def latent_mean(self, key: str) -> float:
        alive = self.agents_alive()
        if not alive:
            return 0.0
        return float(np.mean([a.latent.get(key, 0.5) for a in alive]))

    def _update_metrics(self):
        alive = self.agents_alive()
        pop = len(alive)
        if pop == 0:
            self.last_metrics = {
                "coop_rate": 0.0,
                "violence_rate": 0.0,
                "life_expectancy": 0.0,
                "avg_reasoning": 0.0,
                "avg_empathy": 0.0,
                "avg_dominance": 0.0,
                "male_violence_rate": 0.0,
                "female_violence_rate": 0.0,
                "births": 0.0,
                "sex_ratio": 0.0,
                "coalition_count": 0.0,
                "coalition_wins": 0.0,
                "sneaky_success_rate": 0.0,
            "male_male_conflicts": 0.0,
            "female_indirect_competition": 0.0,
            "mating_inequality": 0.0,
            "mean_harem_size": 0.0,
            "repro_gini_males": 0.0,
            "male_childless_share": 0.0,
            "mean_partners_male": 0.0,
            "mean_partners_female": 0.0,
        }
            self.running = False
            return

        coop = sum(1 for a in alive if a.last_action == "coop")
        viol = sum(1 for a in alive if a.last_action == "violence")
        avg_age = float(np.mean([a.age for a in alive])) if alive else 0.0
        avg_reasoning = self.latent_mean("reasoning")
        avg_empathy = self.latent_mean("empathy")
        avg_dom = self.latent_mean("dominance")
        male_count = sum(1 for a in alive if a.gender == "Male")
        female_count = sum(1 for a in alive if a.gender == "Female")
        leadership_scores = sorted([a.latent.get("reasoning", 0.5) * (1.0 - a.latent.get("sociality", 0.5)) for a in alive], reverse=True)
        top_leaders = leadership_scores[: max(1, int(0.05 * len(leadership_scores)))] if leadership_scores else []
        top_mean_leader = float(np.mean(top_leaders)) if top_leaders else 0.0
        conscious_mean = float(np.mean([a.conscious_core.get("awareness", 0.0) for a in alive])) if alive else 0.0
        allied_share = (sum(1 for a in alive if getattr(a, "alliance_id", None)) / pop) if pop else 0.0
        sneaky_attempts = max(1.0, self.step_events.get("sneaky_attempts", 0.0))
        sneaky_success_rate = self.step_events.get("sneaky_success", 0.0) / sneaky_attempts
        mating_success_vals = [a.mating_success for a in alive if a.gender == "Male"]
        mean_harem_size = (sum(mating_success_vals) / max(1, male_count)) if mating_success_vals else 0.0
        male_children = [len(a.children_ids) for a in alive if a.gender == "Male"]
        repro_gini_m = gini(male_children) if male_children else 0.0
        male_childless = sum(1 for c in male_children if c == 0)
        male_childless_share = male_childless / max(1, len(male_children)) if male_children else 0.0
        mean_partners_male = float(np.mean([len(a.mates_lifetime) for a in alive if a.gender == "Male"])) if male_count else 0.0
        mean_partners_female = float(np.mean([len(a.mates_lifetime) for a in alive if a.gender == "Female"])) if female_count else 0.0

        self.last_metrics = {
            "coop_rate": coop / pop,
            "violence_rate": viol / pop,
            "male_violence_rate": self.step_events.get("male_violence", 0.0) / max(pop, 1),
            "female_violence_rate": self.step_events.get("female_violence", 0.0) / max(pop, 1),
            "life_expectancy": avg_age,
            "avg_reasoning": avg_reasoning,
            "avg_empathy": avg_empathy,
            "avg_dominance": avg_dom,
            "avg_leadership": top_mean_leader,
            "conscious_awareness": conscious_mean,
            "alliances_count": len(self.alliances),
            "allied_share": allied_share,
            "births": self.step_events.get("births", 0.0),
            "sex_ratio": male_count / max(pop, 1),
            "coalition_count": self.step_events.get("coalition_count", 0.0),
            "coalition_wins": self.step_events.get("coalition_wins", 0.0),
            "sneaky_success_rate": sneaky_success_rate,
            "male_male_conflicts": self.step_events.get("male_male_conflicts", 0.0),
            "female_indirect_competition": self.step_events.get("female_indirect_competition", 0.0),
            "mating_inequality": gini(mating_success_vals),
            "mean_harem_size": mean_harem_size,
            "repro_gini_males": repro_gini_m,
            "male_childless_share": male_childless_share,
            "mean_partners_male": mean_partners_male,
            "mean_partners_female": mean_partners_female,
        }

        self.legal_formalism = clamp01(top_mean_leader * (1 + 0.3 * self.institution_pressure))
        self.liberty_index = clamp01(0.6 * avg_empathy + 0.2 * (1 - avg_dom))
        self.gini_wealth = gini([a.wealth for a in alive])
        self._update_regime(avg_reasoning, avg_empathy, avg_dom)
        if hasattr(self, "recent_violence_events") and self.recent_violence_events:
            decay = max(0, len(self.recent_violence_events) - 10)
            if decay > 0:
                self.recent_violence_events = self.recent_violence_events[-10:]

    def step(self):
        self._reset_step_events()
        self.agents.shuffle_do("step")
        # economic growth from high reasoning prosocial agents
        for a in self.agents_alive():
            if a.latent.get("moral_prosocial", 0.5) > 0.5 and a.latent.get("reasoning", 0.5) > 0.6:
                growth = float(self.rng.uniform(0.1, 0.5) * a.latent.get("reasoning", 0.5))
                a.wealth += growth
                self.total_wealth += growth
                a.nd_contribution += growth
        if hasattr(self, "recent_violence_events") and len(self.recent_violence_events) > 5:
            self.legal_formalism = clamp01(self.legal_formalism + 0.05)
        self._update_alliances()
        self._update_metrics()
        self.total_wealth = sum(agent.wealth for agent in self.agents_alive())
        self.datacollector.collect(self)
        if not self.agents_alive():
            self.running = False

    def redistribute_wealth_to_allies(self, amount: float, allies: List[Citizen]):
        if not allies:
            return
        share = amount / len(allies)
        for ally in allies:
            ally.wealth += share

    def log_event(self, tag: str, payload: object):
        if not hasattr(self, "event_log"):
            self.event_log = []
        self.event_log.append((tag, payload))

    def register_violence(self, attacker: Citizen, victim: Citizen):
        self.recent_violence_events.append((attacker.unique_id, victim.unique_id))
        if attacker.gender == "Male":
            self.step_events["male_violence"] += 1
        else:
            self.step_events["female_violence"] += 1
