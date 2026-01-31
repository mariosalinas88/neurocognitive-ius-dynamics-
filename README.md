# Neuro-Societies

Agent-based simulations demonstrating how neurodivergent cognitive styles scale into distinct normative architectures through probabilistic interactions.

## Hypothesis

Heterogeneous cognitive styles (neurodivergences as unique neural "hardware") interacting at scale generate systematically different normative architectures (e.g., cooperation regimes, sanction patterns, dominance equilibria, tolerance thresholds). These emerge from innate biological predispositions modulated by dynamic, probabilistic "software" development based on interaction outcomes, reflecting quantum-like collapse of social potentialities. High-reasoning profiles (e.g., autism) drive innovation and development when empathetic, but may lead to dominance or collapse if dark-biased, while low-resilience profiles (e.g., depression) are more victimized in competitive environments.

## What This Repository Contains

- `model.py`: Core agent-based society model, including probabilistic interactions, neuroplasticity, and metrics for economic/social/political emergence.
- `profiles.json`: Neutral biological hardware profiles (e.g., Neurotípico Neutro, Eliza TDAH Neutro) with innate predispositions and Gaussian variability for realistic simulations.
- `run.py`: Single-run simulation script with configurable arguments (e.g., --spectrum_level, --resilience_bias) and detailed reporting on per-profile outcomes (wealth, leadership, victims, etc.).
- `run_batch.py`: Batch experiment script for comparative analysis across scenarios, generating `test_runs.csv` with inputs/outputs including dominant/victimized profiles and societal metrics.
- `server.py`: Interactive Solara dashboard for visualizing simulation history, per-profile stats, probabilistic distributions, and hypothesis-driven metrics (e.g., ND contribution to development).
- `test_runs.csv`: Sample results from batch runs, showing regime evolution, economic growth (total_wealth), and per-profile dominance/victimization/benefits.
- `summary_evolution.csv`: Per-run model-level metrics (population, cooperation/violence rates, Gini, etc.).
- `per_profile_stats.csv`: Per-run detailed stats by neuroprofile (wealth_avg for economy, leadership_avg for politics, rep_avg for social success, victims_avg for victimization, etc.).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Neuro-Societies
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Unix/Mac
   .venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Assumes `requirements.txt` includes: mesa, numpy, pandas, matplotlib, solara, etc. If not present, generate it via `pip freeze > requirements.txt` after installing manually.)

## Quickstart

- Run a single simulation:
  ```bash
  python run.py --steps 200 --spectrum_level 2 --resilience_bias high
  ```
  Outputs: Console report on per-profile success/victimization, `summary_evolution.csv`, `per_profile_stats.csv`.

- Run batch experiments for comparative analysis:
  ```bash
  python run_batch.py
  ```
  Outputs: `test_runs.csv` with scenario inputs (mix %, levels, biases) and results (regime, total_wealth economy, per-profile dominants/victims/benefited/harmed, etc.).

- Launch interactive dashboard:
  ```bash
  python -m solara run server.py
  ```
  Visualize: Simulation metrics, per-profile bars (wealth/leadership/victims/success), probabilistic distributions (e.g., change deltas), and hypothesis tabs (ND contribution vs. costs).

## Usage Examples

- Test high-reasoning empathetic scenario (expected: stable development, benefited ND):
  ```bash
  python run.py --initial_moral_bias high_prosocial --emotional_bias low
  ```

- Analyze batch results: Open `test_runs.csv` to compare regimes across scenarios (e.g., inclusive mixed yields low victims for ND, competitive harms low-reasoning profiles).

- Custom runs: Modify `run_batch.py` SCENARIOS for new mixes (e.g., 70% autism + 30% ADHD).

## Results Interpretation

Simulations demonstrate the hypothesis:
- High-reasoning ND (autism/high IQ) boost economy (total_wealth/gini low) and dominate politics/social positively in empathetic/inclusive scenarios, but incur costs/victimization in competitive/dark-biased ones.
- Low-resilience profiles (depression/dark predisposition) are more victimized (high violence_received), while empathetic/high-sociality lead social/cultural stability.
- Probabilistic nature ensures variability: Runs show uncertain outcomes, with ND innovation emerging non-deterministically.
- Key metrics: Per-profile success_rate > avg = benefited; < avg = harmed. Dominants: % top wealth/leadership/rep.

For deficiencies: If collapses persist, tune gauss_std lower (less variability) or resilience_bias higher in args.

## Contributing

Fork the repo, create a branch, and submit a PR with changes. Focus on enhancing probabilistic mechanics or adding profiles.

## License

MIT License. See LICENSE file for details.
