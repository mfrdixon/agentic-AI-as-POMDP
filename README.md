# Agentic AI as a Partially Observable Markov Decision Process

## A Model Validation Framework for Belief-State, Forecast, and Policy Validation

This repository accompanies the paper:

> **Matthew Dixon**
>
> *Agentic AI as a Partially Observable Markov Decision Process: A Model Validation Framework for Belief-State, Forecast, and Policy Validation*

The paper develops a mathematical framework for validating autonomous AI systems using concepts from stochastic control, Bayesian decision theory, model risk management, and partially observable Markov decision processes (POMDPs).

The central idea is that autonomous systems should not be evaluated solely through their outputs. Instead, validation should be performed across the complete decision pipeline:

```text
Observations
    ↓
Beliefs
    ↓
Forecasts
    ↓
Actions
    ↓
Utility
```

The repository contains:

* POMDP-based latent-state inference
* LLM-based belief-state estimation
* Belief-conditioned forecasting
* Black–Litterman portfolio construction
* Ablation studies
* Calibration diagnostics
* Coverage tests
* Sensitivity analysis
* Figure and table generation code

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```


export RUN_OPENAI=1

You're need API keys for Massive.com and Open AI. Set them as environmental variables: 

export MASSIVE_API_KEY="..."

export OPENAI_API_KEY=".. "
---

# Running the Experiments

Execute:

```bash
python portfolio-agent-test.py
```

Outputs are written to:

```text
pomdp_results/
```

including:

```text
table_1_performance.tex
table_2_utility.tex
table_5_ablation_study.tex
table_6_parameter_sensitivity.tex
table_7_belief_calibration.tex
table_8_belief_coverage.tex
```

and all journal-quality figures.

---

# Reproducing the Paper

The LaTeX manuscript expects generated tables and figures to be located in:

```text
pomdp_results/
```

Compile with:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

# Citation

If you use this repository, please cite:

```bibtex
@article{dixon2026agentic,
  author = {Dixon, Matthew},
  title = {Agentic AI as a Partially Observable Markov Decision Process:
           A Model Validation Framework for Belief-State, Forecast, and Policy Validation},
  year = {2026},
  journal = {Working Paper},
  note = {Submitted to Journal of Risk Model Validation}
}
```

If citing the arXiv version:

```bibtex
@article{dixon2026agentic_arxiv,
  author = {Dixon, Matthew},
  title = {Agentic AI as a Partially Observable Markov Decision Process:
           A Model Validation Framework for Belief-State, Forecast, and Policy Validation},
  journal = {arXiv preprint},
  year = {2026},
  eprint = {XXXX.XXXXX}
}
```

Replace the arXiv identifier once available.
The paper is also available here: https://quiota.substack.com/p/agentic-ai-as-a-partially-observable?r=9a1il


---

# Validation Framework

The framework validates autonomous systems at four layers:

| Layer     | Validation Method                                   |
| --------- | --------------------------------------------------- |
| Beliefs   | Calibration, Brier Score, Log Score, Coverage Tests |
| Forecasts | Information Coefficient, Ablation Studies           |
| Policies  | Utility and Risk-Adjusted Performance               |
| Utility   | Decision Quality and Benchmark Comparison           |

Model risk is decomposed into:

```text
State-Space Risk
Filtering Risk
Forecast Risk
Policy Risk
Utility-Specification Risk
Parameter Risk
```

---

# License

MIT License.

---

# Contact

Matthew Dixon

Quiota LLC

For questions, collaborations, or consulting engagements related to agentic AI validation, model risk management, and trusted AI systems, please contact the author.



