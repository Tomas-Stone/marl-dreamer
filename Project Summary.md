# Project Summary — Dreamer with Teammate Modeling for Cooperative MARL

**Goal:** Build and evaluate a Dreamer-style **world model architecture** for **multi-agent reinforcement learning** that explicitly models teammates’ behavior (Theory-of-Mind, ToM) to achieve robust **zero-shot and few-shot coordination** in cooperative, partially observable environments.

---

## Motivation

- **Problem:** In cooperative multi-agent tasks, teammates’ actions appear as stochastic uncertainty. Standard Dreamer or model-free MARL agents struggle to generalize to **new partners** or to adapt quickly.
- **Gap:** Prior Dreamer-based MARL works (MA-Dreamer, CoDreamer, MACD, GAWM, MAMBA) focus on centralized training, counterfactuals, or communication, but **none explicitly model a teammate’s intent**.
- **Opportunity:** A ToM-inspired teammate model integrated into a world model could yield **faster adaptation, improved generalization, and stronger human-AI coordination.**

---

## Settings where we aim to contribute

- **Cooperative, partially observable multi-agent environments** without explicit communication.
- **Zero-shot coordination (ZSC):** performing well with unseen partners at test time.
- **Few-shot adaptation:** improving performance after observing a small number of teammate actions.
- **Human-AI interaction:** robustness when the teammate is a human with hidden states unavailable during training.

---

## Architecture idea

- **Core:** Recurrent State Space Model (RSSM) with deterministic memory hth_t and stochastic latent ztz_t.
- **Factorization:** split latent into global dynamics and per-agent (teammate) latents.
- **Teammate Modeling Head:** auxiliary module that predicts teammates’ next actions or latent goals from observed history.
- **Partner Embedding:** inferred online from teammate actions, conditions the actor for quick adaptation.
- **Training:**
    - World model loss = observation + reward + continuation prediction + KL regularization.
    - Auxiliary ToM loss = cross-entropy/log-likelihood on teammates’ actions.
    - Control loss = Dreamer-style actor/critic training with imagination rollouts conditioned on sampled teammate behavior.

---

## Benchmarks

- **Debug:** Multi-Agent Particle Environments (MPE).
- **Main:** Overcooked-AI (canonical ZSC testbed).
- **Generalization:** DeepMind Melting Pot (unseen partner splits).
- **Optional:** Hanabi (explicit Theory-of-Mind, symbolic domain).

---

## What we will measure

- **Task performance:** episodic reward, sample efficiency.
- **Zero-shot coordination (ZSC):** performance with unseen partners without adaptation.
- **Few-shot adaptation:** performance gain after N observed steps of partner interaction.
- **Cross-play matrices:** performance across all partner pairs (robustness/generalization).
- **Calibration of predictions:** how well the teammate head predicts true partner actions (ECE/Brier score).
- **Ablations:** remove ToM head, remove factorization, remove CTDE to isolate contributions.

---

## Expected contributions

- A **Dreamer + ToM** architecture that models teammates explicitly.
- A systematic evaluation on **ZSC + few-shot coordination** benchmarks.
- Evidence that partner modeling improves coordination over Dreamer baselines.
- A public repo and paper showing how world models can support **human-AI generalization** by reducing apparent stochasticity of teammates’ behavior.