marl-dreamer

🚀 Research repo for experimenting with world models (Dreamer-style) in multi-agent reinforcement learning (MARL) environments.

🎯 Goal

To design and evaluate algorithms that enable agents to:

Learn world models of their environment (latent dynamics, imagination rollouts).

Act cooperatively in partially observable, multi-agent settings.

Use teammate modeling (Theory of Mind, opponent shaping, intent prediction) to coordinate better with new partners (Zero-Shot Coordination).

Generalize to human–AI interaction and multi-robot collaboration.

Our motivating example is:

In Overcooked-AI, if two agents both initially plan to cook the meat, one agent should recognize the other’s intent and adapt — instead preparing the rice — so the team completes the task more efficiently.

🛠️ Research Plan

Single-Agent Dreamer

Implement DreamerV3 components (RSSM, imagination, actor/critic).

Verify on single-agent tasks.

Multi-Agent Dreamer (no ToM)

Extend Dreamer to cooperative MARL with CTDE (centralized training, decentralized execution).

Baseline: IPPO / MAPPO.

Dreamer + Teammate Modeling (ToM)

Add auxiliary head to predict teammate actions/intent.

Condition policy on teammate embedding.

Evaluate Zero-Shot Coordination.

Benchmarks

PettingZoo MPE (Simple Spread, Simple Tag).

Overcooked-AI (ZSC, cross-play evaluation).

(Stretch) Melting Pot for broader social generalization.

Deliverables (by Dec 2025)

📦 Clean open-source repo (this one).

📊 Reproducible experiments + figures (curves, cross-play matrices, ZSC scores).

📄 Paper-style writeup (arXiv-ready, 6–8 pages).