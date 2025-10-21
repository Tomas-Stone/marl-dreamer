# Dreaming of Others 🧠

A multi-agent reinforcement learning system that extends world models to cooperative settings by learning to model teammates as structured latent processes. 

## Overview

This project implements a novel architecture for cooperative MARL that treats teammates not as environmental noise, but as learnable components within an agent's world model. By factorizing the latent state into **environment** and **teammate** components, and training a Theory-of-Mind (ToM) head to infer partner behavior, agents can coordinate with unseen collaborators without access to their policies or observations.

### Key Innovation

Traditional world models like Dreamer excel in single-agent settings but struggle with the non-stationarity introduced by changing teammate policies. Our approach:

- **Factorizes latent states** into `z_env` (environment dynamics) and `z_team` (teammate behavior)
- **Learns a ToM head** that predicts teammate actions from observed behavior
- **Conditions imagination** on inferred teammate latents for socially-aware planning
- **Enables zero-shot coordination** with unseen partners through learned teammate representations

## Implementation Roadmap

We're building this from scratch to deeply understand each component:

### Phase 1: Single-Agent Foundation
- ✅ **VAE**: Learn latent representations of observations
- ✅ **Replay Buffer**: Store and sample trajectory sequences
- 🔄 **RSSM**: Recurrent state-space model with prior/posterior
- 🔄 **World Model**: Complete dynamics learning with reconstruction + reward prediction
- 🔄 **Actor-Critic**: Policy learning through imagination

### Phase 2: Multi-Agent Extension
- ⬜ **MARL Environment**: Transition to cooperative multi-agent tasks
- ⬜ **Factorized Latents**: Split RSSM into environment and teammate components
- ⬜ **ToM Head**: Implement teammate action prediction
- ⬜ **Social Imagination**: Sample teammate behaviors during rollouts

## Target Environments

- **Multi-Agent Particle Envs**: Lightweight diagnostic tasks
- **Overcooked-AI**: Primary testbed for zero-shot coordination
- **Melting Pot**: Large-scale robustness evaluation across diverse social contexts

## Research Goals

This implementation aims to demonstrate:
- **Reduced non-stationarity** by explicitly modeling teammate variability
- **Zero-shot coordination** with previously unseen partners
- **Few-shot adaptation** through online inference of teammate latents
- **Interpretable representations** of partner behavior and intent

---

*Building agents that dream not only of the worlds they inhabit, but of the minds that share them.*
