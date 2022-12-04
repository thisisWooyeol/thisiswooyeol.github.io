---
layout: page
title: MBMRL for Flight
description: Review on "Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads"
img: 
importance: 6
category: papers review
---

TL;DR:
- Proposed a **meta-learning approach** that "learns how to learn" **models of various payloads** that a priori unknown physical properties vary dynamics **in an online fashion**.
- **By augmenting the dynamic model with stochastic latent variables**, the authors infused meta-learning approach into MBRL.
- **Usage of unknown latent variables** leads to outperforming results compared to pure model-based methods.
- By **online adaptation mechanism**, the dynamics variables converges and the tracking error also reduces.
- Proposed approach successfully completes the full end-to-end payload trasportation task as well as transporting a suspended payload towards a moving target, around an obstacle by following a predefined path, and along trajectories dictated using a "wand"-like interface.
