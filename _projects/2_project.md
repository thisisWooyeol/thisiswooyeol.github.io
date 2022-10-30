---
layout: page
title: Soft Q-Learning
description: Reinforcement Learning with Deep Energy-Based Policies
img: assets/img/softqlearning.PNG
importance: 1
category: papers review
---

# TL;DR:
- A method for **learning expressive energy-based policies for continuous states and actions**
- Apply the method to learning **maximum entropy policies**
- Used **amortized Sten variational gradient descent** to learn a stochastic sampling network
- **Improved exploration and compositionality** and connection to **actor-critic methods**


# Preliminaries
## Maxmimum Entropy Reinforcement Learning
Maximum Entropy RL objective is

$$
{\pi_\mathrm{MaxEnt}}^*=\underset{\pi}{\mathrm{argmax}}\sum_t \mathbb E_{(s_t,a_t) \sim \rho_\pi} \left[\sum_{l=t}^\infty \gamma^{l-t} \mathbb E_{(s_l,a_l)} \left[r(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t)) \right]\right]
$$

where $$ \alpha $$ is a temperature parameter that determines the relative importance of entropy and reward. This objective aims to maximize the entropy of the **entire trajectory distribution for the policy $$ \pi $$**. See more details of objective with discount factor at Appendix A.

## Soft Value Functions and Energy-Based Models
General energy-based policies of the form

$$
\pi(a_t|s_t) \propto \mathrm{exp}(-\mathcal E(s_t,a_t)),
$$

where $$ \mathcal E $$ is an energy function can represent very general class of distributions. There is a close connection between such energy-based models and *soft* versions of value functions and Q-functions, where we set $$ \mathcal E(s_t,a_t) = \frac{1}{\alpha}Q_{soft}(s_t,a_t) $$ and use the following theorem:
