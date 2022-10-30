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
\pi_\mathrm{MaxEnt}^* =\underset{\pi}{\mathrm{argmax}}\sum_t \mathbb E_{(s_t,a_t) \sim \rho_\pi} \left[\sum_{l=t}^\infty \gamma^{l-t} \mathbb E_{(s_l,a_l)} \left[r(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t)) \right]\right]
$$

where $$ \alpha $$ is a temperature parameter that determines the relative importance of entropy and reward. This objective aims to maximize the entropy of the **entire trajectory distribution for the policy $$ \pi $$**. See more details of objective with discount factor at Appendix A.

## Soft Value Functions and Energy-Based Models
General energy-based policies of the form

$$
\pi(a_t|s_t) \propto \mathrm{exp}(-\mathcal E(s_t,a_t)),
$$

where $$ \mathcal E $$ is an energy function can represent very general class of distributions. There is a close connection between such energy-based models and *soft* versions of value functions and Q-functions, where we set $$ \mathcal E(s_t,a_t) = \frac{1}{\alpha}Q_{soft}(s_t,a_t) $$ and use the following theorem:

**Theorem 1.** *Let the soft Q-function be defined by*

$$
Q_{soft}^* (s_t,a_t)=r_t+\mathbb E_{(s_{t+1},...) \sim \rho_\pi} \left[\sum_{l=1}^\infty \gamma^l(r_{t+l}+\alpha \mathcal H(\pi_\mathrm{MaxEnt}^* (\cdot|s_{t+l})))\right],
$$

*and soft value function by*

$$
V_{soft}^* (s_t)=\alpha \ \mathrm{log}\int_{\mathcal{A}} \mathrm{exp} \left( \frac{1}{\alpha} Q_{soft}^* (s_t,a') \right)\, da'.
$$

*Then the optimal policy for (2) is given by*

$$ 
\pi_\mathrm{MaxEnt}^* (a_t|s_t) = \mathrm{exp} \left(\frac{1}{\alpha} \left( Q_{soft}^* (s_t,a_t) - V_{soft}^* (s_t) \right) \right).
$$

How to prove thm 1.:
1. If one rewrites the maximum entropy objective with soft Q-function, the discounted maximum entropy policy objective can be defined as

$$
J(\pi) \triangleq \sum_t \mathbb E_{(s_t,a_t) \sim \rho_\pi} \left[Q_{soft}^\pi(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t))\right].
$$

2. Check Appendix A.1. Given a policy $$ \pi $$ , defining a new policy $$ \tilde{\pi} $$ as $$ \tilde{\pi} \propto \mathrm{exp} (Q_{soft}^\pi (s, \cdot), \quad \forall s $$ maximize the objective $$ \mathcal H(\pi(\cdot|s)+\mathbb E_{a \sim \pi} \left[ Q_{soft}^\pi (s,a)\right] $$.
3. 
