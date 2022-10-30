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

<br/>
<br/>
<br/>

--------

# Preliminaries
### Maxmimum Entropy Reinforcement Learning
Maximum Entropy RL objective is

$$
\pi_\mathrm{MaxEnt}^* =\underset{\pi}{\mathrm{argmax}}\sum_t \mathbb E_{(s_t,a_t) \sim \rho_\pi} \left[\sum_{l=t}^\infty \gamma^{l-t} \mathbb E_{(s_l,a_l)} \left[r(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t)) \right]\right]
$$

where $$ \alpha $$ is a temperature parameter that determines the relative importance of entropy and reward. This objective aims to maximize the entropy of the **entire trajectory distribution for the policy $$ \pi $$**. See more details of objective with discount factor in Appendix A.

<br/>
<br/>

### Soft Value Functions and Energy-Based Models
General energy-based policies of the form

$$
\pi(a_t|s_t) \propto \mathrm{exp}(-\mathcal E(s_t,a_t)),
$$

where $$ \mathcal E $$ is an energy function can represent very general class of distributions. There is a close connection between such energy-based models and *soft* versions of value functions and Q-functions, where we set $$ \mathcal E(s_t,a_t) = \frac{1}{\alpha}Q_{soft}(s_t,a_t) $$ and use the following theorem:

<br/>

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
- If one rewrites the maximum entropy objective with soft Q-function, the discounted maximum entropy policy objective can be defined as

$$
J(\pi) \triangleq \sum_t \mathbb E_{(s_t,a_t) \sim \rho_\pi} \left[Q_{soft}^\pi(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t))\right].
$$

- Check Appendix A.1 Theorem 4. Given a policy $$ \pi $$ , defining a new policy $$ \tilde{\pi} $$ as 

$$ 
\tilde{\pi} \propto \mathrm{exp} (Q_{soft}^\pi (s, \cdot), \quad \forall s 
$$

maximize the objective $$ \alpha \mathcal H(\pi(\cdot \mid s)+\mathbb E_{a \sim \pi} \left[ Q_{soft}^\pi (s,a)\right] $$.
- From theorem 4., by applying policy iteration $$ \pi_{i+1}(\cdot \mid s) \propto \mathrm{exp}(Q_{soft}^{\pi_i}(s,\cdot)) $$ from an arbitrary policy $$ \pi_0 $$ we can get $$ \pi_\infty(a \mid s) \propto_a \mathrm{exp}(Q^{pi_infty}(s,a)) $$.

<br/>

**Theorem 2.** *The soft Q-function in (4) satisfies the soft bellman equation*

$$
Q_{soft}^* (s_t,a_t)=r_t+\gamma \mathbb E_{s_{t+1} \sim p_s} \left[ V_{soft}^* (s_{t+1}) \right],
$$

*where the soft value function* $$ V_{soft}^* $$ *is given by (5).*

How to prove thm 2.: check Appendix A.2.

<br/>
<br/>
<br/>

-------
# Training Expressive Energy-Based Models via Soft Q-Learning
### Soft Q-Iteration
**Theorem 3.** *Soft Q-iteration. Let $$ Q_{soft}(\cdot \mid \cdot) $$ and $$ V_{soft}(\cdot) $$ be bounded and assume that $$ \int_\mathcal{A} \mathrm{exp}\left(\frac{1}{\alpha}Q_{soft}(\cdot,a')\, da' < \infty $$ and that* $$ Q_{soft}^* < \infty $$ *exists. Then the fixed-point iteration*

$$
\begin{matrix}
Q_{soft}(s_t,a_t) \gets r_t+\gamma \mathbb E_{s_{t+1} \sim p_s} \left[ V_{soft}(s_{t+1})\right], \quad \forall s_t, a_t \\
V_{soft}(s_t) \gets \alpha \ \mathrm{log} \int_\mathcal{A} \mathrm{exp}\left(\frac{1}{\alpha} Q_{soft}(s_t, a')\right)da', \quad \forall s_t
\end{matrix}
$$

*converges to* $$ Q_{soft}^* $$ *and* $$ V_{soft}^* $$ *, respectively.*

How to prove thm 3.: check Appendix A.2 that soft Bellman backup operator $$ \mathcal T $$ is a contraction.

<br/>

**PROBLEM**
1. Soft Bellman backup cannot be performed exactly in **continuous or large state and action spaces**.
2. **Sampling from the energy-based model in (6) is intractable** in general.

<br/>
<br/>

### Soft Q-Learning
To handle problem 1, this paper express the Bellman backup process as a stochastic optimization. For soft value function, expectation via importance sampling is used.

$$
V_{soft}^\theta (s_t)=\alpha \ \mathrm{log} \mathbb E_{q_{a'}} \left[\frac{\mathrm{exp}(\frac{1}{\alpha} Q_{soft}^\theta (s_t,a'))}{q_{a'}(a')} \right].
$$

While $$ q_{a'} $$ can be an arbitrary distribution over the action space, the current policy is used in this paper. More details are in Appendix C.2.

For soft q-iteration, minimizing the following objective is used.:

$$
J_Q(\theta)=\mathbb E_{s_t \sim q_{s_t}, a_t \sim q_{a_t}} \left[ \frac{1}{2} \left(\hat Q_{soft}^{\bar{\theta}} (s_t,a_t)-Q_{soft}^\theta(s_t,a_t)\right)^2\right],
$$

where $$ q_{s_t}, q_{a_t} $$ are positive over $$ \mathcal S $$ and $$ \mathcal A $$ respectively, $$ \hat Q_{soft}^{\bar{\theta}} (s_t,a_t) =r_t+\gamma \mathbb E_{s_{t+1} \sim p_s} \left[ V_{soft}^\bar{\theta} (s_{t+1})\right] $$ is a target Q-value, with $$ V_{soft}^\bar{\theta} (s_{t+1}) $$ given by (10) and $$ \theta $$ being replaced by the target parameters, $$ \bar{\theta} $$.

While ampling distributions $$ q_{s_t} $$ and $$ q_{a_t} $$ can be arbitrary, real samples(=replay memories) are used in this paper.

<br/>
<br/>

### Approximate Sampling and Stein Variational Gradient Descent (SVGD)
