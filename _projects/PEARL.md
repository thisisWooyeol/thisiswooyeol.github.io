---
layout: page
title: PEARL
description: Review on "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables"
img: assets/img/1.jpg
importance: 4
category: papers review
---

# TL;DR:
- Developed an **off-policy meta-RL algorithm** that disentangles **sample efficiency** and **task reasoning** problems.
- Perform **online probabilistic filtering of latent task variables** to infer how to solve a new task from small amounts of experience.

<br/>
<br/>
<br/>

--------

# Introduction
<br/>

### Meta-RL problems
Modern meta-learning is predicated on the assumption that **the distribution of data used for adaptation will match across meta-testing and meta-test**. As on-policy data will be used to adapt at meta-test time, on-policy data should be used during meta-training time as well. This makes algorithms exceedingly inefficient during meta-training and makes them inherently difficult to meta-train a policy to adapt using off-policy data.

<br/>
<br/>

### How to tackle the problem ?
To achieve both meta-training efficiency(off-policy) and rapid adaptation(task inference), they integrate **online inference of probabilistic context variables with existing off-policy RL algorithms**. During meta-training, PEARL learns a probabilistic encoder that **accumulates the necessary statistics from past experience into the context variables**. At meta-test time, trajectories are collected with sampled context variables(="task hypotheses") and the collected trajectories update the posterior over the context variables, achieveing rapid trajectory-level adaptation. 

**Disentangling task inference from action** makes PEARL particularly amenable to off-policy meta-learning; the policy can be optimized with off-policy data while the probabilistic encoder is trained with on-policy data to minimize distribution mismatch between meta-train and meta-test.

<br/>
<br/>

### Experiments summary
Probabilistic embeddings for actor-critic RL (PEARL) achieves state-of-the-art results with 20-100X improvement in meta-training sample efficiency and substantial increases in asymptotic performance over prior state-of-the-art on six continuous control meta-learning environments. They further examine how PEARL conducts **structured exploration** to adapt rapidly to new tasks in a 2-D navigation environments with sparse rewards.

<br/>
<br/>
<br/>

-------

# Related Work
<br/>

### Meta-learning
<br/>

***Context-based* meta-RL**
- Prior works with recurrent and recursive meta-RL method adapt to new tasks by aggregating experience into a latent representation on which the policy is conditioned. (whole experience = task specific context)
- *However*, PEARL represents task contexts with **probabilistic latent variables**, enabling reasoning over task uncertainty.
- Instead of using recurrence, PEARL leverages the Markov property in the **permutation-invariant encoder** to aggregate experience, enabling fast optimization especially for long-horizon tasks while mitigating overfitting.

***Gradient-based* meta-RL**
- Gradient-based meta-RL methods focus on on-policy meta-learning as off-policy data is non-trivial to do with policy gradient methods.
- They found that PEARL (context-based method) is able to reach higher asymptotic performance, in comparison to methods using policy gradients.

**Motivation of permutation-invariant encoder**
- Permutation-invariant embedding function described in this paper is inspired by the embedding function of prototypical networks.
- While they use a distance metric in a learned, *deterministic* embedding space to classify new inputs, embeddings in this paper is *probabilistic* and is used to condition the behavior of an RL agent.

<br/>
<br/>

### Probabilistic meta-learning
- Probabilistic latent task variables: used to adapt model predictions for supervised learning <- *extend this idea to off-policy meta-RL !*
- PEARL infers task variables and explores via **posterior sampling**.

<br/>
<br/>

### Posterior sampling
- In classical RL, posterior sampling maintains a posterior over possible MDPs and enables **temporally extended exploration** by acting optimally according to a sampled MDP.
- PEARL: meta-learned variant of this method; probabilistic context captures the **current uncertainty over the task**, allowing the agent to **explore in new tasks** in a similarly structured manner.

<br/>
<br/>

### Partially observed MDPs
- Adaptation at test time in meta-RL can be viewed as a special case of RL in a POMDP:

$$
\tilde{\mathcal M} = \left\lbrace \mathcal{\tilde{S}, A, \tilde{O}, \tilde{P}, E, r} \right\rbrace \quad \text{where} \quad \mathcal{\tilde{S} = S \times Z}, \tilde{s} = (s,z) \ \text{and} \ \mathcal{\tilde{O} = S}, \tilde o = s
$$
- The explicit state estimation $ p(s_t \mid o_{1:t}) $ from POMDP can be written as task inference $ \hat p(z_t \mid s_{1:t},a_{1:t},r_{1:t}) $, where posterior sampling is used for exploration in new tasks.
- **Variational approach** related to the prior work on solving POMDPs is used to estimate belief over task.

<br/>
<br/>
<br/>

-------

# Problem Statement

- Assume a distribution of tasks $$ p(\mathcal T) $$ , where each task is a MDP.
- A task is defined as $$ \mathcal T = \left\lbrace p(s_0), p(s_{t+1} \mid s_t,a_t), r(s_t,a_t) \right\rbrace $$ . Note that **varying transition functions** and **varying reward functions** constitute task distribution.
- Context $$ c $$ is referred to the history of past transitions. Let $$ c^{\mathcal T}_ n = (s_n, a_n, r_n, s_n') $$ be one transition in the task $$ \mathcal T $$ so that $$ c^{\mathcal T}_ {1:N} $$ comprises the experience collected so far.
- Entirely new tasks are drawn from $$ p(\mathcal T) $$ at test-time.

<br/>
<br/>
<br/>

-------

# Probabilistic Latent Context
<br/>

### Modeling and Learning Latent Contexts
An amortized variational inference approach is used to train an *inference network* $$ q_\phi (z \mid c) $$ , parameterized by $$ \phi $$ , that estimates the posterior $$ p(z \mid c) $$ .
