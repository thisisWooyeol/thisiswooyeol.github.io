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
- By **augmenting the dynamic model with stochastic latent variables**, the authors infused meta-learning approach into MBRL.
- **Usage of unknown latent variables** leads to outperforming results compared to pure model-based methods.
- By **online adaptation mechanism**, the dynamics variables converges and the tracking error also reduces.
- Proposed approach successfully completes the full end-to-end payload trasportation task as well as transporting a suspended payload towards a moving target, around an obstacle by following a predefined path, and along trajectories dictated using a "wand"-like interface.

<br/>
<br/>
<br/>

--------

# Introduction
<br/>

### Task definition
<br/>

<div class="row justify-content-center">
    <div class="col-10">
        {% include figure.html path="assets/img/MBMRL-Flight/MBMRL-Flight-transport-task.PNG" title="Quadcopter Payload Transport Task" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure from "Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads"
</div>

<br/>

### Limits of hand-designed model on the flight tasks

**The complex interaction between the magnetic gripper and the payload** are unlikely to be represented accurately by hand-designed models. Even more unpredictable is **the effect of the payload on the dynamics of the quadcopter** when the payload is lifted off the ground.

<br/>

### Need for fast adaptation

Conventional learning-based methods typically require a large amount of data to learn accurate models, and therefore may be slow to adapt. The payload adaptation task illustrates the need for fast adaptation; the robot must very quickly **determine the payload parameters**, and then **adjust its motor commands accordingly**.

<br/>

### Proposed method: model-based meta-RL

A **predictive dynamics model** which is **augmented with stochastic latent variables** is learned with different payload masses and tether lengths. In test time, it uses **variational inference to estimate the corresponding posterior distribution over these latent variables**.

<br/>
<br/>
<br/>

--------

# Related Work
<br/>

**Controlling Aerial Vehicles**

- Prior works on control for aerial vehicles that relying on **manual system identification** : require *a priori* knowledge of the system
- Automating system identification via **online parameter adaptation** : still rely on domain knowledge for the equations of motion
- Adaptive model-based controller : require *a priori* knowledge of dynamic systems, parameters for state estimation, etc.
- In contrast, the proposed data-driven method only requires **the pixel location of the payload** and **the quadrotor's commanded actions to control and adapt to the suspended payload's dynamics**.

<br/>

**End-to-end Learning-based Approach**

- The learning processes of value-based and gradient-based methods generally take hours or even days, making it **poorly suited for safety-critical and resource-constrained quadcopters**.
- Model-based reinforcement learning (MBRL) can provide **better sample efficiency**, but must MBRL methods are **designed to model a single task with unchanging dynamics**, and therefore do not adapt to rapid online changes in the system dynamics.

<br/>

**Model-based Meta-learning**

- [`O'Connell et al.`](https://arxiv.org/abs/2103.01932) used MAML algorithm for adapting a drone's internal dynamics model. The resulting adapted model, however, **did not improve the performance of the closed-loop controller**. In contrast, the authors demonstrate that their method does improve performance of the model-based controller.
- [`Nagabandi et al. (GrBAL)`](https://arxiv.org/abs/1803.11347) and [`Kaushik et al.`](https://arxiv.org/abs/2003.04663) explored **meta-learning for online adaptation in MBRL** for a legged robot, which demonstrated improved closed-loop controller performance with adapted model.

<br/>
<br/>
<br/>

--------

# Preliminaries
<br/>

Model-based reinforcement learning estimates the underlying dynamics from data by training a dynamics model $$ p_\theta(s_{t+1} \mid s_t, a_t) $$ via maximum likelihood

$$
\begin{align}
\theta* &= \underset{\theta}{\mathrm{argmax}} \ p(\mathcal D^\text{train} \mid \theta) \nonumber\\
 &= \underset{\theta}{\mathrm{argmax}} \sum_{(s_t,a_t,s_{t+1}) \in \mathcal D^\text{train}} \mathrm{log} \ p_\theta(s_{t+1} \mid s_t, a_t).
\end{align}
$$

To instantiate this method, the authors extend the [`PETS algorithm`](https://arxiv.org/abs/1805.12114), which handles expressive **neural network dynamics models** to attain good sample efficiency as model-based algorithms and **asymptotic performance** as model-free algorithms. PETS uses an ensemble of probabilistic neural network models, each parameterizing a Gaussian distribution of $$ s_{t+1} $$ conditioned on both $$ s_t $$ and $$ a_t $$ . The learned dynamics model is used to plan and execute actions via model predictive control (MPC) with trajectory sampling (TS)(due to probabilistic models). Trajectory sampling predicts plausible state trajectories begins by creating $$ P $$ particles from the current state, $$ s_{t=0}^p =s_0 \forall p $$ . Each particle is then propagated by: $$ s_{t+1}^p \sim \tilde{f}_ {\theta_{b(p,t)}} (s_t^p,a_t) $$ , according to a particular bootstrap $$ b(p,t) \text{in} \lbrace 1, \ldots, B \rbrace $$ , where $$ B $$ is the number of bootstrap models in the ensemble. Unlike a common technique to compute the optimal action sequence (random sampling shooting), PETS used cross-entropy method (CEM), which samples actions from a distribution closer to previous action samples that yielded high reward. Now the overall PETS algorithm can be summarized in Algorithm 1.

<div class="row justify-content-center">
    <div class="col-8">
        {% include figure.html path="assets/img/MBMRL-Flight/PETS-algorithm.PNG" title="PETS algorithm" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure from "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models"
</div>

<br/>
<br/>
<br/>

--------

# Model-Based Meta-Learning For Quadcopter Payload Transport
<br/>

<div class="row justify-content-center">
    <div class="col-10">
        {% include figure.html path="assets/img/MBMRL-Flight/MBMRL-Flight-SystemDiagram.PNG" title="System diagram of MBMRL algorithm for flight" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure from "Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads"
</div>

### Data Collection

Data is collected by manually piloting the quadcopter along random paths for each of the $$ K $$ suspended payloads. A dataset $$ \mathcal D^\text{train} $$ consists of $$ K $$ separate datasets $$ \mathcal D^\text{train} \doteq \mathcal D^\text{train}_ {1:K} \doteq \lbrace D^\text{train}_ 1, \ldots , D^\text{train}_ K \rbrace $$ , one per payload task.

<br/>

### Model Training with Known Dynamics Variables
