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
- [`Nagabandi et al.(GrBAL)`](https://arxiv.org/abs/1803.11347) and [`Kaushik et al.`](https://arxiv.org/abs/2003.04663) explored **meta-learning for online adaptation in MBRL** for a legged robot, which demonstrated improved closed-loop controller performance with adapted model.

<br/>
<br/>
<br/>

--------

# Preliminaries
<br/>

Model-based reinforcement learning estimates the underlying dynamics from data by training a dynamics model $$ p_\theta(s_{t+1} \mid s_t, a_t) $$ via maximum likelihood

$$
\begin{align}
\theta^* &= \underset{\theta}{\mathrm{argmax}} \ p(\mathcal D^\text{train} \mid \theta) \nonumber\\ \label{eqn:model-train}
 &= \underset{\theta}{\mathrm{argmax}} \sum_{(s_t,a_t,s_{t+1}) \in \mathcal D^\text{train}} \mathrm{log} \ p_\theta(s_{t+1} \mid s_t, a_t).
\end{align}
$$

To instantiate this method, the authors extend the [`PETS algorithm`](https://arxiv.org/abs/1805.12114), which handles expressive **neural network dynamics models** to attain good sample efficiency as model-based algorithms and **asymptotic performance** as model-free algorithms. PETS uses an ensemble of probabilistic neural network models, each parameterizing a Gaussian distribution of $$ s_{t+1} $$ conditioned on both $$ s_t $$ and $$ a_t $$ . The learned dynamics model is used to plan and execute actions via model predictive control (MPC) with trajectory sampling (TS)(due to probabilistic models). Trajectory sampling predicts plausible state trajectories begins by creating $$ P $$ particles from the current state, $$ s_{t=0}^p =s_0 \forall p $$ . Each particle is then propagated by: $$ s_{t+1}^p \sim \tilde{f}_ {\theta_{b(p,t)}} (s_t^p,a_t) $$ , according to a particular bootstrap $$ b(p,t) \text{in} \lbrace 1, \ldots, B \rbrace $$ , where $$ B $$ is the number of bootstrap models in the ensemble. Unlike a common technique to compute the optimal action sequence (random sampling shooting), PETS used cross-entropy method (CEM), which samples actions from a distribution closer to previous action samples that yielded high reward. Now the overall PETS algorithm can be summarized in Algorithm 1.

<div class="row justify-content-center">
    <div class="col-6">
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

Data is collected by manually piloting the quadcopter along random paths for each of the $$ K $$ suspended payloads. A dataset $$ \mathcal D^\text{train} $$ consists of $$ K $$ separate datasets $$ \mathcal D^\text{train} \doteq \mathcal D^\text{train}_ {1:K} \doteq \lbrace \mathcal D^\text{train}_ 1, \ldots , \mathcal D^\text{train}_ K \rbrace $$ , one per payload task.

<br/>

### Model Training with Known Dynamics Variables

In case we know all the "factors of variation" in the dynamics across tasks at training time, represented explicitly as a "dynamic variable" $$ z_k \in \mathbb R^{d_z} $$ , we can learn a single dynamics model $$ p_\theta $$ across all tasks by using $$ z_k $$ as an auxiliary input to PETS:

$$
\begin{equation}
s_{t+1} \sim p_\theta (s_{t+1} \mid s_t, a_t, z_k ).
\end{equation}
$$

Training is analogous to \eqref{eqn:model-train} , but with an additional conditioning on $$ z_{1:K} \doteq [ z_1, \ldots, z_K ] $$ :

$$
\begin{align}
\theta^* &= \underset{\theta}{\mathrm{argmax}} \ p(\mathcal D^\text{train} \mid z_{1:K}, \theta) \nonumber\\ \label{eqn:known-model-train}
 &= \underset{\theta}{\mathrm{argmax}} \sum_{k=1}^K \sum_{(s_t,a_t,s_{t+1}) \in \mathcal D^\text{train}_ k} \mathrm{log} \ p_\theta(s_{t+1} \mid s_t, a_t, z_k).
\end{align}
$$

<br/>

### Meta-Training with Latent Dynamics Variables

In most cases, we can't know or measure dynamic factors at every training time. Thus, a more general training procedure that ***infers* the dynamics variables $$ z_{1:K} $$ and the model parameters $$ \theta $$ *jointly***. This is begun by placing a prior over $$ z_{1:K} \sim p(z_{1:K}) = \mathcal N(0,I) $$ , and then jointly infer the posterior $$ p(\theta, z_{1:K} \mid \mathcal D^\text{train}_ {1:K} ) $$ .

Unfortunately, inferring $$ p(\theta, z_{1:K} \mid \mathcal D^\text{train}_ {1:K} ) $$ exactly is computationally intractable. Therefore, the authors used an **approximate variational posterior**, which is a Gaussian with diagonal covariance, factored over tasks,

$$
\begin{equation}
q_{\phi_k} (z_k) = \mathcal N(\mu_k, \Sigma_k) \approx p(z_k \mid \mathcal D^\text{train}) \quad \forall k \in [K],
\end{equation}
$$

and parameterized by $$ \phi_k \doteq \lbrace \mu_k, \Sigma_k \rbrace $$ . Unlike the case of known dynamics variables \eqref{eqn:known-model-train}, now we must marginalize out $$ z_{1:K} $$ because it is unknown. Therfore, the model training with latent dynamics variables can be written as:

$$
\begin{align}
\mathrm{log}\ p(\mathcal D^\text{train} \mid \theta) &= \mathrm{log} \int_{z_{1:K}} p(\mathcal D^\text{train} \mid z_{1:K}, \theta) p(z_{1:K})\, dz_{1:K} \nonumber\\
&= \sum_{k=1}^K \mathrm{log}\ \mathbb E_{z_k \sim q_{\phi_k}} p(\mathcal D^\text{train} \mid z_{1:K}, \theta) \cdot p(z_{1:K})/q_{\phi_k}(z_k) \nonumber\\
&\geq \sum_{k=1}^K \mathbb E_{z_k \sim q_{\phi_k}} \sum_{(s_t,a_t,s_{t+1}) \in \mathcal D^\text{train}_ k} \mathrm{log}\ p_\theta(s_{t+1} \mid s_t, a_t, z_k) - \mathrm{KL}(q_{\phi_k}(z_k) \parallel p(z_k)) \quad (\because \text{def. of KL-div & } 0 \leq q \leq 1 \text{, inequality holds)} \nonumber\\ \label{eqn:marginalize-latent}
&\doteq \mathrm{ELBO}(\mathcal D^\text{train} \mid \theta, \phi_{1:K}) .
\end{align}
$$

The propsoed meta-training algorithm then optimizes both $$ \theta $$ and the variational parameters $$ \phi_{1:K} $$ of each task with respect to the evidence lower bound

$$
\begin{equation}
\theta^* \doteq \underset{\theta}{\mathrm{argmax}}\ \underset{\phi_{1:K}}{\mathrm{max}}\ \mathrm{ELBO}(\mathcal D^\text{train} \mid \theta, \phi_{1:K}).
\end{equation}
$$

<br/>

### Test-Time Task Inference

At test time, the robot must infer the unknown dynamics variables $$ z^\text{test} $$ online in order to improve the learned dynamics model $$ p_{\theta*} $$ and the resulting MPC planner. Similarly to meta-training with latent variables, a variational approximation is used for $$ z^\text{test} $$ :

$$
\begin{equation}
q_{\phi^\text{test}}(z^\text{test}) = \mathcal N(\mu^\text{test}, \Sigma^\text{test}) \approx p(z^\text{test} \mid \mathcal D^\text{test}),
\end{equation}
$$

parameterized by $$ \phi^\text{test} \doteq \lbrace \mu^\text{test}, \Sigma^\text{test} \rbrace $$ . Variational inference is used to optimze $$ \phi^\text{text} $$ such that **the approximate distribution $$ q_{\phi^\text{test}}(z^\text{test}) $$ is close to the true distribution $$ p(z^\text{test} \mid \mathcal D^\text{test} $$** , measured by the Kullback-Leibler divergence:

$$
\begin{align}
\phi* &\doteq \underset{\phi}{\mathrm{argmax}}\ -\mathrm{KL}( q_\phi(z^\text{test} \parallel p(z^\text{test} \mid \mathcal D^\text{test}, \theta*)) \nonumber\\
&= \underset{\phi}{\mathrm{argmax}}\ \mathbb E_{z^\text{test} \sim q_\phi} \mathrm{log}\ p(z^\text{test} \mid \mathcal D^\text{test}, \theta*) - \mathrm{log}\ q_\phi(z^\text{test}) \nonumber\\
&= \underset{\phi}{\mathrm{argmax}}\ \mathbb E_{z^\text{test} \sim q_\phi} \mathrm{log}\ p(z^\text{test} \mid \mathcal D^\text{test}, \theta*) - \mathrm{log}\ q_\phi(z^\text{test}) + \mathrm{log}\ p(z^\text{test}) \nonumber\\
&= \underset{\phi}{\mathrm{argmax}}\ \mathbb E_{z^\text{test} \sim q_\phi} \sum_{(s_t,a_t,s_{t+1}) \in \mathcal D^\text{test}} \mathrm{log}\ p_{\theta*}(s_{t+1} \mid s_t, a_t, z^\text{test}) - \mathrm{KL}(q_\phi(z^\text{test} \parallel p(z^\text{test})) \nonumber\\ \label{eqn:test-objective}
&= \underset{\phi}{\mathrm{argmax}}\ \mathrm{ELBO}(\mathcal D^\text{test} \mid \theta*, \phi).
\end{align}
$$

Note the objective \eqref{eqn:test-objective} corresponds to the test-time ELBO of $$ \mathcal D^\text{test} $$ , analogous to training-time ELBO of $$ \mathcal D^\text{train} $$ \eqref{eqn:marginalize-latent}. As equation \eqref{eqn:test-objective} is tractable to optimize, and therefore at test time we perform **gradient descent online** in order to learn $$ \phi^\text{test} $$ and therefore improve the predictions of our learned dynamics model.

The overall training and test time graphical models are summarized in Figure 4.

<div class="row justify-content-center">
    <div class="col-10">
        {% include figure.html path="assets/img/MBMRL-Flight/meta-rl-graphical-model.PNG" title="Probabilistic graphical models of the drone-payload system dynamics" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure from "Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads"
</div>

<br/>

### Method Summary

<div class="row justify-content-center">
    <div class="col-10">
        {% include figure.html path="assets/img/MBMRL-Flight/MBMRL-Flight-algorithm.PNG" title="Algorithm of MBMRL for Quadcopter Payload Transport" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Figure from "Model-Based Meta-Reinforcement Learning for Flight with Suspended Payloads"
</div>

The equation numberings above are based on those in the original paper. In this post, equation (7) means \eqref{eqn:marginalize-latent}, equation (2) means Algorithm 1 from `PETS` and equation (9) means \eqref{eqn:test-objective}.

<br/>

### Method Implementation

**Payload variations**
- 3D printed payloads weighing between 10-15 grams.
- 
