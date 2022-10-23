---
layout: page
title: Review on "Meta-Learning in Neural Networks-A Survey" (1)
description: simple summary on meta-learning survey paper
img: assets/img/meta-example-from-BAIR-edu.PNG
importance: 1
category: \papers review
---

# Key Contents
- [Definitions of meta-learning](#2-1-formalizing-meta-learning)
- Position of meta-learning with respect to related fields
- Previous taxonomies and new taxonomy of meta-learning
- Promising applications and successes of meta-learning
- Challenges and promising areas for future research
    
# 1 Introduction
Contemporary ML models are typically **trained from scratch** for **a specific task** using a fixed learning algorithm **designed by hand**. However, successes of theirs exclude many applications where **data is intrinsically rare or expensive, or compute resources are unavailable**. Meta-learning in NN aims to provide the next step of integrating joint feature, model, and *algorithm* learning; that is, it targets **replacing prior hand-designed learners with leard learning algorithms**. One perspectives on meta-learning is that **learning-to-learn occurs when a learner’s performance at solving tasks drawn from a given task family improves with respect to the number of tasks seen**(by Thrun).

In this paper, they focus specially on where algorithm learning is achieved by *end-to-end* learning of an *explicitly defined objective function*.

# 2 Background
## 2.1 Formalizing Meta-Learning
### Conventional Machine Learning

Training a predictive model $\hat{y}=f_\theta(x)$ parameterized by $\theta$, given a training dataset $\mathcal{D}=\{(x_1,y_1),...,(x_N,y_N)\}$, by solving:

$$
\theta^*=\underset{\theta}{\mathrm{argmin}}\ \mathcal L(\mathcal D; \theta,\omega),
$$

where $\mathcal L$ is loss function, $\omega$ specifies assumptions about ‘**how to learn**’, such as the choice of optimizer for $\theta$ or function class for $f$.

The conventional assumption is that **this optimization is performed *from scratch* for every problem $\mathcal D$; and that $\omega$ is pre-specified**. However, the specification of $\omega$ can drastically affect performance measure(ex: accuracy, data efficiency). → Meta-learning seeks to improve these measures by l**earning the learning algorithm itself**.

Generalization is then measured by evaluating $\mathcal L$ on test points with known labels.

### **Meta-Learning: Task-Distribution View**

Learning how to learn, the performance of $\omega$ over a task distribution $p(\mathcal T)$ where task specifies a dataset and **task specific** loss function $\mathcal{T=\{D,L\}}$, becomes

$$
\underset{\omega}{\mathrm{min}}\ \mathbb E_{\mathcal T\sim p(\mathcal T)} \mathcal {L(D};\omega).
$$

Task specific loss is defined as $\mathcal{L(D};\omega)=\mathcal{L(D^{val}};\theta^*(\mathcal D^{train},\omega), \omega)$; where $\mathcal{D=(D^{train},D^{val})}$ and $\theta^*$ is the parameters of the model **trained using the ‘how to learn’ *meta-knowledge* $\omega$ on dataset $\mathcal D^{train}$.**

The *meta-training* step of ‘**learning how to learn**’, given M sources tasks $\mathcal{ D_{source}=\{(D^{train}_{source}, D^{val}_{source})^{(i)}\}^M_{i=1}} \sim p(\mathcal T)$, can be written as:

$$
\omega^*=\underset{\omega}{\mathrm{argmin}}\sum^M_{i=1}\mathcal {L(D}^{(i)}_{source};\omega).
$$

In the *meta-testing* stage we use the **learned meta-knowledge $\omega^*$** to train the base model on each previously unseen target task $i$ from $\mathcal {D_{target}=\{(D^{train}_{target}, D^{test}_{target})^{(i)}\}^Q_{i=1}}$:

$$
\theta^{*\ (i)}=\underset{\theta}{\mathrm{argmin}}\ \mathcal{L(D^{train\ (i)}_{target}};\theta,\omega^*).
$$

Meta-knowledge $\omega^*$ could be:

- An estimate of the initial parameters
- Entire learning model
- Optimization strategy

We can evaluate **the accuracy of our meta-learner** by the performance of $\theta^{*\ (i)}$ on the test split of each target task $\mathcal D^{test\ (i)}_{target}$.

### Meta-Learning: Bilevel Optimization View

Bilevel optimization refers to a **hierarchical optimization problem**, where one optimization contains another optimization as a constraint. (The way to solve the meta-training step in Eq. (3), multiple task scenario, is commonly done by casting the meta-training step as a bilevel optimization problem.) Meta-training can be formalised as follows:

$$
\omega*=\underset{\omega}{\mathrm{argmin}}\sum^M_{i=1}\mathcal{L^{meta}(D^{val\ (i)}_{source}};\theta^{*\ (i)}(\omega),\omega)
$$

$$
\mathrm{s.t.} \qquad \theta^{*\ (i)}(\omega)=\underset{\theta}{\mathrm{argmin}}\ \mathcal{L^{task}(D^{train\ (i)}_{source}};\theta,\omega),
$$

where $\mathcal L^{meta}$ and $\mathcal L^{task}$ refer to the **outer and inner objectives** respectively. They can be different functions. Note the leader-follower asymmetry between the outer and inner levels: the inner optimization Eq.(6) is conditional on the learning strategy $\omega$ defined by the outer level, but it cannot change $\omega$ during its training.

### Meta-Learning: Feed-Forward Model View

Feed forward model view refers to meta-learning approaches that **synthesize models in a feed-forward manner**, rather than via explicit iterative optimization as Eqs. 5-6 above. While they vary in their degree of complexity it can be instructive to understand this family of approaches by instantiating the abstract objective in Eq. (2) to define a toy example for meta-training linear regression [43].

$$
\underset{\omega}{\mathrm{min}}\ \mathbb E_{\mathcal T\sim p\mathcal{(T)(D^{tr},D^{val})\in T}}\sum_{(x,y)\in D^{val}}\left[(\mathbf x^T\mathbf g_\omega(\mathcal D^{tr})-y)^2\right].
$$

The train set $\mathcal D^{tr}$ is embedded into a vector $\mathbf g_\omega$ which defines the linear regression weights to predict examples $\mathbf x$ from the validation set. By training the function $\mathbf g_\omega$ through Eq.(7), $\mathbf g_\omega$ should provide a good solution for novel meta-test tasks $\mathcal{T^{test} \sim p(T)}$. Methods in this family vary in the complexity of the predictive model $\mathbf g$ used, and how the support set is embedded (e.g., by pooling, CNN, or RNN).

## 2.2 Historical Context of Meta-Learning: Skipped

## 2.3 Related Fields
### Transfer Learning (TL)

TL : Using **past experience from a source task** to improve learning (speed, data-efficiency, accuracy) on a target task.

$\subset$ meta-learning; With use of meta-objective, it deals with a much wider range of meta-representations.

### Domain Adaptation (DA) and Domain Generalization (DG)

DEF Domain-shift: the situation where source and target problems share the same objective, but the **input distribution of the target task is shifted** with respect to the source task.

DA : a variant of TL, **adapting the source-trained model** using sparse or unlabeled data from the target.

DG : training a **source model to be robust to such domain-shift** without further adaptation.

$\subset$ meta-learning; TL, vanilla DA, DG don’t use meta-objective

### Continual Learning (CL)

The ability to learn on a **sequence of tasks** drawn from a potentially non-stationary distribution, and in particular seek to do so **while accelerating learning new tasks and without forgetting old tasks**.

$\subset$ meta-learning ( $\because$ despite usage of task distribution and intention to accelerate learning of a target task, this meta objective is not solved for explicitly)

### Multi-Task Learning (MTL)

To **jointly learn several related tasks**, conventional MTL is a single-level optimization without a meta-objective.

$\subset$ meta-learning; the goal of MTL is **to solve a fixed number of known tasks**, whereas the point of meta-learning is often **to solve unseen future tasks**.

### Hyperparameter Optimization (HO)

$\subset$ meta-learning; hyperparameters describe ‘how to learn’

### Hierarchical Bayesian Models (HBM) : TBD

HBM involve bayesian learning of parameters $\theta$ under a prior $p(\theta|\omega)$ where $\omega \sim p(\omega)$. They feature strongly as models for grouped data $\mathcal{D=\{D_i}|i=1,2,...,M\}$, where each group i has its own $\theta_i$. → Full model : $\left [\prod^M_{i=1}p(\mathcal D_i|\theta_i)p(\theta_i|\omega)\right]p(\omega)$.

Then use some form of Bayesian marginalisation to compute the posterior over $\omega: P(\omega|\mathcal D)\sim p(\omega)\prod^M_{i=1}\int d\theta_ip(\mathcal D_i|\theta_i)p(\theta_i|\omega)$.

HBM provide a valuable viewpoint for meta-learning, by providing a modeling rather than an algorithmic framework for understanding the meta-learning process.

### AutoML

A broad unbrella for approaches aiming to **automate parts of the machine learning process that are typically manual**. (Ex: data preparation, algorithm selection, hyperparameter tuning, and architecture search)

$\supset$ meta-learning; AutoML often makes use of numerous huristics outside the scope of meta-learning.
