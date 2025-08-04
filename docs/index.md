---
title: "Backpropagation in Neural Networks"
mathjax: true
---
<!-- Load Mermaid.js -->
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({ startOnLoad: true });</script>

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>


*A Mathematical Derivation and Implementation Notes*

## **1. Introduction**
Backpropagation is the cornerstone algorithm for training neural networks.
It efficiently computes gradients of the loss function with respect to each weight by applying the chain rule of calculus.
This document derives the math step-by-step and links it to the implementation in this project.

### **Notation**
* $\mathcal{L}$: Loss function (e.g., MSE, Cross-Entropy).
* $a^{(k)}$: Activation at layer $k$ (e.g., $a^{(k)} = ReLU(z^{(k)})$).
* $y$: True label, $a^{(L)}$: Predicted output.
* $W^{(k)}$: Weight matrix for layer $k$.
* $b^{(k)}$: Bias vector for layer $k$.
* $z^{(k)}$: Pre-activation at layer $k$ ($z^{(k)} = W^{(k)}a^{(k-1)} + b^{(k)}$).
* $\sigma$: Activation function (e.g., ReLU, Sigmoid).
* $\odot$: Element-wise (Hadamard) product.

We will be using standard theoretical dimensions for the tensors:
* $a^{(k)} \in \mathbb{R}^{n \times 1}$
* $W^{(k)} \in \mathbb{R}^{m \times n}$
* $b^{(k)} \in \mathbb{R}^{m \times 1}$

## **2. Derivation of Backpropagation**
### **Forward Pass**
```mermaid
graph LR
  X:::invisible -->|"a(k-1)"| A["Linear Layer: W(k), b(k)"]
  A --> |"z(k)"| B["Activation σ"]
  B --> |"a(k)"| Y:::invisible

  classDef invisible fill:none,stroke:none,color:white;
```
<div style="display: flex; justify-content: center;">
  <figure style="margin: 0;">
    <img src="assets/forward.svg" alt="Neural Network Layer">
    <figcaption style="text-align: center;">
  <figcaption>Forward Pass through Neural Network Layer</figcaption>
</figcaption>
  </figure>
</div>
<br> 

$$
\begin{aligned}
z^{(k)} &= W^{(k)}a^{(k-1)} + b^{(k)} \\
a^{(k)} &= \sigma(z^{(k)})
\end{aligned}
$$

### **Backward Pass**
```mermaid
graph RL
  X:::invisible -->|"∂L/∂a(k)"| B["Activation σ"]
  B -->|"δ(k)"| A["Linear Layer"]
  A -->|"∂L/∂a(k-1)"| Y:::invisible

  classDef invisible fill:none,stroke:none,color:white;
```

Let
