---
title: "Backpropagation Derivation"
mathjax: true
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Backpropagation in Neural Networks
*A Mathematical Derivation and Implementation Notes*

## 1. Introduction
Backpropagation is the cornerstone algorithm for training neural networks.
It efficiently computes gradients of the loss function with respect to each weight by applying the chain rule of calculus.
This document derives the math step-by-step and links it to the implementation in this project.

### Notation
$\mathcal{L}$: Loss Function
$a^{(L)}$ be the output<br>
$W \in \mathbb{R}^{n \times m}$
