# Backpropagation in Neural Networks
*A Mathematical Derivation and Implementation Notes*

## 1. Introduction
Backpropagation is the cornerstone algorithm for training neural networks.
It efficiently computes gradients of the loss function with respect to each weight by applying the chain rule of calculus.
This document derives the math step-by-step and links it to the implementation in this project.

### Notation
$\mathscr{L}$: Loss Function
$a^{(L)}$ be the output<br>
$W \in \mathbb{R}^{n \times m}$
