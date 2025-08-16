---
title: "Loss Functions"
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

*An explanation of the loss funtions implemented in the project.*<br>

## Introduction
Each explanation provides the **definition**, **derivative** and **code**.<br>
It is recommended to first read 'Backpropagation in Neural Networks', since it provides deep insights
on the reason behind the need to use these type of functions and the calculation of $\dfrac{\partial \mathcal{L}}{\partial a^{(k)}}$

## **Regression**
These loss functions are used...

### **Mean Squared Error (MSE)**
The MSE loss function can be defined as the following:
$$
\begin{aligned}
Without Batches &= W^{(k)}a^{(k-1)} + b^{(k)} \\
With Batch &= \sigma(z^{(k)}) \\
Vectorized &= 
\end{aligned}
$$

