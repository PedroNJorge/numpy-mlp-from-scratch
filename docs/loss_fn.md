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

*An explanation of the loss funtions implemented in the project.*

## Introduction
Each explanation provides the **definition**, **derivative** and **code**.
It will be provided a **Scalar Form and Vectorized Form** for each function, but when
making further calculations, the later will be used because of it's easier usage.
It is recommended to first read 'Backpropagation in Neural Networks', since it provides deep insights
on the reason behind the need to use these type of functions and the calculation of $\dfrac{\partial \mathcal{L}}{\partial a^{(k)}}$

To put it simply, a loss function is a function that produces a scalar as an output,
quantifying "how wrong" the model is on a set of data. As you can imagine,
there are a variety of different losses, each with their own purpose. The categories of loss functions aborded
in this document are **Regression and Classification**.

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


## **Classification**

### **Binary-Cross Entropy (BCE)**
The BCE takes as input $a^{(L)}$ (or $\hat{y}$) and $y$, both in $\mathbb{R}^{b \times 1}$,
where $b$ is the **batch size**. Note that $a^{(L)}$ is the predicted probability and $y$ the true label.

#### Definition

The BCE loss is computed as:
$$
\begin{aligned}
BCE &= -\frac{1}{b}\sum_{i=1}^{b} \left[y_i \log(a^{(L)}_i) + (1 - y_i) \log(1 - a^{(L)}_i)\right]
\end{aligned}
$$

Here is the vectorized form of the BCE:
$$
\begin{aligned}
BCE &= -\frac{1}{b} \left[y_i \odot \log(a^{(L)}_i) + (1 - y_i) \odot \log(1 - a^{(L)}_i)\right]
\end{aligned}
$$

#### Derivative

Let's calculate it's partial derivative w.r.t. $a^{(L)}$:
$$
\begin{align*}
\frac{\partial BCE}{\partial a^{(L)}} &= -\frac{1}{b} \frac{\partial}{\partial a^{(L)}} \left[ y \odot \log(a^{(L)}) + (1 - y) \odot \log(1 - a^{(L)}) \right] \\
                                      &= -\frac{1}{b} \left[ y \odot \frac{1}{a^{(L)}} - (1 - y) \odot \frac {1}{1 - a^{(L)}} \right] \\
                                      &= -\frac{1}{b} \left[ \frac{y - a^{(L)}}{a^{(L)} \odot (1 - a^{(L)})} \right]
\end{align*}
$$

#### Code

