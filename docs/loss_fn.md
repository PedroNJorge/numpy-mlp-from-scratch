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

To put it simply, a loss function is a function that produces a scalar as an output,
quantifying "how wrong" the model is on a set of data. More formally:

$$
\begin{aligned}
Batch Loss \mathcal{L}: \mathbb{R}^{m \times K} \times \mathbb{R}^{m \times K} \rightarrow \mathbb{R} \\
Per-Example Loss \mathcal{L}_i: \mathbb{R}^{n} \rightarrow \mathbb{R} \\
\mathcal{L}(\mathbf{\hat{Y}}, \mathbf{Y}) &= \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}_i(\mathbf{\hat{y}}_i, \mathbf{y}_i)
\end{aligned}
$$

Where:
* $m$: Batch size (number of samples in the current batch).
* $K$: Number of classes (output dimension for classification).
* $\mathbf{Y}$: Target matrix of shape $(m, K)$.
* $\mathbf{\hat{Y}}$: Prediction matrix of shape $(m, K)$. For Neural Networks, $\mathbf{\hat{Y}} = \mathbf{a^{(L)}}$.
* $\mathbf{y}_i$, $\mathbf{\hat{y}}_i$: The target and prediction vectors for the $i$-th sample (rows from $\mathbf{Y}$ and $\mathbf{\hat{Y}}$).

We can interpret this as a general definition followed by every loss function with $\mathcal{L}$ being the **batch loss**, i.e., the average loss over all examples,
and $\mathcal{L}_i$ the **per-example loss**.

As you can imagine, there are a variety of different losses, each with their own purpose.
The categories of loss functions covered in this document are **Regression and Classification**.

## **Regression**
These loss functions are used...

### **Mean Squared Error (MSE)**


## **Classification**

### **Binary-Cross Entropy (BCE)**
The BCE loss function is used when... ($\mathbf{\hat{Y}}$ is a predicted probability).

#### **Definition**
The BCE loss is computed as:
$$
\begin{aligned}
BCE(\mathbf{\hat{Y}}, \mathbf{Y}) &= -\frac{1}{m}\sum_{i=1}^{m} \left[y_i \log(a^{(L)}_i) + (1 - y_i) \log(1 - a^{(L)}_i)\right]
\end{aligned}
$$

Here is the vectorized form of the BCE:
$$
\begin{aligned}
BCE &= -\frac{1}{b} \left[y_i \odot \log(a^{(L)}_i) + (1 - y_i) \odot \log(1 - a^{(L)}_i)\right]
\end{aligned}
$$

#### **Gradient of Loss**

Let's calculate it's partial derivative w.r.t. $a^{(L)}$:
$$
\begin{align*}
\frac{\partial BCE}{\partial a^{(L)}} &= -\frac{1}{b} \frac{\partial}{\partial a^{(L)}} \left[ y \odot \log(a^{(L)}) + (1 - y) \odot \log(1 - a^{(L)}) \right] \\
                                      &= -\frac{1}{b} \left[ y \odot \frac{1}{a^{(L)}} - (1 - y) \odot \frac {1}{1 - a^{(L)}} \right] \\
                                      &= -\frac{1}{b} \left[ \frac{y - a^{(L)}}{a^{(L)} \odot (1 - a^{(L)})} \right]
\end{align*}
$$

#### Code

