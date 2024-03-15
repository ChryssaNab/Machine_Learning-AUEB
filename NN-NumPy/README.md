# Feedforward Neural Network for Image Classification

### [**Contents**](#)
1. [Project Description](#descr)
2. [Datasets](#data)
3. [The Model](#model)
4. [Feedforward - Objective Function](#feed)
5. [Backpropagation](#back)
6. [Partial Derivatives of Weights](#dev)
7. [Results](#results)

---

### [**Project Description**](#) <a name="descr"></a>

The aim of this project is to implement the Stohastic Gradient Ascent algorithm from scratch in order to maximize the objective function and estimate the parameters of a feedforward neural network with one hidden layer. The implemented architecture is trained on two distinct datasets: the *MNIST* and *CIFAR-10* datasets, with the goal of classifying their images into the correct categories.

The project was implemented in the context of the course "Machine Learning" taught by Prof. Prodromos Malakasiotis at the Department of Informatics, AUEB. For a comprehensive overview of the methodology and final results, please refer to the [notebook](https://github.com/ChryssaNab/Machine_Learning-AUEB/blob/master/NN-NumPy/jupyter_notebook/Project_1.ipynb).

<a name="cont"></a>

---

### [**Datasets**](#) <a name="data"></a>

 > #### [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

The first dataset being examined for image classification is the MNIST dataset, which contains 28x28 grayscale images. It comprises a total of &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=6*10^5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?6*10^5" title="6*10^5" /></a> &nbsp; training examples and <a href="https://www.codecogs.com/eqnedit.php?latex=10^3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?10^3" title="10^3" /></a> testing examples.

> #### [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) 

The second dataset under consideration is the CIFAR-10. CIFAR-10 comprises 32x32 color images categorized into 10 distinct classes. In total it has &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=5*10^4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?5*10^4" title="5*10^4" /></a> training examples and &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=10^4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?10^4" title="10^4" /></a> &nbsp; testing examples.


#### Project Information:

- N: Number of training data
- D+1: Number of features, plus the one of bias
- K: Number of categories (10 for both datasets)
- M: Number of hidden units

---

### [**The Model**](#) <a name="model"></a>

The architecture of a feedforward neural network with one hidden layer, designed to classify each input example into one of ten categories, is illustrated below:

<p align="center">
<img src="https://github.com/ChryssaNab/Machine_Learning-AUEB/blob/master/NN-NumPy/nn.png" height="400"/>
</p>

---

### [**Feedforward - Objective Function**](#) <a name="feed"></a>

The objective function we aim to optimize for the task of classifying *N* data examples into *K* classes is the *log-likelihood function* plus the regularization term, defined as follows:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=E(W)&space;=&space;\sum_{n=1}^N&space;\sum_{k=1}^K&space;t_{nk}&space;\log&space;s_{nk}&space;-&space;\frac{\lambda}{2}&space;\left[&space;\left(&space;\sum_{k=1}^K&space;||\mathbf{w_k^{(2)}}||^2&space;\right)&space;&plus;&space;\left(&space;\sum_{j=1}^M&space;||\mathbf{w_j^{(1)}}||^2&space;\right)&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(W)&space;=&space;\sum_{n=1}^N&space;\sum_{k=1}^K&space;t_{nk}&space;\log&space;s_{nk}&space;-&space;\frac{\lambda}{2}&space;\left[&space;\left(&space;\sum_{k=1}^K&space;||\mathbf{w_k^{(2)}}||^2&space;\right)&space;&plus;&space;\left(&space;\sum_{j=1}^M&space;||\mathbf{w_j^{(1)}}||^2&space;\right)&space;\right]" title="E(W) = \sum_{n=1}^N \sum_{k=1}^K t_{nk} \log s_{nk} - \frac{\lambda}{2} \left[ \left( \sum_{k=1}^K ||\mathbf{w_k^{(2)}}||^2 \right) + \left( \sum_{j=1}^M ||\mathbf{w_j^{(1)}}||^2 \right) \right]" /></a> ,
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=s_{nk}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{nk}" title="s_{nk}" /></a>  is the softmax function defined as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf&space;s_{nk}&space;=&space;\frac{\mathbf{e^{y_{nk}}}}{\sum_{j=1}^K&space;\mathbf{e^{y_{nk}}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf&space;s_{nk}&space;=&space;\frac{\mathbf{e^{y_{nk}}}}{\sum_{j=1}^K&space;\mathbf{e^{y_{nk}}}}" title="\mathbf s_{nk} = \frac{\mathbf{e^{y_{nk}}}}{\sum_{j=1}^K \mathbf{e^{y_{nk}}}}" /></a>  
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=s_{nk}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{nk}" title="s_{nk}" /></a> is the linear combination of the parameters in the hidden layer defined as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf&space;y_{nk}&space;=&space;\mathbf{z}_n&space;\mathbf{({w}_k^{(2)})}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf&space;y_{nk}&space;=&space;\mathbf{z}_n&space;\mathbf{({w}_k^{(2)})}^T" title="\mathbf y_{nk} = \mathbf{z}_n \mathbf{({w}_k^{(2)})}^T" /></a> ,
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=z_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{n}" title="s_{nk}" /></a> is the output of the chosen activation function in the input layer defined as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf&space;z_{n}{(a)},&space;\hspace{3mm}&space;a&space;=&space;\mathbf&space;x_{n}&space;{(\mathbf{w_{j}}^{(1)})}^T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf&space;z_{n}{(a)},&space;\hspace{3mm}&space;a&space;=&space;\mathbf&space;x_{n}&space;{(\mathbf{w_{j}}^{(1)})}^T" title="\mathbf z_{n}{(a)}, \hspace{3mm} a = \mathbf x_{n} {(\mathbf{w_{j}}^{(1)})}^T" /></a> ,
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(2)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(2)}}" title="\mathbf{w^{(2)}}" /></a> &nbsp; is a &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=K&space;\times&space;(M&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K&space;\times&space;(M&plus;1)" title="K \times (M+1)" /></a> &nbsp; matrix and each line represents the vector &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{w}_k}^{(2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{w}_k}^{(2)}" title="\mathbf{{w}_k}^{(2)}" /></a>, 

and <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf&space;{w^{(1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf&space;{w^{(1)}}" title="\mathbf {w^{(1)}}" /></a> &nbsp; is a &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=(M&plus;1)&space;\times&space;(D&plus;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(M&plus;1)&space;\times&space;(D&plus;1)" title="(M+1) \times (D+1)" /></a> &nbsp; matrix and each line represents the vector &nbsp; <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{{w}_j}^{(1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{{w}_j}^{(1)}" title="\mathbf{{w}_j}^{(1)}" /></a>.

---

### [**Backpropagation**](#) <a name="back"></a>

Stochastic Gradient Ascent is implemented as the optimizer to update the parameters of the neural network:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=w^{(1)}&space;=&space;w^{(1)}&space;&plus;&space;\eta&space;\times&space;\frac{{\vartheta&space;E}}{&space;\vartheta&space;w^{(1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{(1)}&space;=&space;w^{(1)}&space;&plus;&space;\eta&space;\times&space;\frac{{\vartheta&space;E}}{&space;\vartheta&space;w^{(1)}}" title="w^{(1)} = w^{(1)} + \eta \times \frac{{\vartheta E}}{ \vartheta w^{(1)}}" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=w^{(2)}&space;=&space;w^{(2)}&space;&plus;&space;\eta&space;\times&space;\frac{{\vartheta&space;E}}{&space;\vartheta&space;w^{(2)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{(2)}&space;=&space;w^{(2)}&space;&plus;&space;\eta&space;\times&space;\frac{{\vartheta&space;E}}{&space;\vartheta&space;w^{(2)}}" title="w^{(2)} = w^{(2)} + \eta \times \frac{{\vartheta E}}{ \vartheta w^{(2)}}" /></a>
</p>

---

### [**Partial Derivatives of Weights**](#) <a name="dev"></a>

The <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(2)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(2)}}" title="\mathbf{w^{(2)}}" /></a> &nbsp; and <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(1)}}" title="\mathbf{w^{(1)}}" /></a> values arise from the following variables of the objective function: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(2)}}&space;:&space;\hspace{2mm}&space;s_{nk}&space;\Longrightarrow&space;{y}_{nk}&space;\Longrightarrow&space;{w}_k^{(2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(2)}}&space;:&space;\hspace{2mm}&space;s_{nk}&space;\Longrightarrow&space;{y}_{nk}&space;\Longrightarrow&space;{w}_k^{(2)}" title="\mathbf{w^{(2)}} : \hspace{2mm} s_{nk} \Longrightarrow {y}_{nk} \Longrightarrow {w}_k^{(2)}" /></a> and the regularization term & <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(1)}}&space;:&space;\hspace{2mm}&space;s_{nk}&space;\Longrightarrow&space;y_{nk}&space;\Longrightarrow&space;z_{n}&space;\Longrightarrow&space;a&space;\Longrightarrow{w}_j^{(1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(1)}}&space;:&space;\hspace{2mm}&space;s_{nk}&space;\Longrightarrow&space;y_{nk}&space;\Longrightarrow&space;z_{n}&space;\Longrightarrow&space;a&space;\Longrightarrow{w}_j^{(1)}" title="\mathbf{w^{(1)}} : \hspace{2mm} s_{nk} \Longrightarrow y_{nk} \Longrightarrow z_{n} \Longrightarrow a \Longrightarrow{w}_j^{(1)}" /></a> &nbsp; and the correspoding regularization term
<br>

So, the partial derivatives of the values <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(2)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(2)}}" title="\mathbf{w^{(2)}}" /></a> of the objective function are given by the following equation:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E}{\partial{W^{(2)}}}&space;=&space;\frac{\partial&space;E}{\partial&space;S}&space;\hspace{3mm}&space;\frac{\partial&space;S}{\partial&space;Y}&space;\hspace{3mm}&space;\frac{\partial&space;Y}{\partial&space;W^{(2)}}&space;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial{W^{(2)}}}&space;=&space;\frac{\partial&space;E}{\partial&space;S}&space;\hspace{3mm}&space;\frac{\partial&space;S}{\partial&space;Y}&space;\hspace{3mm}&space;\frac{\partial&space;Y}{\partial&space;W^{(2)}}&space;" title="\frac{\partial E}{\partial{W^{(2)}}} = \frac{\partial E}{\partial S} \hspace{3mm} \frac{\partial S}{\partial Y} \hspace{3mm} \frac{\partial Y}{\partial W^{(2)}}" /></a></p>
</p>
<br>

As for <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{w^{(1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{w^{(1)}}" title="\mathbf{w^{(1)}}" /></a>, the partial derivatives of these values are given by the following equation:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;E}{\partial&space;W^{(1)}}&space;=&space;\frac{\partial&space;E}{\partial&space;S}&space;\hspace{3mm}&space;\frac{\partial&space;S}{\partial&space;Y}&space;\hspace{3mm}&space;\frac{\partial&space;Y}{\partial&space;Z}&space;*&space;\frac{\partial&space;Z}{\partial&space;A}&space;\hspace{3mm}&space;\frac{\partial&space;A}{\partial&space;W^{(1)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;E}{\partial&space;W^{(1)}}&space;=&space;\frac{\partial&space;E}{\partial&space;S}&space;\hspace{3mm}&space;\frac{\partial&space;S}{\partial&space;Y}&space;\hspace{3mm}&space;\frac{\partial&space;Y}{\partial&space;Z}&space;*&space;\frac{\partial&space;Z}{\partial&space;A}&space;\hspace{3mm}&space;\frac{\partial&space;A}{\partial&space;W^{(1)}}" title="\frac{\partial E}{\partial W^{(1)}} = \frac{\partial E}{\partial S} \hspace{3mm} \frac{\partial S}{\partial Y} \hspace{3mm} \frac{\partial Y}{\partial Z} * \frac{\partial Z}{\partial A} \hspace{3mm} \frac{\partial A}{\partial W^{(1)}}" /></a>
</p>

<a href="https://www.codecogs.com/eqnedit.php?latex=(*)&space;:&space;element-wise\hspace{2mm}&space;product" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(*)&space;:&space;element-wise\hspace{2mm}&space;product" title="(*) : element-wise\hspace{2mm} product" /></a>

---

### [**Results**](#) <a name="results"></a>

<ul>
<li> In the Mnist dataset, the model typically attains an accuracy exceeding 97% in the majority of cases. </li>
<li> The model exhibits a notably low accuracy on CIFAR-10, peaking at 51%, mainly due to the dataset's complexity. Even when adjusting the learning rate parameter, optimizing accuracy remains challenging. Unlike the MNIST dataset, CIFAR-10 has fewer training examples but significantly more features, especially with color images, explaining the disparity in their scores. To potentially enhance the model's performance, the consideration of incorporating a second hidden layer could be explored. </li>
</ul>
