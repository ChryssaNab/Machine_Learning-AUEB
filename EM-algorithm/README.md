# Clustering-based Image Segmentation using Expectation-Maximization algorithm
The aim of this project is the implementation of Expectation-Maximization algorithm from scratch in order to maximize the likelihood function for a mixture of Gaussians of the format below:


<p align="center">
<img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\sum_{k=1}^{K}&space;\pi_{k}\prod_{d=1}^{D}&space;\frac{1}{\sqrt{2&space;\pi&space;\sigma_k^2}}\&space;e^{-\frac{1}{2\sigma_k^2}&space;(x_d&space;-&space;\mu_{kd})^2&space;}&space;\hspace{0.5cm}"/>
</p>


Then, the implemented algorithm is used for the segmentation of the RGB image below:
<p align="center">
<img src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/images/im.jpg" height="314" width="250"/>
</p>


## Results
The algorithm is tested for different number of clusters (K=2,4,8,16,32,64,128):

<p align="center">
    <img title="Clusters: 2" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/2_Categories.jpg" height="314" width="250"/>
        <img title="Clusters: 4" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/4_Categories.jpg" height="314" width="250"/>
        <img title="Clusters: 8" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/8_Categories.jpg" height="314" width="250"/>
    <img title="Clusters: 16" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/16_Categories.jpg" height="314" width="250"/>
    <img title="Clusters: 32" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/32_Categories.jpg" height="314" width="250"/>
    <img title="Clusters: 64" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/64_Categories.jpg" height="314" width="250"/>
    <img title="Clusters: 128" src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/output/128_Categories.jpg" height="314" width="250"/>
   </p>

The reconstruction error for each of the above clusters is defined as:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\textit{error}&space;=&space;\frac{1}{N}&space;\sum_{n=1}^{N}&space;{\lVert{x_{true,n}&space;-&space;x_{r,n}}\rVert}^2" title="\textit{error} = \frac{1}{N} \sum_{n=1}^{N} {\lVert{x_{true,n} - x_{r,n}}\rVert}^2" />
</p>

where <img src="https://latex.codecogs.com/gif.latex?x_{r,n}" /> is the pixel value predicted by the mixture, i.e. the mean value <img src="https://latex.codecogs.com/gif.latex?\mu_k" /> for which the corresponding posteriori probability <img src="https://latex.codecogs.com/gif.latex?\gamma(z_k)" /> is the maximum, while <img src="https://latex.codecogs.com/gif.latex?x_{true,n}" /> is the actual pixel value.



### Cost Curve 
<img src="https://github.com/ChryssaNab/aueb-machine_learning/blob/master/EM-algorithm/report/Project_2_files/cost_curve.png"/>
