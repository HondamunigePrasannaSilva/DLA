#  Laboratory 1: Convolutional neural networks with residual connections

This laboratory use relatively simple architectures to deep down on architecture with and without residual.

## Exercise 1:  A baseline MLP

Implement a *simple* Multilayer Perceptron to classify the 10 digits of MNIST. The purpose of this exercise is to demostrate that deeper architecture **can** work better and converge much more slowley than shallow networks.  

![](img/es1.png)


## Exercise 2: Convolutional nerual network with residual connection
Implement a Convolutional nerual network with residual connection to classify the class of CIFAR10.

A simple test to demostrate that deeper networks without residual connection not always work better and **even deeper** ones *with* residual connections.

![](img/cnnres.png)

In **red** residual CNN with residual, in **green** CNN without residual.

## Exercise 3: Explain the prediction of a CNN
In this exercise will use [*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.): to view where the network focus to classify.

> B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning Deep Features for Discriminative Localization. CVPR'16 (arXiv:1512.04150, 2015).

- This first attempt:  use the parameters of the global average pool, to understand where the network focus globaly.
 
![](img/gifs/prova_4_g.gif)

- The second attempt: use the parameter of the predicted class to find where the network focus to predict it.

![](img/gifs/prova_4_cc.gif)

on the left is the true class and on the right is the class predited from the class at every epoch.

It's interesting noting that the information gather after global average pooling show where the network is looking in general as it sums the spatial information,  before MLP is applied. Meanwhile in the first it showes only the part that the network looks to predict a certain class. And this could be useful for anomaly detection to search where the network focus in general and not only on the predicted class. If the netowrk has '*perfect*' accuracy the two *attention map* should be the same.   


