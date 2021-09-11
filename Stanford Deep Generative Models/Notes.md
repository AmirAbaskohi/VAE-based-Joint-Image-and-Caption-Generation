# Deep Generative Models

*These notes are based on `Stanford` `CS 236: Deep Generative Models` Course by `Stefano Ermon` and `Aditya Grover`. You can access course materials from here: <a href:"deepgenerativemodels.github.io">link</a>*

## Introduction

Challenge: Understand complex, unstructured inputs. This can be in any area:
* Computer Vision
* Natural Language Procssing
* Robotics
* Computational Speech

![image info](./images/Richard.png)

`Richard Feynman`: “What I cannot create, I do not understand”

## Statistical Generative Models
A statistical generative model is a probability distribution p(x)
* Data
* Prior knowledge

It is generative because sampling from p(x) generates new images.

### Disciminative vs. generative
`Discriminative`: classify bedroom vs. dining room. The input image X is always given. Goal: a good decision boundary, via conditional distribution.

`Generative`: generate X. The input X is not given. Requires a model of the joint distribution.

Joint and conditional are related via Bayes Rule.

Class conditional generative models are also possible. It’s often useful to condition on rich side information Y. A discriminative model is a very simple conditional generative model of Y.

## Learning a generative model
We are given a training set of examples, e.g., images of dogs. We want to learn a probability distribution p(x ) over images x such that:
* Generation
* Density estimation
* Unsupervised

But how should we represent p(x)?

We can use basic discrete distributions such as:
* Bernoulli
* Categorical

### Structure through independence
If X1, ..., Xn are independent, then:
```
p(x1, ..., xn) = p(x1) p(x2) p(x3) ... p(xn)
```

How many parameters to specify joint distribution? 2^(n)-1. Because p(x1) needs one parameters and we have 2^(n) states.

Using chain rule:
```
p(x1, ..., xn) = p(x1) p(x2 | x1) p(x3 | x1, x2) ... p(xn | x1, ..., xn-1)
```

So we need: 1 + 2 + ... + 2^(n-1) = 2^(n) - 1 parameters. It is still exponential.

Now suppose `Xi+1 ⊥ X1,...,Xi −1 |Xi` , then:
```
p(x1, ..., xn) = p(x1) p(x2 | x1) p(x3 | x2) ... p(xn | xn-1)
```

Now how many parameters? `2n-1`.

Bayesian Networks: assume an ordering and a set of conditional
independencies to get compact representation.

```
p(x1,...,xn ) = ∏p(xi |xAi )
```

### Beysian networks

A Bayesian network is specified by a directed acyclic graph G = (V ,E ) with:
* One node i ∈ V for each random variable Xi
* One conditional probability distribution (CPD) per node

Graph G = (V ,E ) is called the structure of the Bayesian Network
```
p(x1,...xn ) = ∏p(xi |xPa(i))
```

![image info](./images/BN.png)

Bayesian network structure implies conditional independencies.

## Neural models
```
p(x1,x2,x3,x4) ≈ p(x1)p(x2|x1)pNeural(x3|x1,x2)pNeural(x4|x1,x2,x3)
```

Assumes specific functional form for the conditionals. A sufficiently deep neural net can approximate any function.

In neural models for classification we care about `p(y | x)` and assume that:
```
p(y=1 | x;α) = f(x,α)
```

In logistic regression(linear form):
```
p_logit(Y = 1 |x; α) = σ(z (α,x))

where σ(z ) = 1/(1 + e−z) and 
z(α,x) = α0 + ∑αi*xi .
```

In non-linear form:
```
p_Neural(Y = 1 |x; α,A,b) = σ(α0 + ∑αi*hi)

where h(A,b,x) = f (Ax + b) be a non-linear transformation of the inputs (features)
```


## Autoregressive Models

We can pick an ordering, i.e., order variables (pixels) from top-left (X1) to bottom-right (Xn=784). Using chain rule:
```
p(v1,··· ,v784) = p(v1)p(v2 |v1)p(v3 |v1,v2) ···p(vn |v1,··· ,vn−1)
```
Some conditionals are too complex to be stored in tabular form. So we assume:
```
p(v1,··· ,v784) = pCPT(v1; α1)plogit(v2 |v1; α2)plogit(v3 |v1,v2; α3) ··· plogit(vn |v1,··· ,vn−1; αn )

pCPT(V1 = 1; α1) = α1, p(V1 = 0) = 1 −α1

plogit(V2 = 1 |v1; α2) = σ(α20 + α21v1)

plogit(V3 = 1 |v1,v2; α3) = σ(α30 + α31v1 + α32v2)
.
.
.
```

This is a modeling assumption. We are using a logistic regression to predict next pixel based on the previous ones. Called autoregressive.

The conditional variables `Vi |V1,··· ,Vi −1` are Bernoulli with parameters
```
vi = p(Vi = 1|v1,··· ,vi −1; αi ) = p(Vi = 1|v<i ; αi ) = σ(αi0 + ∑αij*vj )
```
This is called `FVSBN`.

To improve model: use one layer neural network instead of logistic regression:
```
vi = p(vi |v1,··· ,vi −1; Ai ,ci ,αi ,bi) = σ(αi hi + bi )

hi = σ(Ai v<i + ci )
```

This is called `NADE`.

### Autoregressive vs. autoencoders
On the surface, FVSBN and NADE look similar to an autoencoder. Can we get a generative model from an autoencoder?

We need to make sure it corresponds to a valid Bayesian Network (DAG structure), i.e., we need an ordering.

we can use a single neural network (with n outputs) to produce all the parameters. In contrast, NADE requires n passes. Much more efficient on modern hardware.

### `MADE : Masked Autoregressive Density Estimator`

* Parameter sharing: use a single multi-layer neural network
* Challenge: need to make sure it’s autoregressive (DAG structure)
* Solution: use masks to disallow certain paths (Germain et al., 2015).

![image](https://user-images.githubusercontent.com/50926437/132952561-cf8778ad-ea4e-4c6b-8635-cf1d607f33eb.png)

### RNN
![image](https://user-images.githubusercontent.com/50926437/132952613-9ac24226-208f-45d9-a9b3-a67636d80f37.png)

Pros:
* Can be applied to sequences of arbitrary length.
* Very general: For every computable function, there exists a finite
RNN that can compute it

Cons:
* Still requires an ordering
* Sequential likelihood evaluation (very slow for training)
* Sequential generation (unavoidable in an autoregressive model)
* Can be difficult to train (vanishing/exploding gradients)

### CNN
Use convolutional architecture to predict next pixel given context (a neighborhood of pixels).

Challenge: Has to be autoregressive. Masked convolutions preserve raster scan
order. Additional masking for colors order.

## Maximum likelihood learning
Lets assume that the domain is governed by some underlying distribution Pdata. We are given a dataset D of m samples from Pdata. The standard assumption is that the data instances are independent and
identically distributed (IID). We are also given a family of models M, and our task is to learn some
“good” model ˆM∈M (i.e., in this family) that defines a distribution p ˆM. The goal of learning is to return a model ˆM that precisely captures the distribution Pdata from which our data was sampled. This is in general not achievable because of:
* Limited data
* Computational reasons

What is "best"? This depends on what we want to do:
* Density Estimation
* Specific prediction tasks
* Structure or knowledge discovery

We want to learn the full distribution so that later we can answer any probabilistic inference query. In this setting we can view the learning problem as density estimation. We want to construct Pθ as ”close” as possible to Pdata.
 
 
 How should we measure distance between distributions? Kullback-Leibler divergence (KL-divergence)
 ```
D (p‖q) = ∑p(x)*(log (p(x)/q(x)))
 ```
 *Note that KL-duvergence is asymmetric
 
 Now we have:
 ```
D(Pdata||Pθ) = Ex∼Pdata[log(Pdata(x)/Pθ(x))]
              = Ex∼Pdata [log Pdata(x)] −Ex∼Pdata [log Pθ(x)]
 ```
 
 So minimizing KL divergence is equivalent to maximizing the expected log-likelihood.
 ```
 maxPθ [(1/|D|) * ∑log Pθ(x)]
 Pθ(x(1),··· ,x(m)) = ∏Pθ(x)
 ```
 
 ## Latent variable models
Lots of variability in images x due to gender, eye color, hair color,pose, etc. However, unless images are annotated, these factors of variation are not explicitly available (latent)

Explicitly model these factors using latent variables z.

![image](https://user-images.githubusercontent.com/50926437/132957570-588384c3-df4d-4c2c-b25f-1cadcda443c6.png)

Latent variables z correspond to high level features.
* If z chosen properly, p(x|z) could be much simpler than p(x)
* If we had trained this model, then we could identify features via p(z |x), e.g., p(EyeColor = Blue|x)

### Deep latent variable model
* z ∼N(0,1)
* p(x |z) = N (μθ(z),Σθ(z)) where μθ,Σθ are neural network
* Hope that after training, z will correspond to meaningful latent factors of variation (features). Unsupervised representation learning
* As before, features can be computed via p(z |x)

### Mixture of guassians
* z ∼Categorical(1,··· ,K )
* p(x |z = k ) = N (μk ,Σk )

![image](https://user-images.githubusercontent.com/50926437/132957807-6fc1ec83-addc-4641-ac58-847a4c53ad2b.png)

### Micture models
Combine simple models into a more complex and expressive one.

```
p(x) = ∑p(x,z) = ∑p(z)p(x |z) = ∑p(z = k ) N(x; μk ,Σk )
```

### Variational Autoencoder
![image](https://user-images.githubusercontent.com/50926437/132957865-0709ed46-9829-464f-94eb-6de83ce2582c.png)

A mixture of an infinite number of Gaussians:
* z ∼N(0,I )
* p(x |z) = N (μθ(z),Σθ(z)) where μθ,Σθare neural networks
    * μθ(z) = σ(Az + c ) = (σ(a1z + c1),σ(a2z + c2)) = (μ1(z),μ2(z))
    * Σθ(z) = diag (exp(σ(B z + d )))
    * θ = (A,B ,c ,d )
* Even though p(x |z) is simple, the marginal p(x) is very complex/flexible


## Extra: What are VAEs(Variational Autoencoders)

Among these deep generative models, two major families stand out and deserve a special attention: Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).

![image](https://user-images.githubusercontent.com/50926437/132959268-c3b04c19-51f8-40c9-9b16-bec3cd3dbfb4.png)

In a nutshell, a VAE is an autoencoder whose encodings distribution is regularised during the training in order to ensure that its latent space has good properties allowing us to generate some new data. Moreover, the term “variational” comes from the close relation there is between the regularisation and the variational inference method in statistics.

### Dimensionality reduction, PCA and autoencoders
In machine learning, dimensionality reduction is the process of reducing the number of features that describe some data. This reduction is done either by selection (only some existing features are conserved) or by extraction (a reduced number of new features are created based on the old features) and can be useful in many situations that require low dimensional data (data visualisation, data storage, heavy computation…). 

First, let’s call encoder the process that produce the “new features” representation from the “old features” representation (by selection or by extraction) and decoder the reverse process. Dimensionality reduction can then be interpreted as data compression where the encoder compress the data (from the initial space to the encoded space, also called latent space) whereas the decoder decompress them. Of course, depending on the initial data distribution, the latent space dimension and the encoder definition, this compression can be lossy, meaning that a part of the information is lost during the encoding process and cannot be recovered when decoding.

![image](https://user-images.githubusercontent.com/50926437/132959364-6ed92c2e-e6de-4c5e-9c83-d356a6e65c9b.png)

For a given set of possible encoders and decoders, we are looking for the pair that keeps the maximum of information when encoding and, so, has the minimum of reconstruction error when decoding.

PCA is looking for the best linear subspace of the initial space (described by an orthogonal basis of new features) such that the error of approximating the data by their projections on this subspace is as small as possible.

![image](https://user-images.githubusercontent.com/50926437/132959428-3641bee5-36e6-4b72-87f8-a86911bdd5dd.png)


### Autoencoders
Let’s now discuss autoencoders and see how we can use neural networks for dimensionality reduction. The general idea of autoencoders is pretty simple and consists in setting an encoder and a decoder as neural networks and to learn the best encoding-decoding scheme using an iterative optimisation process.

Intuitively, the overall autoencoder architecture (encoder+decoder) creates a bottleneck for data that ensures only the main structured part of the information can go through and be reconstructed.

![image](https://user-images.githubusercontent.com/50926437/132959478-40d2f5a8-9eb9-4802-ba4e-17e14807cde8.png)


### Variational Autoencoders

#### Limitations of autoencoders for content generation

At this point, a natural question that comes in mind is “what is the link between autoencoders and content generation?”. Indeed, once the autoencoder has been trained, we have both an encoder and a decoder but still no real way to produce any new content. At first sight, we could be tempted to think that, if the latent space is regular enough (well “organized” by the encoder during the training process), we could take a point randomly from that latent space and decode it to get a new content. The decoder would then act more or less like the generator of a Generative Adversarial Network.

![image](https://user-images.githubusercontent.com/50926437/132959594-15cc6e5a-3d43-42f7-befa-41b0e974e8cb.png)

However, the regularity of the latent space for autoencoders is a difficult point that depends on the distribution of the data in the initial space, the dimension of the latent space and the architecture of the encoder. So, it is pretty difficult (if not impossible) to ensure, a priori, that the encoder will organize the latent space in a smart way compatible with the generative process we just described.

![image](https://user-images.githubusercontent.com/50926437/132959682-5ccc7a20-ec31-4dfd-a327-2b828856d8bb.png)

#### Definition of variational autoencoders
In order to be able to use the decoder of our autoencoder for generative purpose, we have to be sure that the latent space is regular enough. One possible solution to obtain such regularity is to introduce explicit regularisation during the training process.  A variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.

Just as a standard autoencoder, a variational autoencoder is an architecture composed of both an encoder and a decoder and that is trained to minimise the reconstruction error between the encoded-decoded data and the initial data. However, in order to introduce some regularisation of the latent space, we proceed to a slight modification of the encoding-decoding process: instead of encoding an input as a single point, we encode it as a distribution over the latent space.

![image](https://user-images.githubusercontent.com/50926437/132959956-4a4499bf-1650-4964-811c-e0c9394d8f88.png)

The reason why an input is encoded as a distribution with some variance instead of a single point is that it makes possible to express very naturally the latent space regularisation: the distributions returned by the encoder are enforced to be close to a standard normal distribution. 

![image](https://user-images.githubusercontent.com/50926437/132960004-d5f78f69-91e9-44b8-a3c9-37f23099f21d.png)

With this regularisation term, we prevent the model to encode data far apart in the latent space and encourage as much as possible returned distributions to “overlap”, satisfying this way the expected continuity and completeness conditions. 
