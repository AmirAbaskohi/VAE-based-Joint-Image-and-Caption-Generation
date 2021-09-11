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

![image info](./images/Richard.png)
