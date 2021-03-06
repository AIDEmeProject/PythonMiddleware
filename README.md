# AIDEme Middleware
This project contains the middleware code for the [AIDEme](https://github.com/AIDEmeProject/AIDEme) data exploration system software. More precisely, this
codebase implements the main Active Learning algorithms and optimizations used by AIDEme, including Version Space techniques
and the Dual-Space factorized model.  

Below, you can find a short description of the project, installation instructions, and some useful links.

This software in under the **BSD 3-Clause license**.

# Introduction
[AIDEme](https://github.com/AIDEmeProject/AIDEme) is a scalable interactive data exploration system for efficiently learning a user interest pattern over a large dataset. 
The system is cast in a principled active learning (AL) framework, which iteratively presents strategically selected records for user labeling, 
thereby building an increasingly-more-accurate model of the user interest. However, a challenge in building such a system 
is that existing active learning techniques experience slow convergence when learning the user interest on large datasets. 
To overcome the problem, AIDEme explores properties of the user labeling process and the class distribution of observed 
data to design new active learning algorithms, which come with provable results on model accuracy, convergence, and approximation, 
and have evaluation results showing much improved convergence over existing AL methods while maintaining interactive speed.

## Features
With AIDEme, you can:
  * Run our custom interactive data exploration over large datasets, efficiently retrieving all data points of interest
  * Compare the performance of different Active Learning algorithms over targeted labeled data
  * Easily run your custom Active Learning algorithms or implement new data exploration routines


# Dependencies and Installation

The AIDEme Middleware has the following dependencies:
  * Python (>= 3.7)
  * NumPy
  * SciPy
  * Scikit-learn

After downloading this project, simply open a terminal and run:

```
python setup.py install
python setup.py build_ext --inplace
```
This should take care of installing all the above dependencies. 

## Using our system
If you are interested in running our AL algorithms or comparing them with our own, you can refer to the example jupyter notebook called [example.ipynb](./example.ipynb). There
you can find detailed examples of how to run our active learning algorithms over any dataset (including factorization).

# Websites
We also invite you to check our [website](https://www.lix.polytechnique.fr/aideme), for a more complete description of this project.


# References
[1] 
Enhui Huang, Luciano Palma, Laurent Cetinsoy, Yanlei Diao, Anna Liu.
[AIDEme: An active learning based system for interactive exploration of large datasets](https://nips.cc/Conferences/2019/Schedule?showEvent=15427).
NeurIPS - Thirty-third Conference on Neural Information Processing Systems, Dec 2019, Vancouver, Canada

[2] 
Luciano Di Palma, Yanlei Diao, Anna Liu. 
[A Factorized Version Space Algorithm for "Human-In-the-Loop" Data Exploration](https://hal.inria.fr/hal-02274497v2/document). 
ICDM - 19th IEEE International Conference in Data Mining, Nov 2019, Beijing, China.

[3] 
Enhui Huang, Liping Peng, Luciano Di Palma, Ahmed Abdelkafi, Anna Liu, Yanlei Diao.
[Optimization for Active Learning-based Interactive Database Exploration](http://www.vldb.org/pvldb/vol12/p71-huang.pdf). 
PVLDB - 12th Proceedings of Very Large Database Systems, Sep 2018, Los Angeles, USA.
