# AIDEme - JMLR 2020

This version of our project was used in when running all experiments detailed in our JMLR 2020 submission. 
Below, you can find a short description of the project, installation instructions, and some useful links.   

# Introduction
AIDEme is a scalable interactive data exploration system for efficiently learning a user interest pattern over a large dataset. 
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


# Instructions

## Dependencies
AIDEme has the following dependencies:
  * Python (>= 3.7)
  * NumPy
  * SciPy
  * Scikit-learn

After downloading this project, simply open a terminal and run:

`python setup.py install`

This should take care of installing all the above dependencies. 

# Reproducing our results
If you wish to reproduce our results for JMLR 2020, two things are needed: obtaining the data, and learn how to use our system. 

In our evaluation, two datasets were used:

- [Sloan Digital Sky Survey](http://www.sdss3.org/dr8/) (SDSS, 190 million points): The dataset contains the "PhotoObjAl" table with 510 attributes and 190 million sky observations. 
We used a 1% sample (1.9 million points, 4.9GB) to create a dataset for running active learning algorithms. 

- Car dataset (5622 points): This small dataset was used in [1] to conduct a user study, which generated 18 queries representing the true user interests.
Being a proprietary dataset, we cannot make it public. 


In order learn how to use our system, you can refer to an example jupyter notebook called [example.ipynb](./example.ipynb). There
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
