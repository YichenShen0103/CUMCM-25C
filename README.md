# CUMCM-25C

## Run our code

**Google Colab**

Problem 1 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YichenShen0103/CUMCM-25C/blob/main/problem1.ipynb)

Problem 2 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YichenShen0103/CUMCM-25C/blob/main/problem2.ipynb)

Problem 3 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YichenShen0103/CUMCM-25C/blob/main/problem3.ipynb)

Problem 4 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YichenShen0103/CUMCM-25C/blob/main/problem4.ipynb)


**Local**

Create a new conda environment (or using your other favourite virtual env manager):

```shell
$ conda create -n CUMCM-25C python=3.12
$ conda activate CUMCN-25C
```

Download all packages needed in our code using pip (or your other package manager), be careful choosing your pytorch version:

```shell
$ pip install -r requirements.txt
```

If you need to display the decision tree as a picture, please download `graphviz`. In Ubuntu Linux platform, just do:

```shell
$ sudo apt-get install graphviz
```

After finish downloading, you can use this venv as the kernel of your jupyter notebook to run our code locally.