# Superpixels
In this work, we demonstrate that infusing domain knowledge in the form of superpixels-based radial graph improves the predictive performance of CNN-based classifiers. We would love to know about any issues found on this repository. Please create an issue for any queries, or contact us at sharmabhee@gmail.com.

Pre-print: Coming Soon

## Abstract

<p align="center">
  <img src="./misc/images/paper_front_page.svg" alt="Paper Abstract"/>
</p>

## Updates

- [18 May 2021]: Repository is made public.

## Usage

### Quick Colab Guide

[Colab Notebook](https://colab.research.google.com/drive/1zpgNTe1B_RqPzqVPwxwyia8DOPQfnBv2?usp=sharing)

### Setting Up

Clone the repository.

```sh
git clone https://github.com/abheesht17/super-pixels.git
cd super-pixels
```

Install the requirements.

```sh
make requirements
```
At times, especially on Colab, the above command fails. It is preferable that you run the following:

```sh
pip3 install --upgrade -r requirements.txt --find-links https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html --find-links https://download.pytorch.org/whl/torch_stable.html
```

Note: Find the CUDA version and install accordingly. 

### Datasets

All the datasets are available on [Google Drive](https://drive.google.com/drive/u/0/folders/1CQfPgNtXmRzUqYrz5eFZDwHgW1crbje-).

#### List of available datasets:
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [COVID X-Ray Detection](https://github.com/tawsifur/COVID-19-Chest-X-ray-Detection)
- [LFW Face Recognition](http://vis-www.cs.umass.edu/lfw/)
- [SOCOFing Fingerprint Identification](https://www.kaggle.com/ruizgara/socofing)


### Grid Search

For all models and datasets, the configs are present in the `configs/custom_trainer` directory. Choose a config according to the model and the dataset you want to run. For example, to do a Grid Search on the CNN+GAT model using the LFW dataset, you have to run the following command:

```sh
python train.py --config_dir ./configs/custom_trainer/graph_image/hybrid/cnn_gat_lfw --grid_search --validation
```

Currently, we guarantee that the "cnn" and "cnn_gat" models are in working condition. The rest have to be checked.

This will save the logs and the final model at the path specified in the `logs` folder.

Note: Performing Grid Search is not necessary. The default hyperparameters specified in the configs are the ones obtained after performing Grid Search; one can directly run the training and inference command given in the next subsection.  

### Training and Inference

```sh
python train.py --config_dir ./configs/custom_trainer/graph_image/hybrid/cnn_gat_lfw
```
## Tasks

- [x] Add Usage.
- [x] Add Citation(s).
- [x] Update `README.md`.
- [x] Add Directory Structure.

## Results and Analysis

### Results on Standard Datasets

|  Dataset  |   CNN  | CNN+GNN |
|:---------:|:------:|:-------:|
|   MNIST   | 99.30% |  99.21% |
|   FMNIST  | 91.65% |  91.50% |
|  CIFAR-10 | 77.80% |  76.81% |
| CIFAR-100 | 42.88% |  46.79% |

### Results on Domain-Specific Datasets

|  Dataset |   CNN  | CNN+GNN |
|:--------:|:------:|:-------:|
|   COVID  | 89.09% |  91.01% |
|    LFW   | 60.83% |  66.12% |
| SOCOFing | 65.68% |  93.58% |

## Citation

You can cite our work as:

```sh
@unpublished{chhablani2021superpixels ,
author = {Gunjan Chhablani and Abheesht Sharma and Harshit Pandey and Tirtharaj Dash},
title = {Superpixel-based Domain-Knowledge Infusion in Computer Vision},
note = {Under Review at ESANN 2021},
year = {2021}
}
```
OR

```sh
G. Chhablani, A. Sharma, H. Pandey, T. Dash, "Superpixel-based Domain-Knowledge Infusion in Computer Vision", Under Review at ESANN 2021, 2021.
```

If you use any part of our code in your work, please use the following citation:

```sh
@misc{sharma2021superpixelsgithub,
  author = {Abheesht Sharma and Gunjan Chhablani and Harshit Pandey and Tirtharaj Dash},
  title = {abheesht17/super-pixels},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abheesht17/super-pixels}},
}
```

## Directory Structure

```sh
.
├── configs
│   ├── custom_trainer
│   │   ├── graph
│   │   │   ├── gat_cifar10
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gat_cifar100
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gat_covid
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gat_fmnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gat_lfw
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gat_mnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gat_socofing
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_cifar10
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_cifar100
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_covid
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_fmnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_lfw
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_mnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── gcn_socofing
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_cifar10
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_cifar100
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_covid
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_fmnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_lfw
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_mnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── monet_socofing
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── multigat_cifar10
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── multigat_cifar100
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── multigat_covid
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── multigat_fmnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── multigat_lfw
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── multigat_mnist
│   │   │   │   ├── dataset.yaml
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   └── multigat_socofing
│   │   │       ├── dataset.yaml
│   │   │       ├── model.yaml
│   │   │       └── train.yaml
│   │   ├── graph_image
│   │   │   ├── hybrid
│   │   │   │   ├── cnn_gat_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_mnist_hybrid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gat_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_gcn_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_monet_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── cnn_multigat_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gat_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_gcn_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_monet_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── prevgg_multigat_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gat_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_gcn_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_monet_socofing
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_multigat_cifar10
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_multigat_cifar100
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_multigat_covid
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_multigat_fmnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_multigat_lfw
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   ├── vgg_multigat_mnist
│   │   │   │   │   ├── dataset.yaml
│   │   │   │   │   ├── model.yaml
│   │   │   │   │   └── train.yaml
│   │   │   │   └── vgg_multigat_socofing
│   │   │   │       ├── dataset.yaml
│   │   │   │       ├── model.yaml
│   │   │   │       └── train.yaml
│   │   │   └── projection
│   │   │       ├── cnn_gat_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gat_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gat_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gat_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gat_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gat_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gat_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_gcn_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_monet_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── cnn_multigat_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gat_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_gcn_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_monet_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── prevgg_multigat_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gat_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_gcn_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_monet_socofing
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_multigat_cifar10
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_multigat_cifar100
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_multigat_covid
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_multigat_fmnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_multigat_lfw
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       ├── vgg_multigat_mnist
│   │   │       │   ├── dataset.yaml
│   │   │       │   ├── model.yaml
│   │   │       │   └── train.yaml
│   │   │       └── vgg_multigat_socofing
│   │   │           ├── dataset.yaml
│   │   │           ├── model.yaml
│   │   │           └── train.yaml
│   │   └── image
│   │       ├── cnn_cifar10
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── cnn_cifar100
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── cnn_covid
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── cnn_fmnist
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── cnn_lfw
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── cnn_mnist
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── cnn_socofing
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_cifar10
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_cifar100
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_covid
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_fmnist
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_lfw
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_mnist
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── prevgg_socofing
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── vgg_cifar10
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── vgg_cifar100
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── vgg_covid
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── vgg_fmnist
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── vgg_lfw
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       ├── vgg_mnist
│   │       │   ├── dataset.yaml
│   │       │   ├── model.yaml
│   │       │   └── train.yaml
│   │       └── vgg_socofing
│   │           ├── dataset.yaml
│   │           ├── model.yaml
│   │           └── train.yaml
│   └── templates
│       ├── dataset
│       │   ├── paths
│       │   │   ├── cifar100.txt
│       │   │   ├── cifar10.txt
│       │   │   ├── covid.txt
│       │   │   ├── fmnist.txt
│       │   │   ├── lfw.txt
│       │   │   ├── mnist.txt
│       │   │   └── socofing.txt
│       │   └── transforms
│       │       ├── graph
│       │       │   ├── cifar10
│       │       │   │   ├── monet.txt
│       │       │   │   └── regular.txt
│       │       │   ├── cifar100
│       │       │   │   ├── monet.txt
│       │       │   │   └── regular.txt
│       │       │   ├── covid
│       │       │   │   ├── monet.txt
│       │       │   │   └── regular.txt
│       │       │   ├── fmnist
│       │       │   │   ├── monet.txt
│       │       │   │   └── regular.txt
│       │       │   ├── lfw
│       │       │   │   └── regular.txt
│       │       │   ├── mnist
│       │       │   │   ├── monet.txt
│       │       │   │   └── regular.txt
│       │       │   └── socofing
│       │       │       ├── monet.txt
│       │       │       └── regular.txt
│       │       └── image
│       │           ├── cifar10
│       │           │   ├── prevgg.txt
│       │           │   ├── regular.txt
│       │           │   └── vgg.txt
│       │           ├── cifar100
│       │           │   ├── prevgg.txt
│       │           │   ├── regular.txt
│       │           │   └── vgg.txt
│       │           ├── covid
│       │           │   ├── prevgg.txt
│       │           │   ├── regular.txt
│       │           │   └── vgg.txt
│       │           ├── fmnist
│       │           │   ├── prevgg.txt
│       │           │   ├── regular.txt
│       │           │   └── vgg.txt
│       │           ├── lfw
│       │           │   └── regular.txt
│       │           ├── mnist
│       │           │   ├── prevgg.txt
│       │           │   ├── regular.txt
│       │           │   └── vgg.txt
│       │           └── socofing
│       │               ├── prevgg.txt
│       │               ├── regular.txt
│       │               └── vgg.txt
│       ├── model
│       │   ├── graph
│       │   │   ├── gat
│       │   │   │   ├── cifar100.txt
│       │   │   │   ├── cifar10.txt
│       │   │   │   ├── covid.txt
│       │   │   │   ├── lfw.txt
│       │   │   │   ├── regular.txt
│       │   │   │   └── socofing.txt
│       │   │   ├── gcn
│       │   │   │   ├── cifar100.txt
│       │   │   │   ├── cifar10.txt
│       │   │   │   ├── lfw.txt
│       │   │   │   └── regular.txt
│       │   │   ├── monet
│       │   │   │   ├── cifar100.txt
│       │   │   │   ├── cifar10.txt
│       │   │   │   ├── lfw.txt
│       │   │   │   └── regular.txt
│       │   │   └── multigat
│       │   │       ├── cifar100.txt
│       │   │       ├── cifar10.txt
│       │   │       ├── lfw.txt
│       │   │       └── regular.txt
│       │   └── image
│       │       ├── cnn
│       │       │   ├── cifar100.txt
│       │       │   ├── cifar10.txt
│       │       │   ├── covid.txt
│       │       │   ├── lfw.txt
│       │       │   ├── regular.txt
│       │       │   └── socofing.txt
│       │       ├── prevgg
│       │       │   ├── cifar100.txt
│       │       │   ├── cifar10.txt
│       │       │   ├── covid.txt
│       │       │   ├── lfw.txt
│       │       │   ├── regular.txt
│       │       │   └── socofing.txt
│       │       └── vgg
│       │           ├── cifar100.txt
│       │           ├── cifar10.txt
│       │           ├── covid.txt
│       │           ├── lfw.txt
│       │           ├── regular.txt
│       │           └── socofing.txt
│       └── train
│           ├── graph
│           │   ├── covid.yaml
│           │   ├── lfw.yaml
│           │   └── regular.yaml
│           ├── graph_image
│           │   ├── hybrid
│           │   │   ├── covid.yaml
│           │   │   ├── lfw.yaml
│           │   │   └── regular.yaml
│           │   └── projection
│           │       ├── covid.yaml
│           │       └── regular.yaml
│           └── image
│               ├── covid.yaml
│               ├── lfw.yaml
│               └── regular.yaml
├── CONTRIBUTING.md
├── custom_train.py
├── ex.md
├── generate_configs.py
├── hf_train.py
├── isort.cfg
├── LICENSE
├── Makefile
├── mean_calculator.py
├── misc
│   └── images
│       └── paper_front_page.svg
├── ovh_requirements.sh
├── README.md
├── requirements.txt
└── src
    ├── datasets
    │   ├── cifar_img_slic.py
    │   ├── cifar.py
    │   ├── cifar_slic.py
    │   ├── covid_img_slic.py
    │   ├── covid.py
    │   ├── covid_slic.py
    │   ├── hf_image_classification.py
    │   ├── __init__.py
    │   ├── lfw_img_slic.py
    │   ├── lfw.py
    │   ├── lfw_slic.py
    │   ├── mnist_img_slic.py
    │   ├── mnist.py
    │   ├── mnist_slic.py
    │   ├── socofing_img_slic.py
    │   ├── socofing.py
    │   ├── socofing_slic.py
    │   ├── tg_mnist_slic.py
    │   └── tv_mnist.py
    ├── __init__.py
    ├── models
    │   ├── cnn.py
    │   ├── gat.py
    │   ├── gcn.py
    │   ├── hybrid.py
    │   ├── __init__.py
    │   ├── monet.py
    │   ├── multigat.py
    │   ├── projection.py
    │   └── vgg.py
    ├── modules
    │   ├── activations.py
    │   ├── __init__.py
    │   ├── losses.py
    │   ├── metrics.py
    │   ├── optimizers.py
    │   ├── schedulers.py
    │   └── transforms.py
    ├── trainers
    │   ├── base_trainer.py
    │   ├── hybrid_trainer.py
    │   └── __init__.py
    └── utils
        ├── configuration.py
        ├── __init__.py
        ├── logger.py
        ├── mapper.py
        ├── misc.py
        └── viz.py
```