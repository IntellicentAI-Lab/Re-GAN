## Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration

### Abstract

---
Training Generative Adversarial Networks (GANs) on high-fidelity images usually requires a vast number of training images. Recent research on GAN tickets reveals that dense GANs models contain sparse sub-networks or "lottery tickets" that, when trained separately, yield better results under limited data. However, finding GANs tickets requires an expensive process of train-prune-retrain. In this paper, we propose Re-GAN, a data-efficient GANs training that dynamically reconfigures GANs architecture during training to explore different sub-network structures in training time. Our method repeatedly prunes unimportant connections to regularize GANs network and regrows them to reduce the risk of prematurely pruning important connections. Re-GAN stabilizes the GANs models with less data and offers an alternative to the existing GANs tickets and progressive growing methods. We demonstrate that Re-GAN is a generic training methodology which achieves stability on datasets of varying sizes, domains, and resolutions (CIFAR-10, Tiny-ImageNet, and multiple few-shot generation datasets) as well as different GANs architectures (SNGAN, ProGAN, StyleGAN2 and AutoGAN). Re-GAN also improves performance when combined with the recent augmentation approaches. Moreover, Re-GAN requires fewer floating-point operations (FLOPs) and less training time by removing the unimportant connections during GANs training while maintaining comparable or even generating higher-quality samples. When compared to state-of-the-art StyleGAN2, our method outperforms without requiring any additional fine-tuning step.

### Dependencies

---
1. Linux         (Ubuntu)
2. Python        (3.8.0)
3. Pytorch         (1.13.0+cu116)
4. torchvision         (0.14.0)
5. numpy (1.23.4)

### Usage

---
As our method can be very easily generalized to other GANs, we only provide SNGAN implementation with CIFAR-10 dataset in this repo.

#### Hyper-parameters introduction

| Argument       | Type  | Description                                                              |
|----------------|-------|--------------------------------------------------------------------------|
| `epoch`        | int   | Number of total training epochs                                          |
| `batch_size`   | int   | Batch size of per iteration, choose a proper value by yourselves         |
| `sparsity`     | float | Target sparsity k, e.g. sparsity=0.3 means 30% of weights will be pruned |
| `g`            | int   | The update interval                                                      |
| `warmup_epoch` | int   | Warmup training epochs                                                   |
| `data_ratio`   | float | To simulate a training data limited scenario                             |


#### Data Preparation
Pytorch will download the CIFAR-10 dataset automatically if the dataset is not detected, therefore there is no need to prepare CIFAR-10 dataset.

For Tiny-ImageNet dataset, you may derive it from https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet.

For FFHQ dataset, you may derive it from https://github.com/NVlabs/stylegan.


#### Example

To run a Re-SNGAN model, you may follow:
1. Clone this repo to your local environment.
```
git clone https://github.com/IntellicentAI-Lab/Re-GAN.git
```
2. Run your model! One example can be:
```
python main.py --epoch 1000 --warmup_epoch 200 --g 100 --sparse 0.3
```

### Citation
If you find Re-GAN helps your work, please consider citing our paper by following
```
BiTex
```
