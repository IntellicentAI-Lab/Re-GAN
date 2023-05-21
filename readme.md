## Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration, CVPR 2023

1. give requirement.txt file  and then give instruction like ...For pip users, please type the command pip install -r requirements.txt.
For Conda users, you can create a new Conda environment using conda env create -f environment.yml.
2. GIve main image of the paper after abstract
3. We provide PyTorch implementations...
4. make one .ipnyb file
5. think from a very beginner perspective in this field...it should be like s/he can easily run and get the results
6. what is the main factor for fixing pruning ratio in D and G?
7. How to test the code...?
8. How to get 10%, 20% and 50% of data...any code or done manually?
9. How to compute IS AND FID
10. how to get #real image, Flops and training time
11. Code for Figure 7 and 4
12. give pretrained model also and then tell how to use it
13. give DCGAN experiment...and also give file to draw Figure 2 from the paper
14. Show table of results having significant results and also show qualitative results like FFHQ for few-shot....also StyleGAN with both Augmentation
15. ProGAN and reProGAN for few shot data...still people using ProGAN for lots of purpose...so they give code like now they can explore Re-ProGAN
16. AutoGAN...all three arch....
17. Acknowledgements...to the githubs used as the base




### Abstract

---
Training Generative Adversarial Networks (GANs) on high-fidelity images usually requires a vast number of training images. Recent research on GAN tickets reveals that dense GANs models contain sparse sub-networks or "lottery tickets" that, when trained separately, yield better results under limited data. However, finding GANs tickets requires an expensive process of train-prune-retrain. In this paper, we propose Re-GAN, a data-efficient GANs training that dynamically reconfigures GANs architecture during training to explore different sub-network structures in training time. Our method repeatedly prunes unimportant connections to regularize GANs network and regrows them to reduce the risk of prematurely pruning important connections. Re-GAN stabilizes the GANs models with less data and offers an alternative to the existing GANs tickets and progressive growing methods. We demonstrate that Re-GAN is a generic training methodology which achieves stability on datasets of varying sizes, domains, and resolutions (CIFAR-10, Tiny-ImageNet, and multiple few-shot generation datasets) as well as different GANs architectures (SNGAN, ProGAN, StyleGAN2 and AutoGAN). Re-GAN also improves performance when combined with the recent augmentation approaches. Moreover, Re-GAN requires fewer floating-point operations (FLOPs) and less training time by removing the unimportant connections during GANs training while maintaining comparable or even generating higher-quality samples. When compared to state-of-the-art StyleGAN2, our method outperforms without requiring any additional fine-tuning step.

### Dependencies --->Prerequisites

---
1. Linux         (Ubuntu)
2. Python        (3.8.0)
3. Pytorch         (1.13.0+cu116)
4. torchvision         (0.14.0)
5. numpy (1.23.4)

## Getting Started


### Usage

---
As our method can be very easily generalized to other GANs, we only provide SNGAN implementation with CIFAR-10 dataset in this repo.---->Also provide for StyleGAN2 on FFHQ and Few Shot data

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

For Tiny-ImageNet dataset, you may get it from https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet.

For FFHQ dataset, you may get it from https://github.com/NVlabs/stylegan.


#### Example

To run a Re-SNGAN model, you may follow:
1. Clone this repo to your local environment.
```
git clone https://github.com/IntellicentAI-Lab/Re-GAN.git
```
2. Prepare all the required libraries and datasets.


3. Run your model! One example can be:
```
python main.py --epoch 1000 --warmup_epoch 200 --g 100 --sparse 0.3 --data_ratio 0.1
```
### Citation
If you use this code for your research, please cite our papers.

```
BiTex
```
