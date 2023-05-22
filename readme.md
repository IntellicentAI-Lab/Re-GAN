## Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration, CVPR 2023

1. ~~give requirement.txt file  and then give instruction like ...For pip users, please type the command pip install -r requirements.txt. For Conda users, you can create a new Conda environment using conda env create -f environment.yml.~~

    <font color="green"> You can see our code does not need to use uncommon libraries or dependencies, everyone who wants to do some interesting work with GANs should know to install Pytorch and Torchvision. And Pytorch is usually not installed through a requirement.txt file, installing it from the official website will be better. </font>

2. ~~Give main image of the paper after abstract~~

    <font color="green"> Done </font>
3. ~~We provide PyTorch implementations...~~

    <font color="green"> Done </font>
4. ~~make one .ipnyb file~~

    <font color="green"> Do not know what is this meaning exactly.. </font>
5. ~~think from a very beginner perspective in this field...it should be like s/he can easily run and get the results~~
       
    <font color=Green>Our code cannot be more clear, this is also what I want to do. The reason why I only put a SNGAN implementation is that SNGAN is one of most simple GAN architecture as it do not have much complex components like those in StyleGAN, so reader can very easily clarify what part is our contribution, and which part it belonging to SNGAN implementation itself.</font>

6. ~~what is the main factor for fixing pruning ratio in D and G?~~

    <font color="green"> Do not know what is this meaning exactly, if you want to find where we prune the model, you may refer to regan.py. </font>
7. ~~How to test the code...?~~

    <font color=green>Please refer to issue 12 </font>
8. ~~How to get 10%, 20% and 50% of data...data_ratio...what value to assign....give example~~ 

    <font color=green>Done, check data_ratio </font>
9. ~~How to compute IS AND FID~~

    <font color=green>Done, add them on the acknowledgment part </font>
10. ~~how to get #real image, Flops and training time~~

    <font color="green"> I think they are trivial to be present in our repo.. #Real image can be calculated by training epoch and batch size, Flops can be calculated by some libraries, training time can be also easily calculated. </font>
11. ~~Code for Figure 7 and 4~~

    <font color="green"> Do not know what is this meaning exactly, provide the code of how we draw the figure? </font>
12. ~~give pretrained model also and then tell how to use it~~

    <font color=green> I am afraid we cannot show the pre-trained model in this repo as I reformat the regan.py to make it more readable and clear. Some old methods disappear and some new added. So the model trained based on the old version of regan.py cannot be loaded by the current regan.py. But if you want the pre-trained model for our research, I can give both model and the old version of regan.py which can be successfully run. </font>
13. ~~give DCGAN experiment...and also give file to draw Figure 2 from the paper~~

    <font color=green>Do not know why we need to show the code of drawing figure, but if you need this code for our own research, I can send it to you.</font>
14. ~~Show table of results having significant results and also show qualitative results like FFHQ for few-shot....~~

    <font color="green"> Done </font>
15. ~~ProGAN and reProGAN for few shot data...still people using ProGAN for lots of purpose...so they give code like now they can explore Re-ProGAN~~

    <font color="green"> ProGAN implementation that we used is quite rough, not very suitable for sharing, but if you need it for our own research, I can send it to you. </font>
16. ~~give code for AutoGAN...all three arch....~~

    <font color=green>I show it on acknowledge part, I think it is not good to show all use models in our repo, it will make our repo messy and complex. </font>
17. ~~also StyleGAN with both Augmentation~~

    <font color="green"> Done, and I only add Diffaug, APA is very easy to add if needed. </font>
18. ~~Acknowledgements...to the githubs used as the base~~

    <font color="green">Put them on the codes </font>




### Abstract

---
Training Generative Adversarial Networks (GANs) on high-fidelity images usually requires a vast number of training images. Recent research on GAN tickets reveals that dense GANs models contain sparse sub-networks or "lottery tickets" that, when trained separately, yield better results under limited data. However, finding GANs tickets requires an expensive process of train-prune-retrain. In this paper, we propose Re-GAN, a data-efficient GANs training that dynamically reconfigures GANs architecture during training to explore different sub-network structures in training time. Our method repeatedly prunes unimportant connections to regularize GANs network and regrows them to reduce the risk of prematurely pruning important connections. Re-GAN stabilizes the GANs models with less data and offers an alternative to the existing GANs tickets and progressive growing methods. We demonstrate that Re-GAN is a generic training methodology which achieves stability on datasets of varying sizes, domains, and resolutions (CIFAR-10, Tiny-ImageNet, and multiple few-shot generation datasets) as well as different GANs architectures (SNGAN, ProGAN, StyleGAN2 and AutoGAN). Re-GAN also improves performance when combined with the recent augmentation approaches. Moreover, Re-GAN requires fewer floating-point operations (FLOPs) and less training time by removing the unimportant connections during GANs training while maintaining comparable or even generating higher-quality samples. When compared to state-of-the-art StyleGAN2, our method outperforms without requiring any additional fine-tuning step.

### Impressive results

---
![Mian Figure](./figures/main_figure.png "Main Figure")
![Table5](./figures/Table5.png "Table5")

### Prerequisites

---
Our code is implemented with Pytorch, we list the libraries and their version used in our experiments, but other versions should also be worked.
1. Linux         (Ubuntu)
2. Python        (3.8.0)
3. Pytorch         (1.13.0+cu116)
4. torchvision         (0.14.0)
5. numpy (1.23.4)

## Getting Started


### Usage

---
As our method can be very easily generalized to other GANs, we only provide SNGAN implementation with CIFAR-10 dataset in this repo.---->Also provide for StyleGAN2 on FFHQ and Few Shot data

#### Hyper-parameters introduction for SNGAN

| Argument       | Type  | Description                                                              |
|----------------|-------|--------------------------------------------------------------------------|
| `epoch`        | int   | Number of total training epochs                                          |
| `batch_size`   | int   | Batch size of per iteration, choose a proper value by yourselves         |
| `sparsity`     | float | Target sparsity k, e.g. sparsity=0.3 means 30% of weights will be pruned |
| `g`            | int   | The update interval                                                      |
| `warmup_epoch` | int   | Warmup training epochs                                                   |
| `data_ratio`   | float | To simulate a training data limited scenario                             |


#### Hyper-parameters introduction for StyleGAN2

| Argument      | Type       | Description                                                                            |
|---------------|------------|----------------------------------------------------------------------------------------|
| `iter`        | int        | Number of total training iterations                                                    |
| `batch_size`  | int        | Batch size of per iteration, choose a proper value by yourselves                       |
| `regan`       | score true | Enable ReGAN training or not                                                           |
| `sparsity`    | float      | Target sparsity k, e.g. sparsity=0.3 means 30% of weights will be pruned               |
| `g`           | int        | The update interval                                                                    |
| `warmup_iter` | int        | Warmup training iterations                                                             |
| `diffaug`     | score true | Enable DiffAug or not                                                                  |
| `eva_iter`    | int        | Evaluation frequency                                                                   |
| `eva_size`    | int        | Evaluation size, for few-shot dataset, we use eva_size=5000                            |
| `dataset`     | str        | which dataset to use, please make sure you place the dataset dictionary in right place |
| `size`        | int        | Size of training image, for few-shot dataset, size=256                                 |
| `ckpt`        | str        | If you want to resume your training, you can use this argument                         |


#### Data Preparation
Pytorch will download the CIFAR-10 dataset automatically if the dataset is not detected, therefore there is no need to prepare CIFAR-10 dataset.

For Tiny-ImageNet dataset, you may get it from https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet.

For FFHQ dataset, you may get it from https://github.com/NVlabs/stylegan.

For Few-shot dataset, you may get it from https://github.com/odegeasslbc/FastGAN-pytorch.
Besides, we provide the lmdb data of Few-shot dataset for your convenience, you may get it from [here](https://drive.google.com/file/d/1bDRlddUxytLSElnrlr9IDydsMVDxm5Bm/view?usp=sharing "fewshow_lmdb").


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



To run a Re-StyleGAN2 model on few-shot dataset, you may follow:
1. Clone this repo to your local environment.
```
git clone https://github.com/IntellicentAI-Lab/Re-GAN.git
```
2. Prepare all the required libraries and datasets. For dataset, please place them like following panda example:
   1. ReGAN/dataset/panda
   2. ReGAN/dataset/pandalmdb


3. Run your model! One example can be:
```
python train.py --size 256 --batch 32 --iter 40000 --dataset panda --eva_iter 2000 --eva_size 100 \
--regan --warmup_iter 10000 --g 5000 --sparsity 0.3 \
--diffaug
```

### Citation
If you use this code for your research, please cite our papers.

```
BiTex
```

### Acknowledgment
We would like to thank the work that helps our paper.

1. FID score: https://github.com/bioinf-jku/TTUR.
2. Inception score: https://github.com/w86763777/pytorch-gan-metrics/blob/master/pytorch_gan_metrics/core.py.
3. DiffAugmentation: https://github.com/VITA-Group/Ultra-Data-Efficient-GAN-Training.


We would like to thank the work that used in our paper but not shown in this repo.

1. AutoGAN: https://github.com/VITA-Group/AutoGAN.
2. APA: https://github.com/endlesssora/deceived.

