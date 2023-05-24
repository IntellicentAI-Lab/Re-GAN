from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from model import ResDiscriminator32, ResGenerator32
from regan import Regan_training
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main():
    # Create the dataset
    dataset = dset.CIFAR10(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]), download=True, train=True)

    # Make sub-training dataset
    subset = torch.utils.data.Subset(dataset, np.arange(int(len(dataset) * args.data_ratio)))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.workers)

    netD = ResDiscriminator32().to(device)
    netG = Regan_training(ResGenerator32(args.noise_size).to(device), sparsity=args.sparsity)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), args.lr, (0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), args.lr, (0, 0.9))

    print("Starting Training Loop...")

    flag_g = 1

    for epoch in range(1, args.epoch + 1):

        if args.regan:
            # Warm-up phase, do not enable the ReGAN training
            if epoch < args.warmup_epoch + 1:
                print('current is warmup training')
                netG.train_on_sparse = False

            # Warm-up phase finished, get into Sparse training phase
            elif epoch > args.warmup_epoch and flag_g < args.g + 1:
                print('epoch %d, current is sparse training' % epoch)
                # turn training mode to sparse, update mask
                netG.turn_training_mode(mode='sparse')
                # make sure the learning rate of sparse phase is the original one
                if flag_g == 1:
                    print('turn learning rate to normal')
                    for params in optimizerG.param_groups:
                        params['lr'] = args.lr
                flag_g = flag_g + 1

            # Sparse training phase finished, get into dense training phase
            elif epoch > args.warmup_epoch and flag_g < 2 * args.g + 1:
                print('epoch %d, current is dense training' % epoch)
                # turn training mode to dense
                netG.turn_training_mode(mode='dense')
                # make sure the learning rate of Dense phase is 10 times smaller than the original one
                if flag_g == args.g + 1:
                    print('turn learning rate to 10 times smaller')
                    for params in optimizerG.param_groups:
                        params['lr'] = args.lr * 0.1
                flag_g = flag_g + 1

                # When curren Sparse-Dense pair training finished, get into next pair training
                if flag_g == 2 * args.g + 1:
                    print('clean flag')
                    flag_g = 1

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            output = netD(real_cpu).view(-1)
            errD_real = torch.mean(nn.ReLU(inplace=True)(1.0 - output))
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.noise_size, device=device)
            fake = netG(noise)

            output = netD(fake.detach()).view(-1)
            errD_fake = torch.mean(nn.ReLU(inplace=True)(1 + output))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            if i % args.n_critic == 0:
                netG.zero_grad()
                noise = torch.randn(b_size, args.noise_size, device=device)
                fake = netG(noise)
                output = netD(fake).view(-1)
                errG = -torch.mean(output)
                errG.backward()
                D_G_z2 = output.mean().item()

                # Eliminate weights and their gradients
                if args.regan and netG.train_on_sparse:
                    netG.apply_masks()

                optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%4d/%4d][%3d/%3d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


if __name__ == '__main__':
    model_name = 'SNGAN'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=20)
    argparser.add_argument('--batch_size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=2e-4)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--image_size', type=int, default=32)
    argparser.add_argument('--noise_size', type=int, default=128)
    argparser.add_argument('--dataroot', type=str, default='../dataset')
    argparser.add_argument('--clip_value', type=float, default=0.01)
    argparser.add_argument('--n_critic', type=int, default=5)
    argparser.add_argument('--sparsity', type=float, default=0.3)
    argparser.add_argument('--g', type=int, default=5)
    argparser.add_argument('--warmup_epoch', type=int, default=100)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--regan', action="store_true")
    args = argparser.parse_args()

    if not os.path.exists(args.dataroot):
        os.makedirs(args.dataroot)

    device = "cuda"

    main()
