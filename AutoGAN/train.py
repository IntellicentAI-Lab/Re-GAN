# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
# Modified by Jiahao Xu (jiahxu@polyu.edu.hk)

from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from regan import Regan_training

import cfg
import datasets
import models  # noqa
from functions import copy_params, LinearLrDecay, load_params, train, validate
# from utils.fid_score import check_or_download_inception, create_inception_graph
# from utils.inception_score import _init_inception
from utils.utils import create_logger, save_checkpoint, set_log_dir


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # # set tf env
    # _init_inception()
    # inception_path = check_or_download_inception(None)
    # create_inception_graph(inception_path)

    # import network
    gen_net = Regan_training(eval("models." + args.gen_model + ".Generator")(args=args).cuda(), sparsity=args.sparsity)
    dis_net = eval("models." + args.dis_model + ".Discriminator")(args=args).cuda()

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            if args.init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == "orth":
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == "xavier_uniform":
                nn.init.xavier_uniform(m.weight.data, 1.0)
            else:
                raise NotImplementedError(
                    "{} unknown inital type".format(args.init_type)
                )
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set optimizer
    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()),
        args.g_lr,
        (args.beta1, args.beta2),
    )
    dis_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dis_net.parameters()),
        args.d_lr,
        (args.beta1, args.beta2),
    )
    gen_scheduler = LinearLrDecay(
        gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic
    )
    dis_scheduler = LinearLrDecay(
        dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic
    )

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # fid stat
    if args.dataset.lower() == "cifar10":
        fid_stat = "fid_stat/cifar10.test.npz"
    elif args.dataset.lower() == "stl10":
        fid_stat = "fid_stat/stl10_train_unlabeled_fid_stats_48.npz"
    else:
        raise NotImplementedError(f"no fid stat for {args.dataset.lower()}")
    # assert os.path.exists(fid_stat)

    # epoch number for dis_net
    # args.max_epoch = args.max_epoch * args.n_critic
    # if args.max_iter:
    #     args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4

    assert args.exp_name
    args.path_helper = set_log_dir("logs", args.exp_name)
    logger = create_logger(args.path_helper["log_path"])

    logger.info(args)
    writer_dict = {
        "writer": SummaryWriter(args.path_helper["log_path"]),
        "train_global_steps": start_epoch * len(train_loader),
        "valid_global_steps": start_epoch // args.val_freq,
    }

    # train loop
    flag_g = 1
    print(start_epoch)
    pbar = range(1, 1 + args.max_epoch)
    print(len(pbar))
    pbar = tqdm(pbar, initial=start_epoch, desc="total progress", dynamic_ncols=True, smoothing=0.01)
    for cur_epoch in pbar:
        epoch = cur_epoch + start_epoch

        if epoch > args.max_epoch:
            print("Done!")

            break

        if args.regan:
            # Warm-up phase, do not enable the ReGAN training
            if epoch < args.warmup_epoch + 1:
                print('current is warmup training')
                gen_net.train_on_sparse = False

            # Warm-up phase finished, get into Sparse training phase
            elif epoch > args.warmup_epoch and flag_g < args.g + 1:
                print('epoch %d, current is sparse training' % epoch)
                # turn training mode to sparse, update mask
                gen_net.turn_training_mode(mode='sparse')
                # make sure the learning rate of sparse phase is the original one
                if flag_g == 1:
                    print('turn learning rate to normal')
                    for params in gen_optimizer.param_groups:
                        params['lr'] = args.g_lr
                flag_g = flag_g + 1

            # Sparse training phase finished, get into dense training phase
            elif epoch > args.warmup_epoch and flag_g < 2 * args.g + 1:
                print('epoch %d, current is dense training' % epoch)
                # turn training mode to dense
                gen_net.turn_training_mode(mode='dense')
                # make sure the learning rate of Dense phase is 10 times smaller than the original one
                if flag_g == args.g + 1:
                    print('turn learning rate to 10 times smaller')
                    for params in gen_optimizer.param_groups:
                        params['lr'] = args.g_lr * 0.1
                flag_g = flag_g + 1

                # When curren Sparse-Dense pair training finished, get into next pair training
                if flag_g == 2 * args.g + 1:
                    print('clean flag')
                    flag_g = 1

        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None

        train(
            args,
            gen_net,
            dis_net,
            gen_optimizer,
            dis_optimizer,
            gen_avg_param,
            train_loader,
            epoch,
            writer_dict,
            lr_schedulers,
        )


        if epoch % args.val_freq == 0 or epoch == args.max_epoch:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, is_std, fid_score = validate(
                args, fixed_z, fid_stat, gen_net, writer_dict
            )
            logger.info('IS: %.4f (%.4f) ||  FID: %.4f || @ epoch %d.' % (inception_score, is_std, fid_score, epoch))
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
            if epoch == int(args.max_epoch):
                logger.info(
                    f"Total time cost: {time_count}s.")

        else:
            is_best = False

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param)
        if epoch % args.val_freq == 0 or epoch == args.max_epoch:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "gen_model": args.gen_model,
                    "dis_model": args.dis_model,
                    "gen_state_dict": gen_net.state_dict(),
                    "dis_state_dict": dis_net.state_dict(),
                    "avg_gen_state_dict": avg_gen_net.state_dict(),
                    "gen_optimizer": gen_optimizer.state_dict(),
                    "dis_optimizer": dis_optimizer.state_dict(),
                    "best_fid": best_fid,
                    "path_helper": args.path_helper,
                    "regan": args.regan,
                    "g": args.g,
                    "sparsity": args.sparsity
                },
                is_best,
                args.path_helper["ckpt_path"], epoch
            )
        del avg_gen_net


if __name__ == "__main__":
    main()

