import argparse
from dsd import *
import torch
from torchvision import utils
print('test')
from model import Generator
print('test')

if __name__ == "__main__":
    print('start')
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()
    print('---')

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g = DSDTraining(g, 0.3)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)
    latent = torch.randn(args.n_sample, 512, device=args.device)

    latent = g.get_latent(latent)

    for xx in [10, 20, 30, 50, 100]:

        args.index = xx
        direction = args.degree * eigvec[:, args.index].unsqueeze(0)

        img, _ = g(
            [latent],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img1, _ = g(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img2, _ = g(
            [latent - direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        img3, _ = g(
            [latent + 2 * direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        img4, _ = g(
            [latent - 2 * direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img5, _ = g(
            [latent + 3 * direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        img6, _ = g(
            [latent - 3 * direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        img7, _ = g(
            [latent - 4 * direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        # print(torch.cat([img4, img1, img, img2, img3], 0).shape)
        # print(img4[0].shape)
        # print(img4[0].unsqueeze(0).shape)
        # print(torch.cat([img4[0].unsqueeze(0), img1[0].unsqueeze(0),
        #                  img[0].unsqueeze(0),
        #                  img2[0].unsqueeze(0), img3[0].unsqueeze(0)], 0).shape)

        for i in range(args.n_sample):
            grid = utils.save_image(
                torch.cat([img7[i].unsqueeze(0), img6[i].unsqueeze(0), img4[i].unsqueeze(0), img2[i].unsqueeze(0),
                           img[i].unsqueeze(0),
                           img1[i].unsqueeze(0), img3[i].unsqueeze(0), img5[i].unsqueeze(0)], 0),
                f"./factor/{args.out_prefix}_index-{args.index}_degree-{args.degree}_{i}.png",
                normalize=True,
                # range=(-1, 1),
                nrow=args.n_sample,
            )

        # for i in range(args.n_sample):
        #     grid = utils.save_image(
        #         torch.cat([img[i].unsqueeze(0), img1[i].unsqueeze(0), img2[i].unsqueeze(0), img3[i].unsqueeze(0)], 0),
        #         f"./factor/{args.out_prefix}_index-{args.index}_degree-{args.degree}_{i}.png",
        #         normalize=True,
        #         range=(-1, 1),
        #         nrow=args.n_sample,
        #     )

    # grid = utils.save_image(
    #     torch.cat([img4, img1, img, img2, img3], 0),
    #     f"./factor/{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
    #     normalize=True,
    #     range=(-1, 1),
    #     nrow=args.n_sample,
    # )
