import torch.optim as optim
from model import Discriminator, Generator
import argparse
from utils import get_loader, gradient_penalty
from regan import Regan_training


def train(epoch, num_epochs, critic, gen, loader, step, opt_critic, opt_gen):
    for batch_idx, (real, _) in enumerate(loader, 0):

        alpha = (batch_idx + epoch * len(loader)) / (num_epochs * len(loader))

        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, args.z_dim, 1, 1).to(device)

        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=device)
        loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + args.lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
        )

        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        loss_gen.backward()
        # scaler_gen.step(opt_gen)

        if args.regan and gen.train_on_sparse:
            gen.apply_masks()

        opt_gen.step()

        alpha = min(alpha, 1)

        # Output training stats
        if batch_idx % 50 == 0:
            print('[%d][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tAlpha: %.2f'
                  % (4 * 2 ** step, epoch, num_epochs, batch_idx, len(loader), loss_critic.item(),
                     loss_gen.item(), critic_real.mean().item(), critic_fake.mean().item(),
                     gen_fake.mean().item(), alpha))

    return alpha


def main():
    gen = Regan_training(Generator(args.z_dim, args.in_channels).to(device), sparsity=args.sparsity)
    critic = Discriminator(args.in_channels).to(device)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.99))

    step = 0

    flag_g = 1
    image_size = 4 * 2 ** step
    for num_epochs in progressive_epochs[step:]:
        print(f"Current image size: {image_size}")
        # 4->0, 8->1, 16->2, 32->3, 64 -> 4

        loader, dataset = get_loader(image_size, args.workers, batch_size, '../dataset', args.data_ratio)

        if args.epoch != -1:
            num_epochs = args.epoch
        for epoch in range(1, 1 + num_epochs):
            print(f"Epoch [{epoch}/{num_epochs}]")

            if args.regan and step == len(batch_size) - 1:
                # Warm-up phase, do not enable the ReGAN training
                if epoch < args.warmup_epoch + 1:
                    print('current is warmup training')
                    gen.train_on_sparse = False

                # Warm-up phase finished, get into Sparse training phase
                elif epoch > args.warmup_epoch and flag_g < args.g + 1:
                    print('epoch %d, current is sparse training' % epoch)
                    # turn training mode to sparse, update mask
                    gen.turn_training_mode(mode='sparse')
                    # make sure the learning rate of sparse phase is the original one
                    if flag_g == 1:
                        print('turn learning rate to normal')
                        for params in opt_gen.param_groups:
                            params['lr'] = args.lr
                    flag_g = flag_g + 1

                # Sparse training phase finished, get into dense training phase
                elif epoch > args.warmup_epoch and flag_g < 2 * args.g + 1:
                    print('epoch %d, current is dense training' % epoch)
                    # turn training mode to dense
                    gen.turn_training_mode(mode='dense')
                    # make sure the learning rate of Dense phase is 10 times smaller than the original one
                    if flag_g == args.g + 1:
                        print('turn learning rate to 10 times smaller')
                        for params in opt_gen.param_groups:
                            params['lr'] = args.lr * 0.1
                    flag_g = flag_g + 1

                    # When curren Sparse-Dense pair training finished, get into next pair training
                    if flag_g == 2 * args.g + 1:
                        print('clean flag')
                        flag_g = 1

            alpha = train(epoch, num_epochs, critic, gen, loader, step, opt_critic, opt_gen)

        step += 1  # progress to the next img size
        image_size = 4 * 2 ** step


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--load', type=bool, default=False)
    argparser.add_argument('--load_epoch', type=int, default=0)
    argparser.add_argument('--epoch', type=int, default=-1)
    argparser.add_argument('--z_dim', type=int, default=512)
    argparser.add_argument('--in_channels', type=int, default=512)
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--workers', type=int, default=4)
    argparser.add_argument('--lambda_gp', type=int, default=10)
    argparser.add_argument('--data_ratio', type=float, default=1.0)
    argparser.add_argument('--regan', action="store_true")
    argparser.add_argument('--sparsity', type=float, default=0.3)
    argparser.add_argument('--g', type=int, default=2)
    argparser.add_argument('--warmup_epoch', type=int, default=1)


    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_training_at = 4
    batch_size = [128, 128, 128, 64]
    progressive_epochs = [2, 2, 2, 8]

    main()
