import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Regan_training(nn.Module):

    def __init__(self, model, sparsity, train_on_sparse=False):
        super(Regan_training, self).__init__()

        self.model = model
        self.sparsity = sparsity
        self.train_on_sparse = train_on_sparse
        self.layers = []
        self.masks = []
        layers = list(self.model.named_parameters())

        for i in range(0, len(layers)):
            w = layers[i]
            if "to_rgb" not in w[0]:
                self.layers.append(w[1])
        self.reset_masks()

    def get_latent(self, input):
        return self.model.style(input)

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.model.style_dim, device=self.model.input.input.device
        )
        latent = self.model.style(latent_in).mean(0, keepdim=True)

        return latent

    def reset_masks(self):
        for w in self.layers:
            mask_w = torch.ones_like(w, dtype=bool)
            self.masks.append(mask_w)

        return self.masks

    def update_masks(self):

        for i, w in enumerate(self.layers):
            q_w = torch.quantile(torch.abs(w), q=self.sparsity)
            mask_w = torch.where(torch.abs(w) < q_w, True, False)

            self.masks[i] = mask_w

    def turn_training_mode(self, mode):
        if mode == 'dense':
            self.train_on_sparse = False
        else:
            self.train_on_sparse = True
            self.update_masks()

    def apply_masks(self):
        for w, mask_w in zip(self.layers, self.masks):
            w.data[mask_w] = 0
            w.grad.data[mask_w] = 0

    def forward(self, x, return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,):
        return self.model(x, return_latents,
        inject_index,
        truncation,
        truncation_latent,
        input_is_latent,
        noise,
        randomize_noise)