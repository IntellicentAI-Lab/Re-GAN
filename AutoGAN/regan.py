import torch
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
            self.layers.append(w[1])

        self.reset_masks()

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

    def forward(self, x):
        return self.model(x)
