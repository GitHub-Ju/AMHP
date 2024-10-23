import numpy as np
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class AMHP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        half_input_dim = int(input_dim / 2)
        self.model, _ = clip.load("ViT-B/16", device=device, jit=False)
        self.mlp = nn.Linear(input_dim, half_input_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, text, text_template):

        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        image_features = self.model.encode_image(x)
        text_features = self.model.encode_text(text)
        text_template_feature = self.model.encode_text(text_template)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features.float()
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features = text_features.float()
        text_template_feature = text_template_feature / text_template_feature.norm(dim=1, keepdim=True)
        text_template_feature = text_template_feature.float()

        residual = torch.cat((image_features, text_template_feature), 1)
        residual = self.mlp(residual)
        feature = image_features + residual

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * feature @ text_features.t()

        logits_per_image = F.softmax(logits_per_image, dim=1)

        return logits_per_image
