import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, out_dim, pretrained=True):
        super(ResNetSimCLR, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        dim_mlp = self.backbone.fc.in_features
        # print(f"dim_mlp: {dim_mlp}") # 2048
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))
        self.fc=nn.Linear(out_dim, 100)

    def forward(self, x):
        return self.backbone(x)
