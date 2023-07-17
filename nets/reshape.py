import torch
import torch.nn as nn

class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=nn.ReLU(inplace=True), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)


#-----------------------------------------------------------------------
import torch
import torch.nn as nn

class MultiScaleRepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=nn.ReLU(inplace=True), deploy=False):
        super(MultiScaleRepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = act

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        dense_out = self.act(self.rbr_dense(inputs))
        out_1x1 = self.act(self.rbr_1x1(inputs))
        
        # 多尺度特征融合
        fused_out = torch.cat((dense_out, out_1x1), dim=1)
        out = dense_out + out_1x1 + id_out
        
        return out, fused_out
