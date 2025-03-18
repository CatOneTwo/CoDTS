import torch.nn as nn
from pdb import set_trace as pause

class EncoderDecoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg

        self.feature_num = model_cfg['input_dim']
        layer_num = model_cfg['layer_num']
        self.feature_stride = 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=2, padding=0, bias=False
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU()]

            cur_layers.extend([
                nn.Conv2d(feature_num, feature_num * self.feature_stride,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_num * self.feature_stride,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            ])

            self.encoder.append(nn.Sequential(*cur_layers))
            feature_num = feature_num * self.feature_stride

        feature_num = self.feature_num
        for i in range(layer_num):
            cur_layers = [nn.Sequential(
                nn.ConvTranspose2d(
                    feature_num * 2, feature_num,
                    kernel_size=2,
                    stride=2, bias=False
                ),
                nn.BatchNorm2d(feature_num,
                               eps=1e-3, momentum=0.01),
                nn.ReLU()
            )]

            cur_layers.extend([nn.Sequential(
                nn.Conv2d(
                    feature_num, feature_num, kernel_size=3,
                    stride=1, bias=False, padding=1
                ),
                nn.BatchNorm2d(feature_num, eps=1e-3,
                               momentum=0.01),
                nn.ReLU()
            )])
            self.decoder.append(nn.Sequential(*cur_layers))
            feature_num *= 2

    def forward(self, x):

        for i in range(len(self.encoder)):
            x = self.encoder[i](x)

        for i in range(len(self.decoder)-1, -1, -1):
            x = self.decoder[i](x)

        return x


class LightDecoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.model_cfg = model_cfg
        input_dim = self.model_cfg['input_dim']
        output_dim = self.model_cfg['output_dim']
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                        input_dim, output_dim,
                        kernel_size=3,
                        stride=2, 
                        padding = 1, output_padding= 1,
                        bias=False
                    ),
                nn.BatchNorm2d(output_dim, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        

    def forward(self, x):
        x = self.decoder(x) # 尺寸扩大2倍
        return x