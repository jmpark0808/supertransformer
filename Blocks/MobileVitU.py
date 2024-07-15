
import torch.nn as nn
import torch
from Blocks.MobileVitV2 import MobileViTv3_v2



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileVITV3_unet(nn.Module):
    def __init__(self, input_size, pre_trained='weights/mobilenet_v2.pth.tar'):
        super(MobileVITV3_unet, self).__init__()

        self.backbone = MobileViTv3_v2(image_size=input_size, width_multiplier=0.5, num_classes=1000)
        channels = self.backbone.channels

        self.dconv1 = nn.ConvTranspose2d(channels[-1], channels[-2], 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(channels[-2]*2, channels[-2], 1, 6)

        self.dconv2 = nn.ConvTranspose2d(channels[-2], channels[-3], 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(channels[-3]*2, channels[-3], 1, 6)

        self.dconv3 = nn.ConvTranspose2d(channels[-3], channels[-4], 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(channels[-4]*2, channels[-4], 1, 6)

        self.dconv4 = nn.ConvTranspose2d(channels[-4], channels[-5], 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(channels[-5]*2, channels[-5], 1, 6)

        self.dconv5 = nn.ConvTranspose2d(channels[-5], channels[-5], 4, padding=1, stride=2)

        self.conv_last = nn.Conv2d(channels[-5], 1, 1)

        # self.conv_score = nn.Conv2d(3, 1, 1)

        # self.pe = nn.Sequential(*[nn.Conv2d(22,32, 1, 1), nn.BatchNorm2d(32), nn.ReLU6(), nn.Conv2d(32, 32, 1, 1)])

        # self._init_weights()

        if pre_trained is not None:
            self.backbone.load_state_dict(torch.load(pre_trained, map_location=torch.device('cpu')))

    def forward(self, x):

 
  
        x = self.backbone.layer_1(self.backbone.conv_0(x))
        x1 = x
       
        
        x = self.backbone.layer_2(x)
        x2 = x
      

        
        x = self.backbone.layer_3(x)
        x3 = x
       

        
        x = self.backbone.layer_4(x)
        x4 = x
       

        
        x = self.backbone.layer_5(x)
        x5 = x
    
        
        
        up1 = torch.cat([
            x4,
            self.dconv1(x)
        ], dim=1)
        up1 = self.invres1(up1)
       

        up2 = torch.cat([
            x3,
            self.dconv2(up1)
        ], dim=1)
        up2 = self.invres2(up2)
       

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        

        up4 = torch.cat([
            x1,
            self.dconv4(up3)
        ], dim=1)
        up4 = self.invres4(up4)
       
        
        x = self.dconv5(up4)
        x = self.conv_last(x)


        # x = self.conv_score(x)
      

        # x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # logging.debug((x.shape, 'interpolate'))

        return x