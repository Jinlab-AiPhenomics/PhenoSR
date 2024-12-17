import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.hrnet_backbone import BN_MOMENTUM, hrnet_classification


# https://blog.csdn.net/weixin_38715903/article/details/101629781?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=dis
class HRnet_Backbone(nn.Module):
    def __init__(self, backbone='hrnetv2_w18', pretrained=False):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)
        # print("stage: 2", len(y_list), y_list[0].shape, y_list[1].shape)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)
        # print("stage: 3", len(y_list), y_list[0].shape, y_list[1].shape, y_list[2].shape)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        # print("stage: 4", len(y_list), y_list[0].shape, y_list[1].shape, y_list[2].shape, y_list[3].shape)

        return y_list


class HRnet(nn.Module):
    def __init__(self, num_classes=21, backbone='hrnetv2_w18', pretrained=False):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(backbone=backbone, pretrained=pretrained)

        last_inp_channels = np.int64(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        # #self.softmax = nn.Softmax(dim=1)
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1).cuda()
        # self.img_range = 1.
        
    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        # 在进入basenet之前已经均值化等处理过了，不需要在分割模型里再处理一次
        # inputs = (inputs - self.mean) * self.img_range
        # x的值为stage4输出的结果，包含四个分支最终的特征图
        x = self.backbone(inputs)
        out_feature = x
        
        # Upsampling
        # 第一个分支特征图大小不变
        # 32, 64, 64
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        # 将第二个分支特征图上采样至第一个分支特征图大小，其他分支同理，通道数不变
        # 64, 32, 32 -> 64, 64, 64
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # 128, 16, 16 -> 128, 64, 64
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # 256, 8, 8 -> 256, 64, 64
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        # 480, 64, 64
        cat_x = torch.cat([x[0], x1, x2, x3], 1)
        # out_feature = cat_x
        # semantic segmentation
        x = self.last_layer(cat_x)
        # out_feature = x
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # x = self.softmax(x)
        return x, out_feature


if __name__ == "__main__":
    model = HRnet(num_classes=21, backbone='hrnetv2_w32', pretrained=False)
    model.eval()
    data = torch.rand(1, 3, 256, 256)
    x, cat_x = model(data)
    # from torchkeras import summary
    # summary(model, input_shape=(3, 128, 128))
