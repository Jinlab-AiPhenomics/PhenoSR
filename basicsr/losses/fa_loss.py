import torch
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FALoss(torch.nn.Module):
    def __init__(self, loss_weight, subscale=0.0625):
        super(FALoss, self).__init__()
        self.subscale = int(1 / subscale)
        self.loss_weight = loss_weight

    def forward(self, feature1, feature2):
        # 使用平均池化层对feature进行下采样，缩小特征图的尺寸
        feature1 = torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2 = torch.nn.AvgPool2d(self.subscale)(feature2)

        m_batchsize, C, height, width = feature1.size()
        # 分别对两个特征图进行形状变换，以便计算两个特征图的 Gram 矩阵
        feature1 = feature1.view(m_batchsize, -1, width * height)  # [N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        # 计算feature1的 Gram 矩阵
        mat1 = torch.bmm(feature1.permute(0, 2, 1), feature1)  # [N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width * height)  # [N,C,W*H]
        mat2 = torch.bmm(feature2.permute(0, 2, 1), feature2)  # [N,W*H,W*H]

        # 计算两个 Gram 矩阵的 Frobenius 范数之差的 L1 范数
        L1norm = torch.norm(mat2 - mat1, 1)
        # 返回归一化的 L1 范数作为损失值，以特征图的大小进行归一化
        return L1norm / ((height * width) ** 2)
