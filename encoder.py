import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, output_dim=2048):
        super(Encoder, self).__init__()

        # 使用预训练的 ResNet50 模型
        self.resnet = models.resnet50(pretrained=True)

        # 移除 ResNet50 的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # 添加全局平均池化层
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # 添加全连接层和批标准化
        self.fc = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 创建编码器模型
def getEncoder():
    return Encoder()


if __name__ == '__main__':
    print(getEncoder())
