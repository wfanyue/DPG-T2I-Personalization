import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class FullyConvRegressionModel(nn.Module):
    def __init__(self):
        super(FullyConvRegressionModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)    # 2
        
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # 全连接层
        self.fc = nn.Linear(64*64, 64)
        self.fc2 = nn.Linear(64, 1)
        # 将线性层的权重初始化为在0到1之间的随机值
        nn.init.uniform_(self.fc.weight, a=0.0, b=1 / (64*64))
        nn.init.uniform_(self.fc.bias, a=0.0, b=1 / (64*64))

        nn.init.uniform_(self.fc2.weight, a=0.0, b=1 / (64))
        nn.init.uniform_(self.fc2.bias, a=0.0, b=1 / (64))

        nn.init.uniform_(self.conv1.weight, a = 0.0, b = 1/ (3*3))
        nn.init.uniform_(self.conv1.bias, a = 0.0, b = 1 / (3*3))


    def forward(self, x):
        # 输入特征经过卷积和激活函数
        x = self.conv1(x)
        # remember to reopen
        x = self.leaky_relu(x)
 
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # 将特征展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class SimRegressionModel(nn.Module):
    def __init__(self, output_dim=1):
        super(SimRegressionModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)    # 2
        
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # 全连接层
        self.fc = nn.Linear(64*64, 64*32)
        self.fc2 = nn.Linear(64*32, 32*16)
        # 将线性层的权重初始化为在0到1之间的随机值
        # nn.init.uniform_(self.fc.weight, a=0.0, b=1 / (64*64))
        # nn.init.uniform_(self.fc.bias, a=0.0, b=1 / (64*64))

        # nn.init.uniform_(self.fc2.weight, a=0.0, b=1 / (64))
        # nn.init.uniform_(self.fc2.bias, a=0.0, b=1 / (64))

        # nn.init.uniform_(self.conv1.weight, a = 0.0, b = 1/ (3*3))
        # nn.init.uniform_(self.conv1.bias, a = 0.0, b = 1 / (3*3))


    def forward(self, x):
        # 输入特征经过卷积和激活函数
        x = self.conv1(x)
        # remember to reopen
        x = self.leaky_relu(x)
 
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.conv3(x)
        # 将特征展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x



if __name__ == '__main__':
    pass