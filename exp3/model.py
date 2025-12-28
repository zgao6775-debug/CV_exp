import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1: 输入1通道，输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 卷积层2: 输入32通道，输出64通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout层: 防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层1: 输入 9216 (64 * 12 * 12), 输出 128
        self.fc1 = nn.Linear(9216, 128)
        # 全连接层2: 输入 128, 输出 10 (类别数)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: [batch_size, 1, 28, 28]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # 最大池化: 2x2
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # 展平
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # 输出对数概率
        output = F.log_softmax(x, dim=1)
        return output
