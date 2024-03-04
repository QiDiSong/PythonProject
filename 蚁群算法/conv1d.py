import torch
import torch.nn as nn

# 创建一个Conv1d层
# 输入通道数为1，输出通道数为1，卷积核大小为3
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)

# 创建一个随机的一维输入数据
# 假设有一个批量大小为1的数据，具有1个通道，长度为5
input = torch.randn(1, 1, 5)

# 应用Conv1d卷积
output = conv1d(input)

print("Input shape:", input.shape)
print("Output shape:", output.shape)
print("Output:", output)
