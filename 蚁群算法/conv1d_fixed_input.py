import torch
import torch.nn as nn

# 创建一个固定值的一维输入数据，以便于验证计算结果
# 使用torch.tensor手动指定输入数据的值
input_fixed = torch.tensor([[[0.5, -1.5, 2.0, 0.5, -0.5]]], dtype=torch.float32)

# 创建一个Conv1d层
# 输入通道数为1，输出通道数为1，卷积核大小为3
conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3)

# 为了确定计算结果，我们也需要固定卷积层的权重和偏置
# 这里我们手动设置卷积核的权重和偏置为固定值
conv1d.weight = nn.Parameter(torch.tensor([[[0.2, 0.5, -0.5]]]))
conv1d.bias = nn.Parameter(torch.tensor([0.1]))

# 应用Conv1d卷积
output_fixed = conv1d(input_fixed)

# 打印固定输入的输出结果
print(output_fixed)
print(input_fixed.shape)
print(output_fixed.shape)