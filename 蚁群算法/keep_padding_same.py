import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CustomConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0) # 注意：这里的padding设置为0

    def forward(self, x):
        # F.pad函数的pad参数格式为：(左填充量, 右填充量)
        # 对于一维数据，你需要根据需要调整这些值
        x_padded = F.pad(x, (padding_left, padding_right), 'constant', 0)
        return self.conv(x_padded)

# 示例参数
in_channels = 3
out_channels = 32
kernel_size = 10
stride = 1
padding_left = 4  # 你可以根据需要调整这个值
padding_right = 5  # 你也可以根据需要调整这个值

# 创建模型实例
model = CustomConv1d(in_channels, out_channels, kernel_size, stride)

# 假设你有一个输入张量
input_tensor = torch.randn(1, in_channels, 100)  # 示例输入：batch_size=1, channels=in_channels, width=100

# 应用模型
output = model(input_tensor)

# 输出结果的形状
print(output.shape)
