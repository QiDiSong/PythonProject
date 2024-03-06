from customer_model import DCAE
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random


# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = DCAE()
net.load_state_dict(torch.load("./dcae_encoder_decoder.pth", map_location=device))
net.to(device)
print(net.parameters())




processed_dataset = np.load('processed_dataset.npy')
# num_entries_to_add = 452000 - processed_dataset.shape[0]  # 计算差额
# padded_dataset = np.pad(processed_dataset, ((0, num_entries_to_add), (0, 0), (0, 0)), mode='constant', constant_values=0)
# padded_dataset = padded_dataset.reshape(-1, 1000, 3, 17)
# trainloader = torch.tensor(padded_dataset)
# images = trainloader[0]
inputs = torch.tensor(processed_dataset).float().to(device)
outputs = net(inputs)

# 选择部分数据进行测试和可视化
# dataiter = iter(testloader)
# images, labels = next(dataiter)
# images = images.to(device)
# labels = labels.to(device)
# 选择一个随机索引
# random_index = random.randint(0, images.size(0) - 1)
# print("random index: ", random_index)
# images_random_index = images[random_index].view(-1, 1, 784)  # 调整形状以匹配模型输入
# outputs_random_index = net(images_random_index)



# 可视化随机选择的图像和其重建结果
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1)
plt.imshow(images_random_index.view(28, 28).cpu().detach().numpy(), cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(outputs_random_index.view(28, 28).cpu().detach().numpy(), cmap='gray')
plt.title('Reconstructed Image')
plt.show()