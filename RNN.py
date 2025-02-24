# 虽然图像是一个二维的，但是如果我们把图像拉伸成一维的向量，那么我们可以把向量的长度当作时间的长度。
# 这样，我们就仍然使用 RNN 模型来做 MNIST 数据集上的手写数字识别。
# 先定义一些超参数和导入数据。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# Set hyperparameters
batch_size = 64
input_size = 28  # MNIST images are 28x28 pixels
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 10
learning_rate = 0.001

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = MNIST(root="./data", train=True, download=True, transform=transform)
testset = MNIST(root="./data", train=False, download=True, transform=transform)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 定义 RNN 模型


# Define the RNN model
class RNN(nn.Module):  # 定义一个继承自nn.Module的RNN类
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  # 初始化RNN类的参数
        super(RNN, self).__init__()  # 调用父类的构造函数
        self.hidden_size = hidden_size  # 设置隐藏层大小
        self.num_layers = num_layers  # 设置RNN的层数
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # 定义RNN层
        self.fc = nn.Linear(hidden_size, num_classes)  # 定义全连接层

    def forward(self, x):  # 定义前向传播函数
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        out, _ = self.rnn(x, h0)  # 通过RNN层
        out = self.fc(out[:, -1, :])  # 通过全连接层
        return out  # 返回输出


# 张量的形状
# 在这个 RNN 模型中，out 是 RNN 层的输出张量，其形状通常为 (batch_size, sequence_length, hidden_size)，其中：
# batch_size：批次大小，表示一次输入的样本数量。
# sequence_length：序列长度，表示每个样本的时间步数。
# hidden_size：隐藏层大小，表示每个时间步的输出特征数。
# 切片语法 out[:, -1, :]
# :：表示选择该维度上的所有元素。
# -1：表示选择该维度上的最后一个元素。在这里，-1 用于选择序列的最后一个时间步。
# :：表示选择该维度上的所有元素。
# 因此，out[:, -1, :] 的作用是：
# [:, -1, :]：对于每个样本（批次中的每一行），选择序列的最后一个时间步的所有特征。


# Initialize the model
model = RNN(input_size, hidden_size, num_layers, num_classes)

# 定义损失函数和优化器，并进行训练和测试。

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.reshape(-1, input_size, input_size)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 更新模型参数。

        if (i + 1) % 100 == 0:  # 每 100 个批次输出一次当前的损失，帮助监控训练过程。
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

# Testing
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.reshape(-1, input_size, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # 作用：获取每个样本的预测类别。
        # 解释：torch.max 返回每行的最大值及其索引，这里我们只需要索引（即预测的类别）。
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
