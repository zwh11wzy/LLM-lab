# 这里我们使用 MNIST 数字识别数据集。因为，对于每一个样本而言，我们的输入必须是一个一维的向量，
# 虽然 MNIST 数据集含有分辨率为28×28 的图片，但他们默认被存储成了一个 28×28=784的向量。
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # 添加这一行
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 定义应用于数据的转换
transform = transforms.Compose([transforms.ToTensor()])

# 加载 MNIST 数据集
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 设置超参数
batch_size = 64
input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.001
num_epochs = 10

# 创建数据加载器
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 接下来，我们定义一个两层的 MLP 模型。


# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 初始化模型
model = MLP(input_size, hidden_size, output_size)

# 然后，我们使用交叉熵函数作为损失函数，使用 Adam 算法作为优化器。

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 最后，我们进行训练与测试

# 训练循环
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}")

# 测试
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
