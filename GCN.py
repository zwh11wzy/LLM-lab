# 我们将使用 PyTorch Geometric (PyG) 框架学习 GCN，应用于图分类任务。
# 图分类是指在给定图数据集的情况下，根据某些结构图属性对整个图进行分类的问题。
# 图分类最常见的任务是分子属性预测，其中分子被表示为图，任务可能是推断分子是否抑制HIV病毒复制。
# 多特蒙德工业大学收集了各种不同的图形分类数据集，称为 TUDatasets，
# 可以通过 PyTorch Geometric 中的 torch_geometric.datasets.TUDataset 访问。 让我们加载并检查较小的数据集之一，即 MUTAG 数据集：

import torch
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root="data/TUDataset", name="MUTAG")  # 加载数据集

print()
print(f"Dataset: {dataset}:")
print("====================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

data = dataset[0]  # 得到数据中的第一个图

print()
print(data)
print("=============================================================")

# 获得图的一些统计特征
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Has self-loops: {data.has_self_loops()}")
print(f"Is undirected: {data.is_undirected()}")


# 该数据集提供了 188 个不同的图，任务是将每个图分类为两类中的一类。通过检查数据集的第一个图对象，
# 我们可以看到它有 17 个节点（具有 7 维特征向量）和 38条边（平均节点度为2.24`），它还有一个标签 (y=[1])。除了之前的数据集之外，
# 还提供了额外的 4 维边缘特征 (edge_attr=[38, 4])。 然而，为了简单起见，我们这次不会使用它们。
# PyTorch Geometric 提供了一些有用的实用程序来处理图数据集，例如，我们可以打乱数据集并使用前 150 个图作为训练图，同时使用剩余的图形进行测试：

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
# 语法：dataset[start:end]，其中 start 是起始索引，end 是结束索引（不包括 end 本身）。
# start 省略：表示从序列的开头开始。
# end 为 150：表示提取到索引 149（即前 150 个元素）。
test_dataset = dataset[150:]
# 语法：dataset[start:end]。
# start 为 150：表示从索引 150 开始。
# end 省略：表示一直提取到序列的末尾。


print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(test_dataset)}")

# 批处理对于图数据比较复杂和麻烦。PyTorch Geometric 选择了一种和常见图像数据集不同的方法来实现多个示例的并行化。
# 在这里，邻接矩阵以对角方式堆叠（创建一个包含多个孤立子图的巨型图），并且节点和目标特征在节点维度中简单地连接。
# 与其他批处理程序相比，该程序具有一些关键优势：
# （1）依赖于消息传递方案的 GNN 算子不需要修改，因为属于不同图的两个节点之间不会交换消息；
# （2）由于邻接矩阵以稀疏方式保存，仅保存非零条目（即边），因此不存在计算或内存开销。

# PyTorch Geometric 在 torch_geometric.data.DataLoader 类的帮助下自动将多个图批处理为单个巨型图，我们并不需要手动进行上述的复杂步骤。

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 在这里，我们选择 batch_size 为 64，从而产生 3 个（随机洗牌）小批量，包含所有 2⋅64+22=150 个图。

# 训练 GNN 进行图分类通常遵循一个简单的方法：
# （1）通过执行多轮消息传递来嵌入每个节点。
# （2）将节点嵌入聚合为统一的图嵌入（读出层）。
# （3）在图嵌入上训练最终分类器。
# 对于整图分类，我们需要一个读出层（readout layer），但最常见的一种是简单地取节点嵌入的平均值：

# PyTorch Geometric 通过 torch_geometric.nn.global_mean_pool 提供此功能，它接受小批量中所有节点的节点嵌入和分配向量批量，
# 以计算批量中每个图的大小为 [batch_size, hide_channels] 的图嵌入。也就是说，我们在这里不需要考虑批大小。

# 将 GNN 应用到图分类任务的最终架构如下所示，并允许完整的端到端训练：

from torch.nn import Linear  # 全连接层，用于将特征映射到输出类别。
import torch.nn.functional as F  # 包含各种激活函数和损失函数。
from torch_geometric.nn import GCNConv  # 图卷积神经网络层。
from torch_geometric.nn import global_mean_pool  # 全局平均池化层，用于将节点特征聚合为图特征。


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):  # 隐藏层的通道数，决定了每个卷积层的输出特征数。
        super(GCN, self).__init__()

        # super()：这是一个内置函数，用于调用父类的一个方法。它返回一个代理对象，代表了父类。
        # GCN：这是当前类的名称。在 Python 3 中，super() 可以省略参数，直接写成 super().__init__()，但在 Python 2 中需要显式指定。
        # self：这是当前实例的引用。在 Python 3 中，super() 不需要传递 self，但在 Python 2 中需要。
        # __init__()：这是构造函数，用于初始化对象。在这里，super(GCN, self).__init__() 调用了父类 torch.nn.Module 的构造函数，确保父类的初始化逻辑被执行。

        torch.manual_seed(12345)  # 设置随机种子，确保结果的可重复性。
        # 使用 GCNConv
        # 为了让模型更稳定我们也可以使用带有跳跃链接的 GraphConv
        # from torch_geometric.nn import GraphConv
        self.conv1 = GCNConv(
            dataset.num_node_features, hidden_channels
        )  # 第一个卷积层，输入特征数为节点特征数，输出特征数为隐藏层通道数。
        self.conv2 = GCNConv(
            hidden_channels, hidden_channels
        )  # 第二个卷积层，输入特征数为隐藏层通道数，输出特征数为隐藏层通道数(每个卷积层的输出特征数。)
        self.conv3 = GCNConv(
            hidden_channels, hidden_channels
        )  # 第三个卷积层，输入特征数为隐藏层通道数，输出特征数为隐藏层通道数。
        self.lin = Linear(
            hidden_channels, dataset.num_classes
        )  # 全连接层，输入特征数为隐藏层通道数，输出特征数为类别数。

    def forward(self, x, edge_index, batch):
        # 1. 获得节点的嵌入
        x = self.conv1(x, edge_index)  # 节点特征矩阵和边索引矩阵（描述图的连接关系）作为输入，进行卷积操作。
        x = x.relu()  # 激活函数，将特征值限制在0到1之间。
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. 读出层
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]，对每个图的节点特征取平均，得到图级别的特征。

        # 3. 应用最后的分类器，应用 dropout 正则化，防止过拟合
        x = F.dropout(x, p=0.5, training=self.training)  # 随机丢弃一部分特征。
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
print(model)

# 在这里，我们再次使用 GCNConv 和 ReLU(x)=max(x,0) 激活来获得局部节点嵌入，然后再将最终分类器应用到图读出层之上。
# 让我们训练我们的网络几个周期，看看它在训练和测试集上的表现如何：

from IPython.display import display, Javascript

display(Javascript("""google.colab.output.setIframeHeight(0, true, {maxHeight: 300})"""))

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # 迭代获得各个批数据
        out = model(data.x, data.edge_index, data.batch)  # 前向传播
        loss = criterion(out, data.y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        optimizer.zero_grad()  # 梯度清零


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # 迭代获得各个批数据
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # 取最大概率的类作为预测
        correct += int((pred == data.y).sum())  # 与真实标签做比较
    return correct / len(loader.dataset)  # 计算准确率


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# 模型达到了 78% 左右的测试准确率。 准确率波动的原因可以用相当小的数据集（只有 38 个测试图）来解释，
# 并且一旦将 GNN 应用到更大的数据集，通常就会消失。
