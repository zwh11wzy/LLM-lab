# 这里我们使用 DGL 来对 GAT 模型进行复现，在复现 GAT 模型的时候重点在于以下几点

# 首先对所有的节点特征进行Wh 变换
# 然后计算出每条边的注意力系数 e ij(即，节点 j 对节点 i 的权重)
# 之后对eij使用 softmax 操作进行归一化得到归一化的注意力系数
# 对邻居节点使用归一化后的注意力系数进行加权求和
# 重复上述过程得到多头注意力的多个结果，然后进行拼接或求和

# 我们首先来看实现单头注意力的代码

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset  # 确保导入 CoraGraphDataset

# from dgl import transform  # 如果你需要使用 DGL 的 transform 功能


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        """
        g : dgl的grapg实例
        in_dim : 节点embedding维度
        out_dim : attention编码维度
        """
        self.g = g  # dgl的一个graph实例
        self.fc = nn.Linear(in_dim, out_dim, bias=False)  # 对节点进行通用的映射的fc
        # 这一步是对每个节点的特征进行线性变换，通常用于特征降维或升维。
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)  # 计算edge attention的fc
        # 这是另一个线性变换模块，用于计算边的注意力系数。
        # 2 * out_dim 是输入特征的维度，因为注意力机制通常需要结合源节点和目标节点的特征。
        # 1 是输出特征的维度，表示计算得到的注意力系数。
        # bias=False 表示不使用偏置项。
        # self.attn_fc 是一个线性层，用于计算边的注意力权重。
        self.reset_parameters()  # 参数初始化

    """
    edges 的src表示源节点,dst表示目标节点
    """

    def edge_attention(self, edges):

        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)  # eq.1 里面的拼接操作
        # ["z"]:
        # ["z"] 是字典的键，用于访问源节点的特定特征。
        # 在这段代码中，"z" 是通过线性变换得到的节点特征，存储在图的节点数据中。
        a = self.attn_fc(z2)  # eq.1 里面对e_{ij}的计算
        return {
            "e": F.leaky_relu(a)
        }  # 这里的return实际上等价于 edges.data['e'] =  F.leaky_relu(a),这样后面才能将这个 e 传递给 源节点

    def reset_parameters(self):  # 初始化神经网络层self.fc.weight，self.attn_fc.weight的权重
        gain = nn.init.calculate_gain(
            "relu"
        )  # 不同的激活函数有不同的增益值，calculate_gain 会根据激活函数类型返回合适的增益。
        nn.init.xavier_normal_(self.fc.weight, gain=gain)  # 用于使用 Xavier 正态分布初始化权重。
        # Xavier 初始化是一种常用的权重初始化方法，旨在保持输入和输出的方差相等。
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)  # 通过指定 gain，可以确保初始化适合特定的激活函数

    def message_func(self, edges):
        return {
            "z": edges.src["z"],
            "e": edges.data["e"],
        }  # 将前面 edge_attention 算出来的 e 以及 edges的源节点的 node embedding 都传给 nodes.mailbox

    def reduce_func(self, nodes):  # reduce_func 是 DGL 中用于聚合消息的函数。从而更新每个节点的特征表示。
        # 在 GAT 中，消息传递机制包括从源节点向目标节点发送消息，然后在目标节点上聚合这些消息。
        # 通过 message_func 之后即可得到 源节点的node embedding 与 edges 的 e
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        # softmax归一化，得到 a，nodes.mailbox["e"] 包含了从所有邻居节点传来的注意力系数。
        # dim=1 指定在特征维度上进行 softmax 操作，确保每个节点的邻居权重和为 1。
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)  # 根据a进行加权求和
        return {"h": h}  # 中间隐藏层参数

    def forward(self, h):
        z = self.fc(h)  # eq. 1 这里的h就是输入的node embedding
        # h 是输入的节点特征（或嵌入），z 是经过线性变换后的特征。
        self.g.ndata["z"] = z
        self.g.apply_edges(self.edge_attention)  # eq. 2
        self.g.update_all(self.message_func, self.reduce_func)  # eq. 3 and 4
        # 执行消息传递和节点特征更新。
        # message_func 负责从源节点向目标节点传递消息。
        # reduce_func 负责在目标节点上聚合来自邻居节点的消息。
        return self.g.ndata.pop(
            "h"
        )  # 返回经过attention的 node embedding。返回更新后的节点特征，并从节点数据中移除 "h"。


# 在理解上述代码之后，就大致明白了 GAT 的核心计算逻辑，下面展示多头注意力部分代码


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        """
        g : dgl的grapg实例
        in_dim : 节点embedding维度
        out_dim : attention编码维度
        num_heads : 头的个数
        merge : 最后一层为'mean',其他层为'cat'
        """
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            # 这里简单粗暴，直接声明 num_heads个GATLayer达到 Multi-Head的效果
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]  # head_outs 是一个列表，包含所有注意力头的输出。
        # 根据merge的类别来处理 Multi-Head的逻辑
        if self.merge == "cat":
            return torch.cat(head_outs, dim=1)  # 在特征维度上拼接，将所有注意力头的输出连接在一起。
        else:  # 如果 self.merge 不是 'cat'，则使用平均的方式合并输出。
            return torch.mean(torch.stack(head_outs))  # torch.stack 是 PyTorch 中的一个函数，用于在新维度上堆叠张量。


# 最后，在明白完整版的多头注意力层之后，我们构建图注意力网络 GAT


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        """
        g : dgl的grapg实例
        in_dim : 节点embedding维度
        hidden_dim : 隐层的维度
        out_dim : attention编码维度
        num_heads : 头的个数
        """
        # 这里简简单单的写了一个两层的 MultiHeadGATLayer
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 在 layer1 中，num_heads 决定了并行注意力头的数量。
        # 多头注意力机制通过多个独立的注意力头来捕获不同的特征表示。
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        # 这是因为 layer1 的输出是所有注意力头的拼接结果，其特征维度是 hidden_dim 乘以 num_heads。
        # 在 layer2 中，1 表示只有一个注意力头。
        # 这通常用于输出层，简化输出特征的聚合。

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)  # elu 是 torch.nn.functional 中的一个激活函数，表示指数线性单元。
        # 相较于 ReLU，ELU 在输入为负时不会导致神经元“死亡”（即输出恒为零）。
        # ELU 的输出均值更接近于零，这有助于加速神经网络的收敛。
        h = self.layer2(h)
        return h


"""

我们将使用论文数据集 Core，其基本信息如下:

NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000

可以看到，Cora 数据集一共有 2708 个节点，10556 条边，每个节点的特征维度都是 1433，
并且每个节点都有一个 7 分类的类别的标签，我们这里的任务就是使用 GAT 来对 Cora 的数据进行多分类的预估。
在 DGL 里面直接内置了 Cora 数据集，我们可以直接运行下面的代码读取 Cora 数据集
"""


def load_core_data():
    data = CoraGraphDataset()
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    mask = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    return g, features, labels, mask


"""
g : dgl的cora graph实例
features : 每个节点的向量表征，其维度为[2708,1433]
labels : 每个节点的标签,其维度为[2708,1]
mask : 这是一个长度为2708的取值全为[True,False]的list,True代表使用这个node，False则代表不使用，
通过mask可以区分train/vali/test的数据
"""
g, features, labels, mask = load_core_data()


def evaluate(features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    #     weight_decay 是权重衰减系数，通常用于 L2 正则化。
    # L2 正则化通过在损失函数中添加参数的平方和来防止过拟合。
    # weight_decay 实现了 L2 正则化，通过在损失函数中添加一个惩罚项来限制模型的复杂度。
    # 这个惩罚项是所有模型参数的平方和乘以 weight_decay 系数。

    # training loop
    for epoch in range(100):
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(epoch, loss.item(), acc))


if __name__ == "__main__":
    g, features, labels, masks = load_core_data()
    model = GAT(g, in_dim=features.shape[1], hidden_dim=8, out_dim=7, num_heads=8)  # 根据需要调整参数
    train(features, labels, masks, model)


import matplotlib.pyplot as plt
import numpy as np

# 假设你已经有了以下变量
# features: 节点特征
# labels: 节点标签
# model: 训练好的 GAT 模型

# 获取节点嵌入
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    node_embeddings = model(features)  # 获取节点嵌入

# 将嵌入转换为 NumPy 数组
node_embeddings = node_embeddings.cpu().numpy()
labels = labels.cpu().numpy()

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(node_embeddings[:, 0], node_embeddings[:, 1], c=labels, cmap="tab10", s=50)
plt.colorbar(scatter, label="Classes")
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.title("Node Embeddings Visualization")
plt.grid()
plt.show()
