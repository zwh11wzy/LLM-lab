# 这里的异质图注意力网络 HAN 的代码来源于 DGL 官方。我们在复现 HAN 的时候需要注意以下要点

# 对应于节点级注意力，我们使用 DGL 自带的 GAT 来实现
# 对应于语义级注意力，我们自定义一个 SemanticAttention 来完成相应计算
# 最后，定义一个全连接层得到多分类结果

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_geometric.datasets import HGBDataset
from torch_geometric.transforms import RandomLinkSplit


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        # z 是输入的特征张量，通常是节点的特征表示。它的形状可能是 (N, D * K)，其中：
        # N 是样本的数量（例如，节点的数量）。
        # D 是特征的维度。
        # K 是某种聚合或头的数量。
        w = self.project(z).mean(0)  # (M, 1)线性变换，通常是通过一个全连接层（nn.Linear）实现的。
        # 对第一个维度（样本维度）进行平均，得到一个权重向量 w，其形状为 (M, 1)，其中 M 是特征的数量。
        beta = torch.softmax(w, dim=0)  # (M, 1)对权重 w 应用 softmax 函数，得到归一化的注意力系数 β。
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        # beta.expand((z.shape[0],) + beta.shape) 将 β 的形状扩展到 (N, M, 1)，使其可以与输入特征 z 进行逐元素相乘。
        # 这里，z.shape[0] 是样本数量 N，beta.shape 是 (M, 1)。
        return (beta * z).sum(1)  # (N, D * K)
        # (beta * z) 进行逐元素相乘，得到一个新的张量。
        # sum(1) 对第二个维度（特征维度）进行求和，最终返回的结果形状为 (N, D * K)。


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        # num_meta_paths: 表示元路径的数量。在异质图中，元路径用于定义不同类型的节点和边之间的关系。
        # in_size: 输入特征的维度，表示每个节点的特征向量的大小。
        # out_size: 输出特征的维度，表示经过图注意力层后每个节点特征的大小。
        # layer_num_heads: 注意力头的数量，表示在每个图注意力层中并行计算的注意力机制的数量。
        # dropout: dropout 概率，用于防止过拟合。
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        # self.gat_layers 是一个 ModuleList，用于存储多个图注意力层（GAT）。
        for i in range(num_meta_paths):  # for 循环根据 num_meta_paths 的数量，创建多个 GAT 层，每个层对应一个元路径。
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu))
            # 一个图注意力层的实现
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        # self.semantic_attention 是一个语义注意力层的实例，负责对多个 GAT 层的输出进行加权和聚合。
        # in_size=out_size * layer_num_heads 表示输入特征的维度是所有注意力头的输出特征的拼接结果。
        self.num_meta_paths = num_meta_paths  #  存储元路径的数量，以便在后续的前向传播中使用。

    def forward(self, gs, h):
        # gs: 这是一个图的列表，通常是多个异质图，每个图对应一个元路径。
        # h: 输入特征，通常是节点的特征表示。
        semantic_embeddings = []
        # semantic_embeddings 是一个空列表，用于存储每个图的语义嵌入。
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        # enumerate(gs) 用于遍历图列表 gs，同时获取每个图的索引 i 和图对象 g。
        # self.gat_layers[i](g, h) 调用第 i 个图注意力层（GAT），将当前图 g 和输入特征 h 传递给它。
        # flatten(1) 将输出的特征张量展平，通常是为了将多维特征转换为二维形式，以便后续处理。
        # 将每个图的语义嵌入添加到 semantic_embeddings 列表中。
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N样本数量（节点数量）, M元路径的数量, D * K每个节点的特征维度)
        # torch.stack(semantic_embeddings, dim=1) 将 semantic_embeddings 列表中的所有张量沿着指定的维度（这里是维度 1）堆叠在一起。
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)表示经过语义注意力层处理后的最终节点特征。


class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        # num_meta_paths: 表示元路径的数量。在异质图中，元路径用于定义不同类型的节点和边之间的关系。每个元路径对应一个图注意力层（GAT）。
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(num_meta_paths, hidden_size * num_heads[l - 1], hidden_size, num_heads[l], dropout)
                # 在后续层中，输入特征维度是 hidden_size * num_heads[l - 1]，这意味着每一层的输入特征是前一层的输出特征的拼接结果。
                # 输出特征维度仍然是 hidden_size，并使用 num_heads[l] 作为当前层的注意力头的数量。
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        # 这是一个全连接层，用于将最后一层的输出特征映射到最终的输出特征维度 out_size。
        # 这里的输入特征维度是 hidden_size * num_heads[-1]，即最后一层的输出特征。

    def forward(self, data):
        h = data.x
        for layer in self.layers:
            h = layer(data, h)
        return self.predict(h)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss  # 这里假设较低的 val_loss 是更好的

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# 加载 ACM 数据集
def load_acm_data():
    dataset = HGBDataset(root="/tmp/HGB", name="ACM")
    data = dataset[0]  # 使用数据集中的第一个图
    return data


# 在训练完成后，获取节点嵌入
def visualize_embeddings(model, data):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        node_embeddings = model(data).cpu().numpy()
        labels = data.y.cpu().numpy()

    # 将嵌入转换为 NumPy 数组
    node_embeddings = node_embeddings
    labels = labels

    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(node_embeddings[:, 0], node_embeddings[:, 1], c=labels, cmap="tab10", s=50)
    plt.colorbar(scatter, label="Classes")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.title("Node Embeddings Visualization")
    plt.grid()
    plt.show()


# 我们将使用异质图数据集 ACM，其与同质图数据集 Cora 的直观比较可见下图。我们可以看出 ACM 数据集具有不同类型的节点和不同类型的边。

# ACM 数据集提取了发表在 KDD、SIGMOD、SIGCOMM、MobiCOMM 和 VLDB 上的论文，
# 并将这些论文分为三个类（数据库、无线通信、数据挖掘），其中包括 3025 篇论文（P），5835名作者（A）和56个主题（S）。
# 论文特征对应于词包中的元素，代表关键词。

# 我们采用元路径集 {PAP, PSP} 来进行实验。在这里，我们根据论文发表的会议来标记根据他们发表的会议来标记论文。
# 这里我们将两种元路径分别构造成两张图，两张图的信息如下

"""

[Graph(num_nodes=3025, num_edges=29281,
       ndata_schemes={}
       edata_schemes={}),
 Graph(num_nodes=3025, num_edges=2210761,
       ndata_schemes={}
       edata_schemes={})]
"""

# 模型训练/测试

stopper = EarlyStopping(patience=10)
# patience=10 意味着如果在连续的 10 个训练轮次中，
# 模型在验证集上的性能没有提升（例如，验证损失没有降低或验证准确率没有提高），则训练将停止。
loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

for epoch in range(200):
    model.train()
    logits = model(data)
    loss = loss_fcn(logits[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc, train_micro_f1, train_macro_f1 = score(logits[data.train_mask], data.y[data.train_mask])
    val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, data, data.y, data.val_mask, loss_fcn)
    early_stop = stopper.step(val_loss.data.item(), val_acc, model)

    print(
        "Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | "
        "Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}".format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1
        )
    )

    if early_stop:
        break

stopper.load_checkpoint(model)
test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, data, data.y, data.test_mask, loss_fcn)
print(
    "Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}".format(
        test_loss.item(), test_micro_f1, test_macro_f1
    )
)

if __name__ == "__main__":
    data = load_acm_data()
    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(data)

    model = HAN(
        num_meta_paths=2, in_size=data.num_node_features, hidden_size=8, out_size=7, num_heads=[8, 1], dropout=0.5
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    stopper = EarlyStopping(patience=10)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits[train_data.train_mask], train_data.y[train_data.train_mask])
        loss.backward()
        optimizer.step()

        val_loss = criterion(logits[val_data.val_mask], val_data.y[val_data.val_mask])
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        if stopper(val_loss.item(), model):
            print("Early stopping")
            break

    test_logits = model(test_data)
    test_loss = criterion(test_logits[test_data.test_mask], test_data.y[test_data.test_mask])
    print(f"Test Loss: {test_loss.item():.4f}")

    visualize_embeddings(model, data)
