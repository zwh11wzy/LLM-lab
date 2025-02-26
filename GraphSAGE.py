# GraphSAGE 是 SAmple aggreGatE for Graph

# 在 GraphSAGE 之前的 GCN 模型中，都是采用的全图的训练方式，也就是说每一轮的迭代都要对全图的节点进行更新，当图的规模很大时，
# 这种训练方式无疑是很耗时甚至无法更新的。mini-batch 的训练时深度学习一个非常重要的特点，那么能否将 mini-batch 的思想用到 GraphSAGE 中呢，
# GraphSAGE 提出了一个解决方案。它的流程大致分为3步：
# 1. 对邻居进行随机采样，每一跳抽样的邻居数不多于Sk个；
# 2. 生成目标节点的 embedding：先聚合二跳邻居的特征，生成一跳邻居的embedding，再聚合一跳的 embedding，生成目标节点的 embedding；
# 3. 将目标节点的 embedding 输入全连接网络得到目标节点的预测值。

# 下面我们介绍一下 GraphSAGE 的代码实现，使用的是 DGL 框架（这里我们又引入 DGL 这个框架是因为 PyG 和 DGL 现在都被广泛使用，
# 读者应当对这两个框架都有所了解）。我们用 link prediction 作为模型的任务来举例。我们先简单的介绍一下链接预测这个任务。
# 许多应用，如社交推荐、项目推荐、知识图谱补全等，都可以表述为链接预测，即预测两个特定节点之间是否存在边。


# 导入相关的库
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools  # 迭代工具
import numpy as np
import scipy.sparse as sp  # 稀疏矩阵

# 导入Cora数据集
import dgl.data

dataset = dgl.data.CoraGraphDataset()  # 加载 Cora 数据集。
g = dataset[0]  # 获取数据集中的第一个图。

# 准备training set 和 testing set
u, v = g.edges()  #  返回图中所有边的起始节点和终止节点，分别存储在 u 和 v 中。

eids = np.arange(g.number_of_edges())  # 生成一个数组，包含从 0 到图中边数减一的所有整数。
eids = np.random.permutation(eids)  # 打乱数组 eids 的顺序。
test_size = int(len(eids) * 0.1)  # 计算测试集的大小，为总边数的 10%。
train_size = g.number_of_edges() - test_size  # 计算训练集的大小，为总边数减去测试集的大小。
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]  # 测试集中的正样本边。
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]  # 训练集中的正样本边。

# 分离负样本
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))  # 创建一个稀疏矩阵，表示图的邻接矩阵。
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())  # 计算负样本的邻接矩阵，1 表示没有边，0 表示有边。

# adj.todense()：将稀疏矩阵转换为密集矩阵。
# np.eye(g.number_of_nodes())：生成一个单位矩阵，大小为节点数，用于去除自环（节点与自身的边）。

neg_u, neg_v = np.where(adj_neg != 0)  # 找到邻接矩阵中不为 0 的元素的索引，即不存在边的节点对。

neg_eids = np.random.choice(len(neg_u), g.number_of_edges())  # 随机选择与正样本数量相同的负样本
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]  # 测试集中的负样本边。
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]  # 训练集中的负样本边。

# 训练时，您需要从原始图中删除测试集中的边。您可以通过 dgl.remove_edges 来完成此操作。
# dgl.remove_edges 的工作原理是从原始图创建子图，从而生成副本，因此对于大型图来说可能会很慢。
# 如果是这样，您可以将训练和测试图保存到磁盘，就像预处理一样。

train_g = dgl.remove_edges(g, eids[:test_size])

# 下面我们正式定义一个GraphSAGE模型：

from dgl.nn import SAGEConv


# 构建一个两层的 GraphSAGE 模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):  # in_feats：输入特征的维度，即每个节点的特征向量的长度。
        # h_feats：隐藏层特征的维度，即中间层的节点特征向量的长度。
        super(GraphSAGE, self).__init__()  # 调用父类 nn.Module 的构造函数，初始化模块。
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):  # g：输入的图，in_feat：输入的节点特征矩阵。
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# 然后，该模型通过计算两个节点的表示之间的得分来预测边缘存在的概率，通常通过一层MLP或者直接计算点积。

# 构建正样本和负样本的图
train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

# 构建上面提到的预测函数，如点积和MLP，即 DotPredictor 和 MLPPredictor:

import dgl.function as fn


class DotPredictor(nn.Module):  # DotPredictor 类继承自 nn.Module，用于通过节点特征的点积来预测边的分数。
    def forward(self, g, h):  # g：输入的图对象。h：节点特征矩阵。
        with g.local_scope():  # 在局部作用域内操作图，确保不改变原始图的状态。
            g.ndata["h"] = h  # 将节点特征赋值给图的节点数据。
            # 通过点积计算一个新的边的分数
            g.apply_edges(fn.u_dot_v("h", "h", "score"))  # 对每条边应用点积操作，计算边的分数。
            # u_dot_v 返回了一个 1-element 的向量，所以需要压平它
            return g.edata["score"][:, 0]  # 返回边的分数，压平为一维。


class MLPPredictor(nn.Module):  # MLPPredictor 类继承自 nn.Module，使用多层感知机（MLP）来预测边的分数。
    def __init__(self, h_feats):  # h_feats：隐藏层特征的维度，即中间层的节点特征向量的长度。
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        # h_feats * 2：表示输入特征的维度。因为在 MLPPredictor 中，
        # 我们将每条边的源节点和目标节点的特征拼接在一起，所以输入的特征维度是 h_feats 的两倍。
        # h_feats：表示输出特征的维度。经过这一层线性变换后，特征的维度被压缩回 h_feats。
        self.W2 = nn.Linear(h_feats, 1)  # 定义了两层线性变换，用于 MLP 的前向传播。
        # h_feats：表示输入特征的维度。这里的输入是经过第一层线性变换和激活函数处理后的特征。
        # 1：表示输出特征的维度。因为我们希望预测每条边的一个标量分数，所以输出维度为 1。

    def apply_edges(self, edges):  # 方法计算每条边的分数：edges：包含源节点、目标节点和边的特征。
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)  # 将源节点和目标节点的特征拼接在一起。
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}  # 通过 MLP 计算边的分数，并压平为一维。

    def forward(self, g, h):  # 方法计算每条边的分数：edges：包含源节点、目标节点和边的特征。
        with g.local_scope():  # 在局部作用域内操作图，确保不改变原始图的状态。
            g.ndata["h"] = h  # 将节点特征赋值给图的节点数据。
            g.apply_edges(self.apply_edges)  # 对每条边应用 apply_edges 方法。
            return g.edata["score"]  # 返回边的分数。


# 下面展示整个任务的训练过程

model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
# train_g.ndata['feat'].shape[1]：获取训练图中节点特征的维度，作为 GraphSAGE 模型的输入特征维度。
# 16：隐藏层特征的维度，表示中间层节点特征的长度。

# You can replace DotPredictor with MLPPredictor.
# 可以用 MLPPredictor(16) 替换 DotPredictor，如果想使用多层感知机（MLP）来预测边的分数。
# pred = MLPPredictor(16)
pred = DotPredictor()


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])  # 将正负样本的分数拼接在一起。
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    # 创建标签，正样本为 1，负样本为 0。
    return F.binary_cross_entropy_with_logits(scores, labels)
    # F.binary_cross_entropy_with_logits：计算二元交叉熵损失。


from sklearn.metrics import roc_auc_score


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)  # 计算 ROC 曲线下面积，评估模型的分类性能。
    # 分别是拼接后的预测分数和标签，转换为 NumPy 数组以便使用 roc_auc_score。


# 定义优化器
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
# 将 model 和 pred 的参数组合在一起，以便同时更新。

# 训练
all_logits = []  # 初始化一个空列表，用于存储每个 epoch 的输出。
for e in range(100):
    # 前向传播
    h = model(train_g, train_g.ndata["feat"])  # 通过 GraphSAGE 模型计算节点的隐藏表示。
    pos_score = pred(train_pos_g, h)  # 通过预测器计算正样本的分数。
    neg_score = pred(train_neg_g, h)  # 通过预测器计算负样本的分数。
    loss = compute_loss(pos_score, neg_score)  # 计算损失。

    # 更新参数
    optimizer.zero_grad()  # 清除上一步的梯度。
    loss.backward()  # 反向传播，计算梯度。
    optimizer.step()  # 更新模型参数。

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# 计算AUC
with torch.no_grad():  # 在评估阶段禁用梯度计算，以节省内存和计算资源。
    pos_score = pred(test_pos_g, h)  # 计算测试集正样本边的预测分数。
    neg_score = pred(test_neg_g, h)  # 计算测试集负样本边的预测分数。
    print("AUC", compute_auc(pos_score, neg_score))  # 计算并打印测试集的 AUC 值，评估模型的分类性能。
