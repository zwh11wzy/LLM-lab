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

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# 准备training set 和 testing set
u, v = g.edges()

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# 分离负样本
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
