# 我们将使用 Institute for Applied Informatics and Formal Description Methods (AIFB) dataset 数据集
# 并通过 DGL 框架来完成一个实体分类任务。实体分类的目标是为实体分配类型和分类属性。
# 我们可以通过在实体（节点）的最终嵌入处附加 softmax 分类器来完成的，其训练是通过损失标准交叉熵来进行的。

# 加载数据库和一些必要的库

import os

os.environ["DGLBACKEND"] = "pytorch"
from functools import partial

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

# 加载图数据
dataset = dgl.data.rdf.AIFBDataset()  # 加载 AIFB 数据集，这是一个 RDF 数据集，常用于知识图谱任务。
g = dataset[0]
category = dataset.predict_category
train_mask = g.nodes[category].data.pop("train_mask")  # 从图中指定类别的节点数据中提取并移除训练掩码。
test_mask = g.nodes[category].data.pop("test_mask")
train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()  # 获取训练掩码中非零元素的索引，并将其压缩为一维。
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
# 当 as_tuple=False 时，torch.nonzero() 返回一个二维张量，其中每一行表示一个非零元素的索引。
# 当 as_tuple=True 时，torch.nonzero() 返回一个元组，其中每个元素是一个一维张量，表示非零元素在每个维度上的索引。
# squeeze() 函数用于去除返回张量中大小为 1 的维度，将其压缩为一维张量，以便后续处理。
labels = g.nodes[category].data.pop("label")
num_rels = len(g.canonical_etypes)  # 获取图中所有关系类型的数量。
num_classes = dataset.num_classes
# 归一化因子
for cetype in g.canonical_etypes:  # 遍历图中所有的关系类型。
    g.edges[cetype].data["norm"] = dgl.norm_by_dst(g, cetype).unsqueeze(1)  # 计算以目标节点为基础的归一化因子。
# unsqueeze(1)：将归一化因子扩展为二维，以便与边数据的形状匹配。
# g.edges[cetype].data["norm"]：将计算的归一化因子存储在边数据中。
category_id = g.ntypes.index(category)

# 定义 R-GCN 模型


# 定义 R-GCN 层
class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,  # 关系类型的数量。
        num_bases=-1,  # `num_bases`：基数，用于参数共享，默认为 -1 表示不使用。
        bias=None,
        activation=None,
        is_input_layer=False,
    ):
        super(RGCNLayer, self).__init__()  # 调用父类 nn.Module 的构造函数，初始化模块。
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check合理性检验
        """
        # 参数共享：
        # 当 num_bases 小于 num_rels 时，模型会使用 num_bases 个基础矩阵，
        # 通过线性组合生成 num_rels 个关系特定的权重矩阵。这种方式可以减少参数数量，特别是在关系类型很多的情况下。
        如果 num_bases 小于等于 0，表示没有基础矩阵，这在逻辑上是无效的，因为至少需要一个基础矩阵。
        如果 num_bases 大于 num_rels，则没有必要，因为每个关系类型都可以有自己独立的权重矩阵，不需要共享。
        最大化灵活性：
        当 num_bases 设置为 num_rels 时，每个关系类型都有自己独立的权重矩阵。这种配置在关系类型较少时是可行的，因为参数数量不会过多。
        # 确保模型的有效性：
        # 通过将 num_bases 设置为 num_rels，可以确保即使在没有参数共享的情况下，模型也能正常工作。

        """
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        # weight bases in equation (3)
        self.weight = nn.Parameter(  # 将一个张量注册为模型的参数。
            torch.Tensor(self.num_bases, self.in_feat, self.out_feat)  # 创建一个三维张量，用于存储权重矩阵。
        )
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(  # 则创建一个用于线性组合的系数矩阵 w_comp。
                torch.Tensor(self.num_rels, self.num_bases)
            )
        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))  # 如果使用偏置，则创建一个偏置向量。
        # init trainable parameters
        nn.init.xavier_uniform_(  # 使用 Xavier 均匀分布初始化权重矩阵，gain 参数用于调整初始化的范围。
            self.weight, gain=nn.init.calculate_gain("relu")
        )
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain("relu"))
        if self.bias:
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain("relu"))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)  # 改变张量的形状。
            weight = torch.matmul(self.w_comp, weight).view(  # torch.matmul：矩阵乘法。
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight
        if self.is_input_layer:

            def message_func(edges):  # 用于计算消息传递。
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                #  将 weight 重塑为一个二维张量，其中 -1 表示自动计算该维度的大小，以确保总元素数量不变。
                # 结果是一个形状为 (num_bases * in_feat, out_feat) 的二维张量 embed，每一行对应一个特征向量的输出维度。
                index = edges.data[dgl.ETYPE] * self.in_feat + edges.src["id"]
                # 计算索引 index，它将边类型和源节点 ID 结合起来，形成一个唯一的索引。
                # 结果是一个形状为 (num_rels * in_feat,) 的一维张量 index，其中每个元素表示一个特征向量的输出维度。
                # 计算每条边的索引 index，用于从 embed 中查找对应的嵌入。
                # edges.data[dgl.ETYPE] * self.in_feat 计算边类型在 embed 中的起始位置。
                # + edges.src["id"] 将源节点 ID 加到起始位置上，得到完整的索引。
                return {"msg": embed[index] * edges.data["norm"]}  # 返回一个字典
                # embed[index]：
                # 使用计算得到的 index 从 embed 中查找对应的嵌入向量。
                # * edges.data["norm"]：
                # edges.data["norm"] 是边的归一化因子，用于调整消息的大小。
                # 将嵌入向量与归一化因子相乘，得到最终的消息。

        else:  # 非输入层

            def message_func(edges):
                w = weight[edges.data[dgl.ETYPE]]
                msg = torch.bmm(edges.src["h"].unsqueeze(1), w).squeeze()
                # edges.src["h"] 获取每条边的源节点特征。
                # unsqueeze(1) 将源节点特征扩展为三维，以便进行批量矩阵乘法。
                # torch.bmm 是批量矩阵乘法，将源节点特征与权重矩阵 w 相乘，计算消息。
                # squeeze() 去除多余的维度，将结果压缩为二维。
                msg = msg * edges.data["norm"]
                # edges.data["norm"] 是边的归一化因子。
                # 将消息与归一化因子相乘，调整消息的大小。
                return {"msg": msg}

        def apply_func(nodes):
            h = nodes.data["h"]  # 获取节点的特征数据。
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {"h": h}

        g.update_all(message_func, fn.sum(msg="msg", out="h"), apply_func)
        # g.update_all：
        # 在图上执行消息传递和节点更新。
        # message_func：
        # 定义如何计算每条边的消息。
        # fn.sum(msg="msg", out="h")：
        # 对所有传入的消息进行求和聚合，将结果存储在节点数据 "h" 中。
        # apply_func：
        # 定义如何更新节点特征。


# 定义完整的 R-GCN 模型
class Model(nn.Module):
    def __init__(
        self,
        num_nodes,
        h_dim,  # 隐藏层特征维度。
        out_dim,
        num_rels,
        num_bases=-1,
        num_hidden_layers=1,
    ):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()  # 初始化节点特征

    def build_model(self):  # 创建 R-GCN 的各层
        self.layers = nn.ModuleList()  # 使用 nn.ModuleList 存储各层，自动注册子模块，便于管理和更新参数。
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)  # 将创建的输入层 i2h 添加到 self.layers 中
        # hidden to hidden
        for _ in range(self.num_hidden_layers):  # 使用循环创建多个隐藏层
            h2h = self.build_hidden_layer()  # 隐藏层的作用是进一步处理和提取特征
            self.layers.append(h2h)
        # hidden to output输出层的作用是将隐藏特征映射到输出特征空间，通常用于分类或回归任务
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)  # 生成从 0 到 num_nodes-1 的整数序列。
        return features

    def build_input_layer(self):
        return RGCNLayer(
            self.num_nodes,
            self.h_dim,
            self.num_rels,
            self.num_bases,
            activation=F.relu,
            is_input_layer=True,
        )

    def build_hidden_layer(self):
        return RGCNLayer(
            self.h_dim,
            self.h_dim,
            self.num_rels,
            self.num_bases,
            activation=F.relu,
        )

    """
    partial 是 Python 标准库 functools 模块中的一个函数。
    它用于创建一个新的函数，这个新函数是对原函数的部分应用，即预设了一些参数。
    在这里，partial(F.softmax, dim=1) 创建了一个新的函数，这个函数是 F.softmax 的一个特例，其中 dim=1 已经被固定。
    通过 partial 预设 dim=1，可以简化代码，使得在调用激活函数时不需要每次都指定 dim 参数。
    dim=1 指定了 softmax 操作的维度。
    在多分类任务中，通常对输出的最后一个维度（即类别维度）进行 softmax 操作，以便将每个样本的输出转换为概率分布。
    """

    def build_output_layer(self):
        return RGCNLayer(
            self.h_dim,
            self.out_dim,
            self.num_rels,
            self.num_bases,
            activation=partial(F.softmax, dim=1),
        )

    def forward(self, g):
        if self.features is not None:
            g.ndata["id"] = self.features
            # 将节点特征赋值给图的节点数据，键为 "id"。
            # g.ndata 是 DGLGraph 中用于存储节点数据的字典，"id" 是用于标识节点特征的键。
        for layer in self.layers:
            layer(g)
        return g.ndata.pop("h")

    # "h" 是在每一层中更新的节点特征，最终的输出是经过所有层处理后的节点特征。
    # pop 方法不仅返回特征，还会从节点数据中移除该键，确保图数据的整洁。


# 配置超参数并实例化模型。
# 配置参数
n_hidden = 16  # number of hidden units
n_bases = -1  # use number of relations as number of bases
n_hidden_layers = 0  # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25  # epochs to train
lr = 0.01  # learning rate
l2norm = 0  # L2 norm coefficient

# 创建图
g = dgl.to_homogeneous(g, edata=["norm"])
node_ids = torch.arange(g.num_nodes())
target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]

# node_ids:
# node_ids 是一个张量，包含图中所有节点的索引。
# 通常通过 torch.arange(g.num_nodes()) 创建，表示从 0 到 g.num_nodes() - 1 的整数序列。
# g.ndata[dgl.NTYPE]:
# g.ndata 是 DGLGraph 中用于存储节点数据的字典。
# dgl.NTYPE 是 DGL 中用于标识节点类型的键。
# g.ndata[dgl.NTYPE] 返回一个张量，表示每个节点的类型。
# g.ndata[dgl.NTYPE] == category_id:
# category_id 是目标类别的索引。
# g.ndata[dgl.NTYPE] == category_id 返回一个布尔张量，表示哪些节点的类型与 category_id 匹配。
# node_ids[g.ndata[dgl.NTYPE] == category_id]:
# 使用布尔索引从 node_ids 中选择与 category_id 匹配的节点索引。
# 结果是一个张量，包含所有属于目标类别的节点的索引。


# 创建模型
model = Model(
    g.num_nodes(),
    n_hidden,
    num_classes,
    num_rels,
    num_bases=n_bases,
    num_hidden_layers=n_hidden_layers,
)

# 配置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
# weight_decay 是权重衰减系数，通常用于 L2 正则化。
# L2 正则化通过在损失函数中添加参数的平方和来防止过拟合。
# l2norm 是一个变量，表示 L2 正则化的强度，通常在代码的其他地方定义。

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    logits = model.forward(g)
    logits = logits[target_idx]
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    loss.backward()

    optimizer.step()

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    # logits[train_idx] 提取训练集节点的输出。
    # argmax(dim=1) 返回指定维度上最大值的索引。在这里，它返回每个节点预测类别的索引。
    # labels[train_idx] 是训练集节点的真实类别。
    # torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]) 计算预测正确的数量。
    val_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    val_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx])
    val_acc = val_acc.item() / len(test_idx)
    # .item() 将单元素张量转换为 Python 数值。预测正确的节点数量除以总节点数量。
    print(
        "Epoch {:05d} | ".format(epoch)
        + "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(train_acc, loss.item())
        + "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(val_acc, val_loss.item())
    )
