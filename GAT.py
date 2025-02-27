# 这里我们使用 DGL 来对 GAT 模型进行复现，在复现 GAT 模型的时候重点在于以下几点

# 首先对所有的节点特征进行Wh 变换
# 然后计算出每条边的注意力系数 e ij(即，节点 j 对节点 i 的权重)
# 之后对eij使用 softmax 操作进行归一化得到归一化的注意力系数
# 对邻居节点使用归一化后的注意力系数进行加权求和
# 重复上述过程得到多头注意力的多个结果，然后进行拼接或求和

# 我们首先来看实现单头注意力的代码


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
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)  # 计算edge attention的fc
        self.reset_parameters()  # 参数初始化

    """
    edges 的src表示源节点,dst表示目标节点
    """

    def edge_attention(self, edges):

        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)  # eq.1 里面的拼接操作
        a = self.attn_fc(z2)  # eq.1 里面对e_{ij}的计算
        return {
            "e": F.leaky_relu(a)
        }  # 这里的return实际上等价于 edges.data['e'] =  F.leaky_relu(a),这样后面才能将这个 e 传递给 源节点

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def message_func(self, edges):
        return {
            "z": edges.src["z"],
            "e": edges.data["e"],
        }  # 将前面 edge_attention 算出来的 e 以及 edges的源节点的 node embedding 都传给 nodes.mailbox

    def reduce_func(self, nodes):
        # 通过 message_func 之后即可得到 源节点的node embedding 与 edges 的 e
        alpha = F.softmax(nodes.mailbox["e"], dim=1)  # softmax归一化，得到 a
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)  # 根据a进行加权求和
        return {"h": h}

    def forward(self, h):
        z = self.fc(h)  # eq. 1 这里的h就是输入的node embedding
        self.g.ndata["z"] = z
        self.g.apply_edges(self.edge_attention)  # eq. 2
        self.g.update_all(self.message_func, self.reduce_func)  # eq. 3 and 4
        return self.g.ndata.pop("h")  # 返回经过attention的 node embedding
