# 导入 networkx 包
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个图
g = nx.Graph()
# 添加图的节点
g.add_node(2)
g.add_node(5)
# 添加图的边
g.add_edge(2, 5)
g.add_edge(1, 4)  # 当添加的边对应的节点不存在的时候，会自动创建相应的节点
g.add_edge(1, 2)
g.add_edge(2, 6)
# 绘制图
nx.draw(g)

plt.show()

# 默认情况下，networkX 创建的是无向图
G = nx.Graph()
print(G.is_directed())

# 创建有向图
H = nx.DiGraph()
print(H.is_directed())
