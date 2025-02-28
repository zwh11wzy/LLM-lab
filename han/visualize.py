import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model_hetero import HAN  # 替换为你的模型文件名
from utils import load_acm  # 替换为你的数据加载函数


def visualize_results(logits, labels):
    _, predicted = torch.max(logits, dim=1)
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(logits[:, 0].cpu().numpy(), logits[:, 1].cpu().numpy(), c=predicted, cmap="tab10", s=50)
    plt.colorbar(scatter, label="Classes")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.title("Model Predictions Visualization")
    plt.grid()
    plt.show()


def main():
    # 加载数据
    data = load_acm(remove_self_loop=False)  # 确保这个函数返回正确的数据格式

    # 初始化模型
    model = HAN(
        num_meta_paths=2, in_size=data.num_node_features, hidden_size=8, out_size=7, num_heads=[8, 1], dropout=0.5
    )

    # 加载训练好的模型
    model.load_state_dict(torch.load("early_stop_2025-02-28_17-53-57.pth"))
    model.eval()  # 设置模型为评估模式

    # 获取测试数据
    test_logits = model(data)  # 确保这里传入的是正确的测试数据
    test_labels = data.y  # 假设标签在数据中

    # 可视化结果
    visualize_results(test_logits, test_labels)


if __name__ == "__main__":
    main()
