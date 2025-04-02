import matplotlib.pyplot as plt


def plot_training_metrics(train_losses, test_losses, train_acc_list, test_acc_list, save_path='training_metrics.png'):
    """绘制训练和测试的损失与准确率曲线

    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表
        train_acc_list: 训练准确率列表
        test_acc_list: 测试准确率列表
        save_path: 图表保存路径
    """
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-o', label='Training Loss')
    plt.plot(test_losses, 'r-s', label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, 'b-o', label='Training Accuracy')
    plt.plot(test_acc_list, 'r-s', label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
