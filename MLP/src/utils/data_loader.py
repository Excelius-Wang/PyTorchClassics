from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# ******************** 数据预处理与加载 ********************
def get_dataloaders(config):
    """高效数据加载器

    Args:
        config: 配置字典，包含batch_size等参数

    Returns:
        train_loader, test_loader: 训练集和测试集加载器
    """
    # 数据增强流水线
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为张量并归一化到[0,1]
    ])

    # 数据集加载（自动下载）
    train_set = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transform  # 应用定义的数据增强
    )

    test_set = datasets.MNIST(
        root='data',
        train=False,
        transform=transform
    )

    # 创建数据加载器（最佳实践配置）
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,  # 训练集需要打乱顺序
        num_workers=2,  # 多进程加速数据加载
        pin_memory=True  # 锁页内存加速GPU传输
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,  # 测试集无需打乱
        num_workers=2
    )

    return train_loader, test_loader
