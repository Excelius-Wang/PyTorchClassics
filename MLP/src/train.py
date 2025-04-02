import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import datetime
from utils.plots import plot_training_metrics

from model.MLP import MLPModel
from utils.data_loader import get_dataloaders
from utils.logger import printlog, save_log

# ******************** 环境配置 ********************
# 设备选择（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
printlog(f"Using Device: {device}")


# ******************** 工具函数 ********************
def evaluate(model, data_loader, desc="Testing: "):
    """模型评估函数（优化内存使用版本）
    功能：
        - 在指定数据集上计算模型精度
        - 禁用梯度计算以节省内存
    返回：
        - 分类准确率（百分比形式）和损失值
    """
    # 切换到评估模式（关闭Dropout/BatchNorm）
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0

    # 添加进度条，注意设置更窄的宽度以确保显示完整
    progress_bar = tqdm(data_loader, desc=desc, ncols=100, leave=False)

    # 禁用梯度计算（减少内存消耗）
    with torch.no_grad():
        for images, labels in progress_bar:
            # 数据预处理
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # 前向传播获取预测结果
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算损失
            total_loss += loss.item() * images.size(0)

            # 获取预测类别
            _, predicted = torch.max(outputs.data, 1)

            # 统计正确预测数
            batch_size = labels.size(0)
            total += batch_size
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct

            # 实时更新进度条显示当前准确率(精简显示)
            current_acc = 100 * correct / total
            progress_bar.set_postfix({'acc': f'{current_acc:.2f}%'})

    # 返回百分比精度和平均损失
    return 100 * correct / total, total_loss / total


# ******************** 训练流程（完整封装）********************
def main(_config):
    """主训练流程

    Args:
        _config: 配置字典，包含所有超参数
    """
    # 数据加载
    train_loader, test_loader = get_dataloaders(_config)

    # 模型初始化
    model = MLPModel(input_size=784,
                hidden_size=_config["hidden_dims"],
                num_classes=10,
                layer_num=_config["layer_num"],
                use_dropout=_config['use_dropout'],
                dropout_rate=_config['dropout_rate'],
                use_batch_norm=_config['use_batch_norm']).to(device)

    # 模型参数数量统计
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 1e9:
        printlog(f"Total Parameters: {total_params/1e9:.2f}B")
    elif total_params >= 1e6:
        printlog(f"Total Parameters: {total_params/1e6:.2f}M")
    else:
        printlog(f"Total Parameters: {total_params:,}")

    # 优化器配置(AdamW：带权重衰减修正的Adam)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_config['learning_rate'],
        weight_decay=_config['weight_decay']  # 解耦权重衰减
    )

    # 学习率调度器(根据验证精度调整学习率)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.5,
        patience=2
    ) if _config['use_scheduler'] else None
    
    # 记录初始学习率
    last_lr = _config['learning_rate']

    # 损失函数(内置 Softmax)
    criterion = nn.CrossEntropyLoss()

    train_losses = []  # 记录每个epoch的训练损失
    test_losses = []   # 记录每个epoch的测试损失
    train_acc_list = []  # 记录训练准确率
    test_acc_list = [] # 记录测试准确率
    best_acc = 0.0     # 记录最佳测试精度
    
    # 使用AverageMeter跟踪训练指标
    class AverageMeter:
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    # 训练循环
    for epoch in range(_config['num_epochs']):
        model.train()  # 训练模式
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        # 更新描述信息，确保进度条宽度合适
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{_config['num_epochs']}", 
                           ncols=100, leave=True)

        for images, labels in progress_bar:
            # 数据预处理
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸，提升训练稳定性）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 更新参数
            optimizer.step()

            # 统计训练指标
            batch_size = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_acc = 100 * batch_correct / batch_size

            # 更新平均指标
            train_loss_meter.update(loss.item(), batch_size)
            train_acc_meter.update(batch_acc, batch_size)

            # 只显示关键指标，减少显示宽度
            progress_bar.set_postfix({
                'loss': f'{train_loss_meter.avg:.4f}',
                'acc': f'{train_acc_meter.avg:.2f}%'
            })

        # 记录训练损失
        train_losses.append(train_loss_meter.avg)
        # 记录训练准确率
        train_acc_list.append(train_acc_meter.avg)

        # 计算测试损失和准确率 (只需要一次评估)
        test_acc, test_loss = evaluate(model, test_loader, desc=f"Testing (Epoch {epoch+1})")
        test_acc_list.append(test_acc)
        test_losses.append(test_loss)

        # 学习率调整（基于测试集性能）
        if scheduler:
            # 根据测试准确率调整学习率
            scheduler.step(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存模型参数
            torch.save(model.state_dict(), "checkpoint/best_model.bin")
            printlog(f"✅ The model has been saved! New best accuracy: {best_acc:.2f}%")

        # 打印epoch结果
        epoch_summary = (f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss_meter.avg:.4f} | "
              f"Train Acc: {train_acc_meter.avg:.2f}% | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}"
              )

        # 记录到日志并打印到控制台
        save_log(epoch_summary)
        print(epoch_summary)

    # 输出模型最佳结果
    printlog(f"😍 Best Accuracy: {best_acc:.2f}%")

    # 调用绘图函数
    plot_training_metrics(train_losses, test_losses, train_acc_list, test_acc_list)


# 程序入口
if __name__ == "__main__":
    # 记录运行开始时间
    start_time = datetime.datetime.now()
    printlog(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ******************** 超参数配置字典 ********************
    config = {
        "seed": 42,  # 随机种子（保证可重复性）
        "batch_size": 512,  # 大批量提升训练速度
        "num_epochs": 15,  # 适当增加训练轮次
        "learning_rate": 1e-3,  # Adam优化器的典型学习率
        "weight_decay": 1e-4,  # 权重衰减（L2正则化系数）
        "hidden_dims": 256,  # 隐藏层维度配置（可灵活调整）
        "dropout_rate": 0.2,  # Dropout比例（防止过拟合）
        "layer_num": 5,  # MLP 层数
        "use_batch_norm": True,  # 是否使用批量归一化
        "use_dropout": True,  # 是否使用Dropout
        "use_scheduler": True  # 是否启用学习率调度
    }
    printlog(config)
    main(config)  # 执行训练流程
    
    # 记录运行结束时间
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    printlog(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    printlog(f"Total training time: {duration.total_seconds()/60:.2f} minutes")
