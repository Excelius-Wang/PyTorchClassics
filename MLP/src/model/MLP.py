import torch.nn as nn


# ******************** 模型定义 ********************
class MLPModel(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_classes,
            layer_num=3,
            use_dropout=False,
            dropout_rate=0.2,
            use_batch_norm=False
    ):
        super(MLPModel, self).__init__()

        # 保存参数配置
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # 单独设置输入层
        self.fc_in = nn.Linear(input_size, hidden_size)

        # 批归一化层（如果启用）
        self.bn_in = nn.BatchNorm1d(hidden_size) if use_batch_norm else None

        # 输入层激活函数
        self.relu_in = nn.ReLU()

        # Dropout层（如果启用）
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None

        # 根据 layer_num 参数决定隐藏层的数量, 默认为3-2=1层
        self.fc_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None

        for _ in range(layer_num - 2):
            # 加入隐藏层
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            # 加入激活函数
            self.relu_layers.append(nn.ReLU())
            # 加入批归一化层（如果启用）
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_size))

        # 单独设置输出层
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 输入层
        x = self.fc_in(x)  # [batch_size, 784] -> [batch_size, 256]

        # 使用顺序：先线性变换，再批归一化，然后激活函数，最后Dropout，定义顺序也可以按照这个来
        # 批归一化（如果启用）
        if self.use_batch_norm and self.bn_in is not None:
            x = self.bn_in(x)

        x = self.relu_in(x)  # 应用激活函数

        # Dropout（如果启用）
        if self.use_dropout and self.dropout is not None:
            x = self.dropout(x)

        # 通过所有隐藏层
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)

            # 批归一化（如果启用）
            if self.use_batch_norm and self.bn_layers is not None:
                x = self.bn_layers[i](x)

            x = self.relu_layers[i](x)

            # Dropout（如果启用）
            if self.use_dropout and self.dropout is not None:
                x = self.dropout(x)

        x = self.fc_out(x)  # [batch_size, 256] -> [batch_size, 10]
        return x  # 注：CrossEntropyLoss自带Softmax，此处不需要使用 Softmax
