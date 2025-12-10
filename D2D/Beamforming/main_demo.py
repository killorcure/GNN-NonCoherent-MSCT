import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


# 生成简单的二分类数据
def generate_data(n_samples=100):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # 生成一个简单的异或问题
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)


# 用模型进行训练
def train_on_task(model, data, labels, lr=0.01, epochs=1):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model


# 用MAML的方式训练模型
def meta_train(model, meta_lr=0.001, task_lr=0.01, n_tasks=10, n_inner_steps=100):
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    for meta_epoch in range(10000):  # Meta-training loop
        meta_optimizer.zero_grad()
        meta_loss = 0

        for _ in range(n_tasks):
            # 每个任务
            # 1. 生成任务数据
            data, labels = generate_data(20)

            # 2. 对任务进行内循环训练
            model_copy = SimpleModel(2, 1)
            model_copy.load_state_dict(model.state_dict())  # 使用当前模型的权重作为初始化
            model_copy = train_on_task(model_copy, data, labels, lr=task_lr, epochs=n_inner_steps)

            # 3. 计算模型在该任务上的损失并累加
            outputs = model_copy(data)
            criterion = nn.BCELoss()
            task_loss = criterion(outputs, labels)
            meta_loss += task_loss

        # 4. 计算meta梯度并更新模型
        meta_loss /= n_tasks
        meta_loss.backward()
        meta_optimizer.step()

        if meta_epoch % 10 == 0:
            print(f"Meta-Epoch [{meta_epoch + 1}/100], Meta Loss: {meta_loss.item()}")


# 初始化模型
model = SimpleModel(2, 1)

# 使用MAML进行训练
meta_train(model, meta_lr=0.001, task_lr=0.01, n_tasks=10, n_inner_steps=5)
