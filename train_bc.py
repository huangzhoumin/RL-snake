import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义BC策略网络（简单MLP）
class BCPolicy(nn.Module):
    def __init__(self, state_dim=808, action_dim=4): #  state_dim 从12改成808（根据你的实际维度调整）
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入：12维状态
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)  # 输出：4个动作的logits
        )

    def forward(self, x):
        return self.net(x)

def load_all_npz_data(data_dir="data2"):
    """
    读取 data 目录下所有 .npz 文件，合并 states 和 actions 为 torch 张量
    :param data_dir: 数据目录路径（默认 "data"）
    :return: 合并后的 states (torch.float32)、actions (torch.long)
    """
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    # 存储所有文件的 states 和 actions（先存 numpy 数组，最后统一转张量，效率更高）
    all_states = []
    all_actions = []

    # 遍历 data 目录下所有 .npz 文件
    for filename in os.listdir(data_dir):
        # 只处理 .npz 后缀的文件
        if filename.endswith(".npz"):
            file_path = os.path.join(data_dir, filename)  # 完整文件路径
            print(f"正在加载：{file_path}")

            try:
                # 加载单个 npz 文件
                data = np.load(file_path, allow_pickle=True)  # allow_pickle 兼容旧版本数据

                # 提取 states 和 actions（确保键名正确，与你的文件一致）
                states = data["states"]
                actions = data["actions"]

                # 验证数据有效性（避免空数据或格式错误）
                if len(states) == 0 or len(actions) == 0:
                    print(f"⚠️  跳过空文件：{filename}")
                    continue
                if len(states) != len(actions):
                    print(f"⚠️  跳过数据长度不匹配的文件：{filename}（states: {len(states)}, actions: {len(actions)}）")
                    continue

                # 添加到列表
                all_states.append(states)
                all_actions.append(actions)
                print(f"✅ 加载成功：{filename}（数据量：{len(states)} 条）")

            except Exception as e:
                # 捕获单个文件加载错误，不影响整体流程
                print(f"❌ 加载文件失败：{filename}，错误：{str(e)}")
                continue

    # 检查是否加载到有效数据
    if not all_states or not all_actions:
        raise ValueError("未加载到任何有效数据！请检查 data 目录下的 .npz 文件")

    # 合并所有 numpy 数组（按行拼接，axis=0）
    merged_states_np = np.concatenate(all_states, axis=0)
    merged_actions_np = np.concatenate(all_actions, axis=0)

    # 转换为 torch 张量（匹配你的原始格式：states=float32，actions=long）
    merged_states = torch.tensor(merged_states_np, dtype=torch.float32)
    merged_actions = torch.tensor(merged_actions_np, dtype=torch.long)

    print(f"\n📊 数据合并完成！")
    print(f"总数据量：{len(merged_states)} 条")
    print(f"states 形状：{merged_states.shape}（维度：{merged_states.ndim}）")
    print(f"actions 形状：{merged_actions.shape}（维度：{merged_actions.ndim}）")

    return merged_states, merged_actions

def train_bc_model():
    """用收集的专家数据训练BC模型"""
    # 加载专家数据（确保已运行collect_data.py生成expert_data.npz）
    try:
        states, actions = load_all_npz_data()
        print(f"成功加载专家数据：{len(states)} 条样本")
    except FileNotFoundError:
        print("错误：未找到expert_data.npz！请先运行collect_data.py收集数据")
        return

    # 初始化模型、损失函数、优化器
    model = BCPolicy()
    criterion = nn.CrossEntropyLoss()  # 离散动作用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    # 训练参数
    epochs = 500  # 训练轮数
    batch_size = 32  # 批次大小

    print("\n=== 开始训练BC模型 ===")
    for epoch in range(epochs):
        total_loss = 0.0
        # 批次迭代训练
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]

            # 前向传播：预测动作
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)

            # 反向传播：更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_states)

        # 计算每轮平均损失
        avg_loss = total_loss / len(states)
        print(f"Epoch {epoch+1:2d}/{epochs} | Average Loss: {avg_loss:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "model/bc_policy_230.pth")
    print("\n模型训练完成！已保存为 model/bc_policy_230.pth")

if __name__ == "__main__":
    train_bc_model()