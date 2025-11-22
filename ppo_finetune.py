import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from snake_env import SnakeEnv
from train_bc import BCPolicy  # 复用 BC 策略网络
from helper import plot

class PPO:
    def __init__(self, state_dim=12, action_dim=4, lr=1e-4, gamma=0.99, clip_epsilon=0.2, model="bc_policy_dagger.pth"):
        # 策略网络（加载 DAGGER 预训练权重）
        self.policy = BCPolicy(state_dim, action_dim)
        self.action_dim = action_dim
        # 动作与方向映射（关键：用于计算蛇头新位置）
        self.action_to_dir = [
            (0, -20),   # 0=上：y轴-20（假设格子大小20px）
            (0, 20),    # 1=下：y轴+20
            (-20, 0),   # 2=左：x轴-20
            (20, 0)     # 3=右：x轴+20
        ]
        # 加载模型
        try:
            self.policy.load_state_dict(torch.load(model))
            print("✅ 成功加载预训练模型")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            print("尝试重新初始化模型")
            self.policy = BCPolicy()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # PPO 核心参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.batch_size = 64
        self.epochs = 10

    def get_action(self, state, snake_head_pos, history_positions):
        """
        获取动作（新增：重复路径检测+动作替换）
        :param state: 环境状态
        :param snake_head_pos: 当前蛇头坐标 (x, y)
        :param history_positions: 蛇头已走过的历史位置列表
        :return: 有效动作（不重复路径）、动作概率
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_tensor)
            logits = torch.clamp(logits, min=-10.0, max=10.0)  # 数值裁剪
            temperature = 1.0
            logits = logits / temperature
            action_probs = torch.softmax(logits, dim=1)
            action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True) + 1e-8

            # 步骤1：获取模型预测的动作（按概率排序）
            sorted_actions = torch.argsort(action_probs, dim=1, descending=True).squeeze(0).tolist()
            best_action = sorted_actions[0]

            # 步骤2：预计算执行最佳动作后的蛇头新位置
            dx, dy = self.action_to_dir[best_action]
            new_head_pos = (snake_head_pos[0] + dx, snake_head_pos[1] + dy)

            # 步骤3：重复路径检测（关键逻辑）
            # 判定条件：新位置在历史轨迹中，且不是蛇尾即将离开的位置（避免误判正常移动）
            is_repeat = new_head_pos in history_positions[:-1]  # 排除最后一个位置（蛇尾要离开）
            valid_actions = []

            if not is_repeat:
                # 无重复路径：直接用最佳动作
                valid_actions = [best_action]
            else:
                # 有重复路径：过滤所有会导致重复的动作
                # print(f"⚠️  检测到重复路径！当前位置：{snake_head_pos}，预测位置：{new_head_pos}")
                for action in sorted_actions:
                    dx, dy = self.action_to_dir[action]
                    candidate_pos = (snake_head_pos[0] + dx, snake_head_pos[1] + dy)
                    # 筛选：新位置不在历史轨迹，且在游戏边界内（400x400窗口）
                    if (candidate_pos not in history_positions[:-1] and
                        0 <= candidate_pos[0] < 400 and
                        0 <= candidate_pos[1] < 400):
                        valid_actions.append(action)

            # 步骤4：选择最终动作（兜底逻辑）
            if valid_actions:
                # 从有效动作中选概率最高的（或随机选，增加探索）
                final_action = valid_actions[0]  # 选最优有效动作
                # final_action = np.random.choice(valid_actions)  # 随机选，探索更多路径
            else:
                # 极端情况：所有动作都重复/撞墙，用均匀分布兜底s
                # print("⚠️  无有效动作，均匀分布兜底")
                final_action = np.random.choice(self.action_dim)

            # 获取最终动作的概率
            final_action_prob = action_probs[0, final_action].item()

        return final_action, final_action_prob

    def compute_returns(self, rewards, dones):
        """计算累计折扣奖励"""
        returns = []
        discounted_return = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0.0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = torch.clamp(returns, min=-50.0, max=50.0)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train(self, trajectories):
        """用收集到的轨迹训练 PPO 策略"""
        states = torch.tensor([t[0] for t in trajectories], dtype=torch.float32)
        actions = torch.tensor([t[1] for t in trajectories], dtype=torch.long)
        old_probs = torch.tensor([t[2] for t in trajectories], dtype=torch.float32)
        rewards = [t[3] for t in trajectories]
        dones = [t[4] for t in trajectories]

        returns = self.compute_returns(rewards, dones)

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i + self.batch_size]
                batch_actions = actions[i:i + self.batch_size]
                batch_old_probs = old_probs[i:i + self.batch_size]
                batch_returns = returns[i:i + self.batch_size]

                logits = self.policy(batch_states)
                current_probs = torch.softmax(logits, dim=1)
                current_probs_selected = current_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                ratio = current_probs_selected / (batch_old_probs + 1e-8)
                surr1 = ratio * batch_returns
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_returns
                policy_loss = -torch.min(surr1, surr2).mean()

                self.optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()

        return policy_loss.item()


def ppo_finetune_main():
    """PPO 微调主流程（新增：记录蛇头历史位置）"""
    env = SnakeEnv()
    model_path = "model/ppo_finetuned_policy.pth"
    ppo_agent = PPO(model=model_path)

    total_timesteps = 500000
    timestep = 0
    max_trajectory_len = 500  # 单条轨迹最大步数
    max_history_len = 20  # 历史位置记录长度（只记最近10步，避免内存占用）
    avg_Food = 20

    plot_scores = []
    total_score = 0
    plot_mean_scores = []

    print("=== 开始 PPO 微调（含重复路径检测）===")
    print(f"总训练步数：{total_timesteps}")

    while timestep < total_timesteps:
        trajectory = []
        state = env.reset()
        done = False
        trajectory_reward = 0.0
        eat_food_num = 0
        env.eat_food_num = 0

        # 关键：记录蛇头的历史位置（初始为当前蛇头位置）
        snake_head_pos = tuple(env.snake[0])  # env.snake[0] 是蛇头坐标 (x, y)
        history_positions = [snake_head_pos]  # 历史位置列表

        idx = 0
        while idx < max_trajectory_len:
            # 步骤1：获取有效动作（避免重复路径）
            action, action_prob = ppo_agent.get_action(
                state=state,
                snake_head_pos=snake_head_pos,
                history_positions=history_positions
            )

            # 监控异常概率
            if np.isnan(action_prob) or np.isinf(action_prob) or action_prob < 0:
                print(f"⚠️  异常动作概率：{action_prob}，重置当前轨迹")
                done = True
                break

            # 步骤2：执行动作，更新环境
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, action_prob, reward, done))

            # 步骤3：更新蛇头位置和历史记录
            snake_head_pos = tuple(env.snake[0])  # 执行动作后的新蛇头位置
            history_positions.append(snake_head_pos)
            # 限制历史记录长度（只保留最近max_history_len步，避免误判早期路径）
            if len(history_positions) > max_history_len:
                history_positions.pop(0)

            # 步骤4：更新训练状态
            state = next_state
            trajectory_reward += reward
            timestep += 1
            idx += 1

            # 吃到食物时重置计数
            if env.eat_food_num != eat_food_num:
                eat_food_num = env.eat_food_num
                idx = 0

            if timestep >= total_timesteps or done:
                print(f"Timestep: {timestep:6d} | Reward: {trajectory_reward:4.1f} | Food: {eat_food_num} ")
                break

        # 过滤短轨迹，避免无效训练
        if len(trajectory) >= avg_Food:
            policy_loss = ppo_agent.train(trajectory)
        else:
            policy_loss = -1.0

        # 每1000步打印状态
        if timestep % 1000 == 0:
            print(
                f"Timestep: {timestep:6d} | Reward: {trajectory_reward:4.1f} | Food: {eat_food_num}"
            )

        if done:
            plot_scores.append(eat_food_num)
            total_score += eat_food_num
            mean_score = total_score / len(plot_scores)

            plot_mean_scores.append(mean_score)
            with open("config/config.json", "rb") as f:
                tmp = json.load(f)
                # print(f"tmp = {tmp}")
                if tmp["show"] == "True":
                    plot(plot_scores, plot_mean_scores)

    # 保存最终模型
    print(f"训练轮次： {len(plot_scores)}")
    torch.save(ppo_agent.policy.state_dict(), "ppo_finetuned_policy.pth")
    print("\nPPO 微调完成！模型保存为 ppo_finetuned_policy.pth")


if __name__ == "__main__":
    ppo_finetune_main()