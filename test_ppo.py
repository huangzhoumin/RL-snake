import json

import torch
import pygame
from snake_env import SnakeEnv
from train_bc import BCPolicy
from helper import plot

def test_ppo_ai():
    """测试 PPO 微调后的 AI 性能"""
    # 加载 PPO 微调后的模型
    try:
        model = BCPolicy()
        model.load_state_dict(torch.load("ppo_finetuned_policy.pth"))
        model.eval()
        print("成功加载 PPO 微调模型：ppo_finetuned_policy.pth")
    except FileNotFoundError:
        print("错误：未找到 ppo_finetuned_policy.pth！请先运行 ppo_finetune.py")
        return

    env = SnakeEnv()
    state = env.reset()
    done = False
    total_reward = 0
    print("\n=== PPO 微调 AI 开始游戏 ===")
    print("操作说明：按 ESC 键退出")
    epochs = 100
    for idx in range(epochs):
        plot_scores = []
        plot_mean_scores = []
        eat_food_num = 0

        step_count = 0  # 统计存活步数（更直观反映性能）

        while not done:
            env.render()
            step_count += 1

            # 监听退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    print(f"\n游戏退出 | 累计奖励：{total_reward} | 存活步数：{step_count} | 吃掉食物数量: {eat_food_num}")
                    return

            # AI 预测动作（PPO 微调后策略更优）
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()

            state, reward, done = env.step(action)
            total_reward += reward
            eat_food_num = env.eat_food_num
            plot_scores

        with open("config/config.json", "rb") as f:
            tmp = json.load(f)
            # print(f"tmp = {tmp}")
            if tmp["show"] == "True":
                plot(plot_scores, plot_mean_scores)

    # 游戏结束
    print(f"\n游戏结束 | 累计奖励：{total_reward} | 存活步数：{step_count} | 吃掉食物数量: {eat_food_num}")
    pygame.quit()

if __name__ == "__main__":
    test_ppo_ai()