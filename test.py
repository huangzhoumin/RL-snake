import json

import torch
import pygame
from snake_env import SnakeEnv
from train_bc import BCPolicy
from helper import plot

def test_ai(model_path="model/bc_policy_230.pth"):
    """加载训练好的BC模型，让AI自动玩贪吃蛇"""
    # 加载模型
    try:
        model = BCPolicy()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 切换到推理模式（禁用Dropout等）
        print(f"成功加载模型：{model_path}")
    except FileNotFoundError:
        print(f"错误：未找到{model_path}！请先运行train_bc.py训练模型")
        return

    print("\n=== AI开始游戏 ===")
    print("操作说明：按ESC键退出游戏")

    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    for idx in range(100):
        # 初始化游戏环境
        env = SnakeEnv()
        state = env.reset()
        done = False
        total_reward = 0  # 累计奖励
        foods = 0


        while not done:
            env.render()  # 显示AI游戏画面

            # 监听退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    print(f"\n游戏退出，累计奖励：{total_reward}")
                    return

            # AI预测动作（禁用梯度计算，提高速度）
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 扩展为batch维度
                logits = model(state_tensor)
                action = torch.argmax(logits, dim=1).item()  # 选择概率最大的动作

            # 执行动作，更新状态
            state, reward, done = env.step(action)
            total_reward += reward
            foods = env.eat_food_num

            if done:
                plot_scores.append(foods)
                total_score += foods
                mean_score = total_score / len(plot_scores)

                plot_mean_scores.append(mean_score)
                with open("config/config.json", "rb") as f:
                    tmp = json.load(f)
                    # print(f"tmp = {tmp}")
                    if tmp["show"] == "True":
                        plot(plot_scores, plot_mean_scores)



    # 游戏结束
    print(f"\n游戏结束！累计奖励：{total_reward}, 吃掉的食物数量：{foods}")
    pygame.quit()

if __name__ == "__main__":
    test_ai()