import numpy as np
from snake_env import SnakeEnv
import pygame

def collect_expert_data():
    """手动玩贪吃蛇，收集「状态-动作」专家数据"""
    env = SnakeEnv()
    states = []  # 存储状态
    actions = []  # 存储对应动作
    state = env.reset()  # 重置游戏获取初始状态

    print("=== 专家数据收集 ====")
    print("操作说明：方向键控制蛇移动，Q键保存数据并退出，ESC键直接退出")
    print("建议收集500+条数据（玩5-10局），确保覆盖避障、吃食物等场景")
    epochs = 8
    while True:
        env.render()  # 显示游戏画面

        # 监听键盘事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                # ESC键：退出不保存
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                # Q键：保存数据并退出
                # if event.key == pygame.K_q:
                #
                #     pygame.quit()
                #     return
                # 方向键：映射为动作（0=上，1=下，2=左，3=右）
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                else:
                    continue  # 其他按键忽略

                # 记录状态和动作，执行下一步
                states.append(state)
                actions.append(action)
                state, _, done = env.step(action)

                # 游戏结束则重置
                if done:
                    np.savez(f"expert_data{epochs}.npz", states=np.array(states), actions=np.array(actions))
                    print(f"\n数据保存成功！共收集 {len(states)} 条样本")
                    print(f"游戏结束，重新开始...")
                    state = env.reset()
                    epochs += 1
                    done = False

if __name__ == "__main__":
    collect_expert_data()