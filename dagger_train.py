import torch
import numpy as np
from snake_env import SnakeEnv
from train_bc import BCPolicy, train_bc_model  # 复用BC模型和训练函数
import pygame

def expert_annotator(state, env):
    """
    专家标注器：只响应方向键和Q键，忽略所有鼠标事件
    功能：标注当前状态的最优动作，Q键确认，方向键选择
    """
    print("\n=== 专家标注 ===")
    print("操作说明：方向键选择动作 → Q键确认标注（仅响应字母Q和方向键，忽略鼠标）")
    action = None

    # 修复：渲染完整游戏状态（蛇身+食物），确保标注判断准确
    temp_screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("专家标注 - 仅响应Q键和方向键")

    # 渲染当前状态（完整蛇身+食物）
    def render_state():
        temp_screen.fill((0, 0, 0))  # 黑色背景
        # 画完整蛇身（而非仅蛇头）
        for segment in env.snake:
            pygame.draw.rect(temp_screen, (0, 255, 0), (segment[0], segment[1], 19, 19))
        # 画食物
        pygame.draw.rect(temp_screen, (255, 0, 0), (env.food[0], env.food[1], 19, 19))
        pygame.display.flip()  # 强制刷新画面

    render_state()

    while True:
        # 遍历所有事件，但只处理键盘事件，忽略鼠标事件
        for event in pygame.event.get():
            # 3. 仅响应键盘按下事件（KEYDOWN）
            # print(f"event.type11 = {event.type}")
            # print(f"pygame.KEYDOWN11 = {pygame.KEYDOWN}")
            if event.type == pygame.KEYDOWN:
                print(f"event.type = {event.type}")
                print(f"pygame.KEYDOWN = {pygame.KEYDOWN}")

                # 1. 忽略所有鼠标事件（直接跳过）
                if event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION]:
                    continue  # 鼠标事件不做任何处理

                # 2. 只处理键盘事件和窗口关闭事件
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit("用户关闭窗口，程序退出")

                # 调试输出：仅打印键盘事件关键信息
                print(f"键盘事件 - event.key={event.key}, Q键标准值={pygame.K_q}")

                # 只响应 Q键 和 方向键，其他字母/按键忽略
                if event.key == 49:  # 按1 确认 仅响应Q键（大小写兼容，因pygame.K_q包含小写）
                    if action is not None:
                        pygame.display.quit()
                        print(f"✅ 标注确认：动作={action}（0=上，1=下，2=左，3=右）")
                        return action
                    else:
                        print("❌ 请先按方向键选择动作！")
                elif event.key == pygame.K_UP:  # 上方向键
                    action = 0
                    print("当前选择：上（0）")
                    render_state()  # 选择后刷新画面
                elif event.key == pygame.K_DOWN:  # 下方向键
                    action = 1
                    print("当前选择：下（1）")
                    render_state()
                elif event.key == pygame.K_LEFT:  # 左方向键
                    action = 2
                    print("当前选择：左（2）")
                    render_state()
                elif event.key == pygame.K_RIGHT:  # 右方向键
                    action = 3
                    print("当前选择：右（3）")
                    render_state()
                else:
                    # 其他按键（如A、B、空格等）直接忽略，不打印、不响应
                    pass

            # 持续刷新画面（防止窗口卡死）
            render_state()
            pygame.time.Clock().tick(30)  # 30FPS确保事件响应流畅

def dagger_iteration(n_iter=3):
    """
    DAGGER迭代流程（修复状态记录错误，忽略鼠标事件）
    """
    # 加载初始专家数据（用户指定的 expert_data0.npz）
    try:
        data = np.load("expert_data0.npz")
        states = list(data["states"])
        actions = list(data["actions"])
        print(f"✅ 加载初始专家数据：{len(states)} 条样本")
        if len(states) < 100:
            print("⚠️  警告：初始数据量过少（建议≥500条），可能影响训练效果")
    except FileNotFoundError:
        print("❌ 错误：未找到 expert_data0.npz！请先运行 collect_data.py 收集初始专家数据")
        return

    # 初始化BC模型（首次训练前用初始数据训练）
    print("\n=== 首次训练初始BC模型 ===")
    train_bc_model()  # 复用训练函数
    model = BCPolicy()
    model.load_state_dict(torch.load("bc_policy.pth"))  # 加载初始训练后的模型
    model.eval()

    for iter in range(n_iter):
        print(f"\n=== DAGGER 迭代 {iter+1}/{n_iter} ===")
        new_states = []
        new_actions = []
        env = SnakeEnv()
        state = env.reset()
        done = False
        step_count = 0  # 限制最大步数，避免无限循环
        eat_food_num = 0

        while not done and step_count < 500:
            step_count += 1
            # 模型预测动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = model(state_tensor)
                model_action = torch.argmax(logits, dim=1).item()

            # 执行动作，获取下一个状态
            next_state, _, done = env.step(model_action)
            # 重置计数
            if env.eat_food_num != eat_food_num:
                step_count = 0
                eat_food_num = env.eat_food_num

            # 修复：记录模型决策时的原始状态（而非执行后的next_state），标注更准确
            new_states.append(state)

            # 专家标注（只响应Q键和方向键，忽略鼠标）
            expert_action = expert_annotator(state, env)
            new_actions.append(expert_action)

            # 更新状态
            state = next_state

        # 过滤无效数据（确保状态和动作数量一致）
        valid_len = min(len(new_states), len(new_actions))
        new_states = new_states[:valid_len]
        new_actions = new_actions[:valid_len]

        # 扩充数据集
        states.extend(new_states)
        actions.extend(new_actions)
        print(f"✅ 迭代{iter+1}完成：新增 {valid_len} 条数据，总数据量 {len(states)}")

        # 保存扩充后的数据集
        np.savez("expert_data_dagger.npz", states=np.array(states), actions=np.array(actions))
        np.savez(f"expert_data_dagger_iter{iter+1}.npz", states=np.array(states), actions=np.array(actions))

        # 重新训练BC模型
        print(f"\n=== 迭代{iter+1}：重新训练BC模型 ===")
        train_bc_model()

        # 加载重新训练后的模型（用于下一轮迭代）
        model.load_state_dict(torch.load("bc_policy.pth"))
        model.eval()

    print("\n🎉 DAGGER所有迭代完成！最终数据集：expert_data_dagger.npz，最终模型：bc_policy.pth")

if __name__ == "__main__":
    # 初始化Pygame（必须调用，否则键盘事件无法响应）
    pygame.init()
    try:
        dagger_iteration(n_iter=3)
    finally:
        # 程序结束后清理Pygame资源
        pygame.quit()
        print("🔚 程序正常退出，Pygame资源已释放")