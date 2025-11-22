import pygame
import numpy as np
import random

# 初始化pygame
pygame.init()

# 游戏基础配置
BLOCK_SIZE = 20  # 方块大小
WIDTH = 400      # 窗口宽度
HEIGHT = 400     # 窗口高度
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Imitation Learning")

# 颜色定义
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # 蛇身颜色
BULE = (0, 0, 255)  # 蛇头颜色
RED = (255, 0, 0)    # 食物颜色

class SnakeEnv:
    def __init__(self):
        self.height = HEIGHT
        self.width = WIDTH
        self.eat_food_num = 0
        self.reset()  # 初始化游戏状态

    def reset(self):
        """重置游戏，返回初始状态"""
        # 蛇初始位置（头部在中心，身体向右延伸）
        self.snake = [
            (WIDTH//2, HEIGHT//2),
            (WIDTH//2 - BLOCK_SIZE, HEIGHT//2),
            (WIDTH//2 - 2*BLOCK_SIZE, HEIGHT//2)
        ]
        self.direction = (BLOCK_SIZE, 0)  # 初始方向：向右
        self.food = self._generate_food()  # 生成食物
        self.done = False  # 游戏结束标志
        return self._get_state()

    def _generate_food(self):
        """生成不与蛇身重叠的食物"""
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)


    def _get_state(self):
        """兼容原有12维state，追加蛇身位置特征（避免自撞）"""
        # -------------------------- 原有12维特征（完全保留你的代码）--------------------------
        # 食物方向特征（food_up/food_down/food_left/food_right：0=无，1=有）
        food_x, food_y = self.food
        head_x, head_y = self.snake[0]
        food_up = 1.0 if food_y < head_y else 0.0
        food_down = 1.0 if food_y > head_y else 0.0
        food_left = 1.0 if food_x < head_x else 0.0
        food_right = 1.0 if food_x > head_x else 0.0

        # 障碍物检测特征（obs_up/obs_down/obs_left/obs_right：0=无障碍，1=有障碍）
        grid_size = 20  # 你的格子大小（根据 snake_env.py 实际值修改）
        obs_up = 1.0 if (head_y - grid_size < 0) or ((head_x, head_y - grid_size) in self.snake) else 0.0
        obs_down = 1.0 if (head_y + grid_size >= self.height) or ((head_x, head_y + grid_size) in self.snake) else 0.0
        obs_left = 1.0 if (head_x - grid_size < 0) or ((head_x - grid_size, head_y) in self.snake) else 0.0
        obs_right = 1.0 if (head_x + grid_size >= self.width) or ((head_x + grid_size, head_y) in self.snake) else 0.0

        # 当前朝向特征（dir_up/dir_down/dir_left/dir_right：独热编码）
        dir_up = 1.0 if self.direction == 'UP' else 0.0
        dir_down = 1.0 if self.direction == 'DOWN' else 0.0
        dir_left = 1.0 if self.direction == 'LEFT' else 0.0
        dir_right = 1.0 if self.direction == 'RIGHT' else 0.0

        # 原有12维state
        base_state = [
            food_up, food_down, food_left, food_right,
            obs_up, obs_down, obs_left, obs_right,
            dir_up, dir_down, dir_left, dir_right
        ]

        # -------------------------- 新增蛇身位置特征（追加在后面）--------------------------
        body_features = []
        # max_body_segments = 18  # 最大蛇身段数（根据需求调整，比如蛇最大长度20，去掉蛇头剩18段）
        max_body_segments = 398  # 最大蛇身段数（根据需求调整，比如蛇最大长度20，去掉蛇头剩18段）

        body_segments = self.snake[1:]  # 蛇身（不含蛇头）

        for i, segment in enumerate(body_segments):
            if i >= max_body_segments:
                break  # 超过最大段数，不再追加（避免维度过高）
            seg_x, seg_y = segment
            # 计算蛇身相对蛇头的偏移（归一化到 [-1, 1]，和原有特征尺度一致）
            rel_x = (seg_x - head_x) / self.width  # self.width 是游戏窗口宽度（如400）
            rel_y = (seg_y - head_y) / self.height  # self.height 是游戏窗口高度（如400）
            body_features.extend([rel_x, rel_y])  # 每个蛇身段贡献2维特征

        # 补零到固定长度（确保状态维度统一，比如 12 + 18×2 = 48 维） # (12 + 398 * 2) = 808
        required_body_dim = max_body_segments * 2
        while len(body_features) < required_body_dim:
            body_features.append(0.0)

        # -------------------------- 合并最终state（原有12维 + 新增蛇身特征）--------------------------
        final_state = base_state + body_features
        return np.array(final_state, dtype=np.float32)

    # 这里reward 只有两种 -10 10
    def step(self, action):
        """执行动作，返回下一个状态、奖励、游戏结束标志"""
        # 动作映射：0=上，1=下，2=左，3=右
        action_map = [
            (0, -BLOCK_SIZE),  # 上
            (0, BLOCK_SIZE),   # 下
            (-BLOCK_SIZE, 0),  # 左
            (BLOCK_SIZE, 0)    # 右
        ]
        self.direction = action_map[action]

        # 移动蛇头
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        reward = 0
        # 检测碰撞（撞墙/撞自身）
        if (new_head[0] < 0 or new_head[0] >= WIDTH or
            new_head[1] < 0 or new_head[1] >= HEIGHT or
            new_head in self.snake):
            self.done = True
            return self._get_state(), reward-10, self.done  # 碰撞奖励-10

        reward += 0.1  # 存活奖励
        # 更新蛇身
        self.snake.insert(0, new_head)
        # 检测是否吃食物
        if new_head == self.food:
            self.food = self._generate_food()
            self.eat_food_num += 1
            return self._get_state(), reward+10, self.done  # 吃食物奖励+10
        else:
            self.snake.pop()  # 没吃食物(但是还存活)，移除蛇尾
            return self._get_state(), reward, self.done

    # 示例：在主流程中计算修正奖励
    def calc_reward(self, env, old_head_pos, new_head_pos, food_pos, is_repeat):
        old_dist = np.linalg.norm(np.array(old_head_pos) - np.array(food_pos))
        new_dist = np.linalg.norm(np.array(new_head_pos) - np.array(food_pos))
        reward = 0.0
        if env.eat_food_num > eat_food_num:
            reward += 20.0  # 吃食物奖励
        elif new_dist < old_dist:
            reward += 0.5  # 靠近食物奖励
        reward += 0.1  # 存活奖励
        if is_repeat:
            reward -= 1.0  # 重复路径惩罚
        if done:
            reward -= 10.0  # 死亡惩罚
        return reward

    def render(self):
        """绘制游戏画面"""
        SCREEN.fill(BLACK)
        # 画蛇身
        first = True
        for segment in self.snake:
            if first:
                # 蛇头颜色改下
                pygame.draw.rect(SCREEN, BULE, (segment[0], segment[1], BLOCK_SIZE - 1, BLOCK_SIZE - 1))
                first = False
            else:
                pygame.draw.rect(SCREEN, GREEN, (segment[0], segment[1], BLOCK_SIZE-1, BLOCK_SIZE-1))


        # 画食物
        pygame.draw.rect(SCREEN, RED, (self.food[0], self.food[1], BLOCK_SIZE-1, BLOCK_SIZE-1))
        pygame.display.update()
        pygame.time.Clock().tick(10)  # 游戏速度（10帧/秒）