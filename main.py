import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Item import Star, System
# =====================================用户可修改区域=========================================

# ============ 参数 ============
FPS = 120
BOUND = 80
TRAIL_LEN = 200      # 每个点保留最近多少帧的轨迹
MIU = 1              # 指定快放倍率，智能是大于0的实数
# ============ 指定天体的初始状态 ============
"""pos 位置, radius 半径, vel 速度, mass 质量（可以不指定，会根据半径和密度计算）"""
star1 = Star(
    pos=np.array([0, 0, 0]),
    radius=21,
    vel=np.array([6.0, 0, 0]),
)
star2 = Star(
    pos=np.array([0, 25, 40]),
    radius=15,
    vel=np.array([0, 3.11, 0.11]),
)

star3 = Star(
    pos=np.array([0, 0, -50]),
    radius=9,
    vel=np.array([-2, -1.56, 9.2]),
)
# =========== 指定天体系统初始化 ============
system = System([star1, star2, star3])

n = len(system.stars)
positions = system.plot_pos          # (3, n)
sizes = system.radius.flatten()      # (n,)
colors = system.colors               # (n, 3)



# ========================================================================================

# 下面是渲染逻辑，用户不需要修改区域

# 每个点的历史轨迹
trails = [[] for _ in range(n)]

# ============ 画布 ============
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-BOUND, BOUND)
ax.set_ylim(-BOUND, BOUND)
ax.set_zlim(-BOUND, BOUND)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 当前点
scatter = ax.scatter(
    positions[0], positions[1], positions[2],
    s=sizes, c=colors, alpha=0.9, edgecolors='none'
)

# 每个点一条轨迹线（颜色用该点自己的颜色）
trail_lines = []
for i in range(n):
    ln, = ax.plot(
        [], [], [],
        color=colors[i],
        linewidth=1.2,
        alpha=0.5,
    )
    trail_lines.append(ln)


# ============ 更新函数 ============
def update(frame):
    global positions, sizes, colors

    system.step(dt=1 / FPS)
    positions = system.plot_pos
    sizes = system.radius.flatten()
    # 更新当前点
    scatter._offsets3d = (positions[0], positions[1], positions[2])
    scatter.set_sizes(sizes)
    scatter.set_color(colors)

    # 更新每个点的轨迹
    for i in range(n):
        trails[i].append(positions[:, i].copy())
        if len(trails[i]) > TRAIL_LEN:
            trails[i].pop(0)

        arr = np.array(trails[i])          # (k, 3)
        trail_lines[i].set_data(arr[:, 0], arr[:, 1])
        trail_lines[i].set_3d_properties(arr[:, 2])

    ax.set_title(f'frame {frame}')
    return [scatter] + trail_lines


anim = FuncAnimation(fig, update, frames=2000, interval=1000 / FPS / MIU)
plt.show()