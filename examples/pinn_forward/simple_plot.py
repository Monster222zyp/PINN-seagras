"""简单的热方程可视化"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def heat_eq_exact_solution(x, t):
    """精确解"""
    a = 0.4  # 热扩散系数
    L = 1    # 杆长
    n = 1    # 频率
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)

def create_visualization():
    """创建可视化"""
    print("创建热方程解的可视化...")
    
    # 创建空间和时间网格
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 50)
    X, T = np.meshgrid(x, t)
    
    # 计算精确解
    U_exact = np.zeros_like(X)
    for i in range(len(t)):
        for j in range(len(x)):
            U_exact[i, j] = heat_eq_exact_solution(x[j], t[i])
    
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D表面图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_exact, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('空间 x')
    ax1.set_ylabel('时间 t')
    ax1.set_zlabel('温度 u')
    ax1.set_title('热方程精确解 - 3D视图')
    
    # 2. 等高线图
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contourf(X, T, U_exact, levels=20, cmap='viridis')
    ax2.set_xlabel('空间 x')
    ax2.set_ylabel('时间 t')
    ax2.set_title('热方程精确解 - 等高线图')
    plt.colorbar(contour, ax=ax2)
    
    # 3. 不同时间点的温度分布
    ax3 = fig.add_subplot(2, 3, 3)
    time_points = [0.0, 0.1, 0.2, 0.5, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(time_points)))
    
    for i, t_val in enumerate(time_points):
        t_idx = np.argmin(np.abs(t - t_val))
        ax3.plot(x, U_exact[t_idx, :], color=colors[i], 
                label=f't = {t_val:.1f}', linewidth=2)
    
    ax3.set_xlabel('空间 x')
    ax3.set_ylabel('温度 u')
    ax3.set_title('不同时间点的温度分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 不同位置点的时间演化
    ax4 = fig.add_subplot(2, 3, 4)
    x_points = [0.2, 0.4, 0.5, 0.6, 0.8]
    colors = plt.cm.cool(np.linspace(0, 1, len(x_points)))
    
    for i, x_val in enumerate(x_points):
        x_idx = np.argmin(np.abs(x - x_val))
        ax4.plot(t, U_exact[:, x_idx], color=colors[i], 
                label=f'x = {x_val:.1f}', linewidth=2)
    
    ax4.set_xlabel('时间 t')
    ax4.set_ylabel('温度 u')
    ax4.set_title('不同位置点的时间演化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 初始条件
    ax5 = fig.add_subplot(2, 3, 5)
    initial_condition = np.sin(np.pi * x)
    ax5.plot(x, initial_condition, 'r-', linewidth=3, label='初始条件: sin(πx)')
    ax5.plot(x, U_exact[0, :], 'b--', linewidth=2, label='t=0时的解')
    ax5.set_xlabel('空间 x')
    ax5.set_ylabel('温度 u')
    ax5.set_title('初始条件')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 能量衰减
    ax6 = fig.add_subplot(2, 3, 6)
    # 计算每个时刻的总能量 (L2范数的平方)
    energy = np.array([np.trapz(U_exact[i, :]**2, x) for i in range(len(t))])
    theoretical_energy = np.exp(-2 * np.pi**2 * 0.4 * t) * 0.5  # 理论能量衰减
    
    ax6.plot(t, energy, 'b-', linewidth=2, label='数值能量')
    ax6.plot(t, theoretical_energy, 'r--', linewidth=2, label='理论能量')
    ax6.set_xlabel('时间 t')
    ax6.set_ylabel('能量')
    ax6.set_title('能量随时间的衰减')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('heat_equation_exact_solution.png', dpi=300, bbox_inches='tight')
    print("图像已保存为: heat_equation_exact_solution.png")
    
    # 显示图像
    plt.show()
    
    # 打印一些关键信息
    print(f"\n=== 热方程解的特性 ===")
    print(f"初始最大温度: {np.max(U_exact[0, :]):.6f}")
    print(f"最终最大温度: {np.max(U_exact[-1, :]):.6f}")
    print(f"温度衰减比: {np.max(U_exact[-1, :]) / np.max(U_exact[0, :]):.6f}")
    print(f"理论衰减比: {np.exp(-np.pi**2 * 0.4 * 1):.6f}")
    
    # 验证边界条件
    print(f"\n=== 边界条件验证 ===")
    print(f"x=0处所有时间的最大温度: {np.max(np.abs(U_exact[:, 0])):.10f}")
    print(f"x=1处所有时间的最大温度: {np.max(np.abs(U_exact[:, -1])):.10f}")
    
    return X, T, U_exact

if __name__ == "__main__":
    X, T, U = create_visualization()
    print("可视化完成！")

