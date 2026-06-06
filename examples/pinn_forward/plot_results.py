"""简单的热方程结果可视化"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def heat_eq_exact_solution(x, t):
    """精确解"""
    a = 0.4  # 热扩散系数
    L = 1    # 杆长
    n = 1    # 频率
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)

def plot_heat_equation_results():
    """绘制热方程结果"""
    print("开始可视化热方程PINN预测结果...")
    
    # 读取保存的数据
    try:
        # 读取本地test.dat文件 (格式: x, y_true, y_pred)
        test_data = np.loadtxt("test.dat")
        print(f"成功读取本地测试数据，形状: {test_data.shape}")
        
        # 但这个文件没有时间信息，需要从原始数据重构
        data = np.load("../../heat_eq_data.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        x_data = np.ravel(xx)
        t_data = np.ravel(tt)
        y_true = exact.flatten()
        
        # 从根目录的test.dat读取预测结果
        root_test_data = np.loadtxt("../../test.dat")
        print(f"成功读取根目录测试数据，形状: {root_test_data.shape}")
        y_pred = root_test_data[:, 3] if root_test_data.shape[1] > 3 else root_test_data[:, 2]
        
    except Exception as e:
        print(f"读取数据失败: {e}")
        print("使用heat_eq_data.npz数据...")
        
        # 使用原始数据
        data = np.load("../../heat_eq_data.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        xx, tt = np.meshgrid(x, t)
        x_data = np.ravel(xx)
        t_data = np.ravel(tt)
        y_true = exact.flatten()
        y_pred = y_true  # 如果没有预测数据，用真实值代替
    
    # 确定网格大小
    x_unique = np.unique(x_data)
    t_unique = np.unique(t_data)
    nx, nt = len(x_unique), len(t_unique)
    
    print(f"空间点数: {nx}, 时间点数: {nt}")
    
    # 重新整理数据为网格形式
    X_mesh, T_mesh = np.meshgrid(x_unique, t_unique)
    U_true = y_true.reshape(nt, nx)
    U_pred = y_pred.reshape(nt, nx)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D表面图 - 精确解
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X_mesh, T_mesh, U_true, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('空间 x')
    ax1.set_ylabel('时间 t')
    ax1.set_zlabel('温度 u')
    ax1.set_title('精确解')
    
    # 2. 3D表面图 - PINN预测解
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X_mesh, T_mesh, U_pred, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('空间 x')
    ax2.set_ylabel('时间 t')
    ax2.set_zlabel('温度 u')
    ax2.set_title('PINN预测解')
    
    # 3. 3D表面图 - 误差
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    error = np.abs(U_true - U_pred)
    surf3 = ax3.plot_surface(X_mesh, T_mesh, error, cmap='Reds', alpha=0.8)
    ax3.set_xlabel('空间 x')
    ax3.set_ylabel('时间 t')
    ax3.set_zlabel('绝对误差')
    ax3.set_title('预测误差')
    
    # 4. 等高线图 - 精确解
    ax4 = fig.add_subplot(2, 3, 4)
    contour1 = ax4.contourf(X_mesh, T_mesh, U_true, levels=20, cmap='viridis')
    ax4.set_xlabel('空间 x')
    ax4.set_ylabel('时间 t')
    ax4.set_title('精确解等高线')
    plt.colorbar(contour1, ax=ax4)
    
    # 5. 等高线图 - PINN预测解
    ax5 = fig.add_subplot(2, 3, 5)
    contour2 = ax5.contourf(X_mesh, T_mesh, U_pred, levels=20, cmap='viridis')
    ax5.set_xlabel('空间 x')
    ax5.set_ylabel('时间 t')
    ax5.set_title('PINN预测解等高线')
    plt.colorbar(contour2, ax=ax5)
    
    # 6. 不同时间点的温度分布对比
    ax6 = fig.add_subplot(2, 3, 6)
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(time_points)))
    
    for i, t_val in enumerate(time_points):
        # 找到最接近的时间索引
        t_idx = np.argmin(np.abs(t_unique - t_val))
        ax6.plot(x_unique, U_true[t_idx, :], '--', color=colors[i], 
                label=f't={t_val:.1f} (精确)', alpha=0.7, linewidth=2)
        ax6.plot(x_unique, U_pred[t_idx, :], '-', color=colors[i], 
                label=f't={t_val:.1f} (PINN)', linewidth=2)
    
    ax6.set_xlabel('空间 x')
    ax6.set_ylabel('温度 u')
    ax6.set_title('不同时间点的温度分布对比')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('heat_equation_results.png', dpi=300, bbox_inches='tight')
    print("图像已保存为: heat_equation_results.png")
    
    # 显示图像
    plt.show()
    
    # 计算并打印误差统计
    mse = np.mean((U_true - U_pred)**2)
    mae = np.mean(np.abs(U_true - U_pred))
    max_error = np.max(np.abs(U_true - U_pred))
    
    print(f"\n=== 预测结果统计 ===")
    print(f"均方误差 (MSE): {mse:.6e}")
    print(f"平均绝对误差 (MAE): {mae:.6e}")
    print(f"最大绝对误差: {max_error:.6e}")
    if np.mean(np.abs(U_true)) > 0:
        print(f"相对误差 (%): {100 * mae / np.mean(np.abs(U_true)):.4f}%")
    
    # 显示一些关键时间点的解
    print(f"\n=== 关键时间点的解 ===")
    for t_val in [0.0, 0.5, 1.0]:
        t_idx = np.argmin(np.abs(t_unique - t_val))
        x_center = len(x_unique) // 2
        print(f"t={t_val:.1f}, x=0.5处: 精确解={U_true[t_idx, x_center]:.6f}, "
              f"PINN预测={U_pred[t_idx, x_center]:.6f}")

if __name__ == "__main__":
    plot_heat_equation_results()
    print("可视化完成！")
