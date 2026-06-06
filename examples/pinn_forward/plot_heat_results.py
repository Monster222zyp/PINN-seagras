"""可视化热方程PINN结果"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def heat_eq_exact_solution(x, t):
    """精确解"""
    a = 0.4  # 热扩散系数
    L = 1    # 杆长
    n = 1    # 频率
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)

def plot_comparison():
    """绘制PINN预测结果与精确解的对比"""
    print("开始可视化热方程PINN预测结果...")
    
    # 创建测试网格
    x = np.linspace(0, 1, 50)
    t = np.linspace(0, 1, 30)
    X, T = np.meshgrid(x, t)
    
    # 计算精确解
    U_exact = np.zeros_like(X)
    for i in range(len(t)):
        for j in range(len(x)):
            U_exact[i, j] = heat_eq_exact_solution(x[j], t[i])
    
    # 尝试读取PINN预测结果
    try:
        # 读取根目录的test.dat文件
        test_data = np.loadtxt("../../test.dat")
        print(f"成功读取PINN预测数据，形状: {test_data.shape}")
        
        # 从test.dat提取数据 (格式可能是 x, t, y_true, y_pred)
        if test_data.shape[1] >= 4:
            x_pred = test_data[:, 0]
            t_pred = test_data[:, 1]
            y_true = test_data[:, 2]
            y_pred = test_data[:, 3]
        else:
            # 如果只有3列，假设是 x, y_true, y_pred (没有时间信息)
            print("数据格式不包含时间信息，使用精确解作为对比")
            y_pred = U_exact
            use_pinn_data = False
    except Exception as e:
        print(f"无法读取PINN预测数据: {e}")
        print("仅显示精确解")
        y_pred = U_exact
        use_pinn_data = False
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D表面图 - 精确解
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_exact, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('空间 x')
    ax1.set_ylabel('时间 t')
    ax1.set_zlabel('温度 u')
    ax1.set_title('精确解')
    
    # 2. 3D表面图 - PINN预测解
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, T, y_pred, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('空间 x')
    ax2.set_ylabel('时间 t')
    ax2.set_zlabel('温度 u')
    ax2.set_title('PINN预测解')
    
    # 3. 误差分析
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    error = np.abs(U_exact - y_pred)
    surf3 = ax3.plot_surface(X, T, error, cmap='Reds', alpha=0.8)
    ax3.set_xlabel('空间 x')
    ax3.set_ylabel('时间 t')
    ax3.set_zlabel('绝对误差')
    ax3.set_title('预测误差')
    
    # 4. 等高线图 - 精确解
    ax4 = fig.add_subplot(2, 3, 4)
    contour1 = ax4.contourf(X, T, U_exact, levels=20, cmap='viridis')
    ax4.set_xlabel('空间 x')
    ax4.set_ylabel('时间 t')
    ax4.set_title('精确解等高线')
    plt.colorbar(contour1, ax=ax4)
    
    # 5. 等高线图 - PINN预测
    ax5 = fig.add_subplot(2, 3, 5)
    contour2 = ax5.contourf(X, T, y_pred, levels=20, cmap='viridis')
    ax5.set_xlabel('空间 x')
    ax5.set_ylabel('时间 t')
    ax5.set_title('PINN预测等高线')
    plt.colorbar(contour2, ax=ax5)
    
    # 6. 不同时间点的对比
    ax6 = fig.add_subplot(2, 3, 6)
    time_points = [0.0, 0.2, 0.5, 0.8, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(time_points)))
    
    for i, t_val in enumerate(time_points):
        t_idx = np.argmin(np.abs(t - t_val))
        ax6.plot(x, U_exact[t_idx, :], '--', color=colors[i], 
                label=f't={t_val:.1f} (精确)', alpha=0.7, linewidth=2)
        ax6.plot(x, y_pred[t_idx, :], '-', color=colors[i], 
                label=f't={t_val:.1f} (PINN)', linewidth=2)
    
    ax6.set_xlabel('空间 x')
    ax6.set_ylabel('温度 u')
    ax6.set_title('不同时间点的温度分布对比')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('heat_equation_pinn_results.png', dpi=300, bbox_inches='tight')
    print("图像已保存为: heat_equation_pinn_results.png")
    
    # 计算并打印误差统计
    mse = np.mean((U_exact - y_pred)**2)
    mae = np.mean(np.abs(U_exact - y_pred))
    max_error = np.max(np.abs(U_exact - y_pred))
    
    print(f"\n=== 预测结果统计 ===")
    print(f"均方误差 (MSE): {mse:.6e}")
    print(f"平均绝对误差 (MAE): {mae:.6e}")
    print(f"最大绝对误差: {max_error:.6e}")
    if np.mean(np.abs(U_exact)) > 0:
        print(f"相对误差 (%): {100 * mae / np.mean(np.abs(U_exact)):.4f}%")
    
    # 显示关键点的数值
    print(f"\n=== 关键点数值对比 ===")
    mid_x, mid_t = len(x)//2, len(t)//2
    print(f"中心点 (x=0.5, t=0.5):")
    print(f"  精确解: {U_exact[mid_t, mid_x]:.6f}")
    print(f"  PINN预测: {y_pred[mid_t, mid_x]:.6f}")
    print(f"  绝对误差: {abs(U_exact[mid_t, mid_x] - y_pred[mid_t, mid_x]):.6f}")

def create_training_visualization():
    """创建训练过程可视化"""
    print("\n创建训练过程可视化...")
    
    try:
        # 读取损失历史
        loss_data = np.loadtxt("../../loss.dat")
        print(f"成功读取训练损失数据，形状: {loss_data.shape}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 训练损失
        if loss_data.shape[1] >= 2:
            steps = loss_data[:, 0]
            train_loss = loss_data[:, 1]
            
            axes[0].semilogy(steps, train_loss, 'b-', linewidth=2, label='训练损失')
            if loss_data.shape[1] >= 3:
                test_loss = loss_data[:, 2]
                axes[0].semilogy(steps, test_loss, 'r--', linewidth=2, label='测试损失')
            
            axes[0].set_xlabel('训练步数')
            axes[0].set_ylabel('损失值')
            axes[0].set_title('PINN训练损失曲线')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 损失分量（如果有的话）
            if loss_data.shape[1] > 3:
                for i in range(3, min(loss_data.shape[1], 6)):
                    axes[1].semilogy(steps, loss_data[:, i], linewidth=2, 
                                   label=f'损失分量 {i-2}')
                
                axes[1].set_xlabel('训练步数')
                axes[1].set_ylabel('损失分量')
                axes[1].set_title('PINN损失分量')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, '无损失分量数据', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('损失分量（无数据）')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("训练历史图像已保存为: training_history.png")
        
    except Exception as e:
        print(f"无法创建训练可视化: {e}")

if __name__ == "__main__":
    plot_comparison()
    create_training_visualization()
    print("\n可视化完成！生成的文件:")
    print("- heat_equation_pinn_results.png: PINN预测结果对比")
    print("- training_history.png: 训练过程")
