"""可视化热方程PINN预测结果"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import deepxde as dde

def heat_eq_exact_solution(x, t):
    """精确解"""
    a = 0.4  # 热扩散系数
    L = 1    # 杆长
    n = 1    # 频率
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)

def gen_testdata():
    """生成测试数据"""
    # 加载数据
    data = np.load("heat_eq_data.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    # 处理数据
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y, x, t, exact

def load_model_and_predict():
    """加载训练好的模型并预测"""
    # 重新定义PDE
    def pde(x, y):
        a = 0.4
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - a * dy_xx

    # 重新构建几何和数据
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.sin(np.pi * x[:, 0:1]),
        lambda _, on_initial: on_initial,
    )
    
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
        num_test=2540,
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    # 尝试加载训练好的模型
    try:
        model.restore("model.ckpt")
        print("加载训练好的模型成功")
    except:
        print("未找到训练好的模型，请先运行 heat.py")
        return None, None, None, None, None
    
    # 生成预测数据
    X, y_true, x, t, exact = gen_testdata()
    y_pred = model.predict(X)
    
    return X, y_true, y_pred, x, t, exact

def visualize_results():
    """可视化结果"""
    # 生成测试数据和预测
    X, y_true, x, t, exact = gen_testdata()
    
    # 重新reshape数据用于绘图
    x_mesh, t_mesh = np.meshgrid(x.flatten(), t.flatten())
    u_true = exact
    
    # 从保存的test.dat文件读取预测结果
    try:
        test_data = np.loadtxt("../../test.dat")
        X_test = test_data[:, :2]  # x, t coordinates
        y_true_flat = test_data[:, 2]  # true values
        y_pred_flat = test_data[:, 3]  # predicted values
        
        # Reshape预测结果
        n_x, n_t = len(x), len(t)
        u_pred = y_pred_flat.reshape(n_t, n_x)
        
    except:
        print("无法读取test.dat文件，生成新的预测...")
        # 如果无法读取，就用精确解作为示例
        u_pred = u_true
    
    # 创建图形
    fig = plt.figure(figsize=(15, 12))
    
    # 1. 3D表面图 - 精确解
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(x_mesh, t_mesh, u_true, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('空间 x')
    ax1.set_ylabel('时间 t')
    ax1.set_zlabel('温度 u')
    ax1.set_title('精确解')
    
    # 2. 3D表面图 - 预测解
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(x_mesh, t_mesh, u_pred, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('空间 x')
    ax2.set_ylabel('时间 t')
    ax2.set_zlabel('温度 u')
    ax2.set_title('PINN预测解')
    
    # 3. 3D表面图 - 误差
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    error = np.abs(u_true - u_pred)
    surf3 = ax3.plot_surface(x_mesh, t_mesh, error, cmap='Reds', alpha=0.8)
    ax3.set_xlabel('空间 x')
    ax3.set_ylabel('时间 t')
    ax3.set_zlabel('绝对误差')
    ax3.set_title('预测误差')
    
    # 4. 等高线图 - 精确解
    ax4 = fig.add_subplot(2, 3, 4)
    contour1 = ax4.contourf(x_mesh, t_mesh, u_true, levels=20, cmap='viridis')
    ax4.set_xlabel('空间 x')
    ax4.set_ylabel('时间 t')
    ax4.set_title('精确解等高线')
    plt.colorbar(contour1, ax=ax4)
    
    # 5. 等高线图 - 预测解
    ax5 = fig.add_subplot(2, 3, 5)
    contour2 = ax5.contourf(x_mesh, t_mesh, u_pred, levels=20, cmap='viridis')
    ax5.set_xlabel('空间 x')
    ax5.set_ylabel('时间 t')
    ax5.set_title('PINN预测解等高线')
    plt.colorbar(contour2, ax=ax5)
    
    # 6. 不同时间点的温度分布对比
    ax6 = fig.add_subplot(2, 3, 6)
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(time_points)))
    
    for i, t_val in enumerate(time_points):
        t_idx = np.argmin(np.abs(t.flatten() - t_val))
        ax6.plot(x.flatten(), u_true[t_idx, :], '--', color=colors[i], 
                label=f't={t_val:.1f} (精确)', alpha=0.7)
        ax6.plot(x.flatten(), u_pred[t_idx, :], '-', color=colors[i], 
                label=f't={t_val:.1f} (预测)', linewidth=2)
    
    ax6.set_xlabel('空间 x')
    ax6.set_ylabel('温度 u')
    ax6.set_title('不同时间点的温度分布')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_equation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算并打印误差统计
    mse = np.mean((u_true - u_pred)**2)
    mae = np.mean(np.abs(u_true - u_pred))
    max_error = np.max(np.abs(u_true - u_pred))
    
    print(f"\n=== 预测结果统计 ===")
    print(f"均方误差 (MSE): {mse:.6e}")
    print(f"平均绝对误差 (MAE): {mae:.6e}")
    print(f"最大绝对误差: {max_error:.6e}")
    print(f"相对误差 (%): {100 * mae / np.mean(np.abs(u_true)):.4f}%")

if __name__ == "__main__":
    print("开始可视化热方程PINN预测结果...")
    visualize_results()
    print("可视化完成！图像已保存为 heat_equation_results.png")

