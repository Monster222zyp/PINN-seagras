"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

这是一个基于原始Euler_beam示例修改的版本，用于自定义开发。
该文件求解欧拉-伯努利梁方程的PINN实现。

欧拉-伯努利梁方程: EI * d⁴y/dx⁴ = q(x)
边界条件:
- 左端(x=0): y = 0 (位移为0)
- 左端(x=0): y' = 0 (斜率为0)
- 右端(x=1): y'' = 0 (弯矩为0)
- 右端(x=1): y''' = 0 (剪力为0)

解析解(当EI=1, q(x)=-1时): y(x) = -x⁴/24 + x³/6 - x²/4
"""
# 导入配置文件，自动设置环境
import config

import deepxde as dde
import numpy as np
from scipy.io import loadmat


# 定义材料和几何参数
E = 1.0  # 杨氏模量
I = 1.0  # 截面惯性矩
EI = E * I  # 弯曲刚度

# 定义载荷函数
def load_function(x):
    # 可以定义任意的载荷函数 q(x)
    # 这里使用恒定载荷 q(x) = -1
    return -1.0


# 定义二阶导数算子
def ddy(x, y):
    return dde.grad.hessian(y, x)


# 定义三阶导数算子
def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)


# 定义PDE
def pde(x, y):
    dy_xx = ddy(x, y)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return EI * dy_xxxx - load_function(x)  # EI * d⁴y/dx⁴ = q(x)


# 定义左边界(x=0)
def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


# 定义右边界(x=1)
def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


# 定义解析解函数
def func(x):
    # 当EI=1且q(x)=-1时的解析解
    return -(x**4) / 24 + x**3 / 6 - x**2 / 4


# 加载实验数据
def load_experimental_data(mat_file_path, num_points=None):
    """
    加载.mat文件中的实验数据
    
    参数:
        mat_file_path: .mat文件路径
        num_points: 要使用的数据点数量，None表示使用所有点
        
    返回:
        x_data: 位置数据
        y_data: 位移数据
    """
    data = loadmat(mat_file_path)
    
    # 优先从 pinn_data 结构中读取 X_matrix / Y_matrix
    if "pinn_data" in data:
        pinn_data = data["pinn_data"][0, 0]
        # 期望 X_matrix 形状为 (N, D)，Y_matrix 形状为 (N, 1)
        Xm = pinn_data["X_matrix"]
        Ym = pinn_data["Y_matrix"]
        
        x_raw = Xm[:, 0].astype(float).reshape(-1, 1)  # 使用第一列作为自变量
        y = Ym.astype(float).reshape(-1, 1)
        # 将自变量归一化到 [0, 1]，与计算域一致
        xmin, xmax = float(x_raw.min()), float(x_raw.max())
        x = (x_raw - xmin) / (xmax - xmin + 1e-12)
        
        if num_points is not None and num_points < len(x):
            idx = np.random.choice(len(x), num_points, replace=False)
            return x[idx], y[idx]
        return x, y
    
    # 回退：兼容简单的 {'x','y'} 键
    if "x" in data and "y" in data:
        x = data["x"].flatten()[:, None]
        y = data["y"].flatten()[:, None]
        if num_points is not None and num_points < len(x):
            idx = np.random.choice(len(x), num_points, replace=False)
            return x[idx], y[idx]
        return x, y
    
    raise ValueError("无法在 MAT 文件中找到可用的数据字段。期望存在 'pinn_data' 结构或 'x'/'y' 字段。")


# 定义计算域
geom = dde.geometry.Interval(0, 1)

# 定义边界条件
bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)  # y(0) = 0
bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)    # y'(0) = 0
bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)  # y''(1) = 0
bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)  # y'''(1) = 0

# 是否使用实验数据训练
use_experimental_data = True  # 设置为True使用实验数据，False使用解析解

if use_experimental_data:
    # 加载实验数据
    mat_file_path = "pinn_training_data.mat"  # 您的实验数据文件
    x_exp, y_exp = load_experimental_data(mat_file_path, num_points=50)
    
    # 创建观测数据点约束
    observe_y = dde.icbc.PointSetBC(x_exp, y_exp)
    
    # 创建PDE问题（包含实验数据约束）
    data = dde.data.PDE(
        geom,
        pde,
        [bc1, bc2, bc3, bc4, observe_y],  # 包含实验数据的约束
        num_domain=100,       # 内部采样点数量
        num_boundary=20,      # 边界采样点数量
        anchors=x_exp,        # 将实验数据点作为锚点
    )
else:
    # 使用解析解创建PDE问题
    data = dde.data.PDE(
        geom,
        pde,
        [bc1, bc2, bc3, bc4],
        num_domain=10,      # 内部采样点数量
        num_boundary=2,     # 边界采样点数量
        solution=func,      # 解析解，用于计算误差
        num_test=100,       # 测试点数量
    )

# 定义神经网络
layer_size = [1] + [20] * 3 + [1]  # 输入维度1，3个隐藏层(每层20个节点)，输出维度1
activation = "tanh"                # 激活函数
initializer = "Glorot uniform"     # 权重初始化方法
net = dde.nn.FNN(layer_size, activation, initializer)

# 创建模型
model = dde.Model(data, net)
if use_experimental_data:
    # 无解析解/测试集时不计算 l2 相对误差
    model.compile("adam", lr=0.001)
else:
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# 训练模型
losshistory, train_state = model.train(iterations=10000)

# 保存结果
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# 如果使用了实验数据，额外进行预测和可视化
if use_experimental_data:
    import matplotlib.pyplot as plt
    
    # 预测并与实验数据对比
    x_pred = np.linspace(0, 1, 100)[:, None]
    y_pred = model.predict(x_pred)
    
    # 保存预测结果
    np.savetxt("prediction_results.dat", np.hstack((x_pred, y_pred)))
    
    # 计算PDE残差
    f = model.predict(x_pred, operator=pde)
    print("平均PDE残差:", np.mean(np.absolute(f)))
    
    # 可视化结果
    plt.figure()
    plt.scatter(x_exp, y_exp, label="实验数据")
    plt.plot(x_pred, y_pred, label="PINN预测")
    
    # 绘制解析解作为参考
    x_analytical = np.linspace(0, 1, 100)[:, None]
    y_analytical = func(x_analytical)
    plt.plot(x_analytical, y_analytical, '--', label="解析解")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("comparison.png")
    plt.show() 