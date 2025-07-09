# DeepXDE 使用 .mat 文件训练模型说明

本项目展示如何使用 DeepXDE 框架加载和使用 .mat 文件中的实验数据来训练物理信息神经网络(PINN)模型。

## 文件说明

- `my_euler_beam.py`: 原始的欧拉梁方程求解示例，使用解析解作为参考
- `my_euler_beam_with_mat_data.py`: 展示如何导入 .mat 文件的实验数据来训练欧拉梁模型

## 如何使用

1. 准备一个包含实验数据的 .mat 文件，确保其中包含:
   - 'x': 位置数据
   - 'y': 位移数据

2. 修改 `my_euler_beam_with_mat_data.py` 中的 `mat_file_path` 变量，指向你的 .mat 文件:
   ```python
   mat_file_path = "path/to/your/experimental_data.mat"  # 修改为你的.mat文件路径
   ```

3. 如果你的 .mat 文件中的变量名不是 'x' 和 'y'，请修改 `load_experimental_data` 函数中的对应部分:
   ```python
   # 根据你的.mat文件结构调整以下字段名
   x = data["your_x_variable_name"].flatten()[:, None]
   y = data["your_y_variable_name"].flatten()[:, None]
   ```

4. 运行脚本:
   ```
   python my_euler_beam_with_mat_data.py
   ```

## 输出文件

- `prediction_results.dat`: 包含预测点和预测值的数据
- `comparison.png`: 可视化图，展示实验数据和PINN预测的对比
- `loss.dat` 和其他训练日志文件

## 注意事项

1. 确保你的 .mat 文件中的数据格式正确，如果遇到加载问题，可以使用以下代码检查数据结构:
   ```python
   from scipy.io import loadmat
   data = loadmat("your_file.mat")
   print(data.keys())  # 查看所有变量名
   print(data["x"].shape)  # 查看数据形状
   ```

2. 如果遇到linter警告，比如关于FNN的警告，可以忽略它们。这些警告是由于DeepXDE的动态导入机制引起的，不会影响代码的实际运行。

3. 如果需要调整模型参数，可以修改:
   - 神经网络结构: `layer_size = [1, 20, 20, 20, 1]`
   - 激活函数: `activation = "tanh"`
   - 优化器和学习率: `model.compile("adam", lr=0.001)`
   - 迭代次数: `model.train(iterations=10000)` 