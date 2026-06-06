# 读取文件
with open('train_force_model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 进行替换
for i in range(len(lines)):
    lines[i] = lines[i].replace('Re_soft_tr', 'Re_cyl_tr')
    lines[i] = lines[i].replace('Re_soft_val', 'Re_cyl_val')
    lines[i] = lines[i].replace('Re_soft_all_t', 'Re_cyl_all_t')
    lines[i] = lines[i].replace('Re_soft_np', 'Re_cyl_np')
    lines[i] = lines[i].replace('rho * v_all * L_all / mu', 'rho * v_all * Dc_all / mu')
    lines[i] = lines[i].replace('rho * v_np * X_shuf[:, 4] / mu', 'rho * v_np * X_shuf[:, 2] / mu')
    lines[i] = lines[i].replace('Re_soft,', 'Re_cyl,')
    lines[i] = lines[i].replace('Cd_soft vs Re_soft', 'Cd_soft vs Re_cyl')

# 写回文件
with open('train_force_model.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("修改完成！")
