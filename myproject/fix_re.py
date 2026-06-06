import re

# 读取文件
with open('train_force_model.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换所有Re_soft相关的变量为Re_cyl
replacements = [
    ('Re_soft_tr', 'Re_cyl_tr'),
    ('Re_soft_val', 'Re_cyl_val'), 
    ('Re_soft_all_t', 'Re_cyl_all_t'),
    ('Re_soft_np', 'Re_cyl_np'),
    ('Re_soft_all = rho \* v_all \* L_all / mu', 'Re_soft_all = rho * v_all * Dc_all / mu'),
    ('plt.scatter\(Re_cyl_np, cd_s_all, s=18, label="Cd_soft vs Re_soft"\)', 'plt.scatter(Re_cyl_np, cd_s_all, s=18, label="Cd_soft vs Re_cyl")'),
    ('"idx,v,Hc,Dc,L,Re_cyl,Re_soft,Ca,Cd_cyl,Cd_soft,y_true,y_pred,F_cyl_pred,F_soft_pred,F_soft_pred_col1,F_soft_pred_col2,F_soft_pred_col3,Fc1\(F@Cd=1\),Fs1\(F@Cd=1\),Fs1_col1,Fs1_col2,Fs1_col3"', '"idx,v,Hc,Dc,L,Re_cyl,Re_cyl,Ca,Cd_cyl,Cd_soft,y_true,y_pred,F_cyl_pred,F_soft_pred,F_soft_pred_col1,F_soft_pred_col2,F_soft_pred_col3,Fc1(F@Cd=1),Fs1(F@Cd=1),Fs1_col1,Fs1_col2,Fs1_col3"')
]

for old, new in replacements:
    content = content.replace(old, new)

# 写回文件
with open('train_force_model.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("替换完成！")
