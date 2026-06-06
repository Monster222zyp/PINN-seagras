# -*- coding: utf-8 -*-
import re

# 读取文件
with open("train_force_model.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. 移除 Re_soft_all 的定义，使用 Re_cyl_all
content = re.sub(r"Re_soft_all = rho \* v_all \* L_all / mu", "", content)

# 2. 将所有 Re_soft_tr 替换为 Re_cyl_tr
content = content.replace("Re_soft_tr", "Re_cyl_tr")

# 3. 将所有 Re_soft_val 替换为 Re_cyl_val
content = content.replace("Re_soft_val", "Re_cyl_val")

# 4. 将所有 Re_soft_np 替换为 Re_cyl_np
content = content.replace("Re_soft_np", "Re_cyl_np")

# 5. 将所有 Re_soft_all_t 替换为 Re_cyl_all_t
content = content.replace("Re_soft_all_t", "Re_cyl_all_t")

# 6. 移除 Re_soft_np 的定义
content = re.sub(r"Re_soft_np = rho \* v_np \* X_shuf\[:, 4\] / mu", "", content)

# 7. 修改图表标签
content = content.replace("Cd_soft vs Re_soft", "Cd_soft vs Re_cyl")

# 8. 修改CSV头部
content = content.replace("Re_cyl,Re_soft,", "Re_cyl,Re_cyl,")

# 写回文件
with open("train_force_model.py", "w", encoding="utf-8") as f:
    f.write(content)

print("修改完成！")
