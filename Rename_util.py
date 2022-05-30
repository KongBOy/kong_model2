import os
from kong_util.util import Visit_sub_dir_include_self_and_get_dir_paths

dir_paths = []
Visit_sub_dir_include_self_and_get_dir_paths(src_dir=r"H:\data_dir\result\Exps_7_v3\doc3d\I_to_M_Gk3_no_pad", dir_containor=dir_paths)

change_amount = 0
for dir_path in dir_paths:
    if("seet_" in dir_path):
        wrong = dir_path
        right = dir_path.replace("seet_", "see_")
        print(f"rename seet_ to see_:\n\t{wrong}-->\n\t{right}")
        os.rename(wrong, right)
        change_amount += 1
print(f"總共修改了：{change_amount}個資料夾")
print("finish")