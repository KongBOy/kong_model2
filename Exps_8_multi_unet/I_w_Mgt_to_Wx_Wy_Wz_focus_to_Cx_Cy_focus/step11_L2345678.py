#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
###############################################################################################################################################################################################################
# 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
    os.chdir(code_exe_dir)
# print("current_path:", os.getcwd())
###############################################################################################################################################################################################################

import a_normal.step10_a as block1
import b_limit .step10_a as block1_limit
#################################################################################################################################################################################################################################################################################################################################################################################################
### L2345678_flow
Layer2345678_Ch = [
    [block1.L2_ch128.build().result_obj,
     block1.L2_ch064.build().result_obj,
     block1.L2_ch032.build().result_obj,
     block1.L2_ch016.build().result_obj,
     block1.L2_ch008.build().result_obj,
     block1.L2_ch004.build().result_obj,
     block1.L2_ch002.build().result_obj,
     block1.L2_ch001.build().result_obj],

    [block1.L3_ch128.build().result_obj,
     block1.L3_ch064.build().result_obj,
     block1.L3_ch032.build().result_obj,
     block1.L3_ch016.build().result_obj,
     block1.L3_ch008.build().result_obj,
     block1.L3_ch004.build().result_obj,
     block1.L3_ch002.build().result_obj,
     block1.L3_ch001.build().result_obj],

    [block1.L4_ch128.build().result_obj,
     block1.L4_ch064.build().result_obj,
     block1.L4_ch032.build().result_obj,
     block1.L4_ch016.build().result_obj,
     block1.L4_ch008.build().result_obj,
     block1.L4_ch004.build().result_obj,
     block1.L4_ch002.build().result_obj,
     block1.L4_ch001.build().result_obj],

    [# block1.L5_ch128.build().result_obj,
     block1.L5_ch064.build().result_obj,
     block1.L5_ch032.build().result_obj,
     block1.L5_ch016.build().result_obj,
     block1.L5_ch008.build().result_obj,
     block1.L5_ch004.build().result_obj,
     block1.L5_ch002.build().result_obj,
     block1.L5_ch001.build().result_obj],

    [block1_limit.L6_ch128_limit.build().result_obj,
     # block1.L6_ch064.build().result_obj,
     block1.L6_ch032.build().result_obj,
     block1.L6_ch016.build().result_obj,
     block1.L6_ch008.build().result_obj,
     block1.L6_ch004.build().result_obj,
     block1.L6_ch002.build().result_obj,
     block1.L6_ch001.build().result_obj],

    [block1_limit.L7_ch128_limit.build().result_obj,
     block1_limit.L7_ch064_limit.build().result_obj,
     block1.L7_ch032.build().result_obj,
     block1.L7_ch016.build().result_obj,
     block1.L7_ch008.build().result_obj,
     block1.L7_ch004.build().result_obj,
     block1.L7_ch002.build().result_obj,
     block1.L7_ch001.build().result_obj],

    [block1_limit.L8_ch128_limit.build().result_obj,
     block1_limit.L8_ch064_limit.build().result_obj,
     block1_limit.L8_ch032_limit.build().result_obj,
     block1.L8_ch016.build().result_obj,
     block1.L8_ch008.build().result_obj,
     block1.L8_ch004.build().result_obj,
     block1.L8_ch002.build().result_obj,
     block1.L8_ch001.build().result_obj],
]

# block1_mask_L45678_normal_vs_limit = [
#     [block1.L4_ch128.build().result_obj, block1_limit.L4_ch128_limit.build().result_obj, block1.L6_ch032.build().result_obj, block1_limit.L6_ch032_limit.build().result_obj, block1.L7_ch032.build().result_obj, block1_limit.L7_ch032_limit.build().result_obj, ],
#     [block1.L5_ch064.build().result_obj, block1_limit.L5_ch064_limit.build().result_obj, block1.L6_ch064.build().result_obj, block1_limit.L6_ch064_limit.build().result_obj, block1.L8_ch008.build().result_obj, block1_limit.L8_ch008_limit.build().result_obj, ],
#     [block1.L5_ch128.build().result_obj, block1_limit.L5_ch128_limit.build().result_obj, block1.L7_ch016.build().result_obj, block1_limit.L7_ch016_limit.build().result_obj, block1.L8_ch016.build().result_obj, block1_limit.L8_ch016_limit.build().result_obj, ],
# ]