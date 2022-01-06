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
block1_mask_L2345678 = [
    [block1.L2_ch129.build().result_obj,
     block1.L2_ch066.build().result_obj,
     block1.L2_ch033.build().result_obj,
     block1.L2_ch018.build().result_obj,
     block1.L2_ch009.build().result_obj,
     block1.L2_ch006.build().result_obj,
     block1.L2_ch003.build().result_obj],

    [block1.L3_ch129.build().result_obj,
     block1.L3_ch066.build().result_obj,
     block1.L3_ch033.build().result_obj,
     block1.L3_ch018.build().result_obj,
     block1.L3_ch009.build().result_obj,
     block1.L3_ch006.build().result_obj,
     block1.L3_ch003.build().result_obj],

    [block1.L4_ch129.build().result_obj,
     block1.L4_ch066.build().result_obj,
     block1.L4_ch033.build().result_obj,
     block1.L4_ch018.build().result_obj,
     block1.L4_ch009.build().result_obj,
     block1.L4_ch006.build().result_obj,
     block1.L4_ch003.build().result_obj],

    [block1.L5_ch129.build().result_obj,
     block1.L5_ch066.build().result_obj,
     block1.L5_ch033.build().result_obj,
     block1.L5_ch018.build().result_obj,
     block1.L5_ch009.build().result_obj,
     block1.L5_ch006.build().result_obj,
     block1.L5_ch003.build().result_obj],

    [block1_limit.L6_ch129_limit.build().result_obj,
     block1.L6_ch066.build().result_obj,
     block1.L6_ch033.build().result_obj,
     block1.L6_ch018.build().result_obj,
     block1.L6_ch009.build().result_obj,
     block1.L6_ch006.build().result_obj,
     block1.L6_ch003.build().result_obj],

    [block1_limit.L7_ch129_limit.build().result_obj,
     block1_limit.L7_ch066_limit.build().result_obj,
     block1.L7_ch033.build().result_obj,
     block1.L7_ch018.build().result_obj,
     block1.L7_ch009.build().result_obj,
     block1.L7_ch006.build().result_obj,
     block1.L7_ch003.build().result_obj],

    [block1_limit.L8_ch129_limit.build().result_obj,
     block1_limit.L8_ch066_limit.build().result_obj,
     block1_limit.L8_ch033_limit.build().result_obj,
     block1.L8_ch018.build().result_obj,
     block1.L8_ch009.build().result_obj,
     block1.L8_ch006.build().result_obj,
     block1.L8_ch003.build().result_obj],
]

block1_mask_L45678_normal_vs_limit = [
    [block1.L4_ch129.build().result_obj, block1_limit.L4_ch129_limit.build().result_obj, block1.L6_ch033.build().result_obj, block1_limit.L6_ch033_limit.build().result_obj, block1.L7_ch033.build().result_obj, block1_limit.L7_ch033_limit.build().result_obj, ],
    [block1.L5_ch066.build().result_obj, block1_limit.L5_ch066_limit.build().result_obj, block1.L6_ch066.build().result_obj, block1_limit.L6_ch066_limit.build().result_obj, block1.L8_ch009.build().result_obj, block1_limit.L8_ch009_limit.build().result_obj, ],
    [block1.L5_ch129.build().result_obj, block1_limit.L5_ch129_limit.build().result_obj, block1.L7_ch018.build().result_obj, block1_limit.L7_ch018_limit.build().result_obj, block1.L8_ch018.build().result_obj, block1_limit.L8_ch018_limit.build().result_obj, ],
]