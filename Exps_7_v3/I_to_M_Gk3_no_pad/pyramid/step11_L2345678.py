#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
code_dir = "\\".join(code_exe_path_element[:-1])
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
sys.path.append(code_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    code_dir:", code_dir)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
###############################################################################################################################################################################################################
# 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
    os.chdir(code_exe_dir)
# print("current_path:", os.getcwd())
###############################################################################################################################################################################################################

import bce_s001_tv_s0p1_1_side_0.step10_a as side1_0
import bce_s001_tv_s0p1_1_side_1.step10_a as side1_1
import bce_s001_tv_s0p1_1_side_2.step10_a as side1_2
import bce_s001_tv_s0p1_1_side_3.step10_a as side1_3
import bce_s001_tv_s0p1_1_side_4.step10_a as side1_4
import bce_s001_tv_s0p1_1_side_5.step10_a as side1_5
import bce_s001_tv_s0p1_1_side_6.step10_a as side1_6
import bce_s001_tv_s0p1_1_side_7.step10_a as side1_7
import bce_s001_tv_s0p1_1_side_8.step10_a as side1_8
#################################################################################################################################################################################################################################################################################################################################################################################################
### L2345678_flow
L2_side1 = [
        [side1_0.L2_ch128, side1_0.L2_ch064, side1_0.L2_ch032, side1_0.L2_ch016, side1_0.L2_ch008, side1_0.L2_ch004, side1_0.L2_ch002, side1_0.L2_ch001],
        [side1_1.L2_ch128, side1_1.L2_ch064, side1_1.L2_ch032, side1_1.L2_ch016, side1_1.L2_ch008, side1_1.L2_ch004, side1_1.L2_ch002, side1_1.L2_ch001],
        [side1_2.L2_ch128, side1_2.L2_ch064, side1_2.L2_ch032, side1_2.L2_ch016, side1_2.L2_ch008, side1_2.L2_ch004, side1_2.L2_ch002, side1_2.L2_ch001],
        [side1_3.L2_ch128, side1_3.L2_ch064, side1_3.L2_ch032, side1_3.L2_ch016, side1_3.L2_ch008, side1_3.L2_ch004, side1_3.L2_ch002, side1_3.L2_ch001],
]

L3_side1 = [
        [side1_0.L3_ch128, side1_0.L3_ch064, side1_0.L3_ch032, side1_0.L3_ch016, side1_0.L3_ch008, side1_0.L3_ch004, side1_0.L3_ch002, side1_0.L3_ch001],
        [side1_1.L3_ch128, side1_1.L3_ch064, side1_1.L3_ch032, side1_1.L3_ch016, side1_1.L3_ch008, side1_1.L3_ch004, side1_1.L3_ch002, side1_1.L3_ch001],
        [side1_2.L3_ch128, side1_2.L3_ch064, side1_2.L3_ch032, side1_2.L3_ch016, side1_2.L3_ch008, side1_2.L3_ch004, side1_2.L3_ch002, side1_2.L3_ch001],
        [side1_3.L3_ch128, side1_3.L3_ch064, side1_3.L3_ch032, side1_3.L3_ch016, side1_3.L3_ch008, side1_3.L3_ch004, side1_3.L3_ch002, side1_3.L3_ch001],
        [side1_4.L3_ch128, side1_4.L3_ch064, side1_4.L3_ch032, side1_4.L3_ch016, side1_4.L3_ch008, side1_4.L3_ch004, side1_4.L3_ch002, side1_4.L3_ch001],
]

L4_side1 = [
        [side1_0.L4_ch128, side1_0.L4_ch064, side1_0.L4_ch032, side1_0.L4_ch016, side1_0.L4_ch008, side1_0.L4_ch004, side1_0.L4_ch002, side1_0.L4_ch001],
        [side1_1.L4_ch128, side1_1.L4_ch064, side1_1.L4_ch032, side1_1.L4_ch016, side1_1.L4_ch008, side1_1.L4_ch004, side1_1.L4_ch002, side1_1.L4_ch001],
        [side1_2.L4_ch128, side1_2.L4_ch064, side1_2.L4_ch032, side1_2.L4_ch016, side1_2.L4_ch008, side1_2.L4_ch004, side1_2.L4_ch002, side1_2.L4_ch001],
        [side1_3.L4_ch128, side1_3.L4_ch064, side1_3.L4_ch032, side1_3.L4_ch016, side1_3.L4_ch008, side1_3.L4_ch004, side1_3.L4_ch002, side1_3.L4_ch001],
        [side1_4.L4_ch128, side1_4.L4_ch064, side1_4.L4_ch032, side1_4.L4_ch016, side1_4.L4_ch008, side1_4.L4_ch004, side1_4.L4_ch002, side1_4.L4_ch001],
        [side1_5.L4_ch128, side1_5.L4_ch064, side1_5.L4_ch032, side1_5.L4_ch016, side1_5.L4_ch008, side1_5.L4_ch004, side1_5.L4_ch002, side1_5.L4_ch001],
]


L5_side1 = [
        [side1_0.L5_ch128, side1_0.L5_ch064, side1_0.L5_ch032, side1_0.L5_ch016, side1_0.L5_ch008, side1_0.L5_ch004, side1_0.L5_ch002, side1_0.L5_ch001],
        [side1_1.L5_ch128, side1_1.L5_ch064, side1_1.L5_ch032, side1_1.L5_ch016, side1_1.L5_ch008, side1_1.L5_ch004, side1_1.L5_ch002, side1_1.L5_ch001],
        [side1_2.L5_ch128, side1_2.L5_ch064, side1_2.L5_ch032, side1_2.L5_ch016, side1_2.L5_ch008, side1_2.L5_ch004, side1_2.L5_ch002, side1_2.L5_ch001],
        [side1_3.L5_ch128, side1_3.L5_ch064, side1_3.L5_ch032, side1_3.L5_ch016, side1_3.L5_ch008, side1_3.L5_ch004, side1_3.L5_ch002, side1_3.L5_ch001],
        [side1_4.L5_ch128, side1_4.L5_ch064, side1_4.L5_ch032, side1_4.L5_ch016, side1_4.L5_ch008, side1_4.L5_ch004, side1_4.L5_ch002, side1_4.L5_ch001],
        [side1_5.L5_ch128, side1_5.L5_ch064, side1_5.L5_ch032, side1_5.L5_ch016, side1_5.L5_ch008, side1_5.L5_ch004, side1_5.L5_ch002, side1_5.L5_ch001],
        [side1_6.L5_ch128, side1_6.L5_ch064, side1_6.L5_ch032, side1_6.L5_ch016, side1_6.L5_ch008, side1_6.L5_ch004, side1_6.L5_ch002, side1_6.L5_ch001],
]

L6_side1 = [
        [side1_0.L6_ch128_limit, side1_0.L6_ch064, side1_0.L6_ch032, side1_0.L6_ch016, side1_0.L6_ch008, side1_0.L6_ch004, side1_0.L6_ch002, side1_0.L6_ch001],
        [side1_1.L6_ch128_limit, side1_1.L6_ch064, side1_1.L6_ch032, side1_1.L6_ch016, side1_1.L6_ch008, side1_1.L6_ch004, side1_1.L6_ch002, side1_1.L6_ch001],
        [side1_2.L6_ch128_limit, side1_2.L6_ch064, side1_2.L6_ch032, side1_2.L6_ch016, side1_2.L6_ch008, side1_2.L6_ch004, side1_2.L6_ch002, side1_2.L6_ch001],
        [side1_3.L6_ch128_limit, side1_3.L6_ch064, side1_3.L6_ch032, side1_3.L6_ch016, side1_3.L6_ch008, side1_3.L6_ch004, side1_3.L6_ch002, side1_3.L6_ch001],
        [side1_4.L6_ch128_limit, side1_4.L6_ch064, side1_4.L6_ch032, side1_4.L6_ch016, side1_4.L6_ch008, side1_4.L6_ch004, side1_4.L6_ch002, side1_4.L6_ch001],
        [side1_5.L6_ch128_limit, side1_5.L6_ch064, side1_5.L6_ch032, side1_5.L6_ch016, side1_5.L6_ch008, side1_5.L6_ch004, side1_5.L6_ch002, side1_5.L6_ch001],
        [side1_6.L6_ch128_limit, side1_6.L6_ch064, side1_6.L6_ch032, side1_6.L6_ch016, side1_6.L6_ch008, side1_6.L6_ch004, side1_6.L6_ch002, side1_6.L6_ch001],
        [side1_7.L6_ch128_limit, side1_7.L6_ch064, side1_7.L6_ch032, side1_7.L6_ch016, side1_7.L6_ch008, side1_7.L6_ch004, side1_7.L6_ch002, side1_7.L6_ch001],
]

L7_side1 = [
        [side1_0.L7_ch128_limit, side1_0.L7_ch064_limit, side1_0.L7_ch032, side1_0.L7_ch016, side1_0.L7_ch008, side1_0.L7_ch004, side1_0.L7_ch002, side1_0.L7_ch001],
        [side1_1.L7_ch128_limit, side1_1.L7_ch064_limit, side1_1.L7_ch032, side1_1.L7_ch016, side1_1.L7_ch008, side1_1.L7_ch004, side1_1.L7_ch002, side1_1.L7_ch001],
        [side1_2.L7_ch128_limit, side1_2.L7_ch064_limit, side1_2.L7_ch032, side1_2.L7_ch016, side1_2.L7_ch008, side1_2.L7_ch004, side1_2.L7_ch002, side1_2.L7_ch001],
        [side1_3.L7_ch128_limit, side1_3.L7_ch064_limit, side1_3.L7_ch032, side1_3.L7_ch016, side1_3.L7_ch008, side1_3.L7_ch004, side1_3.L7_ch002, side1_3.L7_ch001],
        [side1_4.L7_ch128_limit, side1_4.L7_ch064_limit, side1_4.L7_ch032, side1_4.L7_ch016, side1_4.L7_ch008, side1_4.L7_ch004, side1_4.L7_ch002, side1_4.L7_ch001],
        [side1_5.L7_ch128_limit, side1_5.L7_ch064_limit, side1_5.L7_ch032, side1_5.L7_ch016, side1_5.L7_ch008, side1_5.L7_ch004, side1_5.L7_ch002, side1_5.L7_ch001],
        [side1_6.L7_ch128_limit, side1_6.L7_ch064_limit, side1_6.L7_ch032, side1_6.L7_ch016, side1_6.L7_ch008, side1_6.L7_ch004, side1_6.L7_ch002, side1_6.L7_ch001],
        [side1_7.L7_ch128_limit, side1_7.L7_ch064_limit, side1_7.L7_ch032, side1_7.L7_ch016, side1_7.L7_ch008, side1_7.L7_ch004, side1_7.L7_ch002, side1_7.L7_ch001],
        [side1_8.L7_ch128_limit, side1_8.L7_ch064_limit, side1_8.L7_ch032, side1_8.L7_ch016, side1_8.L7_ch008, side1_8.L7_ch004, side1_8.L7_ch002, side1_8.L7_ch001],
]

L8_side1 = [
        [side1_0.L8_ch128_limit, side1_0.L8_ch064_limit, side1_0.L8_ch032_limit, side1_0.L8_ch016, side1_0.L8_ch008, side1_0.L8_ch004, side1_0.L8_ch002, side1_0.L8_ch001],
        [side1_1.L8_ch128_limit, side1_1.L8_ch064_limit, side1_1.L8_ch032_limit, side1_1.L8_ch016, side1_1.L8_ch008, side1_1.L8_ch004, side1_1.L8_ch002, side1_1.L8_ch001],
        [side1_2.L8_ch128_limit, side1_2.L8_ch064_limit, side1_2.L8_ch032_limit, side1_2.L8_ch016, side1_2.L8_ch008, side1_2.L8_ch004, side1_2.L8_ch002, side1_2.L8_ch001],
        [side1_3.L8_ch128_limit, side1_3.L8_ch064_limit, side1_3.L8_ch032_limit, side1_3.L8_ch016, side1_3.L8_ch008, side1_3.L8_ch004, side1_3.L8_ch002, side1_3.L8_ch001],
        [side1_4.L8_ch128_limit, side1_4.L8_ch064_limit, side1_4.L8_ch032_limit, side1_4.L8_ch016, side1_4.L8_ch008, side1_4.L8_ch004, side1_4.L8_ch002, side1_4.L8_ch001],
        [side1_5.L8_ch128_limit, side1_5.L8_ch064_limit, side1_5.L8_ch032_limit, side1_5.L8_ch016, side1_5.L8_ch008, side1_5.L8_ch004, side1_5.L8_ch002, side1_5.L8_ch001],
        [side1_6.L8_ch128_limit, side1_6.L8_ch064_limit, side1_6.L8_ch032_limit, side1_6.L8_ch016, side1_6.L8_ch008, side1_6.L8_ch004, side1_6.L8_ch002, side1_6.L8_ch001],
        [side1_7.L8_ch128_limit, side1_7.L8_ch064_limit, side1_7.L8_ch032_limit, side1_7.L8_ch016, side1_7.L8_ch008, side1_7.L8_ch004, side1_7.L8_ch002, side1_7.L8_ch001],
        [side1_8.L8_ch128_limit, side1_8.L8_ch064_limit, side1_8.L8_ch032_limit, side1_8.L8_ch016, side1_8.L8_ch008, side1_8.L8_ch004, side1_8.L8_ch002, side1_8.L8_ch001],
]
