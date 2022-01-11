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

import a_normal.step10_a as mae_block1
import b_limit.step10_a  as mae_block1_limit
#################################################################################################################################################################################################################################################################################################################################################################################################
### Layer_Ch
block1_mask_L2345678 = [
    [
        mae_block1.I_to_Wx_L2_ch128_and_I_to_Wy_L2_ch128_ep060_and_I_to_Wz_L2_ch128_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch064_and_I_to_Wy_L2_ch064_ep060_and_I_to_Wz_L2_ch064_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch032_and_I_to_Wy_L2_ch032_ep060_and_I_to_Wz_L2_ch032_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch016_and_I_to_Wy_L2_ch016_ep060_and_I_to_Wz_L2_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch008_and_I_to_Wy_L2_ch008_ep060_and_I_to_Wz_L2_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch004_and_I_to_Wy_L2_ch004_ep060_and_I_to_Wz_L2_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch002_and_I_to_Wy_L2_ch002_ep060_and_I_to_Wz_L2_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L2_ch001_and_I_to_Wy_L2_ch001_ep060_and_I_to_Wz_L2_ch001_ep060.build().result_obj],
    [
        mae_block1.I_to_Wx_L3_ch128_and_I_to_Wy_L3_ch128_ep060_and_I_to_Wz_L3_ch128_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch064_and_I_to_Wy_L3_ch064_ep060_and_I_to_Wz_L3_ch064_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch032_and_I_to_Wy_L3_ch032_ep060_and_I_to_Wz_L3_ch032_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch016_and_I_to_Wy_L3_ch016_ep060_and_I_to_Wz_L3_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch008_and_I_to_Wy_L3_ch008_ep060_and_I_to_Wz_L3_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch004_and_I_to_Wy_L3_ch004_ep060_and_I_to_Wz_L3_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch002_and_I_to_Wy_L3_ch002_ep060_and_I_to_Wz_L3_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L3_ch001_and_I_to_Wy_L3_ch001_ep060_and_I_to_Wz_L3_ch001_ep060.build().result_obj],
    [
        mae_block1.I_to_Wx_L4_ch128_and_I_to_Wy_L4_ch128_ep060_and_I_to_Wz_L4_ch128_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch064_and_I_to_Wy_L4_ch064_ep060_and_I_to_Wz_L4_ch064_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch032_and_I_to_Wy_L4_ch032_ep060_and_I_to_Wz_L4_ch032_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch016_and_I_to_Wy_L4_ch016_ep060_and_I_to_Wz_L4_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch008_and_I_to_Wy_L4_ch008_ep060_and_I_to_Wz_L4_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch004_and_I_to_Wy_L4_ch004_ep060_and_I_to_Wz_L4_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch002_and_I_to_Wy_L4_ch002_ep060_and_I_to_Wz_L4_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L4_ch001_and_I_to_Wy_L4_ch001_ep060_and_I_to_Wz_L4_ch001_ep060.build().result_obj],
    [
        mae_block1_limit.I_to_Wx_L5_ch128_lim_and_I_to_Wy_L5_ch128_lim_ep060_and_I_to_Wz_L5_ch128_lim_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch064_and_I_to_Wy_L5_ch064_ep060_and_I_to_Wz_L5_ch064_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch032_and_I_to_Wy_L5_ch032_ep060_and_I_to_Wz_L5_ch032_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch016_and_I_to_Wy_L5_ch016_ep060_and_I_to_Wz_L5_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch008_and_I_to_Wy_L5_ch008_ep060_and_I_to_Wz_L5_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch004_and_I_to_Wy_L5_ch004_ep060_and_I_to_Wz_L5_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch002_and_I_to_Wy_L5_ch002_ep060_and_I_to_Wz_L5_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L5_ch001_and_I_to_Wy_L5_ch001_ep060_and_I_to_Wz_L5_ch001_ep060.build().result_obj],
    [
        mae_block1_limit.I_to_Wx_L6_ch128_lim_and_I_to_Wy_L6_ch128_lim_ep060_and_I_to_Wz_L6_ch128_lim_ep060.build().result_obj,
        mae_block1_limit.I_to_Wx_L6_ch064_lim_and_I_to_Wy_L6_ch064_lim_ep060_and_I_to_Wz_L6_ch064_lim_ep060.build().result_obj,
        mae_block1.I_to_Wx_L6_ch032_and_I_to_Wy_L6_ch032_ep060_and_I_to_Wz_L6_ch032_ep060.build().result_obj,
        mae_block1.I_to_Wx_L6_ch016_and_I_to_Wy_L6_ch016_ep060_and_I_to_Wz_L6_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L6_ch008_and_I_to_Wy_L6_ch008_ep060_and_I_to_Wz_L6_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L6_ch004_and_I_to_Wy_L6_ch004_ep060_and_I_to_Wz_L6_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L6_ch002_and_I_to_Wy_L6_ch002_ep060_and_I_to_Wz_L6_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L6_ch001_and_I_to_Wy_L6_ch001_ep060_and_I_to_Wz_L6_ch001_ep060.build().result_obj],
    [
        mae_block1_limit.I_to_Wx_L7_ch128_lim_and_I_to_Wy_L7_ch128_lim_ep060_and_I_to_Wz_L7_ch128_lim_ep060.build().result_obj,
        mae_block1_limit.I_to_Wx_L7_ch064_lim_and_I_to_Wy_L7_ch064_lim_ep060_and_I_to_Wz_L7_ch064_lim_ep060.build().result_obj,
        mae_block1.I_to_Wx_L7_ch032_and_I_to_Wy_L7_ch032_ep060_and_I_to_Wz_L7_ch032_ep060.build().result_obj,
        mae_block1.I_to_Wx_L7_ch016_and_I_to_Wy_L7_ch016_ep060_and_I_to_Wz_L7_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L7_ch008_and_I_to_Wy_L7_ch008_ep060_and_I_to_Wz_L7_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L7_ch004_and_I_to_Wy_L7_ch004_ep060_and_I_to_Wz_L7_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L7_ch002_and_I_to_Wy_L7_ch002_ep060_and_I_to_Wz_L7_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L7_ch001_and_I_to_Wy_L7_ch001_ep060_and_I_to_Wz_L7_ch001_ep060.build().result_obj],
    [
        mae_block1_limit.I_to_Wx_L8_ch128_lim_and_I_to_Wy_L8_ch128_lim_ep060_and_I_to_Wz_L8_ch128_lim_ep060.build().result_obj,
        mae_block1_limit.I_to_Wx_L8_ch064_lim_and_I_to_Wy_L8_ch064_lim_ep060_and_I_to_Wz_L8_ch064_lim_ep060.build().result_obj,
        mae_block1_limit.I_to_Wx_L8_ch032_lim_and_I_to_Wy_L8_ch032_lim_ep060_and_I_to_Wz_L8_ch032_lim_ep060.build().result_obj,
        mae_block1.I_to_Wx_L8_ch016_and_I_to_Wy_L8_ch016_ep060_and_I_to_Wz_L8_ch016_ep060.build().result_obj,
        mae_block1.I_to_Wx_L8_ch008_and_I_to_Wy_L8_ch008_ep060_and_I_to_Wz_L8_ch008_ep060.build().result_obj,
        mae_block1.I_to_Wx_L8_ch004_and_I_to_Wy_L8_ch004_ep060_and_I_to_Wz_L8_ch004_ep060.build().result_obj,
        mae_block1.I_to_Wx_L8_ch002_and_I_to_Wy_L8_ch002_ep060_and_I_to_Wz_L8_ch002_ep060.build().result_obj,
        mae_block1.I_to_Wx_L8_ch001_and_I_to_Wy_L8_ch001_ep060_and_I_to_Wz_L8_ch001_ep060.build().result_obj],
]
