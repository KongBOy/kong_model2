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

import step10_a as side3
import Exps_7_v3.I_to_M_Gk3_no_pad.pyramid_2side.bce_s001_tv_s0p1_L3.step10_a as L3_1_2_side
#################################################################################################################################################################################################################################################################################################################################################################################################
ch032_1side_1__2_3_side_all = [
    [L3_1_2_side.ch032_1side_1__2side_0, L3_1_2_side.ch032_1side_1__2side_1   , ],
    [side3.empty                       , side3.ch032_1side_1__2side_1__3side_1, ],
]

ch032_1side_2__2_3_side_all = [
    [L3_1_2_side.ch032_1side_2__2side_0, L3_1_2_side.ch032_1side_2__2side_1   , L3_1_2_side.ch032_1side_2__2side_2   , ],
    [side3.empty                       , side3.ch032_1side_2__2side_1__3side_1, side3.ch032_1side_2__2side_2__3side_1, ],
    [side3.empty                       , side3.empty                          , side3.ch032_1side_2__2side_2__3side_2, ],
]

ch032_1side_3__2_3_side_all = [
    [L3_1_2_side.ch032_1side_3__2side_0, L3_1_2_side.ch032_1side_3__2side_1   , L3_1_2_side.ch032_1side_3__2side_2    , L3_1_2_side.ch032_1side_3__2side_3   , ],
    [side3.empty                       , side3.ch032_1side_3__2side_1__3side_1, side3.ch032_1side_3__2side_2__3side_1 , side3.ch032_1side_3__2side_3__3side_1, ],
    [side3.empty                       , side3.empty                          , side3.ch032_1side_3__2side_2__3side_2 , side3.ch032_1side_3__2side_3__3side_2, ],
    [side3.empty                       , side3.empty                          , side3.empty                           , side3.ch032_1side_3__2side_3__3side_3, ],
]

ch032_1side_4__2_3_side_all = [
    [L3_1_2_side.ch032_1side_4__2side_0, L3_1_2_side.ch032_1side_4__2side_1   , L3_1_2_side.ch032_1side_4__2side_2    , L3_1_2_side.ch032_1side_4__2side_3   , L3_1_2_side.ch032_1side_4__2side_4   , ],
    [side3.empty                       , side3.ch032_1side_4__2side_1__3side_1, side3.ch032_1side_4__2side_2__3side_1 , side3.ch032_1side_4__2side_3__3side_1, side3.ch032_1side_4__2side_4__3side_1, ],
    [side3.empty                       , side3.empty                          , side3.ch032_1side_4__2side_2__3side_2 , side3.ch032_1side_4__2side_3__3side_2, side3.ch032_1side_4__2side_4__3side_2, ],
    [side3.empty                       , side3.empty                          , side3.empty                           , side3.ch032_1side_4__2side_3__3side_3, side3.ch032_1side_4__2side_4__3side_3, ],
    [side3.empty                       , side3.empty                          , side3.empty                           , side3.empty                          , side3.ch032_1side_4__2side_4__3side_4, ],
]
