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
import Exps_7_v3.W_w_Mgt_to_Cx_Cy_focus.pyramid_tight_crop_size256_pad60_jit15.pyr_0s.bce_s001_tv_s0p1_L3.step10_a as L3_0side
import Exps_7_v3.W_w_Mgt_to_Cx_Cy_focus.pyramid_tight_crop_size256_pad60_jit15.pyr_1s.bce_s001_tv_s0p1_L3.step10_a as L3_1side
import Exps_7_v3.W_w_Mgt_to_Cx_Cy_focus.pyramid_tight_crop_size256_pad60_jit15.pyr_2s.bce_s001_tv_s0p1_L3.step10_a as L3_2side
import Exps_7_v3.W_w_Mgt_to_Cx_Cy_focus.pyramid_tight_crop_size256_pad60_jit15.pyr_3s.bce_s001_tv_s0p1_L3.step10_a as L3_3side
import Exps_7_v3.W_w_Mgt_to_Cx_Cy_focus.pyramid_tight_crop_size256_pad60_jit15.pyr_4s.bce_s001_tv_s0p1_L3.step10_a as L3_4side
import step10_a as L3_5side
#################################################################################################################################################################################################################################################################################################################################################################################################
########
# 1side_1
########
### 2side_1
# 3side_1
ch032_1side_1_2side_1_3side_1_45side_all = [
    [L3_3side.ch032_1side_1__2side_1__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_1__2side_1__3side_1_4side_1, L3_5side.ch032_1side_1__2side_1__3side_1_4side_1_5s1 , ],
]

########
# 1side_2
########
### 2side_1
# 3side_1
ch032_1side_2_2side_1_3side_1_45side_all = [
    [L3_3side.ch032_1side_2__2side_1__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_2__2side_1__3side_1_4side_1, L3_5side.ch032_1side_2__2side_1__3side_1_4side_1_5s1 , ],
]
### 2side_2
# 3side_1
ch032_1side_2_2side_2_3side_1_45side_all = [
    [L3_3side.ch032_1side_2__2side_2__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_2__2side_2__3side_1_4side_1, L3_5side.ch032_1side_2__2side_2__3side_1_4side_1_5s1 , ],
]
# 3side_2
ch032_1side_2_2side_2_3side_2_45side_all = [
    [L3_3side.ch032_1side_2__2side_2__3side_2        , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_2__2side_2__3side_2_4side_1, L3_5side.ch032_1side_2__2side_2__3side_2_4side_1_5s1 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_2__2side_2__3side_2_4side_2, L3_5side.ch032_1side_2__2side_2__3side_2_4side_2_5s1 , L3_5side.ch032_1side_2__2side_2__3side_2_4side_2_5s2 , ],
]

########
# 1side_3
########
### 2side_1
# 3side_1
ch032_1side_3_2side_1_3side_1_45side_all = [
    [L3_3side.ch032_1side_3__2side_1__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_1__3side_1_4side_1, L3_5side.ch032_1side_3__2side_1__3side_1_4side_1_5s1 , ],
]
### 2side_2
# 3side_1
ch032_1side_3_2side_2_3side_1_45side_all = [
    [L3_3side.ch032_1side_3__2side_2__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_2__3side_1_4side_1, L3_5side.ch032_1side_3__2side_2__3side_1_4side_1_5s1 , ],
]
# 3side_2
ch032_1side_3_2side_2_3side_2_45side_all = [
    [L3_3side.ch032_1side_3__2side_2__3side_2        , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_2__3side_2_4side_1, L3_5side.ch032_1side_3__2side_2__3side_2_4side_1_5s1 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_2__3side_2_4side_2, L3_5side.ch032_1side_3__2side_2__3side_2_4side_2_5s1 , L3_5side.ch032_1side_3__2side_2__3side_2_4side_2_5s2 , ],
]
### 2side_3
# 3side_1
ch032_1side_3_2side_3_3side_1_45side_all = [
    [L3_3side.ch032_1side_3__2side_3__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_3__3side_1_4side_1, L3_5side.ch032_1side_3__2side_3__3side_1_4side_1_5s1 , ],
]
# 3side_2
ch032_1side_3_2side_3_3side_2_45side_all = [
    [L3_3side.ch032_1side_3__2side_3__3side_2        , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_3__3side_2_4side_1, L3_5side.ch032_1side_3__2side_3__3side_2_4side_1_5s1 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_3__3side_2_4side_2, L3_5side.ch032_1side_3__2side_3__3side_2_4side_2_5s1 , L3_5side.ch032_1side_3__2side_3__3side_2_4side_2_5s2 , ],
]
# 3side_3
ch032_1side_3_2side_3_3side_3_45side_all = [
    [L3_3side.ch032_1side_3__2side_3__3side_3        , L3_5side.empty                                       , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_3__3side_3_4side_1, L3_5side.ch032_1side_3__2side_3__3side_3_4side_1_5s1 , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_3__3side_3_4side_2, L3_5side.ch032_1side_3__2side_3__3side_3_4side_2_5s1 , L3_5side.ch032_1side_3__2side_3__3side_3_4side_2_5s2 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_3__2side_3__3side_3_4side_3, L3_5side.ch032_1side_3__2side_3__3side_3_4side_3_5s1 , L3_5side.ch032_1side_3__2side_3__3side_3_4side_3_5s2 , L3_5side.ch032_1side_3__2side_3__3side_3_4side_3_5s3 , ],
]

########
# 1side_4
########
### 2side_1
# 3side_1
ch032_1side_4_2side_1_3side_1_45side_all = [
    [L3_3side.ch032_1side_4__2side_1__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_1__3side_1_4side_1, L3_5side.ch032_1side_4__2side_1__3side_1_4side_1_5s1 , ],
]
### 2side_2
# 3side_1
ch032_1side_4_2side_2_3side_1_45side_all = [
    [L3_3side.ch032_1side_4__2side_2__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_2__3side_1_4side_1, L3_5side.ch032_1side_4__2side_2__3side_1_4side_1_5s1 , ],
]
# 3side_2
ch032_1side_4_2side_2_3side_2_45side_all = [
    [L3_3side.ch032_1side_4__2side_2__3side_2        , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_2__3side_2_4side_1, L3_5side.ch032_1side_4__2side_2__3side_2_4side_1_5s1 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_2__3side_2_4side_2, L3_5side.ch032_1side_4__2side_2__3side_2_4side_2_5s1 , L3_5side.ch032_1side_4__2side_2__3side_2_4side_2_5s2 , ],
]
### 2side_3
# 3side_1
ch032_1side_4_2side_3_3side_1_45side_all = [
    [L3_3side.ch032_1side_4__2side_3__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_3__3side_1_4side_1, L3_5side.ch032_1side_4__2side_3__3side_1_4side_1_5s1 , ],
]
# 3side_2
ch032_1side_4_2side_3_3side_2_45side_all = [
    [L3_3side.ch032_1side_4__2side_3__3side_2        , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_3__3side_2_4side_1, L3_5side.ch032_1side_4__2side_3__3side_2_4side_1_5s1 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_3__3side_2_4side_2, L3_5side.ch032_1side_4__2side_3__3side_2_4side_2_5s1 , L3_5side.ch032_1side_4__2side_3__3side_2_4side_2_5s2 , ],
]
# 3side_3
ch032_1side_4_2side_3_3side_3_45side_all = [
    [L3_3side.ch032_1side_4__2side_3__3side_3        , L3_5side.empty                                       , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_3__3side_3_4side_1, L3_5side.ch032_1side_4__2side_3__3side_3_4side_1_5s1 , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_3__3side_3_4side_2, L3_5side.ch032_1side_4__2side_3__3side_3_4side_2_5s1 , L3_5side.ch032_1side_4__2side_3__3side_3_4side_2_5s2 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_3__3side_3_4side_3, L3_5side.ch032_1side_4__2side_3__3side_3_4side_3_5s1 , L3_5side.ch032_1side_4__2side_3__3side_3_4side_3_5s2 , L3_5side.ch032_1side_4__2side_3__3side_3_4side_3_5s3 , ],
]
### 2side_4
# 3side_1
ch032_1side_4_2side_4_3side_1_45side_all = [
    [L3_3side.ch032_1side_4__2side_4__3side_1        , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_1_4side_1, L3_5side.ch032_1side_4__2side_4__3side_1_4side_1_5s1 , ],
]
# 3side_2
ch032_1side_4_2side_4_3side_2_45side_all = [
    [L3_3side.ch032_1side_4__2side_4__3side_2        , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_2_4side_1, L3_5side.ch032_1side_4__2side_4__3side_2_4side_1_5s1 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_2_4side_2, L3_5side.ch032_1side_4__2side_4__3side_2_4side_2_5s1 , L3_5side.ch032_1side_4__2side_4__3side_2_4side_2_5s2 , ],
]
# 3side_3
ch032_1side_4_2side_4_3side_3_45side_all = [
    [L3_3side.ch032_1side_4__2side_4__3side_3        , L3_5side.empty                                       , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_3_4side_1, L3_5side.ch032_1side_4__2side_4__3side_3_4side_1_5s1 , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_3_4side_2, L3_5side.ch032_1side_4__2side_4__3side_3_4side_2_5s1 , L3_5side.ch032_1side_4__2side_4__3side_3_4side_2_5s2 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_3_4side_3, L3_5side.ch032_1side_4__2side_4__3side_3_4side_3_5s1 , L3_5side.ch032_1side_4__2side_4__3side_3_4side_3_5s2 , L3_5side.ch032_1side_4__2side_4__3side_3_4side_3_5s3 , ],
]
# 3side_4
ch032_1side_4_2side_4_3side_4_45side_all = [
    [L3_3side.ch032_1side_4__2side_4__3side_4        , L3_5side.empty                                       , L3_5side.empty                                       , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_4_4side_1, L3_5side.ch032_1side_4__2side_4__3side_4_4side_1_5s1 , L3_5side.empty                                       , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_4_4side_2, L3_5side.ch032_1side_4__2side_4__3side_4_4side_2_5s1 , L3_5side.ch032_1side_4__2side_4__3side_4_4side_2_5s2 , L3_5side.empty                                       , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_4_4side_3, L3_5side.ch032_1side_4__2side_4__3side_4_4side_3_5s1 , L3_5side.ch032_1side_4__2side_4__3side_4_4side_3_5s2 , L3_5side.ch032_1side_4__2side_4__3side_4_4side_3_5s3 , L3_5side.empty                                       , ],
    [L3_4side.ch032_1side_4__2side_4__3side_4_4side_4, L3_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s1 , L3_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s2 , L3_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s3 , L3_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s4 , ],
]