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
import Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop256_pad20_jit15.pyr_0s.L7.step10_a as L7_0side
import Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop256_pad20_jit15.pyr_1s.L7.step10_a as L7_1side
import Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop256_pad20_jit15.pyr_2s.L7.step10_a as L7_2side
import Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop256_pad20_jit15.pyr_3s.L7.step10_a as L7_3side
import step10_a as L7_4side
#################################################################################################################################################################################################################################################################################################################################################################################################
########
# 1side_1
########
### 2side_1
ch032_1side_1_2side_1_34side_all = [
    [L7_2side.ch032_1side_1__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_1__2side_1__3side_1, L7_4side.ch032_1side_1__2side_1__3side_1_4side_1 , ],
]

########
# 1side_2
########
### 2side_1
ch032_1side_2_2side_1_34side_all = [
    [L7_2side.ch032_1side_2__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_2__2side_1__3side_1, L7_4side.ch032_1side_2__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_2_2side_2_34side_all = [
    [L7_2side.ch032_1side_2__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_2__2side_2__3side_1, L7_4side.ch032_1side_2__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_2__2side_2__3side_2, L7_4side.ch032_1side_2__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_2__2side_2__3side_2_4side_2 , ],
]

########
# 1side_3
########
### 2side_1
ch032_1side_3_2side_1_34side_all = [
    [L7_2side.ch032_1side_3__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_3__2side_1__3side_1, L7_4side.ch032_1side_3__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_3_2side_2_34side_all = [
    [L7_2side.ch032_1side_3__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_3__2side_2__3side_1, L7_4side.ch032_1side_3__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_3__2side_2__3side_2, L7_4side.ch032_1side_3__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_3__2side_2__3side_2_4side_2 , ],
]
### 2side_3
ch032_1side_3_2side_3_34side_all = [
    [L7_2side.ch032_1side_3__2side_3         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_3__2side_3__3side_1, L7_4side.ch032_1side_3__2side_3__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_3__2side_3__3side_2, L7_4side.ch032_1side_3__2side_3__3side_2_4side_1 , L7_4side.ch032_1side_3__2side_3__3side_2_4side_2 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_3__2side_3__3side_3, L7_4side.ch032_1side_3__2side_3__3side_3_4side_1 , L7_4side.ch032_1side_3__2side_3__3side_3_4side_2 , L7_4side.ch032_1side_3__2side_3__3side_3_4side_3 , ],
]

########
# 1side_4
########
### 2side_1
ch032_1side_4_2side_1_34side_all = [
    [L7_2side.ch032_1side_4__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_1__3side_1, L7_4side.ch032_1side_4__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_4_2side_2_34side_all = [
    [L7_2side.ch032_1side_4__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_2__3side_1, L7_4side.ch032_1side_4__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_2__3side_2, L7_4side.ch032_1side_4__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_4__2side_2__3side_2_4side_2 , ],
]
### 2side_3
ch032_1side_4_2side_3_34side_all = [
    [L7_2side.ch032_1side_4__2side_3         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_3__3side_1, L7_4side.ch032_1side_4__2side_3__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_3__3side_2, L7_4side.ch032_1side_4__2side_3__3side_2_4side_1 , L7_4side.ch032_1side_4__2side_3__3side_2_4side_2 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_3__3side_3, L7_4side.ch032_1side_4__2side_3__3side_3_4side_1 , L7_4side.ch032_1side_4__2side_3__3side_3_4side_2 , L7_4side.ch032_1side_4__2side_3__3side_3_4side_3 , ],
]
### 2side_4
ch032_1side_4_2side_4_34side_all = [
    [L7_2side.ch032_1side_4__2side_4         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_4__3side_1, L7_4side.ch032_1side_4__2side_4__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_4__3side_2, L7_4side.ch032_1side_4__2side_4__3side_2_4side_1 , L7_4side.ch032_1side_4__2side_4__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_4__3side_3, L7_4side.ch032_1side_4__2side_4__3side_3_4side_1 , L7_4side.ch032_1side_4__2side_4__3side_3_4side_2 , L7_4side.ch032_1side_4__2side_4__3side_3_4side_3 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_4__2side_4__3side_4, L7_4side.ch032_1side_4__2side_4__3side_4_4side_1 , L7_4side.ch032_1side_4__2side_4__3side_4_4side_2 , L7_4side.ch032_1side_4__2side_4__3side_4_4side_3 , L7_4side.ch032_1side_4__2side_4__3side_4_4side_4 , ],
]

########
# 1side_5
########
### 2side_1
ch032_1side_5_2side_1_34side_all = [
    [L7_2side.ch032_1side_5__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_1__3side_1, L7_4side.ch032_1side_5__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_5_2side_2_34side_all = [
    [L7_2side.ch032_1side_5__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_2__3side_1, L7_4side.ch032_1side_5__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_2__3side_2, L7_4side.ch032_1side_5__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_5__2side_2__3side_2_4side_2 , ],
]
### 2side_3
ch032_1side_5_2side_3_34side_all = [
    [L7_2side.ch032_1side_5__2side_3         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_3__3side_1, L7_4side.ch032_1side_5__2side_3__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_3__3side_2, L7_4side.ch032_1side_5__2side_3__3side_2_4side_1 , L7_4side.ch032_1side_5__2side_3__3side_2_4side_2 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_3__3side_3, L7_4side.ch032_1side_5__2side_3__3side_3_4side_1 , L7_4side.ch032_1side_5__2side_3__3side_3_4side_2 , L7_4side.ch032_1side_5__2side_3__3side_3_4side_3 , ],
]
### 2side_4
ch032_1side_5_2side_4_34side_all = [
    [L7_2side.ch032_1side_5__2side_4         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_4__3side_1, L7_4side.ch032_1side_5__2side_4__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_4__3side_2, L7_4side.ch032_1side_5__2side_4__3side_2_4side_1 , L7_4side.ch032_1side_5__2side_4__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_4__3side_3, L7_4side.ch032_1side_5__2side_4__3side_3_4side_1 , L7_4side.ch032_1side_5__2side_4__3side_3_4side_2 , L7_4side.ch032_1side_5__2side_4__3side_3_4side_3 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_4__3side_4, L7_4side.ch032_1side_5__2side_4__3side_4_4side_1 , L7_4side.ch032_1side_5__2side_4__3side_4_4side_2 , L7_4side.ch032_1side_5__2side_4__3side_4_4side_3 , L7_4side.ch032_1side_5__2side_4__3side_4_4side_4 , ],
]
### 2side_5
ch032_1side_5_2side_5_34side_all = [
    [L7_2side.ch032_1side_5__2side_5         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_5__3side_1, L7_4side.ch032_1side_5__2side_5__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_5__3side_2, L7_4side.ch032_1side_5__2side_5__3side_2_4side_1 , L7_4side.ch032_1side_5__2side_5__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_5__3side_3, L7_4side.ch032_1side_5__2side_5__3side_3_4side_1 , L7_4side.ch032_1side_5__2side_5__3side_3_4side_2 , L7_4side.ch032_1side_5__2side_5__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_5__3side_4, L7_4side.ch032_1side_5__2side_5__3side_4_4side_1 , L7_4side.ch032_1side_5__2side_5__3side_4_4side_2 , L7_4side.ch032_1side_5__2side_5__3side_4_4side_3 , L7_4side.ch032_1side_5__2side_5__3side_4_4side_4 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_5__2side_5__3side_5, L7_4side.ch032_1side_5__2side_5__3side_5_4side_1 , L7_4side.ch032_1side_5__2side_5__3side_5_4side_2 , L7_4side.ch032_1side_5__2side_5__3side_5_4side_3 , L7_4side.ch032_1side_5__2side_5__3side_5_4side_4 , L7_4side.ch032_1side_5__2side_5__3side_5_4side_5 , ],
]

########
# 1side_6
########
### 2side_1
ch032_1side_6_2side_1_34side_all = [
    [L7_2side.ch032_1side_6__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_1__3side_1, L7_4side.ch032_1side_6__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_6_2side_2_34side_all = [
    [L7_2side.ch032_1side_6__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_2__3side_1, L7_4side.ch032_1side_6__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_2__3side_2, L7_4side.ch032_1side_6__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_6__2side_2__3side_2_4side_2 , ],
]
### 2side_3
ch032_1side_6_2side_3_34side_all = [
    [L7_2side.ch032_1side_6__2side_3         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_3__3side_1, L7_4side.ch032_1side_6__2side_3__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_3__3side_2, L7_4side.ch032_1side_6__2side_3__3side_2_4side_1 , L7_4side.ch032_1side_6__2side_3__3side_2_4side_2 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_3__3side_3, L7_4side.ch032_1side_6__2side_3__3side_3_4side_1 , L7_4side.ch032_1side_6__2side_3__3side_3_4side_2 , L7_4side.ch032_1side_6__2side_3__3side_3_4side_3 , ],
]
### 2side_4
ch032_1side_6_2side_4_34side_all = [
    [L7_2side.ch032_1side_6__2side_4         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_4__3side_1, L7_4side.ch032_1side_6__2side_4__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_4__3side_2, L7_4side.ch032_1side_6__2side_4__3side_2_4side_1 , L7_4side.ch032_1side_6__2side_4__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_4__3side_3, L7_4side.ch032_1side_6__2side_4__3side_3_4side_1 , L7_4side.ch032_1side_6__2side_4__3side_3_4side_2 , L7_4side.ch032_1side_6__2side_4__3side_3_4side_3 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_4__3side_4, L7_4side.ch032_1side_6__2side_4__3side_4_4side_1 , L7_4side.ch032_1side_6__2side_4__3side_4_4side_2 , L7_4side.ch032_1side_6__2side_4__3side_4_4side_3 , L7_4side.ch032_1side_6__2side_4__3side_4_4side_4 , ],
]
### 2side_5
ch032_1side_6_2side_5_34side_all = [
    [L7_2side.ch032_1side_6__2side_5         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_5__3side_1, L7_4side.ch032_1side_6__2side_5__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_5__3side_2, L7_4side.ch032_1side_6__2side_5__3side_2_4side_1 , L7_4side.ch032_1side_6__2side_5__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_5__3side_3, L7_4side.ch032_1side_6__2side_5__3side_3_4side_1 , L7_4side.ch032_1side_6__2side_5__3side_3_4side_2 , L7_4side.ch032_1side_6__2side_5__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_5__3side_4, L7_4side.ch032_1side_6__2side_5__3side_4_4side_1 , L7_4side.ch032_1side_6__2side_5__3side_4_4side_2 , L7_4side.ch032_1side_6__2side_5__3side_4_4side_3 , L7_4side.ch032_1side_6__2side_5__3side_4_4side_4 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_5__3side_5, L7_4side.ch032_1side_6__2side_5__3side_5_4side_1 , L7_4side.ch032_1side_6__2side_5__3side_5_4side_2 , L7_4side.ch032_1side_6__2side_5__3side_5_4side_3 , L7_4side.ch032_1side_6__2side_5__3side_5_4side_4 , L7_4side.ch032_1side_6__2side_5__3side_5_4side_5 , ],
]
### 2side_6
ch032_1side_6_2side_6_34side_all = [
    [L7_2side.ch032_1side_6__2side_6         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_6__3side_1, L7_4side.ch032_1side_6__2side_6__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_6__3side_2, L7_4side.ch032_1side_6__2side_6__3side_2_4side_1 , L7_4side.ch032_1side_6__2side_6__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_6__3side_3, L7_4side.ch032_1side_6__2side_6__3side_3_4side_1 , L7_4side.ch032_1side_6__2side_6__3side_3_4side_2 , L7_4side.ch032_1side_6__2side_6__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_6__3side_4, L7_4side.ch032_1side_6__2side_6__3side_4_4side_1 , L7_4side.ch032_1side_6__2side_6__3side_4_4side_2 , L7_4side.ch032_1side_6__2side_6__3side_4_4side_3 , L7_4side.ch032_1side_6__2side_6__3side_4_4side_4 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_6__3side_5, L7_4side.ch032_1side_6__2side_6__3side_5_4side_1 , L7_4side.ch032_1side_6__2side_6__3side_5_4side_2 , L7_4side.ch032_1side_6__2side_6__3side_5_4side_3 , L7_4side.ch032_1side_6__2side_6__3side_5_4side_4 , L7_4side.ch032_1side_6__2side_6__3side_5_4side_5 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_6__2side_6__3side_6, L7_4side.ch032_1side_6__2side_6__3side_6_4side_1 , L7_4side.ch032_1side_6__2side_6__3side_6_4side_2 , L7_4side.ch032_1side_6__2side_6__3side_6_4side_3 , L7_4side.ch032_1side_6__2side_6__3side_6_4side_4 , L7_4side.ch032_1side_6__2side_6__3side_6_4side_5 , L7_4side.ch032_1side_6__2side_6__3side_6_4side_6 , ],
]

########
# 1side_7
########
### 2side_1
ch032_1side_7_2side_1_34side_all = [
    [L7_2side.ch032_1side_7__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_1__3side_1, L7_4side.ch032_1side_7__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_7_2side_2_34side_all = [
    [L7_2side.ch032_1side_7__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_2__3side_1, L7_4side.ch032_1side_7__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_2__3side_2, L7_4side.ch032_1side_7__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_7__2side_2__3side_2_4side_2 , ],
]
### 2side_3
ch032_1side_7_2side_3_34side_all = [
    [L7_2side.ch032_1side_7__2side_3         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_3__3side_1, L7_4side.ch032_1side_7__2side_3__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_3__3side_2, L7_4side.ch032_1side_7__2side_3__3side_2_4side_1 , L7_4side.ch032_1side_7__2side_3__3side_2_4side_2 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_3__3side_3, L7_4side.ch032_1side_7__2side_3__3side_3_4side_1 , L7_4side.ch032_1side_7__2side_3__3side_3_4side_2 , L7_4side.ch032_1side_7__2side_3__3side_3_4side_3 , ],
]
### 2side_4
ch032_1side_7_2side_4_34side_all = [
    [L7_2side.ch032_1side_7__2side_4         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_4__3side_1, L7_4side.ch032_1side_7__2side_4__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_4__3side_2, L7_4side.ch032_1side_7__2side_4__3side_2_4side_1 , L7_4side.ch032_1side_7__2side_4__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_4__3side_3, L7_4side.ch032_1side_7__2side_4__3side_3_4side_1 , L7_4side.ch032_1side_7__2side_4__3side_3_4side_2 , L7_4side.ch032_1side_7__2side_4__3side_3_4side_3 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_4__3side_4, L7_4side.ch032_1side_7__2side_4__3side_4_4side_1 , L7_4side.ch032_1side_7__2side_4__3side_4_4side_2 , L7_4side.ch032_1side_7__2side_4__3side_4_4side_3 , L7_4side.ch032_1side_7__2side_4__3side_4_4side_4 , ],
]
### 2side_5
ch032_1side_7_2side_5_34side_all = [
    [L7_2side.ch032_1side_7__2side_5         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_5__3side_1, L7_4side.ch032_1side_7__2side_5__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_5__3side_2, L7_4side.ch032_1side_7__2side_5__3side_2_4side_1 , L7_4side.ch032_1side_7__2side_5__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_5__3side_3, L7_4side.ch032_1side_7__2side_5__3side_3_4side_1 , L7_4side.ch032_1side_7__2side_5__3side_3_4side_2 , L7_4side.ch032_1side_7__2side_5__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_5__3side_4, L7_4side.ch032_1side_7__2side_5__3side_4_4side_1 , L7_4side.ch032_1side_7__2side_5__3side_4_4side_2 , L7_4side.ch032_1side_7__2side_5__3side_4_4side_3 , L7_4side.ch032_1side_7__2side_5__3side_4_4side_4 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_5__3side_5, L7_4side.ch032_1side_7__2side_5__3side_5_4side_1 , L7_4side.ch032_1side_7__2side_5__3side_5_4side_2 , L7_4side.ch032_1side_7__2side_5__3side_5_4side_3 , L7_4side.ch032_1side_7__2side_5__3side_5_4side_4 , L7_4side.ch032_1side_7__2side_5__3side_5_4side_5 , ],
]
### 2side_6
ch032_1side_7_2side_6_34side_all = [
    [L7_2side.ch032_1side_7__2side_6         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_6__3side_1, L7_4side.ch032_1side_7__2side_6__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_6__3side_2, L7_4side.ch032_1side_7__2side_6__3side_2_4side_1 , L7_4side.ch032_1side_7__2side_6__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_6__3side_3, L7_4side.ch032_1side_7__2side_6__3side_3_4side_1 , L7_4side.ch032_1side_7__2side_6__3side_3_4side_2 , L7_4side.ch032_1side_7__2side_6__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_6__3side_4, L7_4side.ch032_1side_7__2side_6__3side_4_4side_1 , L7_4side.ch032_1side_7__2side_6__3side_4_4side_2 , L7_4side.ch032_1side_7__2side_6__3side_4_4side_3 , L7_4side.ch032_1side_7__2side_6__3side_4_4side_4 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_6__3side_5, L7_4side.ch032_1side_7__2side_6__3side_5_4side_1 , L7_4side.ch032_1side_7__2side_6__3side_5_4side_2 , L7_4side.ch032_1side_7__2side_6__3side_5_4side_3 , L7_4side.ch032_1side_7__2side_6__3side_5_4side_4 , L7_4side.ch032_1side_7__2side_6__3side_5_4side_5 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_6__3side_6, L7_4side.ch032_1side_7__2side_6__3side_6_4side_1 , L7_4side.ch032_1side_7__2side_6__3side_6_4side_2 , L7_4side.ch032_1side_7__2side_6__3side_6_4side_3 , L7_4side.ch032_1side_7__2side_6__3side_6_4side_4 , L7_4side.ch032_1side_7__2side_6__3side_6_4side_5 , L7_4side.ch032_1side_7__2side_6__3side_6_4side_6 , ],
]
### 2side_7
ch032_1side_7_2side_7_34side_all = [
    [L7_2side.ch032_1side_7__2side_7         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_1, L7_4side.ch032_1side_7__2side_7__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_2, L7_4side.ch032_1side_7__2side_7__3side_2_4side_1 , L7_4side.ch032_1side_7__2side_7__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_3, L7_4side.ch032_1side_7__2side_7__3side_3_4side_1 , L7_4side.ch032_1side_7__2side_7__3side_3_4side_2 , L7_4side.ch032_1side_7__2side_7__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_4, L7_4side.ch032_1side_7__2side_7__3side_4_4side_1 , L7_4side.ch032_1side_7__2side_7__3side_4_4side_2 , L7_4side.ch032_1side_7__2side_7__3side_4_4side_3 , L7_4side.ch032_1side_7__2side_7__3side_4_4side_4 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_5, L7_4side.ch032_1side_7__2side_7__3side_5_4side_1 , L7_4side.ch032_1side_7__2side_7__3side_5_4side_2 , L7_4side.ch032_1side_7__2side_7__3side_5_4side_3 , L7_4side.ch032_1side_7__2side_7__3side_5_4side_4 , L7_4side.ch032_1side_7__2side_7__3side_5_4side_5 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_6, L7_4side.ch032_1side_7__2side_7__3side_6_4side_1 , L7_4side.ch032_1side_7__2side_7__3side_6_4side_2 , L7_4side.ch032_1side_7__2side_7__3side_6_4side_3 , L7_4side.ch032_1side_7__2side_7__3side_6_4side_4 , L7_4side.ch032_1side_7__2side_7__3side_6_4side_5 , L7_4side.ch032_1side_7__2side_7__3side_6_4side_6 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_7__2side_7__3side_7, L7_4side.ch032_1side_7__2side_7__3side_7_4side_1 , L7_4side.ch032_1side_7__2side_7__3side_7_4side_2 , L7_4side.ch032_1side_7__2side_7__3side_7_4side_3 , L7_4side.ch032_1side_7__2side_7__3side_7_4side_4 , L7_4side.ch032_1side_7__2side_7__3side_7_4side_5 , L7_4side.ch032_1side_7__2side_7__3side_7_4side_6 , L7_4side.ch032_1side_7__2side_7__3side_7_4side_7 , ],
]

########
# 1side_8
########
### 2side_1
ch032_1side_8_2side_1_34side_all = [
    [L7_2side.ch032_1side_8__2side_1         , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_1__3side_1, L7_4side.ch032_1side_8__2side_1__3side_1_4side_1 , ],
]
### 2side_2
ch032_1side_8_2side_2_34side_all = [
    [L7_2side.ch032_1side_8__2side_2         , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_2__3side_1, L7_4side.ch032_1side_8__2side_2__3side_1_4side_1 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_2__3side_2, L7_4side.ch032_1side_8__2side_2__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_2__3side_2_4side_2 , ],
]
### 2side_3
ch032_1side_8_2side_3_34side_all = [
    [L7_2side.ch032_1side_8__2side_3         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_3__3side_1, L7_4side.ch032_1side_8__2side_3__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_3__3side_2, L7_4side.ch032_1side_8__2side_3__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_3__3side_2_4side_2 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_3__3side_3, L7_4side.ch032_1side_8__2side_3__3side_3_4side_1 , L7_4side.ch032_1side_8__2side_3__3side_3_4side_2 , L7_4side.ch032_1side_8__2side_3__3side_3_4side_3 , ],
]
### 2side_4
ch032_1side_8_2side_4_34side_all = [
    [L7_2side.ch032_1side_8__2side_4         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_4__3side_1, L7_4side.ch032_1side_8__2side_4__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_4__3side_2, L7_4side.ch032_1side_8__2side_4__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_4__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_4__3side_3, L7_4side.ch032_1side_8__2side_4__3side_3_4side_1 , L7_4side.ch032_1side_8__2side_4__3side_3_4side_2 , L7_4side.ch032_1side_8__2side_4__3side_3_4side_3 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_4__3side_4, L7_4side.ch032_1side_8__2side_4__3side_4_4side_1 , L7_4side.ch032_1side_8__2side_4__3side_4_4side_2 , L7_4side.ch032_1side_8__2side_4__3side_4_4side_3 , L7_4side.ch032_1side_8__2side_4__3side_4_4side_4 , ],
]
### 2side_5
ch032_1side_8_2side_5_34side_all = [
    [L7_2side.ch032_1side_8__2side_5         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_5__3side_1, L7_4side.ch032_1side_8__2side_5__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_5__3side_2, L7_4side.ch032_1side_8__2side_5__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_5__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_5__3side_3, L7_4side.ch032_1side_8__2side_5__3side_3_4side_1 , L7_4side.ch032_1side_8__2side_5__3side_3_4side_2 , L7_4side.ch032_1side_8__2side_5__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_5__3side_4, L7_4side.ch032_1side_8__2side_5__3side_4_4side_1 , L7_4side.ch032_1side_8__2side_5__3side_4_4side_2 , L7_4side.ch032_1side_8__2side_5__3side_4_4side_3 , L7_4side.ch032_1side_8__2side_5__3side_4_4side_4 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_5__3side_5, L7_4side.ch032_1side_8__2side_5__3side_5_4side_1 , L7_4side.ch032_1side_8__2side_5__3side_5_4side_2 , L7_4side.ch032_1side_8__2side_5__3side_5_4side_3 , L7_4side.ch032_1side_8__2side_5__3side_5_4side_4 , L7_4side.ch032_1side_8__2side_5__3side_5_4side_5 , ],
]
### 2side_6
ch032_1side_8_2side_6_34side_all = [
    [L7_2side.ch032_1side_8__2side_6         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_6__3side_1, L7_4side.ch032_1side_8__2side_6__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_6__3side_2, L7_4side.ch032_1side_8__2side_6__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_6__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_6__3side_3, L7_4side.ch032_1side_8__2side_6__3side_3_4side_1 , L7_4side.ch032_1side_8__2side_6__3side_3_4side_2 , L7_4side.ch032_1side_8__2side_6__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_6__3side_4, L7_4side.ch032_1side_8__2side_6__3side_4_4side_1 , L7_4side.ch032_1side_8__2side_6__3side_4_4side_2 , L7_4side.ch032_1side_8__2side_6__3side_4_4side_3 , L7_4side.ch032_1side_8__2side_6__3side_4_4side_4 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_6__3side_5, L7_4side.ch032_1side_8__2side_6__3side_5_4side_1 , L7_4side.ch032_1side_8__2side_6__3side_5_4side_2 , L7_4side.ch032_1side_8__2side_6__3side_5_4side_3 , L7_4side.ch032_1side_8__2side_6__3side_5_4side_4 , L7_4side.ch032_1side_8__2side_6__3side_5_4side_5 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_6__3side_6, L7_4side.ch032_1side_8__2side_6__3side_6_4side_1 , L7_4side.ch032_1side_8__2side_6__3side_6_4side_2 , L7_4side.ch032_1side_8__2side_6__3side_6_4side_3 , L7_4side.ch032_1side_8__2side_6__3side_6_4side_4 , L7_4side.ch032_1side_8__2side_6__3side_6_4side_5 , L7_4side.ch032_1side_8__2side_6__3side_6_4side_6 , ],
]
### 2side_7
ch032_1side_8_2side_7_34side_all = [
    [L7_2side.ch032_1side_8__2side_7         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_1, L7_4side.ch032_1side_8__2side_7__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_2, L7_4side.ch032_1side_8__2side_7__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_7__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_3, L7_4side.ch032_1side_8__2side_7__3side_3_4side_1 , L7_4side.ch032_1side_8__2side_7__3side_3_4side_2 , L7_4side.ch032_1side_8__2side_7__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_4, L7_4side.ch032_1side_8__2side_7__3side_4_4side_1 , L7_4side.ch032_1side_8__2side_7__3side_4_4side_2 , L7_4side.ch032_1side_8__2side_7__3side_4_4side_3 , L7_4side.ch032_1side_8__2side_7__3side_4_4side_4 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_5, L7_4side.ch032_1side_8__2side_7__3side_5_4side_1 , L7_4side.ch032_1side_8__2side_7__3side_5_4side_2 , L7_4side.ch032_1side_8__2side_7__3side_5_4side_3 , L7_4side.ch032_1side_8__2side_7__3side_5_4side_4 , L7_4side.ch032_1side_8__2side_7__3side_5_4side_5 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_6, L7_4side.ch032_1side_8__2side_7__3side_6_4side_1 , L7_4side.ch032_1side_8__2side_7__3side_6_4side_2 , L7_4side.ch032_1side_8__2side_7__3side_6_4side_3 , L7_4side.ch032_1side_8__2side_7__3side_6_4side_4 , L7_4side.ch032_1side_8__2side_7__3side_6_4side_5 , L7_4side.ch032_1side_8__2side_7__3side_6_4side_6 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_7__3side_7, L7_4side.ch032_1side_8__2side_7__3side_7_4side_1 , L7_4side.ch032_1side_8__2side_7__3side_7_4side_2 , L7_4side.ch032_1side_8__2side_7__3side_7_4side_3 , L7_4side.ch032_1side_8__2side_7__3side_7_4side_4 , L7_4side.ch032_1side_8__2side_7__3side_7_4side_5 , L7_4side.ch032_1side_8__2side_7__3side_7_4side_6 , L7_4side.ch032_1side_8__2side_7__3side_7_4side_7 , ],
]
### 2side_8
ch032_1side_8_2side_8_34side_all = [
    [L7_2side.ch032_1side_8__2side_8         , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_1, L7_4side.ch032_1side_8__2side_8__3side_1_4side_1 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_2, L7_4side.ch032_1side_8__2side_8__3side_2_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_2_4side_2 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_3, L7_4side.ch032_1side_8__2side_8__3side_3_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_3_4side_2 , L7_4side.ch032_1side_8__2side_8__3side_3_4side_3 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_4, L7_4side.ch032_1side_8__2side_8__3side_4_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_4_4side_2 , L7_4side.ch032_1side_8__2side_8__3side_4_4side_3 , L7_4side.ch032_1side_8__2side_8__3side_4_4side_4 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_5, L7_4side.ch032_1side_8__2side_8__3side_5_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_5_4side_2 , L7_4side.ch032_1side_8__2side_8__3side_5_4side_3 , L7_4side.ch032_1side_8__2side_8__3side_5_4side_4 , L7_4side.ch032_1side_8__2side_8__3side_5_4side_5 , L7_4side.empty                                   , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_6, L7_4side.ch032_1side_8__2side_8__3side_6_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_6_4side_2 , L7_4side.ch032_1side_8__2side_8__3side_6_4side_3 , L7_4side.ch032_1side_8__2side_8__3side_6_4side_4 , L7_4side.ch032_1side_8__2side_8__3side_6_4side_5 , L7_4side.ch032_1side_8__2side_8__3side_6_4side_6 , L7_4side.empty                                   , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_7, L7_4side.ch032_1side_8__2side_8__3side_7_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_7_4side_2 , L7_4side.ch032_1side_8__2side_8__3side_7_4side_3 , L7_4side.ch032_1side_8__2side_8__3side_7_4side_4 , L7_4side.ch032_1side_8__2side_8__3side_7_4side_5 , L7_4side.ch032_1side_8__2side_8__3side_7_4side_6 , L7_4side.ch032_1side_8__2side_8__3side_7_4side_7 , L7_4side.empty                                   , ],
    [L7_3side.ch032_1side_8__2side_8__3side_8, L7_4side.ch032_1side_8__2side_8__3side_8_4side_1 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_2 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_3 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_4 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_5 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_6 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_7 , L7_4side.ch032_1side_8__2side_8__3side_8_4side_5 , ],
]
