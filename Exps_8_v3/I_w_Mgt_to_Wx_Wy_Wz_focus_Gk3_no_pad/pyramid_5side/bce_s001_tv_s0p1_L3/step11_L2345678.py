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
import Exps_7_v3.I_to_M_Gk3_no_pad.pyramid_0side.bce_s001_tv_s0p1_L6.step10_a as L6_0side
import Exps_7_v3.I_to_M_Gk3_no_pad.pyramid_1side.bce_s001_tv_s0p1_L6.step10_a as L6_1side
import Exps_7_v3.I_to_M_Gk3_no_pad.pyramid_2side.bce_s001_tv_s0p1_L6.step10_a as L6_2side
import Exps_7_v3.I_to_M_Gk3_no_pad.pyramid_3side.bce_s001_tv_s0p1_L6.step10_a as L6_3side
import Exps_7_v3.I_to_M_Gk3_no_pad.pyramid_4side.bce_s001_tv_s0p1_L6.step10_a as L6_4side
import step10_a as L6_5side
#################################################################################################################################################################################################################################################################################################################################################################################################
##################################
### 5side0
##################################
# "1" (3 3side以前的結果也順便一起看吧)  6 10 15 21 28 36 45 55
# side1 OK 1
ch032_1side_1__2_3_side_all__4side_0_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_1 , L6_2side.ch032_1side_1__2side_1                  , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_3side.ch032_1side_1__2side_1__3side_1         , ],
]

ch032_1side_1__2_3_side_all__4side_1_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_1 , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.ch032_1side_1__2side_1__3side_1_4side_1 , ],
]

# 1 "3" (6 3side以前的結果也順便一起看吧)  10 15 21 28 36 45 55
# side2 OK 4
ch032_1side_2__2_3_side_all__4side_0_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_2 , L6_2side.ch032_1side_2__2side_1                  , L6_2side.ch032_1side_2__2side_2                  , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_3side.ch032_1side_2__2side_1__3side_1         , L6_3side.ch032_1side_2__2side_2__3side_1         , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_3side.empty                                   , L6_3side.ch032_1side_2__2side_2__3side_2         , ],
]

ch032_1side_2__2_3_side_all__4side_1_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_2 , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.ch032_1side_2__2side_1__3side_1_4side_1 , L6_4side.ch032_1side_2__2side_2__3side_1_4side_1 , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.ch032_1side_2__2side_2__3side_2_4side_1 , ],
]
ch032_1side_2__2_3_side_all__4side_2_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_2 , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.ch032_1side_2__2side_2__3side_2_4side_2 , ],
]

# 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
# side3 OK 10
ch032_1side_3__2_3_side_all__4side_0_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_2side.ch032_1side_3__2side_1                  , L6_2side.ch032_1side_3__2side_2                  , L6_2side.ch032_1side_3__2side_3         , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_3side.ch032_1side_3__2side_1__3side_1         , L6_3side.ch032_1side_3__2side_2__3side_1         , L6_3side.ch032_1side_3__2side_3__3side_1, ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_3side.empty                                   , L6_3side.ch032_1side_3__2side_2__3side_2         , L6_3side.ch032_1side_3__2side_3__3side_2, ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_3side.empty                                   , L6_3side.empty                                   , L6_3side.ch032_1side_3__2side_3__3side_3, ],
]

ch032_1side_3__2_3_side_all__4side_1_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                  , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.ch032_1side_3__2side_1__3side_1_4side_1 , L6_4side.ch032_1side_3__2side_2__3side_1_4side_1 , L6_4side.ch032_1side_3__2side_3__3side_1_4side_1, ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.ch032_1side_3__2side_2__3side_2_4side_1 , L6_4side.ch032_1side_3__2side_3__3side_2_4side_1, ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_3__2side_3__3side_3_4side_1, ],
]
ch032_1side_3__2_3_side_all__4side_2_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                  , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                  , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.ch032_1side_3__2side_2__3side_2_4side_2 , L6_4side.ch032_1side_3__2side_3__3side_2_4side_2, ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_3__2side_3__3side_3_4side_2, ],
]
ch032_1side_3__2_3_side_all__4side_3_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                  , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                  , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                  , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_3__2side_3__3side_3_4side_3, ],
]

# 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
# L6_4side OK 20
ch032_1side_4__2_3_side_all__4side_0_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_2side.ch032_1side_4__2side_1                  , L6_2side.ch032_1side_4__2side_2                  , L6_2side.ch032_1side_4__2side_3                  , L6_2side.ch032_1side_4__2side_4                  , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_3side.ch032_1side_4__2side_1__3side_1         , L6_3side.ch032_1side_4__2side_2__3side_1         , L6_3side.ch032_1side_4__2side_3__3side_1         , L6_3side.ch032_1side_4__2side_4__3side_1         , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_3side.empty                                   , L6_3side.ch032_1side_4__2side_2__3side_2         , L6_3side.ch032_1side_4__2side_3__3side_2         , L6_3side.ch032_1side_4__2side_4__3side_2         , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_3side.empty                                   , L6_3side.empty                                   , L6_3side.ch032_1side_4__2side_3__3side_3         , L6_3side.ch032_1side_4__2side_4__3side_3         , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_3side.empty                                   , L6_3side.empty                                   , L6_3side.empty                                   , L6_3side.ch032_1side_4__2side_4__3side_4         , ],
]

ch032_1side_4__2_3_side_all__4side_1_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.ch032_1side_4__2side_1__3side_1_4side_1 , L6_4side.ch032_1side_4__2side_2__3side_1_4side_1 , L6_4side.ch032_1side_4__2side_3__3side_1_4side_1 , L6_4side.ch032_1side_4__2side_4__3side_1_4side_1 , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_2__3side_2_4side_1 , L6_4side.ch032_1side_4__2side_3__3side_2_4side_1 , L6_4side.ch032_1side_4__2side_4__3side_2_4side_1 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_3__3side_3_4side_1 , L6_4side.ch032_1side_4__2side_4__3side_3_4side_1 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_4__3side_4_4side_1 , ],
]
ch032_1side_4__2_3_side_all__4side_2_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_2__3side_2_4side_2 , L6_4side.ch032_1side_4__2side_3__3side_2_4side_2 , L6_4side.ch032_1side_4__2side_4__3side_2_4side_2 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_3__3side_3_4side_2 , L6_4side.ch032_1side_4__2side_4__3side_3_4side_2 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_4__3side_4_4side_2 , ],
]
ch032_1side_4__2_3_side_all__4side_3_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_3__3side_3_4side_3 , L6_4side.ch032_1side_4__2side_4__3side_3_4side_3 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_4__3side_4_4side_3 , ],
]
ch032_1side_4__2_3_side_all__4side_4_5s0 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.empty                                   , L6_4side.ch032_1side_4__2side_4__3side_4_4side_4 , ],
]

##################################
### 5side1
##################################
# "1" (3 3side以前的結果也順便一起看吧)  6 10 15 21 28 36 45 55
# side1 OK 1
ch032_1side_1__2_3_side_all__4side_1_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_1 , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.ch032_1side_1__2side_1__3side_1_4side_1_5s1 , ],
]

# 1 "3" (6 3side以前的結果也順便一起看吧)  10 15 21 28 36 45 55
# side2 OK 4
ch032_1side_2__2_3_side_all__4side_1_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_2 , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.ch032_1side_2__2side_1__3side_1_4side_1_5s1 , L6_5side.ch032_1side_2__2side_2__3side_1_4side_1_5s1 , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_2__2side_2__3side_2_4side_1_5s1 , ],
]
ch032_1side_2__2_3_side_all__4side_2_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_2 , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_2__2side_2__3side_2_4side_2_5s1 , ],
]

# 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
# side3 OK 10
ch032_1side_3__2_3_side_all__4side_1_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.ch032_1side_3__2side_1__3side_1_4side_1_5s1 , L6_5side.ch032_1side_3__2side_2__3side_1_4side_1_5s1 , L6_5side.ch032_1side_3__2side_3__3side_1_4side_1_5s1 , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_2__3side_2_4side_1_5s1 , L6_5side.ch032_1side_3__2side_3__3side_2_4side_1_5s1 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_3__3side_3_4side_1_5s1 , ],
]
ch032_1side_3__2_3_side_all__4side_2_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_2__3side_2_4side_2_5s1 , L6_5side.ch032_1side_3__2side_3__3side_2_4side_2_5s1 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_3__3side_3_4side_2_5s1 , ],
]
ch032_1side_3__2_3_side_all__4side_3_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_3__3side_3_4side_3_5s1 , ],
]

# 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
# L6_4side OK 20
ch032_1side_4__2_3_side_all__4side_1_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.ch032_1side_4__2side_1__3side_1_4side_1_5s1 , L6_5side.ch032_1side_4__2side_2__3side_1_4side_1_5s1 , L6_5side.ch032_1side_4__2side_3__3side_1_4side_1_5s1 , L6_5side.ch032_1side_4__2side_4__3side_1_4side_1_5s1 , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_2__3side_2_4side_1_5s1 , L6_5side.ch032_1side_4__2side_3__3side_2_4side_1_5s1 , L6_5side.ch032_1side_4__2side_4__3side_2_4side_1_5s1 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_3__3side_3_4side_1_5s1 , L6_5side.ch032_1side_4__2side_4__3side_3_4side_1_5s1 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_1_5s1 , ],
]
ch032_1side_4__2_3_side_all__4side_2_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_2__3side_2_4side_2_5s1 , L6_5side.ch032_1side_4__2side_3__3side_2_4side_2_5s1 , L6_5side.ch032_1side_4__2side_4__3side_2_4side_2_5s1 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_3__3side_3_4side_2_5s1 , L6_5side.ch032_1side_4__2side_4__3side_3_4side_2_5s1 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_2_5s1 , ],
]
ch032_1side_4__2_3_side_all__4side_3_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_3__3side_3_4side_3_5s1 , L6_5side.ch032_1side_4__2side_4__3side_3_4side_3_5s1 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_3_5s1 , ],
]
ch032_1side_4__2_3_side_all__4side_4_5s1 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s1 , ],
]

##################################
### 5side2
##################################
# 1 "3" (6 3side以前的結果也順便一起看吧)  10 15 21 28 36 45 55
# side2 OK 4
ch032_1side_2__2_3_side_all__4side_2_5s2 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_2 , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_2__2side_2__3side_2_4side_2_5s2 , ],
]

# 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
# side3 OK 10
ch032_1side_3__2_3_side_all__4side_2_5s2 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_2__3side_2_4side_2_5s2 , L6_5side.ch032_1side_3__2side_3__3side_2_4side_2_5s2 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_3__3side_3_4side_2_5s2 , ],
]
ch032_1side_3__2_3_side_all__4side_3_5s2 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_3__3side_3_4side_3_5s2 , ],
]

# 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
# L6_4side OK 20
ch032_1side_4__2_3_side_all__4side_2_5s2 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_2__3side_2_4side_2_5s2 , L6_5side.ch032_1side_4__2side_3__3side_2_4side_2_5s2 , L6_5side.ch032_1side_4__2side_4__3side_2_4side_2_5s2 , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_3__3side_3_4side_2_5s2 , L6_5side.ch032_1side_4__2side_4__3side_3_4side_2_5s2 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_2_5s2 , ],
]
ch032_1side_4__2_3_side_all__4side_3_5s2 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_3__3side_3_4side_3_5s2 , L6_5side.ch032_1side_4__2side_4__3side_3_4side_3_5s2 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_3_5s2 , ],
]
ch032_1side_4__2_3_side_all__4side_4_5s2 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s2 , ],
]

##################################
### 5side3
##################################
# 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
# side3 OK 10
ch032_1side_3__2_3_side_all__4side_3_5s3 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_3 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_3__2side_3__3side_3_4side_3_5s3 , ],
]

# 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
# L6_4side OK 20
ch032_1side_4__2_3_side_all__4side_3_5s3 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_3__3side_3_4side_3_5s3 , L6_5side.ch032_1side_4__2side_4__3side_3_4side_3_5s3 , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_3_5s3 , ],
]
ch032_1side_4__2_3_side_all__4side_4_5s3 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s3 , ],
]

##################################
### 5side4
##################################
# 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
# L6_4side OK 20
ch032_1side_4__2_3_side_all__4side_4_5s4 = [
    [L6_0side.ch032_0side   , L6_1side.ch032_1side_4 , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_1 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_2 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_3 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , ],
    [L6_1side.ch032_1side_4 , L6_3side.empty         , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.empty                                       , L6_5side.ch032_1side_4__2side_4__3side_4_4side_4_5s4 , ],
]