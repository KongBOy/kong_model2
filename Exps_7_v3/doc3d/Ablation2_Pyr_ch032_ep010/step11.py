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
from step10_a import *
from Exps_7_v3.doc3d.DewarpNet_result.step10_a import Model_run, Google_down, Recti_run
#################################################################################################################################################################################################################################################################################################################################################################################################
##### 前change(參no init) 後fix
I_w_M_to_W_change_have_no_init__W_w_M_t_C_fix_Full_Less_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, ] ,
]
I_w_M_to_W_change_have_no_init__W_w_M_t_C_fix_Full_More_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, ] ,
]
I_w_M_to_W_change_have_no_init__W_w_M_t_C_fix_NoFu_Less_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, ] ,
]
I_w_M_to_W_change_have_no_init__W_w_M_t_C_fix_NoFu_More_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]

I_w_M_to_W_change_have_no_init__W_w_M_t_C_fix_all_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, ] ,
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, ] ,
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, ] ,
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]

##### 前change 後fix
I_w_M_to_W_change__W_w_M_t_C_fix_Full_Less_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, ] ,
]
I_w_M_to_W_change__W_w_M_t_C_fix_Full_More_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, ] ,
]
I_w_M_to_W_change__W_w_M_t_C_fix_NoFu_Less_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, ] ,
]
I_w_M_to_W_change__W_w_M_t_C_fix_NoFu_More_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]

I_w_M_to_W_change__W_w_M_t_C_fix_all_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]
##################################################################################################################################
##### 前fix 後change
I_w_M_to_W_fix_Full_Less_No_init__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]

I_w_M_to_W_fix_Full_Less__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]
I_w_M_to_W_fix_Full_More__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]
I_w_M_to_W_fix_NoFu_Less__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]
I_w_M_to_W_fix_NoFu_More__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]

I_w_M_to_W_fix_all__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]

I_w_M_to_W_fix_all_No_init__W_w_M_t_C_change_analyze = [
    [Google_down],
    [exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_No_inital_train_I_w_M_to_W_ch032_wiDiv_Full_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_Full_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_Less__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
    [exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_Full_More, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_Less, exp_I_w_M_to_W_ch032_wiDiv_NoFu_More__W_w_M_to_C_ch032_wiDiv_NoFu_More, ] ,
]
