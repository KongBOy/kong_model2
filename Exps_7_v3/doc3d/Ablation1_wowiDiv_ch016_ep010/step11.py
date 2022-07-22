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
#################################
##### 前fix 後change
### 分析 Div_3UNet_analyze

# 從 step10a copy 過來的，總共就是要分析這些exps
# ##### 一起訓練
# ### 3UNet
# I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010
# ### wiDiv
# I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010
# ### woD_L
# I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010
# ### woD_M
# I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010
# I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010

comb_change_I_w_M_to_W__fix_W_w_M_t_C__3UNet_analyze = [
    [Google_down, ],
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, ] ,
]
comb_change_I_w_M_to_W__fix_W_w_M_t_C__wiDiv_analyze = [
    [Google_down],
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, ] ,
]
comb_change_I_w_M_to_W__fix_W_w_M_t_C__woD_L_analyze = [
    [Google_down],
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, ] ,
]
comb_change_I_w_M_to_W__fix_W_w_M_t_C__woD_M_analyze = [
    [Google_down],
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, ] ,
]

comb_all__change_I_w_M_to_W_then_changeW_w_M_t_C_analyze = [
    [Google_down],
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_3UNet_Sob_k09_Mae_ep001__ep010, ] ,
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_wiDiv_Sob_k09_Mae_ep001__ep010, ] ,
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_woD_L_Sob_k09_Mae_ep001__ep010, ] ,
    [I_w_M_to_W__3UNet_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__wiDiv_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_L_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, I_w_M_to_W__woD_M_Sob_k09_Mae_ep001__W_w_M_to_C_woD_M_Sob_k09_Mae_ep001__ep010, ] ,
]
