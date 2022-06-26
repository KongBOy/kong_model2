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
from Exps_7_v3.doc3d.DewarpNet_result.step10_a import Model_run, Google_down
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################
##### 前fix 後change
### 分析 Div_3UNet_analyze

comb_change_I_w_M_to_W__fix_W_w_M_t_C__Div_3UNet_analyze = [
    ### woDIv
    [p20_L5_woDiv_Sob_k05_Mae__Sob_k05_Mae, p20_L5_woDiv_Sob_k15_Mae__Sob_k05_Mae, p20_L5_woDiv_Sob_k25_Mae__Sob_k05_Mae, p20_L5_woDiv_Sob_k35_Mae__Sob_k05_Mae, ] ,
    ### wiDIv
    [p20_L5_wiDiv_Sob_k05_Mae__Sob_k05_Mae, p20_L5_wiDiv_Sob_k15_Mae__Sob_k05_Mae, p20_L5_wiDiv_Sob_k25_Mae__Sob_k05_Mae, p20_L5_wiDiv_Sob_k35_Mae__Sob_k05_Mae, ] ,
    ### 3UNet
    [p20_L5_3UNet_Sob_k05_Mae__Sob_k05_Mae, p20_L5_3UNet_Sob_k15_Mae__Sob_k05_Mae, p20_L5_3UNet_Sob_k25_Mae__Sob_k05_Mae, p20_L5_3UNet_Sob_k35_Mae__Sob_k05_Mae, ] ,
]

comb_change_I_w_M_to_W__fix_W_w_M_t_C__Div_3UNet_analyze2 = [
    ### woDIv
    [p20_L5_woDiv_Sob_k03_Mae__Sob_k05_Mae, p20_L5_woDiv_Sob_k05_Mae__Sob_k05_Mae, p20_L5_woDiv_Sob_k09_Mae__Sob_k05_Mae, p20_L5_woDiv_Sob_k11_Mae__Sob_k05_Mae, ] ,
    ### wiDIv
    [p20_L5_wiDiv_Sob_k03_Mae__Sob_k05_Mae, p20_L5_wiDiv_Sob_k05_Mae__Sob_k05_Mae, p20_L5_wiDiv_Sob_k09_Mae__Sob_k05_Mae, p20_L5_wiDiv_Sob_k11_Mae__Sob_k05_Mae, ] ,
    ### 3UNet
    [p20_L5_3UNet_Sob_k03_Mae__Sob_k05_Mae, p20_L5_3UNet_Sob_k05_Mae__Sob_k05_Mae, p20_L5_3UNet_Sob_k09_Mae__Sob_k05_Mae, p20_L5_3UNet_Sob_k11_Mae__Sob_k05_Mae, ] ,
]
