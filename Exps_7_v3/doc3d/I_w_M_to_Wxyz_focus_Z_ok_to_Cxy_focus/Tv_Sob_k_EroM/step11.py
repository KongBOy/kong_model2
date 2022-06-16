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
#################################################################################################################################################################################################################################################################################################################################################################################################
#################################
##### 前fix 後change
### 分析 EroM
combine1_fix_I_w_M_to_W__change_W_w_M_t_C_EroM_analyze = [
    [p20_wiColorJ_L5_MaeSob_k03__MaeSob_k05        , p20_wiColorJ_L5_MaeSob_k05__MaeSob_k05        , p20_wiColorJ_L5_MaeSob_k09__MaeSob_k05        , p20_wiColorJ_L5_MaeSob_k11__MaeSob_k05        , ],
    [p20_wiColorJ_L5_MaeSob_k03_EroM__MaeSob_k05   , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05   , p20_wiColorJ_L5_MaeSob_k09_EroM__MaeSob_k05   , p20_wiColorJ_L5_MaeSob_k11_EroM__MaeSob_k05   , ],
    [p20_wiColorJ_L5_MaeSob_k03_EroMore__MaeSob_k05, p20_wiColorJ_L5_MaeSob_k05_EroMore__MaeSob_k05, p20_wiColorJ_L5_MaeSob_k09_EroMore__MaeSob_k05, p20_wiColorJ_L5_MaeSob_k11_EroMore__MaeSob_k05, ],
]


### 分析TV
combine2_fix_I_w_M_to_W__change_W_w_M_t_C_Tv_s001_analyze = [
    [W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k03         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k05         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k09         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k11         , ],
    [W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k03_Tv_s001 , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k05_Tv_s001 , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k09_Tv_s001 , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k11_Tv_s001 , ],
]
combine = [
    [W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k03_EroM         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k05_EroM         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k09_EroM         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k11_EroM         , ],
    [W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k03_Tv_s001_EroM , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k05_Tv_s001_EroM , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k09_Tv_s001_EroM , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k11_Tv_s001_EroM , ],
]
combine = [
    [W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k03_EroMore         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k05_EroMore         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k09_EroMore         , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k11_EroMore         , ],
    [W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k03_Tv_s001_EroMore , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k05_Tv_s001_EroMore , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k09_Tv_s001_EroMore , W_w_M_to_C_Tcrop255_p20_wiColorJ_2s_L5_MaeSob_k11_Tv_s001_EroMore , ],
]
