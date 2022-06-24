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
####### 目前先用 127.29 跑
##### 前fix 後change
### 分析 EroM
comb1b_fix_I_w_M_to_W__change_W_w_M_t_C__EroM_type_analyze = [
    [Model_run                                          , Google_down                                        , empty                                              , empty                                              , ],
    [p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03        , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05        , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09        , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11        , ],
    [p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_EroM   , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroM   , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_EroM   , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_EroM   , ],
    [p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_EroMore, p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroMore, p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_EroMore, p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_EroMore, ],
]

### 分析TV
comb2a_fix_I_w_M_to_W__change_W_w_M_t_C__wo_wiTv_s001_woEroM = [
    [Model_run  , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11         , ],
    [Google_down, p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_Tv_s001 , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_Tv_s001 , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_Tv_s001 , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_Tv_s001 , ],
]
comb2b_fix_I_w_M_to_W__change_W_w_M_t_C__wo_wiTv_s001_wiEroM = [
    [Model_run  , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_EroM         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_EroM         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_EroM         , ],
    [Google_down, p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_Tv_s001_EroM , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_Tv_s001_EroM , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_Tv_s001_EroM , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_Tv_s001_EroM , ],
]
comb2c_fix_I_w_M_to_W__change_W_w_M_t_C__wo_wiTv_s001_wiEroMore = [
    [Model_run  , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_EroMore         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroMore         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_EroMore         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_EroMore         , ],
    [Google_down, p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_Tv_s001_EroMore , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_Tv_s001_EroMore , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_Tv_s001_EroMore , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_Tv_s001_EroMore , ],
]


comb2d_fix_I_w_M_to_W__change_W_w_M_t_C_tboard_all = [[
    Model_run,
    Google_down,

    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03                 ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_EroM            ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_EroMore         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_Tv_s001         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_Tv_s001_EroM    ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k03_Tv_s001_EroMore ,

    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05                 ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroM            ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroMore         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_Tv_s001         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_Tv_s001_EroM    ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_Tv_s001_EroMore ,

    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09                 ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_EroM            ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_EroMore         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_Tv_s001         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_Tv_s001_EroM    ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k09_Tv_s001_EroMore ,

    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11                 ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_EroM            ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_EroMore         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_Tv_s001         ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_Tv_s001_EroM    ,
    p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k11_Tv_s001_EroMore ,
]]
#################################
##### 前change 後fix
### 分析 EroM
combine3b_fix_I_w_M_to_W__change_W_w_M_t_C__EroM_type_analyze = [
    [Model_run                                          , Google_down                                        , empty                                              , empty                                              , ],
    [p20_wiColorJ_L5_MaeSob_k03__MaeSob_k05_EroM        , p20_wiColorJ_L5_MaeSob_k05__MaeSob_k05_EroM        , p20_wiColorJ_L5_MaeSob_k09__MaeSob_k05_EroM        , p20_wiColorJ_L5_MaeSob_k11__MaeSob_k05_EroM        , ],
    [p20_wiColorJ_L5_MaeSob_k03_EroM__MaeSob_k05_EroM   , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroM   , p20_wiColorJ_L5_MaeSob_k09_EroM__MaeSob_k05_EroM   , p20_wiColorJ_L5_MaeSob_k11_EroM__MaeSob_k05_EroM   , ],
    [p20_wiColorJ_L5_MaeSob_k03_EroMore__MaeSob_k05_EroM, p20_wiColorJ_L5_MaeSob_k05_EroMore__MaeSob_k05_EroM, p20_wiColorJ_L5_MaeSob_k09_EroMore__MaeSob_k05_EroM, p20_wiColorJ_L5_MaeSob_k11_EroMore__MaeSob_k05_EroM, ],
]

### 分析TV
combine4a_fix_I_w_M_to_W__change_W_w_M_t_C__wo_wiTv_s001_woEroM = [
    [Model_run  , p20_wiColorJ_L5_MaeSob_k03__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k05__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k09__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k11__MaeSob_k05_EroM         , ],
    [Google_down, p20_wiColorJ_L5_MaeSob_k03_Tv_s001__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k05_Tv_s001__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k09_Tv_s001__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k11_Tv_s001__MaeSob_k05_EroM , ],
]
combine4b_fix_I_w_M_to_W__change_W_w_M_t_C__wo_wiTv_s001_wiEroM = [
    [Model_run  , p20_wiColorJ_L5_MaeSob_k03_EroM__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k05_EroM__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k09_EroM__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k11_EroM__MaeSob_k05_EroM         , ],
    [Google_down, p20_wiColorJ_L5_MaeSob_k03_Tv_s001_EroM__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k05_Tv_s001_EroM__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k09_Tv_s001_EroM__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k11_Tv_s001_EroM__MaeSob_k05_EroM , ],
]
combine4c_fix_I_w_M_to_W__change_W_w_M_t_C__wo_wiTv_s001_wiEroMore = [
    [Model_run  , p20_wiColorJ_L5_MaeSob_k03_EroMore__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k05_EroMore__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k09_EroMore__MaeSob_k05_EroM         , p20_wiColorJ_L5_MaeSob_k11_EroMore__MaeSob_k05_EroM         , ],
    [Google_down, p20_wiColorJ_L5_MaeSob_k03_Tv_s001_EroMore__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k05_Tv_s001_EroMore__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k09_Tv_s001_EroMore__MaeSob_k05_EroM , p20_wiColorJ_L5_MaeSob_k11_Tv_s001_EroMore__MaeSob_k05_EroM , ],
]
