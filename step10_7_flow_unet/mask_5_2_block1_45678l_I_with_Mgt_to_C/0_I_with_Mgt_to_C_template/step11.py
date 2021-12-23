#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])    ### 定位出 kong_model2 的 dir
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

import mae_s001.a_normal.step10_a as mae_block1
import mae_s001.b_limit .step10_a as mae_block1_limit
#################################################################################################################################################################################################################################################################################################################################################################################################
### L2345678_flow
mae_block1_flow_s001_L345678 = [
### L3 之前直接跳過 因為發現幾乎無法 rec #######################################################################
    [mae_block1.L3_ch128.build().result_obj,
     mae_block1.L3_ch064.build().result_obj,
     mae_block1.L3_ch032.build().result_obj,
     mae_block1.L3_ch016.build().result_obj,
     mae_block1.L3_ch008.build().result_obj,
     mae_block1.L3_ch004.build().result_obj,
     mae_block1.L3_ch002.build().result_obj,
     mae_block1.L3_ch001.build().result_obj],

    [mae_block1.L4_ch128.build().result_obj,
     mae_block1.L4_ch064.build().result_obj,
     mae_block1.L4_ch032.build().result_obj,
     mae_block1.L4_ch016.build().result_obj,
     mae_block1.L4_ch008.build().result_obj,
     mae_block1.L4_ch004.build().result_obj,
     mae_block1.L4_ch002.build().result_obj,
     mae_block1.L4_ch001.build().result_obj],

    [mae_block1.L5_ch128.build().result_obj,
     mae_block1.L5_ch064.build().result_obj,
     mae_block1.L5_ch032.build().result_obj,
     mae_block1.L5_ch016.build().result_obj,
     mae_block1.L5_ch008.build().result_obj,
     mae_block1.L5_ch004.build().result_obj,
     mae_block1.L5_ch002.build().result_obj,
     mae_block1.L5_ch001.build().result_obj],

    [mae_block1_limit.L6_ch128_limit.build().result_obj,
     mae_block1.L6_ch064.build().result_obj,
     mae_block1.L6_ch032.build().result_obj,
     mae_block1.L6_ch016.build().result_obj,
     mae_block1.L6_ch008.build().result_obj,
     mae_block1.L6_ch004.build().result_obj,
     mae_block1.L6_ch002.build().result_obj,
     mae_block1.L6_ch001.build().result_obj],

    [mae_block1_limit.L7_ch128_limit.build().result_obj,
     mae_block1_limit.L7_ch064_limit.build().result_obj,
     mae_block1.L7_ch032.build().result_obj,
     mae_block1.L7_ch016.build().result_obj,
     mae_block1.L7_ch008.build().result_obj,
     mae_block1.L7_ch004.build().result_obj,
     mae_block1.L7_ch002.build().result_obj,
     mae_block1.L7_ch001.build().result_obj],

    [mae_block1_limit.L8_ch128_limit.build().result_obj,
     mae_block1_limit.L8_ch064_limit.build().result_obj,
     mae_block1_limit.L8_ch032_limit.build().result_obj,
     mae_block1.L8_ch016.build().result_obj,
     mae_block1.L8_ch008.build().result_obj,
     mae_block1.L8_ch004.build().result_obj,
     mae_block1.L8_ch002.build().result_obj,
     mae_block1.L8_ch001.build().result_obj],
]


#################################################################################################################################################################################################################################################################################################################################################################################################
mae_block1_rec_s001_L45678 = [
    ### L3 之前直接跳過 因為發現幾乎無法 rec #######################################################################
    [
        mae_block1.L3_ch128.build().result_obj,
        mae_block1.L3_ch064.build().result_obj,
        mae_block1.L3_ch032.build().result_obj,
        mae_block1.L3_ch016.build().result_obj,
        mae_block1.L3_ch008.build().result_obj,
        mae_block1.L3_ch004.build().result_obj,
        mae_block1.L3_ch002.build().result_obj,
        # mae_block1.L3_ch001.build().result_obj, ### 因為 L8 只有7個， 所以這個也註解掉要不然跑步起來， 除非有空把這個bug修掉
    ],

    [
        mae_block1.L4_ch128.build().result_obj,
        mae_block1.L4_ch064.build().result_obj,
        mae_block1.L4_ch032.build().result_obj,
        mae_block1.L4_ch016.build().result_obj,
        mae_block1.L4_ch008.build().result_obj,
        mae_block1.L4_ch004.build().result_obj,
        mae_block1.L4_ch002.build().result_obj,
        mae_block1.L4_ch001.build().result_obj,
    ],

    [
        mae_block1.L5_ch128.build().result_obj,
        mae_block1.L5_ch064.build().result_obj,
        mae_block1.L5_ch032.build().result_obj,
        mae_block1.L5_ch016.build().result_obj,
        mae_block1.L5_ch008.build().result_obj,
        mae_block1.L5_ch004.build().result_obj,
        mae_block1.L5_ch002.build().result_obj,
        # mae_block1.L5_ch001.build().result_obj,  ###  做不起來
    ],

    [
        mae_block1_limit.L6_ch128_limit.build().result_obj,
        mae_block1.L6_ch064.build().result_obj,
        mae_block1.L6_ch032.build().result_obj,
        mae_block1.L6_ch016.build().result_obj,
        mae_block1.L6_ch008.build().result_obj,
        mae_block1.L6_ch004.build().result_obj,
        mae_block1.L6_ch002.build().result_obj,
        # mae_block1.L6_ch001.build().result_obj,  ### 做不起來
    ],

    [
        mae_block1_limit.L7_ch128_limit.build().result_obj,
        mae_block1_limit.L7_ch064_limit.build().result_obj,
        mae_block1.L7_ch032.build().result_obj,
        mae_block1.L7_ch016.build().result_obj,
        mae_block1.L7_ch008.build().result_obj,
        mae_block1.L7_ch004.build().result_obj,
        mae_block1.L7_ch002.build().result_obj,
        # mae_block1.L7_ch001.build().result_obj ### 因為 L8 只有7個， 所以這個也註解掉要不然跑步起來， 除非有空把這個bug修掉
    ],

    [
        mae_block1_limit.L8_ch128_limit.build().result_obj,
        mae_block1_limit.L8_ch064_limit.build().result_obj,
        mae_block1_limit.L8_ch032_limit.build().result_obj,
        mae_block1.L8_ch016.build().result_obj,
        mae_block1.L8_ch008.build().result_obj,
        mae_block1.L8_ch004.build().result_obj,
        mae_block1.L8_ch002.build().result_obj,
        # mae_block1.L8_ch001.build().result_obj,  ### 做不起來
    ],
]
# ### L2345_E_relu
# mae_block1_L2345_E_relu = [
#     [mae_block1.L2_ch128_E_relu.build().result_obj,
#      mae_block1.L2_ch064_E_relu.build().result_obj,
#      mae_block1.L2_ch032_E_relu.build().result_obj,
#      mae_block1.L2_ch016_E_relu.build().result_obj,
#      mae_block1.L2_ch008_E_relu.build().result_obj],
#     [mae_block1.L3_ch128_E_relu.build().result_obj,
#      mae_block1.L3_ch064_E_relu.build().result_obj,
#      mae_block1.L3_ch032_E_relu.build().result_obj,
#      mae_block1.L3_ch016_E_relu.build().result_obj,
#      mae_block1.L3_ch008_E_relu.build().result_obj],
#     [mae_block1.L4_ch064_E_relu.build().result_obj,
#      mae_block1.L4_ch032_E_relu.build().result_obj,
#      mae_block1.L4_ch016_E_relu.build().result_obj,
#      mae_block1.L4_ch008_E_relu.build().result_obj,
#      mae_block1.L4_ch004_E_relu.build().result_obj],
#     [mae_block1.L5_ch032_E_relu.build().result_obj,
#      mae_block1.L5_ch016_E_relu.build().result_obj,
#      mae_block1.L5_ch008_E_relu.build().result_obj,
#      mae_block1.L5_ch004_E_relu.build().result_obj,
#      mae_block1.L5_ch002_E_relu.build().result_obj],
# ]
# ### L2345_E_lrelu vs E_relu
# mae_block1_L2345__E_lrelu_vs_E_relu = [
#     [mae_block1.L2_ch128.build().result_obj,
#      mae_block1.L2_ch064.build().result_obj,
#      mae_block1.L2_ch032.build().result_obj,
#      mae_block1.L2_ch016.build().result_obj,
#      mae_block1.L2_ch008.build().result_obj],
#     [mae_block1.L2_ch128_E_relu.build().result_obj,
#      mae_block1.L2_ch064_E_relu.build().result_obj,
#      mae_block1.L2_ch032_E_relu.build().result_obj,
#      mae_block1.L2_ch016_E_relu.build().result_obj,
#      mae_block1.L2_ch008_E_relu.build().result_obj],

#     [mae_block1.L3_ch128.build().result_obj,
#      mae_block1.L3_ch064.build().result_obj,
#      mae_block1.L3_ch032.build().result_obj,
#      mae_block1.L3_ch016.build().result_obj,
#      mae_block1.L3_ch008.build().result_obj],
#     [mae_block1.L3_ch128_E_relu.build().result_obj,
#      mae_block1.L3_ch064_E_relu.build().result_obj,
#      mae_block1.L3_ch032_E_relu.build().result_obj,
#      mae_block1.L3_ch016_E_relu.build().result_obj,
#      mae_block1.L3_ch008_E_relu.build().result_obj],

#     [mae_block1.L4_ch064.build().result_obj,
#      mae_block1.L4_ch032.build().result_obj,
#      mae_block1.L4_ch016.build().result_obj,
#      mae_block1.L4_ch008.build().result_obj,
#      mae_block1.L4_ch004.build().result_obj],
#     [mae_block1.L4_ch064_E_relu.build().result_obj,
#      mae_block1.L4_ch032_E_relu.build().result_obj,
#      mae_block1.L4_ch016_E_relu.build().result_obj,
#      mae_block1.L4_ch008_E_relu.build().result_obj,
#      mae_block1.L4_ch004_E_relu.build().result_obj],

#     [mae_block1.L5_ch032.build().result_obj,
#      mae_block1.L5_ch016.build().result_obj,
#      mae_block1.L5_ch008.build().result_obj,
#      mae_block1.L5_ch004.build().result_obj,
#      mae_block1.L5_ch002.build().result_obj],
#     [mae_block1.L5_ch032_E_relu.build().result_obj,
#      mae_block1.L5_ch016_E_relu.build().result_obj,
#      mae_block1.L5_ch008_E_relu.build().result_obj,
#      mae_block1.L5_ch004_E_relu.build().result_obj,
#      mae_block1.L5_ch002_E_relu.build().result_obj],
# ]
