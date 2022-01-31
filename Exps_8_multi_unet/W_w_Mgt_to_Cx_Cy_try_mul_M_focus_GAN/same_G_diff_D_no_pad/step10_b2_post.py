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
### 所有 指令 統一寫這邊
from step10_c_exp_command import *
######################################################################################################################
import subprocess as sb

# sb.run(cmd_python_step10_a + [f"L1_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch004.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch008.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch032.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch064.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L1_ch128.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"L2_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch004.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch008.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch032.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch064.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch128.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"L3_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch004.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch008.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch032.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch064.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch128.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"L4_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch004.{compress_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch008.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch032.{compress_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch064.{compress_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch128.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"L5_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch004.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch008.{compress_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch064.{compress_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch128.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"L6_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch004.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch008.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch032.{compress_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch064.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch001.{run}"])
# sb.run(cmd_python_step10_a + [f"L6_ch128.{compress_all}"])

# sb.run(cmd_python_step10_a + [f"L7_ch002.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch004.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch008.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch016.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch032.{run}"])
# sb.run(cmd_python_step10_a + [f"L7_ch064.{run}"])
# sb.run(cmd_python_step10_a + [f"L7_ch001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch128.{run}"])
