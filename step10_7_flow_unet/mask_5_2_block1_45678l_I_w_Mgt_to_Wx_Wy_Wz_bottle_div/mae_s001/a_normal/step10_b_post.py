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
### 所有 指令 統一寫這邊
from step10_c_exp_command import *
######################################################################################################################
import subprocess as sb


#### 3l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L2_ch066.{run}"])
# sb.run(cmd_python_step10_a + [f"L2_ch033.{run}"])
# sb.run(cmd_python_step10_a + [f"L2_ch018.{run}"])
# sb.run(cmd_python_step10_a + [f"L2_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L2_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L2_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L2_ch129.{run}"])
#### 3l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L3_ch066.{run}"])
# sb.run(cmd_python_step10_a + [f"L3_ch033.{run}"])
# sb.run(cmd_python_step10_a + [f"L3_ch018.{run}"])
# sb.run(cmd_python_step10_a + [f"L3_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L3_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L3_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L3_ch129.{run}"])
#### 4l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L4_ch066.{run}"])
# sb.run(cmd_python_step10_a + [f"L4_ch033.{run}"])
# sb.run(cmd_python_step10_a + [f"L4_ch018.{run}"])
# sb.run(cmd_python_step10_a + [f"L4_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L4_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L4_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L4_ch129.{run}"])
#### 5l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L5_ch066.{run}"])
# sb.run(cmd_python_step10_a + [f"L5_ch033.{run}"])
# sb.run(cmd_python_step10_a + [f"L5_ch018.{run}"])
# sb.run(cmd_python_step10_a + [f"L5_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L5_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L5_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L5_ch129.{run}"])
#### 6l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L6_ch033.{run}"])
# sb.run(cmd_python_step10_a + [f"L6_ch018.{run}"])
# sb.run(cmd_python_step10_a + [f"L6_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L6_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L6_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L6_ch066.{run}"])
#### 7l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L7_ch018.{run}"])
# sb.run(cmd_python_step10_a + [f"L7_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L7_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L7_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L7_ch033.{run}"])
#### 8l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L8_ch009.{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch006.{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch003.{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch018.{run}"])
