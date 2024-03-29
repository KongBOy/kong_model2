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

### 先看極端的 沒eroM 和 eroMore
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k03_s001_Tv_s001                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k05_s001_Tv_s001                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001_Tv_s001                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k11_s001_Tv_s001                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k03_s001_eroMore_Tv_s001_ero_More.{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k05_s001_eroMore_Tv_s001_ero_More.{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001_eroMore_Tv_s001_ero_More.{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k11_s001_eroMore_Tv_s001_ero_More.{train}"])

# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k03_s001                         .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k05_s001                         .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001                         .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k11_s001                         .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k03_s001_eroMore                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k05_s001_eroMore                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001_eroMore                 .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k11_s001_eroMore                 .{train}"])

### 在看 介於中間的 eroM
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k03_s001_eroM_Tv_s001_ero_M      .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k05_s001_eroM_Tv_s001_ero_M      .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001_eroM_Tv_s001_ero_M      .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k11_s001_eroM_Tv_s001_ero_M      .{train}"])

# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k03_s001_eroM                    .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k05_s001_eroM                    .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001_eroM                    .{train}"])
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k11_s001_eroM                    .{train}"])

### good ep010
# sb.run(cmd_python_step10_a + [f"L5_ch032_2blk__Mae_s001_Sob_k09_s001_ep010                   .{train}"])