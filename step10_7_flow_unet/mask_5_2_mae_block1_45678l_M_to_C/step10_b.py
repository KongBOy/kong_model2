### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                       ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")                ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])  ### 定位出 kong_model2 的 dir
import sys                                                       ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print("step10b")
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
######################################################################################################################
### 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
    os.chdir(code_exe_dir)
# print("current_path:", os.getcwd())
######################################################################################################################
### 所有 指令 統一寫這邊
from step10_c_exp_command import *
######################################################################################################################
import subprocess as sb


# sb.run(cmd_python_step10_a + [f"L7_ch016_mae_s001_copy.{compress_and_bm_rec_all}"])
# exit()
#### l2 ############################################################################################
# sb.run(cmd_python_step10_a + [f"L2_ch128_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch064_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch032_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L2_ch001_mae_s001.{compress_and_bm_rec_all}"])
#### 3l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L3_ch128_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch064_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch032_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L3_ch001_mae_s001.{compress_and_bm_rec_all}"])
#### 4l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L4_ch064_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch032_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L4_ch001_mae_s001.{compress_and_bm_rec_all}"])
#### 5l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L5_ch032_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L5_ch001_mae_s001.{compress_and_bm_rec_all}"])
#### 6l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L6_ch128_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch128_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch064_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch064_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch032_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch032_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L6_ch001_mae_s001.{compress_and_bm_rec_all}"])
#### 7l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L7_ch128_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch128_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch064_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch064_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch032_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch032_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch016_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch016_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L7_ch001_mae_s001.{compress_and_bm_rec_all}"])
#### 8l ############################################################################################
# sb.run(cmd_python_step10_a + [f"L8_ch128_mae_s001      .{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch128_mae_s001_limit.{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch064_mae_s001      .{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch064_mae_s001_limit.{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch032_mae_s001      .{run}"])
# sb.run(cmd_python_step10_a + [f"L8_ch032_mae_s001_limit.{run}"])
sb.run(cmd_python_step10_a + [f"L8_ch016_mae_s001      .{compress_and_bm_rec_all}"])
sb.run(cmd_python_step10_a + [f"L8_ch016_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L8_ch008_mae_s001      .{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L8_ch008_mae_s001_limit.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L8_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L8_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"L8_ch001_mae_s001.{compress_and_bm_rec_all}"])
