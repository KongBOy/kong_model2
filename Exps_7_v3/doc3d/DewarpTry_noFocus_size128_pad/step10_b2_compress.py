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


### I_w_M_to_C 3UNet/wiDiv/woDiv %& FL/FM/NL/NM
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_0s_L5__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_0s_L6__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_0s_L7__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_1s_L5__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_1s_L6__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_1s_L7__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_2s_L5__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_2s_L6__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_I_w_M_to_W__ch016_blk_2s_L7__woD_L_in_have_bg .{compress_all}"])

### W_w_M_to_C woDiv
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_0s_L5__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_0s_L6__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_0s_L7__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_1s_L5__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_1s_L6__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_1s_L7__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_2s_L5__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_2s_L6__woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_W_w_M_to_C__ch016_blk_2s_L7__woD_L_in_have_bg .{compress_all}"])

##### 一起訓練
### 4. woD_L woD_L(記得 woD_L 的 seperate 要設 False)，第二個測這個
# 這個是我意想不到竟然做得更好的結果， 我想看看他可以做得多好
# sb.run(cmd_python_step10_a + [f"exp_blk_0s__L5_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_0s__L6_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_0s__L7_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_1s__L5_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_1s__L6_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_1s__L7_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_2s__L5_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_2s__L6_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_blk_2s__L7_I_w_M_to_W_ch016_woD_L__W_w_M_to_C_ch016_woD_L_in_have_bg .{compress_all}"])

### 目前最接近 DewarpNet的架構
# sb.run(cmd_python_step10_a + [f"exp_DewarpUNet_I_w_M_to_W_IN__in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_DewarpUNet_W_w_M_to_C_IN__in_have_bg .{compress_all}"])
# sb.run(cmd_python_step10_a + [f"exp_DewarpUNet_Gather_IN__in_have_bg     .{compress_all}"])
