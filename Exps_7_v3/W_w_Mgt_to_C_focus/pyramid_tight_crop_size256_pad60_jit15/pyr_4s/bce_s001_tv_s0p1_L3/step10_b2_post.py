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

# "1" 3 6 10 15 21 28 36 45 55
# side1 OK 1
# sb.run(cmd_python_step10_a + [f"ch032_1side_1__2side_1__3side_1_4side_1.{compress_and_bm_rec_all}"])

# 1 "3" 6 10 15 21 28 36 45 55
# side2 OK 4
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_1__3side_1_4side_1.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_1_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_2_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_2_4side_2.{compress_and_bm_rec_all}"])

# 1 3 "6" 10 15 21 28 36 45 55
# side3 OK 10
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_1__3side_1_4side_1.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_1_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_2_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_2_4side_2.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_1_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_2_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_2_4side_2.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_2.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3.{compress_and_bm_rec_all}"])

# 1 3 6 "10" 15 21 28 36 45 55
# side4 OK 20
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_1__3side_1_4side_1.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_1_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_2_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_2_4side_2.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_1_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_2_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_2_4side_2.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_2.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3.{compress_and_bm_rec_all}"])

# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_1_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_2_4side_1.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_2_4side_2.{compress_and_bm_rec_all}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_1.{compress_and_bm_rec_all_from_begin}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_2.{compress_and_bm_rec_all_from_begin}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3.{compress_and_bm_rec_all_from_begin}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_1.{compress_and_bm_rec_all_from_begin}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_2.{compress_and_bm_rec_all_from_begin}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3.{compress_and_bm_rec_all_from_begin}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4.{compress_and_bm_rec_all_from_begin}"])
