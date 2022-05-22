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

###################
############# 1s1
######### 2s1
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_1__2side_1__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])

###################
############# 1s2
######### 2s1
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_1__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])

######### 2s1
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])

##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_2__2side_2__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])

###################
############# 1s3
######### 2s1
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_1__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
######### 2s2
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_2__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
######### 2s3
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
##### 3s3
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_3__2side_3__3side_3_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])

###################
############# 1s4
######### 2s1
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_1__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
######### 2s2
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_2__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
######### 2s3
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
##### 3s3
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_3__3side_3_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
######### 2s4
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
##### 3s3
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_3_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
##### 3s4
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
### 4s4
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s3_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s4_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s4_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s4_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_4__2side_4__3side_4_4side_4_5s4_6s4.{test % ('', 'knpy_save=False')}"])

###################
############# 1s5
######### 2s1
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_1__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
######### 2s2
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_2__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_2__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_2__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_2__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_2__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
######### 2s3
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
##### 3s3
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_3__3side_3_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
######### 2s4
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
##### 3s3
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_3_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
##### 3s4
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
### 4s4
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s3_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s4_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s4_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s4_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_4__3side_4_4side_4_5s4_6s4.{test % ('', 'knpy_save=False')}"])
######### 2s5
##### 3s1
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_1_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
##### 3s2
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_2_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_2_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_2_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_2_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
##### 3s3
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_3_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
##### 3s4
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
### 4s4
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s3_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s4_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s4_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s4_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_4_4side_4_5s4_6s4.{test % ('', 'knpy_save=False')}"])
##### 3s5
### 4s1
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_1_5s1_6s1.{test % ('', 'knpy_save=False')}"])
### 4s2
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_2_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_2_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_2_5s2_6s2.{test % ('', 'knpy_save=False')}"])
### 4s3
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_3_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_3_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_3_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_3_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_3_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_3_5s3_6s3.{test % ('', 'knpy_save=False')}"])
### 4s4
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s3_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s4_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s4_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s4_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_4_5s4_6s4.{test % ('', 'knpy_save=False')}"])
### 4s5
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s1_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s2_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s2_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s3_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s3_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s3_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s4_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s4_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s4_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s4_6s4.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s5_6s1.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s5_6s2.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s5_6s3.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s5_6s4.{test % ('', 'knpy_save=False')}"])
# sb.run(cmd_python_step10_a + [f"ch032_1side_5__2side_5__3side_5_4side_5_5s5_6s5.{test % ('', 'knpy_save=False')}"])