#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################

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

############################  have_bg  #################################
### 1a. ch
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch128_sig_L6_ep060.{run}"])  ### 127.37跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_L6_ep060.{run}"])  ### 127.37跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_sig_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch016_sig_L6_ep060.{run}"])  ### 127.35跑
''' 做完許多實驗以後覺得 ch太少沒辦法， 加入 tv, sobel 很容易壞掉 所以就不train了
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch008_sig_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch004_sig_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch002_sig_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch001_sig_L6_ep060.{run}"])  ### 127.35跑
'''

### 3. no-concat
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_2to2noC_sig_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_2to3noC_sig_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_2to4noC_sig_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_2to5noC_sig_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_2to6noC_sig_ep060.{run}"])  ### 127.35跑

### 4. skip use add
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_skipAdd_sig_ep060.{run}"])  ### 127.35跑

### 1b. ch and epoch
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch128_sig_L6_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_L6_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_sig_L6_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch016_sig_L6_ep200.{run}"])  ### 127.35跑
''' 做完許多實驗以後覺得 ch太少沒辦法， 加入 tv, sobel 很容易壞掉 所以就不train了
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch008_sig_L6_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch004_sig_L6_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch002_sig_L6_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch001_sig_L6_ep200.{run}"])  ### 127.35跑
'''
