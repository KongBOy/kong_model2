### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print("step10b")
# print("code_exe_path:", code_exe_path)
# print("code_exe_path_element:", code_exe_path_element)
# print("kong_layer:", kong_layer)
# print("kong_model2_dir:", kong_model2_dir)

### 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
    os.chdir(code_exe_dir)
######################################################################################################################
### 所有 指令 統一寫這邊
from step10_c_exp_command import *
######################################################################################################################

import subprocess as sb


#### l2 ############################################################################################
# sb.run(same_command + [f"L2_ch128_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch128_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch128_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch128_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch128_mae_s001_coord_conv.{run}"])
###################
# sb.run(same_command + [f"L2_ch064_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch064_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch064_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch064_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch064_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch032_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch032_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch032_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch032_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch032_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch016_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch016_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch016_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch008_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch008_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch008_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch004_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s100.{run}"])

# sb.run(same_command + [f"L2_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L2_ch002_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s100.{run}"])

# sb.run(same_command + [f"L2_ch001_mae_s001.{compress_and_bm_rec_all}"])  ### 跑不動
# sb.run(same_command + [f"L2_ch001_mae_s001_copy.{compress_and_bm_rec_4}"])
# sb.run(same_command + [f"L2_ch001_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s100.{run}"])
#### 3l ############################################################################################
# sb.run(same_command + [f"L3_ch128_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch128_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch128_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch128_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch128_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch064_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch064_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch064_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch064_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch064_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L3_ch032_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch032_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch032_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch032_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch032_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L3_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch016_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch016_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch016_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch016_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L3_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch008_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch008_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch008_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch004_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch004_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s100.{run}"])

# sb.run(same_command + [f"L3_ch002_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L3_ch002_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s100.{run}"])

# sb.run(same_command + [f"L3_ch001_mae_s001.{compress_and_bm_rec_all}"])  ### 跑失敗
# sb.run(same_command + [f"L3_ch001_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s100.{run}"])
#### 4l ############################################################################################
sb.run(same_command + [f"L4_ch064_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L4_ch064_mae_s020.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s040.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s060.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s080.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L4_ch064_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L4_ch064_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L4_ch064_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L4_ch032_mae_s001.{compress_and_bm_rec_all}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s100.{run}"])  ### 127.27
### E_relu
# sb.run(same_command + [f"L4_ch032_mae_s001_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s020_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s040_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s060_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s080_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s100_E_relu .{run}"])  ### 127.27
### no_Bias
# sb.run(same_command + [f"L4_ch032_mae_s001_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s020_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s040_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s060_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s080_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s100_no_Bias.{run}"])  ### 127.27
### coord_conv
# sb.run(same_command + [f"L4_ch032_mae_s001_coord_conv.{run}"])  ### 127.27
####################
sb.run(same_command + [f"L4_ch016_mae_s001.{compress_and_bm_rec_all}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s020.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s040.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s060.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s080.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s100.{run}"])  ### 127.37
### E_relu
# sb.run(same_command + [f"L4_ch016_mae_s001_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s020_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s040_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s060_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s080_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s100_E_relu .{run}"])  ### 127.27
### no_Bias
# sb.run(same_command + [f"L4_ch016_mae_s001_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s020_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s040_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s060_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s080_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s100_no_Bias.{run}"])  ### 127.27
### coord_conv
# sb.run(same_command + [f"L4_ch016_mae_s001_coord_conv.{run}"])  ### 127.27
####################
sb.run(same_command + [f"L4_ch008_mae_s001.{compress_and_bm_rec_all}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s100.{run}"])  ### 127.49
### E_relu
# sb.run(same_command + [f"L4_ch008_mae_s001_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s020_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s040_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s060_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s080_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s100_E_relu .{run}"])  ### 127.27
### no_Bias
# sb.run(same_command + [f"L4_ch008_mae_s001_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s020_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s040_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s060_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s080_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s100_no_Bias.{run}"])  ### 127.27
### coord_conv
# sb.run(same_command + [f"L4_ch008_mae_s001_coord_conv.{run}"])  ### 127.27
####################
sb.run(same_command + [f"L4_ch004_mae_s001.{compress_and_bm_rec_all}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s100.{run}"])  ### 127.49
### E_relu
# sb.run(same_command + [f"L4_ch004_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L4_ch004_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L4_ch004_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L4_ch002_mae_s001.{compress_and_bm_rec_all}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s100.{run}"])  ### 127.49
####################
sb.run(same_command + [f"L4_ch001_mae_s001.{compress_and_bm_rec_all}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s100.{run}"])  ### 127.49

#### 5l ############################################################################################
sb.run(same_command + [f"L5_ch032_mae_s001.{compress_and_bm_rec_all}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch032_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch032_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch032_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L5_ch016_mae_s001.{compress_and_bm_rec_all}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch016_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch016_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch016_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L5_ch008_mae_s001.{compress_and_bm_rec_all}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch008_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch008_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch008_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L5_ch004_mae_s001.{compress_and_bm_rec_all}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch004_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch004_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch004_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L5_ch002_mae_s001.{compress_and_bm_rec_all}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch002_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch002_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch002_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L5_ch001_mae_s001.{compress_and_bm_rec_all}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s100.{run}"])  ### 127.55

#### 6l ############################################################################################
### coord_conv
# sb.run(same_command + [f"L6_ch064_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L6_ch032_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L6_ch016_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L6_ch016_mae_s020.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s040.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s060.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s080.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s100.{run}"])
### coord_conv
# sb.run(same_command + [f"L6_ch016_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L6_ch008_mae_s001.{compress_and_bm_rec_all}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch008_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L6_ch004_mae_s001.{compress_and_bm_rec_all}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch004_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L6_ch002_mae_s001.{compress_and_bm_rec_all}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch002_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L6_ch001_mae_s001.{compress_and_bm_rec_all}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch001_mae_s001_coord_conv.{run}"])

#### 7l ############################################################################################
### coord_conv
# sb.run(same_command + [f"L7_ch032_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L7_ch016_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L7_ch008_mae_s001.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L7_ch008_mae_s020.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s040.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s060.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s080.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s100.{run}"])
### coord_conv
# sb.run(same_command + [f"L7_ch008_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L7_ch004_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L7_ch002_mae_s001_coord_conv.{run}"])
#### 8l ############################################################################################
### coord_conv
# sb.run(same_command + [f"L8_ch016_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L8_ch008_mae_s001_coord_conv.{run}"])
####################
sb.run(same_command + [f"L8_ch004_mae_s001.{compress_and_bm_rec_all}"])
sb.run(same_command + [f"L8_ch004_mae_s001_copy.{compress_and_bm_rec_all}"])
# sb.run(same_command + [f"L8_ch004_mae_s020.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s040.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s060.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s080.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s100.{run}"])
### coord_conv
# sb.run(same_command + [f"L8_ch004_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L8_ch002_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L8_ch001_mae_s001_coord_conv.{run}"])