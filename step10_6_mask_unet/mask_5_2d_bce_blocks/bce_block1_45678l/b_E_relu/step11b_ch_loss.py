#############################################################################################################################################################################################################
import os
code_exe_path = os.path.realpath(__file__)     ### 目前執行 step11.py 的 path
code_exe_dir = os.path.dirname(code_exe_path)  ### 目前執行 step11.py 的 dir
import sys                                     ### 把 step11 加入 的 dir sys.path， 在外部 import 這裡的 step11 才 import 的到 step10_a.py
sys.path.append(code_exe_dir)
#############################################################################################################################################################################################################
import step10_a_b_E_relu as bce_block1


#################################################################################################################################################################################################################################################################################################################################################################################################
### 4l
### E_relu
bce_L4_block1_ch032_E_relu = [
    bce_block1.L4_ch032_sig_ep060_bce_s001_E_relu.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s020_E_relu.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s040_E_relu.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s060_E_relu.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s080_E_relu.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s100_E_relu.build(),
]
bce_L4_block1_ch032_E_relu = [ exp.result_obj for exp in bce_L4_block1_ch032_E_relu]

####################

### E_relu
bce_L4_block1_ch016_E_relu = [
    bce_block1.L4_ch016_sig_ep060_bce_s001_E_relu.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s020_E_relu.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s040_E_relu.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s060_E_relu.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s080_E_relu.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s100_E_relu.build(),
]
bce_L4_block1_ch016_E_relu = [ exp.result_obj for exp in bce_L4_block1_ch016_E_relu]
####################
### E_relu
bce_L4_block1_ch008_E_relu = [
    bce_block1.L4_ch008_sig_ep060_bce_s001_E_relu.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s020_E_relu.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s040_E_relu.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s060_E_relu.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s080_E_relu.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s100_E_relu.build(),
]
bce_L4_block1_ch008_E_relu = [ exp.result_obj for exp in bce_L4_block1_ch008_E_relu]
