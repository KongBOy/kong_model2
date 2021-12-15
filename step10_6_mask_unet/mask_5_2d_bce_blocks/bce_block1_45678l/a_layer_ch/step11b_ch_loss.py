#############################################################################################################################################################################################################
import os
code_exe_path = os.path.realpath(__file__)     ### 目前執行 step11.py 的 path
code_exe_dir = os.path.dirname(code_exe_path)  ### 目前執行 step11.py 的 dir
import sys                                     ### 把 step11 加入 的 dir sys.path， 在外部 import 這裡的 step11 才 import 的到 step10_a.py
sys.path.append(code_exe_dir)
#############################################################################################################################################################################################################
import step10_a_a_layer_ch as bce_block1

#################################################################################################################################################################################################################################################################################################################################################################################################
### 2l
bce_L2_block1_ch128 = [
    bce_block1.L2_ch128_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch128_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch128_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch128_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch128_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch128_sig_ep060_bce_s100.build()]
bce_L2_block1_ch128 = [ exp.result_obj for exp in bce_L2_block1_ch128]
####################
bce_L2_block1_ch064 = [
    bce_block1.L2_ch064_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch064_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch064_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch064_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch064_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch064_sig_ep060_bce_s100.build()]
bce_L2_block1_ch064 = [ exp.result_obj for exp in bce_L2_block1_ch064]
####################
bce_L2_block1_ch032 = [
    bce_block1.L2_ch032_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch032_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch032_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch032_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch032_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch032_sig_ep060_bce_s100.build()]
bce_L2_block1_ch032 = [ exp.result_obj for exp in bce_L2_block1_ch032]
####################
bce_L2_block1_ch016 = [
    bce_block1.L2_ch016_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch016_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch016_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch016_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch016_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch016_sig_ep060_bce_s100.build()]
bce_L2_block1_ch016 = [ exp.result_obj for exp in bce_L2_block1_ch016]
####################
bce_L2_block1_ch008 = [
    bce_block1.L2_ch008_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch008_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch008_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch008_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch008_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch008_sig_ep060_bce_s100.build()]
bce_L2_block1_ch008 = [ exp.result_obj for exp in bce_L2_block1_ch008]
####################
bce_L2_block1_ch004 = [
    bce_block1.L2_ch004_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch004_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch004_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch004_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch004_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch004_sig_ep060_bce_s100.build()]
bce_L2_block1_ch004 = [ exp.result_obj for exp in bce_L2_block1_ch004]
####################
bce_L2_block1_ch002 = [
    bce_block1.L2_ch002_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch002_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch002_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch002_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch002_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch002_sig_ep060_bce_s100.build()]
bce_L2_block1_ch002 = [ exp.result_obj for exp in bce_L2_block1_ch002]
####################
bce_L2_block1_ch001 = [
    bce_block1.L2_ch001_sig_ep060_bce_s001.build(),
    bce_block1.L2_ch001_sig_ep060_bce_s020.build(),
    bce_block1.L2_ch001_sig_ep060_bce_s040.build(),
    bce_block1.L2_ch001_sig_ep060_bce_s060.build(),
    bce_block1.L2_ch001_sig_ep060_bce_s080.build(),
    bce_block1.L2_ch001_sig_ep060_bce_s100.build()]
bce_L2_block1_ch001 = [ exp.result_obj for exp in bce_L2_block1_ch001]
#################################################################################################################################################################################################################################################################################################################################################################################################
### 3l
bce_L3_block1_ch128 = [
    bce_block1.L3_ch128_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch128_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch128_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch128_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch128_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch128_sig_ep060_bce_s100.build()]
bce_L3_block1_ch128 = [ exp.result_obj for exp in bce_L3_block1_ch128]
####################
bce_L3_block1_ch064 = [
    bce_block1.L3_ch064_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch064_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch064_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch064_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch064_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch064_sig_ep060_bce_s100.build()]
bce_L3_block1_ch064 = [ exp.result_obj for exp in bce_L3_block1_ch064]
####################
bce_L3_block1_ch032 = [
    bce_block1.L3_ch032_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch032_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch032_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch032_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch032_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch032_sig_ep060_bce_s100.build()]
bce_L3_block1_ch032 = [ exp.result_obj for exp in bce_L3_block1_ch032]
####################
bce_L3_block1_ch016 = [
    bce_block1.L3_ch016_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch016_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch016_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch016_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch016_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch016_sig_ep060_bce_s100.build()]
bce_L3_block1_ch016 = [ exp.result_obj for exp in bce_L3_block1_ch016]
####################
bce_L3_block1_ch008 = [
    bce_block1.L3_ch008_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch008_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch008_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch008_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch008_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch008_sig_ep060_bce_s100.build()]
bce_L3_block1_ch008 = [ exp.result_obj for exp in bce_L3_block1_ch008]
####################
bce_L3_block1_ch004 = [
    bce_block1.L3_ch004_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch004_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch004_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch004_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch004_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch004_sig_ep060_bce_s100.build()]
bce_L3_block1_ch004 = [ exp.result_obj for exp in bce_L3_block1_ch004]
####################
bce_L3_block1_ch002 = [
    bce_block1.L3_ch002_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch002_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch002_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch002_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch002_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch002_sig_ep060_bce_s100.build()]
bce_L3_block1_ch002 = [ exp.result_obj for exp in bce_L3_block1_ch002]
####################
bce_L3_block1_ch001 = [
    bce_block1.L3_ch001_sig_ep060_bce_s001.build(),
    bce_block1.L3_ch001_sig_ep060_bce_s020.build(),
    bce_block1.L3_ch001_sig_ep060_bce_s040.build(),
    bce_block1.L3_ch001_sig_ep060_bce_s060.build(),
    bce_block1.L3_ch001_sig_ep060_bce_s080.build(),
    bce_block1.L3_ch001_sig_ep060_bce_s100.build()]
bce_L3_block1_ch001 = [ exp.result_obj for exp in bce_L3_block1_ch001]
#################################################################################################################################################################################################################################################################################################################################################################################################
### 4l
bce_L4_block1_ch064 = [
    bce_block1.L4_ch064_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch064_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch064_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch064_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch064_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch064_sig_ep060_bce_s100.build()]
bce_L4_block1_ch064 = [ exp.result_obj for exp in bce_L4_block1_ch064]
####################
bce_L4_block1_ch032 = [
    bce_block1.L4_ch032_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch032_sig_ep060_bce_s100.build()]
bce_L4_block1_ch032 = [ exp.result_obj for exp in bce_L4_block1_ch032]
####################
bce_L4_block1_ch016 = [
    bce_block1.L4_ch016_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch016_sig_ep060_bce_s100.build()]
bce_L4_block1_ch016 = [ exp.result_obj for exp in bce_L4_block1_ch016]

####################
bce_L4_block1_ch008 = [
    bce_block1.L4_ch008_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch008_sig_ep060_bce_s100.build()]
bce_L4_block1_ch008 = [ exp.result_obj for exp in bce_L4_block1_ch008]

####################
bce_L4_block1_ch004 = [
    bce_block1.L4_ch004_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch004_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch004_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch004_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch004_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch004_sig_ep060_bce_s100.build()]
bce_L4_block1_ch004 = [ exp.result_obj for exp in bce_L4_block1_ch004]
####################
bce_L4_block1_ch002 = [
    bce_block1.L4_ch002_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch002_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch002_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch002_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch002_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch002_sig_ep060_bce_s100.build()]
bce_L4_block1_ch002 = [ exp.result_obj for exp in bce_L4_block1_ch002]
####################
bce_L4_block1_ch001 = [
    bce_block1.L4_ch001_sig_ep060_bce_s001.build(),
    bce_block1.L4_ch001_sig_ep060_bce_s020.build(),
    bce_block1.L4_ch001_sig_ep060_bce_s040.build(),
    bce_block1.L4_ch001_sig_ep060_bce_s060.build(),
    bce_block1.L4_ch001_sig_ep060_bce_s080.build(),
    bce_block1.L4_ch001_sig_ep060_bce_s100.build()]
bce_L4_block1_ch001 = [ exp.result_obj for exp in bce_L4_block1_ch001]
#################################################################################################################################################################################################################################################################################################################################################################################################
### 5l
bce_L5_block1_ch032 = [
    bce_block1.L5_ch032_sig_ep060_bce_s001.build(),
    bce_block1.L5_ch032_sig_ep060_bce_s020.build(),
    bce_block1.L5_ch032_sig_ep060_bce_s040.build(),
    bce_block1.L5_ch032_sig_ep060_bce_s060.build(),
    bce_block1.L5_ch032_sig_ep060_bce_s080.build(),
    bce_block1.L5_ch032_sig_ep060_bce_s100.build()]
bce_L5_block1_ch032 = [ exp.result_obj for exp in bce_L5_block1_ch032]
####################
bce_L5_block1_ch016 = [
    bce_block1.L5_ch016_sig_ep060_bce_s001.build(),
    bce_block1.L5_ch016_sig_ep060_bce_s020.build(),
    bce_block1.L5_ch016_sig_ep060_bce_s040.build(),
    bce_block1.L5_ch016_sig_ep060_bce_s060.build(),
    bce_block1.L5_ch016_sig_ep060_bce_s080.build(),
    bce_block1.L5_ch016_sig_ep060_bce_s100.build()]
bce_L5_block1_ch016 = [ exp.result_obj for exp in bce_L5_block1_ch016]
####################
bce_L5_block1_ch008 = [
    bce_block1.L5_ch008_sig_ep060_bce_s001.build(),
    bce_block1.L5_ch008_sig_ep060_bce_s020.build(),
    bce_block1.L5_ch008_sig_ep060_bce_s040.build(),
    bce_block1.L5_ch008_sig_ep060_bce_s060.build(),
    bce_block1.L5_ch008_sig_ep060_bce_s080.build(),
    bce_block1.L5_ch008_sig_ep060_bce_s100.build()]
bce_L5_block1_ch008 = [ exp.result_obj for exp in bce_L5_block1_ch008]
####################
bce_L5_block1_ch004 = [
    bce_block1.L5_ch004_sig_ep060_bce_s001.build(),
    bce_block1.L5_ch004_sig_ep060_bce_s020.build(),
    bce_block1.L5_ch004_sig_ep060_bce_s040.build(),
    bce_block1.L5_ch004_sig_ep060_bce_s060.build(),
    bce_block1.L5_ch004_sig_ep060_bce_s080.build(),
    bce_block1.L5_ch004_sig_ep060_bce_s100.build()]
bce_L5_block1_ch004 = [ exp.result_obj for exp in bce_L5_block1_ch004]
####################
bce_L5_block1_ch002 = [
    bce_block1.L5_ch002_sig_ep060_bce_s001.build(),
    bce_block1.L5_ch002_sig_ep060_bce_s020.build(),
    bce_block1.L5_ch002_sig_ep060_bce_s040.build(),
    bce_block1.L5_ch002_sig_ep060_bce_s060.build(),
    bce_block1.L5_ch002_sig_ep060_bce_s080.build(),
    bce_block1.L5_ch002_sig_ep060_bce_s100.build()]
bce_L5_block1_ch002 = [ exp.result_obj for exp in bce_L5_block1_ch002]
####################
bce_L5_block1_ch001 = [
    bce_block1.L5_ch001_sig_ep060_bce_s001.build(),
    bce_block1.L5_ch001_sig_ep060_bce_s020.build(),
    bce_block1.L5_ch001_sig_ep060_bce_s040.build(),
    bce_block1.L5_ch001_sig_ep060_bce_s060.build(),
    bce_block1.L5_ch001_sig_ep060_bce_s080.build(),
    bce_block1.L5_ch001_sig_ep060_bce_s100.build()]
bce_L5_block1_ch001 = [ exp.result_obj for exp in bce_L5_block1_ch001]
#################################################################################################################################################################################################################################################################################################################################################################################################
### 6l
bce_L6_block1_ch016 = [
    bce_block1.L6_ch016_sig_ep060_bce_s001.build(),
    bce_block1.L6_ch016_sig_ep060_bce_s020.build(),
    bce_block1.L6_ch016_sig_ep060_bce_s040.build(),
    bce_block1.L6_ch016_sig_ep060_bce_s060.build(),
    bce_block1.L6_ch016_sig_ep060_bce_s080.build(),
    bce_block1.L6_ch016_sig_ep060_bce_s100.build()]
bce_L6_block1_ch016 = [ exp.result_obj for exp in bce_L6_block1_ch016]
####################
bce_L6_block1_ch008 = [
    bce_block1.L6_ch008_sig_ep060_bce_s001.build(),
    bce_block1.L6_ch008_sig_ep060_bce_s020.build(),
    bce_block1.L6_ch008_sig_ep060_bce_s040.build(),
    bce_block1.L6_ch008_sig_ep060_bce_s060.build(),
    bce_block1.L6_ch008_sig_ep060_bce_s080.build(),
    bce_block1.L6_ch008_sig_ep060_bce_s100.build()]
bce_L6_block1_ch008 = [ exp.result_obj for exp in bce_L6_block1_ch008]
####################
bce_L6_block1_ch004 = [
    bce_block1.L6_ch004_sig_ep060_bce_s001.build(),
    bce_block1.L6_ch004_sig_ep060_bce_s020.build(),
    bce_block1.L6_ch004_sig_ep060_bce_s040.build(),
    bce_block1.L6_ch004_sig_ep060_bce_s060.build(),
    bce_block1.L6_ch004_sig_ep060_bce_s080.build(),
    bce_block1.L6_ch004_sig_ep060_bce_s100.build()]
bce_L6_block1_ch004 = [ exp.result_obj for exp in bce_L6_block1_ch004]
####################
bce_L6_block1_ch002 = [
    bce_block1.L6_ch002_sig_ep060_bce_s001.build(),
    bce_block1.L6_ch002_sig_ep060_bce_s020.build(),
    bce_block1.L6_ch002_sig_ep060_bce_s040.build(),
    bce_block1.L6_ch002_sig_ep060_bce_s060.build(),
    bce_block1.L6_ch002_sig_ep060_bce_s080.build(),
    bce_block1.L6_ch002_sig_ep060_bce_s100.build()]
bce_L6_block1_ch002 = [ exp.result_obj for exp in bce_L6_block1_ch002]
####################
bce_L6_block1_ch001 = [
    bce_block1.L6_ch001_sig_ep060_bce_s001.build(),
    bce_block1.L6_ch001_sig_ep060_bce_s020.build(),
    bce_block1.L6_ch001_sig_ep060_bce_s040.build(),
    bce_block1.L6_ch001_sig_ep060_bce_s060.build(),
    bce_block1.L6_ch001_sig_ep060_bce_s080.build(),
    bce_block1.L6_ch001_sig_ep060_bce_s100.build()]
bce_L6_block1_ch001 = [ exp.result_obj for exp in bce_L6_block1_ch001]
#################################################################################################################################################################################################################################################################################################################################################################################################
### 7l
bce_L7_block1_ch008 = [
    bce_block1.L7_ch008_sig_ep060_bce_s001.build(),
    bce_block1.L7_ch008_sig_ep060_bce_s020.build(),
    bce_block1.L7_ch008_sig_ep060_bce_s040.build(),
    bce_block1.L7_ch008_sig_ep060_bce_s060.build(),
    bce_block1.L7_ch008_sig_ep060_bce_s080.build(),
    bce_block1.L7_ch008_sig_ep060_bce_s100.build()]
bce_L7_block1_ch008 = [ exp.result_obj for exp in bce_L7_block1_ch008]
#################################################################################################################################################################################################################################################################################################################################################################################################
### 8l
bce_L8_block1_ch004 = [
    bce_block1.L8_ch004_sig_ep060_bce_s001.build(),
    bce_block1.L8_ch004_sig_ep060_bce_s020.build(),
    bce_block1.L8_ch004_sig_ep060_bce_s040.build(),
    bce_block1.L8_ch004_sig_ep060_bce_s060.build(),
    bce_block1.L8_ch004_sig_ep060_bce_s080.build(),
    bce_block1.L8_ch004_sig_ep060_bce_s100.build()]
bce_L8_block1_ch004 = [ exp.result_obj for exp in bce_L8_block1_ch004]
#################################################################################################################################################################################################################################################################################################################################################################################################