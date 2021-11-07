import os
### 自動抓目前的資料夾 當 exp_dir
print("os.path.realpath(__file__) ", os.path.realpath(__file__)  )  ### 舉例：os.path.realpath(__file__)  C:\Users\TKU\Desktop\kong_model2\step10_6_mask\mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_bce\step10_a.py
exp_dir = "6_mask_unet/" +  os.path.realpath(__file__).split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
print("exp_dir~~~~~~~~~~", exp_dir)  ### 舉例：exp_dir~~~~~~~~~~ 6_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_bce
#############################################################################################################################################################################################################
### 把 current_dir 轉回到 kong_model 裡面
import os
import sys
curr_path = os.getcwd()
curr_layer = len(curr_path.split("\\")) - 1              ### 看 目前執行python的位置在哪一層， -1 是 因為 為了配合下面.index() 從0開始算
kong_layer = curr_path.split("\\").index("kong_model2")  ### 看kong_model2 在哪一層
back_to_kong_layer_amount = curr_layer - kong_layer      ### 看 目前執行python的位置在哪一層 到 kong_model2 差幾層
for _ in range(back_to_kong_layer_amount): os.chdir("..")  ### 看差幾層 往前跳 幾次dir
sys.path.append(".")                                           ### 把 kong_model2 加進 sys.path
print("os.getcwd()", os.getcwd())
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step08_e_model_obj import *
from step09_b_loss_info_obj import *
from step10_a_load_and_train_and_test import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

use_db_obj = type9_try_flow_mask_have_bg_dtd_hdr_mix_and_paper
############################  have_bg  #################################
### 1a. ch
ch128_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_1", describe_end="block1_ch128_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch064_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_2", describe_end="block1_ch064_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_3", describe_end="block1_ch032_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch016_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_4", describe_end="block1_ch016_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch008_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_5", describe_end="block1_ch008_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch004_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_6", describe_end="block1_ch004_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch002_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_7", describe_end="block1_ch002_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch001_sig_4l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_4l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4l_8", describe_end="block1_ch001_sig_tv_s001_bce_s001_4l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

ch128_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_1", describe_end="block1_ch128_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch064_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_2", describe_end="block1_ch064_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_3", describe_end="block1_ch032_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch016_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_4", describe_end="block1_ch016_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch008_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_5", describe_end="block1_ch008_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch004_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_6", describe_end="block1_ch004_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch002_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_7", describe_end="block1_ch002_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch001_sig_5l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_5l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_5l_8", describe_end="block1_ch001_sig_tv_s001_bce_s001_5l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

ch128_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_1", describe_end="block1_ch128_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch064_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_2", describe_end="block1_ch064_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_3", describe_end="block1_ch032_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch016_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_4", describe_end="block1_ch016_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch008_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_5", describe_end="block1_ch008_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch004_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_6", describe_end="block1_ch004_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch002_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_7", describe_end="block1_ch002_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch001_sig_6l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_6l_8", describe_end="block1_ch001_sig_tv_s001_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

ch128_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_1", describe_end="block1_ch128_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch064_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_2", describe_end="block1_ch064_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_3", describe_end="block1_ch032_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch016_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_4", describe_end="block1_ch016_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch008_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_5", describe_end="block1_ch008_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch004_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_6", describe_end="block1_ch004_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch002_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_7", describe_end="block1_ch002_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch001_sig_7l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_7l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_7l_8", describe_end="block1_ch001_sig_tv_s001_bce_s001_7l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

ch128_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_1", describe_end="block1_ch128_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch064_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_2", describe_end="block1_ch064_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_3", describe_end="block1_ch032_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch016_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_4", describe_end="block1_ch016_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch008_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_5", describe_end="block1_ch008_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch004_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_6", describe_end="block1_ch004_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch002_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_7", describe_end="block1_ch002_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch001_sig_8l_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_8l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_8l_8", describe_end="block1_ch001_sig_tv_s001_bce_s001_8l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

'''
### 1b. ch and epoch_6l
ch128_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch128_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_1", describe_end="block1_ch128_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch064_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch064_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_2", describe_end="block1_ch064_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch032_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_3", describe_end="block1_ch032_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch016_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch016_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_4", describe_end="block1_ch016_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch008_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch008_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_5", describe_end="block1_ch008_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch004_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch004_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_6", describe_end="block1_ch004_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch002_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch002_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_7", describe_end="block1_ch002_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch001_sig_6l_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_ch001_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_8", describe_end="block1_ch001_sig_tv_s001_bce_s001_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

### 3. no-concat
ch032_6l_2to2noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_6l_ch32_2to2noC_sig, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_1", describe_end="block1_ch032_6l_2to2noC_sig_tv_s001_bce_s001_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_6l_2to3noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_6l_ch32_2to3noC_sig, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_2", describe_end="block1_ch032_6l_2to3noC_sig_tv_s001_bce_s001_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_6l_2to4noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_6l_ch32_2to4noC_sig, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_3", describe_end="block1_ch032_6l_2to4noC_sig_tv_s001_bce_s001_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_6l_2to5noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_6l_ch32_2to5noC_sig, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_4", describe_end="block1_ch032_6l_2to5noC_sig_tv_s001_bce_s001_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
ch032_6l_2to6noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_IN_6l_ch32_2to6noC_sig, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_5", describe_end="block1_ch032_6l_2to6noC_sig_tv_s001_bce_s001_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

### 4. skip use add
ch032_6l_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet2_block1_6_level_skip_use_add_sig_6l, G_tv_s01_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_5", describe_end="block1_ch032_6l_skipAdd_sig_tv_s001_bce_s001_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
'''

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        ch016_sig_4l_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
