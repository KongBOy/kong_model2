import os
### 自動抓目前的資料夾 當 exp_dir
print("os.path.realpath(__file__) ", os.path.realpath(__file__)  )  ### 舉例：os.path.realpath(__file__)  C:\Users\TKU\Desktop\kong_model2\step10_6_mask\mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s80_bce\step10_a.py
exp_dir = "6_mask_unet/" +  os.path.realpath(__file__).split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
print("exp_dir~~~~~~~~~~", exp_dir)  ### 舉例：exp_dir~~~~~~~~~~ 6_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s80_bce
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
mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s20_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_1", describe_end="mask_h_bg_ch032_sig_tv_s20_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s20_bce_s020_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_2", describe_end="mask_h_bg_ch032_sig_tv_s20_bce_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s20_bce_s040_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_tv_s20_bce_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s20_bce_s060_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_4", describe_end="mask_h_bg_ch032_sig_tv_s20_bce_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s20_bce_s080_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_5", describe_end="mask_h_bg_ch032_sig_tv_s20_bce_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s001 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s30_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_1", describe_end="mask_h_bg_ch032_sig_tv_s30_bce_s001_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s020 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s30_bce_s020_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_2", describe_end="mask_h_bg_ch032_sig_tv_s30_bce_s020_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s040 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s30_bce_s040_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_3", describe_end="mask_h_bg_ch032_sig_tv_s30_bce_s040_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s30_bce_s060_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_4", describe_end="mask_h_bg_ch032_sig_tv_s30_bce_s060_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s080 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_sobel_6l, G_tv_s30_bce_s080_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_5", describe_end="mask_h_bg_ch032_sig_tv_s30_bce_s080_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s001.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
