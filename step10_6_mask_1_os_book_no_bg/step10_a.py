import os
import sys
os.chdir("..")
sys.path.append(".")

from step06_a_datas_obj import *
from step08_e_model_obj import *
from step09_a_loss_info_obj import *
from step10_a_load_and_train_and_test import Exp_builder
#############################################################################################################################################################################################################
### 1 試試看 mask
exp_dir16 = "6_mask_unet/1 see_no_bg_easy"
#############################  no-bg  ##################################
### 1. ch 結果超棒就直接結束了 沒有做其他嘗試
mask_ch032_tanh_mae_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch032_tanh, G_mae_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch032_tanh_mae_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210914_183346-flow_unet-mask_ch032_tanh_mae_ep060")
mask_ch032_sig_bce_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch032_sigmoid, G_bce_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch032_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210915_031618-flow_unet-mask_ch032_sigmoid_bce_ep060")
mask_ch016_sig_bce_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch016_sigmoid, G_bce_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch016_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210915_021235-flow_unet-mask_ch016_sigmoid_bce_ep060")
mask_ch008_sig_bce_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch008_sigmoid, G_bce_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch008_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210915_011429-flow_unet-mask_ch008_sigmoid_bce_ep060")
mask_ch004_sig_bce_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch004_sigmoid, G_bce_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch004_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210915_001642-flow_unet-mask_ch004_sigmoid_bce_ep060")
mask_ch002_sig_bce_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch002_sigmoid, G_bce_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch002_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210914_231839-flow_unet-mask_ch002_sigmoid_bce_ep060")
mask_ch001_sig_bce_ep060 = Exp_builder().set_basic("train", type9_try_flow_mask, mask_unet_ch001_sigmoid, G_bce_loss_info_builder, exp_dir=exp_dir16, describe_mid="6_1", describe_end="mask_ch001_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1-20210914_222003-flow_unet-mask_ch001_sigmoid_bce_ep060")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch128_sig_bce_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
