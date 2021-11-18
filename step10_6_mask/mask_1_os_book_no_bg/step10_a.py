#############################################################################################################################################################################################################
### 把 current_dir 轉回到 kong_model 裡面
import os
import sys
curr_path = os.getcwd()
curr_layer = len(curr_path.split("\\")) - 1                ### 看 目前執行python的位置在哪一層， -1 是 因為 為了配合下面.index() 從0開始算
kong_layer = curr_path.split("\\").index("kong_model2")    ### 看kong_model2 在哪一層
back_to_kong_layer_amount = curr_layer - kong_layer        ### 看 目前執行python的位置在哪一層 到 kong_model2 差幾層
for _ in range(back_to_kong_layer_amount): os.chdir("..")  ### 看差幾層 往前跳 幾次dir
sys.path.append(".")                                         ### 把 kong_model2 加進 sys.path
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_e_model_obj import *
from step09_b_loss_info_obj import *
from step10_b_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

exp_dir = "6_mask_unet/1_os_book_no_bg"
use_db_obj = type9_mask_flow
#############################  no-bg  ##################################
### 1. ch 結果超棒就直接結束了 沒有做其他嘗試
mask_ch032_tanh_mae_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_tanh_7l, G_mae_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_0_1", describe_end="mask_ch032_tanh_mae_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_0_1-20210914_183346-flow_unet-mask_ch032_tanh_mae_ep060")
mask_ch032_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_7l, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="mask_ch032_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_1-20210915_031618-flow_unet-mask_ch032_sigmoid_bce_ep060")
mask_ch016_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_7l, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_2", describe_end="mask_ch016_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_2-20210915_021235-flow_unet-mask_ch016_sigmoid_bce_ep060")
mask_ch008_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_7l, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3", describe_end="mask_ch008_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_3-20210915_011429-flow_unet-mask_ch008_sigmoid_bce_ep060")
mask_ch004_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_7l, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_4", describe_end="mask_ch004_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_4-20210915_001642-flow_unet-mask_ch004_sigmoid_bce_ep060")
mask_ch002_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_7l, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_5", describe_end="mask_ch002_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_5-20210914_231839-flow_unet-mask_ch002_sigmoid_bce_ep060")
mask_ch001_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_7l, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_6", describe_end="mask_ch001_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_6-20210914_222003-flow_unet-mask_ch001_sigmoid_bce_ep060")

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
