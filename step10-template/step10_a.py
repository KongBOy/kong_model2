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

exp_dir = "6_mask_unet/自己命的名字"
use_db_obj = type9_try_flow_mask_have_bg_and_paper
############################  have_bg  #################################
### 1a. ch
mask_h_bg_ch128_sig_bce_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch128_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_1", describe_end="mask_h_bg_ch128_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_bce_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch064_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_2", describe_end="mask_h_bg_ch064_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch032_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch016_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch016_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_4", describe_end="mask_h_bg_ch016_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch008_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch008_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_5", describe_end="mask_h_bg_ch008_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch004_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch004_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_6", describe_end="mask_h_bg_ch004_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch002_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch002_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_7", describe_end="mask_h_bg_ch002_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch001_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch001_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1_8", describe_end="mask_h_bg_ch001_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
### 1b. ch and epoch
mask_h_bg_ch128_sig_bce_ep200 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_ch128_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_1", describe_end="mask_h_bg_ch128_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch064_sig_bce_ep200 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_ch064_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_2", describe_end="mask_h_bg_ch064_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_sig_bce_ep200 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch032_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_3", describe_end="mask_h_bg_ch032_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch016_sig_bce_ep200 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch016_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_4", describe_end="mask_h_bg_ch016_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch008_sig_bce_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch008_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_5", describe_end="mask_h_bg_ch008_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch004_sig_bce_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch004_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_6", describe_end="mask_h_bg_ch004_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch002_sig_bce_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch002_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_7", describe_end="mask_h_bg_ch002_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch001_sig_bce_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch001_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_1b_8", describe_end="mask_h_bg_ch001_sig_bce_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

### 2. level
mask_h_bg_ch032_2l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_2_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_1", describe_end="mask_h_bg_ch032_2l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_3l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_3_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_2", describe_end="mask_h_bg_ch032_3l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_4l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_4_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_3", describe_end="mask_h_bg_ch032_4l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_5l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_5_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_4", describe_end="mask_h_bg_ch032_5l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_6l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_6_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_5", describe_end="mask_h_bg_ch032_6l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_7_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_6", describe_end="mask_h_bg_ch032_7l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_8l_sig_bce_ep060 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_8_level_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_2_7", describe_end="mask_h_bg_ch032_8l_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

### 3. no-concat
mask_h_bg_ch032_7l_2to2noC_sig_bce_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to2noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_1", describe_end="mask_h_bg_ch032_7l_2to2noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_2to3noC_sig_bce_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to3noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_2", describe_end="mask_h_bg_ch032_7l_2to3noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_2to4noC_sig_bce_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to4noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_3", describe_end="mask_h_bg_ch032_7l_2to4noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_2to5noC_sig_bce_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to5noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_4", describe_end="mask_h_bg_ch032_7l_2to5noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_2to6noC_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to6noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_5", describe_end="mask_h_bg_ch032_7l_2to6noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_2to7noC_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to7noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_6", describe_end="mask_h_bg_ch032_7l_2to7noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_2to8noC_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_IN_7l_ch32_2to8noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_3_7", describe_end="mask_h_bg_ch032_7l_2to8noC_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

### 4. skip use add
mask_h_bg_ch032_2l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_2_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_1", describe_end="mask_h_bg_ch032_2l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_3l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_3_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_2", describe_end="mask_h_bg_ch032_3l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_4l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_4_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_3", describe_end="mask_h_bg_ch032_4l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_5l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_5_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_4", describe_end="mask_h_bg_ch032_5l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_6_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_5", describe_end="mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_7l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_7_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_6", describe_end="mask_h_bg_ch032_7l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")
mask_h_bg_ch032_8l_skipAdd_sig_bce_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_8_level_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, describe_mid="6_4_7", describe_end="mask_h_bg_ch032_8l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="")

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
