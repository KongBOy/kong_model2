import os
### 自動抓目前的資料夾 當 exp_dir
code_exe_path = os.path.realpath(__file__)
print("code_exe_path~~~~~~~~~~", code_exe_path  )  ### 舉例：C:\Users\TKU\Desktop\kong_model2\step10_6_mask\mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_bce\step10_a.py
exp_dir = "6_mask_unet/" +  code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
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
from step09_e2_mask_unet2_obj import *
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

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
############################  have_bg  #################################
### 1a. ch
mask_h_bg_ch128_sig_L6_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch128_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_1-20210929_034930-flow_unet-mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch064_sig_L6_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch064_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_2", describe_end="mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_2-20210929_025913-flow_unet-mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch032_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_3-20210929_031441-flow_unet-mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch016_sig_L6_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch016_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_4", describe_end="mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_4-20210929_022728-flow_unet-mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch008_sig_L6_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch008_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_5", describe_end="mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_5-20210929_015650-flow_unet-mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch004_sig_L6_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch004_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_6", describe_end="mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_6-20210929_012621-flow_unet-mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch002_sig_L6_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch002_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_7", describe_end="mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_7-20210929_005546-flow_unet-mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch001_sig_L6_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch001_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_8", describe_end="mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_8-20210929_002459-flow_unet-mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep060")
### 1b. ch and epoch_6l
mask_h_bg_ch128_sig_L6_ep200 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_ch128_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_1", describe_end="mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_1-20210929_115738-flow_unet-mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch064_sig_L6_ep200 = Exp_builder().set_com("127.37").set_basic("train", use_db_obj, mask_unet_ch064_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_2", describe_end="mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_2-20210929_092400-flow_unet-mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch032_sig_L6_ep200 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch032_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_3", describe_end="mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_3-20210929_073236-flow_unet-mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch016_sig_L6_ep200 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_ch016_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_4", describe_end="mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_4-20210929_123345-flow_unet-mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch008_sig_L6_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch008_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_5", describe_end="mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_5-20210929_105354-flow_unet-mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch004_sig_L6_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch004_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_6", describe_end="mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_6-20210929_091346-flow_unet-mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch002_sig_L6_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch002_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_7", describe_end="mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_7-20210929_073310-flow_unet-mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep200")
mask_h_bg_ch001_sig_L6_ep200 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_ch001_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_8", describe_end="mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_8-20210929_061337-flow_unet-mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep200")

### 3. no-concat
mask_h_bg_ch032_L6_2to2noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_6l_ch32_2to2noC_sig, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_1", describe_end="mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k3_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_1-20210929_003219-flow_unet-mask_h_bg_ch032_2to2noC_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch032_L6_2to3noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_6l_ch32_2to3noC_sig, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_2", describe_end="mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k3_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_2-20210929_010655-flow_unet-mask_h_bg_ch032_2to3noC_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch032_L6_2to4noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_6l_ch32_2to4noC_sig, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_3", describe_end="mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k3_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_3-20210929_014125-flow_unet-mask_h_bg_ch032_2to4noC_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch032_L6_2to5noC_sig_ep060 = Exp_builder().set_com("127.55").set_basic("train", use_db_obj, mask_unet_IN_6l_ch32_2to5noC_sig, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_4", describe_end="mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k3_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_4-20210929_021553-flow_unet-mask_h_bg_ch032_2to5noC_sig_bce_sobel_k3_6l_ep060")
mask_h_bg_ch032_L6_2to6noC_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_IN_6l_ch32_2to6noC_sig, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_5", describe_end="mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k3_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_5-20210929_025022-flow_unet-mask_h_bg_ch032_2to6noC_sig_bce_sobel_k3_6l_ep060")

### 4. skip use add
mask_h_bg_ch032_L6_skipAdd_sig_ep060 = Exp_builder().set_com("127.35").set_basic("train", use_db_obj, mask_unet_6_level_skip_use_add_sig_6l, G_bce_sobel_k3_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_5", describe_end="mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k3_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_5-20210929_055321-flow_unet-mask_h_bg_ch032_skipAdd_sig_bce_sobel_k3_6l_ep060")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch128_sig_L6_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
