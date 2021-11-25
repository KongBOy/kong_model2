#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print("step10a")
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
exp_dir = code_exe_path_element[5][7:] + "/" + code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
# print("    exp_dir:", exp_dir)  ### 舉例：exp_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
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
mask_h_bg_ch128_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch128_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="mask_h_bg_ch128_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_1-20211025_155057-flow_unet-mask_h_bg_ch128_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch064_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_2", describe_end="mask_h_bg_ch064_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_2-20211018_153052-flow_unet-mask_h_bg_ch064_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch032_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_3-20211018_145602-flow_unet-mask_h_bg_ch032_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch016_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_4", describe_end="mask_h_bg_ch016_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_4-20211018_142333-flow_unet-mask_h_bg_ch016_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch008_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_5", describe_end="mask_h_bg_ch008_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_5-20211025_184737-flow_unet-mask_h_bg_ch008_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch004_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_6", describe_end="mask_h_bg_ch004_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_6-20211025_181827-flow_unet-mask_h_bg_ch004_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch002_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_7", describe_end="mask_h_bg_ch002_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_7-20211025_174908-flow_unet-mask_h_bg_ch002_sig_tv_bce_sobel_k5_6l_ep060")
mask_h_bg_ch001_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_8", describe_end="mask_h_bg_ch001_sig_tv_bce_sobel_k5_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1_8-20211025_171957-flow_unet-mask_h_bg_ch001_sig_tv_bce_sobel_k5_6l_ep060")
### 1b. ch and epoch_6l
mask_h_bg_ch128_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch128_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_1", describe_end="mask_h_bg_ch128_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_1-20211026_075254-flow_unet-mask_h_bg_ch128_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch064_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_2", describe_end="mask_h_bg_ch064_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_2-20211026_051515-flow_unet-mask_h_bg_ch064_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch032_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_3", describe_end="mask_h_bg_ch032_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_3-20211026_032558-flow_unet-mask_h_bg_ch032_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch016_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_4", describe_end="mask_h_bg_ch016_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_4-20211026_014509-flow_unet-mask_h_bg_ch016_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch008_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_5", describe_end="mask_h_bg_ch008_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_5-20211026_000737-flow_unet-mask_h_bg_ch008_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch004_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_6", describe_end="mask_h_bg_ch004_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_6-20211025_223040-flow_unet-mask_h_bg_ch004_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch002_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_7", describe_end="mask_h_bg_ch002_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_7-20211025_205346-flow_unet-mask_h_bg_ch002_sig_tv_bce_sobel_k5_6l_ep200")
mask_h_bg_ch001_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_L6, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_8", describe_end="mask_h_bg_ch001_sig_tv_bce_sobel_k5_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_1b_8-20211025_191654-flow_unet-mask_h_bg_ch001_sig_tv_bce_sobel_k5_6l_ep200")

### 3. no-concat
mask_h_bg_ch032_L6_2to2noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to2noC_sig, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_1", describe_end="mask_h_bg_ch032_6l_2to2noC_sig_tv_bce_sobel_k5_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_1-20211018_164153-flow_unet-mask_h_bg_ch032_6l_2to2noC_sig_tv_bce_sobel_k5_ep060")
mask_h_bg_ch032_L6_2to3noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to3noC_sig, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_2", describe_end="mask_h_bg_ch032_6l_2to3noC_sig_tv_bce_sobel_k5_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_2-20211018_171640-flow_unet-mask_h_bg_ch032_6l_2to3noC_sig_tv_bce_sobel_k5_ep060")
mask_h_bg_ch032_L6_2to4noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to4noC_sig, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_3", describe_end="mask_h_bg_ch032_6l_2to4noC_sig_tv_bce_sobel_k5_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_3-20211018_175131-flow_unet-mask_h_bg_ch032_6l_2to4noC_sig_tv_bce_sobel_k5_ep060")
mask_h_bg_ch032_L6_2to5noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to5noC_sig, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_4", describe_end="mask_h_bg_ch032_6l_2to5noC_sig_tv_bce_sobel_k5_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_4-20211018_182634-flow_unet-mask_h_bg_ch032_6l_2to5noC_sig_tv_bce_sobel_k5_ep060")
mask_h_bg_ch032_L6_2to6noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to6noC_sig, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_5", describe_end="mask_h_bg_ch032_6l_2to6noC_sig_tv_bce_sobel_k5_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_3_5-20211018_194403-flow_unet-mask_h_bg_ch032_6l_2to6noC_sig_tv_bce_sobel_k5_ep060")

### 4. skip use add
mask_h_bg_ch032_L6_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L6_skip_use_add_sig, G_tv_bce_sobel_k5_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_5", describe_end="mask_h_bg_ch032_6l_skipAdd_sig_tv_bce_sobel_k5_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-6_4_5-20211018_194409-flow_unet-mask_h_bg_ch032_6l_skipAdd_sig_tv_bce_sobel_k5_ep060")

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
