#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
# print("    kong_to_py_layer:", kong_to_py_layer)
if  (kong_to_py_layer == 0): template_dir = ""
elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][0:]  ### [7:] 是為了去掉 step1x_， 後來覺得好像改有意義的名字不去掉也行所以 改 0
elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子， 後來覺得 自動排的順序也可以接受， 所以 改0
elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])
# print("    template_dir:", template_dir)  ### 舉例： template_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
exp_dir = template_dir
#############################################################################################################################################################################################################

from step06_a_datas_obj import *
from step09_e2_mask_unet2_obj import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder
#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_and_paper
############################  have_bg  #################################
### 1a. ch
mask_h_bg_ch128_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch128_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="mask_h_bg_ch128_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch064_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_2", describe_end="mask_h_bg_ch064_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch016_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_4", describe_end="mask_h_bg_ch016_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch008_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_5", describe_end="mask_h_bg_ch008_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch004_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_6", describe_end="mask_h_bg_ch004_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch002_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_7", describe_end="mask_h_bg_ch002_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch001_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_8", describe_end="mask_h_bg_ch001_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
### 1b. ch and epoch
mask_h_bg_ch128_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch128_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_1", describe_end="mask_h_bg_ch128_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch064_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_2", describe_end="mask_h_bg_ch064_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_3", describe_end="mask_h_bg_ch032_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch016_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_4", describe_end="mask_h_bg_ch016_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch008_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_5", describe_end="mask_h_bg_ch008_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch004_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_6", describe_end="mask_h_bg_ch004_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch002_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_7", describe_end="mask_h_bg_ch002_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch001_sig_bce_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_L7, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_8", describe_end="mask_h_bg_ch001_sig_bce_ep200") .set_train_args(epochs=200).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

### 2. level
mask_h_bg_ch032_L2_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L2_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_1", describe_end="mask_h_bg_ch032_L2_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L3_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L3_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_2", describe_end="mask_h_bg_ch032_L3_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L4_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L4_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_3", describe_end="mask_h_bg_ch032_L4_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L5_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L5_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_4", describe_end="mask_h_bg_ch032_L5_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L6_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L6_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_5", describe_end="mask_h_bg_ch032_L6_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L7_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_6", describe_end="mask_h_bg_ch032_L7_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L8_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L8_ch32_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_2_7", describe_end="mask_h_bg_ch032_L8_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

### 3. no-concat
mask_h_bg_ch032_L7_2to2noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to2noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_1", describe_end="mask_h_bg_ch032_7l_2to2noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_2to3noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to3noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_2", describe_end="mask_h_bg_ch032_7l_2to3noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_2to4noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to4noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_3", describe_end="mask_h_bg_ch032_7l_2to4noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_2to5noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to5noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_4", describe_end="mask_h_bg_ch032_7l_2to5noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_2to6noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to6noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_5", describe_end="mask_h_bg_ch032_7l_2to6noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_2to7noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to7noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_6", describe_end="mask_h_bg_ch032_7l_2to7noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_2to8noC_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L7_ch32_2to8noC_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_7", describe_end="mask_h_bg_ch032_7l_2to8noC_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

### 4. skip use add
mask_h_bg_ch032_2l_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L2_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_1", describe_end="mask_h_bg_ch032_2l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_3l_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L3_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_2", describe_end="mask_h_bg_ch032_3l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_4l_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L4_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_3", describe_end="mask_h_bg_ch032_4l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_5l_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L5_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_4", describe_end="mask_h_bg_ch032_5l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L6_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_5", describe_end="mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_L7_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L7_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_6", describe_end="mask_h_bg_ch032_7l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch032_8l_skipAdd_sig_bce_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L8_skip_use_add_sig, G_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_7", describe_end="mask_h_bg_ch032_8l_skipAdd_sig_bce_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch128_sig_bce_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
