#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])    ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer
# print("    kong_to_py_layer:", kong_to_py_layer)
if  (kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][7:]  ### [7:] 是為了去掉 step1x_
elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
# print("    template_dir:", template_dir)  ### 舉例： template_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
exp_dir = template_dir
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
mask_h_bg_ch128_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch128_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_1", describe_end="mask_h_bg_ch128_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_1-flow_unet-mask_h_bg_ch128_sig_tv_s08_bce_6l_ep060-20211020_224645")  #.result_name_v1_to_v2()
mask_h_bg_ch064_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_2", describe_end="mask_h_bg_ch064_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_2-flow_unet-mask_h_bg_ch064_sig_tv_s08_bce_6l_ep060-20211020_220724")  #.result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3", describe_end="mask_h_bg_ch032_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3-flow_unet-mask_h_bg_ch032_sig_tv_s08_bce_6l_ep060-20211020_213604")  #.result_name_v1_to_v2()
mask_h_bg_ch016_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_4", describe_end="mask_h_bg_ch016_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_4-flow_unet-mask_h_bg_ch016_sig_tv_s08_bce_6l_ep060-20211020_210636")  #.result_name_v1_to_v2()
mask_h_bg_ch008_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_5", describe_end="mask_h_bg_ch008_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_5-flow_unet-mask_h_bg_ch008_sig_tv_s08_bce_6l_ep060-20211021_043607")  #.result_name_v1_to_v2()
mask_h_bg_ch004_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_6", describe_end="mask_h_bg_ch004_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_6-flow_unet-mask_h_bg_ch004_sig_tv_s08_bce_6l_ep060-20211021_040717")  #.result_name_v1_to_v2()
mask_h_bg_ch002_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_7", describe_end="mask_h_bg_ch002_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_7-flow_unet-mask_h_bg_ch002_sig_tv_s08_bce_6l_ep060-20211021_033826")  #.result_name_v1_to_v2()
mask_h_bg_ch001_sig_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_8", describe_end="mask_h_bg_ch001_sig_tv_s08_bce_6l_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_8-flow_unet-mask_h_bg_ch001_sig_tv_s08_bce_6l_ep060-20211021_030936")  #.result_name_v1_to_v2()
### 1b. ch and epoch_6l
mask_h_bg_ch128_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch128_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_1", describe_end="mask_h_bg_ch128_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_1-flow_unet-mask_h_bg_ch128_sig_tv_s20_bce_6l_ep200-20211021_140727")  #.result_name_v1_to_v2()
mask_h_bg_ch064_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_2", describe_end="mask_h_bg_ch064_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_2-flow_unet-mask_h_bg_ch064_sig_tv_s20_bce_6l_ep200-20211021_123236")  #.result_name_v1_to_v2()
mask_h_bg_ch032_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch032_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_3", describe_end="mask_h_bg_ch032_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_3-flow_unet-mask_h_bg_ch032_sig_tv_s20_bce_6l_ep200-20211021_104809")  #.result_name_v1_to_v2()
mask_h_bg_ch016_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch016_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_4", describe_end="mask_h_bg_ch016_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_4-flow_unet-mask_h_bg_ch016_sig_tv_s20_bce_6l_ep200-20211021_102348")  #.result_name_v1_to_v2()
mask_h_bg_ch008_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch008_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_5", describe_end="mask_h_bg_ch008_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_5-flow_unet-mask_h_bg_ch008_sig_tv_s20_bce_6l_ep200-20211021_084915")  #.result_name_v1_to_v2()
mask_h_bg_ch004_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch004_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_6", describe_end="mask_h_bg_ch004_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_6-flow_unet-mask_h_bg_ch004_sig_tv_s20_bce_6l_ep200-20211021_071336")  #.result_name_v1_to_v2()
mask_h_bg_ch002_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch002_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_7", describe_end="mask_h_bg_ch002_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_7-flow_unet-mask_h_bg_ch002_sig_tv_s20_bce_6l_ep200-20211021_053827")  #.result_name_v1_to_v2()
mask_h_bg_ch001_sig_L6_ep200 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch001_sig_L6, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1b_8", describe_end="mask_h_bg_ch001_sig_tv_s08_bce_6l_ep200") .set_train_args(epochs=200, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1b_8-flow_unet-mask_h_bg_ch001_sig_tv_s20_bce_6l_ep200-20211021_050450")  #.result_name_v1_to_v2()

### 3. no-concat
mask_h_bg_ch032_L6_2to2noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to2noC_sig, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_1", describe_end="mask_h_bg_ch032_6l_2to2noC_sig_tv_s08_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_3_1-flow_unet-mask_h_bg_ch032_6l_2to2noC_sig_tv_s08_bce_ep060-20211021_000021")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L6_2to3noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to3noC_sig, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_2", describe_end="mask_h_bg_ch032_6l_2to3noC_sig_tv_s08_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_3_2-flow_unet-mask_h_bg_ch032_6l_2to3noC_sig_tv_s08_bce_ep060-20211021_003137")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L6_2to4noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to4noC_sig, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_3", describe_end="mask_h_bg_ch032_6l_2to4noC_sig_tv_s08_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_3_3-flow_unet-mask_h_bg_ch032_6l_2to4noC_sig_tv_s08_bce_ep060-20211021_010244")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L6_2to5noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to5noC_sig, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_4", describe_end="mask_h_bg_ch032_6l_2to5noC_sig_tv_s08_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_3_4-flow_unet-mask_h_bg_ch032_6l_2to5noC_sig_tv_s08_bce_ep060-20211021_013349")  #.result_name_v1_to_v2()
mask_h_bg_ch032_L6_2to6noC_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_IN_L6_ch32_2to6noC_sig, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_3_5", describe_end="mask_h_bg_ch032_6l_2to6noC_sig_tv_s08_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_3_5-flow_unet-mask_h_bg_ch032_6l_2to6noC_sig_tv_s08_bce_ep060-20211021_020456")  #.result_name_v1_to_v2()

### 4. skip use add
mask_h_bg_ch032_L6_skipAdd_sig_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_L6_skip_use_add_sig, G_tv_s08_bce_s001_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_4_5", describe_end="mask_h_bg_ch032_6l_skipAdd_sig_tv_s08_bce_ep060") .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_4_5-flow_unet-mask_h_bg_ch032_6l_skipAdd_sig_tv_s08_bce_ep060-20211021_023546")  #.result_name_v1_to_v2()

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
