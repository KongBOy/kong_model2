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

rm_paths = [path for path in sys.path if code_dir in path]
for rm_path in rm_paths: sys.path.remove(rm_path)
rm_moduless = [module for module in sys.modules if "step09" in module]
for rm_module in rm_moduless: del sys.modules[rm_module]
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
### ch064
mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s120_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_1", describe_end="mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3_1-flow_unet-mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060-20211031_200413")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_sobel_k5_s140_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s140_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_2", describe_end="mask_h_bg_ch064_sig_sobel_k5_s140_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3_2-flow_unet-mask_h_bg_ch064_sig_sobel_k5_s140_L6_ep060-20211031_204232")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_sobel_k5_s160_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s160_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_3", describe_end="mask_h_bg_ch064_sig_sobel_k5_s160_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3_3-flow_unet-mask_h_bg_ch064_sig_sobel_k5_s160_L6_ep060-20211031_200458")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_sobel_k5_s180_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s180_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_4", describe_end="mask_h_bg_ch064_sig_sobel_k5_s180_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3_4-flow_unet-mask_h_bg_ch064_sig_sobel_k5_s180_L6_ep060-20211031_204434")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_sobel_k5_s200_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s200_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_5", describe_end="mask_h_bg_ch064_sig_sobel_k5_s200_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3_5-flow_unet-mask_h_bg_ch064_sig_sobel_k5_s200_L6_ep060-20211031_200542")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_sobel_k5_s220_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s220_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_6", describe_end="mask_h_bg_ch064_sig_sobel_k5_s220_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-6_1_3_6-flow_unet-mask_h_bg_ch064_sig_sobel_k5_s220_L6_ep060-20211031_204559")  #.change_result_name_v1_to_v2()
mask_h_bg_ch064_sig_sobel_k5_s240_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s240_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_7", describe_end="mask_h_bg_ch064_sig_sobel_k5_s240_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
mask_h_bg_ch064_sig_sobel_k5_s260_L6_ep060 = Exp_builder().set_basic("train", use_db_obj, mask_unet_ch064_sig_L6, G_sobel_k5_s260_loss_info_builder, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_mid="6_1_3_8", describe_end="mask_h_bg_ch064_sig_sobel_k5_s260_L6_ep060") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
