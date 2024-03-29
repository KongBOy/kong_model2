#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
code_dir = "\\".join(code_exe_path_element[:-1])
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
sys.path.append(code_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    code_dir:", code_dir)
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
from step09_2side_L4 import *
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

use_db_obj = type8_blender_kong_doc3d_in_W_gt_W_ch_norm_v2
use_loss_obj = [G_sobel_k5_erose_M_loss_info_builder.set_loss_target("UNet_Wz").copy()]
#############################################################
### 為了resul_analyze畫空白的圖，建一個empty的 Exp_builder
empty = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_1__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_1__2side_1.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="為了resul_analyze畫空白的圖，建一個empty的 Exp_builder")
#############################################################
ch032_1side_1__2side_1 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_1__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_1__2side_1.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

ch032_1side_2__2side_1 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_2__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_2__2side_1.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_2__2side_2 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_2__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_2__2side_2.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

ch032_1side_3__2side_1 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_3__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_3__2side_1.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_3__2side_2 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_3__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_3__2side_2.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_3__2side_3 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_3__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_3__2side_3.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

ch032_1side_4__2side_1 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_4__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_1.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_4__2side_2 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_4__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_2.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_4__2side_3 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_4__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_3.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_4__2side_4 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_4__2side_4, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_4.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

ch032_1side_5__2side_1 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_5__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_1.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_5__2side_2 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_5__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_2.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_5__2side_3 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_5__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_3.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_5__2side_4 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_5__2side_4, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_4.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
ch032_1side_5__2side_5 = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_5__2side_5, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_5.kong_model.model_describe) .set_train_args(epochs=  1) .set_train_iter_args(it_see_fq=900, it_save_fq=900 * 2, it_down_step="half", it_down_fq=900).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
#############################################################
if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        ch032_1side_1__2side_0.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
