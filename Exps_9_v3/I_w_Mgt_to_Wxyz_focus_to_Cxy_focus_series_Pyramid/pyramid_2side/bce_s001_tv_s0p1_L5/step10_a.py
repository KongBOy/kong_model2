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
### Basic builder
from step06_a_datas_obj import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder

### Model_builder
from step09_2side_L5 import *

### Exp_builder
import Exps_8_v3.I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad.pyramid_2side.bce_s001_tv_s0p1_L5.step10_a as I_to_Wxyz_exp_builder
import Exps_8_multi_unet.W_w_Mgt_to_Cx_Cy_try_mul_M_focus.a_normal.step10_a                      as W_to_Cxy_exp_builder
''' I_to_Wxyz 變動， W_to_Cxy固定用 W_w_Mgt_to_Cxy_focus 的 L5_ch064 '''
Reload_from_which_Exp = W_to_Cxy_exp_builder.L5_ch064
#############################################################################################################################################################################################################

###############################################################################################################################
use_db_obj = type8_blender_dis_wc_flow_try_mul_M
use_loss_obj = [G_mae_s001_loss_info_builder.set_loss_target("UNet_z").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_y").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_x").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cx").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cy").copy()]  ### z, y, x 順序是看 step07_b_0b_Multi_UNet 來對應的喔

#############################################################
### 為了resul_analyze畫空白的圖，建一個empty的 Exp_builder
empty = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_1__2side_0, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_1__2side_0.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="為了resul_analyze畫空白的圖，建一個empty的 Exp_builder")
#############################################################
ch032_1side_1__2side_1 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_1__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_1__2side_1.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_1__2side_1, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s1__2s1")

ch032_1side_2__2side_1 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_2__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_2__2side_1.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_2__2side_1, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s2__2s1")
ch032_1side_2__2side_2 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_2__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_2__2side_2.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_2__2side_2, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s2__2s2")

ch032_1side_3__2side_1 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_3__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_3__2side_1.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_3__2side_1, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s3__2s1")
ch032_1side_3__2side_2 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_3__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_3__2side_2.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_3__2side_2, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s3__2s2")
ch032_1side_3__2side_3 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_3__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_3__2side_3.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_3__2side_3, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s3__2s3")

ch032_1side_4__2side_1 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_4__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_1.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_4__2side_1, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s4__2s1")
ch032_1side_4__2side_2 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_4__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_2.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_4__2side_2, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s4__2s2")
ch032_1side_4__2side_3 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_4__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_3.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_4__2side_3, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s4__2s3")
ch032_1side_4__2side_4 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_4__2side_4, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_4__2side_4.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_4__2side_4, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s4__2s4")

ch032_1side_5__2side_1 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_5__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_1.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_5__2side_1, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s5__2s1")
ch032_1side_5__2side_2 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_5__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_2.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_5__2side_2, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s5__2s2")
ch032_1side_5__2side_3 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_5__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_3.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_5__2side_3, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s5__2s3")
ch032_1side_5__2side_4 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_5__2side_4, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_4.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_5__2side_4, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s5__2s4")
ch032_1side_5__2side_5 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_5__2side_5, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_5__2side_5.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_5__2side_5, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s5__2s5")

ch032_1side_6__2side_1 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_6__2side_1, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_1.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_6__2side_1, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s6__2s1")
ch032_1side_6__2side_2 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_6__2side_2, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_2.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_6__2side_2, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s6__2s2")
ch032_1side_6__2side_3 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_6__2side_3, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_3.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_6__2side_3, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s6__2s3")
ch032_1side_6__2side_4 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_6__2side_4, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_4.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_6__2side_4, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s6__2s4")
ch032_1side_6__2side_5 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_6__2side_5, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_5.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_6__2side_5, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s6__2s5")
ch032_1side_6__2side_6 = Exp_builder().set_basic("train_run_final_see", use_db_obj, ch032_pyramid_1side_6__2side_6, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_6.kong_model.model_describe) .set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_to_Wxyz_exp_builder.ch032_1side_6__2side_6, W_to_Cx_Cy=Reload_from_which_Exp).set_result_name(result_name="ch032_1s6__2s6")
#############################################################
if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_1side_4__2side_2
    use_model = use_model.build()
    result = use_model.generator(data, Mask=data)
    print(result[0].shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
