#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
from tkinter import S
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
from step08_b_use_G_generate_W_w_M_to_Cx_Cy_combine import W_w_M_to_Cx_Cy
from step08_b_use_G_generate_0_util import Tight_crop
from step09_c_train_step import Train_step_W_w_M_to_Cx_Cy
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import Exps_7_v3.Basic_Pyramid_1ch_model_for_import.pyr_2s.L7.step09_2side_L7 as pyr_1ch_model
use_gen_op     =            W_w_M_to_Cx_Cy(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=20, resize=(256, 256), jit_scale=  0) )
use_train_step = Train_step_W_w_M_to_Cx_Cy(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=20, resize=(256, 256), jit_scale= 15) )

import time
start_time = time.time()
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### Block1
### Block1
#########################################################################################
pyramid_1side_1__2side_0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
pyramid_1side_1__2side_1 = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]

pyramid_1side_2__2side_0 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
pyramid_1side_2__2side_1 = [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
pyramid_1side_2__2side_2 = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]

pyramid_1side_3__2side_0 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
pyramid_1side_3__2side_1 = [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
pyramid_1side_3__2side_2 = [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2]
pyramid_1side_3__2side_3 = [2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2]

pyramid_1side_4__2side_0 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
pyramid_1side_4__2side_1 = [2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2]
pyramid_1side_4__2side_2 = [2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
pyramid_1side_4__2side_3 = [2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2]
pyramid_1side_4__2side_4 = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

pyramid_1side_5__2side_0 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
pyramid_1side_5__2side_1 = [2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2]
pyramid_1side_5__2side_2 = [2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
pyramid_1side_5__2side_3 = [2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2]
pyramid_1side_5__2side_4 = [2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2]
pyramid_1side_5__2side_5 = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]

pyramid_1side_6__2side_0 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
pyramid_1side_6__2side_1 = [2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2]
pyramid_1side_6__2side_2 = [2, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 2, 2]
pyramid_1side_6__2side_3 = [2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2]
pyramid_1side_6__2side_4 = [2, 2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 2]
pyramid_1side_6__2side_5 = [2, 2, 2, 2, 2, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2]
pyramid_1side_6__2side_6 = [2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2]

pyramid_1side_7__2side_0 = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
pyramid_1side_7__2side_1 = [2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2]
pyramid_1side_7__2side_2 = [2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2]
pyramid_1side_7__2side_3 = [2, 2, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 2, 2]
pyramid_1side_7__2side_4 = [2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2]
pyramid_1side_7__2side_5 = [2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2]
pyramid_1side_7__2side_6 = [2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2]
pyramid_1side_7__2side_7 = [2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2]

pyramid_1side_8__2side_0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pyramid_1side_8__2side_1 = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
pyramid_1side_8__2side_2 = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
pyramid_1side_8__2side_3 = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
pyramid_1side_8__2side_4 = [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
pyramid_1side_8__2side_5 = [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
pyramid_1side_8__2side_6 = [2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]
pyramid_1side_8__2side_7 = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2]
pyramid_1side_8__2side_8 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

#########################################################################################
ch032_pyramid_1side_1__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_1__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_1__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_2__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_2__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_3__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_5__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_5__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_5__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_5__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_5__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_5__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_5__2side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_5__2side_3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_5__2side_3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_5__2side_4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_5__2side_4, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_5__2side_4) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_5__2side_5 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_5__2side_5, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_5__2side_5) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_6__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_6__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_6__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_6__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_6__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_6__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_6__2side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_6__2side_3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_6__2side_3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_6__2side_4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_6__2side_4, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_6__2side_4) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_6__2side_5 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_6__2side_5, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_6__2side_5) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_6__2side_6 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_6__2side_6, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_6__2side_6) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_7__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_7__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_7__2side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_7__2side_4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_4, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_4) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_7__2side_5 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_5, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_5) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_7__2side_6 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_6, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_6) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_7__2side_7 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_7__2side_7, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_7__2side_7) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_8__2side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_4, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_4) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_5 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_5, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_5) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_6 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_6, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_6) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_7 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_7, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_7) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_8__2side_8 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_8__2side_8, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_8__2side_8) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
#########################################################################################
###############################################################################################################################################################################################
if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_1side_1__2side_1
    use_model = use_model.build()
    result = use_model.generator(data)
    print(result[0].shape)
    print(result[1].shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
