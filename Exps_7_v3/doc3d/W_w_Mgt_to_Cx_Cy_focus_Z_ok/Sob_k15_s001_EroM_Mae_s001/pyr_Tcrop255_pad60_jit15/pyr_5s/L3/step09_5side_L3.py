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

import Exps_7_v3.Basic_Pyramid_1ch_model_for_import.pyr_5s.L3.step09_5side_L3 as pyr_1ch_model
use_gen_op     =            W_w_M_to_Cx_Cy(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=60, resize=(255, 255), jit_scale=  0) )
use_train_step = Train_step_W_w_M_to_Cx_Cy(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=60, resize=(255, 255), jit_scale= 15) )

import time
start_time = time.time()
###############################################################################################################################################################################################
##################################
### 5side1
##################################
# "1" 3 6 10 15 21 28 36 45 55
# side1 OK 1
pyramid_1side_1__2side_1__3side_1_4side_1_5s1 = [5, 0, 0, 0, 0, 0, 5]

# 1 "3" 6 10 15 21 28 36 45 55
# side2 OK 4
pyramid_1side_2__2side_1__3side_1_4side_1_5s1 = [5, 1, 0, 0, 0, 1, 5]
pyramid_1side_2__2side_2__3side_1_4side_1_5s1 = [5, 2, 0, 0, 0, 2, 5]
pyramid_1side_2__2side_2__3side_2_4side_1_5s1 = [5, 3, 0, 0, 0, 3, 5]

pyramid_1side_2__2side_2__3side_2_4side_2_5s1 = [5, 4, 0, 0, 0, 4, 5]

# 1 3 "6" 10 15 21 28 36 45 55
# side3 OK 10
pyramid_1side_3__2side_1__3side_1_4side_1_5s1 = [5, 1, 1, 0, 1, 1, 5]
pyramid_1side_3__2side_2__3side_1_4side_1_5s1 = [5, 2, 1, 0, 1, 2, 5]
pyramid_1side_3__2side_2__3side_2_4side_1_5s1 = [5, 3, 1, 0, 1, 3, 5]
pyramid_1side_3__2side_3__3side_1_4side_1_5s1 = [5, 2, 2, 0, 2, 2, 5]
pyramid_1side_3__2side_3__3side_2_4side_1_5s1 = [5, 3, 2, 0, 2, 3, 5]
pyramid_1side_3__2side_3__3side_3_4side_1_5s1 = [5, 3, 3, 0, 3, 3, 5]

pyramid_1side_3__2side_2__3side_2_4side_2_5s1 = [5, 4, 1, 0, 1, 4, 5]
pyramid_1side_3__2side_3__3side_2_4side_2_5s1 = [5, 4, 2, 0, 2, 4, 5]
pyramid_1side_3__2side_3__3side_3_4side_2_5s1 = [5, 4, 3, 0, 3, 4, 5]

pyramid_1side_3__2side_3__3side_3_4side_3_5s1 = [5, 4, 4, 0, 4, 4, 5]

# 1 3 6 "10" 15 21 28 36 45 55
# side4 OK 20
pyramid_1side_4__2side_1__3side_1_4side_1_5s1 = [5, 1, 1, 1, 1, 1, 5]
pyramid_1side_4__2side_2__3side_1_4side_1_5s1 = [5, 2, 1, 1, 1, 2, 5]
pyramid_1side_4__2side_2__3side_2_4side_1_5s1 = [5, 3, 1, 1, 1, 3, 5]
pyramid_1side_4__2side_3__3side_1_4side_1_5s1 = [5, 2, 2, 1, 2, 2, 5]
pyramid_1side_4__2side_3__3side_2_4side_1_5s1 = [5, 3, 2, 1, 2, 3, 5]
pyramid_1side_4__2side_3__3side_3_4side_1_5s1 = [5, 3, 3, 1, 3, 3, 5]
pyramid_1side_4__2side_4__3side_1_4side_1_5s1 = [5, 2, 2, 2, 2, 2, 5]
pyramid_1side_4__2side_4__3side_2_4side_1_5s1 = [5, 3, 2, 2, 2, 3, 5]
pyramid_1side_4__2side_4__3side_3_4side_1_5s1 = [5, 3, 3, 2, 3, 3, 5]
pyramid_1side_4__2side_4__3side_4_4side_1_5s1 = [5, 3, 3, 3, 3, 3, 5]

pyramid_1side_4__2side_2__3side_2_4side_2_5s1 = [5, 4, 1, 1, 1, 4, 5]
pyramid_1side_4__2side_3__3side_2_4side_2_5s1 = [5, 4, 2, 1, 2, 4, 5]
pyramid_1side_4__2side_3__3side_3_4side_2_5s1 = [5, 4, 3, 1, 3, 4, 5]
pyramid_1side_4__2side_4__3side_2_4side_2_5s1 = [5, 4, 2, 2, 2, 4, 5]
pyramid_1side_4__2side_4__3side_3_4side_2_5s1 = [5, 4, 3, 2, 3, 4, 5]
pyramid_1side_4__2side_4__3side_4_4side_2_5s1 = [5, 4, 3, 3, 3, 4, 5]

pyramid_1side_4__2side_3__3side_3_4side_3_5s1 = [5, 4, 4, 1, 4, 4, 5]
pyramid_1side_4__2side_4__3side_3_4side_3_5s1 = [5, 4, 4, 2, 4, 4, 5]
pyramid_1side_4__2side_4__3side_4_4side_3_5s1 = [5, 4, 4, 3, 4, 4, 5]

pyramid_1side_4__2side_4__3side_4_4side_4_5s1 = [5, 4, 4, 4, 4, 4, 5]

##################################
### 5side2
##################################
# "1" 3 6 10 15 21 28 36 45 55
# side3 OK 1
pyramid_1side_2__2side_2__3side_2_4side_2_5s2 = [5, 5, 0, 0, 0, 5, 5]

# 1 "3" 6 10 15 21 28 36 45 55
# side3 OK 4
pyramid_1side_3__2side_2__3side_2_4side_2_5s2 = [5, 5, 1, 0, 1, 5, 5]
pyramid_1side_3__2side_3__3side_2_4side_2_5s2 = [5, 5, 2, 0, 2, 5, 5]
pyramid_1side_3__2side_3__3side_3_4side_2_5s2 = [5, 5, 3, 0, 3, 5, 5]

pyramid_1side_3__2side_3__3side_3_4side_3_5s2 = [5, 5, 4, 0, 4, 5, 5]

# 1 3 "6" 10 15 21 28 36 45 55
# side3 OK 10
pyramid_1side_4__2side_2__3side_2_4side_2_5s2 = [5, 5, 1, 1, 1, 5, 5]
pyramid_1side_4__2side_3__3side_2_4side_2_5s2 = [5, 5, 2, 1, 2, 5, 5]
pyramid_1side_4__2side_3__3side_3_4side_2_5s2 = [5, 5, 3, 1, 3, 5, 5]
pyramid_1side_4__2side_4__3side_2_4side_2_5s2 = [5, 5, 2, 2, 2, 5, 5]
pyramid_1side_4__2side_4__3side_3_4side_2_5s2 = [5, 5, 3, 2, 3, 5, 5]
pyramid_1side_4__2side_4__3side_4_4side_2_5s2 = [5, 5, 3, 3, 3, 5, 5]

pyramid_1side_4__2side_3__3side_3_4side_3_5s2 = [5, 5, 4, 1, 4, 5, 5]
pyramid_1side_4__2side_4__3side_3_4side_3_5s2 = [5, 5, 4, 2, 4, 5, 5]
pyramid_1side_4__2side_4__3side_4_4side_3_5s2 = [5, 5, 4, 3, 4, 5, 5]

pyramid_1side_4__2side_4__3side_4_4side_4_5s2 = [5, 5, 4, 4, 4, 5, 5]

##################################
### 5side3
##################################
# "1" 3 6 10 15 21 28 36 45 55
# side3 OK 1
pyramid_1side_3__2side_3__3side_3_4side_3_5s3 = [5, 5, 5, 0, 5, 5, 5]

# 1 "3" 6 10 15 21 28 36 45 55
# side3 OK 4
pyramid_1side_4__2side_3__3side_3_4side_3_5s3 = [5, 5, 5, 1, 5, 5, 5]
pyramid_1side_4__2side_4__3side_3_4side_3_5s3 = [5, 5, 5, 2, 5, 5, 5]
pyramid_1side_4__2side_4__3side_4_4side_3_5s3 = [5, 5, 5, 3, 5, 5, 5]

pyramid_1side_4__2side_4__3side_4_4side_4_5s3 = [5, 5, 5, 4, 5, 5, 5]

##################################
### 5side4
##################################
# "1" 3 6 10 15 21 28 36 45 55
# side3 OK 1
pyramid_1side_4__2side_4__3side_4_4side_4_5s4 = [5, 5, 5, 5, 5, 5, 5]

###############################################################################################################################################################################################
###############################################################################################################################################################################################
###############################################################################################################################################################################################
### 種子程式碼： set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model., I_to_Cy=pyr_1ch_model.) 
##################################
### 1side1
##################################
# "1" 3 6 10 15 21 28 36 45 55
# 2side1 OK 1
ch032_pyramid_1side_1__2side_1__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_1__2side_1__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_1__2side_1__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

##################################
### 1side2
##################################
# "1" 3 6 10 15 21 28 36 45 55
# 2side1 OK 1
ch032_pyramid_1side_2__2side_1__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_1__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_1__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

# 1 "3" 6 10 15 21 28 36 45 55
# 2side2 OK 4
ch032_pyramid_1side_2__2side_2__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_2__2side_2__3side_2_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_2_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_2_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )


##################################
### 1side3
##################################
# "1" 3 6 10 15 21 28 36 45 55
# 2side1 OK 1
ch032_pyramid_1side_3__2side_1__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_1__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_1__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )


# 1 "3" 6 10 15 21 28 36 45 55
# 2side2 OK 4
ch032_pyramid_1side_3__2side_2__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_3__2side_2__3side_2_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_2_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_2_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

# 1 3 "6" 10 15 21 28 36 45 55
# 2side3 OK 10
ch032_pyramid_1side_3__2side_3__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_3__2side_3__3side_2_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_2_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_2_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_3__2side_3__3side_3_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )


##################################
### 1side4
##################################
# "1" 3 6 10 15 21 28 36 45 55
# 2side1 OK 1
ch032_pyramid_1side_4__2side_1__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_1__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_1__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

# 1 "3" 6 10 15 21 28 36 45 55
# 2side2 OK 4
ch032_pyramid_1side_4__2side_2__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_2__3side_2_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_2_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_2_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )


# 1 3 "6" 10 15 21 28 36 45 55
# 2side3 OK 10
ch032_pyramid_1side_4__2side_3__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_3__3side_2_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_2_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_2_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_3__3side_3_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

# 1 3 6 "10" 15 21 28 36 45 55
# 2side4 OK 20
ch032_pyramid_1side_4__2side_4__3side_1_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_1_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_1_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_4__3side_2_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_2_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_2_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_4__3side_3_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )

ch032_pyramid_1side_4__2side_4__3side_4_4side_1_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_1_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_1_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s1, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s1) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s2, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s2) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s3, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s3) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4, I_to_Cy=pyr_1ch_model.ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4) .set_gen_op( use_gen_op ).set_train_step( use_train_step )
###############################################################################################################################################################################################
###############################################################################################################################################################################################
if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_1side_1__2side_1__3side_1_4side_1_5s1
    use_model = use_model.build()
    result = use_model.generator(data)
    print(result[0].shape)
    print(result[1].shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
