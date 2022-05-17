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
from step08_b_use_G_generate_Wxy_w_M_to_Wz_combine import Wyx_w_M_to_Wz
from step08_b_use_G_generate_0_util import Tight_crop
from step09_c_train_step import Train_step_Wyx_w_M_to_Wz
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

from step10_a1_loss import Sobel_MAE
Sob_k5_s001_erose_M = Sobel_MAE(sobel_kernel_size=5, sobel_kernel_scale=1, erose_M=True)

use_gen_op     =            Wyx_w_M_to_Wz( focus=True, tight_crop=Tight_crop(pad_size=60, resize=(255, 255), jit_scale= 0), sobel=Sob_k5_s001_erose_M, sobel_only=False)
use_train_step = Train_step_Wyx_w_M_to_Wz( focus=True, tight_crop=Tight_crop(pad_size=60, resize=(255, 255), jit_scale=15), sobel=Sob_k5_s001_erose_M, sobel_only=False)

import time
start_time = time.time()
###############################################################################################################################################################################################
##################################
### 6side1
##################################
##### 5side1
# side1, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_1__2side_1__3side_1_4side_1_5s1_6s1 = [6, 0, 0, 0, 0, 0, 6]

# side2, 1 "3" 6 10 15 21 28 36 45 55, 4
pyramid_1side_2__2side_1__3side_1_4side_1_5s1_6s1 = [6, 1, 0, 0, 0, 1, 6]
pyramid_1side_2__2side_2__3side_1_4side_1_5s1_6s1 = [6, 2, 0, 0, 0, 2, 6]
pyramid_1side_2__2side_2__3side_2_4side_1_5s1_6s1 = [6, 3, 0, 0, 0, 3, 6]

pyramid_1side_2__2side_2__3side_2_4side_2_5s1_6s1 = [6, 4, 0, 0, 0, 4, 6]

# side3, 1 3 "6" 10 15 21 28 36 45 55, 10
pyramid_1side_3__2side_1__3side_1_4side_1_5s1_6s1 = [6, 1, 1, 0, 1, 1, 6]
pyramid_1side_3__2side_2__3side_1_4side_1_5s1_6s1 = [6, 2, 1, 0, 1, 2, 6]
pyramid_1side_3__2side_2__3side_2_4side_1_5s1_6s1 = [6, 3, 1, 0, 1, 3, 6]
pyramid_1side_3__2side_3__3side_1_4side_1_5s1_6s1 = [6, 2, 2, 0, 2, 2, 6]
pyramid_1side_3__2side_3__3side_2_4side_1_5s1_6s1 = [6, 3, 2, 0, 2, 3, 6]
pyramid_1side_3__2side_3__3side_3_4side_1_5s1_6s1 = [6, 3, 3, 0, 3, 3, 6]

pyramid_1side_3__2side_2__3side_2_4side_2_5s1_6s1 = [6, 4, 1, 0, 1, 4, 6]
pyramid_1side_3__2side_3__3side_2_4side_2_5s1_6s1 = [6, 4, 2, 0, 2, 4, 6]
pyramid_1side_3__2side_3__3side_3_4side_2_5s1_6s1 = [6, 4, 3, 0, 3, 4, 6]

pyramid_1side_3__2side_3__3side_3_4side_3_5s1_6s1 = [6, 4, 4, 0, 4, 4, 6]

# side4, 1 3 6 "10" 15 21 28 36 45 55, 20
pyramid_1side_4__2side_1__3side_1_4side_1_5s1_6s1 = [6, 1, 1, 1, 1, 1, 6]
pyramid_1side_4__2side_2__3side_1_4side_1_5s1_6s1 = [6, 2, 1, 1, 1, 2, 6]
pyramid_1side_4__2side_2__3side_2_4side_1_5s1_6s1 = [6, 3, 1, 1, 1, 3, 6]
pyramid_1side_4__2side_3__3side_1_4side_1_5s1_6s1 = [6, 2, 2, 1, 2, 2, 6]
pyramid_1side_4__2side_3__3side_2_4side_1_5s1_6s1 = [6, 3, 2, 1, 2, 3, 6]
pyramid_1side_4__2side_3__3side_3_4side_1_5s1_6s1 = [6, 3, 3, 1, 3, 3, 6]
pyramid_1side_4__2side_4__3side_1_4side_1_5s1_6s1 = [6, 2, 2, 2, 2, 2, 6]
pyramid_1side_4__2side_4__3side_2_4side_1_5s1_6s1 = [6, 3, 2, 2, 2, 3, 6]
pyramid_1side_4__2side_4__3side_3_4side_1_5s1_6s1 = [6, 3, 3, 2, 3, 3, 6]
pyramid_1side_4__2side_4__3side_4_4side_1_5s1_6s1 = [6, 3, 3, 3, 3, 3, 6]

pyramid_1side_4__2side_2__3side_2_4side_2_5s1_6s1 = [6, 4, 1, 1, 1, 4, 6]
pyramid_1side_4__2side_3__3side_2_4side_2_5s1_6s1 = [6, 4, 2, 1, 2, 4, 6]
pyramid_1side_4__2side_3__3side_3_4side_2_5s1_6s1 = [6, 4, 3, 1, 3, 4, 6]
pyramid_1side_4__2side_4__3side_2_4side_2_5s1_6s1 = [6, 4, 2, 2, 2, 4, 6]
pyramid_1side_4__2side_4__3side_3_4side_2_5s1_6s1 = [6, 4, 3, 2, 3, 4, 6]
pyramid_1side_4__2side_4__3side_4_4side_2_5s1_6s1 = [6, 4, 3, 3, 3, 4, 6]

pyramid_1side_4__2side_3__3side_3_4side_3_5s1_6s1 = [6, 4, 4, 1, 4, 4, 6]
pyramid_1side_4__2side_4__3side_3_4side_3_5s1_6s1 = [6, 4, 4, 2, 4, 4, 6]
pyramid_1side_4__2side_4__3side_4_4side_3_5s1_6s1 = [6, 4, 4, 3, 4, 4, 6]

pyramid_1side_4__2side_4__3side_4_4side_4_5s1_6s1 = [6, 4, 4, 4, 4, 4, 6]

##### 5side2
# side2, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_2__2side_2__3side_2_4side_2_5s2_6s1 = [6, 5, 0, 0, 0, 5, 6]

# side3, 1 "3" 6 10 15 21 28 36 45 55, 4
pyramid_1side_3__2side_2__3side_2_4side_2_5s2_6s1 = [6, 5, 1, 0, 1, 5, 6]
pyramid_1side_3__2side_3__3side_2_4side_2_5s2_6s1 = [6, 5, 2, 0, 2, 5, 6]
pyramid_1side_3__2side_3__3side_3_4side_2_5s2_6s1 = [6, 5, 3, 0, 3, 5, 6]

pyramid_1side_3__2side_3__3side_3_4side_3_5s2_6s1 = [6, 5, 4, 0, 4, 5, 6]

# side4, 1 3 "6" 10 15 21 28 36 45 55, 10
pyramid_1side_4__2side_2__3side_2_4side_2_5s2_6s1 = [6, 5, 1, 1, 1, 5, 6]
pyramid_1side_4__2side_3__3side_2_4side_2_5s2_6s1 = [6, 5, 2, 1, 2, 5, 6]
pyramid_1side_4__2side_3__3side_3_4side_2_5s2_6s1 = [6, 5, 3, 1, 3, 5, 6]
pyramid_1side_4__2side_4__3side_2_4side_2_5s2_6s1 = [6, 5, 2, 2, 2, 5, 6]
pyramid_1side_4__2side_4__3side_3_4side_2_5s2_6s1 = [6, 5, 3, 2, 3, 5, 6]
pyramid_1side_4__2side_4__3side_4_4side_2_5s2_6s1 = [6, 5, 3, 3, 3, 5, 6]

pyramid_1side_4__2side_3__3side_3_4side_3_5s2_6s1 = [6, 5, 4, 1, 4, 5, 6]
pyramid_1side_4__2side_4__3side_3_4side_3_5s2_6s1 = [6, 5, 4, 2, 4, 5, 6]
pyramid_1side_4__2side_4__3side_4_4side_3_5s2_6s1 = [6, 5, 4, 3, 4, 5, 6]

pyramid_1side_4__2side_4__3side_4_4side_4_5s2_6s1 = [6, 5, 4, 4, 4, 5, 6]

##### 5side3
# side3, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s1 = [6, 5, 5, 0, 5, 5, 6]

# side4, 1 "3" 6 10 15 21 28 36 45 55, 4
pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s1 = [6, 5, 5, 1, 5, 5, 6]
pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s1 = [6, 5, 5, 2, 5, 5, 6]
pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s1 = [6, 5, 5, 3, 5, 5, 6]

pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s1 = [6, 5, 5, 4, 5, 5, 6]

##### 5side4
# side4, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s1 = [6, 5, 5, 5, 5, 5, 6]

##################################
### 6side2
##################################
##### 5side2
# side2, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_2__2side_2__3side_2_4side_2_5s2_6s2 = [6, 6, 0, 0, 0, 6, 6]

# side3, 1 "3" 6 10 15 21 28 36 45 55, 4
pyramid_1side_3__2side_2__3side_2_4side_2_5s2_6s2 = [6, 6, 1, 0, 1, 6, 6]
pyramid_1side_3__2side_3__3side_2_4side_2_5s2_6s2 = [6, 6, 2, 0, 2, 6, 6]
pyramid_1side_3__2side_3__3side_3_4side_2_5s2_6s2 = [6, 6, 3, 0, 3, 6, 6]

pyramid_1side_3__2side_3__3side_3_4side_3_5s2_6s2 = [6, 6, 4, 0, 4, 6, 6]

# side4, 1 3 "6" 10 15 21 28 36 45 55, 10
pyramid_1side_4__2side_2__3side_2_4side_2_5s2_6s2 = [6, 6, 1, 1, 1, 6, 6]
pyramid_1side_4__2side_3__3side_2_4side_2_5s2_6s2 = [6, 6, 2, 1, 2, 6, 6]
pyramid_1side_4__2side_3__3side_3_4side_2_5s2_6s2 = [6, 6, 3, 1, 3, 6, 6]
pyramid_1side_4__2side_4__3side_2_4side_2_5s2_6s2 = [6, 6, 2, 2, 2, 6, 6]
pyramid_1side_4__2side_4__3side_3_4side_2_5s2_6s2 = [6, 6, 3, 2, 3, 6, 6]
pyramid_1side_4__2side_4__3side_4_4side_2_5s2_6s2 = [6, 6, 3, 3, 3, 6, 6]

pyramid_1side_4__2side_3__3side_3_4side_3_5s2_6s2 = [6, 6, 4, 1, 4, 6, 6]
pyramid_1side_4__2side_4__3side_3_4side_3_5s2_6s2 = [6, 6, 4, 2, 4, 6, 6]
pyramid_1side_4__2side_4__3side_4_4side_3_5s2_6s2 = [6, 6, 4, 3, 4, 6, 6]

pyramid_1side_4__2side_4__3side_4_4side_4_5s2_6s2 = [6, 6, 4, 4, 4, 6, 6]

##### 5side3
# side3, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s2 = [6, 6, 5, 0, 5, 6, 6]

# side4, 1 "3" 6 10 15 21 28 36 45 55, 4
pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s2 = [6, 6, 5, 1, 5, 6, 6]
pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s2 = [6, 6, 5, 2, 5, 6, 6]
pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s2 = [6, 6, 5, 3, 5, 6, 6]

pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s2 = [6, 6, 5, 4, 5, 6, 6]

##### 5side4
# side4, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s2 = [6, 6, 5, 5, 5, 6, 6]

##################################
### 6side3
##################################
##### 5side3
# side3, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s3 = [6, 6, 6, 0, 6, 6, 6]

# side4, 1 "3" 6 10 15 21 28 36 45 55, 4
pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s3 = [6, 6, 6, 1, 6, 6, 6]
pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s3 = [6, 6, 6, 2, 6, 6, 6]
pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s3 = [6, 6, 6, 3, 6, 6, 6]

pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s3 = [6, 6, 6, 4, 6, 6, 6]

##### 5side4
# side4, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s3 = [6, 6, 6, 5, 6, 6, 6]

##################################
### 6side4
##################################
##### 5side4
# side4, "1" 3 6 10 15 21 28 36 45 55, 1
pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s4 = [6, 6, 6, 6, 6, 6, 6]

###############################################################################################################################################################################################
###############################################################################################################################################################################################
###############################################################################################################################################################################################
###################
############# 1s1
######### 2s1
##### 3s1
### 4s1
ch032_pyramid_1side_1__2side_1__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_1__2side_1__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )

###################
############# 1s2
######### 2s1
##### 3s1
### 4s1
ch032_pyramid_1side_2__2side_1__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_1__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )

######### 2s1
##### 3s1
### 4s1
ch032_pyramid_1side_2__2side_2__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )

##### 3s2
### 4s1
ch032_pyramid_1side_2__2side_2__3side_2_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_2_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_2_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_2_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_2__2side_2__3side_2_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_2_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )

###################
############# 1s3
######### 2s1
##### 3s1
### 4s1
ch032_pyramid_1side_3__2side_1__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_1__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
######### 2s2
##### 3s1
### 4s1
ch032_pyramid_1side_3__2side_2__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s2
### 4s1
ch032_pyramid_1side_3__2side_2__3side_2_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_2_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_2_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_2_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_2__3side_2_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_2_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
######### 2s3
##### 3s1
### 4s1
ch032_pyramid_1side_3__2side_3__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s2
### 4s1
ch032_pyramid_1side_3__2side_3__3side_2_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_2_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_2_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_2_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_2_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_2_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s3
### 4s1
ch032_pyramid_1side_3__2side_3__3side_3_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s3
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3_5s3_6s3, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )

###################
############# 1s4
######### 2s1
##### 3s1
### 4s1
ch032_pyramid_1side_4__2side_1__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_1__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
######### 2s2
##### 3s1
### 4s1
ch032_pyramid_1side_4__2side_2__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s2
### 4s1
ch032_pyramid_1side_4__2side_2__3side_2_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_2_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_2_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_2_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_2__3side_2_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_2_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
######### 2s3
##### 3s1
### 4s1
ch032_pyramid_1side_4__2side_3__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s2
### 4s1
ch032_pyramid_1side_4__2side_3__3side_2_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_2_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_2_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_2_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_2_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_2_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s3
### 4s1
ch032_pyramid_1side_4__2side_3__3side_3_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s3
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3_5s3_6s3, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
######### 2s4
##### 3s1
### 4s1
ch032_pyramid_1side_4__2side_4__3side_1_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_1_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s2
### 4s1
ch032_pyramid_1side_4__2side_4__3side_2_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_2_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_2_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_2_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_2_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_2_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s3
### 4s1
ch032_pyramid_1side_4__2side_4__3side_3_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s3
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3_5s3_6s3, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
##### 3s4
### 4s1
ch032_pyramid_1side_4__2side_4__3side_4_4side_1_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_1_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s2
ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_2_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_2_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_2_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_2_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s3
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3_5s3_6s3, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
### 4s4
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s1_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s1_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s2_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s2_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s2_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s2_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s3_6s3, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s1, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s2, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s3, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
ch032_pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4_5s4_6s4, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
###############################################################################################################################################################################################
###############################################################################################################################################################################################
if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_1side_1__2side_1__3side_1_4side_1_5s1_6s1
    use_model = use_model.build()
    result = use_model.generator(data)
    print(result.shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
