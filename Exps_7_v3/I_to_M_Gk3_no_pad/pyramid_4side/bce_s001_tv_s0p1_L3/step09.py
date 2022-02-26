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
from step08_b_use_G_generate_I_to_M import I_Generate_M_see
from step09_c_train_step import train_step_Single_output_I_to_M
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import time
start_time = time.time()
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### Block1
### Block1
#########################################################################################
# "1" 3 6 10 15 21 28 36 45 55
# side1 OK 1
pyramid_1side_1__2side_1__3side_1_4side_1 = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4]

# 1 "3" 6 10 15 21 28 36 45 55
# side2 OK 4
pyramid_1side_2__2side_1__3side_1_4side_1 = [4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4]
pyramid_1side_2__2side_2__3side_1_4side_1 = [4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4]
pyramid_1side_2__2side_2__3side_2_4side_1 = [4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4]

pyramid_1side_2__2side_2__3side_2_4side_2 = [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4]

# 1 3 "6" 10 15 21 28 36 45 55
# side3 OK 10
pyramid_1side_3__2side_1__3side_1_4side_1 = [4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4]
pyramid_1side_3__2side_2__3side_1_4side_1 = [4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4]
pyramid_1side_3__2side_2__3side_2_4side_1 = [4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4]
pyramid_1side_3__2side_3__3side_1_4side_1 = [4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4]
pyramid_1side_3__2side_3__3side_2_4side_1 = [4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4]
pyramid_1side_3__2side_3__3side_3_4side_1 = [4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4]

pyramid_1side_3__2side_2__3side_2_4side_2 = [4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4]
pyramid_1side_3__2side_3__3side_2_4side_2 = [4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4]
pyramid_1side_3__2side_3__3side_3_4side_2 = [4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4]

pyramid_1side_3__2side_3__3side_3_4side_3 = [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4]

# 1 3 6 "10" 15 21 28 36 45 55
# side4 OK 20
pyramid_1side_4__2side_1__3side_1_4side_1 = [4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4]
pyramid_1side_4__2side_2__3side_1_4side_1 = [4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 4]
pyramid_1side_4__2side_2__3side_2_4side_1 = [4, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 4]
pyramid_1side_4__2side_3__3side_1_4side_1 = [4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 4]
pyramid_1side_4__2side_3__3side_2_4side_1 = [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
pyramid_1side_4__2side_3__3side_3_4side_1 = [4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 4]
pyramid_1side_4__2side_4__3side_1_4side_1 = [4, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 4]
pyramid_1side_4__2side_4__3side_2_4side_1 = [4, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 4]
pyramid_1side_4__2side_4__3side_3_4side_1 = [4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 4]
pyramid_1side_4__2side_4__3side_4_4side_1 = [4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 4]

pyramid_1side_4__2side_2__3side_2_4side_2 = [4, 4, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4]
pyramid_1side_4__2side_3__3side_2_4side_2 = [4, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 4]
pyramid_1side_4__2side_3__3side_3_4side_2 = [4, 4, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 4]
pyramid_1side_4__2side_4__3side_2_4side_2 = [4, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4]
pyramid_1side_4__2side_4__3side_3_4side_2 = [4, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 4]
pyramid_1side_4__2side_4__3side_4_4side_2 = [4, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4]

pyramid_1side_4__2side_3__3side_3_4side_3 = [4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 4]
pyramid_1side_4__2side_4__3side_3_4side_3 = [4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4]
pyramid_1side_4__2side_4__3side_4_4side_3 = [4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4]

pyramid_1side_4__2side_4__3side_4_4side_4 = [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4]

#########################################################################################
# "1" 3 6 10 15 21 28 36 45 55
# side1 OK 1
ch032_pyramid_1side_1__2side_1__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_1__2side_1__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

# 1 "3" 6 10 15 21 28 36 45 55
# side2 OK 4
ch032_pyramid_1side_2__2side_1__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_1__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_2__2side_2__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_2__2side_2__3side_2_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_2_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

ch032_pyramid_1side_2__2side_2__3side_2_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_2__2side_2__3side_2_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

# 1 3 "6" 10 15 21 28 36 45 55
# side3 OK 10
ch032_pyramid_1side_3__2side_1__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_1__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_2__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_2__3side_2_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_2_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_3__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_3__3side_2_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_2_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_3__3side_3_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

ch032_pyramid_1side_3__2side_2__3side_2_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_2__3side_2_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_3__3side_2_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_2_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_3__2side_3__3side_3_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

ch032_pyramid_1side_3__2side_3__3side_3_4side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_3__2side_3__3side_3_4side_3, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

# 1 3 6 "10" 15 21 28 36 45 55
# side4 OK 20
ch032_pyramid_1side_4__2side_1__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_1__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_2__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_2__3side_2_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_2_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_3__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_3__3side_2_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_2_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_3__3side_3_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_1_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_1_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_2_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_2_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_3_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_4_4side_1 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_1, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

ch032_pyramid_1side_4__2side_2__3side_2_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_2__3side_2_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_3__3side_2_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_2_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_3__3side_3_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_2_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_2_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_3_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_4_4side_2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_2, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

ch032_pyramid_1side_4__2side_3__3side_3_4side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_3__3side_3_4side_3, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_3_4side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_3_4side_3, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
ch032_pyramid_1side_4__2side_4__3side_4_4side_3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_3, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)

ch032_pyramid_1side_4__2side_4__3side_4_4side_4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_1side_4__2side_4__3side_4_4side_4, ch_upper_bound= 2 ** 14).set_gen_op(I_Generate_M_see).set_train_step(train_step_Single_output_I_to_M)
#########################################################################################
###############################################################################################################################################################################################

if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_1side_1__2side_1__3side_1_4side_1
    use_model = use_model.build()
    result = use_model.generator(data)
    print(result.shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
