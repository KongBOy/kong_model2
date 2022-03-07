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
from step08_c_use_G_generate_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus import I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see
from step09_c_train_step import train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus

### Basic builder
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

### Model builder
from Exps_8_v3.I_w_Mgt_to_Wx_Wy_Wz_focus_Gk3_no_pad.pyramid_3side.bce_s001_tv_s0p1_L4.step09_3side_L4 import *
import step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus  ### 用 block1_L5_ch064_sig

import time
start_time = time.time()
''' I_to_Wxyz 變動， W_to_Cxy固定用 W_w_Mgt_to_Cxy_focus 的 L5_ch064 '''
Use_what_Model = step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch064_sig
###############################################################################################################################################################################################
###############################################################################################################################################################################################
ch032_pyramid_1side_1__2side_1__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_1__2side_1__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

ch032_pyramid_1side_2__2side_1__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_2__2side_1__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_2__2side_2__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_2__2side_2__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_2__2side_2__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_2__2side_2__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

ch032_pyramid_1side_3__2side_1__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_3__2side_1__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_3__2side_2__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_3__2side_2__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_3__2side_2__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_3__2side_2__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_3__2side_3__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_3__2side_3__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_3__2side_3__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_3__2side_3__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_3__2side_3__3side_3 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_3__2side_3__3side_3, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

ch032_pyramid_1side_4__2side_1__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_1__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_2__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_2__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_2__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_2__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_3__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_3__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_3__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_3__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_3__3side_3 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_3__3side_3, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_4__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_4__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_4__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_4__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_4__3side_3 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_4__3side_3, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_4__2side_4__3side_4 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_4__2side_4__3side_4, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

ch032_pyramid_1side_5__2side_1__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_1__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_2__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_2__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_2__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_2__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_3__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_3__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_3__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_3__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_3__3side_3 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_3__3side_3, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_4__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_4__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_4__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_4__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_4__3side_3 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_4__3side_3, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_4__3side_4 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_4__3side_4, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_5__3side_1 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_5__3side_1, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_5__3side_2 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_5__3side_2, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_5__3side_3 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_5__3side_3, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_5__3side_4 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_5__3side_4, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
ch032_pyramid_1side_5__2side_5__3side_5 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=ch032_pyramid_1side_5__2side_5__3side_5, W_to_Cx_Cy=Use_what_Model).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
#########################################################################################
###############################################################################################################################################################################################
if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_1side_1__2side_1__3side_1
    use_model = use_model.build()
    result = use_model.generator(data, Mask=data)
    print(result[0].shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
