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

from step08_c_use_G_generate_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus import I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see

from step09_c_train_step import train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import step09_g2_multi_unet2_obj_I_w_Mgt_to_Wx_Wy_Wz_focus
import step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus

import time
start_time = time.time()
''' I_to_Wxyz 固定， 變動 W_to_Cxy '''
Use_what_Model = step09_g2_multi_unet2_obj_I_w_Mgt_to_Wx_Wy_Wz_focus.block1_L7_ch064_sig_limit
###############################################################################################################################
block1_L2_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L2_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L2_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L3_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L3_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L3_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L4_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L4_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L5_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L6_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L7_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L8_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch128_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch064_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch032_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch016_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch008_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch004_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch002_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch001_sig).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
#####################################################################################################
block1_L4_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L4_ch128_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L5_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch128_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L5_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L5_ch064_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L6_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch128_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch064_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L6_ch032_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L6_ch032_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L7_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch128_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch064_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch032_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch032_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L7_ch016_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L7_ch016_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)

block1_L8_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch128_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch064_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch032_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch032_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch016_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch016_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)
block1_L8_ch008_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus", I_to_Wx_Wy_Wz=Use_what_Model, W_to_Cx_Cy=step09_g3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus.block1_L8_ch008_sig_limit,).set_gen_op(I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see).set_train_step(train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus)