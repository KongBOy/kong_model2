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
from step08_b_use_G_generate_W_w_M_to_Cx_Cy_focus import W_w_M_Gen_Cx_Cy_focus_see

from step09_c_train_step import Ttrain_step_w_GAN
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import step09_e5_flow_unet2_obj_I_to_M
import step09_i3_discriminator
import time
start_time = time.time()

disc_L1_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L1_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L1_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L2_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L2_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L2_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L3_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L3_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L3_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L4_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L4_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L4_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L5_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L5_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L5_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L6_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L6_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L6_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L7_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L7_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L7_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L8_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L8_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L8_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

disc_L9_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch128_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch064_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch032_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch016_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch008_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch004_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch002_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))
disc_L9_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op(W_w_M_Gen_Cx_Cy_focus_see).set_discriminator_by_exist_builder(step09_i3_discriminator.disc_L9_ch001_sig).set_train_step(Ttrain_step_w_GAN(op_type="W_w_Mgt_to_Cx_Cy_focus"))

if(__name__ == "__main__"):
    import numpy as np
    data = np.zeros(shape=(1, 512, 512, 1), dtype=np.float32)

    use_model = disc_L2_ch016_sig
    use_model = use_model.build()
    print(use_model.generator(data)[0].shape)
