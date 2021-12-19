from step09_c_train_step import train_step_pure_G_split_mask_move_I_to_M_w_I_to_C
from step09_d_KModel_builder import KModel_builder, MODEL_NAME

import step09_e5_flow_unet2_obj_I_to_M
import step09_e5_flow_unet2_obj_I_with_Mgt_to_C

import time
start_time = time.time()



I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch128_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch064_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch016_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch008_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch004_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch002_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)

I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch128_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch064_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch032_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch016_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch004_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch002_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
I_to_M_L4_ch001_and_M_w_I_to_C_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_M_w_I_to_C", I_to_M=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, M_w_I_to_C=step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch001_sig_L5).hook_build_and_gen_op(I_to_M_w_I_to_C=True).set_train_step(train_step_pure_G_split_mask_move_I_to_M_w_I_to_C)
