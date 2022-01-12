from step09_c_train_step import train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import step09_e5_flow_unet2_obj_I_to_M

import time
start_time = time.time()



I_to_Wx_L2_ch128_and_I_to_Wy_L2_ch128_and_I_to_Wz_L2_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch064_and_I_to_Wy_L2_ch064_and_I_to_Wz_L2_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch032_and_I_to_Wy_L2_ch032_and_I_to_Wz_L2_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch016_and_I_to_Wy_L2_ch016_and_I_to_Wz_L2_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch008_and_I_to_Wy_L2_ch008_and_I_to_Wz_L2_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch004_and_I_to_Wy_L2_ch004_and_I_to_Wz_L2_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch002_and_I_to_Wy_L2_ch002_and_I_to_Wz_L2_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L2_ch001_and_I_to_Wy_L2_ch001_and_I_to_Wz_L2_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L3_ch128_and_I_to_Wy_L3_ch128_and_I_to_Wz_L3_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch064_and_I_to_Wy_L3_ch064_and_I_to_Wz_L3_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch032_and_I_to_Wy_L3_ch032_and_I_to_Wz_L3_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch016_and_I_to_Wy_L3_ch016_and_I_to_Wz_L3_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch008_and_I_to_Wy_L3_ch008_and_I_to_Wz_L3_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch004_and_I_to_Wy_L3_ch004_and_I_to_Wz_L3_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch002_and_I_to_Wy_L3_ch002_and_I_to_Wz_L3_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L3_ch001_and_I_to_Wy_L3_ch001_and_I_to_Wz_L3_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L4_ch128_and_I_to_Wy_L4_ch128_and_I_to_Wz_L4_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch064_and_I_to_Wy_L4_ch064_and_I_to_Wz_L4_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch032_and_I_to_Wy_L4_ch032_and_I_to_Wz_L4_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch016_and_I_to_Wy_L4_ch016_and_I_to_Wz_L4_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch008_and_I_to_Wy_L4_ch008_and_I_to_Wz_L4_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch004_and_I_to_Wy_L4_ch004_and_I_to_Wz_L4_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch002_and_I_to_Wy_L4_ch002_and_I_to_Wz_L4_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L4_ch001_and_I_to_Wy_L4_ch001_and_I_to_Wz_L4_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L5_ch128_and_I_to_Wy_L5_ch128_and_I_to_Wz_L5_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch064_and_I_to_Wy_L5_ch064_and_I_to_Wz_L5_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch032_and_I_to_Wy_L5_ch032_and_I_to_Wz_L5_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch016_and_I_to_Wy_L5_ch016_and_I_to_Wz_L5_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch008_and_I_to_Wy_L5_ch008_and_I_to_Wz_L5_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch004_and_I_to_Wy_L5_ch004_and_I_to_Wz_L5_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch002_and_I_to_Wy_L5_ch002_and_I_to_Wz_L5_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch001_and_I_to_Wy_L5_ch001_and_I_to_Wz_L5_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L6_ch128_and_I_to_Wy_L6_ch128_and_I_to_Wz_L6_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch064_and_I_to_Wy_L6_ch064_and_I_to_Wz_L6_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch032_and_I_to_Wy_L6_ch032_and_I_to_Wz_L6_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch016_and_I_to_Wy_L6_ch016_and_I_to_Wz_L6_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch008_and_I_to_Wy_L6_ch008_and_I_to_Wz_L6_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch004_and_I_to_Wy_L6_ch004_and_I_to_Wz_L6_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch002_and_I_to_Wy_L6_ch002_and_I_to_Wz_L6_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch001_and_I_to_Wy_L6_ch001_and_I_to_Wz_L6_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L7_ch128_and_I_to_Wy_L7_ch128_and_I_to_Wz_L7_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch064_and_I_to_Wy_L7_ch064_and_I_to_Wz_L7_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch032_and_I_to_Wy_L7_ch032_and_I_to_Wz_L7_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch016_and_I_to_Wy_L7_ch016_and_I_to_Wz_L7_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch008_and_I_to_Wy_L7_ch008_and_I_to_Wz_L7_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch004_and_I_to_Wy_L7_ch004_and_I_to_Wz_L7_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch002_and_I_to_Wy_L7_ch002_and_I_to_Wz_L7_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch001_and_I_to_Wy_L7_ch001_and_I_to_Wz_L7_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L8_ch128_and_I_to_Wy_L8_ch128_and_I_to_Wz_L8_ch128 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch064_and_I_to_Wy_L8_ch064_and_I_to_Wz_L8_ch064 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch032_and_I_to_Wy_L8_ch032_and_I_to_Wz_L8_ch032 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch016_and_I_to_Wy_L8_ch016_and_I_to_Wz_L8_ch016 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch008_and_I_to_Wy_L8_ch008_and_I_to_Wz_L8_ch008 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch004_and_I_to_Wy_L8_ch004_and_I_to_Wz_L8_ch004 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch004_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch004_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch004_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch002_and_I_to_Wy_L8_ch002_and_I_to_Wz_L8_ch002 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch002_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch002_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch002_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch001_and_I_to_Wy_L8_ch001_and_I_to_Wz_L8_ch001 = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch001_sig, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch001_sig, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch001_sig).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

##############################################################################################################################################################################################################################################################################################################################################################
I_to_Wx_L4_ch128_lim_and_I_to_Wy_L4_ch128_lim_and_I_to_Wz_L4_ch128_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L5_ch128_lim_and_I_to_Wy_L5_ch128_lim_and_I_to_Wz_L5_ch128_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L5_ch064_lim_and_I_to_Wy_L5_ch064_lim_and_I_to_Wz_L5_ch064_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)


I_to_Wx_L6_ch128_lim_and_I_to_Wy_L6_ch128_lim_and_I_to_Wz_L6_ch128_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch064_lim_and_I_to_Wy_L6_ch064_lim_and_I_to_Wz_L6_ch064_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L6_ch032_lim_and_I_to_Wy_L6_ch032_lim_and_I_to_Wz_L6_ch032_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L7_ch128_lim_and_I_to_Wy_L7_ch128_lim_and_I_to_Wz_L7_ch128_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch064_lim_and_I_to_Wy_L7_ch064_lim_and_I_to_Wz_L7_ch064_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch032_lim_and_I_to_Wy_L7_ch032_lim_and_I_to_Wz_L7_ch032_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L7_ch016_lim_and_I_to_Wy_L7_ch016_lim_and_I_to_Wz_L7_ch016_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)

I_to_Wx_L8_ch128_lim_and_I_to_Wy_L8_ch128_lim_and_I_to_Wz_L8_ch128_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch064_lim_and_I_to_Wy_L8_ch064_lim_and_I_to_Wz_L8_ch064_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch032_lim_and_I_to_Wy_L8_ch032_lim_and_I_to_Wz_L8_ch032_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch016_lim_and_I_to_Wy_L8_ch016_lim_and_I_to_Wz_L8_ch016_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)
I_to_Wx_L8_ch008_lim_and_I_to_Wy_L8_ch008_lim_and_I_to_Wz_L8_ch008_lim = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wx_Wy_Wz", I_to_Wx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig_limit, I_to_Wy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig_limit, I_to_Wz=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig_limit).hook_build_and_gen_op(I_w_Mgt_to_Wx_Wy_Wz=True).set_train_step(train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz)