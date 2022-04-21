from step08_b_use_G_generate_W_w_M_to_Cx_Cy_combine import W_w_M_to_Cx_Cy

from step09_c_train_step import Train_step_W_w_M_to_Cx_Cy
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import step09_e5_flow_unet2_obj_I_to_M

import time
start_time = time.time()



block1_L2_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L2_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L2_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L3_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L3_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L3_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L4_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L4_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L5_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L6_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L7_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L8_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch004_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch004_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch002_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch002_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch001_sig, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch001_sig).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

##############################################################################################################################################################################################################################################################################################################################################################
block1_L4_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L4_ch128_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L5_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch128_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L5_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L5_ch064_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )


block1_L6_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch128_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch064_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L6_ch032_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L6_ch032_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L7_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch128_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch064_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch032_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch032_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L7_ch016_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L7_ch016_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )

block1_L8_ch128_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch128_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch064_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch064_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch032_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch032_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch016_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch016_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )
block1_L8_ch008_sig_limit = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_or_W_to_Cx_Cy", I_to_Cx=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig_limit, I_to_Cy=step09_e5_flow_unet2_obj_I_to_M.block1_L8_ch008_sig_limit).set_gen_op( W_w_M_to_Cx_Cy(separate_out=True, focus=True) ).set_train_step( Train_step_W_w_M_to_Cx_Cy(separate_out=True, focus=True) )