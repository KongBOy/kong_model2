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
from step08_b_use_G_generate_0_util import Tight_crop, Color_jit
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME
color_jit = Color_jit(do_ratio=0.6)
use_hid_ch = 16

### I_w_M_to_W
from step08_b_use_G_generate_I_w_M_to_Wx_Wy_Wz_combine import I_w_M_to_W
from step09_c_train_step import Train_step_I_w_M_to_W
I_w_M_to_W_normal_use_gen_op           =            I_w_M_to_W(  separate_out=False, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0))
I_w_M_to_W_normal_use_train_step       = Train_step_I_w_M_to_W(  separate_out=False, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit)
I_w_M_to_W_ChBg_use_gen_op             =            I_w_M_to_W(  separate_out=False, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0), remove_in_bg=False)
I_w_M_to_W_ChBg_use_train_step         = Train_step_I_w_M_to_W(  separate_out=False, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit, remove_in_bg=False)
I_w_M_to_W_noFocus_use_gen_op          =            I_w_M_to_W(  separate_out=False, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0) )
I_w_M_to_W_noFocus_use_train_step      = Train_step_I_w_M_to_W(  separate_out=False, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit )
I_w_M_to_W_noFocus_ChBg_use_gen_op     =            I_w_M_to_W(  separate_out=False, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0), remove_in_bg=False )
I_w_M_to_W_noFocus_ChBg_use_train_step = Train_step_I_w_M_to_W(  separate_out=False, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit, remove_in_bg=False )

### W_w_M_to_C
from step08_b_use_G_generate_W_w_M_to_Cx_Cy_combine import W_w_M_to_Cx_Cy
from step09_c_train_step import Train_step_W_w_M_to_Cx_Cy
### 沒有 ChBg 和 ColorJit
W_w_M_to_C_normal_use_gen_op           =            W_w_M_to_Cx_Cy( separate_out=False, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0))
W_w_M_to_C_normal_use_train_step       = Train_step_W_w_M_to_Cx_Cy( separate_out=False, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15))
W_w_M_to_C_noFocus_ChBg_use_gen_op     =            W_w_M_to_Cx_Cy( separate_out=False, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0), remove_in_bg=False )
W_w_M_to_C_noFocus_ChBg_use_train_step = Train_step_W_w_M_to_Cx_Cy( separate_out=False, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), remove_in_bg=False )

import time
start_time = time.time()
###############################################################################################################################################################################################
L5_2blk = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
#########################################################################################
### I_w_M_to_C woDiv
I_w_M_to_W_ch016_L5_normal       = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch=use_hid_ch , depth_level=5, out_ch=3, unet_acti="sigmoid", conv_block_num=L5_2blk, ch_upper_bound= 2 ** 14) .set_gen_op( I_w_M_to_W_normal_use_gen_op       ).set_train_step( I_w_M_to_W_normal_use_train_step )
I_w_M_to_W_ch016_L5_ChBg         = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch=use_hid_ch , depth_level=5, out_ch=3, unet_acti="sigmoid", conv_block_num=L5_2blk, ch_upper_bound= 2 ** 14) .set_gen_op( I_w_M_to_W_ChBg_use_gen_op         ).set_train_step( I_w_M_to_W_ChBg_use_train_step )
I_w_M_to_W_ch016_L5_noFocus      = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch=use_hid_ch , depth_level=5, out_ch=3, unet_acti="sigmoid", conv_block_num=L5_2blk, ch_upper_bound= 2 ** 14) .set_gen_op( I_w_M_to_W_noFocus_use_gen_op      ).set_train_step( I_w_M_to_W_noFocus_use_train_step )
I_w_M_to_W_ch016_L5_noFocus_ChBg = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch=use_hid_ch , depth_level=5, out_ch=3, unet_acti="sigmoid", conv_block_num=L5_2blk, ch_upper_bound= 2 ** 14) .set_gen_op( I_w_M_to_W_noFocus_ChBg_use_gen_op ).set_train_step( I_w_M_to_W_noFocus_ChBg_use_train_step )
### W_w_M_to_C woDiv， 沒有 ChBg 和 ColorJit
W_w_M_to_C_ch016_L5_normal       = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch=use_hid_ch , depth_level=5, out_ch=2, unet_acti="sigmoid", conv_block_num=L5_2blk, ch_upper_bound= 2 ** 14) .set_gen_op( W_w_M_to_C_normal_use_gen_op       ).set_train_step( W_w_M_to_C_normal_use_train_step )
W_w_M_to_C_ch016_L5_noFocus_ChBg = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch=use_hid_ch , depth_level=5, out_ch=2, unet_acti="sigmoid", conv_block_num=L5_2blk, ch_upper_bound= 2 ** 14) .set_gen_op( W_w_M_to_C_noFocus_ChBg_use_gen_op ).set_train_step( W_w_M_to_C_noFocus_ChBg_use_train_step )
#########################################################################################
###############################################################################################################################################################################################
### 合起來也寫一起好了拉
from step08_c_use_G_generate_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus_combine import I_w_M_to_W_to_C
from step09_c_train_step import Train_step_I_w_M_to_W_to_C
gather_normal_use_gen_op_p20           =            I_w_M_to_W_to_C(  separate_out=True, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0)                                          )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_normal_use_train_step_p20       = Train_step_I_w_M_to_W_to_C(  separate_out=True, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit                     )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_ChBg_use_gen_op_p20             =            I_w_M_to_W_to_C(  separate_out=True, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0)                     , remove_in_bg=False )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_ChBg_use_train_step_p20         = Train_step_I_w_M_to_W_to_C(  separate_out=True, focus=True , tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit, remove_in_bg=False )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_noFocus_use_gen_op_p20          =            I_w_M_to_W_to_C(  separate_out=True, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0)                                          )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_noFocus_use_train_step_p20      = Train_step_I_w_M_to_W_to_C(  separate_out=True, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit                     )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_noFocus_ChBg_use_gen_op_p20     =            I_w_M_to_W_to_C(  separate_out=True, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0),                      remove_in_bg=False )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了
gather_noFocus_ChBg_use_train_step_p20 = Train_step_I_w_M_to_W_to_C(  separate_out=True, focus=False, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit, remove_in_bg=False )  ### 我目前的 multi_model 的 I_to_Wxyz_to_Cxy_general 是 全部都回傳 Wz_pre_w_M, Wy_pre_w_M, Wx_pre_w_M, Cx_pre_w_M, Cy_pre_w_M， 所以不管 wi/woDIV， Separate 全設 True 就對了

#########################################################################################
L5_gather_normal__I_w_M_to_W_normal__W_w_M_to_C_normal                   = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_ch016_L5_normal      , W_to_Cx_Cy=W_w_M_to_C_ch016_L5_normal      ).set_multi_model_separate_focus(I_to_W_separ=False , I_to_W_focus=True  , W_to_C_separ=False , W_to_C_focus=True  ).set_gen_op( gather_normal_use_gen_op_p20       ).set_train_step( gather_normal_use_train_step_p20       )
L5_gather_ChBg__I_w_M_to_W_ChBg__W_w_M_to_C_normal                       = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_ch016_L5_ChBg        , W_to_Cx_Cy=W_w_M_to_C_ch016_L5_normal      ).set_multi_model_separate_focus(I_to_W_separ=False , I_to_W_focus=True  , W_to_C_separ=False , W_to_C_focus=True  ).set_gen_op( gather_ChBg_use_gen_op_p20         ).set_train_step( gather_ChBg_use_train_step_p20         )
L5_gather_noFocus__I_w_M_to_W_noFocus__W_w_M_to_C_noFocus_ChBg           = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_ch016_L5_noFocus     , W_to_Cx_Cy=W_w_M_to_C_ch016_L5_noFocus_ChBg).set_multi_model_separate_focus(I_to_W_separ=False , I_to_W_focus=False , W_to_C_separ=False , W_to_C_focus=False ).set_gen_op( gather_noFocus_use_gen_op_p20      ).set_train_step( gather_noFocus_use_train_step_p20      )
L5_gather_noFocus_ChBg__I_w_M_to_W_noFocus_ChBg__W_w_M_to_C_noFocus_ChBg = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_ch016_L5_noFocus_ChBg, W_to_Cx_Cy=W_w_M_to_C_ch016_L5_noFocus_ChBg).set_multi_model_separate_focus(I_to_W_separ=False , I_to_W_focus=False , W_to_C_separ=False , W_to_C_focus=False ).set_gen_op( gather_noFocus_ChBg_use_gen_op_p20 ).set_train_step( gather_noFocus_ChBg_use_train_step_p20 )
###############################################################################################################################################################################################

if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 255, 255, 1))
    use_model = I_w_M_to_W_ch016_L5_noFocus
    use_model = use_model.build()
    result = use_model.generator(data)
    print(result.shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)

    import tensorflow as tf
    import datetime
    code_exe_dir = "\\".join(code_exe_path_element[:-1])
    log_dir = f"{code_exe_dir}/use_Tensorboard_see_Graph/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    img_inputs = tf.keras.Input(shape=(255, 255, 1))
    use_model.generator(img_inputs)
    use_model.generator.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    use_model.generator.fit    (data, data, epochs=1, callbacks=[tboard_callback])
    print(f"tensorboard --logdir={log_dir}")
