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
from step08_c_use_G_generate_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus_combine import I_w_M_to_W_to_C
from step08_b_use_G_generate_0_util import Tight_crop
from step09_c_train_step import Train_step_I_w_M_to_W_to_C

### Basic builder
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

### Model builder， 跟loss沒關係， 只要model 取到自己要的即可
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok    .ch096.wiColorJ.Add2Loss.Sob_k05_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step09_2side_L5 import ch032_pyramid_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_woDiv
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok_div.ch032.wiColorJ.Add2Loss.Sob_k05_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step09_2side_L5 import ch032_pyramid_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_wiDiv
from Exps_7_v3.doc3d.I_w_M_to_Wxyz_focus_Z_ok.pyr_Tcrop255_pad20_jit15.Mae_s001.pyr_2s.L5.step09_2side_L5                                       import ch032_pyramid_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_3UNet

from Exps_7_v3.doc3d.W_w_Mgt_to_Cx_Cy_focus_Z_ok.Mae_s001.pyr_Tcrop255_pad20_jit15.pyr_2s.L5.step09_2side_L5 import ch032_pyramid_1side_6__2side_6 as W_w_M_to_Cxy_Tcrop255_p20_2s_L5

use_gen_op_p20     =            I_w_M_to_W_to_C(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0) )
use_train_step_p20 = Train_step_I_w_M_to_W_to_C(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15) )

use_gen_op_p60     =            I_w_M_to_W_to_C(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=60, resize=(255, 255), jit_scale=  0) )
use_train_step_p60 = Train_step_I_w_M_to_W_to_C(  separate_out=True, focus=True, tight_crop=Tight_crop(pad_size=60, resize=(255, 255), jit_scale= 15) )


import time
start_time = time.time()
''' I_to_Wxyz 變動， W_to_Cxy固定用 W_w_Mgt_to_Cxy_focus 的 L5_ch064 '''
###############################################################################################################################################################################################
###############################################################################################################################################################################################
ch032_pyramid_1side_6__2side_6_p20_L5_woDiv = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv, W_to_Cx_Cy=W_w_M_to_Cxy_Tcrop255_p20_2s_L5).set_multi_model_separate_focus(I_to_W_separ=False, I_to_W_focus=True, W_to_C_separ=True, W_to_C_focus=True).set_gen_op( use_gen_op_p20 ).set_train_step( use_train_step_p20 )
ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv, W_to_Cx_Cy=W_w_M_to_Cxy_Tcrop255_p20_2s_L5).set_multi_model_separate_focus(I_to_W_separ=True , I_to_W_focus=True, W_to_C_separ=True, W_to_C_focus=True).set_gen_op( use_gen_op_p20 ).set_train_step( use_train_step_p20 )
ch032_pyramid_1side_6__2side_6_p20_L5_3UNet = KModel_builder().set_model_name(MODEL_NAME.multi_flow_unet).set_multi_model_builders(op_type="I_to_Wxyz_to_Cxy_general", I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet, W_to_Cx_Cy=W_w_M_to_Cxy_Tcrop255_p20_2s_L5).set_multi_model_separate_focus(I_to_W_separ=True , I_to_W_focus=True, W_to_C_separ=True, W_to_C_focus=True).set_gen_op( use_gen_op_p20 ).set_train_step( use_train_step_p20 )
#########################################################################################
###############################################################################################################################################################################################
if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 511, 511, 1), dtype=np.float32)
    use_model = ch032_pyramid_1side_6__2side_6_p20_L5_woDiv
    use_model = use_model.build()
    result = use_model.generator(data, Mask=data)
    print(result[0].shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
    print(use_model.model_describe)
