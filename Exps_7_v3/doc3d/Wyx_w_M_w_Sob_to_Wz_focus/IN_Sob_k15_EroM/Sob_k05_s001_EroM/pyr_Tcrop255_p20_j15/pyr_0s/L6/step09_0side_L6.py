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
Sob_k15_s001_erose_M = Sobel_MAE(sobel_kernel_size=15, sobel_kernel_scale=1, erose_M=True)

use_gen_op     =            Wyx_w_M_to_Wz( focus=True, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 0), sobel=Sob_k5_s001_erose_M, sobel_only=True )
use_train_step = Train_step_Wyx_w_M_to_Wz( focus=True, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=15), sobel=Sob_k5_s001_erose_M, sobel_only=True )

import time
start_time = time.time()
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### Block1
### Block1
#########################################################################################
pyramid_0side = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#########################################################################################
ch032_pyramid_0side = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=pyramid_0side, ch_upper_bound= 2 ** 14).set_gen_op( use_gen_op ).set_train_step( use_train_step )
#########################################################################################
###############################################################################################################################################################################################

if(__name__ == "__main__"):
    import numpy as np

    print("build_model cost time:", time.time() - start_time)
    data = np.zeros(shape=(1, 512, 512, 1))
    use_model = ch032_pyramid_0side
    use_model = use_model.build()
    result = use_model.generator(data)
    print(result.shape)

    from kong_util.tf_model_util import Show_model_weights
    Show_model_weights(use_model.generator)
    use_model.generator.summary()
