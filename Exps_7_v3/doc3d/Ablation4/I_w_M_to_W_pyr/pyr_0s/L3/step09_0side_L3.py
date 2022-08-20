#############################################################################################################################################################################################################
from step08_b_use_G_generate_I_w_M_to_Wx_Wy_Wz_combine import I_w_M_to_W
from step08_b_use_G_generate_0_util import Tight_crop, Color_jit
from step09_c_train_step import Train_step_I_w_M_to_W
from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

color_jit = Color_jit(do_ratio=0.6)
use_what_gen_op     =            I_w_M_to_W(  separate_out=False, focus=True, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale=  0) )
use_what_train_step = Train_step_I_w_M_to_W(  separate_out=False, focus=True, tight_crop=Tight_crop(pad_size=20, resize=(255, 255), jit_scale= 15), color_jit=color_jit )

import time
start_time = time.time()
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### Block1
### Block1
#########################################################################################
pyramid_0side = [0, 0, 0, 0, 0, 0, 0]
#########################################################################################
ch032_pyramid_0side = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet3(out_conv_block=True, concat_before_down=True, kernel_size=3, padding="valid", hid_ch= 32, depth_level=3, out_ch=3, unet_acti="sigmoid", conv_block_num=pyramid_0side, ch_upper_bound= 2 ** 14).set_gen_op( use_what_gen_op ).set_train_step( use_what_train_step )
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
