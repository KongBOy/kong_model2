import tensorflow as tf
from step07_a_unet_component import Conv_block
from tensorflow.keras.layers import Activation, Conv2D, Concatenate

from tensorflow.keras.optimizers import Adam
class Discriminator(tf.keras.models.Model):
    def __init__(self,
                 hid_ch=64, depth_level=7,
                 kernel_size=4, strides=2, padding="same", norm="in",
                 D_first_concat=False, out_acti="sigmoid", **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        '''
        norm: bn/ in
        out_acti: tanh/ relu/ sigmoid
        '''
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.depth_level = depth_level

        self.out_acti = out_acti
        self.D_first_concat = D_first_concat
        if(self.D_first_concat): self.concat = Concatenate()

        self.D_layers = {}

        for i in range(depth_level):
            layer_id = i + 1
            D_layers_name = f"Disc_L{layer_id}"
            if(layer_id == 1): self.D_layers[D_layers_name] = Conv_block(out_ch=hid_ch * 2 ** (layer_id - 1), kernel_size=kernel_size, strides=2, acti="lrelu", padding=padding, norm=False, use_bias=True, name= D_layers_name )
            else:              self.D_layers[D_layers_name] = Conv_block(out_ch=hid_ch * 2 ** (layer_id - 1), kernel_size=kernel_size, strides=2, acti="lrelu", padding=padding, norm=norm,  use_bias=True, name= D_layers_name )

        self.conv_map = Conv2D(1   , kernel_size=kernel_size, strides=1, padding="same", name="1x1_Conv")
        if(self.out_acti == "tanh"):    self.tanh    = Activation(tf.nn.tanh,    name="out_tanh")
        if(self.out_acti == "relu"):    self.tanh    = Activation(tf.nn.relu,    name="out_relu")
        if(self.out_acti == "sigmoid"): self.sigmoid = Activation(tf.nn.sigmoid, name="out_sigmoid")

    def call(self, in_data, gt_img=None, training=None):
        # print("in_data",in_data.shape)
        # print("gt_img",gt_img.shape)
        for i, (name, D_layer) in enumerate(self.D_layers.items()):
            if(i == 0):
                if(self.D_first_concat):
                    concat_img = self.concat([in_data, gt_img])
                    # print("concat_img",concat_img.shape)
                    x = D_layer(concat_img)
                else:
                    x = D_layer(in_data)
            else:
                x = D_layer(x)

        final_x = self.conv_map(x)

        if  (self.out_acti == "tanh"):    return self.tanh(final_x)
        elif(self.out_acti == "sigmoid"): return self.sigmoid(final_x)


if(__name__ == "__main__"):
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import time
    # from kong_util.tf_model_util import Show_model_layer_names, Show_model_weights
    # data = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)
    # start_time = time.time()  # 看資料跑一次花多少時間
    # # test_g = Discriminator(hid_ch=64, depth_level=7, use_bias=False)
    # test_g = Discriminator(hid_ch= 128, depth_level=4, kernel_size=4, out_acti="sigmoid")
    # print("cost time", time.time() - start_time)
    # disc_out = test_g(data)
    # test_g.summary()
    # print("disc_out:", disc_out)
    # plt.imshow(disc_out[0, ..., -1], vmin=0, vmax=1, cmap='RdBu_r')
    # plt.colorbar()
    # plt.show()


    ############################################################################################################################
    ### 嘗試 真的 load tf_data 進來 train 看看
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from kong_util.tf_model_util import Show_model_layer_names, Show_model_weights

    from step06_a_datas_obj import *
    from step06_b_data_pipline import tf_Data_builder
    from step10_a2_loss_info_obj import *
    from step09_c_train_step import *


    ###################################################################
    # from Exps_8_multi_unet.\
    #         W_w_Mgt_to_Cx_Cy_try_mul_M_focus_GAN.\
    #             diff_G_same_D.\
    #                 step09_i3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus_GAN import *
    # model_obj = block1_L2_ch016_sig

    ###################################################################
    # from Exps_8_multi_unet.\
    #         W_w_Mgt_to_Cx_Cy_try_mul_M_focus_GAN.\
    #             same_G_diff_D.\
    #                 step09_i3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus_GAN_G_L5_ch032_D_all import *
    # model_obj = disc_L8_ch016_sig

    ###################################################################
    from Exps_8_multi_unet.\
            W_w_Mgt_to_Cx_Cy_try_mul_M_focus_GAN.\
                same_G_diff_D_no_pad.\
                    step09_i3_multi_unet2_obj_W_w_Mgt_to_Cx_Cy_focus_GAN_G_L5_ch032_D_all_no_pad import *
    '''
    no_pad 的 
        L8 output shape 為 1, 0, 0, 1
        L9 直接完全跑步起來
    '''
    model_obj = disc_L7_ch016_sig

    ###################################################################
    model_obj = model_obj.build()  ### 可替換成 上面 想測試的 model

    ### 2. db_obj 和 tf_data
    db_obj  = type8_blender_wc_flow_try_mul_M.build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1, train_shuffle=False).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    ### 3. loss_info_obj
    loss_info_objs = [G_mae_s001_loss_info_builder.set_loss_target("UNet_Cx").copy().build(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cy").copy().build(), GAN_s001_loss_info_builder.set_loss_target("D_Cxy").copy().build()]  ### z, y, x 順序是看 step07_b_0b_Multi_UNet 來對應的喔

    ### 4. 跑起來試試看
    for n, (train_in, train_in_pre, train_gt, train_gt_pre, _) in enumerate(tqdm(tf_data.train_db_combine)):
        # print("train_in.numpy().min():", train_in.numpy().min())
        # print("train_in.numpy().max():", train_in.numpy().max())
        # print("train_in_pre.numpy().min():", train_in_pre.numpy().min())
        # print("train_in_pre.numpy().max():", train_in_pre.numpy().max())
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=loss_info_objs)
        if(n ==  0):
            model_obj.discriminator.summary()
            Show_model_weights(model_obj.generator)

            Cgt = train_gt_pre[..., 1:3]
            disc_out = model_obj.discriminator(Cgt)
            plt.figure()
            plt.imshow(disc_out[0, ..., -1], vmin=0, vmax=1, cmap='RdBu_r')
            plt.colorbar()
            plt.tight_layout()

        if(n == 10):
            model_obj.discriminator.save_weights("debug_data/try_save/disc_weights")
            iter10 = model_obj.discriminator.layers[0].weights[1]
            print("iter10:", iter10)

            Cgt = train_gt_pre[..., 1:3]
            disc_out = model_obj.discriminator(Cgt)
            plt.figure()
            plt.imshow(disc_out[0, ..., -1], vmin=0, vmax=1, cmap='RdBu_r')
            plt.colorbar()
            plt.tight_layout()

        if(n == 20):
            iter20 = model_obj.discriminator.layers[0].weights[1]
            print("iter20:", iter20)

            Cgt = train_gt_pre[..., 1:3]
            disc_out = model_obj.discriminator(Cgt)
            plt.figure()
            plt.imshow(disc_out[0, ..., -1], vmin=0, vmax=1, cmap='RdBu_r')
            plt.colorbar()
            plt.tight_layout()

            model_obj.discriminator.load_weights("debug_data/try_save/disc_weights")
            iter20_load10 = model_obj.discriminator.layers[0].weights[1]
            print("iter20_load10:", iter20_load10)
            
            Cgt = train_gt_pre[..., 1:3]
            disc_out = model_obj.discriminator(Cgt)
            plt.figure()
            plt.imshow(disc_out[0, ..., -1], vmin=0, vmax=1, cmap='RdBu_r')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
