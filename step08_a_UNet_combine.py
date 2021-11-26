import tensorflow as tf
from step07_unet_component import UNet_down, UNet_up
from tensorflow.keras.layers import Activation
### 參考 DewarpNet 的 train_wc 用的 UNet
### 所有 pytorch BN 裡面有兩個參數的設定不確定～： affine=True, track_running_stats=True，目前思考覺得改道tf2全拿掉也可以
### 目前 總共用7層，所以size縮小 2**7 ，也就是 1/128 這樣子！例如256*256*3丟進去，最中間的feature map長寬2*2*512喔！
class Generator(tf.keras.models.Model):
    def __init__(self, hid_ch=64, depth_level=7, out_ch=3, no_concat_layer=0,
                 kernel_size=4, strides=2, norm="in",
                 d_acti="lrelu", u_acti="relu", unet_acti="tanh",
                 use_bias=True,
                 conv_block_num=0,
                 skip_op=None, skip_merge_op="concat",
                 ch_upper_bound=512,
                 coord_conv=False,
                 #  out_tanh=True,
                 #  skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 **kwargs):
        '''
        d_acti: lrelu/ relu
        u_acti: relu/ lrelu
        unet_acti: tanh/ sigmoid
        skip_op: cse/ sse/ scse
        '''
        if(depth_level < 2):
            print("UNet 不可以小於 2層， 因為如果只有 1層 沒辦法做 skip connection")
            exit()
        super(Generator, self).__init__(**kwargs)
        self.depth_level = depth_level
        self.hid_ch = hid_ch
        self.no_concat_layer = no_concat_layer
        self.unet_out_ch = out_ch
        self.unet_acti = unet_acti
        kwargs = dict(kernel_size=kernel_size, strides=strides, norm=norm, conv_block_num=conv_block_num,
                    #   d_acti=d_acti, u_acti=u_acti,
                      use_bias=use_bias,
                      coord_conv=coord_conv,
                    #   skip_op=skip_op,
                    #   skip_merge_op=skip_merge_op
                      )
        ### 最基本(比如最少層depth_level=2)的一定有 top, bottle
        self.d_top    = UNet_down(at_where="top"   , out_ch=self.Get_Layer_hid_ch(to_L=1              , ch_upper_bound=ch_upper_bound), name="D_0->1_top", **kwargs)  ### Layer 0 -> 1， to_L=1 代表 走進 第1層
        self.u_top    = UNet_up  (at_where="top"   , out_ch=self.Get_Layer_hid_ch(to_L=0              , ch_upper_bound=ch_upper_bound), name="U_1->0_top", **kwargs)  ### Layer 1 -> 0， to_L=0 代表 返回 第0層
        self.d_bottle = UNet_down(at_where="bottle", out_ch=self.Get_Layer_hid_ch(to_L=depth_level    , ch_upper_bound=ch_upper_bound), name=f"D_{depth_level-1}->{depth_level}_bottle", **kwargs)
        self.u_bottle = UNet_up  (at_where="bottle", out_ch=self.Get_Layer_hid_ch(to_L=depth_level - 1, ch_upper_bound=ch_upper_bound), name=f"U_{depth_level}->{depth_level-1}_bottle", **kwargs)  ### 因為是返回上一層， 所以 -1
        # self.d_bottle = UNet_down(at_where="bottle", out_ch=min(hid_ch * 2**(depth_level - 1)    , 512), name=f"D{depth_level} bottle")  ### L0(3), L1(hid_ch*2**0), L2(hid_ch*2**1), ..., L2(hid_ch*2**depth_level - 1)
        # self.u_bottle = UNet_up  (at_where="bottle", out_ch=min(hid_ch * 2**(depth_level - 1 - 1), 512), name=f"U{depth_level} bottle")  ### L0(3), L1(hid_ch*2**0), L2(hid_ch*2**1), ..., L2(hid_ch*2**depth_level - 1)， 因為是返回上一層， out_ch 2的冪次要再 -1

        ### depth_level >=3 以後在 top 和 bottle 之間 加入 middle 層的概念， 連接順序是 D 從 top層 往 bottle層 ， U 從 bottle層 到 top，
        ###     小心 這邊 U的宣告 是 top 到 bottle 和 連接順序相反， 宣告完要記得reverse一下順序喔
        # self.d_middles = []
        # self.u_middles = []
        self.d_middles = {}
        self.u_middles = {}
        if(depth_level >= 3):
            for i in range(depth_level - 2):  ### -2 是 -top 和 -bottle 共兩層
                layer_id = i + 1 + 1  ### +1 是 index轉layer_id， 再+1 是因為前面有top層。 middle 至少 一定從 走入Layer2開始(Down) 或 從Layer2開始返回(Up)
                d_middle_name = f"D_{layer_id-1}->{layer_id}_middle"
                u_middle_name = f"U_{layer_id}->{layer_id-1}_middle"
                self.d_middles[d_middle_name] = UNet_down(at_where="middle", out_ch=self.Get_Layer_hid_ch(to_L=layer_id    , ch_upper_bound=ch_upper_bound), name=d_middle_name, **kwargs )
                self.u_middles[u_middle_name] = UNet_up  (at_where="middle", out_ch=self.Get_Layer_hid_ch(to_L=layer_id - 1, ch_upper_bound=ch_upper_bound), name=u_middle_name, **kwargs )
                # self.d_middles.append( UNet_down(at_where="middle", out_ch=self.Get_Layer_hid_ch(to_L=layer_id    , ch_upper_bound=ch_upper_bound), name=d_middle_name, **kwargs ) )
                # self.u_middles.append( UNet_up  (at_where="middle", out_ch=self.Get_Layer_hid_ch(to_L=layer_id - 1, ch_upper_bound=ch_upper_bound), name=u_middle_name, **kwargs ) )
                # self.d_middles.append( UNet_down(at_where="middle", out_ch=( min(hid_ch * 2**(layer_id - 1    ), 512) ), name=f"D{layer_id} middle" ) )
                # self.u_middles.append( UNet_up  (at_where="middle", out_ch=( min(hid_ch * 2**(layer_id - 1 - 1), 512) ), name=f"U{layer_id} middle" ) )

        ### 注意 up 的部分是 從bottle層 往top連接， 跟宣告順序相反， 要reverse一下順序喔
        # self.u_middles.reverse()
        ############################################################################################################################################################
        if(self.unet_acti == "tanh"):    self.tanh    = Activation(tf.nn.tanh,    name="out_tanh")
        if(self.unet_acti == "sigmoid"): self.sigmoid = Activation(tf.nn.sigmoid, name="out_sigmoid")

    def Get_Layer_hid_ch(self, to_L, ch_upper_bound=512):
        '''
        L_0 (3 或 1),
        L_1 (hid_ch*2**0),
        L_2 (hid_ch*2**1),
        L_3 (hid_ch*2**2),
        ...,
        L_depth_level (hid_ch*2**depth_level - 1)
        '''
        if(to_L == 0): return self.unet_out_ch
        else:          return min(self.hid_ch * 2**(to_L - 1), ch_upper_bound)

    def call(self, input_tensor, training=None):
        skips = []

        #####################################################
        ### Down top
        # print(self.d_top.name)  ### debug 用
        x, skip = self.d_top(input_tensor)
        skips.append(skip)
        ### Down middle
        for name, d_middle in self.d_middles.items():
            # print(name)  ### debug 用
            x, skip = d_middle(x)
            skips.append(skip)
        ### Down bottle
        # print(self.d_bottle.name)  ### debug 用
        x = self.d_bottle(x)  ### down 的 bottle沒有 skip
        #####################################################
        ### Up bottle
        # print(self.u_bottle.name)  ### debug 用
        if(self.no_concat_layer >= self.depth_level - 1): x = self.u_bottle(x)
        else:                                             x = self.u_bottle(x, skips.pop())

        ### Up middle
        for go, (name, u_middle) in enumerate(list(self.u_middles.items())[::-1]):
            # print(name)  ### debug 用
            layer_id = self.depth_level - 1 - go
            if (layer_id <= self.no_concat_layer): x = u_middle(x)
            else:                                  x = u_middle(x, skips.pop())

        ### Up top
        # print(self.u_top.name)  ### debug 用
        x = self.u_top(x)  ### up 的 top 沒有 skip
        #####################################################
        ### UNet out
        if  (self.unet_acti == "tanh"):    return self.tanh(x)
        elif(self.unet_acti == "sigmoid"): return self.sigmoid(x)


if(__name__ == "__main__"):
    import numpy as np
    import time
    from kong_util.tf_model_util import Show_model_layer_names
    # data = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)
    # start_time = time.time()  # 看資料跑一次花多少時間
    # # test_g = Generator(hid_ch=64, depth_level=7, use_bias=False)
    # test_g = Generator(hid_ch= 128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14)
    # test_g(data)
    # print("cost time", time.time() - start_time)
    # test_g.summary()
    # print(test_g(data))

    

    ############################################################################################################################
    ### 嘗試 真的 load tf_data 進來 train 看看
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import *
    from step06_b_data_pipline import tf_Data_builder
    from step09_b_loss_info_obj import Loss_info_builder
    from step09_c_train_step import *

    # from step09_e2_mask_unet2_obj import *
    # from step09_e3_flow_unet2_obj_I_to_C import *
    # from step09_e4_flow_unet2_obj_Mgt_to_C import *
    # from step09_e4_flow_unet2_obj_I_with_Mgt_to_C import *


    # model_obj = flow_unet2_ch032_tanh_L7
    # model_obj = flow_unet2_ch128_sig_L7
    # model_obj = flow_unet2_ch064_sig_L7
    # model_obj = flow_unet2_ch032_sig_L7
    # model_obj = flow_unet2_ch016_sig_L7
    # model_obj = flow_unet2_ch008_sig_L7
    # model_obj = flow_unet2_ch004_sig_L7
    # model_obj = flow_unet2_ch002_sig_L7
    # model_obj = flow_unet2_ch001_sig_L7

    # model_obj = flow_unet2_L2_ch32_sig
    # model_obj = flow_unet2_L3_ch32_sig
    # model_obj = flow_unet2_L4_ch32_sig
    # model_obj = flow_unet2_L5_ch32_sig
    # model_obj = flow_unet2_L6_ch32_sig
    # model_obj = flow_unet2_L7_ch32_sig
    # model_obj = flow_unet2_L8_ch32_sig

    # model_obj = flow_unet2_IN_L7_ch32_2to2noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to3noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to4noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to5noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to6noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to7noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to8noC_sig

    # model_obj = flow_unet2_L8_skip_use_add_sig
    # model_obj = flow_unet2_L7_skip_use_add_sig
    # model_obj = flow_unet2_L6_skip_use_add_sig
    # model_obj = flow_unet2_L5_skip_use_add_sig
    # model_obj = flow_unet2_L4_skip_use_add_sig
    # model_obj = flow_unet2_L3_skip_use_add_sig
    # model_obj = flow_unet2_L2_skip_use_add_sig

    model_obj = flow_unet2_block2_ch004_sig_8l

    model_obj = model_obj.build()  ### 可替換成 上面 想測試的 model

    ### 2. db_obj 和 tf_data
    db_obj  = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1, train_shuffle=False).set_data_use_range(in_use_range="0~1", gt_use_range="0~1").set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    ### 3. loss_info_obj
    G_mae_loss_info = Loss_info_builder().set_loss_type("mae").build()
    ### 4. 跑起來試試看
    for n, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        print("train_in.numpy().min():", train_in.numpy().min())
        print("train_in.numpy().max():", train_in.numpy().max())
        print("train_in_pre.numpy().min():", train_in_pre.numpy().min())
        print("train_in_pre.numpy().max():", train_in_pre.numpy().max())
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_obj=G_mae_loss_info)
        if(n ==  0):
            model_obj.generator.summary()
            Show_model_layer_names(model_obj.generator)
        if(n == 10):
            model_obj.generator.save_weights("try_save/weights")
            iter10 = model_obj.generator.layers[0].weights[1]
            print(iter10)
        if(n == 20):
            iter20 = model_obj.generator.layers[0].weights[1]
            print(iter20)
            model_obj.generator.load_weights("try_save/weights")
            iter20_load10 = model_obj.generator.layers[0].weights[1]
            print(iter20_load10)
