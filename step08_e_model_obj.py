from step09_c_train_step import *

from enum import Enum
import tensorflow as tf

import time
start_time = time.time()

class KModel:
    def __init__(self):  ### 共通有的 元件，其實這邊只留model_name好像也可以
        self.model_name = None
        self.epoch_log = tf.Variable(1)  ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
        self.train_step = None

    def __str__(self):
        print("model_name:", self.model_name)
        print("generator:", self.__dict__)
        return ""

class KModel_init_builder:
    def __init__(self, kong_model=None):
        if(kong_model is None): self.kong_model = KModel()
        else: self.kong_model = kong_model

        self.build = None

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.kong_model.model_name = model_name
        return self

    def set_train_step(self, train_step):
        self.kong_model.train_step = train_step
        return self

    # def build(self):
    #     return self.kong_model

class KModel_Unet_builder(KModel_init_builder):
    def build_unet(self):
        def _build_unet():
            from step08_a_1_UNet_BN_512to256 import Generator512to256, generate_sees, generate_results
            self.kong_model.generator           = Generator512to256(out_ch=2)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            self.kong_model.max_train_move = tf.Variable(1)  ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
            self.kong_model.min_train_move = tf.Variable(1)  ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
            self.kong_model.max_db_move_x  = tf.Variable(1)  ### 在test時 rec_img需要，所以需要存起來
            self.kong_model.max_db_move_y  = tf.Variable(1)  ### 在test時 rec_img需要，所以需要存起來

            self.kong_model.generate_results = generate_results  ### 不能checkpoint
            self.kong_model.generate_sees   = generate_sees    ### 不能checkpoint

            ### 建立 tf 存模型 的物件： checkpoint物件
            self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator,
                                                    optimizer_G=self.kong_model.optimizer_G,
                                                    max_train_move=self.kong_model.max_train_move,
                                                    min_train_move=self.kong_model.min_train_move,
                                                    max_db_move_x=self.kong_model.max_db_move_x,
                                                    max_db_move_y=self.kong_model.max_db_move_y,
                                                    epoch_log=self.kong_model.epoch_log)
            print("build_unet", "finish")
            return self.kong_model
        self.build = _build_unet
        return self

class KModel_Mask_Generator_builder(KModel_Unet_builder):
    def _build_mask_part(self):
        ### 生成flow的部分
        from step08_b_use_G_generate import generate_mask_flow_results, generate_mask_flow_sees_without_rec
        # self.kong_model.generate_results = generate_flow_results           ### 不能checkpoint  ### 好像用不到
        self.kong_model.generate_results = generate_mask_flow_results             ### 不能checkpoint
        self.kong_model.generate_sees    = generate_mask_flow_sees_without_rec    ### 不能checkpoint
class KModel_Flow_Generator_builder(KModel_Mask_Generator_builder):
    def _build_flow_part(self):
        ### 生成flow的部分
        from step08_b_use_G_generate import generate_flow_results, generate_flow_sees_without_rec
        # self.kong_model.generate_results = generate_flow_results           ### 不能checkpoint  ### 好像用不到
        self.kong_model.generate_results = generate_flow_results             ### 不能checkpoint
        self.kong_model.generate_sees    = generate_flow_sees_without_rec    ### 不能checkpoint
        self.kong_model.train_step       = train_step_pure_G                 ### 不能checkpoint

class KModel_UNet_Generator_builder(KModel_Flow_Generator_builder):
    def set_unet(self, hid_ch=64, depth_level=7, true_IN=False, use_bias=True, no_concat_layer=0,
                 skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False,
                 skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 out_tanh=True, out_ch=3, concat_Activation=False):
        self.hid_ch = hid_ch
        self.depth_level = depth_level
        self.no_concat_layer = no_concat_layer
        self.skip_use_add  = skip_use_add
        self.skip_use_cSE  = skip_use_cSE
        self.skip_use_sSE  = skip_use_sSE
        self.skip_use_scSE = skip_use_scSE
        self.skip_use_cnn  = skip_use_cnn
        self.skip_cnn_k    = skip_cnn_k
        self.skip_use_Acti = skip_use_Acti
        self.out_tanh = out_tanh
        self.out_ch = out_ch
        self.true_IN = true_IN
        self.concat_Activation = concat_Activation
        return self

    def set_unet2(self, hid_ch=64, depth_level=7, out_ch=3, no_concat_layer=0,
                 kernel_size=4, strides=2, norm="in",
                 d_acti="lrelu", u_acti="relu", unet_acti="tanh",
                 use_bias=True,
                 conv_block_num=0,
                 skip_op=None, skip_merge_op="concat",
                 ch_upper_bound=512,
                 #  out_tanh=True,
                 #  skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 **kwargs):
        self.hid_ch          = hid_ch
        self.depth_level     = depth_level
        self.out_ch          = out_ch
        self.no_concat_layer = no_concat_layer
        self.kernel_size     = kernel_size     ### 多的
        self.strides         = strides         ### 多的

        self.d_acti          = d_acti          ### 多的
        self.u_acti          = u_acti          ### 多的
        self.unet_acti       = unet_acti       ### 對應 out_tanh
        self.norm            = norm            ### 對應 true_IN

        self.use_bias        = use_bias        ### 之前漏的
        self.conv_block_num  = conv_block_num  ### 多的
        self.skip_op         = skip_op         ### 對應 skip_use_add, skip_use_cSE...
        self.skip_merge_op   = skip_merge_op   ### 對應 concat_Activation

        self.ch_upper_bound  = ch_upper_bound

        return self

    def _build_unet_part(self):
        ### model_part
        ### 檢查 build KModel 的時候 參數有沒有正確的傳進來~~
        # print("hid_ch", self.hid_ch)
        # print("skip_use_cSE" , self.skip_use_cSE)
        # print("skip_use_sSE" , self.skip_use_sSE)
        # print("skip_use_scSE", self.skip_use_scSE)
        # print("skip_use_cnn", self.skip_use_cnn)
        # print("skip_cnn_k", self.skip_cnn_k)
        # print("skip_use_Acti", self.skip_use_Acti)
        # print("true_IN", self.true_IN)
        # print("out_ch", self.out_ch)
        # print()
        if  (self.true_IN and self.concat_Activation is False): from step08_a_1_UNet_IN                   import Generator   ### 目前最常用這個
        elif(self.true_IN and self.concat_Activation is True) : from step08_a_1_UNet_IN_concat_Activation import Generator
        else:                                                   from step08_a_1_UNet_BN                   import Generator
        self.kong_model.generator   = Generator(hid_ch=self.hid_ch, depth_level=self.depth_level, use_bias=True, no_concat_layer=self.no_concat_layer,
                                                skip_use_add=self.skip_use_add, skip_use_cSE=self.skip_use_cSE, skip_use_sSE=self.skip_use_sSE, skip_use_scSE=self.skip_use_scSE,
                                                skip_use_cnn=self.skip_use_cnn, skip_cnn_k=self.skip_cnn_k, skip_use_Acti=self.skip_use_Acti,
                                                out_tanh=self.out_tanh, out_ch=self.out_ch)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("build_unet", "finish")
        return self

    def _build_unet_part2(self):
        ### model_part
        ### 檢查 build KModel 的時候 參數有沒有正確的傳進來~~
        # print("hid_ch         =", self.hid_ch         )
        # print("depth_level    =", self.depth_level    )
        # print("out_ch         =", self.out_ch         )
        # print("no_concat_layer=", self.no_concat_layer)
        # print("kernel_size    =", self.kernel_size    )
        # print("strides        =", self.strides        )
        # print()
        # print("d_acti         =", self.d_acti          )
        # print("u_acti         =", self.u_acti          )
        # print("unet_acti      =", self.unet_acti       )
        # print("norm           =", self.norm            )
        # print()
        # print("use_bias       =", self.use_bias        )
        # print("conv_block_num =", self.conv_block_num  )
        # print("skip_op        =", self.skip_op         )
        # print("skip_merge_op  =", self.skip_merge_op   )
        # print()
        from step08_a_UNet_combine import Generator
        self.kong_model.generator   = Generator(hid_ch=self.hid_ch, depth_level=self.depth_level, out_ch=self.out_ch, no_concat_layer=self.no_concat_layer,
                                                d_acti=self.d_acti, u_acti=self.u_acti, unet_acti=self.unet_acti, norm=self.norm,
                                                use_bias=self.use_bias, conv_block_num=self.conv_block_num, skip_op=self.skip_op, skip_merge_op=self.skip_merge_op,
                                                ch_upper_bound=self.ch_upper_bound)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("build_unet", "finish")
        return self

    def _build_ckpt_part(self):
        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)


    def use_flow_unet(self):
        def _build_flow_unet():
            self._build_flow_part()
            self._build_unet_part()  ### 先
            self._build_ckpt_part()  ### 後
            print("build_flow_unet", "finish")
            return self.kong_model
        self.build = _build_flow_unet
        return self

    def use_mask_unet(self):
        def _build_mask_unet():
            self._build_mask_part()
            self._build_unet_part()  ### 先
            self._build_ckpt_part()  ### 後
            print("build_mask_unet", "finish")
            return self.kong_model
        self.build = _build_mask_unet
        return self

    def use_mask_unet2(self):
        def _build_mask_unet():
            self._build_mask_part()
            self._build_unet_part2()  ### 先
            self._build_ckpt_part()  ### 後
            print("build_mask_unet2", "finish")
            return self.kong_model
        self.build = _build_mask_unet
        return self



    def use_flow_rect_7_level(self, first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_ReLU=False, use_res_learning=True, resb_num=9, out_ch=3):
        self.first_k = first_k
        self.hid_ch = hid_ch
        self.depth_level = depth_level
        self.true_IN = true_IN
        self.use_ReLU = use_ReLU
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.out_ch = out_ch

        def _build_flow_rect_7_level():
            '''
            depth_level=2 的情況 已經做到 和 Rect 幾乎一樣了，下面的 flow_rect 還留著是因為 裡面還有 MRFB 和 CoordConv 的東西
            '''
            ### model_part
            from step08_a_3_EResD_7_level import Rect_7_layer as Generator
            self.kong_model.generator   = Generator(first_k=self.first_k, hid_ch=self.hid_ch, depth_level=self.depth_level, true_IN=self.true_IN, use_ReLU=self.use_ReLU, use_res_learning=self.use_res_learning, resb_num=self.resb_num, out_ch=self.out_ch)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

            self._build_flow_part()
            self._build_ckpt_part()
            print("build_flow_rect_7_level", "finish")
            return self.kong_model
        self.build = _build_flow_rect_7_level
        return self

    def use_flow_rect(self, first_k3=False, hid_ch=64, true_IN=True, mrfb=None, mrf_replace=False, coord_conv=False, use_res_learning=True, resb_num=9, out_ch=3):
        self.first_k3 = first_k3
        self.hid_ch = hid_ch
        self.true_IN = true_IN
        self.mrfb = mrfb
        self.mrf_replace = mrf_replace
        self.coord_conv = coord_conv
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.out_ch = out_ch

        def _build_flow_rect():
            '''
            flow_rect 還留著是因為 裡面還有 MRFB 和 CoordConv 的東西
            '''
            ### model_part
            from step08_a_2_Rect2 import Generator
            self.kong_model.generator   = Generator(first_k3=first_k3, hid_ch=hid_ch, true_IN=true_IN, mrfb=mrfb, mrf_replace=mrf_replace, coord_conv=coord_conv, use_res_learning=use_res_learning, resb_num=resb_num, out_ch=out_ch)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

            self._build_flow_part()
            self._build_ckpt_part()
            print("build_flow_rect", "finish")
            return self.kong_model
        self.build = _build_flow_rect
        return self

class KModel_Mask_Flow_Generator_builder(KModel_UNet_Generator_builder):
    def use_mask_flow_unet(self):
        pass

class KModel_GD_and_mrfGD_builder(KModel_Mask_Flow_Generator_builder):
    def _kong_model_GD_setting(self):
        self.kong_model.generator = self.kong_model.rect.generator
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        from step08_b_use_G_generate import  generate_img_sees
        self.kong_model.generate_sees  = generate_img_sees     ### 不能checkpoint
        # from step09_c_train_step import train_step_GAN, train_step_GAN2

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(rect=self.kong_model.rect,
                                                   generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   optimizer_D=self.kong_model.optimizer_D,
                                                   epoch_log=self.kong_model.epoch_log)

    def use_rect2(self, first_k3=False, use_res_learning=True, resb_num=9, coord_conv=False, D_first_concat=True, D_kernel_size=4):
        self.first_k3 = first_k3
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.coord_conv = coord_conv
        self.D_first_concat = D_first_concat
        self.D_kernel_size = D_kernel_size

        def _build_rect2():
            from step08_a_2_Rect2 import Generator, Discriminator, Rect2
            gen_obj = Generator(first_k3=first_k3, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 建立 Generator物件
            dis_obj = Discriminator(D_first_concat=D_first_concat, D_kernel_size=D_kernel_size)
            self.kong_model.rect = Rect2(gen_obj, dis_obj)   ### 把 Generator物件 丟進 Rect建立 Rect物件
            self._kong_model_GD_setting()  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_rect2", "finish")
            return self.kong_model
        self.build = _build_rect2
        return self

    def use_rect2_mrf(self, first_k3=False, mrf_replace=False, use_res_learning=True, resb_num=9, coord_conv=False, use1=False, use3=False, use5=False, use7=False, use9=False, D_first_concat=True, D_kernel_size=4):
        self.first_k3 = first_k3
        self.mrf_replace = mrf_replace
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.coord_conv = coord_conv
        self.use1 = use1
        self.use3 = use3
        self.use5 = use5
        self.use7 = use7
        self.use9 = use9
        self.D_first_concat = D_first_concat
        self.D_kernel_size = D_kernel_size

        def _build_rect2_mrf():
            from step08_a_2_Rect2 import MRFBlock, Generator, Discriminator, Rect2
            mrfb    = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
            gen_obj = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)   ### 把 mrf物件 丟進 Generator 建立 Generator物件
            dis_obj = Discriminator(D_first_concat=D_first_concat, D_kernel_size=D_kernel_size)
            self.kong_model.rect = Rect2(gen_obj, dis_obj)   ### 再把 Generator物件 丟進 Rect建立 Rect物件
            self._kong_model_GD_setting()  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_rect2_mrf", "finish")
            return self.kong_model
        self.build = _build_rect2_mrf
        return self


class KModel_justG_and_mrf_justG_builder(KModel_GD_and_mrfGD_builder):
    def _kong_model_G_setting(self):
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        from step08_b_use_G_generate import generate_img_sees, generate_img_results
        self.kong_model.generate_results = generate_img_results  ### 不能checkpoint
        self.kong_model.generate_sees  = generate_img_sees     ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator, 
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)

    def use_justG(self, first_k3=False, use_res_learning=True, resb_num=9, coord_conv=False):
        self.first_k3 = first_k3
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.coord_conv = coord_conv

        def _build_justG():
            from step08_a_2_Rect2 import Generator
            self.kong_model.generator   = Generator(first_k3=first_k3, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 建立 Generator物件
            self._kong_model_G_setting()  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_justG", "finish")
            return self.kong_model
        self.build = _build_justG
        return self

    def use_justG_mrf(self, first_k3=False, mrf_replace=False, use_res_learning=True, resb_num=9, coord_conv=False, use1=False, use3=False, use5=False, use7=False, use9=False):
        self.first_k3 = first_k3
        self.mrf_replace = mrf_replace
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.coord_conv = coord_conv
        self.use1 = use1
        self.use3 = use3
        self.use5 = use5
        self.use7 = use7
        self.use9 = use9

        def _build_justG_mrf():
            from step08_a_2_Rect2 import MRFBlock, Generator
            mrfb = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
            self.kong_model.generator = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 把 mrf物件 丟進 Generator 建立 Generator物件
            self._kong_model_G_setting()  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_justG_mrf", "finish")
            return self.kong_model

        self.build = _build_justG_mrf
        return self


class KModel_builder(KModel_justG_and_mrf_justG_builder): pass


class MODEL_NAME(Enum):
    unet  = "unet"
    rect  = "rect"
    justG = "justG"
    flow_unet = "flow_unet"    ### 包含這flow 關鍵字就沒問題
    flow_unet2 = "flow_unet2"  ### 包含這flow 關鍵字就沒問題
    flow_rect = "flow_rect"    ### 包含這flow 關鍵字就沒問題
    mask_flow_unet = "mask_flow_unet"


### 直接先建好 obj 給外面import囉！
unet                     = KModel_builder().set_model_name(MODEL_NAME.unet ).build_unet().set_train_step(train_step_first)
#######################################################################################################################
rect                     = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=False).set_train_step(train_step_GAN)   ### G 只train 1次
rect_firstk3             = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True ).set_train_step(train_step_GAN)   ### G 只train 1次
rect_Gk4_many_Dk4_concat = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=False).set_train_step(train_step_GAN2)  ### G 可train 多次， 目前G_train幾次要手動改喔！

rect_mrfall         = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use1=True, use3=True, use5=True, use7=True, use9=True).set_train_step(train_step_GAN)  ### G 只train 1次
rect_mrf7           = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use7=True) .set_train_step(train_step_GAN)  ### G 只train 1次
rect_mrf79          = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True).set_train_step(train_step_GAN)  ### G 只train 1次
rect_replace_mrf7   = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=True , use7=True).set_train_step(train_step_GAN)  ### G 只train 1次
rect_replace_mrf79  = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, use9=True).set_train_step(train_step_GAN)  ### G 只train 1次
#######################################################################################################################
justG               = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=False).set_train_step(train_step_pure_G)
justG_firstk3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True ).set_train_step(train_step_pure_G)
########################################################### 2
justG_mrf7          = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=False, use7=True).set_train_step(train_step_pure_G)
justG_mrf7_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use7=True).set_train_step(train_step_pure_G)
justG_mrf5_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use5=True).set_train_step(train_step_pure_G)
justG_mrf3_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True).set_train_step(train_step_pure_G)
########################################################### 3
justG_mrf79         = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True).set_train_step(train_step_pure_G)
justG_mrf79_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use7=True, use9=True).set_train_step(train_step_pure_G)
justG_mrf57_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use5=True, use7=True).set_train_step(train_step_pure_G)
justG_mrf35_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True).set_train_step(train_step_pure_G)
########################################################### 4
justG_mrf_replace7  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True).set_train_step(train_step_pure_G)
justG_mrf_replace5  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use5=True).set_train_step(train_step_pure_G)
justG_mrf_replace3  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use3=True).set_train_step(train_step_pure_G)
########################################################### 5
justG_mrf_replace79 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use9=True).set_train_step(train_step_pure_G)
justG_mrf_replace75 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use5=True).set_train_step(train_step_pure_G)
justG_mrf_replace35 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use3=True, use5=True).set_train_step(train_step_pure_G)
########################################################### 2c
justG_mrf135_k3     = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True).set_train_step(train_step_pure_G)
justG_mrf357_k3     = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True).set_train_step(train_step_pure_G)
justG_mrf3579_k3    = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True).set_train_step(train_step_pure_G)

rect_mrf35_Gk3_DnoC_k4     = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
rect_mrf135_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
rect_mrf357_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
rect_mrf3579_Gk3_DnoC_k4   = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
########################################################### 9a
# rect_D_concat_k4    = "rect_D_concat_k4" ### 原始版本
rect_Gk4_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=True , D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk4_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk4_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=False, D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
########################################################### 9b
rect_Gk3_D_concat_k4       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=True , D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk3_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=True , D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk3_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk3_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
########################################################### 10
rect_Gk3_train3_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN2)  ### G 可train 多次， 目前G_train幾次要手動改喔！
rect_Gk3_train5_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN2)  ### G 可train 多次， 目前G_train幾次要手動改喔！
########################################################### 11
justG_fk3_no_res             = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=False).set_train_step(train_step_pure_G)  ### 127.51
rect_fk3_no_res_D_no_concat  = KModel_builder().set_model_name(MODEL_NAME.rect  ).use_rect2(first_k3=True, use_res_learning=False, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次  ### 127.28
justG_fk3_no_res_mrf357      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True, mrf_replace=False, use_res_learning=False, use3=True, use5=True, use7=True).set_train_step(train_step_pure_G)   ### 128.246
########################################################### 12
Gk3_resb00  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=0).set_train_step(train_step_pure_G)   ### 127.48
Gk3_resb01  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=1).set_train_step(train_step_pure_G)   ### 127.35
Gk3_resb03  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=3).set_train_step(train_step_pure_G)   ### 127.55
Gk3_resb05  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=5).set_train_step(train_step_pure_G)   ### 128.246
Gk3_resb07  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=7).set_train_step(train_step_pure_G)   ### 127.28
Gk3_resb09  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=9).set_train_step(train_step_pure_G)   ### 127.51
Gk3_resb11  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=11).set_train_step(train_step_pure_G)   ### 127.51
Gk3_resb15  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=15).set_train_step(train_step_pure_G)   ### 127.28
Gk3_resb20  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=20).set_train_step(train_step_pure_G)   ### 127.51

########################################################### 13 加coord_conv試試看
justGk3_coord_conv        = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG    (first_k3=True, coord_conv=True).set_train_step(train_step_pure_G)
justGk3_mrf357_coord_conv = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True, coord_conv=True, mrf_replace=False, use3=True, use5=True, use7=True).set_train_step(train_step_pure_G)


###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 14 快接近IN了
flow_unet       = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

### 在 git commit bc25e 之前都是 old ch 喔！ 最大都是 32*8=256, 16*8=128, 8*8=64 而已， 而128*8=1024 又有點太大， new_ch 就是根據層 做 2**layer，最大取512 囉！
### 如果想回復的話，要用 git 回復到 bc25e 或 之前的版本囉！
flow_unet_old_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_old_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_old_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_old_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 8 , out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 測試 subprocess
flow_unet_epoch2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=4, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_epoch3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=6, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_epoch4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=8, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 真的IN
flow_unet_IN_ch64 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)


flow_unet_IN_new_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  8, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch004 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  4, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 真的IN，跟DewarpNet一樣 CNN 不用 bias
flow_unet_IN_ch64_cnnNoBias = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, use_bias=False).use_flow_unet().set_train_step(train_step_pure_G)


########################################################### 14 看 concat Activation 有沒有差
flow_unet_ch64_in_concat_A = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, out_ch=3, concat_Activation=True).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 看 不同level 的效果
flow_unet_2_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_3_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_4_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_5_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_6_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_7_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_8_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
########################################################### 14 看 unet 的 concat 改成 + 會有什麼影響
flow_unet_8_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_7_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_6_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_5_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_4_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_3_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_2_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 看 unet 的 output 改成sigmoid
flow_unet_IN_ch64_sigmoid  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, out_tanh=False, true_IN=True).use_flow_unet()
########################################################### 14 看 unet 的 第一層試試看 不 concat 效果如何
flow_unet_IN_7l_ch64_2to2noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch32_2to2noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=2).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to3noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to4noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=4).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to5noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=5).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to6noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=6).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to7noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=7).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to8noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=8).use_flow_unet().set_train_step(train_step_pure_G)
########################################################### 14 看 unet 的 skip 中間接 cnn 的效果
flow_unet_IN_7l_ch64_skip_use_cnn1_NO_relu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=None).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_skip_use_cnn1_USErelu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=tf.nn.relu).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_skip_use_cnn1_USEsigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=tf.nn.sigmoid).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_skip_use_cnn3_USErelu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=3, skip_use_Acti=tf.nn.relu).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_skip_use_cnn3_USEsigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=3, skip_use_Acti=tf.nn.sigmoid).use_flow_unet().set_train_step(train_step_pure_G)


########################################################### 14 看 unet 的 skip 中間接 cSE, sSE, csSE 的效果
flow_unet_IN_7l_ch64_2to3noC_sk_cSE  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=3, skip_use_cSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to3noC_sk_sSE  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2, skip_use_sSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_2to3noC_sk_scSE = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2, skip_use_scSE=True).use_flow_unet().set_train_step(train_step_pure_G)

flow_unet_IN_7l_ch64_skip_use_cSE    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_skip_use_sSE    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_sSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_7l_ch64_skip_use_scSE   = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_scSE=True).use_flow_unet().set_train_step(train_step_pure_G)

###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 15 用 resblock 來試試看
flow_rect_fk3_ch64_tfIN_resb_ok9 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect(first_k3=True, hid_ch=64, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

flow_rect_7_level_fk7 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

flow_rect_2_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_3_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_4_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_5_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_6_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_7_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

flow_rect_2_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_3_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_4_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_5_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_6_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_7_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

###############################################################################################################################################################################################
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 1 嘗試看看 mask_unet 搭配 BCE 這裡try 7l
mask_unet_ch032_tanh_7l = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch128_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch064_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 64, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch032_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch016_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch008_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  8, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch004_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  4, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch002_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  2, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch001_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  1, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
### 從這裡 嘗試 2l ~ 7l try出 6l最好
mask_unet_2_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=2, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_3_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=3, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_4_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=4, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_5_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=5, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_6_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_7_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=7, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_8_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=8, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)

mask_unet_IN_7l_ch32_2to2noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=2, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_7l_ch32_2to3noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=3, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_7l_ch32_2to4noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=4, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_7l_ch32_2to5noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=5, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_7l_ch32_2to6noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=6, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_7l_ch32_2to7noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=7, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_7l_ch32_2to8noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=8, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)

mask_unet_8_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_7_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_6_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_5_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_4_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_3_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_2_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
########################################################### 1  嘗試看看 mask_unet，從上面已知 6l 最好， 所以下面固定6l
mask_unet_ch032_tanh_6l = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, depth_level=6, true_IN=True, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch128_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch064_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 64, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch032_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch016_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch008_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  8, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch004_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  4, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch002_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  2, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_ch001_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  1, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)

mask_unet_IN_6l_ch32_2to2noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=2, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_6l_ch32_2to3noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=3, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_6l_ch32_2to4noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=4, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_6l_ch32_2to5noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=5, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
mask_unet_IN_6l_ch32_2to6noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=6, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)

mask_unet_6_level_skip_use_add_sig_6l = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 2 嘗試看看 mask_unet2 搭配 BCE 這裡try 7l
mask_unet2_ch032_tanh_7l = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, out_ch=1, depth_level=7, unet_acti="tanh").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch128_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch064_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch032_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch016_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch008_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch004_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch002_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_ch001_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
### 從這裡 嘗試 2l ~ 7l try出 6l最好
mask_unet2_2_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=2, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_3_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=3, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_4_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=4, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_5_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=5, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_6_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=6, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_7_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_8_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=8, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)

mask_unet2_IN_7l_ch32_2to2noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=2, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_IN_7l_ch32_2to3noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=3, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_IN_7l_ch32_2to4noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=4, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_IN_7l_ch32_2to5noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=5, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_IN_7l_ch32_2to6noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=6, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_IN_7l_ch32_2to7noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=7, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_IN_7l_ch32_2to8noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=8, unet_acti="sigmoid").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)

mask_unet2_8_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=8, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_7_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=7, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_6_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=6, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_5_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=5, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_4_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=4, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_3_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=3, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_2_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=2, unet_acti="sigmoid", skip_merge_op="add").use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
########################################################### 2 嘗試看看 mask_unet2 搭配 BCE 這裡try 7l
mask_unet2_block1_ch032_tanh_7l = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, out_ch=1, depth_level=7, unet_acti="tanh", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch128_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
### 從這裡 嘗試 2l ~ 7l try出 6l最好
mask_unet2_block1_2_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=2, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_3_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=3, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_4_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=4, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_5_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=5, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_6_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=6, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_7_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_8_level_ch32_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=8, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)

mask_unet2_block1_IN_7l_ch32_2to2noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=2, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_7l_ch32_2to3noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=3, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_7l_ch32_2to4noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=4, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_7l_ch32_2to5noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=5, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_7l_ch32_2to6noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=6, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_7l_ch32_2to7noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=7, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_7l_ch32_2to8noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, out_ch=1, depth_level=7, no_concat_layer=8, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)

mask_unet2_block1_8_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=8, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_7_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=7, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_6_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=6, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_5_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=5, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_4_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=4, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_3_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=3, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_2_level_skip_use_add_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, out_ch=1, depth_level=2, unet_acti="sigmoid", skip_merge_op="add", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
########################################################### 2  嘗試看看 mask_unet2，從上面已知 4,6,7,8l 不錯，所以都來試試看
### Block1
mask_unet2_block1_ch128_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block1_ch128_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block1_ch128_sig_4l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch128_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)

mask_unet2_block1_ch016_sig_4l_E_relu  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=4, out_ch=1, d_acti="relu", unet_acti="sigmoid",  conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_4l_no_Bias = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=4, out_ch=1, unet_acti="sigmoid", use_bias=False, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block1_ch128_sig_5l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_5l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch128_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block1_ch128_sig_6l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_6l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_6l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch128_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block1_ch064_sig_7l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_7l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_7l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch128_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block1_ch032_sig_8l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_8l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_8l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch128_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch064_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch032_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch016_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch008_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch004_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch002_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_ch001_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 2  嘗試看看 mask_unet2，從上面已知 4,6,7,8l 不錯，所以都來試試看
### Block2
mask_unet2_block2_ch128_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_2l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=2, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block2_ch128_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_3l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=3, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block2_ch128_sig_4l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch128_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_4l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block2_ch128_sig_5l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_5l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch128_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_5l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=5, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block2_ch128_sig_6l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_6l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_6l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch128_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_6l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=6, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block2_ch064_sig_7l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_7l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_7l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch128_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_7l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=7, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
###############################################################################################################################################################################################
mask_unet2_block2_ch032_sig_8l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_8l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_8l_no_limit = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2, ch_upper_bound= 2**14).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch128_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=128, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch064_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 64, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch032_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 32, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch016_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch= 16, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch008_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  8, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch004_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  4, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch002_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  2, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block2_ch001_sig_8l  = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=  1, depth_level=8, out_ch=1, unet_acti="sigmoid", conv_block_num=2).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)


'''  等試完 layer 再來看這些變化
mask_unet2_block1_IN_6l_ch32_2to2noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=2, out_tanh=False, out_ch=1, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_6l_ch32_2to3noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=3, out_tanh=False, out_ch=1, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_6l_ch32_2to4noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=4, out_tanh=False, out_ch=1, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_6l_ch32_2to5noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=5, out_tanh=False, out_ch=1, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
mask_unet2_block1_IN_6l_ch32_2to6noC_sig = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=32, depth_level=6, true_IN=True, no_concat_layer=6, out_tanh=False, out_ch=1, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)

mask_unet2_block1_6_level_skip_use_add_sig_6l = KModel_builder().set_model_name(MODEL_NAME.flow_unet2).set_unet2(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1, conv_block_num=1).use_mask_unet2().set_train_step(train_step_pure_G_split_mask_move)
'''

if(__name__ == "__main__"):
    # print(flow_rect_2_level_fk3.build())
    # print(mask_unet_ch032_tanh_7l.build())
    print(mask_unet2_ch128_sig_7l.build())
    print("build_model cost time:", time.time() - start_time)
    pass
