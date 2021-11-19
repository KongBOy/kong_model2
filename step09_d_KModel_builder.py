from enum import Enum
import tensorflow as tf

import time
start_time = time.time()

class KModel:
    def __init__(self):  ### 共通有的 元件，其實這邊只留model_name好像也可以
        self.model_name = None
        self.model_describe = ""
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

class Old_512_256_Unet_builder(KModel_init_builder):
    def build_unet(self):
        def _build_unet():
            from step08_a_1_UNet_BN_512to256 import Generator512to256, generate_sees, generate_results
            self.kong_model.generator   = Generator512to256(out_ch=2)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            self.kong_model.max_train_move = tf.Variable(1)  ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
            self.kong_model.min_train_move = tf.Variable(1)  ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
            self.kong_model.max_db_move_x  = tf.Variable(1)  ### 在test時 rec_img需要，所以需要存起來
            self.kong_model.max_db_move_y  = tf.Variable(1)  ### 在test時 rec_img需要，所以需要存起來

            self.kong_model.generate_results = generate_results  ### 不能checkpoint
            self.kong_model.generate_sees    = generate_sees    ### 不能checkpoint

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

class G_Mask_op_builder(Old_512_256_Unet_builder):
    def _build_mask_op_part(self):
        ### 生成 mask 的 operation
        from step08_b_use_G_generate import generate_mask_flow_results, generate_mask_flow_sees_without_rec
        # self.kong_model.generate_results = generate_flow_results           ### 不能checkpoint  ### 好像用不到
        self.kong_model.generate_results = generate_mask_flow_results             ### 不能checkpoint
        self.kong_model.generate_sees    = generate_mask_flow_sees_without_rec    ### 不能checkpoint
class G_Flow_op_builder(G_Mask_op_builder):
    def _build_flow_op_part(self, M_to_C=False):
        ### 生成 flow 的 operation
        if(M_to_C):
            from step08_b_use_G_generate import gt_mask_Generate_gt_flow, gt_mask_Generate_gt_flow_see
            self.kong_model.generate_results = gt_mask_Generate_gt_flow        ### 不能checkpoint
            self.kong_model.generate_sees    = gt_mask_Generate_gt_flow_see    ### 不能checkpoint
        else:
            from step08_b_use_G_generate import generate_flow_results, generate_flow_sees_without_rec
            # self.kong_model.generate_results = generate_flow_results           ### 不能checkpoint  ### 好像用不到
            self.kong_model.generate_results = generate_flow_results             ### 不能checkpoint
            self.kong_model.generate_sees    = generate_flow_sees_without_rec    ### 不能checkpoint
class G_Ckpt_op_builder(G_Flow_op_builder):
    def _build_ckpt_part(self):
        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)

class G_Unet_Body_builder(G_Ckpt_op_builder):
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
                 coord_conv=False,
                 #  out_tanh=True,
                 #  skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 **kwargs):
        self.kong_model.model_describe = "_L%i_ch%03i_block%i_%s" % (depth_level, hid_ch, conv_block_num, unet_acti)
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
        self.coord_conv      = coord_conv

        return self

    def _build_unet_body_part(self):
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
                                                ch_upper_bound=self.ch_upper_bound, coord_conv=self.coord_conv)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("build_unet", "finish")
        return self


class G_Unet_Purpose_builder(G_Unet_Body_builder):
    def use_flow_unet(self):
        def _build_flow_unet():
            self._build_unet_body_part()  ### 先， 用 step08_a_1_UNet_BN or step08_a_1_UNet_BN or step08_a_1_UNet_IN_concat_Activation or step08_a_1_UNet_IN
            self._build_ckpt_part()       ### 後
            self._build_flow_op_part()    ### 用 flow_op
            print("build_flow_unet", "finish")
            return self.kong_model
        self.build = _build_flow_unet
        return self

    def use_mask_unet(self):
        def _build_mask_unet():
            self._build_unet_body_part()  ### 先， 用 step08_a_1_UNet_BN or step08_a_1_UNet_BN or step08_a_1_UNet_IN_concat_Activation or step08_a_1_UNet_IN
            self._build_ckpt_part()       ### 後
            self._build_mask_op_part()    ### 用 mask_op
            print("build_mask_unet", "finish")
            return self.kong_model
        self.build = _build_mask_unet
        return self

    def use_mask_unet2(self):
        def _build_mask_unet():
            self._build_unet_part2()    ### 先， 用 step08_a_UNet_combine
            self._build_ckpt_part()     ### 後
            self._build_mask_op_part()  ### 用 mask_op
            print("build_mask_unet2", "finish")
            return self.kong_model
        self.build = _build_mask_unet
        return self

    def use_flow_unet2(self, M_to_C=False):
        def _build_mask_unet():
            self._build_unet_part2()    ### 先， 用 step08_a_UNet_combine
            self._build_ckpt_part()     ### 後
            self._build_flow_op_part(M_to_C=M_to_C)  ### 用 flow_op
            print("build_flow_unet2", "finish")
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

            self._build_flow_op_part()
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

            self._build_flow_op_part()
            self._build_ckpt_part()
            print("build_flow_rect", "finish")
            return self.kong_model
        self.build = _build_flow_rect
        return self

class KModel_Mask_Flow_Generator_builder(G_Unet_Purpose_builder):
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
