from enum import Enum
import tensorflow as tf

import time
start_time = time.time()

class KModel:
    def __init__(self):  ### 共通有的 元件，其實這邊只留model_name好像也可以
        self.model_name = None
        self.model_describe = ""
        self.epoch_log = tf.Variable(1)  ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch

        self.model_describe_elements = []
        self.model_describe = None

        self.generator        = None
        self.optimizer_G      = None

        self.ckpt             = None

        self.train_step       = None

        self.generate_results = None
        self.generate_sees    = None
        self.generate_tests   = None

    def __str__(self):
        print("model_name:", self.model_name)
        print("generator:", self.__dict__)
        return ""

class KModel_init_builder:
    def __init__(self, kong_model=None):
        if(kong_model is None): self.kong_model = KModel()
        else: self.kong_model = kong_model

        self.build_ops = []

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.kong_model.model_name = model_name
        return self

    def set_train_step(self, train_step):
        self.kong_model.train_step = train_step
        return self

    def build(self):
        for op in self.build_ops: op()
        return self.kong_model

    # def build(self):
    #     return self.kong_model


class G_Ckpt_op_builder(KModel_init_builder):
    def _build_ckpt_part(self):
        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)
        print("ckpt finish")

class G_Unet_Body_builder(G_Ckpt_op_builder):
    def set_unet(self, hid_ch=64, depth_level=7, true_IN=False, use_bias=True, no_concat_layer=0,
                 skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False,
                 skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 out_tanh=True, out_ch=3, concat_Activation=False):

        self.true_IN = true_IN
        self.concat_Activation = concat_Activation

        g_args = {
            "hid_ch"            : hid_ch,
            "depth_level"       : depth_level,
            "no_concat_layer"   : no_concat_layer,
            "skip_use_add"      : skip_use_add,
            "skip_use_cSE"      : skip_use_cSE,
            "skip_use_sSE"      : skip_use_sSE,
            "skip_use_scSE"     : skip_use_scSE,
            "skip_use_cnn"      : skip_use_cnn,
            "skip_cnn_k"        : skip_cnn_k,
            "skip_use_Acti"     : skip_use_Acti,
            "out_tanh"          : out_tanh,
            "out_ch"            : out_ch}

        def _build_unet_body_part():
            ### model_part
            ### 檢查 build KModel 的時候 參數有沒有正確的傳進來~~
            if  (self.true_IN and self.concat_Activation is False): from step08_a_1_UNet_IN                   import Generator   ### 目前最常用這個
            elif(self.true_IN and self.concat_Activation is True) : from step08_a_1_UNet_IN_concat_Activation import Generator
            else:                                                   from step08_a_1_UNet_BN                   import Generator
            self.kong_model.generator   = Generator(**g_args)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            print("build_unet", "finish")

        self.build_ops.append(_build_unet_body_part)
        self.build_ops.append(self._build_ckpt_part)
        return self

    def set_unet2(self, hid_ch=64, depth_level=7, out_ch=3, no_concat_layer=0,
                 kernel_size=4, strides=2, norm="in",
                 d_acti="lrelu", u_acti="relu", unet_acti="tanh",
                 use_bias=True,
                 conv_block_num=0,
                 skip_op=None, skip_merge_op="concat",
                 ch_upper_bound=512,
                 coord_conv=False,
                 d_amount = 1,
                 bottle_divide=False,
                 #  out_tanh=True,
                 #  skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 **kwargs):
        self.kong_model.model_describe_elements = ["L%i" % depth_level, "ch%03i" % hid_ch, "block%i" % conv_block_num, unet_acti[:3], "out_%i" % out_ch]
        self.kong_model.model_describe = "_".join(self.kong_model.model_describe_elements)
        g_args = {
            "hid_ch"          : hid_ch,
            "depth_level"     : depth_level,
            "out_ch"          : out_ch,
            "no_concat_layer" : no_concat_layer,
            "kernel_size"     : kernel_size,     ### 多的
            "strides"         : strides,         ### 多的
            "d_acti"          : d_acti,          ### 多的
            "u_acti"          : u_acti,          ### 多的
            "unet_acti"       : unet_acti,       ### 對應 out_tanh
            "norm"            : norm,            ### 對應 true_IN
            "use_bias"        : use_bias,        ### 之前漏的
            "conv_block_num"  : conv_block_num,  ### 多的
            "skip_op"         : skip_op,         ### 對應 skip_use_add, skip_use_cSE...
            "skip_merge_op"   : skip_merge_op,   ### 對應 concat_Activation
            "ch_upper_bound"  : ch_upper_bound,
            "coord_conv"      : coord_conv,
            "d_amount"        : d_amount,
            "bottle_divide"   : bottle_divide, }


        def _build_unet_body_part():
            # for key, value in kwargs.items(): print(f"{key}: {value}")  ### 檢查 build KModel 的時候 參數有沒有正確的傳進來~~
            from step08_a_0a_UNet_combine import Generator
            self.kong_model.generator   = Generator(**g_args)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            print("build_unet", "finish")

        self.build_ops.append(_build_unet_body_part)  ### 先
        self.build_ops.append(self._build_ckpt_part)  ### 後
        return self

    def set_multi_model_builders(self, op_type, **model_builders_dict):
        '''
        參數名字要丟什麼可以參考 step08_a_0b_Multi_UNet 的 Multi_Generator 的 call()
        舉例：set_multi_model_builders(op_type     = "I_to_M_w_I_to_C",
                                       I_to_M     = step09_e2_mask_unet2_obj.mask_unet2_block1_ch008_sig_L2,
                                       M_w_I_to_C = step09_e5_flow_unet2_obj_I_with_Mgt_to_C.flow_unet2_block1_ch008_sig_L2)
        '''
        for gen_name, model_builder in model_builders_dict.items():
            self.kong_model.model_describe_elements += [gen_name] + model_builder.kong_model.model_describe_elements + ["&&"]
        self.kong_model.model_describe_elements.pop()  ### 把結尾的 & pop 掉
        self.kong_model.model_describe = "_".join(self.kong_model.model_describe_elements)

        def _build_multi_unet_body_part():
            from step08_a_0b_Multi_UNet import Multi_Generator
            gens_dict = {}
            for gen_name, model_builder in model_builders_dict.items():
                gens_dict[gen_name] = model_builder.build().generator
            self.kong_model.generator   = Multi_Generator(op_type, gens_dict)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            print("build_multi_unet", "finish")

        self.build_ops.append(_build_multi_unet_body_part)  ### 先
        self.build_ops.append(self._build_ckpt_part)  ### 後
        return self

class G_Unet_Purpose_builder(G_Unet_Body_builder):
    def _hook_Gen_op_part(self, I_to_M=False,
                            I_to_C_with_Mgt_to_F=False,
                            I_to_W=False,
                            I_with_Mgt_to_C_with_Mgt_to_F=False,
                            I_to_M_w_I_to_C=False,
                            I_to_Cx_Cy = False,
                            Mgt_to_C_with_gt_M_to_F=False,
                            I_w_Mgt_to_Cx=False,
                            I_w_Mgt_to_Wx_Wy_Wz=False):
        if  (I_to_M):
            from step08_b_use_G_generate_I_to_M import  I_Generate_M_see
            self.kong_model.generate_sees    = I_Generate_M_see    ### 不能checkpoint
        ### 生成 flow 的 operation
        elif(I_to_C_with_Mgt_to_F):
            from step08_b_use_G_generate_I_to_C import  I_Generate_C_with_Mgt_to_F_see
            self.kong_model.generate_sees    = I_Generate_C_with_Mgt_to_F_see    ### 不能checkpoint
        elif(Mgt_to_C_with_gt_M_to_F):
            from step08_b_use_G_generate_M_to_C import Mgt_Generate_C_with_Mgt_to_F_see
            self.kong_model.generate_sees    = Mgt_Generate_C_with_Mgt_to_F_see    ### 不能checkpoint
        elif(I_to_M_w_I_to_C):
            from step08_b_use_G_generate_I_to_M_w_I_to_C import  I_gen_M_w_I_gen_C_w_M_to_F_see
            self.kong_model.generate_sees    = I_gen_M_w_I_gen_C_w_M_to_F_see    ### 不能checkpoint
        elif(I_with_Mgt_to_C_with_Mgt_to_F):
            from step08_b_use_G_generate_I_w_M_to_C import  I_with_Mgt_Generate_C_with_Mgt_to_F_see
            self.kong_model.generate_sees    = I_with_Mgt_Generate_C_with_Mgt_to_F_see    ### 不能checkpoint
        elif(I_to_W):
            from step08_b_use_G_generate_I_to_W import I_Generate_W_see
            self.kong_model.generate_sees    = I_Generate_W_see    ### 不能checkpoint
        elif(I_to_Cx_Cy):
            from step08_b_use_G_generate_I_w_M_to_Cx_Cy import  I_w_Mgt_Gen_Cx_Cy_to_C_with_Mgt_to_F_see
            self.kong_model.generate_sees    = I_w_Mgt_Gen_Cx_Cy_to_C_with_Mgt_to_F_see
        elif(I_w_Mgt_to_Cx):
            from step08_b_use_G_generate_I_w_M_to_Cx import  I_w_Mgt_to_Cx_see
            self.kong_model.generate_sees    = I_w_Mgt_to_Cx_see   ### 不能checkpoint
        elif(I_w_Mgt_to_Wx_Wy_Wz):
            from step08_b_use_G_generate_I_w_M_to_Wx_Wy_Wz import  I_w_M_Gen_Wx_Wy_Wz_to_W_see
            self.kong_model.generate_sees    = I_w_M_Gen_Wx_Wy_Wz_to_W_see   ### 不能checkpoint
        else:
            from step08_b_use_G_generate_I_to_F import I_Generate_F_see
            self.kong_model.generate_sees    = I_Generate_F_see  ### 不能checkpoint

    def hook_build_and_gen_op(self, I_to_M=False, I_to_C_with_Mgt_to_F=False, I_to_W=False, I_with_Mgt_to_C_with_Mgt_to_F=False, I_to_M_w_I_to_C=False, Mgt_to_C_with_gt_M_to_F=False, I_to_Cx_Cy=False, I_w_Mgt_to_Cx=False, I_w_Mgt_to_Wx_Wy_Wz=False):
        def _hook_Gen_op():
            self._hook_Gen_op_part( I_to_M=I_to_M,
                                    I_to_C_with_Mgt_to_F=I_to_C_with_Mgt_to_F,
                                    I_to_W=I_to_W,
                                    I_with_Mgt_to_C_with_Mgt_to_F=I_with_Mgt_to_C_with_Mgt_to_F,
                                    I_to_M_w_I_to_C=I_to_M_w_I_to_C,
                                    Mgt_to_C_with_gt_M_to_F=Mgt_to_C_with_gt_M_to_F,
                                    I_to_Cx_Cy=I_to_Cx_Cy,
                                    I_w_Mgt_to_Cx=I_w_Mgt_to_Cx,
                                    I_w_Mgt_to_Wx_Wy_Wz=I_w_Mgt_to_Wx_Wy_Wz)  ### 用 flow_op
            print("build_flow_unet2~~", "finish")
            # return self.kong_model
        self.build_ops.append(_hook_Gen_op)
        return self


class Old_model_and_512_256_Unet_builder(G_Unet_Purpose_builder):
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
        self.build_ops.append(_build_unet)
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

            self._hook_Gen_op_part()
            self._build_ckpt_part()
            print("build_flow_rect_7_level", "finish")
            return self.kong_model
        self.build_ops.append(_build_flow_rect_7_level)
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

            self._hook_Gen_op_part()
            self._build_ckpt_part()
            print("build_flow_rect", "finish")
            return self.kong_model
        self.build_ops.append(_build_flow_rect)
        return self

class KModel_Mask_Flow_Generator_builder(Old_model_and_512_256_Unet_builder):
    def use_mask_flow_unet(self):
        pass

class KModel_GD_and_mrfGD_builder(KModel_Mask_Flow_Generator_builder):
    def _kong_model_GD_setting(self):
        self.kong_model.generator = self.kong_model.rect.generator
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        from step08_b_use_G_generate_I_to_R import  I_Generate_R_see
        self.kong_model.generate_sees  = I_Generate_R_see     ### 不能checkpoint
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
        self.build_ops.append(_build_rect2)
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
        self.build_ops.append(_build_rect2_mrf)
        return self


class KModel_justG_and_mrf_justG_builder(KModel_GD_and_mrfGD_builder):
    def _kong_model_G_setting(self):
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        from step08_b_use_G_generate_I_to_R import I_Generate_R_see, I_Generate_R
        self.kong_model.generate_results = I_Generate_R  ### 不能checkpoint
        self.kong_model.generate_sees  = I_Generate_R_see     ### 不能checkpoint

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
        self.build_ops.append(_build_justG)
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

        self.build_ops.append(_build_justG_mrf)
        return self


class KModel_builder(KModel_justG_and_mrf_justG_builder): pass


class MODEL_NAME(Enum):
    '''
    目前會參考 model_name 做事情的地方：
        step6b_data_pipline 的 set_img_resize
        step10a train想用以前的 512_to_256 和  train_step1_see_current_img 
    '''
    unet  = "unet"
    rect  = "rect"
    justG = "justG"
    flow_unet = "flow_unet"    ### 包含這flow 關鍵字就沒問題
    flow_unet2 = "flow_unet2"  ### 包含這flow 關鍵字就沒問題
    flow_rect = "flow_rect"    ### 包含這flow 關鍵字就沒問題
    mask_flow_unet = "mask_flow_unet"
    multi_flow_unet = "multi_flow_unet"
