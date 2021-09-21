from enum import Enum
import tensorflow as tf

import time
start_time = time.time()

class KModel:
    def __init__(self):  ### 共通有的 元件，其實這邊只留model_name好像也可以
        self.model_name = None
        self.epoch_log = tf.Variable(1)  ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch

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

    # def build(self):
    #     return self.kong_model

class KModel_Unet_builder(KModel_init_builder):
    def build_unet(self):
        def _build_unet():
            from step08_a_1_UNet_BN_512to256 import Generator512to256, generate_sees, generate_results, train_step
            self.kong_model.generator           = Generator512to256(out_ch=2)
            self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            self.kong_model.max_train_move = tf.Variable(1)  ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
            self.kong_model.min_train_move = tf.Variable(1)  ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
            self.kong_model.max_db_move_x  = tf.Variable(1)  ### 在test時 rec_img需要，所以需要存起來
            self.kong_model.max_db_move_y  = tf.Variable(1)  ### 在test時 rec_img需要，所以需要存起來

            self.kong_model.generate_results = generate_results  ### 不能checkpoint
            self.kong_model.generate_sees   = generate_sees    ### 不能checkpoint
            self.kong_model.train_step      = train_step       ### 不能checkpoint

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
        from step09_b_train_step import train_step_pure_G_split_mask_move
        # self.kong_model.generate_results = generate_flow_results           ### 不能checkpoint  ### 好像用不到
        self.kong_model.generate_results = generate_mask_flow_results             ### 不能checkpoint
        self.kong_model.generate_sees    = generate_mask_flow_sees_without_rec    ### 不能checkpoint
        self.kong_model.train_step       = train_step_pure_G_split_mask_move      ### 不能checkpoint

class KModel_Flow_Generator_builder(KModel_Mask_Generator_builder):
    def _build_flow_part(self):
        ### 生成flow的部分
        from step08_b_use_G_generate import generate_flow_results, generate_flow_sees_without_rec
        from step09_b_train_step import train_step_pure_G
        # self.kong_model.generate_results = generate_flow_results           ### 不能checkpoint  ### 好像用不到
        self.kong_model.generate_results = generate_flow_results             ### 不能checkpoint
        self.kong_model.generate_sees    = generate_flow_sees_without_rec    ### 不能checkpoint
        self.kong_model.train_step       = train_step_pure_G                 ### 不能checkpoint

class KModel_UNet_Generator_builder(KModel_Flow_Generator_builder):
    def set_unet(self, hid_ch=64, depth_level=7, true_IN=False, cnn_bias=True, no_concat_layer=0, skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None, out_tanh=True, out_ch=3, concat_Activation=False):
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
        self.kong_model.generator   = Generator(hid_ch=self.hid_ch, depth_level=self.depth_level, cnn_bias=True, no_concat_layer=self.no_concat_layer, skip_use_add=self.skip_use_add, skip_use_cSE=self.skip_use_cSE, skip_use_sSE=self.skip_use_sSE, skip_use_scSE=self.skip_use_scSE, skip_use_cnn=self.skip_use_cnn, skip_cnn_k=self.skip_cnn_k, skip_use_Acti=self.skip_use_Acti, out_tanh=self.out_tanh, out_ch=self.out_ch)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        print("build_flow_unet", "finish")
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
    def _kong_model_GD_setting(self, g_train_many=False):
        self.kong_model.generator = self.kong_model.rect.generator
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        from step08_b_use_G_generate import  generate_img_sees
        self.kong_model.generate_sees  = generate_img_sees     ### 不能checkpoint
        from step09_b_train_step import train_step_GAN, train_step_GAN2
        if  (g_train_many): self.kong_model.train_step = train_step_GAN2  ### 不能checkpoint
        else:               self.kong_model.train_step = train_step_GAN   ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(rect=self.kong_model.rect,
                                                   generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   optimizer_D=self.kong_model.optimizer_D,
                                                   epoch_log=self.kong_model.epoch_log)

    def use_rect2(self, first_k3=False, use_res_learning=True, resb_num=9, coord_conv=False, g_train_many=False, D_first_concat=True, D_kernel_size=4):
        self.first_k3 = first_k3
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.coord_conv = coord_conv
        self.g_train_many = g_train_many
        self.D_first_concat = D_first_concat
        self.D_kernel_size = D_kernel_size

        def _build_rect2():
            from step08_a_2_Rect2 import Generator, Discriminator, Rect2
            gen_obj = Generator(first_k3=first_k3, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 建立 Generator物件
            dis_obj = Discriminator(D_first_concat=D_first_concat, D_kernel_size=D_kernel_size)
            self.kong_model.rect = Rect2(gen_obj, dis_obj)   ### 把 Generator物件 丟進 Rect建立 Rect物件
            self._kong_model_GD_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_rect2", "finish")
            return self.kong_model
        self.build = _build_rect2
        return self

    def use_rect2_mrf(self, first_k3=False, mrf_replace=False, use_res_learning=True, resb_num=9, coord_conv=False, use1=False, use3=False, use5=False, use7=False, use9=False, g_train_many=False, D_first_concat=True, D_kernel_size=4):
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
        self.g_train_many = g_train_many
        self.D_first_concat = D_first_concat
        self.D_kernel_size = D_kernel_size

        def _build_rect2_mrf():
            from step08_a_2_Rect2 import MRFBlock, Generator, Discriminator, Rect2
            mrfb    = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
            gen_obj = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)   ### 把 mrf物件 丟進 Generator 建立 Generator物件
            dis_obj = Discriminator(D_first_concat=D_first_concat, D_kernel_size=D_kernel_size)
            self.kong_model.rect = Rect2(gen_obj, dis_obj)   ### 再把 Generator物件 丟進 Rect建立 Rect物件
            self._kong_model_GD_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_rect2_mrf", "finish")
            return self.kong_model
        self.build = _build_rect2_mrf
        return self


class KModel_justG_and_mrf_justG_builder(KModel_GD_and_mrfGD_builder):
    def _kong_model_G_setting(self, g_train_many=False):
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        from step08_b_use_G_generate import generate_img_sees, generate_img_results
        self.kong_model.generate_results = generate_img_results  ### 不能checkpoint
        self.kong_model.generate_sees  = generate_img_sees     ### 不能checkpoint

        from step09_b_train_step import train_step_pure_G
        self.kong_model.train_step = train_step_pure_G           ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator, 
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)

    def use_justG(self, first_k3=False, use_res_learning=True, resb_num=9, coord_conv=False, g_train_many=False):
        self.first_k3 = first_k3
        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.coord_conv = coord_conv
        self.g_train_many = g_train_many

        def _build_justG():
            from step08_a_2_Rect2 import Generator
            self.kong_model.generator   = Generator(first_k3=first_k3, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 建立 Generator物件
            self._kong_model_G_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_justG", "finish")
            return self.kong_model
        self.build = _build_justG
        return self

    def use_justG_mrf(self, first_k3=False, mrf_replace=False, use_res_learning=True, resb_num=9, coord_conv=False, use1=False, use3=False, use5=False, use7=False, use9=False, g_train_many=False):
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
        self.g_train_many = g_train_many

        def _build_justG_mrf():
            from step08_a_2_Rect2 import MRFBlock, Generator
            mrfb = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
            self.kong_model.generator = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 把 mrf物件 丟進 Generator 建立 Generator物件
            self._kong_model_G_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
            print("build_justG_mrf", "finish")
            return self.kong_model

        self.build = _build_justG_mrf
        return self


class KModel_builder(KModel_justG_and_mrf_justG_builder): pass


class MODEL_NAME(Enum):
    unet  = "unet"
    rect  = "rect"
    justG = "justG"
    flow_unet = "flow_unet"   ### 包含這flow 關鍵字就沒問題
    flow_rect = "flow_rect"   ### 包含這flow 關鍵字就沒問題
    mask_flow_unet = "mask_flow_unet"


### 直接先建好 obj 給外面import囉！
unet                     = KModel_builder().set_model_name(MODEL_NAME.unet               ).build_unet()
#######################################################################################################################
rect                     = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=False, g_train_many=False)
rect_firstk3             = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True)
rect_Gk4_many_Dk4_concat = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=False, g_train_many=True)  ### 目前G_train幾次要手動改喔！

rect_mrfall         = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use1=True, use3=True, use5=True, use7=True, use9=True, g_train_many=False)
rect_mrf7           = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use7=True)
rect_mrf79          = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True)
rect_replace_mrf7   = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, g_train_many=False)
rect_replace_mrf79  = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, use9=True)
#######################################################################################################################
justG               = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=False)
justG_firstk3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True)
########################################################### 2
justG_mrf7          = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=False, use7=True)
justG_mrf7_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use7=True)
justG_mrf5_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use5=True)
justG_mrf3_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True)
########################################################### 3
justG_mrf79         = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True)
justG_mrf79_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use7=True, use9=True)
justG_mrf57_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use5=True, use7=True)
justG_mrf35_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True)
########################################################### 4
justG_mrf_replace7  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True)
justG_mrf_replace5  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use5=True)
justG_mrf_replace3  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use3=True)
########################################################### 5
justG_mrf_replace79 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use9=True)
justG_mrf_replace75 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use5=True)
justG_mrf_replace35 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use3=True, use5=True)
########################################################### 2c
justG_mrf135_k3     = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True)
justG_mrf357_k3     = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True)
justG_mrf3579_k3    = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True)

rect_mrf35_Gk3_DnoC_k4     = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, D_first_concat=False, D_kernel_size=4)
rect_mrf135_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True, D_first_concat=False, D_kernel_size=4)
rect_mrf357_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, D_first_concat=False, D_kernel_size=4)
rect_mrf3579_Gk3_DnoC_k4   = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True, D_first_concat=False, D_kernel_size=4)
########################################################### 9a
# rect_D_concat_k4    = "rect_D_concat_k4" ### 原始版本
rect_Gk4_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=True , D_kernel_size=3)
rect_Gk4_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=False, D_kernel_size=4)
rect_Gk4_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=False, D_kernel_size=3)
########################################################### 9b
rect_Gk3_D_concat_k4       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=True , D_kernel_size=4)
rect_Gk3_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=True , D_kernel_size=3)
rect_Gk3_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4)
rect_Gk3_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=3)
########################################################### 10
rect_Gk3_train3_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4, g_train_many=True)
rect_Gk3_train5_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4, g_train_many=True)
########################################################### 11
justG_fk3_no_res             = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=False)  ### 127.51
rect_fk3_no_res_D_no_concat  = KModel_builder().set_model_name(MODEL_NAME.rect  ) .use_rect2(first_k3=True, use_res_learning=False, D_first_concat=False, D_kernel_size=4)  ### 127.28
justG_fk3_no_res_mrf357      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True, mrf_replace=False, use_res_learning=False, use3=True, use5=True, use7=True)  ### 128.246
########################################################### 12
Gk3_resb00  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=0)  ### 127.48
Gk3_resb01  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=1)  ### 127.35
Gk3_resb03  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=3)  ### 127.55
Gk3_resb05  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=5)  ### 128.246
Gk3_resb07  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=7)  ### 127.28
Gk3_resb09  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=9)  ### 127.51
Gk3_resb11  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=11)  ### 127.51
Gk3_resb15  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=15)  ### 127.28
Gk3_resb20  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=20)  ### 127.51

########################################################### 13 加coord_conv試試看
justGk3_coord_conv        = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG    (first_k3=True, coord_conv=True)
justGk3_mrf357_coord_conv = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True, coord_conv=True, mrf_replace=False, use3=True, use5=True, use7=True)


###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 14 快接近IN了
flow_unet       = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, out_ch=3).use_flow_unet()

### 在 git commit bc25e 之前都是 old ch 喔！ 最大都是 32*8=256, 16*8=128, 8*8=64 而已， 而128*8=1024 又有點太大， new_ch 就是根據層 做 2**layer，最大取512 囉！
### 如果想回復的話，要用 git 回復到 bc25e 或 之前的版本囉！
flow_unet_old_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, out_ch=3).use_flow_unet()
flow_unet_old_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, out_ch=3).use_flow_unet()
flow_unet_old_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, out_ch=3).use_flow_unet()
flow_unet_old_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 8 , out_ch=3).use_flow_unet()

########################################################### 14 測試 subprocess
flow_unet_epoch2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=4, out_ch=3).use_flow_unet()
flow_unet_epoch3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=6, out_ch=3).use_flow_unet()
flow_unet_epoch4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=8, out_ch=3).use_flow_unet()

########################################################### 14 真的IN
flow_unet_IN_ch64 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True).use_flow_unet()


flow_unet_IN_new_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, true_IN=True).use_flow_unet()
flow_unet_IN_new_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True).use_flow_unet()
flow_unet_IN_new_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, true_IN=True).use_flow_unet()
flow_unet_IN_new_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  8, true_IN=True).use_flow_unet()
flow_unet_IN_new_ch004 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  4, true_IN=True).use_flow_unet()

########################################################### 14 真的IN，跟DewarpNet一樣 CNN 不用 bias
flow_unet_IN_ch64_cnnNoBias = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, cnn_bias=False).use_flow_unet()


########################################################### 14 看 concat Activation 有沒有差
flow_unet_ch64_in_concat_A = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, out_ch=3, concat_Activation=True).use_flow_unet()

########################################################### 14 看 不同level 的效果
flow_unet_2_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, out_ch=3).use_flow_unet()
flow_unet_3_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, out_ch=3).use_flow_unet()
flow_unet_4_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, out_ch=3).use_flow_unet()
flow_unet_5_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, out_ch=3).use_flow_unet()
flow_unet_6_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, out_ch=3).use_flow_unet()
flow_unet_7_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, out_ch=3).use_flow_unet()
flow_unet_8_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, out_ch=3).use_flow_unet()
########################################################### 14 看 unet 的 concat 改成 + 會有什麼影響
flow_unet_8_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()
flow_unet_7_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()
flow_unet_6_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()
flow_unet_5_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()
flow_unet_4_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()
flow_unet_3_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()
flow_unet_2_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet()

########################################################### 14 看 unet 的 output 改成sigmoid
flow_unet_IN_ch64_sigmoid  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, out_tanh=False, true_IN=True).use_flow_unet()
########################################################### 14 看 unet 的 第一層試試看 不 concat 效果如何
flow_unet_IN_7l_ch64_2to2noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2).use_flow_unet()
flow_unet_IN_7l_ch32_2to2noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=2).use_flow_unet()
flow_unet_IN_7l_ch64_2to3noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=3).use_flow_unet()
flow_unet_IN_7l_ch64_2to4noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=4).use_flow_unet()
flow_unet_IN_7l_ch64_2to5noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=5).use_flow_unet()
flow_unet_IN_7l_ch64_2to6noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=6).use_flow_unet()
flow_unet_IN_7l_ch64_2to7noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=7).use_flow_unet()
flow_unet_IN_7l_ch64_2to8noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=8).use_flow_unet()
########################################################### 14 看 unet 的 skip 中間接 cnn 的效果
flow_unet_IN_7l_ch64_skip_use_cnn1_NO_relu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=None).use_flow_unet()
flow_unet_IN_7l_ch64_skip_use_cnn1_USErelu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=tf.nn.relu).use_flow_unet()
flow_unet_IN_7l_ch64_skip_use_cnn1_USEsigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=tf.nn.sigmoid).use_flow_unet()
flow_unet_IN_7l_ch64_skip_use_cnn3_USErelu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=3, skip_use_Acti=tf.nn.relu).use_flow_unet()
flow_unet_IN_7l_ch64_skip_use_cnn3_USEsigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=3, skip_use_Acti=tf.nn.sigmoid).use_flow_unet()


########################################################### 14 看 unet 的 skip 中間接 cSE, sSE, csSE 的效果
flow_unet_IN_7l_ch64_2to3noC_sk_cSE  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=3, skip_use_cSE=True).use_flow_unet()
flow_unet_IN_7l_ch64_2to3noC_sk_sSE  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2, skip_use_sSE=True).use_flow_unet()
flow_unet_IN_7l_ch64_2to3noC_sk_scSE = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2, skip_use_scSE=True).use_flow_unet()

flow_unet_IN_7l_ch64_skip_use_cSE    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cSE=True).use_flow_unet()
flow_unet_IN_7l_ch64_skip_use_sSE    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_sSE=True).use_flow_unet()
flow_unet_IN_7l_ch64_skip_use_scSE   = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_scSE=True).use_flow_unet()

###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 15 用 resblock 來試試看
flow_rect_fk3_ch64_tfIN_resb_ok9 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect(first_k3=True, hid_ch=64, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)

flow_rect_7_level_fk7 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)

flow_rect_2_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_3_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_4_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_5_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_6_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_7_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)

flow_rect_2_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_3_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_4_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_5_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_6_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_7_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3)

###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 1 嘗試看看 mask_unet
mask_unet_ch032_tanh    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True, out_ch=1).use_mask_unet()
mask_unet_ch128_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch064_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 64, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch032_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch016_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch008_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  8, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch004_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  4, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch002_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  2, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
mask_unet_ch001_sigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  1, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()


flow_unet_2_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=2, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_3_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=3, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_4_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=4, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_5_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=5, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_6_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=6, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_7_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=7, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_8_level_ch32_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, depth_level=8, true_IN=True, out_tanh=False, out_ch=1).use_mask_unet()


flow_unet_IN_7l_ch32_2to2noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=2, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_IN_7l_ch32_2to3noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=3, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_IN_7l_ch32_2to4noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=4, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_IN_7l_ch32_2to5noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=5, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_IN_7l_ch32_2to6noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=6, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_IN_7l_ch32_2to7noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=7, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_IN_7l_ch32_2to8noC_sig_mask = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=8, out_tanh=False, out_ch=1).use_mask_unet()

flow_unet_8_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_7_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_6_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_5_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_4_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_3_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()
flow_unet_2_level_skip_use_add_sig_mask  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, skip_use_add=True, out_tanh=False, out_ch=1).use_mask_unet()


if(__name__ == "__main__"):
    # print(flow_rect_2_level_fk3.build())
    print(mask_unet_ch032_tanh.build())
    print("build_model cost time:", time.time() - start_time)
    pass
