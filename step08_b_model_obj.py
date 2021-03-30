from enum import Enum
import tensorflow as tf

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

    def set_model_name(self, model_name):
        self.kong_model.model_name = model_name
        return self

    def build(self):
        return self.kong_model

class KModel_Unet_builder(KModel_init_builder):
    def build_unet(self):
        from step08_a_1_UNet_512to256 import Generator512to256, generate_sees, generate_results, train_step
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
        return self.kong_model

class KModel_Flow_Generator_builder(KModel_Unet_builder):
    def _build_flow_part(self):
        ### 
        from step08_a_4_Flow_UNet import train_step, generate_results, generate_sees_without_rec
        self.kong_model.generate_results = generate_results             ### 不能checkpoint
        self.kong_model.generate_sees    = generate_sees_without_rec    ### 不能checkpoint
        self.kong_model.train_step       = train_step                   ### 不能checkpoint

    def _build_ckpt_part(self):
        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)

    def build_flow_unet(self, hid_ch=64, depth_level=7, out_ch=3, true_IN=False, concat_Activation=False):
        ### model_part
        if(true_IN): from step08_a_1_UNet_IN   import Generator
        else:        from step08_a_1_UNet      import Generator  #generate_sees, generate_results, train_step
        if(concat_Activation): from step08_a_1_UNet_IN_concat_Activation import Generator
        self.kong_model.generator   = Generator(hid_ch=hid_ch, depth_level=depth_level, out_ch=out_ch)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


        self._build_flow_part()
        self._build_ckpt_part()
        return self.kong_model

    def build_flow_rect(self, first_k3=False, hid_ch=64, true_IN=True, mrfb=None, mrf_replace=False, coord_conv=False, use_res_learning=True, resb_num=9, out_ch=3):
        ### model_part
        from step08_a_5_Flow_Rect import Generator
        self.kong_model.generator   = Generator(first_k3=first_k3, hid_ch=hid_ch, true_IN=true_IN, mrfb=mrfb, mrf_replace=mrf_replace, coord_conv=coord_conv, use_res_learning=use_res_learning, resb_num=resb_num, out_ch=out_ch)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


        self._build_flow_part()
        self._build_ckpt_part()
        return self.kong_model

    def build_flow_rect_7_level(self, first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3):
        ### model_part
        from step08_a_5_Flow_Rect_7_level import Rect_7_layer as Generator
        self.kong_model.generator   = Generator(first_k=first_k, hid_ch=hid_ch, depth_level=depth_level, true_IN=true_IN, use_res_learning=use_res_learning, resb_num=resb_num, out_ch=out_ch)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


        self._build_flow_part()
        self._build_ckpt_part()
        return self.kong_model

class KModel_GD_and_mrfGD_builder(KModel_Flow_Generator_builder):
    def build_rect2(self, first_k3=False, use_res_learning=True, resb_num=9, coord_conv=False, g_train_many=False, D_first_concat=True, D_kernel_size=4):
        from step08_a_2_Rect2 import Generator, Discriminator, Rect2
        gen_obj = Generator(first_k3=first_k3, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 建立 Generator物件
        dis_obj = Discriminator(D_first_concat=D_first_concat, D_kernel_size=D_kernel_size)
        self.kong_model.rect = Rect2(gen_obj, dis_obj)   ### 把 Generator物件 丟進 Rect建立 Rect物件
        self._kong_model_GD_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def build_rect2_mrf(self, first_k3=False, mrf_replace=False, use_res_learning=True, resb_num=9, coord_conv=False, use1=False, use3=False, use5=False, use7=False, use9=False, g_train_many=False, D_first_concat=True, D_kernel_size=4):
        from step08_a_2_Rect2 import MRFBlock, Generator, Discriminator, Rect2
        mrfb = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
        gen_obj = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)   ### 把 mrf物件 丟進 Generator 建立 Generator物件
        dis_obj = Discriminator(D_first_concat=D_first_concat, D_kernel_size=D_kernel_size)
        self.kong_model.rect = Rect2(gen_obj, dis_obj)   ### 再把 Generator物件 丟進 Rect建立 Rect物件
        self._kong_model_GD_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def _kong_model_GD_setting(self, g_train_many=False):
        self.kong_model.generator = self.kong_model.rect.generator
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        from step08_a_2_Rect2 import generate_sees, generate_results
        self.kong_model.generate_results = generate_results  ### 不能checkpoint
        self.kong_model.generate_sees  = generate_sees     ### 不能checkpoint
        from step08_a_2_Rect2 import train_step, train_step2
        if  (g_train_many): self.kong_model.train_step = train_step2  ### 不能checkpoint
        else:               self.kong_model.train_step = train_step   ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(rect=self.kong_model.rect,
                                                   generator=self.kong_model.generator,
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   optimizer_D=self.kong_model.optimizer_D,
                                                   epoch_log=self.kong_model.epoch_log)

class KModel_justG_and_mrf_justG_builder(KModel_GD_and_mrfGD_builder):
    def build_justG(self, first_k3=False, use_res_learning=True, resb_num=9, coord_conv=False, g_train_many=False):
        from step08_a_3_justG import Generator
        self.kong_model.generator   = Generator(first_k3=first_k3, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 建立 Generator物件
        self._kong_model_G_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def build_justG_mrf(self, first_k3=False, mrf_replace=False, use_res_learning=True, resb_num=9, coord_conv=False, use1=False, use3=False, use5=False, use7=False, use9=False, g_train_many=False):
        from step08_a_2_Rect2 import MRFBlock, Generator
        mrfb = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
        self.kong_model.generator = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace, use_res_learning=use_res_learning, resb_num=resb_num, coord_conv=coord_conv)  ### 把 mrf物件 丟進 Generator 建立 Generator物件
        self._kong_model_G_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def _kong_model_G_setting(self, g_train_many=False):
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        from step08_a_2_Rect2 import generate_sees, generate_results
        self.kong_model.generate_results = generate_results  ### 不能checkpoint
        self.kong_model.generate_sees  = generate_sees     ### 不能checkpoint

        from step08_a_3_justG import train_step
        self.kong_model.train_step = train_step           ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator=self.kong_model.generator, 
                                                   optimizer_G=self.kong_model.optimizer_G,
                                                   epoch_log=self.kong_model.epoch_log)



class KModel_builder(KModel_justG_and_mrf_justG_builder): pass


class MODEL_NAME(Enum):
    unet     = "unet"
    #######################################################################################################################
    rect     = "rect"
    rect_firstk3       = "rect_firstk3"
    rect_mrfall        = "rect_mrfall"
    rect_mrf7          = "rect_mrf7"
    rect_mrf79         = "rect_mrf79"
    rect_mrf_replace7  = "rect_mrf_replace7"
    rect_mrf_replace79 = "rect_mrf_replace79"
    #######################################################################################################################
    justG                 = "justG"
    justG_firstk3         = "justG_firstk3"
    ########################################################### 08b_2
    justG_mrf7            = "justG_mrf7"     ### ord finish
    justG_mrf7_k3         = "justG_mrf7_k3"  ### 127.51
    justG_mrf5_k3         = "justG_mrf5_k3"  ### 沒機器
    justG_mrf3_k3         = "justG_mrf3_k3"  ### 沒機器
    ########################################################### 08b_3
    justG_mrf79           = "justG_mrf79"   ### ord finish
    justG_mrf79_k3        = "justG_mrf79"   ### 128.246
    justG_mrf57_k3        = "justG_mrf57"   ### 沒機器
    justG_mrf35_k3        = "justG_mrf35"   ### 沒機器
    ########################################################### 08b_4
    justG_mrf_replace7    = "justG_mrf_replace7"   ### ord finish
    justG_mrf_replace5    = "justG_mrf_replace5"   ### 127.35
    justG_mrf_replace3    = "justG_mrf_replace3"   ### 127.48
    ########################################################### 08b_5
    justG_mrf_replace79   = "justG_mrf_replace79"  ### ord finish
    justG_mrf_replace75   = "justG_mrf_replace75"  ### 127.55
    justG_mrf_replace35   = "justG_mrf_replace35"  ### 127.28

    ########################################################### 08c
    justG_mrf135_k3       = "justG_mrf135"   ### 128.246
    justG_mrf357_k3       = "justG_mrf357"   ### 127.51
    justG_mrf3579_k3      = "justG_mrf357"  ### 127.28

    ########################################################### 08d
    rect_mrf35_Gk3_DnoC_k4        = "rect_mrf35_Gk3_DnoC_k4"    ### 127.55
    rect_mrf135_Gk3_DnoC_k4       = "rect_mrf135_Gk3_DnoC_k4"   ### 128.246
    rect_mrf357_Gk3_DnoC_k4       = "rect_mrf357_Gk3_DnoC_k4"   ### 127.51
    rect_mrf3579_Gk3_DnoC_k4      = "rect_mrf3579_Gk3_DnoC_k4"  ### 127.28


    ########################################################### 9a
    # rect_D_concat_k4    = "rect_D_concat_k4" ### 原始版本
    rect_Gk4_D_concat_k3       = "rect_Gk4_D_concat_k3"     ### 127.51
    rect_Gk4_D_no_concat_k4    = "rect_Gk4_D_no_concat_k4"  ### 128.246
    rect_Gk4_D_no_concat_k3    = "rect_Gk4_D_no_concat_k3"  ### 127.28

    ########################################################### 9b
    rect_Gk3_D_concat_k4       = "rect_Gk3_D_concat_k4"  ###
    rect_Gk3_D_concat_k3       = "rect_Gk3_D_concat_k3"     ### 127.51
    rect_Gk3_D_no_concat_k4    = "rect_Gk3_D_no_concat_k4"  ### 128.246
    rect_Gk3_D_no_concat_k3    = "rect_Gk3_D_no_concat_k3"  ### 127.28

    ########################################################### 10
    rect_Gk4_many_Dk4_concat    = "rect_Gk4_many_Dk4_concat"    ### 以前的

    rect_Gk3_train3_Dk4_no_concat = "rect_Gk3_train3_Dk4_no_concat"  ###
    rect_Gk3_train5_Dk4_no_concat = "rect_Gk3_train5_Dk4_no_concat"  ###


    ########################################################### 11
    Gk3_no_res             = "justGk3_no_res"
    Gk3_no_res_D_no_concat = "rect_Gk3_no_res_D_no_concat"
    Gk3_no_res_mrf357      = "justGk3_no_res_mrf357"

    ########################################################### 12
    Gk3_resb00 = "justGk3_resb00"  ### 127.48
    Gk3_resb01 = "justGk3_resb01"  ### 127.35
    Gk3_resb03 = "justGk3_resb03"  ### 127.55
    Gk3_resb05 = "justGk3_resb05"  ### 128.246
    Gk3_resb07 = "justGk3_resb07"  ### 127.28
    Gk3_resb09 = "justGk3_resb09"  ### 127.51
    Gk3_resb11 = "justGk3_resb11"  ### 127.51
    Gk3_resb15 = "justGk3_resb15"  ### 127.51
    Gk3_resb20 = "justGk3_resb20"  ### 127.51

    ########################################################### 13 加coord_conv試試看
    justGk3_coord_conv   = "justGk3_coord_conv"        ### 127.35
    justGk3_mrf357_coord_conv = "justGk3_mrf357_corrd_conv"  ### 127.28

    ########################################################### 14
    flow_unet = "flow_unet"   ### 包含這關鍵字就沒問題 ### hid_ch=64
    flow_unet_epoch2 = "flow_unet_epoch2"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=4
    flow_unet_epoch3 = "flow_unet_epoch3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=6
    flow_unet_epoch4 = "flow_unet_epoch4"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=8

    flow_unet_ch128 = "flow_unet_ch128"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=128
    flow_unet_ch032 = "flow_unet_ch032"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=032
    flow_unet_ch016 = "flow_unet_ch016"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=016
    flow_unet_ch008 = "flow_unet_ch008"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=008

    flow_unet_IN_ch64 = "flow_unet_IN_ch64"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=364

    flow_unet_concat_A = "flow_unet_concat_A"

    ########################################################### 15
    flow_rect_fk3_ch64_tfIN_resb_ok9 = "flow_rect_fk3_ch64_tfIN_resb_ok9"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    flow_rect = "flow_rect"
    flow_rect_7_level = "flow_rect_7_level"
    # flow_rect_7_level_fk7 = "flow_rect_7_level_fk7"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    # flow_rect_2_level_fk3 = "flow_rect_2_level_fk3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    # flow_rect_3_level_fk3 = "flow_rect_3_level_fk3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    # flow_rect_4_level_fk3 = "flow_rect_4_level_fk3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    # flow_rect_5_level_fk3 = "flow_rect_5_level_fk3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    # flow_rect_6_level_fk3 = "flow_rect_6_level_fk3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64
    # flow_rect_7_level_fk3 = "flow_rect_7_level_fk3"   ### 包含這flow_unet關鍵字就沒問題 ### hid_ch=64

### 直接先建好 obj 給外面import囉！
unet                = KModel_builder().set_model_name(MODEL_NAME.unet               ).build_unet()
#######################################################################################################################
rect                = KModel_builder().set_model_name(MODEL_NAME.rect               ).build_rect2(first_k3=False, g_train_many=False)
rect_firstk3        = KModel_builder().set_model_name(MODEL_NAME.rect               ).build_rect2(first_k3=True)
rect_Gk4_many_Dk4_concat   = KModel_builder().set_model_name(MODEL_NAME.rect        ).build_rect2(first_k3=False, g_train_many=True)  ### 目前G_train幾次要手動改喔！

rect_mrfall         = KModel_builder().set_model_name(MODEL_NAME.rect_mrfall        ).build_rect2_mrf(first_k3=False, mrf_replace=False, use1=True, use3=True, use5=True, use7=True, use9=True, g_train_many=False)
rect_mrf7           = KModel_builder().set_model_name(MODEL_NAME.rect_mrf7          ).build_rect2_mrf(first_k3=False, mrf_replace=False, use7=True)
rect_mrf79          = KModel_builder().set_model_name(MODEL_NAME.rect_mrf79         ).build_rect2_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True)
rect_replace_mrf7   = KModel_builder().set_model_name(MODEL_NAME.rect_mrf_replace7  ).build_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, g_train_many=False)
rect_replace_mrf79  = KModel_builder().set_model_name(MODEL_NAME.rect_mrf_replace79 ).build_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, use9=True)
#######################################################################################################################
justG               = KModel_builder().set_model_name(MODEL_NAME.justG              ).build_justG(first_k3=False)
justG_firstk3       = KModel_builder().set_model_name(MODEL_NAME.justG              ).build_justG(first_k3=True)
########################################################### 2
justG_mrf7          = KModel_builder().set_model_name(MODEL_NAME.justG_mrf7         ).build_justG_mrf(first_k3=False, mrf_replace=False, use7=True)
justG_mrf7_k3       = KModel_builder().set_model_name(MODEL_NAME.justG_mrf7_k3      ).build_justG_mrf(first_k3=True , mrf_replace=False, use7=True)
justG_mrf5_k3       = KModel_builder().set_model_name(MODEL_NAME.justG_mrf5_k3      ).build_justG_mrf(first_k3=True , mrf_replace=False, use5=True)
justG_mrf3_k3       = KModel_builder().set_model_name(MODEL_NAME.justG_mrf3_k3      ).build_justG_mrf(first_k3=True , mrf_replace=False, use3=True)
########################################################### 3
justG_mrf79         = KModel_builder().set_model_name(MODEL_NAME.justG_mrf79        ).build_justG_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True)
justG_mrf79_k3      = KModel_builder().set_model_name(MODEL_NAME.justG_mrf79_k3     ).build_justG_mrf(first_k3=True , mrf_replace=False, use7=True, use9=True)
justG_mrf57_k3      = KModel_builder().set_model_name(MODEL_NAME.justG_mrf57_k3     ).build_justG_mrf(first_k3=True , mrf_replace=False, use5=True, use7=True)
justG_mrf35_k3      = KModel_builder().set_model_name(MODEL_NAME.justG_mrf35_k3     ).build_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True)
########################################################### 4
justG_mrf_replace7  = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace7 ).build_justG_mrf(first_k3=False, mrf_replace=True, use7=True)
justG_mrf_replace5  = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace5 ).build_justG_mrf(first_k3=False, mrf_replace=True, use5=True)
justG_mrf_replace3  = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace3 ).build_justG_mrf(first_k3=False, mrf_replace=True, use3=True)
########################################################### 5
justG_mrf_replace79 = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace79).build_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use9=True)
justG_mrf_replace75 = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace75).build_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use5=True)
justG_mrf_replace35 = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace35).build_justG_mrf(first_k3=False, mrf_replace=True, use3=True, use5=True)

########################################################### 2c
justG_mrf135_k3     = KModel_builder().set_model_name(MODEL_NAME.justG_mrf357_k3    ).build_justG_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True)
justG_mrf357_k3     = KModel_builder().set_model_name(MODEL_NAME.justG_mrf357_k3    ).build_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True)
justG_mrf3579_k3    = KModel_builder().set_model_name(MODEL_NAME.justG_mrf3579_k3   ).build_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True)

rect_mrf35_Gk3_DnoC_k4     = KModel_builder().set_model_name(MODEL_NAME.rect_mrf35_Gk3_DnoC_k4  ).build_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, D_first_concat=False, D_kernel_size=4)
rect_mrf135_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect_mrf135_Gk3_DnoC_k4 ).build_rect2_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True, D_first_concat=False, D_kernel_size=4)
rect_mrf357_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect_mrf357_Gk3_DnoC_k4 ).build_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, D_first_concat=False, D_kernel_size=4)
rect_mrf3579_Gk3_DnoC_k4   = KModel_builder().set_model_name(MODEL_NAME.rect_mrf3579_Gk3_DnoC_k4).build_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True, D_first_concat=False, D_kernel_size=4)

########################################################### 9a
# rect_D_concat_k4    = "rect_D_concat_k4" ### 原始版本
rect_Gk4_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect_Gk4_D_concat_k3).build_rect2   (D_first_concat=True , D_kernel_size=3)
rect_Gk4_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect_Gk4_D_no_concat_k4).build_rect2(D_first_concat=False, D_kernel_size=4)
rect_Gk4_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect_Gk4_D_no_concat_k3).build_rect2(D_first_concat=False, D_kernel_size=3)
########################################################### 9b
rect_Gk3_D_concat_k4       = KModel_builder().set_model_name(MODEL_NAME.rect_Gk3_D_concat_k4).build_rect2   (first_k3=True, D_first_concat=True , D_kernel_size=4)
rect_Gk3_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect_Gk3_D_concat_k3).build_rect2   (first_k3=True, D_first_concat=True , D_kernel_size=3)
rect_Gk3_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect_Gk3_D_no_concat_k4).build_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4)
rect_Gk3_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect_Gk3_D_no_concat_k3).build_rect2(first_k3=True, D_first_concat=False, D_kernel_size=3)

########################################################### 10
rect_Gk3_train3_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect_Gk3_train3_Dk4_no_concat).build_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4, g_train_many=True)
rect_Gk3_train5_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect_Gk3_train5_Dk4_no_concat).build_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4, g_train_many=True)

########################################################### 11
Gk3_no_res             = KModel_builder().set_model_name(MODEL_NAME.Gk3_no_res)            .build_justG(first_k3=True, use_res_learning=False)  ### 127.51
Gk3_no_res_D_no_concat = KModel_builder().set_model_name(MODEL_NAME.Gk3_no_res_D_no_concat).build_rect2(first_k3=True, use_res_learning=False, D_first_concat=False, D_kernel_size=4)  ### 127.28
Gk3_no_res_mrf357      = KModel_builder().set_model_name(MODEL_NAME.Gk3_no_res_mrf357)     .build_justG_mrf(first_k3=True, mrf_replace=False, use_res_learning=False, use3=True, use5=True, use7=True)  ### 128.246


########################################################### 12
Gk3_resb00  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb00).build_justG(first_k3=True, use_res_learning=True, resb_num=0)  ### 127.48
Gk3_resb01  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb01).build_justG(first_k3=True, use_res_learning=True, resb_num=1)  ### 127.35
Gk3_resb03  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb03).build_justG(first_k3=True, use_res_learning=True, resb_num=3)  ### 127.55
Gk3_resb05  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb05).build_justG(first_k3=True, use_res_learning=True, resb_num=5)  ### 128.246
Gk3_resb07  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb07).build_justG(first_k3=True, use_res_learning=True, resb_num=7)  ### 127.28
Gk3_resb09  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb09).build_justG(first_k3=True, use_res_learning=True, resb_num=9)  ### 127.51
Gk3_resb11  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb11).build_justG(first_k3=True, use_res_learning=True, resb_num=11)  ### 127.51
Gk3_resb15  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb15).build_justG(first_k3=True, use_res_learning=True, resb_num=15)  ### 127.28
Gk3_resb20  = KModel_builder().set_model_name(MODEL_NAME.Gk3_resb20).build_justG(first_k3=True, use_res_learning=True, resb_num=20)  ### 127.51

########################################################### 13 加coord_conv試試看
justGk3_coord_conv        = KModel_builder().set_model_name(MODEL_NAME.justG              ).build_justG    (first_k3=True, coord_conv=True)
justGk3_mrf357_coord_conv = KModel_builder().set_model_name(MODEL_NAME.justG_mrf357_k3    ).build_justG_mrf(first_k3=True, coord_conv=True, mrf_replace=False, use3=True, use5=True, use7=True)

########################################################### 14 快接近IN了
flow_unet = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, out_ch=3)
flow_unet_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=128, out_ch=3)
flow_unet_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch= 32, out_ch=3)
flow_unet_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch= 16, out_ch=3)
flow_unet_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch= 8 , out_ch=3)

########################################################### 14 真的IN
flow_unet_IN_ch64 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, out_ch=3, true_IN=True)


########################################################### 14 測試 subprocess
flow_unet_epoch2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=4, out_ch=3)
flow_unet_epoch3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=6, out_ch=3)
flow_unet_epoch4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=8, out_ch=3)

########################################################### 14 測試 subprocess
flow_unet_concat_A = KModel_builder().set_model_name(MODEL_NAME.flow_unet_concat_A).build_flow_unet(hid_ch=64, out_ch=3, true_IN=True, concat_Activation=True)

flow_unet_2_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, depth_level=2,  out_ch=3, true_IN=True)
flow_unet_3_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, depth_level=3,  out_ch=3, true_IN=True)
flow_unet_4_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, depth_level=4,  out_ch=3, true_IN=True)
flow_unet_5_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, depth_level=5,  out_ch=3, true_IN=True)
flow_unet_6_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, depth_level=6,  out_ch=3, true_IN=True)
flow_unet_7_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, depth_level=7,  out_ch=3, true_IN=True)



########################################################### 15 用 resblock 來試試看
flow_rect_fk3_ch64_tfIN_resb_ok9 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_fk3_ch64_tfIN_resb_ok9).build_flow_rect(first_k3=True, hid_ch=64, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)

flow_rect_7_level_fk7 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)

flow_rect_2_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_3_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_4_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_5_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_6_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)
flow_rect_7_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3)

if(__name__ == "__main__"):
    pass
