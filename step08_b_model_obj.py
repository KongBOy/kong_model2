from enum import Enum
import tensorflow as tf

class KModel:
    def __init__(self):### 共通有的 元件，其實這邊只留model_name好像也可以
        self.model_name = None
        self.epoch_log = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
        
    def __str__(self):
        print("model_name:", self.model_name)
        print("generator:", self.__dict__)
        return ""

class KModel_init_builder:
    def __init__(self, kong_model=None):
        if(kong_model is None):self.kong_model = KModel()
        else: self.kong_model = kong_model

    def set_model_name(self, model_name):
        self.kong_model.model_name = model_name
        return self

    def build(self):
        return self.kong_model

class KModel_Unet_builder(KModel_init_builder):
    def build_unet(self):
        from step08_a_1_UNet_512to256 import Generator512to256, generate_sees, generate_images, train_step
        self.kong_model.generator           = Generator512to256(out_channel=2)
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.max_train_move = tf.Variable(1) ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
        self.kong_model.min_train_move = tf.Variable(1) ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
        self.kong_model.max_db_move_x  = tf.Variable(1) ### 在test時 rec_img需要，所以需要存起來
        self.kong_model.max_db_move_y  = tf.Variable(1) ### 在test時 rec_img需要，所以需要存起來

        self.kong_model.generate_images = generate_images ### 不能checkpoint
        self.kong_model.generate_sees   = generate_sees   ### 不能checkpoint
        self.kong_model.train_step      = train_step      ### 不能checkpoint
        
        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator = self.kong_model.generator, 
                                                   optimizer_G = self.kong_model.optimizer_G, 
                                                   max_train_move = self.kong_model.max_train_move,
                                                   min_train_move = self.kong_model.min_train_move,
                                                   max_db_move_x  = self.kong_model.max_db_move_x,
                                                   max_db_move_y  = self.kong_model.max_db_move_y,
                                                   epoch_log = self.kong_model.epoch_log)
        return self.kong_model
    

class KModel_GD_and_mrfGD_builder(KModel_Unet_builder):                                                   
    def build_rect2(self, first_k3=False, g_train_many=False):
        from step08_a_2_Rect2 import Generator, Rect2
        gen_obj = Generator(first_k3=first_k3)  ### 建立 Generator物件
        self.kong_model.rect = Rect2(gen_obj)   ### 把 Generator物件 丟進 Rect建立 Rect物件
        self._kong_model_GD_setting(g_train_many=g_train_many) ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def build_rect2_mrf(self, first_k3=False, mrf_replace=False, use1=False, use3=False, use5=False, use7=False, use9=False, g_train_many=False):
        from step08_a_2_Rect2 import MRFBlock, Generator, Rect2
        mrfb = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9) ### 先建立 mrf物件
        gen_obj = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace)   ### 把 mrf物件 丟進 Generator 建立 Generator物件

        self.kong_model.rect = Rect2(gen_obj)   ### 再把 Generator物件 丟進 Rect建立 Rect物件 
        self._kong_model_GD_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model
    
    def _kong_model_GD_setting(self,g_train_many=False):
        self.kong_model.generator = self.kong_model.rect.generator
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        from step08_a_2_Rect2 import generate_sees, generate_images
        self.kong_model.generate_images = generate_images ### 不能checkpoint
        self.kong_model.generate_sees  = generate_sees    ### 不能checkpoint
        from step08_a_2_Rect2 import train_step, train_step2
        if  (g_train_many): self.kong_model.train_step = train_step2  ### 不能checkpoint
        else:               self.kong_model.train_step = train_step   ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(rect = self.kong_model.rect, 
                                                   generator = self.kong_model.generator, 
                                                   optimizer_G = self.kong_model.optimizer_G,
                                                   optimizer_D = self.kong_model.optimizer_D,
                                                   epoch_log = self.kong_model.epoch_log)

class KModel_justG_and_mrf_justG_builder(KModel_GD_and_mrfGD_builder):
    def build_justG(self, first_k3=False, g_train_many=False):
        from step08_a_3_justG import Generator, generate_sees, generate_images, train_step
        self.kong_model.generator   = Generator(first_k3=first_k3) ### 建立 Generator物件
        self._kong_model_G_setting(g_train_many=g_train_many) ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def build_justG_mrf(self, first_k3=False, mrf_replace=False, use1=False, use3=False, use5=False, use7=False, use9=False, g_train_many=False):
        from step08_a_2_Rect2 import MRFBlock, Generator
        mrfb = MRFBlock(c_num=64, use1=use1, use3=use3, use5=use5, use7=use7, use9=use9)  ### 先建立 mrf物件
        self.kong_model.generator = Generator(first_k3=first_k3, mrfb=mrfb, mrf_replace=mrf_replace) ### 把 mrf物件 丟進 Generator 建立 Generator物件
        self._kong_model_G_setting(g_train_many=g_train_many)  ### 去把kong_model 剩下的oprimizer, util_method, ckpt 設定完
        return self.kong_model

    def _kong_model_G_setting(self,g_train_many=False):
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        from step08_a_2_Rect2 import generate_sees, generate_images
        self.kong_model.generate_images = generate_images ### 不能checkpoint
        self.kong_model.generate_sees  = generate_sees    ### 不能checkpoint

        from step08_a_3_justG import train_step
        self.kong_model.train_step = train_step           ### 不能checkpoint

        ### 建立 tf 存模型 的物件： checkpoint物件
        self.kong_model.ckpt = tf.train.Checkpoint(generator = self.kong_model.generator, 
                                                   optimizer_G = self.kong_model.optimizer_G,
                                                   epoch_log = self.kong_model.epoch_log)



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
    rect_g_train_many  = "rect_g_train_many"
    #######################################################################################################################
    justG                 = "justG"
    justG_firstk3         = "justG_firstk3"
    ########################################################### 1
    justG_mrf7            = "justG_mrf7"     ### ord finish
    justG_mrf7_k3         = "justG_mrf7_k3"  ### 127.51
    justG_mrf5_k3         = "justG_mrf5_k3"  ### 沒機器
    justG_mrf3_k3         = "justG_mrf3_k3"  ### 沒機器  
    ########################################################### 2
    justG_mrf79           = "justG_mrf79"   ### ord finish
    justG_mrf79_k3        = "justG_mrf79"   ### 128.246
    justG_mrf57_k3        = "justG_mrf57"   ### 沒機器
    justG_mrf35_k3        = "justG_mrf35"   ### 沒機器
    ########################################################### 3
    justG_mrf_replace7    = "justG_mrf_replace7"   ### ord finish
    justG_mrf_replace5    = "justG_mrf_replace5"   ### 127.35
    justG_mrf_replace3    = "justG_mrf_replace3"   ### 127.48
    ########################################################### 4
    justG_mrf_replace79   = "justG_mrf_replace79"  ### ord finish
    justG_mrf_replace75   = "justG_mrf_replace79"  ### 127.55
    justG_mrf_replace35   = "justG_mrf_replace79"  ### 127.28


    justG_g_train_many    = "justG_g_train_many"


### 直接先建好 obj 給外面import囉！
unet                = KModel_builder().set_model_name(MODEL_NAME.unet               ).build_unet()
#######################################################################################################################
rect                = KModel_builder().set_model_name(MODEL_NAME.rect               ).build_rect2(first_k3=False, g_train_many=False)
rect_firstk3        = KModel_builder().set_model_name(MODEL_NAME.rect               ).build_rect2    (first_k3=True )
rect_g_train_many   = KModel_builder().set_model_name(MODEL_NAME.rect               ).build_rect2    (first_k3=False, g_train_many=True) ### 目前G_train幾次要手動改喔！

rect_mrfall         = KModel_builder().set_model_name(MODEL_NAME.rect_mrfall        ).build_rect2_mrf(first_k3=False, mrf_replace=False, use1=True, use3=True, use5=True, use7=True, use9=True, g_train_many=False)
rect_mrf7           = KModel_builder().set_model_name(MODEL_NAME.rect_mrf7          ).build_rect2_mrf(first_k3=False, mrf_replace=False, use7=True)
rect_mrf79          = KModel_builder().set_model_name(MODEL_NAME.rect_mrf79         ).build_rect2_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True)
rect_replace_mrf7   = KModel_builder().set_model_name(MODEL_NAME.rect_mrf_replace7  ).build_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, g_train_many=False)
rect_replace_mrf79  = KModel_builder().set_model_name(MODEL_NAME.rect_mrf_replace79 ).build_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, use9=True)
#######################################################################################################################
justG               = KModel_builder().set_model_name(MODEL_NAME.justG              ).build_justG(first_k3=False)
justG_firstk3       = KModel_builder().set_model_name(MODEL_NAME.justG              ).build_justG(first_k3=True)
########################################################### 1
justG_mrf7          = KModel_builder().set_model_name(MODEL_NAME.justG_mrf7         ).build_justG_mrf(first_k3=False, mrf_replace=False, use7=True)
justG_mrf7_k3       = KModel_builder().set_model_name(MODEL_NAME.justG_mrf7_k3      ).build_justG_mrf(first_k3=True , mrf_replace=False, use7=True)
justG_mrf5_k3       = KModel_builder().set_model_name(MODEL_NAME.justG_mrf5_k3      ).build_justG_mrf(first_k3=True , mrf_replace=False, use5=True)
########################################################### 2
justG_mrf79         = KModel_builder().set_model_name(MODEL_NAME.justG_mrf79        ).build_justG_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True)
justG_mrf79_k3      = KModel_builder().set_model_name(MODEL_NAME.justG_mrf79_k3     ).build_justG_mrf(first_k3=True , mrf_replace=False, use7=True, use9=True)
########################################################### 3
justG_mrf_replace7  = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace7 ).build_justG_mrf(first_k3=False, mrf_replace=True, use7=True)
justG_mrf_replace5  = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace5 ).build_justG_mrf(first_k3=False, mrf_replace=True, use5=True)
justG_mrf_replace3  = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace3 ).build_justG_mrf(first_k3=False, mrf_replace=True, use3=True)
########################################################### 4
justG_mrf_replace79 = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace79).build_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use9=True)
justG_mrf_replace75 = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace75).build_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use5=True)
justG_mrf_replace35 = KModel_builder().set_model_name(MODEL_NAME.justG_mrf_replace35).build_justG_mrf(first_k3=False, mrf_replace=True, use3=True, use5=True)

if(__name__=="__main__"):
    pass