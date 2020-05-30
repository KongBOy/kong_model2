from enum import Enum
import tensorflow as tf

class MODEL_NAME(Enum):
    unet     = "unet"
    rect     = "rect"
    mrf_rect = "mrf_rect"
    just_G   = "just_G"

class KModel:
    def __init__(self):### 共通有的 元件，其實這邊只留model_name好像也可以
        self.model_name = None
        self.epoch_log = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
        # self.generate_images = None
        # self.train_step = None
        # self.generator  = None
        # self.optimizer_G = None
        
    def __str__(self):
        print("model_name:", self.model_name)
        print("generator:", self.__dict__)
        return ""

class KModel_init_builder:
    def __init__(self, kong_model=None):
        if(kong_model is None):self.kong_model = KModel()
        else: self.kong_model = kong_model

    def build(self):
        return self.kong_model

    def set_model_name(self, model_name):
        self.kong_model.model_name = model_name
        return self


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
        self.kong_model.generate_sees   = generate_sees
        self.kong_model.train_step      = train_step
        return self
    

class KModel_rect_builder(KModel_Unet_builder):
    def build_rect(self):
        from step08_a_2_Rect2 import Rect2, generate_sees, generate_images, train_step
        self.kong_model.rect = Rect2() ### 只有這行跟mrf_rect不一樣，剩下都一樣喔！
        self.kong_model.generator       = self.kong_model.rect.generator
        self.kong_model.optimizer_G     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.generate_images = generate_images
        self.kong_model.generate_sees   = generate_sees
        self.kong_model.train_step = train_step
        return self

class KModel_mrf_rect_builder(KModel_rect_builder):
    def build_mrf_rect(self):
        from step08_a_2_Rect2 import Rect2, generate_sees, generate_images, train_step
        self.kong_model.rect = Rect2(use_mrfb=True) ### 只有這行跟rect不一樣，剩下都一樣喔！
        self.kong_model.generator = self.kong_model.rect.generator
        self.kong_model.optimizer_G     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.optimizer_D = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.generate_images = generate_images
        self.kong_model.generate_sees  = generate_sees
        self.kong_model.train_step = train_step
        return self

class KModel_just_G_builder(KModel_mrf_rect_builder):
    def build_just_G(self):
        from step08_a_3_just_G import Generator, generate_sees, generate_images, train_step
        self.kong_model.generator           = Generator()
        self.kong_model.optimizer_G = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.kong_model.generate_images = generate_images ### 不能checkpoint
        self.kong_model.generate_sees   = generate_sees
        self.kong_model.train_step      = train_step
        return self


class KModel_builder(KModel_just_G_builder): 
    def build_by_model_name(self):
        if  (self.kong_model.model_name==MODEL_NAME.unet)    :self.build_unet()
        elif(self.kong_model.model_name==MODEL_NAME.rect)    :self.build_rect()
        elif(self.kong_model.model_name==MODEL_NAME.mrf_rect):self.build_mrf_rect()
        elif(self.kong_model.model_name==MODEL_NAME.just_G)  :self.build_just_G()

        ### 不是 tf 的物件無法存進ckpt裡面！ 要先pop出來喔～
        temp = self.kong_model.__dict__.copy()
        temp.pop("model_name")
        temp.pop("generate_images")
        temp.pop("generate_sees")
        temp.pop("train_step")
        self.kong_model.ckpt = tf.train.Checkpoint(**temp)
        return self.kong_model


### 直接先建好 obj 給外面import囉！
unet     = KModel_builder().set_model_name(MODEL_NAME.unet).build_by_model_name()
rect     = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
mrf_rect = KModel_builder().set_model_name(MODEL_NAME.mrf_rect).build_by_model_name()
just_G   = KModel_builder().set_model_name(MODEL_NAME.just_G).build_by_model_name()
if(__name__=="__main__"):
    model_obj = KModel_builder().set_model_name(MODEL_NAME.just_G).build_by_model_name()
    print(model_obj)