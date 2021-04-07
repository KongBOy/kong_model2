import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, Conv2DTranspose, Activation
from  tensorflow_addons.layers import InstanceNormalization

from step08_a_2_Rect2 import InstanceNorm_kong, ResBlock
from step08_d_loss_funs_and_train_step import  train_step_pure_G

### 模仿UNet
class Rect_7_layer(tf.keras.models.Model):
    def __init__(self, first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_ReLU=False, use_res_learning=True, resb_num=9, out_tanh=True, out_ch=3, **kwargs):
        """
        depth_level: 0~7
        out_tanh：想實驗看看 output 是 tanh 和 sigmoid 的效果，out_tanh=False 就是用 sigmoid
        """
        super(Rect_7_layer, self).__init__(**kwargs)
        ########################################################################################################################################################
        self.first_k = first_k


        ### 還是想實驗看看 tensorflow_addon 跟自己寫的 IN 有沒有差
        self.true_IN = true_IN
        use_what_IN = InstanceNorm_kong  ### 原本架構使用InstanceNorm_kong，用它來當 default IN
        if(self.true_IN): use_what_IN = InstanceNormalization

        ### 跟 step08_a_2_Rect2 好像就是差 ReLU了
        self.use_ReLU = use_ReLU

        ########################################################################################################################################################
        self.depth_level = depth_level

        self.out_tanh = out_tanh


        self.conv0  = Conv2D(hid_ch, kernel_size=self.first_k, strides=(1, 1), padding="valid")  ### in:x1, 3, out:x1, 64
        self.in0    = use_what_IN()          ### x1, 64
        if(self.use_ReLU):
            self.relud0 = ReLU()
        else:
            self.lrelu0 = LeakyReLU(alpha=0.2)   ### x1, 64

        if(self.depth_level >= 1):
            self.convd1  = Conv2D(hid_ch * 2, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1, 64, out:x1/2, 128
            self.ind1    = use_what_IN()         ### x1/2, 128
            if(self.use_ReLU):
                self.relud1 = ReLU()
            else:
                self.lrelud1 = LeakyReLU(alpha=0.2)  ### x1/2, 128

        if(self.depth_level >= 2):
            self.convd2  = Conv2D(hid_ch * 4, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/2, 128, out:x1/4, 256
            self.ind2    = use_what_IN()         ### x1/4, 256
            if(self.use_ReLU):
                self.relud2 = ReLU()
            else:
                self.lrelud2 = LeakyReLU(alpha=0.2)  ### x1/4, 256

        if(self.depth_level >= 3):
            self.convd3  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/4, 256, out:x1/8, 512
            self.ind3    = use_what_IN()         ### x1/8, 512
            if(self.use_ReLU):
                self.relud3 = ReLU()
            else:
                self.lrelud3 = LeakyReLU(alpha=0.2)  ### x1/8, 512

        if(self.depth_level >= 4):
            self.convd4  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/8, 512, out:x1/16, 512
            self.ind4    = use_what_IN()         ### x1/16, 512
            if(self.use_ReLU):
                self.relud4 = ReLU()
            else:
                self.lrelud4 = LeakyReLU(alpha=0.2)  ### x1/16, 512

        if(self.depth_level >= 5):
            self.convd5  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/16, 512, out:x1/32, 512
            self.ind5    = use_what_IN()         ### x1/32, 512
            if(self.use_ReLU):
                self.relud5 = ReLU()
            else:
                self.lrelud5 = LeakyReLU(alpha=0.2)  ### x1/32, 512

        if(self.depth_level >= 6):
            self.convd6  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/32, 512, out:x1/64, 512
            self.ind6    = use_what_IN()         ### x1/64, 512
            if(self.use_ReLU):
                self.relud6 = ReLU()
            else:
                self.lrelud6 = LeakyReLU(alpha=0.2)  ### x1/64, 512

        if(self.depth_level >= 7):
            self.convd7  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/64, 512, out:x1/128, 512
            self.ind7    = use_what_IN()         ### x1/128, 512
            if(self.use_ReLU):
                self.relud7 = ReLU()
            else:
                self.lrelud7 = LeakyReLU(alpha=0.2)  ### x1/128, 512


        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.resbs = []
        res_ch = 64 * (2 ** self.depth_level)
        if(self.depth_level > 3): res_ch = 64 * 8

        for go_r in range(resb_num): self.resbs.append(ResBlock(c_num=res_ch, use_what_IN=use_what_IN, use_res_learning=self.use_res_learning))


        if(self.depth_level >= 7):
            self.convT7  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/128, 512, out:x1/064, 512
            self.in_cT7  = use_what_IN()  ### x1/064, 512
            self.reluu7  = ReLU()         ### x1/064, 512

        if(self.depth_level >= 6):
            self.convT6  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/064, 512, out:x1/032, 512
            self.in_cT6  = use_what_IN()  ### x1/032, 512
            self.reluu6  = ReLU()         ### x1/032, 512

        if(self.depth_level >= 5):
            self.convT5  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/032, 512, out:x1/016, 512
            self.in_cT5  = use_what_IN()  ### x1/016, 512
            self.reluu5  = ReLU()         ### x1/016, 512

        if(self.depth_level >= 4):
            self.convT4  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/016, 512, out:x1/008, 512
            self.in_cT4  = use_what_IN()  ### x1/008, 512
            self.reluu4  = ReLU()         ### x1/008, 512

        if(self.depth_level >= 3):
            self.convT3  = Conv2DTranspose(filters=hid_ch * 4, kernel_size=3, strides=2, padding="same")  ### in: x1/008, 512, out:x1/004, 256
            self.in_cT3  = use_what_IN()  ### x1/004, 256
            self.reluu3  = ReLU()         ### x1/004, 256

        if(self.depth_level >= 2):
            self.convT2  = Conv2DTranspose(filters=hid_ch * 2, kernel_size=3, strides=2, padding="same")  ### in: x1/004, 256, out:x1/002, 128
            self.in_cT2  = use_what_IN()  ### x1/002, 128
            self.reluu2  = ReLU()         ### x1/002, 128

        if(self.depth_level >= 1):
            self.convT1  = Conv2DTranspose(filters=hid_ch * 1, kernel_size=3, strides=2, padding="same")  ### in: x1/002, 128, out:x1/001, 064
            self.in_cT1  = use_what_IN()  ### x1/001, 064
            self.reluu1  = ReLU()         ### x1/001, 064

        self.convRGB = Conv2D(filters=out_ch  , kernel_size=self.first_k, strides=1, padding="valid")  ### in: x1/001, 064, out:x1/001, out_ch

        if(self.out_tanh): self.tanh    = Activation(tf.nn.tanh)
        else:              self.sigmoid = Activation(tf.nn.sigmoid)

    def call(self, input_tensor, training=None):  ### 這裡的training只是為了介面統一，實際上沒用到喔，因為IN不需要指定 train/test mode

        first_pad_size = int((self.first_k - 1) / 2)
        x = tf.pad(input_tensor, [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")

        ### c0
        x = self.conv0(x)
        x = self.in0(x)
        if(self.use_ReLU):
            x = self.relud0(x)
        else:
            x = self.lrelu0(x)

        ### cd1
        if(self.depth_level >= 1):
            x = self.convd1(x)
            x = self.ind1(x)
            if(self.use_ReLU):
                x = self.relud1(x)
            else:
                x = self.lrelud1(x)

        ### cd2
        if(self.depth_level >= 2):
            x = self.convd2(x)
            x = self.ind2(x)
            if(self.use_ReLU):
                x = self.relud2(x)
            else:
                x = self.lrelud2(x)

        ### cd3
        if(self.depth_level >= 3):
            x = self.convd3(x)
            x = self.ind3(x)
            if(self.use_ReLU):
                x = self.relud3(x)
            else:
                x = self.lrelud3(x)

        ### cd4
        if(self.depth_level >= 4):
            x = self.convd4(x)
            x = self.ind4(x)
            if(self.use_ReLU):
                x = self.relud4(x)
            else:
                x = self.lrelud4(x)

        ### cd5
        if(self.depth_level >= 5):
            x = self.convd5(x)
            x = self.ind5(x)
            if(self.use_ReLU):
                x = self.relud5(x)
            else:
                x = self.lrelud5(x)

        ### cd6
        if(self.depth_level >= 6):
            x = self.convd6(x)
            x = self.ind6(x)
            if(self.use_ReLU):
                x = self.relud6(x)
            else:
                x = self.lrelud6(x)

        ### cd7
        if(self.depth_level >= 7):
            x = self.convd7(x)
            x = self.ind7(x)
            if(self.use_ReLU):
                x = self.relud7(x)
            else:
                x = self.lrelud7(x)

        for go_r in range(self.resb_num):
            x = self.resbs[go_r](x)

        ### ct7
        if(self.depth_level >= 7):
            x = self.convT7(x)
            x = self.in_cT7(x)
            x = self.reluu7(x)

        ### ct6
        if(self.depth_level >= 6):
            x = self.convT6(x)
            x = self.in_cT6(x)
            x = self.reluu6(x)

        ### ct5
        if(self.depth_level >= 5):
            x = self.convT5(x)
            x = self.in_cT5(x)
            x = self.reluu5(x)

        ### ct4
        if(self.depth_level >= 4):
            x = self.convT4(x)
            x = self.in_cT4(x)
            x = self.reluu4(x)

        ### ct3
        if(self.depth_level >= 3):
            x = self.convT3(x)
            x = self.in_cT3(x)
            x = self.reluu3(x)

        ### ct2
        if(self.depth_level >= 2):
            x = self.convT2(x)
            x = self.in_cT2(x)
            x = self.reluu2(x)

        ### ct1
        if(self.depth_level >= 1):
            x = self.convT1(x)
            x = self.in_cT1(x)
            x = self.reluu1(x)

        x = tf.pad(x, [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")
        x_RGB = self.convRGB(x)

        if(self.out_tanh): return self.tanh(x_RGB)
        else:              return self.sigmoid(x_RGB)


if(__name__ == "__main__"):
    import time
    import numpy as np
    generator = Rect_7_layer(out_ch=2)  # 建G
    in_img = np.ones(shape=(1, 768, 768, 3), dtype=np.float32)  # 建 假資料
    gt_img = np.ones(shape=(1, 768, 768, 2), dtype=np.float32)  # 建 假資料
    start_time = time.time()  # 看資料跑一次花多少時間
    y = generator(in_img)
    print(y)
    print("cost time", time.time() - start_time)


    from tqdm import tqdm
    from step06_a_datas_obj import DB_C, DB_N, DB_GM
    from step06_b_data_pipline import Dataset_builder, tf_Data_builder
    from step08_e_model_obj import MODEL_NAME, KModel_builder
    from step08_d_loss_funs_and_train_step import mae_kong
    from step08_c_loss_info_obj import Loss_info_builder

    db_obj = Dataset_builder().set_basic(DB_C.type8_blender_os_book, DB_N.blender_os_hw768 , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_type(in_type="png", gt_type="knpy", see_type=None).set_detail(have_train=True, have_see=True).build()
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_rect).build_flow_rect_7_level()
    tf_data = tf_Data_builder().set_basic(db_obj, 1 , train_shuffle=False).set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    loss_info_obj = Loss_info_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_by_model_name(model_obj.model_name).build()  ###step3 建立tensorboard，只有train 和 train_reload需要
    # ###     step2 訓練
    for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_fun=mae_kong, loss_info_obj=loss_info_obj)
