import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, ReLU, Conv2DTranspose
from  tensorflow_addons.layers import InstanceNormalization

from step08_a_2_Rect2 import InstanceNorm_kong, ResBlock
from step08_a_4_Flow_UNet import generator_loss, train_step, generate_results, generate_sees_without_rec

### 模仿UNet
class Rect_7_layer(tf.keras.models.Model):
    def __init__(self, hid_ch=64, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3, **kwargs):
        super(Rect_7_layer, self).__init__(**kwargs)
        ########################################################################################################################################################
        ### 還是想實驗看看 tensorflow_addon 跟自己寫的 IN 有沒有差
        self.true_IN = true_IN
        use_what_IN = InstanceNorm_kong  ### 原本架構使用InstanceNorm_kong，用它來當 default IN
        if(self.true_IN): use_what_IN = InstanceNormalization


        self.conv0  = Conv2D(hid_ch, kernel_size=7, strides=(1, 1), padding="valid")  ### in:x1, 3, out:x1, 64
        self.in0    = use_what_IN()          ### x1, 64
        self.lrelu0 = LeakyReLU(alpha=0.2)   ### x1, 64

        self.convd1  = Conv2D(hid_ch * 2, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1, 64, out:x1/2, 128
        self.ind1    = use_what_IN()         ### x1/2, 128
        self.lrelud1 = LeakyReLU(alpha=0.2)  ### x1/2, 128

        self.convd2  = Conv2D(hid_ch * 4, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/2, 128, out:x1/4, 256
        self.ind2    = use_what_IN()         ### x1/4, 256
        self.lrelud2 = LeakyReLU(alpha=0.2)  ### x1/4, 256

        self.convd3  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/4, 256, out:x1/8, 512
        self.ind3    = use_what_IN()         ### x1/8, 512
        self.lrelud3 = LeakyReLU(alpha=0.2)  ### x1/8, 512

        self.convd4  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/8, 512, out:x1/16, 512
        self.ind4    = use_what_IN()         ### x1/16, 512
        self.lrelud4 = LeakyReLU(alpha=0.2)  ### x1/16, 512

        self.convd5  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/16, 512, out:x1/32, 512
        self.ind5    = use_what_IN()         ### x1/32, 512
        self.lrelud5 = LeakyReLU(alpha=0.2)  ### x1/32, 512

        self.convd6  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/32, 512, out:x1/64, 512
        self.ind6    = use_what_IN()         ### x1/64, 512
        self.lrelud6 = LeakyReLU(alpha=0.2)  ### x1/64, 512

        self.convd7  = Conv2D(hid_ch * 8, kernel_size=3, strides=(2, 2), padding="same")  ### in:x1/64, 512, out:x1/128, 512
        self.ind7    = use_what_IN()         ### x1/128, 512
        self.lrelud7 = LeakyReLU(alpha=0.2)  ### x1/128, 512


        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.resbs = []
        for go_r in range(resb_num): self.resbs.append(ResBlock(c_num=hid_ch * 8, use_what_IN=use_what_IN, use_res_learning=self.use_res_learning))



        self.convT7  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/128, 512, out:x1/064, 512
        self.in_cT7  = use_what_IN()  ### x1/064, 512
        self.relud7  = ReLU()         ### x1/064, 512

        self.convT6  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/064, 512, out:x1/032, 512
        self.in_cT6  = use_what_IN()  ### x1/032, 512
        self.relud6  = ReLU()         ### x1/032, 512

        self.convT5  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/032, 512, out:x1/016, 512
        self.in_cT5  = use_what_IN()  ### x1/016, 512
        self.relud5  = ReLU()         ### x1/016, 512

        self.convT4  = Conv2DTranspose(filters=hid_ch * 8, kernel_size=3, strides=2, padding="same")  ### in: x1/016, 512, out:x1/008, 512
        self.in_cT4  = use_what_IN()  ### x1/008, 512
        self.relud4  = ReLU()         ### x1/008, 512

        self.convT3  = Conv2DTranspose(filters=hid_ch * 4, kernel_size=3, strides=2, padding="same")  ### in: x1/008, 512, out:x1/004, 256
        self.in_cT3  = use_what_IN()  ### x1/004, 256
        self.relud3  = ReLU()         ### x1/004, 256

        self.convT2  = Conv2DTranspose(filters=hid_ch * 2, kernel_size=3, strides=2, padding="same")  ### in: x1/004, 256, out:x1/002, 128
        self.in_cT2  = use_what_IN()  ### x1/002, 128
        self.relud2  = ReLU()         ### x1/002, 128

        self.convT1  = Conv2DTranspose(filters=hid_ch * 2, kernel_size=3, strides=2, padding="same")  ### in: x1/002, 128, out:x1/001, 064
        self.in_cT1  = use_what_IN()  ### x1/001, 064
        self.relud1  = ReLU()         ### x1/001, 064

        self.convRGB = Conv2D(filters=out_ch  , kernel_size=7, strides=1, padding="valid")  ### in: x1/001, 064, out:x1/001, out_ch

    def call(self, input_tensor):

        first_pad_size = int((7 - 1) / 2)
        x = tf.pad(input_tensor, [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")

        ### c0
        x = self.conv0(x)
        x = self.in0(x)
        x = self.lrelu0(x)

        ### cd1
        x = self.convd1(x)
        x = self.ind1(x)
        x = self.lrelud1(x)

        ### cd2
        x = self.convd2(x)
        x = self.ind2(x)
        x = self.lrelud2(x)

        ### cd3
        x = self.convd3(x)
        x = self.ind3(x)
        x = self.lrelud3(x)

        ### cd4
        x = self.convd4(x)
        x = self.ind4(x)
        x = self.lrelud4(x)

        ### cd5
        x = self.convd5(x)
        x = self.ind5(x)
        x = self.lrelud5(x)

        ### cd6
        x = self.convd6(x)
        x = self.ind6(x)
        x = self.lrelud6(x)

        ### cd7
        x = self.convd7(x)
        x = self.ind7(x)
        x = self.lrelud7(x)

        for go_r in range(self.resb_num):
            x = self.resbs[go_r](x)

        ### ct7
        x = self.convT7(x)
        x = self.in_cT7(x)
        x = self.relud7(x)

        ### ct6
        x = self.convT6(x)
        x = self.in_cT6(x)
        x = self.relud6(x)

        ### ct5
        x = self.convT5(x)
        x = self.in_cT5(x)
        x = self.relud5(x)

        ### ct4
        x = self.convT4(x)
        x = self.in_cT4(x)
        x = self.relud4(x)

        ### ct3
        x = self.convT3(x)
        x = self.in_cT3(x)
        x = self.relud3(x)

        ### ct2
        x = self.convT2(x)
        x = self.in_cT2(x)
        x = self.relud2(x)

        ### ct1
        x = self.convT1(x)
        x = self.in_cT1(x)
        x = self.relud1(x)

        x = tf.pad(x, [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")
        x_RGB = self.convRGB(x)
        return tf.nn.tanh(x_RGB)


if(__name__ == "__main__"):
    import time
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import DB_C, DB_N, DB_GM
    from step06_b_data_pipline import Dataset_builder, tf_Data_builder
    from step08_b_model_obj import MODEL_NAME, KModel_builder
    from step09_board_obj import Board_builder


    db_obj = Dataset_builder().set_basic(DB_C.type8_blender_os_book, DB_N.blender_os_hw768 , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_type(in_type="png", gt_type="knpy", see_type=None).set_detail(have_train=True, have_see=True).build()
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_rect_7_level).build_flow_rect_7_level()
    tf_data = tf_Data_builder().set_basic(db_obj, 1 , train_shuffle=False).set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    board_obj = Board_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_by_model_name(model_obj.model_name).build()  ###step3 建立tensorboard，只有train 和 train_reload需要
    # ###     step2 訓練
    for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        model_obj.train_step(model_obj, train_in_pre, train_gt_pre, board_obj)

    # generator = Rect_7_layer(out_ch=2)  # 建G
    # in_img = np.ones(shape=(1, 768, 768, 3), dtype=np.float32)  # 建 假資料
    # gt_img = np.ones(shape=(1, 768, 768, 2), dtype=np.float32)  # 建 假資料
    # start_time = time.time()  # 看資料跑一次花多少時間
    # y = generator(in_img)
    # print(y)
    # print("cost time", time.time() - start_time)
