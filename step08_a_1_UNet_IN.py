import sys
sys.path.append("kong_util")
import tensorflow as tf
from  tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Concatenate, Activation
from tensorflow.keras import Sequential
import time



### 參考 DewarpNet 的 train_wc 用的 UNet
### 所有 pytorch BN 裡面有兩個參數的設定不確定～： affine=True, track_running_stats=True，目前思考覺得改道tf2全拿掉也可以
### 目前 總共用7層，所以size縮小 2**7 ，也就是 1/128 這樣子！例如256*256*3丟進去，最中間的feature map長寬2*2*512喔！
class Generator(tf.keras.models.Model):
    def __init__(self, hid_ch=64, depth_level=7, first_concat=True, second_concat=True, skip_use_add=False, skip_use_cnn3_relu=False, out_tanh=True, out_ch=3, **kwargs):
        """
        depth_level: 2~8, 9有點難，因為要是512的倍數，不是512就1024，只能等我研究好512的dataset才有機會式
        skip_use_add：把 concat 改成 用 + 的看看效果如何
        out_tanh：想實驗看看 output 是 tanh 和 sigmoid 的效果，out_tanh=False 就是用 sigmoid
        """
        super(Generator, self).__init__(**kwargs)
        self.depth_level = depth_level
        self.first_concat = first_concat
        self.second_concat = second_concat
        self.skip_use_add = skip_use_add
        self.skip_use_cnn3_relu = skip_use_cnn3_relu
        self.out_tanh = out_tanh

        self.conv1 = Conv2D(hid_ch * 1, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1")  #,bias=False) ### in_channel:3

        if(self.depth_level >= 2):
            self.lrelu2 = LeakyReLU(alpha=0.2, name="lrelu2")
            self.conv2  = Conv2D(hid_ch * 2, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2")  #,bias=False) ### in_channel:64
            if(self.depth_level > 2): self.in2    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            # if(self.skip_use_cnn3_relu): self.skip_cnn3 = Sequential([Conv2D(hid_ch * 1, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1"), ])

        if(self.depth_level >= 3):
            self.lrelu3 = LeakyReLU(alpha=0.2, name="lrelu3")
            self.conv3  = Conv2D(hid_ch * 4, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv3")  #,bias=False) ### in_channel:128
            if(self.depth_level > 3): self.in3    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")

        if(self.depth_level >= 4):
            self.lrelu4 = LeakyReLU(alpha=0.2, name="lrelu4")
            self.conv4  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv4")  #,bias=False) ### in_channel:256
            if(self.depth_level > 4): self.in4    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")

        if(self.depth_level >= 5):
            self.lrelu5 = LeakyReLU(alpha=0.2, name="lrelu5")
            self.conv5  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv5")  #,bias=False) ### in_channel:512
            if(self.depth_level > 5): self.in5    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")

        if(self.depth_level >= 6):
            self.lrelu6 = LeakyReLU(alpha=0.2, name="lrelu6")
            self.conv6  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv6")  #,bias=False) ### in_channel:512
            if(self.depth_level > 6): self.in6    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")

        if(self.depth_level >= 7):
            self.lrelu7  = LeakyReLU(alpha=0.2, name="lrelu7")
            self.conv7   = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv7")  #,bias=False) ### in_channel:512
            if(self.depth_level > 7): self.in7    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")

        if(self.depth_level >= 8):
            self.lrelu8  = LeakyReLU(alpha=0.2, name="lrelu8")
            self.conv8   = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv8")  #,bias=False) ### in_channel:512
            if(self.depth_level > 8): self.in8    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")

        ###################
        # 最底層
        if(self.depth_level >= 9):
            self.lrelu9  = LeakyReLU(alpha=0.2, name="lrelu9")
            self.conv9   = Conv2D(hid_ch * 9, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv9")  #,bias=False) ### in_channel:512


        if(self.depth_level >= 9):
            self.relu9t  = ReLU(name="relu9t")
            self.conv9t  = Conv2DTranspose(hid_ch * 9, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv9t")  #,bias=False) ### in_channel:512
            self.in9t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat9 = Concatenate(name="concat9")

        ###################
        if(self.depth_level >= 8):
            self.relu8t  = ReLU(name="relu8t")
            self.conv8t  = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv8t")  #,bias=False) ### in_channel:512
            self.in8t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat8 = Concatenate(name="concat8")

        if(self.depth_level >= 7):
            self.relu7t  = ReLU(name="relu7t")
            self.conv7t  = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv7t")  #,bias=False) ### in_channel:512
            self.in7t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat7 = Concatenate(name="concat7")

        if(self.depth_level >= 6):
            self.relu6t  = ReLU(name="relu6t")
            self.conv6t  = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv6t")  #,bias=False) ### in_channel:1024
            self.in6t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat6 = Concatenate(name="concat6")

        if(self.depth_level >= 5):
            self.relu5t  = ReLU(name="relu5t")
            self.conv5t  = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv5t")  #,bias=False) ### in_channel:1024
            self.in5t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat5 = Concatenate(name="concat5")

        if(self.depth_level >= 4):
            self.relu4t  = ReLU(name="relu4t")
            self.conv4t  = Conv2DTranspose(hid_ch * 4, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv4t")  #,bias=False) ### in_channel:1024
            self.in4t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat4 = Concatenate(name="concat4")

        if(self.depth_level >= 3):
            self.relu3t  = ReLU(name="relu3t")
            self.conv3t  = Conv2DTranspose(hid_ch * 2, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv3t")  #,bias=False) ### in_channel:512
            self.in3t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat3 = Concatenate(name="concat3")


        if(self.depth_level >= 2):
            self.relu2t  = ReLU(name="relu2t")
            self.conv2t  = Conv2DTranspose(hid_ch * 1, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2t")  #,bias=False) ### in_channel:256
            self.in2t    = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")
            if(self.skip_use_add is False): self.concat2 = Concatenate(name="concat2")


        self.relu1t = ReLU(name="relu1t")
        self.conv1t = Conv2DTranspose(out_ch, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1t")  ### in_channel:128

        if(self.out_tanh): self.tanh    = Activation(tf.nn.tanh)
        else:              self.sigmoid = Activation(tf.nn.sigmoid)

    def call(self, input_tensor, training=None):  ### 這裡的training只是為了介面統一，實際上沒用到喔，因為IN不需要指定 train/test mode
        x = self.conv1(input_tensor)

        if(self.depth_level >= 2):
            skip2 = x
            x = self.lrelu2(skip2)
            x = self.conv2(x)
            if(self.depth_level > 2): x = self.in2(x)

        if(self.depth_level >= 3):
            skip3 = x
            x = self.lrelu3(skip3)
            x = self.conv3(x)
            if(self.depth_level > 3): x = self.in3(x)

        if(self.depth_level >= 4):
            skip4 = x
            x = self.lrelu4(skip4)
            x = self.conv4(x)
            if(self.depth_level > 4): x = self.in4(x)

        if(self.depth_level >= 5):
            skip5 = x
            x = self.lrelu5(skip5)
            x = self.conv5(x)
            if(self.depth_level > 5): x = self.in5(x)

        if(self.depth_level >= 6):
            skip6 = x
            x = self.lrelu6(skip6)
            x = self.conv6(x)
            if(self.depth_level > 6): x = self.in6(x)

        if(self.depth_level >= 7):
            skip7 = x
            x = self.lrelu7(skip7)
            x = self.conv7(x)
            if(self.depth_level > 7): x = self.in7(x)

        if(self.depth_level >= 8):
            skip8 = x
            x = self.lrelu8(skip8)
            x = self.conv8(x)
            if(self.depth_level > 8): x = self.in8(x)

        ###############################
        if(self.depth_level >= 9):
            skip9 = x
            x = self.lrelu9(skip9)
            x = self.conv9(x)

        if(self.depth_level >= 9):
            x = self.relu9t(x)
            x = self.conv9t(x)
            x = self.in9t(x)
            if(self.skip_use_add is False):
                # x = self.concat9([skip9,x])
                x = self.concat9([x, skip9])
            else: x = x + skip9
        ###############################
        if(self.depth_level >= 8):
            x = self.relu8t(x)
            x = self.conv8t(x)
            x = self.in8t(x)
            if(self.skip_use_add is False):
                # x = self.concat8([skip8,x])
                x = self.concat8([x, skip8])
            else: x = x + skip8

        if(self.depth_level >= 7):
            x = self.relu7t(x)
            x = self.conv7t(x)
            x = self.in7t(x)
            if(self.skip_use_add is False):
                # x = self.concat7([skip7,x])
                x = self.concat7([x, skip7])
            else: x = x + skip7

        if(self.depth_level >= 6):
            x = self.relu6t(x)
            x = self.conv6t(x)
            x = self.in6t(x)
            if(self.skip_use_add is False):
                # x = self.concat6([skip6,x])
                x = self.concat6([x, skip6])
            else: x = x + skip6

        if(self.depth_level >= 5):
            x = self.relu5t(x)
            x = self.conv5t(x)
            x = self.in5t(x)
            if(self.skip_use_add is False):
                # x = self.concat5([skip5,x])
                x = self.concat5([x, skip5])
            else: x = x + skip5


        if(self.depth_level >= 4):
            x = self.relu4t(x)
            x = self.conv4t(x)
            x = self.in4t(x)
            if(self.skip_use_add is False):
                # x = self.concat4([skip4,x])
                x = self.concat4([x, skip4])
            else: x = x + skip4


        if(self.depth_level >= 3):
            x = self.relu3t(x)
            x = self.conv3t(x)
            x = self.in3t(x)
            if(self.second_concat):
                if(self.skip_use_add is False):
                    # x = self.concat3([skip3,x])
                    x = self.concat3([x, skip3])
                else: x = x + skip3


        if(self.depth_level >= 2):
            x = self.relu2t(x)
            x = self.conv2t(x)
            x = self.in2t(x)
            if(self.first_concat):
                if(self.skip_use_add is False):
                    # x = self.concat2([skip2,x])
                    x = self.concat2([x, skip2])
                else: x = x + skip2

        x = self.relu1t(x)
        x = self.conv1t(x)

        if(self.out_tanh):
            # print("tanh")
            return self.tanh(x)
        else:
            # print("sigmoid")
            return self.sigmoid(x)


    def model(self, x):  ### 看summary用的
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))


#######################################################################################################################################
if(__name__ == "__main__"):
    ### 直接用 假資料 嘗試 model 跑不跑得過
    import numpy as np

    generator = Generator(depth_level=9, out_tanh=False)  # 建G
    img = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)  # 建 假資料
    start_time = time.time()  # 看資料跑一次花多少時間
    y = generator(img)
    print(y.shape)
    print(y.numpy().max())   ### 可以在這裡先看看 out_tanh 有沒有設定成功
    print(y.numpy().min())   ### 可以在這裡先看看 out_tanh 有沒有設定成
    print("cost time", time.time() - start_time)

#######################################################################################################################################
    ### 嘗試 真的 load tf_data 進來 train 看看
    import time
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import DB_C, DB_N, DB_GM
    from step06_b_data_pipline import Dataset_builder, tf_Data_builder
    from step08_e_model_obj import MODEL_NAME, KModel_builder
    from step09_a_loss_info_obj import Loss_info_builder

    ### 1. model_obj
    model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_flow_unet()
    flow_unet_IN_ch64 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).build_flow_unet(hid_ch=64, out_ch=3, true_IN=True)
    ### 2. db_obj 和 tf_data
    db_obj = Dataset_builder().set_basic(DB_C.type8_blender_os_book                      , DB_N.blender_os_hw768      , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy").set_detail(have_train=True, have_see=True).build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1 , train_shuffle=False).set_data_use_range(in_use_range="-1~1", gt_use_range="-1~1").set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    ### 3. loss_info_obj
    G_mse_loss_info = Loss_info_builder().build_g_mse_loss_fun_and_containor().build_g_loss_containors().build()

    ### 4. 跑起來試試看
    for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        flow_unet_IN_ch64.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_obj=G_mse_loss_info)
