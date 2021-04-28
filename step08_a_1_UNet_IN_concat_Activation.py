import sys
sys.path.append("kong_util")
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Concatenate, Activation
import time

"""
因為 要改的部分太多，所以才多寫一個.py喔！
"""


### 參考 DewarpNet 的 train_wc 用的 UNet
### 試試看 activation 完 再去 concate
class Generator(tf.keras.models.Model):
    def __init__(self, hid_ch=64, depth_level=7, first_concat=True, second_concat=True, skip_use_add=False, skip_use_cnn3_relu=False, out_tanh=True, out_ch=3, **kwargs):
        '''
        depth_level, skip_use_add 還沒有實作喔，有用到再做吧~

        out_tanh：想實驗看看 output 是 tanh 和 sigmoid 的效果，out_tanh=False 就是用 sigmoid
        '''
        self.out_tanh = out_tanh

        super(Generator, self).__init__(**kwargs)
        self.conv1 = Conv2D(hid_ch * 1, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1")  #,bias=False) ### in_channel:3

        self.lrelu2 = LeakyReLU(alpha=0.2, name="lrelu2")
        self.conv2  = Conv2D(hid_ch * 2, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2")  #,bias=False) ### in_channel:64
        self.in2    = InstanceNormalization(name="in2")  ### b_in_channel:128

        self.lrelu3 = LeakyReLU(alpha=0.2, name="lrelu3")
        self.conv3  = Conv2D(hid_ch * 4, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv3")  #,bias=False) ### in_channel:128
        self.in3    = InstanceNormalization(name="in3")  ### b_in_channel:256

        self.lrelu4 = LeakyReLU(alpha=0.2, name="lrelu4")
        self.conv4  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv4")  #,bias=False) ### in_channel:256
        self.in4    = InstanceNormalization(name="in4")  ### b_in_channel:512

        self.lrelu5 = LeakyReLU(alpha=0.2, name="lrelu5")
        self.conv5  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv5")  #,bias=False) ### in_channel:512
        self.in5    = InstanceNormalization(name="in5")  ### b_in_channel:512

        self.lrelu6 = LeakyReLU(alpha=0.2, name="lrelu6")
        self.conv6  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv6")  #,bias=False) ### in_channel:512
        self.in6    = InstanceNormalization(name="in6")  ### b_in_channel:512

        ###################
        # 最底層
        self.lrelu7 = LeakyReLU(alpha=0.2, name="lrelu7")
        self.conv7  = Conv2D(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv7")  #,bias=False) ### in_channel:512

        self.relu7t = ReLU(name="relu7t")
        self.conv7t = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv7t")  #,bias=False) ### in_channel:512
        self.in7t   = InstanceNormalization(name="in7t")  ### b_in_channel:512
        self.concat7 = Concatenate(name="concat7")
        ###################

        self.relu6t = ReLU(name="relu6t")
        self.conv6t = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv6t")  #,bias=False) ### in_channel:1024
        self.in6t   = InstanceNormalization(name="in6t")  ### b_in_channel:512
        self.concat6 = Concatenate(name="concat6")

        self.relu5t = ReLU(name="relu5t")
        self.conv5t = Conv2DTranspose(hid_ch * 8, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv5t")  #,bias=False) ### in_channel:1024
        self.in5t   = InstanceNormalization(name="in5t")  ### b_in_channel:512
        self.concat5 = Concatenate(name="concat5")

        self.relu4t = ReLU(name="relu4t")
        self.conv4t = Conv2DTranspose(hid_ch * 4, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv4t")  #,bias=False) ### in_channel:1024
        self.in4t   = InstanceNormalization(name="in4t")  ### b_in_channel:256
        self.concat4 = Concatenate(name="concat4")

        self.relu3t = ReLU(name="relu3t")
        self.conv3t = Conv2DTranspose(hid_ch * 2, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv3t")  #,bias=False) ### in_channel:512
        self.in3t   = InstanceNormalization(name="in3t")  ### b_in_channel:128
        self.concat3 = Concatenate(name="concat3")


        self.relu2t = ReLU(name="relu2t")
        self.conv2t = Conv2DTranspose(hid_ch * 1, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2t")  #,bias=False) ### in_channel:256
        self.in2t   = InstanceNormalization(name="in2t")  ### b_in_channel:64
        self.concat2 = Concatenate(name="concat2")


        self.relu1t = ReLU(name="relu1t")
        self.conv1t = Conv2DTranspose(out_ch, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1t")  ### in_channel:128

        if(self.out_tanh): self.tanh    = Activation(tf.nn.tanh)
        else:              self.sigmoid = Activation(tf.nn.sigmoid)

    def call(self, input_tensor, training=True):  ### 這裡的training只是為了介面統一，實際上沒用到喔，因為IN不需要指定 train/test mode
        x = self.conv1(input_tensor)

        x = self.lrelu2(x)
        skip2 = x
        x = self.conv2(skip2)
        x = self.in2(x)

        x = self.lrelu3(x)
        skip3 = x
        x = self.conv3(skip3)
        x = self.in3(x)

        x = self.lrelu4(x)
        skip4 = x
        x = self.conv4(skip4)
        x = self.in4(x)

        x = self.lrelu5(x)
        skip5 = x
        x = self.conv5(skip5)
        x = self.in5(x)

        x = self.lrelu6(x)
        skip6 = x
        x = self.conv6(skip6)
        x = self.in6(x)
        ###############################
        x = self.lrelu7(x)
        skip7 = x
        x = self.conv7(skip7)

        x = self.relu7t(x)
        x = self.conv7t(x)
        x = self.in7t(x)
        # x = self.concat7([skip7,x])
        x = self.concat7([x, skip7])
        ###############################
        x = self.relu6t(x)
        x = self.conv6t(x)
        x = self.in6t(x)
        # x = self.concat6([skip6,x])
        x = self.concat6([x, skip6])

        x = self.relu5t(x)
        x = self.conv5t(x)
        x = self.in5t(x)
        # x = self.concat5([skip5,x])
        x = self.concat5([x, skip5])


        x = self.relu4t(x)
        x = self.conv4t(x)
        x = self.in4t(x)
        # x = self.concat4([skip4,x])
        x = self.concat4([x, skip4])


        x = self.relu3t(x)
        x = self.conv3t(x)
        x = self.in3t(x)
        # x = self.concat3([skip3,x])
        x = self.concat3([x, skip3])


        x = self.relu2t(x)
        x = self.conv2t(x)
        x = self.in2t(x)
        # x = self.concat2([skip2,x])
        x = self.concat2([x, skip2])

        x = self.relu1t(x)
        x = self.conv1t(x)

        if(self.out_tanh): return self.tanh(x)
        else:              return self.sigmoid(x)


    def model(self, x):  ### 看summary用的
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))


#######################################################################################################################################
if(__name__ == "__main__"):
    ### 直接用 假資料 嘗試 model 跑不跑得過
    import numpy as np

    generator = Generator()  # 建G
    img = np.ones(shape=(1, 256, 256, 3), dtype=np.float32)  # 建 假資料
    start_time = time.time()  # 看資料跑一次花多少時間
    y = generator(img)
    print(y)
    print("cost time", time.time() - start_time)

    ### 沒有真的常在用，所以就沒有測試train_step囉！
