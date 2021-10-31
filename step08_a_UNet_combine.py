import tensorflow as tf
from step07_unet_component import UNet_down, UNet_up
from tensorflow.keras.layers import Activation
### 參考 DewarpNet 的 train_wc 用的 UNet
### 所有 pytorch BN 裡面有兩個參數的設定不確定～： affine=True, track_running_stats=True，目前思考覺得改道tf2全拿掉也可以
### 目前 總共用7層，所以size縮小 2**7 ，也就是 1/128 這樣子！例如256*256*3丟進去，最中間的feature map長寬2*2*512喔！
class Generator(tf.keras.models.Model):
    def __init__(self, hid_ch=64, depth_level=7, out_ch=3, no_concat_layer=0,
                 kernel_size=4, strides=2,
                 d_acti="lrelu", u_acti="relu", unet_acti="tanh", norm="in",
                 cnn_bias=True,
                 conv_block_num=0,
                 skip_op=None, skip_merge_op="concat",
                 #  out_tanh=True,
                 #  skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 **kwargs):
        '''
        d_acti: lrelu/ relu
        u_acti: relu/ lrelu
        unet_acti: tanh/ sigmoid
        skip_op: cse/ sse/ scse
        '''
        if(depth_level < 2):
            print("UNet 不可以小於 2層， 因為如果只有 1層 沒辦法做 skip connection")
            exit()
        super(Generator, self).__init__(**kwargs)
        self.depth_level = depth_level
        self.hid_ch = hid_ch
        self.no_concat_layer = no_concat_layer
        self.unet_out_ch = out_ch
        self.unet_acti = unet_acti
        ### 最基本(比如最少層depth_level=2)的一定有 top, bottle
        self.d_top    = UNet_down(at_where="top"   , out_ch=self.Get_Layer_hid_ch(to_L=1              ), name="D 0->1 top")  ### Layer 0 -> 1， to_L=1 代表 走進 第1層
        self.u_top    = UNet_up  (at_where="top"   , out_ch=self.Get_Layer_hid_ch(to_L=0              ), name="U 1->0 top")  ### Layer 1 -> 0， to_L=0 代表 返回 第0層
        self.d_bottle = UNet_down(at_where="bottle", out_ch=self.Get_Layer_hid_ch(to_L=depth_level    ), name=f"D {depth_level-1}->{depth_level} bottle")
        self.u_bottle = UNet_up  (at_where="bottle", out_ch=self.Get_Layer_hid_ch(to_L=depth_level - 1), name=f"U {depth_level}->{depth_level-1} bottle")  ### 因為是返回上一層， 所以 -1
        # self.d_bottle = UNet_down(at_where="bottle", out_ch=min(hid_ch * 2**(depth_level - 1)    , 512), name=f"D{depth_level} bottle")  ### L0(3), L1(hid_ch*2**0), L2(hid_ch*2**1), ..., L2(hid_ch*2**depth_level - 1)
        # self.u_bottle = UNet_up  (at_where="bottle", out_ch=min(hid_ch * 2**(depth_level - 1 - 1), 512), name=f"U{depth_level} bottle")  ### L0(3), L1(hid_ch*2**0), L2(hid_ch*2**1), ..., L2(hid_ch*2**depth_level - 1)， 因為是返回上一層， out_ch 2的冪次要再 -1

        ### depth_level >=3 以後在 top 和 bottle 之間 加入 middle 層的概念， 連接順序是 D 從 top層 往 bottle層 ， U 從 bottle層 到 top，
        ###     小心 這邊 U的宣告 是 top 到 bottle 和 連接順序相反， 宣告完要記得reverse一下順序喔
        self.d_middles = []
        self.u_middles = []
        if(depth_level >= 3):
            for i in range(depth_level - 2):  ### -2 是 -top 和 -bottle 共兩層
                layer_id = i + 1 + 1  ### +1 是 index轉layer_id， 再+1 是因為前面有top層。 middle 至少 一定從 走入Layer2開始(Down) 或 從Layer2開始返回(Up)
                self.d_middles.append( UNet_down(at_where="middle", out_ch=self.Get_Layer_hid_ch(to_L=layer_id    ), name=f"D {layer_id-1}->{layer_id} middle" ) )
                self.u_middles.append( UNet_up  (at_where="middle", out_ch=self.Get_Layer_hid_ch(to_L=layer_id - 1), name=f"U {layer_id}->{layer_id-1} middle" ) )
                # self.d_middles.append( UNet_down(at_where="middle", out_ch=( min(hid_ch * 2**(layer_id - 1    ), 512) ), name=f"D{layer_id} middle" ) )
                # self.u_middles.append( UNet_up  (at_where="middle", out_ch=( min(hid_ch * 2**(layer_id - 1 - 1), 512) ), name=f"U{layer_id} middle" ) )

        ### 注意 up 的部分是 從bottle層 往top連接， 跟宣告順序相反， 要reverse一下順序喔
        self.u_middles.reverse()
        ############################################################################################################################################################
        if(self.unet_acti == "tanh"):    self.tanh    = Activation(tf.nn.tanh,    name="out_tanh")
        if(self.unet_acti == "sigmoid"): self.sigmoid = Activation(tf.nn.sigmoid, name="out_sigmoid")

    def Get_Layer_hid_ch(self, to_L, ch_upper_bound=512):
        '''
        L_0 (3 或 1),
        L_1 (hid_ch*2**0),
        L_2 (hid_ch*2**1),
        L_3 (hid_ch*2**2),
        ...,
        L_depth_level (hid_ch*2**depth_level - 1)
        '''
        if(to_L == 0): return self.unet_out_ch
        else:          return min(self.hid_ch * 2**(to_L - 1), ch_upper_bound)

    def call(self, input_tensor, training=None):
        skips = []

        #####################################################
        ### Down top
        x, skip = self.d_top(input_tensor)
        skips.append(skip)
        ### Down middle
        for d_middle in self.d_middles:
            x, skip = d_middle(x)
            skips.append(skip)
        ### Down bottle
        x = self.d_bottle(x)  ### down 的 bottle沒有 skip
        #####################################################
        ### Up bottle
        if(self.no_concat_layer >= self.depth_level - 1): x = self.u_bottle(x)
        else:                                             x = self.u_bottle(x, skips.pop())

        ### Up middle
        for go, u_middle in enumerate(self.u_middles):
            layer_id = self.depth_level - 1 - go
            if (layer_id <= self.no_concat_layer): x = u_middle(x)
            else:                                  x = u_middle(x, skips.pop())

        ### Up top
        x = self.u_top(x)  ### up 的 top 沒有 skip
        #####################################################
        ### UNet out
        if  (self.unet_acti == "tanh"):    return self.tanh(x)
        elif(self.unet_acti == "sigmoid"): return self.sigmoid(x)


if(__name__ == "__main__"):
    import numpy as np

    data = np.ones(shape=(1, 768, 768, 128), dtype=np.float32)

    test_g = Generator(hid_ch=64, depth_level=7, cnn_bias=False)
    print(test_g(data))
    test_g.summary()
