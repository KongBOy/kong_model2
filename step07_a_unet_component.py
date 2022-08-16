import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, BatchNormalization, Concatenate
from tensorflow_addons.layers import InstanceNormalization
from step07_a_small_component import *

def Use_what_acti(acti):
    if  (acti == "lrelu"): return LeakyReLU(alpha=0.2, name="lrelu")
    elif(acti == "relu"):  return ReLU(name="relu")
    else:
        print("Use_what_acti 出錯了， 請輸入正確 acti")

def Use_what_nrom(norm):
    if  (norm == "bn"): return BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn")  ### b_in_channel:64
    elif(norm == "in"): return InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform", name="in")

def Use_what_skip_op(skip_op):
    if(skip_op is None): return None
    elif(skip_op == "cse"):  return cSE
    elif(skip_op == "sse"):  return sSE
    elif(skip_op == "scse"): return scSE

class Conv_block(tf.keras.layers.Layer):
    def __init__(self, out_ch, kernel_size=4, strides=2, padding="same", acti="lrelu", norm="in", use_bias=True, coord_conv=False, **kwargs):
        """
        acti: lrelu/ relu
        norm: bn/ in/ False
        """
        super(Conv_block, self).__init__(**kwargs)
        self.norm = norm
        self.CoordConv = None

        if(coord_conv): self.CoordConv = CoordConv()
        self.Conv = Conv2D(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_down")  #,bias=False) ### in_channel:3
        self.Acti = Use_what_acti(acti)
        if(norm is not False): self.Norm = Use_what_nrom(self.norm)

    def call(self, x, training=None):
        if(self.CoordConv is not None): x = self.CoordConv(x)
        x = self.Conv(x)
        x = self.Acti(x)
        if(self.norm is not False):
            if  (self.norm == "bn"): x = self.Norm(x, training)
            elif(self.norm == "in"): x = self.Norm(x)

        return x

class Conv_Blocks(tf.keras.layers.Layer):
    def __init__(self, in_ch,
                       out_ch,
                       kernel_size=3, strides=1, padding="same", norm="in",
                       acti="lrelu",
                       use_bias=True,
                       conv_block_num=1,
                       coord_conv=False, **kwargs):
        super(Conv_Blocks, self).__init__(**kwargs)

        self.Conv_blocks = [ Conv_block(out_ch=in_ch,  kernel_size=kernel_size, strides=strides, padding=padding, acti=acti, norm=norm, use_bias=use_bias, coord_conv=coord_conv, name=f"Conv_block_{i}") for i in range(conv_block_num - 1) ] + \
                           [ Conv_block(out_ch=out_ch, kernel_size=kernel_size, strides=strides, padding=padding, acti=acti, norm=norm, use_bias=use_bias, coord_conv=coord_conv, name=f"Conv_block_{conv_block_num - 1}")]

    def call(self, x, training=None):
        for block in self.Conv_blocks: x = block(x, training)
        return x

class UNet_down(tf.keras.layers.Layer):
    def __init__(self, at_where, in_ch, out_ch,
                 kernel_size=4, strides=2, padding="same", norm="in",
                 acti="lrelu",
                 use_bias=True,
                 conv_block_num=0,
                 skip_op = None,
                 coord_conv=False,
                 **kwargs):
        """
        at_where: top/ middle/ bottle
        acti: lrelu/ relu
        norm: bn/ in
        skip_op: None/ cse/ sse/ scse/ cnn
        """
        self.CoordConv = None
        self.Skip_op = skip_op
        if(coord_conv): self.CoordConv = CoordConv()
        super(UNet_down, self).__init__(**kwargs)
        self.at_where = at_where
        self.norm = norm
        self.Conv_blocks = [ Conv_block(out_ch=in_ch, kernel_size=kernel_size, strides=1, padding="same", acti=acti, norm=norm, use_bias=use_bias, name=f"Conv_block_{i}") for i in range(conv_block_num) ]
        ''' 目前覺得這樣子展開來比較好看 '''
        if  (self.at_where == "top"):
            self.Conv    = Conv2D(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_down")  #,bias=False) ### in_channel:3
            self.Skip_op = Use_what_skip_op(skip_op)  ### cse/sse/scse， 看用哪個物件
            if(self.Skip_op is not None): self.Skip_op = self.Skip_op(in_ch=out_ch, ratio=out_ch // 32)  ### 把物件創造出來
        elif(self.at_where == "middle"):
            self.Acti    = Use_what_acti(acti)
            self.Conv    = Conv2D(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_down")  #,bias=False) ### in_channel:3
            self.Norm    = Use_what_nrom(self.norm)
            self.Skip_op = Use_what_skip_op(skip_op)  ### cse/sse/scse
            if(self.Skip_op is not None): self.Skip_op = self.Skip_op(in_ch=out_ch, ratio=out_ch // 32)
        elif(self.at_where == "bottle"):
            self.Acti = Use_what_acti(acti)
            self.Conv = Conv2D(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_down")  #,bias=False) ### in_channel:3

        ''' 簡短版大概長這樣，不過不好直觀的理解 '''
        # if(self.at_where != "top"): self.Acti = Use_what_acti(acti)
        # if(self.at_where == "middle"): self.Norm = Use_what_nrom(self.norm)
        # self.Conv = Conv2D(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_down")  #,bias=False) ### in_channel:3
        # if(self.at_where != "bottle"):
        #   self.Skip_op = Use_what_skip_op(skip_op)  ### cse/sse/scse
        #   if(self.Skip_op is not None): self.Skip_op = self.Skip_op(in_ch=out_ch, ratio=out_ch // 32)

    def call(self, x, training=None):
        ### Activation
        if  (self.at_where != "top"):  x = self.Acti(x)

        ### Conv_Blocks
        for block in self.Conv_blocks: x = block(x, training)
        x_before_down = x

        ### Down
        if(self.CoordConv is not None): x = self.CoordConv(x)
        x = self.Conv(x)

        ### Norm
        if(self.at_where == "middle"):
            if  (self.norm == "bn"): x = self.Norm(x, training)
            elif(self.norm == "in"): x = self.Norm(x)

        ### Skip
        if(self.Skip_op is None):
            x_after_down = x
        else:
            x_after_down          = self.Skip_op(x)
            x_before_down = self.Skip_op(x_before_down)

        return x, x_after_down, x_before_down

class UNet_up(tf.keras.layers.Layer):
    def __init__(self, at_where, in_ch, out_ch,
                 kernel_size=4, strides=2, padding="same", norm="in",
                 acti="relu",
                 use_bias=True,
                 conv_block_num=0,
                 skip_merge_op="concat",
                 coord_conv=False,
                 **kwargs):
        """
        at_where: top/ middle/ bottle
        acti: lrelu/ relu
        norm: bn/ in
        skip_merge_op: concat/ add
        """
        self.CoordConv = None
        if(coord_conv): self.CoordConv = CoordConv()
        super(UNet_up, self).__init__(**kwargs)
        self.at_where = at_where
        self.norm = norm
        self.Conv_blocks = [ Conv_block(out_ch=in_ch, kernel_size=kernel_size, strides=1, padding="same", acti=acti, norm=norm, use_bias=use_bias, name=f"Conv_block_{i}") for i in range(conv_block_num) ]

        if  (self.at_where == "top"):
            self.Acti = Use_what_acti(acti)
            self.Conv_T = Conv2DTranspose(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_up")  #,bias=False) ### in_channel:3
        elif(self.at_where == "middle" or self.at_where == "bottle"):
            self.Acti = Use_what_acti(acti)
            self.Conv_T = Conv2DTranspose(out_ch, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name="conv_up")  #,bias=False) ### in_channel:3
            self.Norm = Use_what_nrom(self.norm)

            self.Skip_merge_op = skip_merge_op
            if(self.Skip_merge_op == "concat"): self.Concat = Concatenate(name="concat")

    def call(self, x, skip=None, training=None):
        ### Activation
        x = self.Acti(x)

        ### Conv_Blocks
        for block in self.Conv_blocks: x = block(x, training)


        if(self.CoordConv is not None): x = self.CoordConv(x)
        # print(x.shape)

        ### Up
        x = self.Conv_T(x)
        # print(x.shape)

        ### Norm and Concat with Skip
        if(self.at_where != "top"):
            if  (self.norm == "bn"): x = self.Norm(x, training)
            elif(self.norm == "in"): x = self.Norm(x)

            if(skip is not None):
                if  (self.Skip_merge_op == "concat"): x = self.Concat([skip, x])
                elif(self.Skip_merge_op == "add"):    x = x + skip

        return x


if(__name__ == "__main__"):
    import numpy as np

    data = np.ones(shape=(1, 192, 192, 128), dtype=np.float32)

    d1_l = UNet_down(at_where= "top", out_ch=128, conv_block_num=2, skip_op="cse")
    d2_l = UNet_down(at_where="middle", out_ch=20, conv_block_num=2)
    u2_l = UNet_up(at_where="middle", out_ch=30, conv_block_num=2)

    d2, skip2 = d1_l(data)
    d3, skip3 = d2_l(d2)
    u2 = u2_l(d3, skip2)
    print(u2)
