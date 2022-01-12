import tensorflow as tf
from tensorflow.keras.layers import GlobalAvgPool2D, Conv2D, Dense, Activation, ReLU

class cSE(tf.keras.layers.Layer):
    def __init__(self, in_ch, ratio, **kwargs):
        super(cSE, self).__init__(**kwargs)
        # self.in_ch = in_ch
        self.squeeze_sp_avg_pool = GlobalAvgPool2D(name = "squeeze_sp_avg_pool")
        self.excite_1 = Dense(in_ch // ratio, use_bias=False, name="excite_1")
        self.excite_r = ReLU(                                 name="excite_ReLU")
        self.excite_2 = Dense(in_ch         , use_bias=False, name="excite_2")
        self.excite_ch_sigmoid = Activation(tf.nn.sigmoid, name="excite_sigmoid")

    def call(self, input_tensor):
        # print("self.in_ch", self.in_ch)
        # print("input_tensor.shape", input_tensor.shape)
        x = self.squeeze_sp_avg_pool(input_tensor)
        # print(x.shape)
        x = self.excite_1(x)
        # print(x.shape)
        x = self.excite_r(x)
        x = self.excite_2(x)
        # print(x.shape)
        x = self.excite_ch_sigmoid(x)
        # print(x.shape)
        # print(x)
        return input_tensor * x

class sSE(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(sSE, self).__init__(**kwargs)
        self.squeeze_ch_1x1conv = Conv2D(1, kernel_size=1, use_bias=False, name="squeeze_ch_1x1conv")
        self.excite_sp_sigmoid = Activation(tf.nn.sigmoid, name="excite_sigmoid")

    def call(self, input_tensor):
        # print("input_tensor.shape", input_tensor.shape)
        x = self.squeeze_ch_1x1conv(input_tensor)
        # print("x.shape", x.shape)
        x = self.excite_sp_sigmoid(x)
        # print("x.shape", x.shape)
        return input_tensor * x

class scSE(tf.keras.layers.Layer):
    def __init__(self, in_ch, ratio, **kwargs):
        super(scSE, self).__init__(**kwargs)
        self.cSE = cSE(in_ch, ratio, **kwargs)
        self.sSE = sSE(**kwargs)

    def call(self, input_tensor):
        # print("input_tensor.shape", input_tensor.shape)
        cse = self.cSE(input_tensor)
        # print("x.shape", x.shape)
        sse = self.sSE(input_tensor)
        # print("x.shape", x.shape)
        return cse + sse

class CoordConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CoordConv, self).__init__(**kwargs)

    def call(self, input_tensor):
        height = input_tensor.shape[1]
        width  = input_tensor.shape[2]
        x = tf.range(start=0, limit=width, dtype=tf.float32)
        x = tf.reshape(x, [1, -1])
        x = tf.tile(x, [height, 1])
        x = tf.expand_dims(x, axis=-1)
        x = x / (width - 1)
        x = x * 2 - 1  ### 值域 弄到 -1~1
        # print(x)

        y = tf.range(start=0, limit=height, dtype=tf.float32)
        y = tf.reshape(y, [-1, 1])
        y = tf.tile(y, [1, width])
        y = tf.expand_dims(y, axis=-1)
        y = y / (height - 1)
        y = y * 2 - 1  ### 值域 弄到 -1~1
        # print(y)

        yx = tf.concat([y, x], axis=-1)
        yx = tf.expand_dims(yx, axis=0)
        # print(yx)

        return tf.concat([input_tensor, yx], axis=-1)

class InstanceNorm_kong(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNorm_kong, self).__init__(**kwargs)

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale  = self.add_weight("scale", shape=[depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02), dtype=tf.float32)
        self.offset = self.add_weight("offset", shape=[depth], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1, 2], keepdims=True)
        epsilon = tf.constant(1e-5, dtype=tf.float32)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return self.scale * normalized + self.offset
        # return tf.matmul(input, self.kernel)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, use_what_IN=InstanceNorm_kong, ks=3, s=1, use_res_learning=True, coord_conv=False, **kwargs):
        super(ResBlock, self).__init__()
        self.ks = ks
        self.use_res_learning = use_res_learning
        self.coord_conv = coord_conv
        # if(self.coord_conv): self.coord_conv_res_layer1 = CoordConv()
        self.conv_1 = Conv2D(c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c1 = use_what_IN()
        # if(self.coord_conv): self.coord_conv_res_layer2 = CoordConv()
        self.conv_2 = Conv2D(c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c2 = use_what_IN()

    def call(self, input_tensor):
        p = int((self.ks - 1) / 2)
        # if(self.coord_conv): input_tensor = self.coord_conv_res_layer1(input_tensor)
        x = tf.pad(input_tensor, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        x = self.conv_1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)

        # if(self.coord_conv): x = self.coord_conv_res_layer2(x)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        x = self.conv_2(x)
        x = self.in_c2(x)
        if(self.use_res_learning): return x + input_tensor[..., :]
        else: return x


if(__name__ == "__main__"):
    import numpy as np

    data = np.ones(shape=(1, 192, 192, 128), dtype=np.float32)

    cse = cSE(in_ch=128, ratio=4)
    out = cse(data)
    print(out[0, ..., 0])
    print(out[0, ..., 1])
    print(out[0, ..., 2])

    sse = sSE()
    data_hw = np.arange(16).reshape(1, 4, 4, 1).astype(np.float32)
    data_hw = np.tile(data_hw, (1, 1, 1, 1024))
    print("data_hw.shape", data_hw.shape)
    out_sse = sse(data_hw)
    print(out_sse[0, ..., 0])
    print(out_sse[0, 0, 0, :])
    print(out_sse[0, 0, 1, :])

    scse = scSE(in_ch=128, ratio=4)
    out_scse = scse(data)
    print(out_scse)
