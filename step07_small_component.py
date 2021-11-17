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
