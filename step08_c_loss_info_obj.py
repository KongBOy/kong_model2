import tensorflow as tf
from step08_e_model_obj import MODEL_NAME
import matplotlib.pyplot as plt
import numpy as np


class Loss_info:
    def __init__(self):
        self.logs_dir = None
        self.summary_writer = None
        self.loss_containors = {}

    def see_loss(self, epochs):
        for loss_name in self.loss_containors.keys():
            plt.figure(figsize=(20, 6))                              ### 建立畫布
            plt.ylim(0, 0.01)
            plt.ylabel(loss_name)
            y_loss_array = np.load(self.logs_dir + "/" + loss_name + ".npy")

            plt.xlim(0, epochs)
            plt.xlabel("epoch_num")
            x_epoch = np.arange(len(y_loss_array))

            plt.plot(x_epoch, y_loss_array)
            plt.savefig(self.logs_dir + "/" + loss_name + ".png")
            plt.close()
            # print("plot %s loss ok~"%loss_name )
        print("plot loss ok~")



class Loss_info_init_builder:
    def __init__(self, loss_info=None):
        if(loss_info is None): self.loss_info = Loss_info()
        else:                  self.loss_info = loss_info

    def set_logs_dir_and_summary_writer(self, logs_dir):
        self.loss_info.logs_dir = logs_dir
        self.loss_info.summary_writer = tf.summary.create_file_writer(logs_dir)  ### 建tensorboard，這會自動建資料夾喔！
        return self

    def build(self):
        return self.loss_info

class Loss_info_G_loss_builder(Loss_info_init_builder):
    def build_g_loss(self):
        self.loss_info.loss_containors["gen_loss" ] = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
        return self

class Loss_info_GAN_loss_builder(Loss_info_G_loss_builder):
    def build_gan_loss(self):
        self.loss_info.loss_containors["1_loss_rec"    ] = tf.keras.metrics.Mean('1_loss_rec'    , dtype=tf.float32)
        self.loss_info.loss_containors["2_loss_g2d"    ] = tf.keras.metrics.Mean('2_loss_g2d'    , dtype=tf.float32)
        self.loss_info.loss_containors["3_g_total_loss"] = tf.keras.metrics.Mean('3_g_total_loss', dtype=tf.float32)
        self.loss_info.loss_containors["4_loss_d_fake" ] = tf.keras.metrics.Mean('4_loss_d_fake' , dtype=tf.float32)
        self.loss_info.loss_containors["5_loss_d_real" ] = tf.keras.metrics.Mean('5_loss_d_real' , dtype=tf.float32)
        self.loss_info.loss_containors["6_d_total_loss"] = tf.keras.metrics.Mean('6_d_total_loss', dtype=tf.float32)
        return self

# class Loss_info_justG_builder(Loss_info_GAN_loss_builder):
#     def build_justG(self):
#         self.loss_info.loss_containors["loss_rec"    ] = tf.keras.metrics.Mean('loss_rec'    , dtype=tf.float32)
#         return self


class Loss_info_builder(Loss_info_GAN_loss_builder):
    def build_by_model_name(self, model_name):
        print("model_name:", model_name.value)
        if  ("unet"  in model_name.value or "flow" in model_name.value) : self.build_g_loss()
        elif("rect"  in model_name.value) : self.build_gan_loss()
        elif("justG" in model_name.value) : self.build_g_loss()
        return self


if(__name__ == "__main__"):
    loss_info_obj = Loss_info_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_by_model_name(MODEL_NAME.rect).build()
    print(loss_info_obj.loss_containors)
    print(loss_info_obj.summary_writer)
