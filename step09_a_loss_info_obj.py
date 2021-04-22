import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def mse_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.square(tensor1 - tensor2))
    return loss * lamb

def mae_kong(tensor1, tensor2, lamb=tf.constant(1., tf.float32)):
    loss = tf.reduce_mean(tf.math.abs(tensor1 - tensor2))
    return loss * lamb


class Loss_info:
    def __init__(self):
        self.logs_dir = None
        self.summary_writer = None
        self.loss_funs_dict = {}
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


###########################################################################################################
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
    '''
    想多加嘗試 G loss 的話 就在這裡多加 method 就好囉！
    '''
    def build_g_mae_loss_fun_and_containor(self):
        self.loss_info.loss_funs_dict["G"] = mae_kong
        self.loss_info.loss_containors["gen_mae_loss" ] = tf.keras.metrics.Mean('gen_mae_loss', dtype=tf.float32)
        return self

    def build_g_mse_loss_fun_and_containor(self):
        self.loss_info.loss_funs_dict["G"] = mse_kong
        self.loss_info.loss_containors["gen_mse_loss" ] = tf.keras.metrics.Mean('gen_mse_loss', dtype=tf.float32)
        return self


class Loss_info_GAN_loss_builder(Loss_info_G_loss_builder):
    def build_gan_loss(self):
        self.loss_info.loss_funs_dict["G"] = mae_kong
        self.loss_info.loss_funs_dict["G_to_D"] = mse_kong
        self.loss_info.loss_funs_dict["D_Real"] = mse_kong
        self.loss_info.loss_funs_dict["D_Fake"] = mse_kong
        return self

    def build_gan_loss_containors(self):
        self.loss_info.loss_containors["1_loss_rec"    ] = tf.keras.metrics.Mean('1_loss_rec'    , dtype=tf.float32)
        self.loss_info.loss_containors["2_loss_g2d"    ] = tf.keras.metrics.Mean('2_loss_g2d'    , dtype=tf.float32)
        self.loss_info.loss_containors["3_g_total_loss"] = tf.keras.metrics.Mean('3_g_total_loss', dtype=tf.float32)
        self.loss_info.loss_containors["4_loss_d_fake" ] = tf.keras.metrics.Mean('4_loss_d_fake' , dtype=tf.float32)
        self.loss_info.loss_containors["5_loss_d_real" ] = tf.keras.metrics.Mean('5_loss_d_real' , dtype=tf.float32)
        self.loss_info.loss_containors["6_d_total_loss"] = tf.keras.metrics.Mean('6_d_total_loss', dtype=tf.float32)
        return self


class Loss_info_builder(Loss_info_GAN_loss_builder):
    pass
    # def build_loss_containors_by_model_name(self, model_name):
    #     print("model_name:", model_name.value)
    #     if  ("unet" in model_name.value or
    #          "flow" in model_name.value) : self.build_g_loss_containors()
    #     elif("rect"  in model_name.value): self.build_gan_loss_containors()
    #     elif("justG" in model_name.value): self.build_g_loss_containors()
    #     return self

    ### 這好像有點多餘，直接在上面 寫 method 就好啦
    # def custom_loss_funs_dict(self, loss_funs_dict):
    #     self.loss_info.loss_funs_dict = loss_funs_dict
    #     return self

    # def custom_loss_containors(self, loss_containors):
    #     self.loss_info.loss_containors = loss_containors
    #     return self


### 並不是 model 決定 Loss， 而是由 我想 怎麼設計決定，
# 所以 不能只寫 build_by_model_name，也要寫 我自己指定的method
# 所以 也可以跟 model 一樣 先建好
# 然後還要在 exp 裡面 再次設定喔！
G_mse_loss_info = Loss_info_builder().build_g_mse_loss_fun_and_containor().build()
G_mae_loss_info = Loss_info_builder().build_g_mae_loss_fun_and_containor().build()
GAN_mae_loss_info = Loss_info_builder().build_gan_loss().build_gan_loss_containors().build()


if(__name__ == "__main__"):
    # from step08_e_model_obj import MODEL_NAME
    loss_info_obj = Loss_info_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_g_mse_loss_fun_and_containor().build()
    print(loss_info_obj.loss_containors)
    print(loss_info_obj.summary_writer)
