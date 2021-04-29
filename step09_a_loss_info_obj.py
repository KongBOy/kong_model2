import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build_new_dir
import copy

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

        self.loss_npy_dict = {}

    def _load_loss_npy_dict(self):
        for loss_name in self.loss_containors.keys():
            self.loss_npy_dict[loss_name] = np.load(self.logs_dir + "/" + loss_name + ".npy")

    def see_loss(self, epochs):
        for loss_name in self.loss_containors.keys():
            plt.figure(figsize=(20, 6))                              ### 建立畫布
            plt.ylim(0, 0.01)
            plt.ylabel(loss_name)
            npy_loss = np.load(self.logs_dir + "/" + loss_name + ".npy")

            plt.xlim(0, epochs)
            plt.xlabel("epoch_num")
            x_epoch = np.arange(len(npy_loss))

            plt.plot(x_epoch, npy_loss)
            plt.savefig(self.logs_dir + "/" + loss_name + ".png")
            plt.close()
            # print("plot %s loss ok~"%loss_name )
        print("plot loss ok~")

    def use_npy_rebuild_justG_tensorboard_loss(self, exp, dst_dir=".", rebuild_board_name="gen_loss_and_lr"):
        '''
        最終會建立的樣子   ：舉例 result/logs/rebuild_board_name/event...
        exp               ：用來抓出當初exp用的 epochs, lr_start, epoch_down_step
        dst_dir           ：通常會在 對應的 result/logs，但 analyze 我也想 用這個fun，所以還是多個參數來控制喔！
        result_board_name ：目前是想用來區分 用什麼loss，所以default gen_loss ， 下面會 根據 使用的 loss 區別出 gen_mae/gen_mse
        '''
        self._load_loss_npy_dict()  ### 讀取 result/logs 的 .npy
        epochs          = exp.epochs
        lr_start        = exp.lr_start
        epoch_down_step = exp.epoch_down_step
        ### 在外部資料夾 區分出 用什麼loss，內部統一用 gen_loss 當名字，這樣才能 mae/mse 放在同個圖裡面比較喔！
        if  ("gen_mae_loss" in self.loss_containors.keys()): rebuild_board_name = "gen_mae_loss_and_lr"
        elif("gen_mse_loss" in self.loss_containors.keys()): rebuild_board_name = "gen_mse_loss_and_lr"

        #######################################################################################################################################
        ### write to summary writer
        Check_dir_exist_and_build_new_dir(dst_dir + "/" + rebuild_board_name)       ### 雖然會自動建立資料夾，如果重複執行 我想刪掉 上次的執行結果， 所以還是 Check_dir_exist_and_build_new_dir() 囉！
        writer = tf.summary.create_file_writer(dst_dir + "/" + rebuild_board_name)  ### 建tensorboard，這會自動建資料夾喔！
        for loss_name in self.loss_containors.keys():   ### 根據 loss_name 讀出 相對應的 loss
            for go_epoch in range( min(len(self.loss_npy_dict[loss_name]), exp.epochs)):  ### loss_index 本身就有 epoch 的概念 囉，所以用for 來 跑 len(loss_npy) 模擬 epoch！取 min 是因為 可能當初 exp 寫錯，train到超出 epochs 了
                epoch = go_epoch + 1  ###  +1是因為 我在主程式 都是 train完 才紀錄 loss，所以 模擬的時候也要 +1
                with writer.as_default():
                    ### 內部統一用 gen_loss 當名字，這樣才能 mae/mse 放在同個圖裡面比較喔！
                    tf.summary.scalar("gen_loss", self.loss_npy_dict[loss_name][go_epoch], step=epoch)   ### 把 loss 值 讀出來 並寫進 tensorboard
                    if  (epochs == epoch_down_step):  ### lr 沒有下降
                        tf.summary.scalar("lr", lr_start, step=epoch)   ### 把lr模擬值 寫入 tensorboard
                    elif(epochs != epoch_down_step):  ### lr 有下降，目前就一種，到epoch_down_step後 用直線 下降到 0
                        lr_current = lr_start if epoch < epoch_down_step else lr_start * (epochs - epoch) / (epochs - epoch_down_step)  ### 模擬訓練時的lr下降
                        tf.summary.scalar("lr", lr_current, step=epoch)   ### 把lr模擬值 寫入 tensorboard



###########################################################################################################
class Loss_info_init_builder:
    def __init__(self, loss_info=None, in_obj_copy=True):
        '''
        一定要注意需不需要 in_obj_copy喔！
        因為雖然 loss_fun 是共用的， 但是每個exp 算出來的 loss_value是不同的！因此 logs_dir 也是不同的！
        所以還是需要每個 exp 都有自己的 loss_info_obj 喔！才不會 logs_dir 被共用 這樣子！
        所以 step10_a 的 exp_builder.build() 裡面 呼叫 Loss_info_builder() 更新 loss_info_obj時，in_obj_copy 就要指定 True 拉！
        '''
        if(loss_info is None): self.loss_info = Loss_info()
        else:
            self.loss_info = loss_info
            if(in_obj_copy): self.loss_info = copy.deepcopy(self.loss_info)

    def set_logs_dir(self, logs_dir):
        self.loss_info.logs_dir = logs_dir
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
    loss_info_obj = Loss_info_builder().set_logs_dir(logs_dir="abc").build_g_mse_loss_fun_and_containor().build()
    print(loss_info_obj.loss_containors)
    print(loss_info_obj.summary_writer)
