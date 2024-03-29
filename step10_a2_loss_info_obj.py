import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import Check_dir_exist_and_build_new_dir
import copy

import time
start_time = time.time()

from step10_a1_loss import *


class Loss_info:
    def __init__(self):
        self.loss_target = "UNet1"
        self.loss_type = None
        self.loss_describe_eles = []  ### 給 result命名的時候用
        self.loss_describe  = None
        self.logs_read_dir  = None
        self.logs_write_dir = None
        self.summary_writer = None
        self.loss_funs_dict = {}
        self.loss_containors = {}

        self.loss_npy_dict = {}

    def see_loss_during_train(self, epochs):
        for loss_name in self.loss_containors.keys():
            npy_loss_array = np.load(self.logs_write_dir + "/" + loss_name + ".npy", allow_pickle=True)  ### logs_read/write_dir 這較特別！因為這是在 "training 過程中執行的 read" ，  我們想read 的 npy_loss_array 在train中 是使用  logs_write_dir 來存， 所以就要去 logs_write_dir 來讀囉！ 所以這邊 np.load 裡面適用 logs_write_dir 是沒問題的！
            iter_indexes = [ index for index, data in enumerate(npy_loss_array) if data is not None]  ### 把有值的地方 的 index 記錄下來
            iter_values  = npy_loss_array[iter_indexes]                                               ### 把有值的地方 抓出來

            plt.figure(figsize=(20, 6))            ### 建立畫布
            plt.ylim(0, iter_values.max() + 0.05)  ### 多一點點邊邊
            plt.ylabel(loss_name)

            plt.xlim(0, len(npy_loss_array))
            plt.xlabel("total_iters")

            plt.scatter(iter_indexes, iter_values, s=10)                 ### 畫出 所有loss點的位置
            plt.scatter(iter_indexes[-1], iter_values[-1], s=20, c="r")  ### 標出 目前所在的loss位置
            plt.plot   (iter_indexes, iter_values)                       ### 連接 所有loss點

            for go_iter, iter_index in enumerate(iter_indexes):  ### 標出 所有loss點的數值
                ### 讓字不要疊在一起
                text_y_shift = 5
                if(go_iter % 2): text_y_shift = 15

                ### 標出 最後一個點以外的loss點的數值
                if(go_iter <= (len(iter_indexes) - 1 - 1) ):### 第一個-1 是 len() 轉index， 第二個-1 是 只做到 倒數第二個
                    plt.annotate( text="%.3f" %  iter_values[go_iter],        ### 顯示的文字
                                  xy=(iter_index, iter_values[go_iter]),      ### 要標註的目標點
                                  xytext=( 0 , 3 + text_y_shift),                            ### 顯示的文字放哪裡
                                  textcoords='offset points',                 ### 目前東西放哪裡的坐標系用什麼
                                  arrowprops=dict(arrowstyle="->",            ### 畫箭頭的資訊
                                                  connectionstyle= "arc3",
                                                ))

                ### 標出 最後一個 loss點的數值 
                else:
                    plt.annotate( text="%.3f" %  iter_values[go_iter],        ### 顯示的文字
                                  xy=(iter_index, iter_values[go_iter]),      ### 要標註的目標點
                                  xytext=( 20 , 10),                          ### 顯示的文字放哪裡
                                  textcoords='offset points',                 ### 目前東西放哪裡的坐標系用什麼
                                  arrowprops=dict(arrowstyle="->",            ### 畫箭頭的資訊
                                                  connectionstyle= "arc3",
                                                ))
            plt.tight_layout()
            plt.savefig(self.logs_write_dir + "/" + loss_name + ".png")
            plt.close()
            # print("plot %s loss ok~"%loss_name )
        # print("plot loss ok~")


    def _load_loss_npy_dict(self):
        for loss_name in self.loss_containors.keys():
            self.loss_npy_dict[loss_name] = np.load(self.logs_read_dir + "/" + loss_name + ".npy")

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
    def __init__(self, loss_info_obj=None):
        '''
        一定要注意需不需要 in_obj_copy喔！
        因為雖然 loss_fun 是共用的， 但是每個exp 算出來的 loss_value是不同的！因此 logs_read/write_dir 也是不同的！
        所以還是需要每個 exp 都有自己的 loss_info_obj 喔！才不會 logs_read/write_dir 被共用 這樣子！
        所以 step10_a 的 exp_builder.build() 裡面 呼叫 Loss_info_builder() 更新 loss_info_obj時，in_obj_copy 就要指定 True 拉！
        '''
        if(loss_info_obj is None): self.loss_info_obj = Loss_info()
        else: self.loss_info_obj = loss_info_obj
        # self._build = None

    def set_loss_target(self, loss_target):
        '''
        loss_target的目的 只是 讓儲存 loss 進 logs 時 的一個tag， 讓 UNet1, 2 可以分開資料夾存這樣子喔～
        目前 在training時 還是靠 list 的擺放順序 要和 model 裡 UNet 的擺放順序對應到 來匹配訓練這樣子
        還沒有 高級到用 name 來指定 誰用誰
        '''
        self.loss_info_obj.loss_target = loss_target
        return self

    def set_loss_type(self, loss_type, **args):
        self.loss_info_obj.loss_type = loss_type
        self.args = args
        return self

    def set_logs_dir(self, logs_read_dir, logs_write_dir):
        self.loss_info_obj.logs_read_dir  = f"{logs_read_dir}/{self.loss_info_obj.loss_target}"
        self.loss_info_obj.logs_write_dir = f"{logs_write_dir}/{self.loss_info_obj.loss_target}"
        return self

    def copy(self):
        return copy.deepcopy(self)


class Loss_info_GAN_loss_builder(Loss_info_init_builder):
    def build_gan_loss(self):
        print("self.loss_info_obj.logs_read_dir ~  ~  ~  ~  ", self.loss_info_obj.logs_read_dir)
        print("self.loss_info_obj.logs_write_dir~  ~  ~  ~  ", self.loss_info_obj.logs_write_dir)
        self.loss_info_obj.loss_funs_dict["G"]      = mae_kong
        self.loss_info_obj.loss_funs_dict["G_to_D"] = mse_kong
        self.loss_info_obj.loss_funs_dict["D_Real"] = mse_kong
        self.loss_info_obj.loss_funs_dict["D_Fake"] = mse_kong
        return self.loss_info_obj


    def build_gan_loss_containors(self):
        print("self.loss_info_obj.logs_read_dir ~  ~  ~  ~  ", self.loss_info_obj.logs_read_dir)
        print("self.loss_info_obj.logs_write_dir~  ~  ~  ~  ", self.loss_info_obj.logs_write_dir)
        self.loss_info_obj.loss_containors["1_loss_rec"    ] = tf.keras.metrics.Mean('1_loss_rec'    , dtype=tf.float32)
        self.loss_info_obj.loss_containors["2_loss_g2d"    ] = tf.keras.metrics.Mean('2_loss_g2d'    , dtype=tf.float32)
        self.loss_info_obj.loss_containors["3_g_total_loss"] = tf.keras.metrics.Mean('3_g_total_loss', dtype=tf.float32)
        self.loss_info_obj.loss_containors["4_loss_d_fake" ] = tf.keras.metrics.Mean('4_loss_d_fake' , dtype=tf.float32)
        self.loss_info_obj.loss_containors["5_loss_d_real" ] = tf.keras.metrics.Mean('5_loss_d_real' , dtype=tf.float32)
        self.loss_info_obj.loss_containors["6_d_total_loss"] = tf.keras.metrics.Mean('6_d_total_loss', dtype=tf.float32)
        return self.loss_info_obj


class Loss_info_G_loss_builder(Loss_info_GAN_loss_builder):
    '''
    想多加嘗試 G loss 的話 就在這裡多加 method 就好囉！
    '''
    def _update_loss_describe(self, loss_describe):
        self.loss_info_obj.loss_describe_eles.append(loss_describe)
        self.loss_info_obj.loss_describe = "_".join(self.loss_info_obj.loss_describe_eles)

    ### before ################################################################################################################################################
    def build_mae_loss_fun_and_containor(self):
        if (isinstance(self.args["mae_scale"], int)): loss_describe = "Mae_s%03i"
        else                                        : loss_describe = "Mae_s%.2f"
        loss_describe = loss_describe % self.args["mae_scale"]
        self._update_loss_describe(loss_describe)
        self.loss_info_obj.loss_funs_dict [loss_describe]  = MAE(**self.args)
        self.loss_info_obj.loss_containors[loss_describe] = tf.keras.metrics.Mean(loss_describe, dtype=tf.float32)
        return self

    def build_mse_loss_fun_and_containor(self):
        if (isinstance(self.args["mse_scale"], int)): loss_describe = "Mse_s%03i"
        else                                        : loss_describe = "Mse_s%.2f"
        loss_describe = loss_describe % self.args["mse_scale"]
        self._update_loss_describe(loss_describe)
        self.loss_info_obj.loss_funs_dict[loss_describe]  = mse_kong
        self.loss_info_obj.loss_containors[loss_describe] = tf.keras.metrics.Mean(loss_describe, dtype=tf.float32)
        return self

    ### 2 ################################################################################################################################################
    def build_bce_loss_fun_and_containor(self):
        if (isinstance(self.args["bce_scale"], int)): loss_describe = "Bce_s%03i"
        else                                        : loss_describe = "Bce_s%.2f"
        loss_describe = loss_describe % self.args["bce_scale"]
        self._update_loss_describe(loss_describe)
        self.loss_info_obj.loss_funs_dict [loss_describe] = BCE(**self.args)
        self.loss_info_obj.loss_containors[loss_describe] = tf.keras.metrics.Mean(name=loss_describe, dtype=tf.float32)
        return self

    ### 3 ################################################################################################################################################
    def build_sobel_mae_loss_fun_and_containor(self):
        if (isinstance(self.args["sobel_kernel_scale"], int)): loss_describe = "Sob_k%02i_s%03i"
        else                                                 : loss_describe = "Sob_k%02i_s%.2f"
        loss_describe = loss_describe % (self.args["sobel_kernel_size"], self.args["sobel_kernel_scale"])

        ### 看 erose_M 是什麼設定
        erose_M    = False
        erose_More = False
        if("erose_M"    in self.args.keys()): erose_M    = self.args["erose_M"]
        if("erose_More" in self.args.keys()): erose_More = self.args["erose_More"]
        if  (erose_M is True and erose_More is False): loss_describe += "_EroM"
        elif(erose_M is True and erose_More is True):  loss_describe += "_EroMore"

        self._update_loss_describe(loss_describe)
        self.loss_info_obj.loss_funs_dict [loss_describe] = Sobel_MAE(**self.args)
        self.loss_info_obj.loss_containors[loss_describe] = tf.keras.metrics.Mean(name=loss_describe, dtype=tf.float32)
        return self

    ### 4 ################################################################################################################################################
    def build_tv_loss_fun_and_containor(self):
        if (isinstance(self.args["tv_scale"], int)): loss_describe = "Tv_s%03i"
        else                                       : loss_describe = "Tv_s%.2f"
        loss_describe = loss_describe % self.args["tv_scale"]

        ### 看 erose_M 是什麼設定
        erose_M    = False
        erose_More = False
        if("erose_M"    in self.args.keys()): erose_M    = self.args["erose_M"]
        if("erose_More" in self.args.keys()): erose_More = self.args["erose_More"]
        if  (erose_M is True and erose_More is False): loss_describe += "_EroM"
        elif(erose_M is True and erose_More is True):  loss_describe += "_EroMore"

        self._update_loss_describe(loss_describe)
        self.loss_info_obj.loss_funs_dict ["tv"]  = Total_Variance(**self.args)
        self.loss_info_obj.loss_containors["tv" ] = tf.keras.metrics.Mean(name='tv', dtype=tf.float32)
        return self

    ### 10 ################################################################################################################################################
    def build_GAN_loss_fun_and_containor(self):
        if (isinstance(self.args["GAN_scale"], int)): loss_describe = "GAN_s%03i"
        else                                        : loss_describe = "GAN_s%.2f"
        loss_describe = loss_describe % self.args["GAN_scale"]
        self._update_loss_describe(loss_describe)
        self.loss_info_obj.loss_funs_dict ["BCE_D_fake"] = BCE(bce_scale=self.args["GAN_scale"], **self.args)
        self.loss_info_obj.loss_containors["BCE_D_fake"] = tf.keras.metrics.Mean(name='2_D_fake', dtype=tf.float32)
        self.loss_info_obj.loss_funs_dict ["BCE_D_real"] = BCE(bce_scale=self.args["GAN_scale"], **self.args)
        self.loss_info_obj.loss_containors["BCE_D_real"] = tf.keras.metrics.Mean(name='1_D_real', dtype=tf.float32)
        self.loss_info_obj.loss_funs_dict ["BCE_G_to_D"] = BCE(bce_scale=self.args["GAN_scale"], **self.args)
        self.loss_info_obj.loss_containors["BCE_G_to_D"] = tf.keras.metrics.Mean(name='3_G_to_D', dtype=tf.float32)
        return self

    ### 5 ################################################################################################################################################
    # def build_bce_and_sobel_mae_loss_fun_and_containor(self):
    #     ''' 因為命名關係，要注意先後順序喔 '''
    #     return self

    ### 6 ################################################################################################################################################
    # def build_tv_bce_loss_fun_and_containor(self):
    #     ''' 因為命名關係，要注意先後順序喔 '''
    #     self.build_tv_loss_fun_and_containor()
    #     self.build_bce_loss_fun_and_containor()
    #     return self

    ### 7 ################################################################################################################################################
    # def build_tv_sobel_loss_fun_and_containor(self):
    #     ''' 因為命名關係，要注意先後順序喔 '''
    #     self.build_tv_loss_fun_and_containor()
    #     self.build_sobel_mae_loss_fun_and_containor()
    #     return self

    ### 8 ################################################################################################################################################
    # def build_tv_bce_sobel_k5_loss_fun_and_containor(self):
    #     ''' 因為命名關係，要注意先後順序喔 '''
    #     self.build_tv_loss_fun_and_containor()
    #     self.build_bce_loss_fun_and_containor()
    #     self.build_sobel_mae_loss_fun_and_containor()
    #     return self

class Loss_info_builder(Loss_info_G_loss_builder):
    def build(self):
        # print("self.loss_info_obj.logs_read_dir ~  ~  ~  ~  ", self.loss_info_obj.logs_read_dir)
        # print("self.loss_info_obj.logs_write_dir~  ~  ~  ~  ", self.loss_info_obj.logs_write_dir)
        ''' 多loss 的部分 因為命名關係，要注意先後順序喔 '''
        if(self.loss_info_obj.loss_type   == "mse"):   self.build_mse_loss_fun_and_containor()
        elif(self.loss_info_obj.loss_type == "mae"):   self.build_mae_loss_fun_and_containor()
        ### 2
        elif(self.loss_info_obj.loss_type == "bce"):   self.build_bce_loss_fun_and_containor()
        ### 3
        elif(self.loss_info_obj.loss_type == "sobel"): self.build_sobel_mae_loss_fun_and_containor()
        ### 4
        elif(self.loss_info_obj.loss_type == "tv"):    self.build_tv_loss_fun_and_containor()

        ### 5
        elif(self.loss_info_obj.loss_type == "bce+sobel"):
            self.build_bce_loss_fun_and_containor()
            self.build_sobel_mae_loss_fun_and_containor()
        ### 6
        elif(self.loss_info_obj.loss_type == "tv+bce"):
            self.build_tv_loss_fun_and_containor()
            self.build_bce_loss_fun_and_containor()
        ### 7
        elif(self.loss_info_obj.loss_type == "tv+sobel_k5"):
            self.build_tv_loss_fun_and_containor()
            self.build_sobel_mae_loss_fun_and_containor()
        ### 8
        elif(self.loss_info_obj.loss_type == "tv+bce+sobel_k5"):
            self.build_tv_loss_fun_and_containor()
            self.build_bce_loss_fun_and_containor()
            self.build_sobel_mae_loss_fun_and_containor()

        ### 9
        elif(self.loss_info_obj.loss_type == "mae+sobel"):
            self.build_mae_loss_fun_and_containor()
            self.build_sobel_mae_loss_fun_and_containor()
        ### 10
        elif(self.loss_info_obj.loss_type == "mae+tv"):
            self.build_mae_loss_fun_and_containor()
            self.build_tv_loss_fun_and_containor()
        ### 11
        elif(self.loss_info_obj.loss_type == "mae+sobel+tv"):
            self.build_mae_loss_fun_and_containor()
            self.build_sobel_mae_loss_fun_and_containor()
            self.build_tv_loss_fun_and_containor()

        ### 12
        elif(self.loss_info_obj.loss_type == "bce+mae"):
            self.build_bce_loss_fun_and_containor()
            self.build_mae_loss_fun_and_containor()

        elif(self.loss_info_obj.loss_type == "justG"): self.build_gan_loss()
        # elif(self.loss_info_obj.loss_type == "GAN"):   self.build_gan_loss_containors()
        elif(self.loss_info_obj.loss_type == "GAN"):   self.build_GAN_loss_fun_and_containor()

        return self.loss_info_obj

GAN_s0p1_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  0.1)
GAN_s0p2_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  0.2)
GAN_s0p3_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  0.3)
GAN_s0p5_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  0.5)
GAN_s0p7_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  0.7)
GAN_s0p7_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  0.9)
GAN_s001_loss_info_builder = Loss_info_builder().set_loss_type("GAN", GAN_scale=  1)

### 並不是 model 決定 Loss， 而是由 我想 怎麼設計決定，
# 所以 不能只寫 build_by_model_name，也要寫 我自己指定的method
# 所以 也可以跟 model 一樣 先建好
# 然後還要在 exp 裡面 再次設定喔！
G_mse_s001_loss_info_builder = Loss_info_builder().set_loss_type("mse", mae_scale=  1)

G_mae_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae", mae_scale=  1)
G_mae_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae", mae_scale= 20)
G_mae_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae", mae_scale= 40)
G_mae_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae", mae_scale= 60)
G_mae_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae", mae_scale= 80)
G_mae_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae", mae_scale=100)
##########################################################################################################################################################################
### 2
G_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale=  1)
G_bce_s010_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale= 10)
G_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale= 20)
G_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale= 40)
G_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale= 60)
G_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale= 80)
G_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce", bce_scale=100)
##########################################################################################################################################################################
### 3
G_sobel_k3_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=3, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()

G_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 20)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 40)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 60)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 80)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=100)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=120)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=140)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s160_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=160)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s180_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=180)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s200_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=200)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s220_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=220)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s240_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=240)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_s260_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=260)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k7_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=7, sobel_kernel_scale=260)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k7_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=7, sobel_kernel_scale=260)  #.build_gan_loss().build_gan_loss_containors()

G_sobel_k07_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size= 7, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k09_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size= 9, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k11_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=11, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k13_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=13, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k15_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=15, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k17_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=17, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k19_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=19, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k21_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=21, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k23_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=23, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k25_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=25, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k27_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=27, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k29_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=29, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k31_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=31, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k33_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=33, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k35_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=35, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
### 3-Erose_M
G_sobel_k3_erose_M_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=3, sobel_kernel_scale=  1, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s010_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 10, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s020_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 20, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s040_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 40, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s060_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 60, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s080_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 80, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s100_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=100, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s120_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=120, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s140_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=140, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s160_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=160, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s180_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=180, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s200_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=200, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s220_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=220, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s240_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=240, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_M_s260_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=260, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k7_erose_M_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=7, sobel_kernel_scale=260, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k15_erose_M_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=15, sobel_kernel_scale=  1, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k25_erose_M_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=25, sobel_kernel_scale=  1, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k35_erose_M_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=35, sobel_kernel_scale=  1, erose_M=True)  #.build_gan_loss().build_gan_loss_containors()
### 3-Erose_More
G_sobel_k3_erose_More_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=3, sobel_kernel_scale=  1, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s010_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 10, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s020_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 20, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s040_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 40, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s060_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 60, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s080_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale= 80, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s100_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=100, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s120_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=120, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s140_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=140, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s160_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=160, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s180_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=180, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s200_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=200, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s220_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=220, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s240_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=240, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k5_erose_More_s260_loss_info_builder = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=5, sobel_kernel_scale=260, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k7_erose_More_loss_info_builder      = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=7, sobel_kernel_scale=260, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k15_erose_More_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=15, sobel_kernel_scale=  1, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k25_erose_More_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=25, sobel_kernel_scale=  1, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()
G_sobel_k35_erose_More_loss_info_builder     = Loss_info_builder().set_loss_type("sobel", sobel_kernel_size=35, sobel_kernel_scale=  1, erose_M=True, erose_More=True)  #.build_gan_loss().build_gan_loss_containors()

##########################################################################################################################################################################
### 4
G_tv_loss_info_builder = Loss_info_builder().set_loss_type("tv")  #.build_gan_loss().build_gan_loss_containors()


##########################################################################################################################################################################
### 5
G_bce_sobel_k3_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel")  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel")  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k7_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel")  #.build_gan_loss().build_gan_loss_containors()

G_bce_sobel_k5_s20_loss_info_builder  = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale= 20)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s40_loss_info_builder  = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale= 40)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s60_loss_info_builder  = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale= 60)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s80_loss_info_builder  = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale= 80)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale=100)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale=120)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale=140)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s160_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale=160)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k5_s180_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=5, sobel_kernel_scale=180)  #.build_gan_loss().build_gan_loss_containors()
G_bce_sobel_k7_s780_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", sobel_kernel_size=7, sobel_kernel_scale=780)  #.build_gan_loss().build_gan_loss_containors()

G_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=100)
G_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 20, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 20, sobel_kernel_size=5, sobel_kernel_scale=100)
G_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 40, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 40, sobel_kernel_size=5, sobel_kernel_scale=100)
G_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 60, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 60, sobel_kernel_size=5, sobel_kernel_scale=100)
G_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 80, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale= 80, sobel_kernel_size=5, sobel_kernel_scale=100)
G_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=100, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=100, sobel_kernel_size=5, sobel_kernel_scale=100)

### 大kernel測試
G_bce_s001_sobel_k05_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size= 5, sobel_kernel_scale=  1)
G_bce_s001_sobel_k07_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size= 7, sobel_kernel_scale=  1)
G_bce_s001_sobel_k09_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size= 9, sobel_kernel_scale=  1)
G_bce_s001_sobel_k11_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=11, sobel_kernel_scale=  1)
G_bce_s001_sobel_k13_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=13, sobel_kernel_scale=  1)
G_bce_s001_sobel_k15_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=15, sobel_kernel_scale=  1)
G_bce_s001_sobel_k17_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=17, sobel_kernel_scale=  1)
G_bce_s001_sobel_k19_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=19, sobel_kernel_scale=  1)
G_bce_s001_sobel_k21_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=21, sobel_kernel_scale=  1)
G_bce_s001_sobel_k23_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=23, sobel_kernel_scale=  1)
G_bce_s001_sobel_k25_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=25, sobel_kernel_scale=  1)
G_bce_s001_sobel_k27_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=27, sobel_kernel_scale=  1)
G_bce_s001_sobel_k29_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=29, sobel_kernel_scale=  1)
G_bce_s001_sobel_k31_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=31, sobel_kernel_scale=  1)
G_bce_s001_sobel_k33_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=33, sobel_kernel_scale=  1)
G_bce_s001_sobel_k35_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=35, sobel_kernel_scale=  1)

### Erose_M ， 注意 不能給 I_to_M 用喔， 是給 I_to_M 之後的 model 用的
G_bce_s001_sobel_k05_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size= 5, sobel_kernel_scale=  1, erose_M=True)
G_bce_s001_sobel_k15_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=15, sobel_kernel_scale=  1, erose_M=True)
G_bce_s001_sobel_k25_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=25, sobel_kernel_scale=  1, erose_M=True)
G_bce_s001_sobel_k35_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("bce+sobel", bce_scale=  1, sobel_kernel_size=35, sobel_kernel_scale=  1, erose_M=True)
##########################################################################################################################################################################
### 6
G_tv_s0p1_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 0.1, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s01_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 1, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 4, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s08_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 8, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s12_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=12, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s16_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=16, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s20_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=20, bce_scale= 1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_bce_s020_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=20, bce_scale=20)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_bce_s040_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=20, bce_scale=40)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_bce_s060_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=20, bce_scale=60)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_bce_s080_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=20, bce_scale=80)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=30, bce_scale= 1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_bce_s020_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=30, bce_scale=20)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_bce_s040_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=30, bce_scale=40)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_bce_s060_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=30, bce_scale=60)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_bce_s080_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=30, bce_scale=80)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s40_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=40, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s60_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=60, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s80_bce_s001_loss_info_builder  = Loss_info_builder().set_loss_type("tv+bce", tv_scale=80, bce_scale=1)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s001_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=  1, bce_scale=  1)
G_tv_s001_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=  1, bce_scale= 20)
G_tv_s001_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=  1, bce_scale= 40)
G_tv_s001_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=  1, bce_scale= 60)
G_tv_s001_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=  1, bce_scale= 80)
G_tv_s001_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=  1, bce_scale=100)
G_tv_s020_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale=  1)
G_tv_s020_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale= 20)
G_tv_s020_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale= 40)
G_tv_s020_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale= 60)
G_tv_s020_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale= 80)
G_tv_s020_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale=100)
G_tv_s020_bce_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale=120)
G_tv_s020_bce_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 20, bce_scale=140)
G_tv_s040_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale=  1)
G_tv_s040_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale= 20)
G_tv_s040_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale= 40)
G_tv_s040_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale= 60)
G_tv_s040_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale= 80)
G_tv_s040_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale=100)
G_tv_s040_bce_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale=120)
G_tv_s040_bce_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 40, bce_scale=140)
G_tv_s060_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale=  1)
G_tv_s060_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale= 20)
G_tv_s060_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale= 40)
G_tv_s060_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale= 60)
G_tv_s060_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale= 80)
G_tv_s060_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale=100)
G_tv_s060_bce_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale=120)
G_tv_s060_bce_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale=140)
G_tv_s060_bce_s160_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale=160)
G_tv_s060_bce_s180_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 60, bce_scale=180)
G_tv_s080_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale=  1)
G_tv_s080_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale= 20)
G_tv_s080_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale= 40)
G_tv_s080_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale= 60)
G_tv_s080_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale= 80)
G_tv_s080_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale=100)
G_tv_s080_bce_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale=120)
G_tv_s080_bce_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale=140)
G_tv_s080_bce_s160_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale=160)
G_tv_s080_bce_s180_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale= 80, bce_scale=180)
G_tv_s100_bce_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=  1)
G_tv_s100_bce_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale= 20)
G_tv_s100_bce_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale= 40)
G_tv_s100_bce_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale= 60)
G_tv_s100_bce_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale= 80)
G_tv_s100_bce_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=100)
G_tv_s100_bce_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=120)
G_tv_s100_bce_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=140)
G_tv_s100_bce_s160_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=160)
G_tv_s100_bce_s180_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=180)
G_tv_s100_bce_s200_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce", tv_scale=100, bce_scale=200)


##########################################################################################################################################################################
### 7
G_tv_s01_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=1, sobel_kernel_size=5, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s01_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=1, sobel_kernel_size=5, sobel_kernel_scale= 80)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s01_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=1, sobel_kernel_size=5, sobel_kernel_scale=100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s01_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=1, sobel_kernel_size=5, sobel_kernel_scale=120)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s01_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=1, sobel_kernel_size=5, sobel_kernel_scale=140)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=4, sobel_kernel_size=5, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=4, sobel_kernel_size=5, sobel_kernel_scale= 80)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=4, sobel_kernel_size=5, sobel_kernel_scale=100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=4, sobel_kernel_size=5, sobel_kernel_scale=120)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=4, sobel_kernel_size=5, sobel_kernel_scale=140)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s20_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=20, sobel_kernel_size=5, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=20, sobel_kernel_size=5, sobel_kernel_scale= 80)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=20, sobel_kernel_size=5, sobel_kernel_scale=100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=20, sobel_kernel_size=5, sobel_kernel_scale=120)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s20_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=20, sobel_kernel_size=5, sobel_kernel_scale=140)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=30, sobel_kernel_size=5, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=30, sobel_kernel_size=5, sobel_kernel_scale= 80)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=30, sobel_kernel_size=5, sobel_kernel_scale=100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=30, sobel_kernel_size=5, sobel_kernel_scale=120)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s30_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=30, sobel_kernel_size=5, sobel_kernel_scale=140)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_tv_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_tv_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_tv_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_tv_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_tv_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=100)
G_tv_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 20, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_tv_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_tv_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_tv_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_tv_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_tv_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 20, sobel_kernel_size=5, sobel_kernel_scale=100)
G_tv_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 40, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_tv_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_tv_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_tv_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_tv_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_tv_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 40, sobel_kernel_size=5, sobel_kernel_scale=100)
G_tv_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 60, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_tv_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_tv_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_tv_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_tv_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_tv_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 60, sobel_kernel_size=5, sobel_kernel_scale=100)
G_tv_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 80, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_tv_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_tv_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_tv_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_tv_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_tv_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale= 80, sobel_kernel_size=5, sobel_kernel_scale=100)
G_tv_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=100, sobel_kernel_size=5, sobel_kernel_scale=  1)
G_tv_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 20)
G_tv_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 40)
G_tv_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 60)
G_tv_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 80)
G_tv_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+sobel_k5", tv_scale=100, sobel_kernel_size=5, sobel_kernel_scale=100)
##########################################################################################################################################################################
### 8
G_tv_bce_sobel_k5_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=1, sobel_kernel_size=5, sobel_kernel_scale=  1)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s01_bce_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 1, sobel_kernel_size=5, sobel_kernel_scale= 100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_bce_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 4, sobel_kernel_size=5, sobel_kernel_scale=   1)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s04_bce_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 4, sobel_kernel_size=5, sobel_kernel_scale= 100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s08_bce_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 8, sobel_kernel_size=5, sobel_kernel_scale= 100)  #.build_gan_loss().build_gan_loss_containors()
G_tv_s12_bce_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=12, sobel_kernel_size=5, sobel_kernel_scale= 100)  #.build_gan_loss().build_gan_loss_containors()

G_tv_s01_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s01_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s01_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s01_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s01_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s01_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s01_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s01_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s01_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s01_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s01_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s01_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s01_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s01_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s01_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s01_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s01_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s01_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s01_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s01_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s01_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s01_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s01_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s01_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s01_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s01_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s01_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s01_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s01_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s01_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s01_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s01_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s01_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s01_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s01_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s01_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  1, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s04_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s04_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s04_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s04_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s04_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s04_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s04_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s04_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s04_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s04_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s04_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s04_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s04_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s04_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s04_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s04_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s04_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s04_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s04_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s04_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s04_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s04_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s04_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s04_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s04_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s04_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s04_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s04_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s04_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s04_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s04_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s04_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s04_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s04_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s04_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s04_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  4, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s08_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s08_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s08_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s08_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s08_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s08_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s08_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s08_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s08_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s08_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s08_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s08_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s08_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s08_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s08_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s08_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s08_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s08_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s08_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s08_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s08_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s08_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s08_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s08_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s08_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s08_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s08_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s08_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s08_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s08_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s08_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s08_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s08_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s08_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s08_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s08_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=  8, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s12_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s12_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s12_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s12_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s12_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s12_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s12_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s12_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s12_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s12_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s12_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s12_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s12_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s12_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s12_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s12_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s12_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s12_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s12_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s12_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s12_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s12_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s12_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s12_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s12_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s12_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s12_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s12_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s12_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s12_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s12_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s12_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s12_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s12_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s12_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s12_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 12, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s20_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s20_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s20_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s20_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s20_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s20_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s20_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s20_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s20_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s20_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s20_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s20_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s20_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s20_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s20_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s20_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s20_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s20_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s20_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s20_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s20_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s20_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s20_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s20_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s20_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s20_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s20_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s20_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s20_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s20_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s20_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s20_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s20_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s20_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s20_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s20_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 20, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s30_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s30_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s30_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s30_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s30_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s30_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s30_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s30_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s30_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s30_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s30_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s30_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s30_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s30_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s30_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s30_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s30_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s30_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s30_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s30_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s30_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s30_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s30_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s30_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s30_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s30_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s30_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s30_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s30_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s30_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s30_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s30_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s30_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s30_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s30_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s30_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 30, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s40_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s40_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s40_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s40_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s40_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s40_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s40_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s40_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s40_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s40_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s40_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s40_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s40_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s40_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s40_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s40_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s40_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s40_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s40_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s40_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s40_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s40_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s40_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s40_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s40_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s40_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s40_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s40_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s40_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s40_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s40_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s40_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s40_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s40_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s40_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s40_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 40, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s60_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s60_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s60_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s60_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s60_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s60_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s60_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s60_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s60_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s60_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s60_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s60_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s60_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s60_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s60_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s60_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s60_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s60_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s60_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s60_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s60_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s60_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s60_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s60_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s60_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s60_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s60_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s60_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s60_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s60_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s60_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s60_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s60_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s60_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s60_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s60_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 60, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s80_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s80_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s80_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s80_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s80_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s80_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s80_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s80_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s80_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s80_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s80_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s80_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s80_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s80_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s80_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s80_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s80_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s80_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s80_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s80_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s80_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s80_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s80_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s80_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s80_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s80_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s80_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s80_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s80_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s80_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s80_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s80_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s80_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s80_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s80_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s80_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale= 80, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
#########################################################################################################################################################
G_tv_s100_bce_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s100_bce_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s100_bce_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s100_bce_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s100_bce_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s100_bce_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=   1, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s100_bce_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s100_bce_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s100_bce_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s100_bce_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s100_bce_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s100_bce_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  20, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s100_bce_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s100_bce_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s100_bce_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s100_bce_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s100_bce_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s100_bce_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  40, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s100_bce_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s100_bce_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s100_bce_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s100_bce_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s100_bce_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s100_bce_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  60, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s100_bce_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s100_bce_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s100_bce_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s100_bce_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s100_bce_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s100_bce_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale=  80, sobel_kernel_size=5, sobel_kernel_scale= 100)
G_tv_s100_bce_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=   1)
G_tv_s100_bce_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  20)
G_tv_s100_bce_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  40)
G_tv_s100_bce_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  60)
G_tv_s100_bce_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale=  80)
G_tv_s100_bce_s100_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("tv+bce+sobel_k5", tv_scale=100, bce_scale= 100, sobel_kernel_size=5, sobel_kernel_scale= 100)
##########################################################################################################################################################################
### 9
mae_s001_sobel_k3_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=3, sobel_kernel_scale=  1)
mae_s001_sobel_k7_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=7, sobel_kernel_scale=  1)
mae_s001_sobel_k9_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=9, sobel_kernel_scale=  1)
mae_s001_sobel_k7_s780_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=7, sobel_kernel_scale=780)

mae_s001_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1)
mae_s001_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 20)
mae_s001_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 40)
mae_s001_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 60)
mae_s001_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale= 80)
mae_s001_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=100)
mae_s001_sobel_k5_s120_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=120)
mae_s001_sobel_k5_s140_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=140)
mae_s001_sobel_k5_s160_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=160)
mae_s001_sobel_k5_s180_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=180)

mae_s020_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 20, sobel_kernel_size=5, sobel_kernel_scale=  1)
mae_s020_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 20)
mae_s020_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 40)
mae_s020_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 60)
mae_s020_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 20, sobel_kernel_size=5, sobel_kernel_scale= 80)
mae_s020_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 20, sobel_kernel_size=5, sobel_kernel_scale=100)

mae_s040_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 40, sobel_kernel_size=5, sobel_kernel_scale=  1)
mae_s040_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 20)
mae_s040_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 40)
mae_s040_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 60)
mae_s040_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 40, sobel_kernel_size=5, sobel_kernel_scale= 80)
mae_s040_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 40, sobel_kernel_size=5, sobel_kernel_scale=100)

mae_s060_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 60, sobel_kernel_size=5, sobel_kernel_scale=  1)
mae_s060_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 20)
mae_s060_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 40)
mae_s060_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 60)
mae_s060_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 60, sobel_kernel_size=5, sobel_kernel_scale= 80)
mae_s060_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 60, sobel_kernel_size=5, sobel_kernel_scale=100)

mae_s080_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 80, sobel_kernel_size=5, sobel_kernel_scale=  1)
mae_s080_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 20)
mae_s080_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 40)
mae_s080_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 60)
mae_s080_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 80, sobel_kernel_size=5, sobel_kernel_scale= 80)
mae_s080_sobel_k5_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale= 80, sobel_kernel_size=5, sobel_kernel_scale=100)

mae_s100_sobel_k5_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=100, sobel_kernel_size=5, sobel_kernel_scale=  1)
mae_s100_sobel_k5_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 20)
mae_s100_sobel_k5_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 40)
mae_s100_sobel_k5_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 60)
mae_s100_sobel_k5_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=100, sobel_kernel_size=5, sobel_kernel_scale= 80)

mae_s001_sobel_k05_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True)
mae_s001_sobel_k15_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True)
mae_s001_sobel_k25_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True)
mae_s001_sobel_k35_s001_EroseM_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True)

mae_s001_sobel_k05_s001_EroseMore_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True, erose_More=True)
mae_s001_sobel_k15_s001_EroseMore_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True, erose_More=True)
mae_s001_sobel_k25_s001_EroseMore_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True, erose_More=True)
mae_s001_sobel_k35_s001_EroseMore_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel", mae_scale=  1, sobel_kernel_size=5, sobel_kernel_scale=  1, erose_M=True, erose_More=True)
##########################################################################################################################################################################
### 10
mae_s001_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale=  1)
mae_s001_tv_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale= 20)
mae_s001_tv_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale= 40)
mae_s001_tv_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale= 60)
mae_s001_tv_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale= 80)
mae_s001_tv_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale=100)
mae_s001_tv_s120_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale=120)
mae_s001_tv_s140_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale=140)
mae_s001_tv_s160_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale=160)
mae_s001_tv_s180_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=  1, tv_scale=180)

mae_s020_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 20, tv_scale=  1)
mae_s020_tv_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 20, tv_scale= 20)
mae_s020_tv_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 20, tv_scale= 40)
mae_s020_tv_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 20, tv_scale= 60)
mae_s020_tv_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 20, tv_scale= 80)
mae_s020_tv_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 20, tv_scale=100)

mae_s040_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 40, tv_scale=  1)
mae_s040_tv_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 40, tv_scale= 20)
mae_s040_tv_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 40, tv_scale= 40)
mae_s040_tv_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 40, tv_scale= 60)
mae_s040_tv_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 40, tv_scale= 80)
mae_s040_tv_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 40, tv_scale=100)

mae_s060_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 60, tv_scale=  1)
mae_s060_tv_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 60, tv_scale= 20)
mae_s060_tv_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 60, tv_scale= 40)
mae_s060_tv_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 60, tv_scale= 60)
mae_s060_tv_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 60, tv_scale= 80)
mae_s060_tv_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 60, tv_scale=100)

mae_s080_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 80, tv_scale=  1)
mae_s080_tv_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 80, tv_scale= 20)
mae_s080_tv_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 80, tv_scale= 40)
mae_s080_tv_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 80, tv_scale= 60)
mae_s080_tv_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 80, tv_scale= 80)
mae_s080_tv_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale= 80, tv_scale=100)

mae_s100_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=100, tv_scale=  1)
mae_s100_tv_s020_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=100, tv_scale= 20)
mae_s100_tv_s060_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=100, tv_scale= 40)
mae_s100_tv_s040_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=100, tv_scale= 60)
mae_s100_tv_s080_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=100, tv_scale= 80)
mae_s100_tv_s100_loss_info_builder = Loss_info_builder().set_loss_type("mae+tv", mae_scale=100, tv_scale=100)

##########################################################################################################################################################################
### 10
mae_s001_sobel_k5_s001_tv_s001_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel+tv", mae_scale=  1, sobel_kernel_size=3, sobel_kernel_scale=  1, tv_scale=  1)
mae_s0p1_sobel_k5_s0p1_tv_s0p1_loss_info_builder = Loss_info_builder().set_loss_type("mae+sobel+tv", mae_scale=0.1, sobel_kernel_size=3, sobel_kernel_scale=0.1, tv_scale=0.1)

##########################################################################################################################################################################
### 12
bce_s001_mae_s001_loss_info_builder = Loss_info_builder().set_loss_type("bce+mae", bce_scale=1, mae_scale=  1)

GAN_mae_loss_info                  = Loss_info_builder().set_loss_type("justG")  #.build_gan_loss().build_gan_loss_containors()

if(__name__ == "__main__"):
    # from step09_d_KModel_builder_combine_step789 import MODEL_NAME
    # loss_info_obj = Loss_info_builder().set_logs_dir(logs_read_dir="abc", logs_write_dir="abc").build_mse_loss_fun_and_containor().build()
    # print(Loss_info_builder().set_logs_dir(logs_read_dir="abc", logs_write_dir="abc").build_mse_loss_fun_and_containor().build())
    # print(loss_info_obj.loss_containors)
    # print(loss_info_obj.summary_writer)

    # loss_info_b1 = Loss_info_builder().set_logs_dir(logs_read_dir="abc", logs_write_dir="abc").build_mse_loss_fun_and_containor()
    # loss_info_b2 = loss_info_b1.copy().set_logs_dir(logs_read_dir="def", logs_write_dir="def")   ### 如果 不copy() 的話，原本的 "abc" 會被改調喔！
    # print(loss_info_b1.loss_info_obj.logs_read_dir)
    # print(loss_info_b2.loss_info_obj.logs_read_dir)
    # loss_info_obj = Loss_info_builder().set_logs_dir(logs_read_dir="abc", logs_write_dir="abc").set_loss_type("mse").build()  #.build_mse_loss_fun_and_containor().build()

    loss_info_obj = G_bce_sobel_k7_s780_loss_info_builder.build()
    print(loss_info_obj.loss_funs_dict["mask_Sobel_MAE"].sobel_kernel_scale)
    loss_info_obj = G_bce_sobel_k7_loss_info_builder.build()
    print(loss_info_obj.loss_funs_dict["mask_Sobel_MAE"].sobel_kernel_scale)


    print("cost time:", time.time() - start_time)
