import tensorflow as tf
from step08_model_obj import MODEL_NAME
import matplotlib.pyplot as plt 
import numpy as np 
class Board:
    def __init__(self):
        self.logs_dir = None 
        self.summary_writer = None
        self.losses = {}

    def see_loss(self, epochs):
        for loss_name in self.losses.keys():
            plt.figure(figsize=(20,6))                              ### 建立畫布
            plt.ylim(0,1.3)
            plt.ylabel(loss_name)
            y_loss_array = np.load(self.logs_dir + "/" + loss_name + ".npy")
            
            plt.xlim(0,epochs)
            plt.xlabel("epoch_num")
            x_epoch = np.arange(len(y_loss_array))

            plt.plot(x_epoch, y_loss_array)
            plt.savefig(self.logs_dir + "/" + loss_name + ".png")
            plt.close()
            # print("plot %s loss ok~"%loss_name )
        print("plot loss ok~" )

class Board_init_builder:
    def __init__(self, board=None):
        if(board is None):
            self.board = Board()
        else:
            self.board = board
        
    def set_logs_dir_and_summary_writer(self, logs_dir):
        self.board.logs_dir = logs_dir
        self.board.summary_writer = tf.summary.create_file_writer( logs_dir ) ### 建tensorboard，這會自動建資料夾喔！
        return self

    def build(self):
        return self.board

class Board_unet_builder(Board_init_builder):
    def build_unet_board(self):
        self.board.losses["gen_l1_loss" ]= tf.keras.metrics.Mean('gen_l1_loss', dtype=tf.float32)
        return self

class Board_rect_builder(Board_unet_builder):
    def build_rect_board(self):
        self.board.losses["1_loss_rec"    ]= tf.keras.metrics.Mean('1_loss_rec'    , dtype=tf.float32)
        self.board.losses["2_loss_g2d"    ]= tf.keras.metrics.Mean('2_loss_g2d'    , dtype=tf.float32)
        self.board.losses["3_g_total_loss"]= tf.keras.metrics.Mean('3_g_total_loss', dtype=tf.float32)
        self.board.losses["4_loss_d_fake" ]= tf.keras.metrics.Mean('4_loss_d_fake' , dtype=tf.float32)
        self.board.losses["5_loss_d_real" ]= tf.keras.metrics.Mean('5_loss_d_real' , dtype=tf.float32)
        self.board.losses["6_d_total_loss"]= tf.keras.metrics.Mean('6_d_total_loss', dtype=tf.float32)
        return self

class Board_just_G_builder(Board_rect_builder):
    def build_just_G_board(self):
        self.board.losses["loss_rec"    ]= tf.keras.metrics.Mean('loss_rec'    , dtype=tf.float32)
        return self


class Board_builder(Board_just_G_builder):
    def build_by_model_name(self, model_name):
        if  (model_name==MODEL_NAME.unet)     : self.build_unet_board()
        elif(model_name==MODEL_NAME.rect or
             model_name==MODEL_NAME.mrf_rect ): self.build_rect_board()
        elif(model_name==MODEL_NAME.just_G   ): self.build_just_G_board()
        
        return self

if(__name__=="__main__"):
    board_obj = Board_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_by_model_name(MODEL_NAME.rect).build()
    print(board_obj.losses)
    print(board_obj.summary_writer)