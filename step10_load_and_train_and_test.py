import tensorflow as tf
import numpy as np
from enum import Enum
import time

from step06_b_data_pipline import tf_Data_builder
from step08_b_model_obj import MODEL_NAME
from step09_board_obj import Board_builder
from step11_a2_result_obj import Result
from step11_b_result_obj_builder import Result_builder
import sys
sys.path.append("kong_util")
from util import time_util

from tqdm import tqdm


class Experiment():
    def step0_save_code(self):
        import shutil
        from build_dataset_combine import Check_dir_exist_and_build
        code_dir = self.result_obj.result_dir + "/" + "train_code"
        Check_dir_exist_and_build(code_dir)
        shutil.copy("step06_a_datas_obj.py"             , code_dir + "/" + "step06_a_datas_obj.py")
        shutil.copy("step06_b_data_pipline.py"          , code_dir + "/" + "step06_b_data_pipline.py")
        shutil.copy("step08_a_1_UNet.py"                , code_dir + "/" + "step08_a_1_UNet.py")
        shutil.copy("step08_a_1_UNet_512to256.py"       , code_dir + "/" + "step7_kong_model1_UNet.py")
        shutil.copy("step08_a_2_Rect2.py"               , code_dir + "/" + "step08_a_2_Rect2.py")
        shutil.copy("step08_a_3_justG.py"               , code_dir + "/" + "step08_a_3_justG.py")
        shutil.copy("step08_b_model_obj.py"             , code_dir + "/" + "step08_b_model_obj.py")
        shutil.copy("step09_board_obj.py"               , code_dir + "/" + "step09_board_obj.py")
        shutil.copy("step10_load_and_train_and_test.py" , code_dir + "/" + "step10_load_and_train_and_test.py")
        shutil.copy("step11_a1_see_obj.py"              , code_dir + "/" + "step11_a1_see_obj.py")
        shutil.copy("step11_a2_result_obj.py"           , code_dir + "/" + "step11_a2_result_obj.py")
        shutil.copy("step11_b_result_obj_builder.py"    , code_dir + "/" + "step11_b_result_obj_builder.py")
        shutil.copy("step11_c_result_instance.py"       , code_dir + "/" + "step11_c_result_instance.py")
        shutil.copy("step11_d_result_do_something.py"   , code_dir + "/" + "step11_d_result_do_something.py")
        shutil.copy("step12_result_analyzer.py"         , code_dir + "/" + "step12_result_analyzer.py")

################################################################################################################################################
################################################################################################################################################
    def __init__(self):
        self.phase        = "train"
        self.db_obj       = None
        self.model_obj    = None
        self.exp_dir      = None
        self.describe_mid = None
        self.describe_end = "try_try_try_enum"
        self.result_name  = None
        self.result_obj   = None
        ##############################################################################################################################
        ### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數
        ### train, train_reload 參數
        self.batch_size      = 1
        self.train_shuffle   = True
        self.epochs          = 1300  ### 看opencv合成的video覺得1300左右就沒變了
        self.epoch_down_step = 100   ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
        self.epoch_save_freq = 1     ### 訓練 epoch_save_freq 個 epoch 存一次模型
        self.start_epoch     = 0


        # self.phase = "train_reload" ### 要記得去決定 result_name 喔！
        # self.phase = "test"         ### test是用固定 train/test 資料夾架構的讀法 ### 要記得去決定 result_name 喔！
        ### self.phase = "test_indicate" ### 這應該是可以刪掉了，因為在取db的時候，已經有包含了test_indecate的取法了！不用在特別區分出來 取資料囉！
        ### 參數設定結束
        ####################################################################################################################
        self.tf_data      = None
        self.ckpt_manager = None

################################################################################################################################################
################################################################################################################################################
    def train_init(self, train_reload=False):  ### 1.result, 2.data, 3.model(reload), 4.board, 5.save_code
        ### 1.result
        if(train_reload):
            print("self.exp_dir", self.exp_dir)
            print("self.result_name", self.result_name)
            self.result_obj = Result_builder().set_by_result_name(self.exp_dir + "/" + self.result_name).build()  ### 直接用 自己指定好的 result_name
        else:             self.result_obj = Result_builder().set_by_exp(self).build()  ### 需要 db_obj 和 exp本身的describe_mid/end
        ### 2.data，在這邊才建立而不在step6_b 就先建好是因為 要參考 model_name 來決定如何 resize 喔！
        self.tf_data      = tf_Data_builder().set_basic(self.db_obj).set_img_resize(self.model_obj.model_name).build_by_db_get_method().build()  ### tf_data 抓資料
        ### 3.model
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_dir, max_to_keep=2)  ###step4 建立checkpoint manager 設定最多存2份
        if(train_reload):  ### 看需不需要reload model
            self.model_obj.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()
            print("reload ok~~ start_epoch=", self.start_epoch)

        ####################################################################################################################
        ### 4.board, 5.save_code；train時才需要 board_obj 和 把code存起來喔！test時不用～
        self.board_obj = Board_builder().set_logs_dir_and_summary_writer(self.result_obj.logs_dir).build_by_model_name(self.model_obj.model_name).build()  ###step3 建立tensorboard，只有train 和 train_reload需要
        self.step0_save_code()  ### 把source code存起來

    def train_reload(self):
        self.train(train_reload=True)

    def train(self, train_reload=False):
        self.train_init(train_reload)
        ################################################################################################################################################
        ### 第三階段：train 和 test
        ###  training 的部分 ###################################################################################################
        ###     以下的概念就是，每個模型都有自己的 generate_results 和 train_step，根據model_name 去各別import 各自的 function過來用喔！
        total_start = time.time()

        ### 多這 這段if 是因為 unet 有move_map的部分，所以要多做以下操作 把 move_map相關會用到的東西存起來
        if("unet" in self.model_obj.model_name.value and "flow" not in self.model_obj.model_name.value):
            from util import get_max_db_move_xy
            self.model_obj.ckpt.max_train_move.assign(self.tf_data.max_train_move)  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            self.model_obj.ckpt.min_train_move.assign(self.tf_data.min_train_move)  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            max_db_move_x, max_db_move_y = get_max_db_move_xy(db_dir=self.db_obj.category, db_name=self.db_obj.db_name)  ### g生成的結果 做 apply_rec_move用
            self.model_obj.ckpt.max_db_move_x.assign(max_db_move_x)  ### 在test時 rec_img需要，所以要存起來
            self.model_obj.ckpt.max_db_move_y.assign(max_db_move_y)  ### 在test時 rec_img需要，所以要存起來
            self.ckpt_manager.save()
            print("save ok ~~~~~~~~~~~~~~~~~")

        for epoch in range(self.start_epoch, self.epochs):
            ###############################################################################################################################
            ###    step0 紀錄epoch開始訓練的時間
            epoch_start_timestamp = time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime())
            print("Epoch: ", epoch, "start at", epoch_start_timestamp)
            e_start = time.time()
            ###############################################################################################################################
            ###    step0 設定learning rate
            lr = 0.0002 if epoch < self.epoch_down_step else 0.0002 * (self.epochs - epoch) / (self.epochs - self.epoch_down_step)
            self.model_obj.optimizer_G.lr = lr
            ###############################################################################################################################
            if(epoch == 0): print("Initializing Model~~~")  ### sample的時候就會initial model喔！
            ###############################################################################################################################
            ###     step1 用來看目前訓練的狀況
            self.train_step1_see_current_img(epoch)
            ###############################################################################################################################
            ###     step2 訓練
            for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(self.tf_data.train_db_combine)):
                self.model_obj.train_step(self.model_obj, train_in_pre, train_gt_pre, self.board_obj)
            ###############################################################
            ###     step3 整個epoch 的 loss 算平均，存進tensorboard
            self.train_step3_board_save_loss(epoch)
            ###############################################################################################################################
            ###     step4 儲存模型 (checkpoint) the model every "epoch_save_freq" epochs
            if (epoch + 1) % self.epoch_save_freq == 0:
                # print("save epoch_log :", epoch + 1)
                self.model_obj.ckpt.epoch_log.assign(epoch + 1)  ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                self.ckpt_manager.save()
                print("save ok ~~~~~~~~~~~~~~~~~")
            ###############################################################################################################################
            ###    step5 紀錄、顯示 訓練相關的時間
            self.train_step5_show_time(epoch, e_start, total_start, epoch_start_timestamp)

    def train_step1_see_current_img(self, epoch):
        # sample_start_time = time.time()
        see_in_pre = self.tf_data.test_in_db_pre
        see_gt     = self.tf_data.test_gt_db
        see_amount = 1
        if(self.db_obj.have_see):
            see_in_pre = self.tf_data.see_in_db_pre
            see_gt     = self.tf_data.see_gt_db
            see_amount = self.tf_data.see_amount

        for see_index, (test_in_pre, test_gt) in enumerate(tqdm(zip(see_in_pre.take(see_amount), see_gt.take(see_amount)))):
            if  ("unet"  in self.model_obj.model_name.value and "flow" not in self.model_obj.model_name.value): 
                self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in_pre, test_gt, self.tf_data.max_train_move, self.tf_data.min_train_move, epoch, result_obj.result_dir, result_obj)  ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif("rect"  in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.rect.generator, see_index, test_in_pre, test_gt, epoch, self.result_obj)
            elif("justG" in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in_pre, test_gt, epoch, self.result_obj)
            elif("flow"  in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in_pre, test_gt, epoch, self.result_obj)

        # self.result_obj.save_all_single_see_as_matplot_visual_multiprocess() ### 不行這樣搞，對當掉！但可以分開用別的python執行喔～
        # print("sample all see time:", time.time()-sample_start_time)

    def train_step3_board_save_loss(self, epoch):
        with self.board_obj.summary_writer.as_default():
            for loss_name, loss_containor in self.board_obj.losses.items():
                tf.summary.scalar(loss_name, loss_containor.result(), step=epoch)
                loss_value = loss_containor.result().numpy()
                if(epoch == 0):  ### 第一次 直接把值存成np.array
                    np.save(self.result_obj.logs_dir + "/" + loss_name, np.array(loss_value.reshape(1)))

                else:  ### 第二次後，先把np.array先讀出來append值後 再存進去
                    loss_array = np.load(self.result_obj.logs_dir + "/" + loss_name + ".npy")
                    loss_array = loss_array[:epoch]   ### 這是為了防止 如果程式在 step3,4之間中斷 這種 loss已經存完 但 model還沒存 的狀況，loss 會比想像中的多一步，所以加這行防止這種情況發生喔
                    loss_array = np.append(loss_array, loss_value)
                    np.save(self.result_obj.logs_dir + "/" + loss_name, np.array(loss_array))
                    # print(loss_array)

        ###    reset tensorboard 的 loss紀錄容器
        for loss_containor in self.board_obj.losses.values():
            loss_containor.reset_states()
        ###############################################################
        self.board_obj.see_loss(self.epochs)  ### 把 loss資訊 用 matplot畫出來
        ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了 # self.result_obj.Draw_loss_during_train(epoch, self.epochs)  ### 在 train step1 generate_see裡已經把see的 matplot_visual圖畫出來了，再把 loss資訊加進去

    def train_step5_show_time(self, epoch, e_start, total_start, epoch_start_timestamp):
        epoch_cost_time = time.time() - e_start
        total_cost_time = time.time() - total_start
        print('epoch %i start at:%s'         % (epoch, epoch_start_timestamp))
        print('epoch %i cost time:%.2f'      % (epoch, epoch_cost_time))
        print("batch cost time:%.2f average" % (epoch_cost_time / self.tf_data.train_amount))
        print("total cost time:%s"           % (time_util(total_cost_time)))
        print("esti total time:%s"           % (time_util(epoch_cost_time * self.epochs)))
        print("esti least time:%s"           % (time_util(epoch_cost_time * (self.epochs - (epoch + 1)))))
        print("")
        with open(self.result_obj.result_dir + "/" + "cost_time.txt", "a") as f:
            f.write(self.phase)                                                                                  ; f.write("\n")
            f.write('epoch %i start at:%s'         % (epoch, epoch_start_timestamp))                             ; f.write("\n")
            f.write('epoch cost time:%.2f'         % (epoch_cost_time))                                          ; f.write("\n")
            f.write("batch cost time:%.2f average" % (epoch_cost_time / self.tf_data.train_amount))              ; f.write("\n")
            f.write("total cost time:%s"           % (time_util(total_cost_time)))                               ; f.write("\n")
            f.write("esti total time:%s"           % (time_util(epoch_cost_time * self.epochs)))                 ; f.write("\n")
            f.write("esti least time:%s"           % (time_util(epoch_cost_time * (self.epochs - (epoch + 1))))) ; f.write("\n")
            f.write("\n")



    def test(self, result_name):  ### 1.result, 2.data, 3.model且reload
        ### 1.result
        self.result_name  = result_name
        self.result_obj   = Result_builder().set_by_result_name(self.exp_dir + "/" + result_name).build()
        ### 2.data
        self.tf_data      = tf_Data_builder().set_basic(self.db_obj).set_img_resize(self.model_obj.model_name).build_by_db_get_method().build()  ### tf_data 抓資料
        ### 3.model且reload
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_dir, max_to_keep=2)  ###step4 建立checkpoint manager 設定最多存2份
        self.model_obj.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()
        ### 待完成

    def run(self):
        if  (self.phase == "train"):          self.train()
        elif(self.phase == "train_reload"):   self.train_reload()
        elif(self.phase == "test"):           self.test()
        elif(self.phase == "train_indicate"): pass  ### 待完成


class Exp_builder():
    def __init__(self, exp=None):
        if(exp is None):
            self.exp = Experiment()
        else: self.exp = exp

    def set_basic(self, phase, db_obj, model_obj, exp_dir=".", describe_mid=None, describe_end=None, result_name=None):
        self.exp.phase = phase
        self.exp.db_obj = db_obj
        self.exp.model_obj = model_obj
        self.exp.exp_dir = exp_dir
        self.exp.describe_mid = describe_mid
        self.exp.describe_end = describe_end
        return self

    def set_train_args(self, batch_size=1, train_shuffle=True, epochs=700, epoch_down_step=100):
        # self.exp.phase = "train"
        self.exp.batch_size = batch_size
        self.exp.train_shuffle = train_shuffle
        self.exp.epochs = epochs
        self.exp.epoch_down_step = epochs / 2
        self.exp.start_epoch = 0
        return self

    def set_train_args_reload(self, result_name):
        self.exp.phase = "train_reload"
        self.result_name = result_name
        return self

    def set_test(self, result_name):
        self.exp.phase = "test"
        self.result_name = result_name
        return self

    def build(self, result_name=None):
        if(result_name is not None): 
            print("build")
            print("self.exp.exp_dir", self.exp.exp_dir)
            print("result_name", result_name)
            self.exp.result_name = result_name
        return self.exp


if(__name__ == "__main__"):
    from step06_a_datas_obj import type5c_real_have_see_no_bg_gt_color,\
                              type7_h472_w304_real_os_book_400data,\
                              type7b_h500_w332_real_os_book_1532data,\
                              type7b_h500_w332_real_os_book_1532data_focus,\
                              type7b_h500_w332_real_os_book_1532data_big,\
                              type7b_h500_w332_real_os_book_800data,\
                              type7b_h500_w332_real_os_book_400data,\
                              type8_blender_os_book_768

    from step08_b_model_obj import *


    ##########################################################################################################################################
    ### 5_1_GD_Gmae136_epoch700
    # os_book_1532_rect_mae1 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_1_1", describe_end="1532data_mae1_127.28").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_1_2", describe_end="1532data_mae1_127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mae6 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_1_3", describe_end="1532data_mae1_127.51").set_train_args(epochs=700).build(result_name="")

    ##########################################################################################################################################
    ### 5_2_GD_vs_justG
    # os_book_1532_rect_D_05  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_2_2", describe_end="1532data_D_0.5_128.245").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_D_025 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_2_3", describe_end="1532data_D_0.25_127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_D_01  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_2_4", describe_end="1532data_D_0.1_127.28").set_train_args(epochs=700).build(result_name="")

    ##########################################################################################################################################
    ### 5_3_just_G_136920 ### 目前mae部分還是需要手動調(20200626)
    # os_book_1532_justG_mae1 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_1", describe_end="1532data_mae1_127.28").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_2", describe_end="1532data_mae3_127.51").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mae6 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_3", describe_end="1532data_mae6_128.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mae9 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_4", describe_end="1532data_mae9_127.35").set_train_args(epochs=700).build(result_name="")

    ##########################################################################################################################################
    ### 5_4_just_G_a_bigger  ### 目前其他 smaller, smaller2 的高度 400, 300 都要手動去調喔 resize大小喔！
    # os_book_1532_justG_mae3_big      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_big, justG, describe_mid="5_4_1", describe_end="1532data_mae3_big_127.35").set_train_args(epochs=700).build(result_name="type7b_h500_w332_real_os_book-20200615_030658-justG-1532data_mae3_big_127.35")
    # os_book_1532_justG_mae3_smaller  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_big, justG, describe_mid="5_4_3", describe_end="1532data_mae3_big_127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mae3_smaller2 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_big, justG, describe_mid="5_4_4", describe_end="1532data_mae3_big_127.35").set_train_args(epochs=700).build(result_name="")

    ##########################################################################################################################################
    ### 5_5_focus_GD_vs_G
    # os_book_1532_rect_mae3_focus     = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect , describe_mid="5_5_2", describe_end="1532data_mae3_focus_127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mae3_focus    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, justG, describe_mid="5_5_4", describe_end="1532data_mae3_focus_127.35").set_train_args(epochs=700).build(result_name="")


    ##########################################################################################################################################
    ### 5_6_a_400_page
    # os_book_400_rect_mae3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_400data, rect, describe_mid="5_6_1", describe_end="400data_mae3_127.35").set_train_args(epochs=2681).build(result_name="")
    # os_book_800_rect_mae3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_800data, rect, describe_mid="no time to train", describe_end="800data_mae3_127.35").set_train_args(epochs=1341).build(result_name="")
    # os_book_400_justG_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_400data, justG, describe_mid="5_6_2", describe_end="400data_justG_mae3_127.28").set_train_args(epochs=2681).build(result_name="")

    ##########################################################################################################################################
    ### 5_7_first_k7_vs_k3
    # os_book_1532_rect_firstk3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_firstk3 , describe_mid="5_7_2", describe_end="127.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_firstk3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_firstk3, describe_mid="5_7_4", describe_end="128.246").set_train_args(epochs=700).build(result_name="")

    ##########################################################################################################################################
    ### 5_8a_GD_mrf
    # os_book_1532_rect_mrf7         = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf7         , describe_mid="5_8a_2", describe_end="127.48").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mrf79        = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf79        , describe_mid="5_8a_3", describe_end="128.245").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_replace_mrf7 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_replace_mrf7 , describe_mid="5_8a_4", describe_end="127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_replace_mrf79 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_replace_mrf79, describe_mid="5_8a_5", describe_end="127.51").set_train_args(epochs=700).build(result_name="")

    ### 5_8b_G_mrf
    ########################################################### 08b2
    # os_book_1532_justG_mrf7          = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf7         , describe_mid="5_8b_2" , describe_end="128.245").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf7_k3       = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf7_k3      , describe_mid="5_8b_2b", describe_end="128.51" ).set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf5_k3       = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf5_k3      , describe_mid="5_8b_2c", describe_end="128.51" ).set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf3_k3       = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf3_k3      , describe_mid="5_8b_2d", describe_end="128.48" ).set_train_args(epochs=700).build(result_name="")

    ########################################################### 08b3
    # os_book_1532_justG_mrf79         = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf79        , describe_mid="5_8b_3" , describe_end="128.245").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf79_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf79_k3     , describe_mid="5_8b_3b", describe_end="128.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf57_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf57_k3     , describe_mid="5_8b_3c", describe_end="128.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf35_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf35_k3     , describe_mid="5_8b_3d", describe_end="127.35" ).set_train_args(epochs=700).build(result_name="")

    ########################################################### 08b4
    # os_book_1532_justG_mrf_replace7  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace7 , describe_mid="5_8b_4" , describe_end="127.40").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf_replace5  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace5 , describe_mid="5_8b_4b", describe_end="127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf_replace3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace3 , describe_mid="5_8b_4c", describe_end="127.48").set_train_args(epochs=700).build(result_name="")
    ########################################################### 08b5
    # os_book_1532_justG_mrf_replace79 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace79, describe_mid="5_8b_5" , describe_end="128.55").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf_replace75 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace75, describe_mid="5_8b_5b", describe_end="128.55").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf_replace35 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace35, describe_mid="5_8b_5c", describe_end="128.28").set_train_args(epochs=700).build(result_name="")

    ########################################################### 08c
    # os_book_1532_justG_mrf135_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c1_Gk3mrf135" , describe_end="128.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf357_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c2_Gk3mrf357" , describe_end="127.51" ).set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justG_mrf3579_k3     = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c3_Gk3mrf3579", describe_end="127.28" ).set_train_args(epochs=700).build(result_name="")

    ########################################################### 08d
    # os_book_1532_rect_mrf35_Gk3_DnoC_k4   = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf35_Gk3_DnoC_k4    , describe_mid="5_8d1_Gmrf35"  , describe_end="127.55"  ).set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mrf135_Gk3_DnoC_k4  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf135_Gk3_DnoC_k4   , describe_mid="5_8d2_Gmrf135" , describe_end="128.246" ).set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mrf357_Gk3_DnoC_k4  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf357_Gk3_DnoC_k4   , describe_mid="5_8d3_Gmrf357" , describe_end="127.51"  ).set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mrf3579_Gk3_DnoC_k4 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf3579_Gk3_DnoC_k4  , describe_mid="5_8d4_Gmrf3579", describe_end="127.28"  ).set_train_args(epochs=700).build(result_name="")

    ########################################################### 09a
    # os_book_1532_rect_Gk4_D_concat_k3    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_concat_k3   , describe_mid="5_9a_2", describe_end="127.51") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_Gk4_D_no_concat_k4 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_no_concat_k4, describe_mid="5_9a_3", describe_end="128.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_Gk4_D_no_concat_k3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_no_concat_k3, describe_mid="5_9a_4", describe_end="127.28") .set_train_args(epochs=700).build(result_name="")

    ########################################################### 09b
    # os_book_1532_rect_Gk3_D_concat_k4    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_concat_k4   , describe_mid="5_9b_1", describe_end="no_machine") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_Gk3_D_concat_k3    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_concat_k3   , describe_mid="5_9b_2", describe_end="no_machine") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_Gk3_D_no_concat_k4 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_no_concat_k4   , describe_mid="5_9b_3", describe_end="127.55") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_Gk3_D_no_concat_k3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_no_concat_k3   , describe_mid="5_9b_4", describe_end="127.48") .set_train_args(epochs=700).build(result_name="")

    ########################################################### 10
    ### 5_10_GD_D_train1_G_train_135
    ### 舊版，如果要重train記得改資料庫喔(拿掉focus)！
    # os_book_1532_rect_mae3_focus_G03D01 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect, describe_mid="5_9_2", describe_end="1532data_mae3_focus_G03D01_127.35").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_mae3_focus_G05D01 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect, describe_mid="5_9_3", describe_end="1532data_mae3_focus_G05D01_127.35").set_train_args(epochs=700).build(result_name="")

    ### 新版
    # os_book_1532_rect_Gk3_train3_Dk4_no_concat = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_train3_Dk4_no_concat, describe_mid="5_10_2", describe_end="127.246").set_train_args(epochs=700).build(result_name="")
    # os_book_1532_rect_Gk3_train5_Dk4_no_concat = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_train5_Dk4_no_concat, describe_mid="5_10_3", describe_end="no_machine").set_train_args(epochs=700).build(result_name="")
    ########################################################### 11
    # os_book_1532_Gk3_no_res              =Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_no_res            , describe_mid="5_11_1", describe_end="127.51") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_no_res_D_no_concat  =Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_no_res_D_no_concat, describe_mid="5_11_2", describe_end="127.28") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_no_res_mrf357       =Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_mrf357_no_res     , describe_mid="5_11_3", describe_end="128.246").set_train_args(epochs=700).build(result_name="")

    # ########################################################### 12
    # exp_dir12 = "5_12_resb_num"
    # os_book_1532_Gk3_resb00 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb00 , exp_dir=exp_dir12, describe_mid="5_12_1", describe_end="finish" ) .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_resb01 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb01 , exp_dir=exp_dir12, describe_mid="5_12_2", describe_end="127.48" ) .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_resb03 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb03 , exp_dir=exp_dir12, describe_mid="5_12_3", describe_end="127.35" ) .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_resb05 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb05 , exp_dir=exp_dir12, describe_mid="5_12_4", describe_end="no_machine") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_resb07 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb07 , exp_dir=exp_dir12, describe_mid="5_12_5", describe_end="no_machine") .set_train_args(epochs=700).build(result_name="")
    # # os_book_1532_Gk3_resb09 ### 原本已經訓練過了，但為了確保沒train錯，還是建了resb_09來train看看囉
    # os_book_1532_Gk3_resb09 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb09 , exp_dir=exp_dir12, describe_mid="5_12_6", describe_end="finish") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_Gk3_resb11 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb11 , exp_dir=exp_dir12, describe_mid="5_12_7", describe_end="127.55") .set_train_args(epochs=700).build(result_name="")
    # ### 13
    # os_book_1532_Gk3_resb15 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb15 , exp_dir=exp_dir12, describe_mid="5_12_7_3", describe_end="127.28") .set_train_args(epochs=700).build(result_name="")
    # ### 17
    # os_book_1532_Gk3_resb20 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb20 , exp_dir=exp_dir12, describe_mid="5_12_8", describe_end="128.244") .set_train_args(epochs=700).build(result_name="")

    # ########################################################### 12
    # exp_dir13 = "5_13_coord_conv"
    # os_book_1532_justGk3_coord_conv        = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justGk3_coord_conv        , exp_dir=exp_dir13, describe_mid="5_13_1", describe_end="127.35") .set_train_args(epochs=700).build(result_name="")
    # os_book_1532_justGk3_mrf357_coord_conv = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justGk3_mrf357_coord_conv , exp_dir=exp_dir13, describe_mid="5_13_2", describe_end="127.28") .set_train_args(epochs=700).build(result_name="")

    ########################################################### 14
    exp_dir14 = "5_14_flow_unet"
    # blender_os_book_flow_unet = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="127.35") .set_train_args(epochs=700).build(result_name="type8_blender_os_book-5_14_1-20210225_204416-flow_unet-127.35")
    # blender_os_book_flow_unet = Exp_builder().set_basic("train"       , type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="127.35") .set_train_args(epochs=700).build(result_name="")
    blender_os_book_flow_unet_epoch050 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch050") .set_train_args(epochs= 50).build(result_name="")
    blender_os_book_flow_unet_epoch100 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch100") .set_train_args(epochs=100).build(result_name="type8_blender_os_book-5_14_1-20210228_161403-flow_unet-epoch100")
    blender_os_book_flow_unet_epoch200 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch200") .set_train_args(epochs=200).build(result_name="")
    blender_os_book_flow_unet_epoch300 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch300") .set_train_args(epochs=300).build(result_name="")
    blender_os_book_flow_unet_epoch700 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch700") .set_train_args(epochs=700).build(result_name="")
    
    blender_os_book_flow_unet_epoch002 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch2, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch002") .set_train_args(epochs=2).build(result_name="")
    blender_os_book_flow_unet_epoch003 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch3, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch003") .set_train_args(epochs=3).build(result_name="")
    blender_os_book_flow_unet_epoch004 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch4, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch004") .set_train_args(epochs=4).build(result_name="")

    blender_os_book_flow_unet_hid_ch_32 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_hid_ch_32, exp_dir=exp_dir14, describe_mid="5_14_2_1", describe_end="hid_ch_32") .set_train_args(epochs=500).build(result_name="")
    blender_os_book_flow_unet_hid_ch_16 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_unet_hid_ch_16, exp_dir=exp_dir14, describe_mid="5_14_2_2", describe_end="hid_ch_16") .set_train_args(epochs=500).build(result_name="type8_blender_os_book-5_14_2_2-20210303_083630-flow_unet-hid_ch_16")
    blender_os_book_flow_unet_hid_ch_08 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_hid_ch_16, exp_dir=exp_dir14, describe_mid="5_14_2_3", describe_end="hid_ch_08") .set_train_args(epochs=500).build(result_name="")
    
if(__name__ == "__main__"):
    ########################################################### 08b2
    # os_book_1532_justG_mrf7_k3.run()   ### 128.51
    # os_book_1532_justG_mrf5_k3.run()   ### 128.51
    # os_book_1532_justG_mrf3_k3.run()   ### 128.48
    ########################################################### 08b3
    # os_book_1532_justG_mrf79_k3.run()  ### 127.246
    # os_book_1532_justG_mrf57_k3.run()  ### 127.246
    # os_book_1532_justG_mrf35_k3.run()  ### 127.35
    ########################################################### 08b4
    # os_book_1532_justG_mrf_replace5.run()  ### 127.35
    # os_book_1532_justG_mrf_replace3.run()  ### 127.48
    ########################################################### 08b5
    # os_book_1532_justG_mrf_replace75.run()  ### 127.55_to127.28
    # os_book_1532_justG_mrf_replace35.run()  ### 127.28

    ########################################################### 08c
    # os_book_1532_justG_mrf135_k3.run()  ### 128.246
    # os_book_1532_justG_mrf357_k3.run()  ### 127.51
    # os_book_1532_justG_mrf3579_k3.run() ### 127.28

    ########################################################### 08d
    # os_book_1532_rect_mrf135_Gk3_DnoC_k4.run() ### 128.246
    # os_book_1532_rect_mrf357_Gk3_DnoC_k4.run() ### 127.51
    # os_book_1532_rect_mrf3579_Gk3_DnoC_k4.run() ### 127.28
    # os_book_1532_rect_mrf35_Gk3_DnoC_k4.run()   ### 127.48

    ########################################################### 09a Gk4的情況下，D try concat 和 k_size
    # os_book_1532_rect_Gk4_D_concat_k3.run()    ### 127.51
    # os_book_1532_rect_Gk4_D_no_concat_k4.run() ### 128.246
    # os_book_1532_rect_Gk4_D_no_concat_k3.run() ### 127.28

    ########################################################### 09b Gk3的情況下，D try concat 和 k_size
    # os_book_1532_rect_Gk3_D_concat_k4.run()    ###
    # os_book_1532_rect_Gk3_D_concat_k3.run()    ###
    # os_book_1532_rect_Gk3_D_no_concat_k4.run() ### 127.55
    # os_book_1532_rect_Gk3_D_no_concat_k3.run() ### 127.48

    ########################################################### 10 GAN裡的 G訓練多次有沒有用
    # os_book_1532_rect_Gk3_train3_Dk4_no_concat.run() ### 128.246
    # os_book_1532_rect_Gk3_train5_Dk4_no_concat.run() ### no machine

    ########################################################### 11 resblock的add有沒有用
    # os_book_1532_Gk3_no_res.run()               ### 127.51
    # os_book_1532_Gk3_no_res_D_no_concat.run()   ### 127.28
    # os_book_1532_Gk3_no_res_mrf357.run()        ### 128.246

    ########################################################### 12 resblock用多少個
    # os_book_1532_Gk3_resb00.run()  ### 127.48 ###完成
    # os_book_1532_Gk3_resb01.run()  ### 127.48
    # os_book_1532_Gk3_resb03.run()  ### 127.35
    # os_book_1532_Gk3_resb05.run()  ### no
    # os_book_1532_Gk3_resb07.run()  ### no
    # os_book_1532_Gk3_resb09.run()  ### finish
    # os_book_1532_Gk3_resb11.run()  ### 127.55
    # os_book_1532_Gk3_resb15.run()  ### 127.28
    # os_book_1532_Gk3_resb20.run()  ### 128.244

    ########################################################### 13 加coord_conv試試看
    # os_book_1532_justGk3_coord_conv.run()         ### 127.35
    # os_book_1532_justGk3_mrf357_coord_conv.run()  ### 127.28
    ########################################################### 14
    # blender_os_book_flow_unet.run()            ### 127.35  60.5  GB   最低loss:0.0000945
    # blender_os_book_flow_unet_epoch050.run()   ### 127.35  05.38 GB   最低loss:0.00035705  total cost time:01:29:33
    # blender_os_book_flow_unet_epoch100.run()   ### 127.35  09.72 GB   最低loss:0.00023004  total cost time:02:41:56
    # blender_os_book_flow_unet_epoch200.run()   ### 127.35  18.30 GB   最低loss:0.00015143  total cost time:05:45:19
    # blender_os_book_flow_unet_epoch300.run()   ### 127.35  27.00 GB   最低loss:0.00012906  total cost time:08:51:23
    # blender_os_book_flow_unet_epoch700.run()   ### 127.35  27.00 GB   最低loss:0.00012906  total cost time:08:51:23

    ########################################################### 14
    # blender_os_book_flow_unet_hid_ch_32.run()
    blender_os_book_flow_unet_hid_ch_16.run()
    # blender_os_book_flow_unet_hid_ch_08.run()
    pass
