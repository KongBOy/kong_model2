import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import time
from step06_a_datas_obj import type5c_real_have_see_no_bg_gt_color,\
                            type7_h472_w304_real_os_book_400data,\
                            type7b_h500_w332_real_os_book_1532data,\
                            type7b_h500_w332_real_os_book_1532data_focus,\
                            type7b_h500_w332_real_os_book_1532data_big,\
                            type7b_h500_w332_real_os_book_800data,\
                            type7b_h500_w332_real_os_book_400data,\
                            type8_blender_os_book_768

from step06_b_data_pipline import tf_Data_builder
from step08_e_model_obj import *
from step09_a_loss_info_obj import *
from step11_b_result_obj_builder import Result_builder
import sys
sys.path.append("kong_util")
from util import time_util, get_dir_certain_file_name

from tqdm import tqdm

import socket    ### 取得 本機 IP   給 train_step5_show_time 紀錄
import getpass   ### 取得 本機 User 給 train_step5_show_time 紀錄

class Experiment():
    def step0_save_code(self):
        '''
        把 step.py 和 kong_util資料夾 存一份進result
        '''
        import shutil
        from build_dataset_combine import Check_dir_exist_and_build
        code_dir = self.result_obj.result_dir + "/" + "train_code"  ### 定位出 result存code的目的地
        Check_dir_exist_and_build(code_dir)                         ### 建立目的地資料夾
        py_file_names = get_dir_certain_file_name(".", certain_word="step")  ### 抓取目前目錄所有 有含 "step" 的檔名
        for py_file_name in py_file_names:
            ### 這兩行可以 抓 step"幾" 然後只存 step"幾".py 喔，但還沒整理以前的code，保險起見還是全存好了，有空再細細處理~~
            # py_file_name_step = int(py_file_name.split("_")[0][4:])          ### 抓出 step "幾"
            # if(py_file_name_step >= 6 or py_file_name_step == 0):            ### step06 以上
            shutil.copy(py_file_name, code_dir + "/" + py_file_name)     ### 存起來

        if(os.path.isdir(code_dir + "/" + "kong_util")):  ### 在train_reload時 如果有舊的kong_util，把舊的刪掉換新的
            shutil.rmtree(code_dir + "/" + "kong_util")
        shutil.copytree("kong_util", code_dir + "/" + "kong_util")  


################################################################################################################################################
################################################################################################################################################
    def __init__(self):
        self.machine_ip = None
        self.machine_user = None
        ##############################################################################################################################
        self.phase        = "train"
        self.db_obj       = None
        self.model_obj    = None
        self.loss_info_obj = G_mae_loss_info
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
        self.epochs          = 500    ### 看opencv合成的video覺得1300左右就沒變了
        self.epoch_down_step = self.epochs // 2    ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
        self.epoch_stop      = self.epochs
        self.epoch_save_freq = 10     ### 訓練 epoch_save_freq 個 epoch 存一次模型
        self.start_epoch     = 0

        self.exp_bn_see_arg = False  ### 本來 bn 在 test 的時候就應該要丟 false，只是 現在batch_size=1， 丟 True 會變 IN ， 所以加這個flag 來控制

        self.in_use_range = "0~1"
        self.gt_use_range = "0~1"
        '''
        覺得不要 mo_use_range，因為本來就應該要 model_output 就直接拿來用了呀，所以直接根據 gt_use_range 來做後處理即可
        output如果不能拿來用 或者 拿來用出問題，代表model設計有問題
        '''

        self.lr_start = 0.0002  ### learning rate
        self.lr_current = self.lr_start

        # self.phase = "train_reload" ### 要記得去決定 result_name 喔！
        # self.phase = "test"         ### test是用固定 train/test 資料夾架構的讀法 ### 要記得去決定 result_name 喔！
        ### self.phase = "test_indicate" ### 這應該是可以刪掉了，因為在取db的時候，已經有包含了test_indecate的取法了！不用在特別區分出來 取資料囉！
        ### 參數設定結束
        ####################################################################################################################
        self.tf_data      = None
        self.ckpt_manager = None

################################################################################################################################################
################################################################################################################################################
    def exp_init(self, reload_result=False):  ### 共作五件事： 1.result, 2.data, 3.model(reload), 4.Loss_info, 5.save_code
        ### 1.result
        if(reload_result):
            # print("self.exp_dir:", self.exp_dir)
            # print("self.result_name:", self.result_name)
            self.result_obj   = Result_builder().set_by_result_name(self.exp_dir + "/" + self.result_name, self.in_use_range, self.gt_use_range).build()  ### 直接用 自己指定好的 result_name
        else: self.result_obj = Result_builder().set_by_exp(self).build()  ### exp在train時 要自動建新的 result，才不會覆蓋到之前訓練的result，Result_builder()需要 db_obj 和 exp本身的describe_mid/end
        ### 2.data，在這邊才建立而不在step6_b 就先建好是因為 要參考 model_name 來決定如何 resize 喔！
        self.tf_data      = tf_Data_builder().set_basic(self.db_obj, batch_size=self.batch_size, train_shuffle=self.train_shuffle).set_data_use_range(in_use_range=self.in_use_range, gt_use_range=self.gt_use_range).set_img_resize(self.model_obj.model_name).build_by_db_get_method().build()  ### tf_data 抓資料
        ### 3.model
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_dir, max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
        if(reload_result):  ### 看需不需要reload model
            self.model_obj.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()
            print("Reload: %s ok~~ start_epoch=%i" % (self.result_obj.result_name, self.start_epoch))

        ####################################################################################################################
        ### 4.Loss_info, 5.save_code；train時才需要 loss_info_obj 和 把code存起來喔！test時不用～所以把存code部分拿進train裡囉
        self.loss_info_obj = Loss_info_builder(G_mae_loss_info).set_logs_dir_and_summary_writer(self.result_obj.logs_dir).build_loss_containors_by_model_name(self.model_obj.model_name).build()  ###step3 建立tensorboard，只有train 和 train_reload需要


    def train_reload(self):
        self.train(reload_result=True)

    def train(self, reload_result=False):
        self.exp_init(reload_result)
        self.step0_save_code()  ### training 才需要把source code存起來，所以從exp_init移下來囉
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

        current_epoch = self.start_epoch   ### 因為 epoch 的狀態 在 train 前後是不一樣的，所以需要用一個變數來記住，就用這個current_epoch來記錄囉！
        for epoch in range(self.start_epoch, self.epoch_stop):
            ###############################################################################################################################
            ###    step0 紀錄epoch開始訓練的時間
            epoch_start_timestamp = time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime())
            print("Epoch: ", current_epoch, "start at", epoch_start_timestamp)
            e_start = time.time()
            ###############################################################################################################################
            ###    step0 設定learning rate
            self.lr_current = self.lr_start if current_epoch < self.epoch_down_step else self.lr_start * (self.epochs - current_epoch) / (self.epochs - self.epoch_down_step)
            self.model_obj.optimizer_G.lr = self.lr_current
            ###############################################################################################################################
            if(current_epoch == 0): print("Initializing Model~~~")  ### sample的時候就會initial model喔！
            ###############################################################################################################################
            ###     step1 用來看目前訓練的狀況
            self.train_step1_see_current_img(current_epoch, training=self.exp_bn_see_arg)   ### 介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～
            ###############################################################################################################################
            ### 以上 current_epoch = epoch   ### 代表還沒訓練
            ###     step2 訓練
            for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(self.tf_data.train_db_combine)):
                self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_obj=self.loss_info_obj)
                # break   ### debug用，看subprocess成不成功
            current_epoch += 1  ### 超重要！別忘了加呀！
            ### 以下 current_epoch = epoch + 1   ### +1 代表訓練完了！變成下個epoch了！
            ###############################################################
            ###     step3 整個epoch 的 loss 算平均，存進tensorboard
            self.train_step3_Loss_info_save_loss(current_epoch)
            ###############################################################################################################################
            ###     step4 儲存模型 (checkpoint) the model every "epoch_save_freq" epochs
            if (current_epoch) % self.epoch_save_freq == 0:
                # print("save epoch_log :", current_epoch)
                self.model_obj.ckpt.epoch_log.assign(current_epoch)  ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                self.ckpt_manager.save()
                print("save ok ~~~~~~~~~~~~~~~~~")
            ###############################################################################################################################
            ###    step5 紀錄、顯示 訓練相關的時間
            self.train_step5_show_time(current_epoch, e_start, total_start, epoch_start_timestamp)
            # break  ### debug用，看subprocess成不成功

            if(current_epoch == self.epoch_stop): break   ### 想要有lr 下降，但又不想train滿 中途想離開就 設 epcoh_stop 囉！

        ### 最後train完 記得也要看結果喔！
        self.train_step1_see_current_img(current_epoch, training=self.exp_bn_see_arg)   ### 介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～

    def train_step1_see_current_img(self, epoch, training=False, see_reset_init=False):
        """
        epoch：         目前 model 正處在 被更新了幾次epoch 的狀態
        training：      可以設定 bn 的動作為 train 還是 test，當batch_size=1時，設為True可以模擬IN，設False圖會壞掉！也可由此得知目前的任務是適合用BN的
                        介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～
        see_reset_init：是給 test_see 用的，有時候製作 fake_exp 的時候，只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要 重建一份see，這時see_reset_init要設True就會重建一下囉
        """
        # sample_start_time = time.time()
        see_in     = self.tf_data.test_in_db
        see_in_pre = self.tf_data.test_in_db_pre
        see_gt     = self.tf_data.test_gt_db
        see_amount = 1
        if(self.db_obj.have_see):
            see_in     = self.tf_data.see_in_db
            see_in_pre = self.tf_data.see_in_db_pre
            see_gt     = self.tf_data.see_gt_db
            see_amount = self.tf_data.see_amount

        for see_index, (test_in, test_in_pre, test_gt) in enumerate(tqdm(zip(see_in.take(see_amount), see_in_pre.take(see_amount), see_gt.take(see_amount)))):
            if  ("unet"  in self.model_obj.model_name.value and
                 "flow" not in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in, test_in_pre, test_gt, self.tf_data.max_train_move, self.tf_data.min_train_move, epoch, self.result_obj.result_dir, result_obj, see_reset_init)  ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif("flow"  in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in, test_in_pre, test_gt, epoch, self.result_obj, training, see_reset_init)
            elif("rect"  in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.rect.generator, see_index, test_in, test_in_pre, test_gt, epoch, self.result_obj, see_reset_init)
            elif("justG" in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in, test_in_pre, test_gt, epoch, self.result_obj, see_reset_init)

        # self.result_obj.save_all_single_see_as_matplot_visual_multiprocess() ### 不行這樣搞，對當掉！但可以分開用別的python執行喔～
        # print("sample all see time:", time.time()-sample_start_time)

    def train_step3_Loss_info_save_loss(self, epoch):
        with self.loss_info_obj.summary_writer.as_default():
            for loss_name, loss_containor in self.loss_info_obj.loss_containors.items():
                ### tensorboard
                tf.summary.scalar(loss_name, loss_containor.result(), step=epoch)
                tf.summary.scalar("lr", self.lr_current, step=epoch)

                ### 自己另存成 npy
                loss_value = loss_containor.result().numpy()
                if(epoch == 1):  ### 第一次 直接把值存成np.array
                    np.save(self.result_obj.logs_dir + "/" + loss_name, np.array(loss_value.reshape(1)))

                else:  ### 第二次後，先把np.array先讀出來append值後 再存進去
                    loss_array = np.load(self.result_obj.logs_dir + "/" + loss_name + ".npy")
                    loss_array = loss_array[:epoch]   ### 這是為了防止 如果程式在 step3,4之間中斷 這種 loss已經存完 但 model還沒存 的狀況，loss 會比想像中的多一步，所以加這行防止這種情況發生喔
                    loss_array = np.append(loss_array, loss_value)
                    np.save(self.result_obj.logs_dir + "/" + loss_name, np.array(loss_array))
                    # print(loss_array)

        ###    reset tensorboard 的 loss紀錄容器
        for loss_containor in self.loss_info_obj.loss_containors.values():
            loss_containor.reset_states()
        ###############################################################
        self.loss_info_obj.see_loss(self.epochs)  ### 把 loss資訊 用 matplot畫出來
        ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了
        # self.result_obj.Draw_loss_during_train(epoch, self.epochs)  ### 在 train step1 generate_see裡已經把see的 matplot_visual圖畫出來了，再把 loss資訊加進去

    def train_step5_show_time(self, epoch, e_start, total_start, epoch_start_timestamp):
        print("current exp:", self.result_obj.result_name)
        epoch_cost_time = time.time() - e_start
        total_cost_time = time.time() - total_start
        print(self.phase)
        print('epoch %i start at:%s, %s, %s' % (epoch, epoch_start_timestamp, self.machine_ip, self.machine_user))
        print('epoch %i cost time:%.2f'      % (epoch, epoch_cost_time))
        print("batch cost time:%.2f average" % (epoch_cost_time / self.tf_data.train_amount))
        print("total cost time:%s"           % (time_util(total_cost_time)))
        print("esti total time:%s"           % (time_util(epoch_cost_time * self.epochs)))
        print("esti least time:%s"           % (time_util(epoch_cost_time * (self.epochs - (epoch + 1)))))
        print("")
        with open(self.result_obj.result_dir + "/" + "cost_time.txt", "a") as f:
            f.write(self.phase)                                                                                         ; f.write("\n")
            f.write('epoch %i start at:%s, %s, %s' % (epoch, epoch_start_timestamp, self.machine_ip, self.machine_user)); f.write("\n")
            f.write('epoch cost time:%.2f'         % (epoch_cost_time))                                                 ; f.write("\n")
            f.write("batch cost time:%.2f average" % (epoch_cost_time / self.tf_data.train_amount))                     ; f.write("\n")
            f.write("total cost time:%s"           % (time_util(total_cost_time)))                                      ; f.write("\n")
            f.write("esti total time:%s"           % (time_util(epoch_cost_time * self.epochs)))                        ; f.write("\n")
            f.write("esti least time:%s"           % (time_util(epoch_cost_time * (self.epochs - (epoch + 1)))))        ; f.write("\n")
            f.write("\n")

    def test_see(self):
        """
        用最後儲存的 Model 來產生see～
        也常常拿來 reset in/gt see 喔！
        想設定 testing 時的 bn 使用的 training arg 的話， 麻煩用 exp.exp_bn_see_arg 來指定， 因為要用test_see 就要先建exp， 就統一寫在exp裡 個人覺得比較連貫， 因此 就不另外開一個 arg 給 test_see 用囉！
        """
        self.exp_init(reload_result=True)
        self.train_step1_see_current_img(self.start_epoch, training=self.exp_bn_see_arg, see_reset_init=True)  ### 有時候製作 fake_exp 的時候 ， 只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要reset一下
        print("test see finish")

    def run(self):
        self.machine_ip   = socket.gethostbyname(socket.gethostname())  ### 取得 本機 IP   給 train_step5_show_time 紀錄
        self.machine_user = getpass.getuser()                           ### 取得 本機 User 給 train_step5_show_time 紀錄
        if  (self.phase == "train"):          self.train()
        elif(self.phase == "train_reload"):   self.train_reload()
        elif(self.phase == "test_see"):       self.test_see()
        elif(self.phase == "train_indicate"): pass  ### 待完成Z
        elif(self.phase.lower() == "ok"): pass      ### 不做事情，只是個標記而以這樣子
        else: print("ㄘㄋㄇㄉ phase 打錯字了拉~~~")



class Exp_builder():
    def __init__(self, exp=None):
        if(exp is None):
            self.exp = Experiment()
        else: self.exp = exp

    def set_basic(self, phase, db_obj, model_obj, loss_info_obj, exp_dir=".", describe_mid=None, describe_end=None, result_name=None):
        self.exp.phase = phase
        self.exp.db_obj = db_obj
        self.exp.model_obj = model_obj
        self.exp.loss_info_obj = loss_info_obj
        self.exp.exp_dir = exp_dir
        self.exp.describe_mid = describe_mid
        self.exp.describe_end = describe_end
        return self

    def set_train_args(self, batch_size=1, train_shuffle=True, epochs=500, epoch_down_step=None, epoch_stop=500, exp_bn_see_arg=False):
        """
        train_shuffle：注意一下，這裡的train_shuffle無法重現 old shuffle 喔
        epochs：train的 總epoch數， epoch_down_step 設定為 epoch_down_step//2
        epoch_down_step：第幾個epoch後 lr 下降
        epoch_stop：想要有lr 下降，但又不想 花時間 train滿 中途想離開就 設 epcoh_stop 囉！
        exp_bn_see_arg：在 train/test 生成see 的時候， 決定 bn 的 training = True 或 False， 詳情看 train_step1_see_current_img～
        """
        # self.exp.phase = "train"
        self.exp.batch_size = batch_size
        self.exp.train_shuffle = train_shuffle
        self.exp.epochs = epochs
        if(epoch_down_step is None): self.exp.epoch_down_step = epochs / 2
        else: self.exp.epoch_down_step = epoch_down_step
        self.exp.epoch_stop = epoch_stop
        self.exp.start_epoch = 0
        self.exp.exp_bn_see_arg = exp_bn_see_arg
        return self

    def set_train_in_gt_use_range(self, in_use_range="0~1", gt_use_range="0~1"):
        self.exp.in_use_range = in_use_range
        self.exp.gt_use_range = gt_use_range
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
        '''
        這邊先建好 result_obj 的話就可以給 step11, step12 用喔，
        且 因為我有寫 在train時 會自動建新的 result，所以不用怕 用到這邊 隨便建立的 default result_obj 囉！
        '''
        if(result_name is not None):
            # print("self.exp.exp_dir", self.exp.exp_dir)
            # print("1.result_name", result_name, ", self.exp.gt_use_range~~~~~~~~~~~~~~~~~~~~~~~~~", self.exp.gt_use_range)  ### 追蹤see的建立過程
            self.exp.result_name = result_name
            self.exp.result_obj  = Result_builder().set_by_result_name(self.exp.exp_dir + "/" + self.exp.result_name, self.exp.in_use_range, self.exp.gt_use_range).build()  ### 直接用 自己指定好的 result_name
            # print()  ### 追蹤see的建立過程
        return self.exp


# if(__name__ == "__main__"):



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
### 以下 old_shuffle (先 batch 再 shuffle)，注意目前無法重現 old shuffle，且也沒必要重現喔！因為 old/new shuffle 因為 batch_size==1 ， 理論上是一樣的！
### 測epoch數，hid_ch=64, bn=1
old_shuf_dir = exp_dir14 + "/" + "0 old_shuf_dir"
epoch050_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_1_1", describe_end="epoch050_bn_see_arg_T") .set_train_args(epochs= 50, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_1_1-20210228_144200-flow_unet-epoch050_bn_see_arg_T")
epoch100_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_1_2", describe_end="epoch100_bn_see_arg_T") .set_train_args(epochs=100, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_1_2-20210228_161403-flow_unet-epoch100_bn_see_arg_T")
epoch200_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_1_3", describe_end="epoch200_bn_see_arg_T") .set_train_args(epochs=200, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_1_3-20210301_015045-flow_unet-epoch200_bn_see_arg_T")
epoch300_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_1_4", describe_end="epoch300_bn_see_arg_T") .set_train_args(epochs=300, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_1_4-20210228_164701-flow_unet-epoch300_bn_see_arg_T")
epoch700_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_1_6", describe_end="epoch700_bn_see_arg_T") .set_train_args(epochs=700, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_1_6-20210225_204416-flow_unet-epoch700_bn_see_arg_T")
### epoch=500, 測hid_ch, bn=1
ch128_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch128, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_2_1", describe_end="ch128_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_2_1-20210304_082556-flow_unet-ch128_bn_see_arg_T")
ch032_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_2_3", describe_end="ch032_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_2_3-20210302_234709-flow_unet-ch032_bn_see_arg_T")
ch016_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch016, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_2_4", describe_end="ch016_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_2_4-20210303_083630-flow_unet-ch016_bn_see_arg_T")
ch008_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch008, G_mse_loss_info, exp_dir=old_shuf_dir, describe_mid="5_14_0_2_5", describe_end="ch008_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_0_2_5-20210303_161150-flow_unet-ch008_bn_see_arg_T")

#############################################################
### 以下 new_shuffle (先 shuffle 再 batch)
### 測epoch數，hid_ch=64, bn=1
epoch050_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_1", describe_end="epoch050_new_shuf_bn_see_arg_T").set_train_args(epochs= 50, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1") .build(result_name="type8_blender_os_book-5_14_1_1_1-20210306_190321-flow_unet-epoch050_new_shuf_bn_see_arg_T")
epoch100_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_2", describe_end="epoch100_new_shuf_bn_see_arg_T").set_train_args(epochs=100, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1") .build(result_name="type8_blender_os_book-5_14_1_1_2-20210306_203154-flow_unet-epoch100_new_shuf_bn_see_arg_T")
epoch200_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_3", describe_end="epoch200_new_shuf_bn_see_arg_T").set_train_args(epochs=200, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1") .build(result_name="type8_blender_os_book-5_14_1_1_3-20210306_232534-flow_unet-epoch200_new_shuf_bn_see_arg_T")
epoch300_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_4", describe_end="epoch300_new_shuf_bn_see_arg_T").set_train_args(epochs=300, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1") .build(result_name="type8_blender_os_book-5_14_1_1_4-20210307_051136-flow_unet-epoch300_new_shuf_bn_see_arg_T")
epoch500_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_5", describe_end="epoch500_new_shuf_bn_see_arg_T").set_train_args(epochs=500, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1") .build(result_name="type8_blender_os_book-5_14_1_1_5-20210318_211827f-flow_unet-epoch500_new_shuf_bn_see_arg_T")
epoch500_new_shuf_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_5", describe_end="epoch500_new_shuf_bn_see_arg_F").set_train_args(epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_1_5-20210318_211827-flow_unet-epoch500_new_shuf_bn_see_arg_F")
epoch700_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_6", describe_end="epoch700_new_shuf_bn_see_arg_T").set_train_args(epochs=700, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1") .build(result_name="type8_blender_os_book-5_14_1_1_6-20210308_100044-flow_unet-epoch700_new_shuf_bn_see_arg_T")
epoch700_bn_see_arg_T_no_down  = Exp_builder().set_basic("train",    type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_1_7", describe_end="epoch700_no_down") .set_train_args(epochs=700, exp_bn_see_arg=True, epoch_down_step=700).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_1_7-20210330_093032-flow_unet-epoch700_no_epoch_down")

### epoch=500, 測hid_ch, bn=1
### 這裡的 bn 當初照著 bn 正常的使用方式使用：在testing 時 bn 使用上的 training要指定 False，但發現效果奇差，透過這結果應該可以知道是因為每張影像都是獨特的存在，差異性太大，每一張影像應該要跟自己的 mu, sigma 作用才對，因此應該要用 in 比較好，所以這些結果就先不用了，用的是下面的結果～
new_shuffle_ch_but_wrong_see_arg_dir = exp_dir14 + "/" + "1 new_shuffle_ch_but_wrong_see_arg_dir"
ch128_new_shuf_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch128, G_mse_loss_info, exp_dir=new_shuffle_ch_but_wrong_see_arg_dir, describe_mid="5_14_1_2_1", describe_end="ch128_new_shuf_bn_see_arg_F") .set_train_args(epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_1-20210310_230448-flow_unet-ch128_new_shuf_bn_see_arg_F")
ch032_new_shuf_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=new_shuffle_ch_but_wrong_see_arg_dir, describe_mid="5_14_1_2_3", describe_end="ch032_new_shuf_bn_see_arg_F") .set_train_args(epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_3-20210309_214404-flow_unet-ch032_new_shuf_bn_see_arg_F")
ch016_new_shuf_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch016, G_mse_loss_info, exp_dir=new_shuffle_ch_but_wrong_see_arg_dir, describe_mid="5_14_1_2_4", describe_end="ch016_new_shuf_bn_see_arg_F") .set_train_args(epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_4-20210309_140134-flow_unet-ch016_new_shuf_bn_see_arg_F")
ch008_new_shuf_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch008, G_mse_loss_info, exp_dir=new_shuffle_ch_but_wrong_see_arg_dir, describe_mid="5_14_1_2_5", describe_end="ch008_new_shuf_bn_see_arg_F") .set_train_args(epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_5-20210309_061533-flow_unet-ch008_new_shuf_bn_see_arg_F")  ### 127.28
### 如何用 bn 模擬 in 就是在 testing 時 training 仍要設 True， 而這剛好就是我舊的 old shuffle 的寫法(當時還不會設bn 剛好 default 就為 True XD) 所以就從 old shuffle 裡 copy 出 ckpt, log, ... 當作假的 new shuffle 囉！
ch128_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch128, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_2_1", describe_end="ch128_new_shuf_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_1-20210304_082556f-flow_unet-ch128_new_shuf_bn_see_arg_T")
ch032_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_2_3", describe_end="ch032_new_shuf_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_3-20210302_234709f-flow_unet-ch032_new_shuf_bn_see_arg_T")
ch016_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch016, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_2_4", describe_end="ch016_new_shuf_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_4-20210303_083630f-flow_unet-ch016_new_shuf_bn_see_arg_T")
ch008_new_shuf_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch008, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_2_5", describe_end="ch008_new_shuf_bn_see_arg_T") .set_train_args(epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_2_5-20210303_161150f-flow_unet-ch008_new_shuf_bn_see_arg_T")  ### 127.28


### epoch=500, hid_ch=64, 測bn, call沒設定training=True/False
ch64_bn04_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3a_3", describe_end="ch64_bn04_bn_see_arg_F") .set_train_args(batch_size= 4, epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3a_3-20210304_102528f-flow_unet-ch64_bn04_bn_see_arg_F")   ### 已經是new_shuf
ch64_bn04_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3a_3", describe_end="ch64_bn04_bn_see_arg_T") .set_train_args(batch_size= 4, epochs=500, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3a_3-20210304_102528-flow_unet-ch64_bn04_bn_see_arg_T")   ### 已經是new_shuf
ch64_bn08_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3a_3", describe_end="ch64_bn08_bn_see_arg_F") .set_train_args(batch_size= 8, epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3a_3-20210304_232248f-flow_unet-ch64_bn08_bn_see_arg_F")   ### 已經是new_shuf
ch64_bn08_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3a_3", describe_end="ch64_bn08_bn_see_arg_T") .set_train_args(batch_size= 8, epochs=500, exp_bn_see_arg=True) .set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3a_3-20210304_232248-flow_unet-ch64_bn08_bn_see_arg_T")   ### 已經是new_shuf
### bn=16在127.28已超出記憶體

### 把hid_ch 變小，bn可以多一點點
### epoch=500, hid_ch=32, 測bn, call沒設定training=True/False
ch32_bn04_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3b_2", describe_end="ch32_bn04_bn_see_arg_T") .set_train_args(batch_size= 4, epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3b_2-20210306_111439-flow_unet-ch32_bn04_bn_see_arg_T")  ### 已經是new_shuf
ch32_bn08_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3b_3", describe_end="ch32_bn08_bn_see_arg_T") .set_train_args(batch_size= 8, epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3b_3-20210306_171735-flow_unet-ch32_bn08_bn_see_arg_T")  ### 已經是new_shuf
ch32_bn16_bn_see_arg_T = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3b_4", describe_end="ch32_bn16_bn_see_arg_T") .set_train_args(batch_size=16, epochs=500, exp_bn_see_arg=True).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3b_4-20210306_231628-flow_unet-ch32_bn16_bn_see_arg_T")  ### 已經是new_shuf
### bn=32在127.28也超出記憶體

### epoch=500, hid_ch=32, 測bn, call沒設定training=True/False
ch32_bn04_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3b_2", describe_end="ch32_bn04_bn_see_arg_F") .set_train_args(batch_size= 4, epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3b_2-20210308_101945-flow_unet-ch32_bn04_bn_see_arg_F")  ### 已經是new_shuf
ch32_bn08_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3b_3", describe_end="ch32_bn08_bn_see_arg_F") .set_train_args(batch_size= 8, epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3b_3-20210308_163036-flow_unet-ch32_bn08_bn_see_arg_F")  ### 已經是new_shuf
ch32_bn16_bn_see_arg_F = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_ch032, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_3b_4", describe_end="ch32_bn16_bn_see_arg_F") .set_train_args(batch_size=16, epochs=500, exp_bn_see_arg=False).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_3b_4-20210308_223123-flow_unet-ch32_bn16_bn_see_arg_F")  ### 已經是new_shuf
#############################################################################################################################################################################################################

ch64_in_epoch500 = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_IN_ch64, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_4_5b", describe_end="ch64_in_epoch500") .set_train_args(epochs=500, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_4_5b-20210309_135755-flow_unet-ch64_in_epoch500")
ch64_in_epoch700 = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_IN_ch64, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_4_6b", describe_end="ch64_in_epoch700") .set_train_args(epochs=700, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_4_6b-20210310_012428-flow_unet-ch64_in_epoch700")

#############################################################################################################################################################################################################
### 測試 concate 用 Activation 後的 feature map 來 concat, 且 unet 本身已用 IN 的 UNet 沒有設定 training=True/False問題
ch64_in_concat_A = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_ch64_in_concat_A, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_5_1", describe_end="ch64_in_concat_A") .set_train_args(epochs=500, exp_bn_see_arg=None).build(result_name="type8_blender_os_book-5_14_1_5_1-20210329_212042-flow_unet-ch64_in_concat_A")

### 測試 UNet 用 2~7 層的效果，且 unet 本身已用 IN 的 UNet 沒有設定 training=True/False問題
unet_2l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_2_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_2l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210330_230045-flow_unet-unet_2l")
unet_3l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_3_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_3l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210331_140947-flow_unet-unet_3l")
unet_4l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_4_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_4l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210331_175806-flow_unet-unet_4l")
unet_5l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_5_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_5l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210331_221927-flow_unet-unet_5l")
unet_6l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_6_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_6l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210401_031859-flow_unet-unet_6l")
unet_7l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_7_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_7l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210401_124509-flow_unet-unet_7l")
unet_8l = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_8_level, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_6_1", describe_end="unet_8l") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_6_1-20210402_020604-flow_unet-unet_8l")

### 測試 UNet 用 output 用 sigmoid 的效果
t1_in_01_mo_th_gt_01_mse = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_1", describe_end="t1_01_th_01") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_8_1-20210309_135755f-flow_unet-01_th_01")
t2_in_01_mo_01_gt_01_mse = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_2", describe_end="t2_01_01_01") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_8_2-20210406_175350-flow_unet-01_01_01")
t3_in_01_mo_th_gt_th_mse = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_3", describe_end="t3_01_th_th") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="-1~1").build(result_name="type8_blender_os_book-5_14_1_8_3-20210408_050717-flow_unet-01_th_th")
t4_in_01_mo_01_gt_th_mse = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_4", describe_end="t4_01_01_th") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="-1~1").build(result_name="type8_blender_os_book-5_14_1_8_4-20210408_100237-flow_unet-01_01_th")
t5_in_th_mo_th_gt_01_mse = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_5", describe_end="t5_th_th_01") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_8_5-20210408_141452-flow_unet-th_th_01")
t6_in_th_mo_01_gt_01_mse = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_6", describe_end="t6_th_01_01") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="0~1"). build(result_name="type8_blender_os_book-5_14_1_8_6-20210408_001841-flow_unet-th_01_01")
t7_in_th_mo_th_gt_th_mse = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_7", describe_end="t7_th_th_th") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="-1~1").build(result_name="type8_blender_os_book-5_14_1_8_7-20210408_001245-flow_unet-th_th_th")
t8_in_th_mo_01_gt_th_mse = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_8", describe_end="t8_th_01_th") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="-1~1").build(result_name="type8_blender_os_book-5_14_1_8_8-20210408_054355-flow_unet-th_01_th")

t1_in_01_mo_th_gt_01_mae = Exp_builder().set_basic("ok", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_1", describe_end="t1_01_th_01_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="")
t2_in_01_mo_01_gt_01_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_2", describe_end="t2_01_01_01_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="")
t3_in_01_mo_th_gt_th_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_3", describe_end="t3_01_th_th_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="-1~1").build(result_name="")
t4_in_01_mo_01_gt_th_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_4", describe_end="t4_01_01_th_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="-1~1").build(result_name="")
t5_in_th_mo_th_gt_01_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_5", describe_end="t5_th_th_01_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="0~1").build(result_name="")
t6_in_th_mo_01_gt_01_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_6", describe_end="t6_th_01_01_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="0~1"). build(result_name="")
t7_in_th_mo_th_gt_th_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64        , G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_7", describe_end="t7_th_th_th_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="-1~1").build(result_name="")
t8_in_th_mo_01_gt_th_mae = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_ch64_sigmoid, G_mae_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_8_8", describe_end="t8_th_01_th_mae") .set_train_args(epochs=500, epoch_down_step=250, epoch_stop=500).set_train_in_gt_use_range(in_use_range="-1~1", gt_use_range="-1~1").build(result_name="")


### 看看 UNet 的 concat 改用 + 會有什麼影響
unet_8l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_8_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_8l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210403_141018-flow_unet-unet_8l_skip_use_add")
unet_7l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_7_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_7l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210401_115231-flow_unet-unet_7l_skip_use_add")
unet_6l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_6_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_6l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210401_225248-flow_unet-unet_6l_skip_use_add")
unet_5l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_5_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_5l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210402_023315-flow_unet-unet_5l_skip_use_add")
unet_4l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_4_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_4l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210402_055832-flow_unet-unet_4l_skip_use_add")
unet_3l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_3_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_3l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210402_090649-flow_unet-unet_3l_skip_use_add")
unet_2l_skip_use_add = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_2_level_skip_use_add, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_7_1", describe_end="unet_2l_skip_use_add") .set_train_args(epochs=200, epoch_down_step=200, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_14_1_7_1-20210404_110035-flow_unet-unet_2l_skip_use_add")
### 看看 UNet 的 concat 第一個如果不用會不會更好
unet_IN_7l_firstnoC = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_IN_7l_ch64_firstnoC, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1_9_1", describe_end="unet_IN_7l_firstnoC") .set_train_args(epochs=500, epoch_down_step=250, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="")
#############################################################################################################################################################################################################
########################################################### 15
exp_dir15 = "5_15_flow_rect"
rect_fk3_ch64_tfIN_resb_ok9_epoch500  = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_fk3_ch64_tfIN_resb_ok9, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5_15_1", describe_end="epoch500"        ).set_train_args(epochs=500, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_15_1-20210327_162414-flow_rect_fk3_ch64_tfIN_resb_ok9-epoch500_okok")
rect_7_level_fk7 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_7_level_fk7, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5_15_2", describe_end="epoch700ND").set_train_args(epochs=700, epoch_down_step=700, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5_15_2-20210330_140254-flow_rect_7_level-7l_fk7")

rect_2_level_fk3 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_rect_2_level_fk3, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="2l_fk3").set_train_args(epochs=30, epoch_down_step=30, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210330_211817-flow_rect-2l_fk3_35to28")
rect_3_level_fk3 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_rect_3_level_fk3, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="3l_fk3").set_train_args(epochs=30, epoch_down_step=30, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210331_015433-flow_rect-3l_fk3")
rect_4_level_fk3 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_rect_4_level_fk3, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="4l_fk3").set_train_args(epochs=30, epoch_down_step=30, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210331_033121-flow_rect-4l_fk3")
rect_5_level_fk3 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_rect_5_level_fk3, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="5l_fk3").set_train_args(epochs=30, epoch_down_step=30, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210331_043804-flow_rect-5l_fk3")
rect_6_level_fk3 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_rect_6_level_fk3, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="6l_fk3").set_train_args(epochs=30, epoch_down_step=30, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210331_053654-flow_rect-6l_fk3")
rect_7_level_fk3 = Exp_builder().set_basic("train_reload", type8_blender_os_book_768, flow_rect_7_level_fk3, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="7l_fk3").set_train_args(epochs=30, epoch_down_step=30, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210331_063414-flow_rect-7l_fk3")

rect_2_level_fk3_ReLU = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_2_level_fk3_ReLU, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="2l_fk3_ReLU").set_train_args(epochs=100, epoch_down_step=100, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210402_114147-flow_rect-2l_fk3_ReLU")
rect_3_level_fk3_ReLU = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_3_level_fk3_ReLU, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="3l_fk3_ReLU").set_train_args(epochs=100, epoch_down_step=100, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210402_144837-flow_rect-3l_fk3_ReLU")
rect_4_level_fk3_ReLU = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_4_level_fk3_ReLU, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="4l_fk3_ReLU").set_train_args(epochs=100, epoch_down_step=100, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210402_200821-flow_rect-4l_fk3_ReLU")
rect_5_level_fk3_ReLU = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_5_level_fk3_ReLU, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="5l_fk3_ReLU").set_train_args(epochs=100, epoch_down_step=100, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210402_214520-flow_rect-5l_fk3_ReLU")
rect_6_level_fk3_ReLU = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_6_level_fk3_ReLU, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="6l_fk3_ReLU").set_train_args(epochs=100, epoch_down_step=100, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210402_231047-flow_rect-6l_fk3_ReLU")
rect_7_level_fk3_ReLU = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_rect_7_level_fk3_ReLU, G_mse_loss_info, exp_dir=exp_dir15, describe_mid="5", describe_end="7l_fk3_ReLU").set_train_args(epochs=100, epoch_down_step=100, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").build(result_name="type8_blender_os_book-5-20210403_003408-flow_rect-7l_fk3_ReLU")



### 測試subprocessing 有沒有用
# blender_os_book_flow_unet_epoch002 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch2, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch002") .set_train_args(batch_size=30, epochs=1).build(result_name="")
# blender_os_book_flow_unet_epoch003 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch3, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch003") .set_train_args(epochs=3).build(result_name="")
# blender_os_book_flow_unet_epoch004 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch4, G_mse_loss_info, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch004") .set_train_args(epochs=4).build(result_name="")

import sys
if(__name__ == "__main__"):
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        # blender_os_book_flow_unet_new_shuf_ch008_fake.run()
        # rect_fk3_ch64_tfIN_resb_ok9_epoch500.run()
        # rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.run()
        # concat_A.run()
        # flow_rect_7_level_fk7.run()
        # flow_rect_7_level_fk3.run()
        # flow_rect_2_level_fk3.run()
        # unet_2l.run()
        # unet_7l.run()
        # unet_7l_skip_use_add.run()
        # unet_8l.run()
        # rect_2_level_fk3_ReLU.run()
        # ch64_in_epoch500_sigmoid.run()
        in_th_mo_th_gt_th.run()

        print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.run()
    eval(sys.argv[1])
