import tensorflow as tf
import numpy as np 
from enum import Enum 
import time
from step11_result_analyze_after_train import Result
from step08_model_obj import MODEL_NAME
from step09_board_obj import Board_builder
from step06_data_pipline import tf_Data_builder
from step11_result_analyze_after_train import Result_builder
import sys
sys.path.append("kong_util")
from util import time_util


class Experiment():
    def step0_save_code(self):
        import shutil
        from build_dataset_combine import Check_dir_exist_and_build
        code_dir = self.result_obj.result_dir+"/"+"train_code"
        Check_dir_exist_and_build(code_dir)
        shutil.copy("step06_data_pipline.py" ,code_dir + "/" + "step06_data_pipline.py")
        shutil.copy("step06_datas_obj.py"    ,code_dir + "/" + "step06_datas_obj.py")
        shutil.copy("step07_1_UNet_512to256.py",code_dir + "/" + "step7_kong_model1_UNet.py")
        shutil.copy("step07_2_Rect2.py",code_dir + "/" + "step07_2_Rect2.py")
        shutil.copy("step07_2_Rect2.py",code_dir + "/" + "step07_2_Rect2.py")
        shutil.copy("step07_3_just_G.py",code_dir + "/" + "step07_3_just_G.py")
        shutil.copy("step08_model_obj.py" ,code_dir + "/" + "step08_model_obj.py")
        shutil.copy("step09_board_obj.py" ,code_dir + "/" + "step09_board_obj.py")
        shutil.copy("step10_load_and_train_and_test.py" ,code_dir + "/" + "step10_load_and_train_and_test.py")
        shutil.copy("step11_result_analyze_after_train.py" ,code_dir + "/" + "step11_result_analyze_after_train.py")

################################################################################################################################################
################################################################################################################################################
    def __init__(self):
        self.phase        = "train"
        self.db_obj       = None
        self.model_obj    = None                
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
        self.epoch_down_step = 100 ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
        self.epoch_save_freq = 1   ### 訓練 epoch_save_freq 個 epoch 存一次模型
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
    def train_init(self, train_reload=False):### 1.result, 2.data, 3.model(reload), 4.board, 5.save_code 
        ### 1.result
        if(train_reload):self.result_obj = Result_builder().set_by_result_name(self.result_name).build() ### 直接用 自己指定好的 result_name
        else:            self.result_obj = Result_builder().set_by_exp(self).build() ### 需要 db_obj 和 exp本身的describe_mid/end
        ### 2.data
        self.tf_data      = tf_Data_builder().set_basic(self.db_obj).set_img_resize(self.model_obj.model_name).build_by_db_get_method().build() ### tf_data 抓資料
        ### 3.model
        self.ckpt_manager = tf.train.CheckpointManager (checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_dir, max_to_keep=2)  ###step4 建立checkpoint manager 設定最多存2份
        if(train_reload): ### 看需不需要reload model
            self.model_obj.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()

        ####################################################################################################################
        ### 4.board, 5.save_code；train時才需要 board_obj 和 把code存起來喔！test時不用～
        self.board_obj = Board_builder().set_logs_dir_and_summary_writer(self.result_obj.logs_dir).build_by_model_name(self.model_obj.model_name).build() ###step3 建立tensorboard，只有train 和 train_reload需要
        self.step0_save_code() ###    把source code存起來

    def train_reload(self,result_name):
        self.result_name = result_name
        self.train(train_reload=True )

    def train(self, train_reload=False):
        self.train_init(train_reload)
        ################################################################################################################################################
        ### 第三階段：train 和 test
        ###  training 的部分 ###################################################################################################
        ###     以下的概念就是，每個模型都有自己的 generate_images 和 train_step，根據model_name 去各別import 各自的 function過來用喔！
        total_start = time.time()

        ### 多這 這段if 是因為 unet 有move_map的部分，所以要多做以下操作 把 move_map相關會用到的東西存起來
        if(self.model_obj.model_name == MODEL_NAME.unet ):
            from util import get_max_db_move_xy
            self.model_obj.ckpt.max_train_move.assign(self.tf_data.max_train_move)  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            self.model_obj.ckpt.min_train_move.assign(self.tf_data.min_train_move)  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            max_db_move_x, max_db_move_y = get_max_db_move_xy(db_dir=self.db_obj.category, db_name=self.db_obj.db_name) ### g生成的結果 做 apply_rec_move用
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
            lr = 0.0002 if epoch < self.epoch_down_step else 0.0002*(self.epochs-epoch)/(self.epochs-self.epoch_down_step)
            self.model_obj.optimizer_G.lr = lr
            ###############################################################################################################################
            if(epoch==0):print("Initializing Model~~~") ### sample的時候就會initial model喔！
            ###############################################################################################################################
            ###     step1 用來看目前訓練的狀況 
            self.train_step1_see_current_img(epoch)
            ###############################################################################################################################
            ###     step2 訓練
            for n, (_, train_in_pre, _, train_gt_pre) in enumerate( self.tf_data.train_db_combine ):
                print('.', end='')
                if (n+1) % 100 == 0: print()
                if  (self.model_obj.model_name == MODEL_NAME.unet)    :self.model_obj.train_step(self.model_obj, train_in_pre, train_gt_pre, self.board_obj)
                elif(self.model_obj.model_name == MODEL_NAME.rect)    :self.model_obj.train_step(self.model_obj, train_in_pre, train_gt_pre, self.board_obj)
                elif(self.model_obj.model_name == MODEL_NAME.mrf_rect):self.model_obj.train_step(self.model_obj, train_in_pre, train_gt_pre, self.board_obj)
                elif(self.model_obj.model_name == MODEL_NAME.just_G)  :self.model_obj.train_step(self.model_obj, train_in_pre, train_gt_pre, self.board_obj)

            ###############################################################
            ###     step3 整個epoch 的 loss 算平均，存進tensorboard
            self.train_step3_board_save_loss(epoch)
            ###############################################################################################################################
            ###     step4 儲存模型 (checkpoint) the model every "epoch_save_freq" epochs
            if (epoch + 1) % self.epoch_save_freq == 0:
                self.model_obj.ckpt.epoch_log.assign(epoch+1) ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                self.ckpt_manager.save()
                print("save ok ~~~~~~~~~~~~~~~~~")
            ###############################################################################################################################
            ###    step5 紀錄、顯示 訓練相關的時間
            self.train_step5_show_time(epoch, e_start, total_start, epoch_start_timestamp)
            
    def train_step1_see_current_img(self, epoch):
        sample_start_time = time.time()
        see_in_pre = self.tf_data.test_in_db_pre
        see_gt_pre = self.tf_data.test_gt_db_pre
        see_amount = 1
        if(self.db_obj.have_see):
            see_in_pre = self.tf_data.see_in_db_pre
            see_gt_pre = self.tf_data.see_gt_db_pre
            see_amount = self.tf_data.see_amount
        
        for see_index, (test_in_pre, test_gt_pre) in enumerate(zip(see_in_pre.take(see_amount), see_gt_pre.take(see_amount))): 
            if  (self.model_obj.model_name == MODEL_NAME.unet ):     self.model_obj.generate_sees( self.model_obj.generator, see_index, test_in_pre, test_gt_pre, self.tf_data.max_train_move, self.tf_data.min_train_move,  epoch, result_obj.result_dir, result_obj) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif(self.model_obj.model_name == MODEL_NAME.rect ):     self.model_obj.generate_sees( self.model_obj.rect.generator, see_index, test_in_pre, test_gt_pre, epoch, self.result_obj) 
            elif(self.model_obj.model_name == MODEL_NAME.mrf_rect ): self.model_obj.generate_sees( self.model_obj.rect.generator, see_index, test_in_pre, test_gt_pre, epoch, self.result_obj) 
            elif(self.model_obj.model_name == MODEL_NAME.just_G   ): self.model_obj.generate_sees( self.model_obj.generator, see_index, test_in_pre, test_gt_pre, epoch, self.result_obj) 
                                                                                
        print("sample all see time:", time.time()-sample_start_time)

    def train_step3_board_save_loss(self, epoch):
        with self.board_obj.summary_writer.as_default():
            for loss_name, loss_containor in self.board_obj.losses.items():
                tf.summary.scalar(loss_name, loss_containor.result(), step=epoch)
                loss_value = loss_containor.result().numpy()
                if(epoch == 0): ### 第一次 直接把值存成np.array
                    np.save(self.result_obj.logs_dir + "/" + loss_name, np.array(loss_value.reshape(1)))

                else: ### 第二次後，先把np.array先讀出來append值後 再存進去
                    loss_array = np.load(self.result_obj.logs_dir + "/" + loss_name + ".npy")
                    loss_array = np.append(loss_array, loss_value)
                    np.save(self.result_obj.logs_dir + "/" + loss_name, np.array(loss_array))
                    # print(loss_array)

        ###    reset tensorboard 的 loss紀錄容器
        for loss_containor in self.board_obj.losses.values():
            loss_containor.reset_states()
        ###############################################################
        ### 把 loss資訊 用 matplot畫出來
        self.board_obj.see_loss(self.epochs)

    def train_step5_show_time(self, epoch, e_start, total_start, epoch_start_timestamp):
        epoch_cost_time = time.time()-e_start
        total_cost_time = time.time()-total_start
        print('epoch %i start at:%s'         %(epoch, epoch_start_timestamp ))
        print('epoch %i cost time:%.2f'      %(epoch ,epoch_cost_time     ))
        print("batch cost time:%.2f average" %(epoch_cost_time/self.tf_data.train_amount ))
        print("total cost time:%s"           %(time_util(total_cost_time)  ))
        print("esti total time:%s"           %(time_util(epoch_cost_time*self.epochs)))
        print("esti least time:%s"           %(time_util(epoch_cost_time*(self.epochs-(epoch+1)))))
        print("")
        with open(self.result_obj.result_dir + "/" +"cost_time.txt","a") as f:
            f.write( self.phase )                                                                    ;f.write("\n") 
            f.write('epoch %i start at:%s'         %(epoch, epoch_start_timestamp ))                 ;f.write("\n")
            f.write('epoch cost time:%.2f'         %(epoch_cost_time             ))                  ;f.write("\n")
            f.write("batch cost time:%.2f average" %(epoch_cost_time/self.tf_data.train_amount ))    ;f.write("\n")
            f.write("total cost time:%s"           %(time_util(total_cost_time)  ))                  ;f.write("\n")
            f.write("esti total time:%s"           %(time_util(epoch_cost_time*self.epochs)))             ;f.write("\n")
            f.write("esti least time:%s"           %(time_util(epoch_cost_time*(self.epochs-(epoch+1))))) ;f.write("\n")
            f.write("\n")



    def test(self, result_name): ### 1.result, 2.data, 3.model且reload
        ### 1.result 
        self.result_name  = result_name
        self.result_obj   = Result_builder().set_by_result_name(result_name).build()
        ### 2.data
        self.tf_data      = tf_Data_builder().set_basic(self.db_obj).set_img_resize(self.model_obj.model_name).build_by_db_get_method().build() ### tf_data 抓資料
        ### 3.model且reload
        self.ckpt_manager = tf.train.CheckpointManager (checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_dir, max_to_keep=2)  ###step4 建立checkpoint manager 設定最多存2份
        self.model_obj.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()
        ### 待完成

    def run(self):
        if  (self.phase == "train"):          self.train()
        elif(self.phase == "train_reload"):   self.train_reload()
        elif(self.phase == "test"):           self.test()
        elif(self.phase == "train_indicate"): pass ### 待完成


class Exp_builder():
    def __init__(self, exp=None):
        if(exp is None):
            self.exp = Experiment()
        else:self.exp = exp 

    def set_basic(self, phase, db_obj, model_obj, describe_mid=None, describe_end=None, result_name=None):
        self.exp.phase = phase 
        self.exp.db_obj = db_obj 
        self.exp.model_obj = model_obj
        self.exp.describe_mid = describe_mid
        self.exp.describe_end = describe_end
        return self 

    def set_train_args(self, batch_size=1, train_shuffle=True, epochs=700, epoch_down_step=100):
        # self.exp.phase = "train"
        self.exp.batch_size = batch_size
        self.exp.train_shuffle = train_shuffle
        self.exp.epochs = epochs
        self.exp.epoch_down_step=epoch_down_step
        self.exp.start_epoch=0
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
        if(result_name is not None):self.exp.result_name = result_name
        return self.exp

if(__name__=="__main__"):
    from step06_datas_obj import type5c_real_have_see_no_bg_gt_color,\
                              type7_h472_w304_real_os_book_400data,\
                              type7b_h500_w332_real_os_book_1532data
                              
    from step08_model_obj import unet, rect, mrf_rect, just_G


    # using_db_obj = type5c_real_have_see_no_bg_gt_color
    # using_db_obj = type7_h472_w304_real_os_book_400data
    # using_db_obj = type7b_h500_w332_real_os_book_1532data

    # using_model_obj = rect
    # using_model_obj = just_G
    # exp = Exp_builder().set_basic("train", using_db_obj, using_model_obj, describe_end="1532data").set_train_args(epochs=700).build().run(result_name="type7b_h500_w332_real_os_book-20200524-181909-just_G-1532data")
    
    
    os_book_1532_rect_mae1 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_end="1532data_mae1_127.28").set_train_args(epochs=700).build(result_name="")
    os_book_1532_rect_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_end="1532data_mae1_127.35").set_train_args(epochs=700).build(result_name="")
    os_book_1532_rect_mae6 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_end="1532data_mae1_127.51").set_train_args(epochs=700).build(result_name="")
    os_book_1532_just_g_mae1 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, just_G, describe_end="1532data_mae1_127.28").set_train_args(epochs=700).build(result_name="")
    os_book_1532_just_g_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, just_G, describe_end="1532data_mae3_127.51").set_train_args(epochs=700).build(result_name="")
    os_book_1532_just_g_mae6 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, just_G, describe_end="1532data_mae6_128.246").set_train_args(epochs=700).build(result_name="")
    os_book_1532_just_g_mae9 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, just_G, describe_end="1532data_mae9_127.35").set_train_args(epochs=700).build(result_name="")


    os_book_1532_just_g_mae9.run()