import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import time
import datetime
import math
from step0_access_path import Syn_write_to_read_dir, Result_Read_Dir, Result_Write_Dir, kong_model2_dir, JPG_QUALITY
from step06_a_datas_obj import *
from step06_cFinal_tf_Data_builder import tf_Data_builder
from step10_a2_loss_info_obj import *
from step11_b_result_obj_builder import Result_builder
import sys
sys.path.append("kong_util")
from kong_util.util import time_util, get_dir_certain_file_names
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_as_jpg, Find_ltrd_and_crop

from tqdm import tqdm, trange

import socket    ### 取得 本機 IP   給 train_step5_show_time 紀錄
import getpass   ### 取得 本機 User 給 train_step5_show_time 紀錄

start_time = time.time()

class Experiment():
    def __init__(self):
        self.machine_ip   = socket.gethostbyname(socket.gethostname())  ### 取得 本機 IP   給 train_step5_show_time 紀錄
        self.machine_user = getpass.getuser()
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.code_exe_path = None
        ##############################################################################################################################
        self.phase         = "train"
        self.db_builder    = None
        self.db_obj        = None
        self.tf_data       = None
        self.img_resize    = None
        self.model_builder = None
        self.model_obj     = None
        self.loss_info_builders = None
        self.loss_info_objs = []
        self.exp_dir      = None
        self.describe_mid = None
        self.describe_end = "try_try_try_enum"
        self.result_name  = None
        self.result_obj   = None
        ##############################################################################################################################
        self.multi_model_by_history = False
        self.multi_model_reload_exp_builders_dict = {}
        ##############################################################################################################################
        self.board_create_flag = False
        ##############################################################################################################################
        ### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數
        ### train, train_reload 參數
        self.batch_size      = 1
        self.train_shuffle   = True
        self.epochs          = 500    ### 看opencv合成的video覺得1300左右就沒變了
        self.epoch_down_step = self.epochs // 2    ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
        self.epoch_stop      = self.epochs
        self.epoch_save_freq = 5     ### 訓練 epoch_save_freq 個 epoch 存一次模型
        self.start_epoch     = 0
        self.current_ep      = self.start_epoch
        self.current_ep_it   = 0  ### 在 當下 epopch 當中的 哪一個it
        self.current_ex_it   = 0  ### 在 整個 exp    當中的 哪一個 it
        self.ep_see_fq       = 1  ### 因為下面寫了 I_see_fq， 覺得ep 應該也要有， 所以就先寫著備用， 目前還沒用到ˊ口ˋ

        ##################################################################################################
        ### iter 是後來才加入的概念，原先沒有， 所以初始值設為None
        self.one_ep_iters = None  ### (算出來的1) 一個epoch內 總共會  更新 幾次， 建立完tf_data後 才會知道所以現在指定None
        self.total_iters  = None  ### (算出來的2) 整個訓練 包含所有 epochs 總共會  更新 幾次， 建立完tf_data後 才會知道所以現在指定None

        self.it_see_fq       = None  ### (人工指定) 執行幾個 iter 就要存一次 see
        self.it_save_fq      = None  ### (人工指定) 執行幾個 iter 就要存一次 model
        self.it_restart      = None  ### (人工指定) 有設定 it_save_fq 才會有 self.it_start_train， 在 reload 時 要跳到 第 it_restart 個 iter 開始訓練， 所以不給從外面設定喔！
        self.it_down_step    = None  ### (人工指定) 執行到 第幾個iter 就開始 lr 下降， 可輸入 "half" 或 數字
        self.it_down_fq      = None  ### (人工指定) 執行幾個 iter 就要 down 一次 lr
        self.it_show_time_fq = None  ### (人工指定) 執行幾個 iter 顯示一次時間資訊
        ######################################
        ### 給 step5 show_time 用的
        self.total_start_time   = None  ### (跑才知道)
        self.ep_start_time      = None  ### (跑才知道)
        self.ep_start_timestamp = None  ### (跑才知道)
        self.it_start_time      = None  ### (跑才知道)
        self.it_start_timestamp = None  ### (跑才知道)
        self.it_sees_amo_in_one_epoch = None  ### (算出來的3) 1個epoch內 會有幾個 it_see_fq
        self.it_sees_cur_in_one_epoch = None  ### (跑才知道) 目前正在哪個 it_see_fq

        ##################################################################################################
        # self.exp_bn_see_arg = False  ### 本來 bn 在 test 的時候就應該要丟 false，只是 現在batch_size=1， 丟 True 會變 IN ， 所以加這個flag 來控制

        self.use_in_range = Range(min=0, max=1)
        self.use_gt_range = Range(min=0, max=1)
        '''
        覺得不要 mo_use_range，因為本來就應該要 model_output 就直接拿來用了呀，所以直接根據 use_gt_range 來做後處理即可
        output如果不能拿來用 或者 拿來用出問題，代表model設計有問題
        '''

        self.lr_start = 0.0002  ### learning rate
        self.lr_current = self.lr_start

        # self.phase = "train_reload" ### 要記得去決定 result_name 喔！
        # self.phase = "test"         ### test是用固定 train/test 資料夾架構的讀法 ### 要記得去決定 result_name 喔！
        ### self.phase = "test_indicate" ### 這應該是可以刪掉了，因為在取db的時候，已經有包含了test_indecate的取法了！不用在特別區分出來 取資料囉！
        ### 參數設定結束
        ####################################################################################################################
        self.ckpt_read_manager  = None
        self.ckpt_write_manager = None
        self.ckpt_D_read_manager  = None
        self.ckpt_D_write_manager = None

################################################################################################################################################
################################################################################################################################################
    def step0_save_code(self):
        '''
        把 step.py 和 kong_util資料夾 存一份進result
        '''
        import shutil
        from kong_util.build_dataset_combine import Check_dir_exist_and_build
        code_save_path = self.result_obj.train_code_write_dir                 ### 定位出 result存code的目的地
        print("code_save_path:", code_save_path)
        Check_dir_exist_and_build(code_save_path)                             ### 建立目的地資料夾

        py_file_names = get_dir_certain_file_names(kong_model2_dir, certain_word="step")  ### 抓取目前目錄所有 有含 "step" 的檔名
        for py_file_name in py_file_names:
            ### 這兩行可以 抓 step"幾" 然後只存 step"幾".py 喔，但還沒整理以前的code，保險起見還是全存好了，有空再細細處理~~
            # py_file_name_step = int(py_file_name.split("_")[0][4:])          ### 抓出 step "幾"
            # if(py_file_name_step >= 6 or py_file_name_step == 0):            ### step06 以上
            shutil.copy(f"{kong_model2_dir}/{py_file_name}", f"{code_save_path}/{py_file_name}")     ### 存起來

        ### 因為我現在有加 timestamp， 所以 不會有上一版 kong_util的問題， 所以可以註解掉囉～
        # if(os.path.isdir(code_save_path + "/" + "kong_util")):  ### 在train_reload時 如果有舊的kong_util，把舊的刪掉換新的
        #     shutil.rmtree(code_save_path + "/" + "kong_util")
        # print("self.exp_dir:", self.exp_dir )
        shutil.copytree(f"{kong_model2_dir}/kong_util", code_save_path + "/" + "kong_util")
        # shutil.copytree(f"{kong_model2_dir}/kong_Blender", code_save_path + "/" + "kong_Blender")  ### 因為檔名很容易太長， 這個到目前為止都還沒改動到， 覺得先不copy好了 (2022/04/22 決定的)
        # shutil.copytree(f"{kong_model2_dir}/SIFT_dev", code_save_path + "/" + "SIFT_dev")          ### 因為檔名很容易太長， 這個到目前為止都還沒改動到， 覺得先不copy好了 (2022/04/21 決定的)

        code_exe_copy_src = "/".join(self.code_exe_path.split("\\")[:-1])
        code_exe_copy_dst = code_save_path + "/" + "/".join(self.code_exe_path.split("\\")[-3:-1])
        # print("self.code_exe_path:", self.code_exe_path)  ### 舉例： c:\Users\CVML\Desktop\kong_model2\step10_6_mask\mask_5_8b_ch032_tv_s100_bce_s001_100_sobel_k5_s001_100\step10_a.py
        # print("code_exe_copy_src:", code_exe_copy_src)    ### 舉例： c:/Users/CVML/Desktop/kong_model2/step10_6_mask/mask_5_8b_ch032_tv_s100_bce_s001_100_sobel_k5_s001_100
        # print("code_exe_copy_dst:", code_exe_copy_dst)    ### 舉例： data_dir/result/6_mask_unet/5_8b_ch032_tv_s100_bce_s001_100_sobel_k5_s001_100/type8_blender_os_book-8b_6_6-flow_unet-mask_h_bg_ch032_sig_tv_s100_bce_s100_sobel_k5_s100_6l_ep060-20211116_125312/train_code_20211116_125312/step10_6_mask/mask_5_8b_ch032_tv_s100_bce_s001_100_sobel_k5_s001_100
        shutil.copytree(code_exe_copy_src, code_exe_copy_dst)

################################################################################################################################################
################################################################################################################################################
    def exp_init(self, reload_result=False, reload_model=False):  ### 共作四件事： 1.result, 2.data, 3.model(reload), 4.Loss_info
        ### 0.真的建立出 model_obj， 在這之前都還是 KModel_builder喔！ 會寫這麼麻煩是為了 想實現 "真的用到的時候再建構出來！" 這樣才省時間！
        self.model_obj = self.model_builder.build()
        if(self.multi_model_by_history):
            for model_type, generator in self.model_obj.generator.gens_dict.items():
                little_model_loader = tf.train.Checkpoint(generator=generator)
                little_model_path = self.multi_model_reload_exp_builders_dict[model_type].build().result_obj.ckpt_read_dir
                print(f"little_model_path: {little_model_path}")
                ckpt_read_manager = tf.train.CheckpointManager(little_model_loader, little_model_path, max_to_keep=1)
                little_model_loader.restore(ckpt_read_manager.latest_checkpoint)
                print(f"{model_type} load ckpt from:{ckpt_read_manager.latest_checkpoint}")

            if( self.model_obj.discriminator is not None):
                for model_type, discriminator in self.model_obj.discriminator.disc_dict.items():
                    little_model_D_loader = tf.train.Checkpoint(discriminator=discriminator)
                    little_model_D_path   = self.multi_model_reload_exp_builders_dict[model_type].build().result_obj.ckpt_D_read_dir
                    print(f"little_model_D_path: {little_model_D_path}")
                    ckpt_D_read_manager = tf.train.CheckpointManager(little_model_D_loader, little_model_D_path, max_to_keep=1)
                    little_model_D_loader.restore(ckpt_D_read_manager.latest_checkpoint)
                    print(f"{model_type} load ckpt from:{ckpt_D_read_manager.latest_checkpoint}")
        self.db_obj = self.db_builder.build()
        # print("self.model_obj", self.model_obj)  ### 檢查 KModel 有沒有正確的建立出來

        ### 1.result
        if(reload_result):
            # print("self.exp_dir:", self.exp_dir)
            # print("self.result_name:", self.result_name)
            self.result_obj   = Result_builder().set_exp_obj_use_gt_range(self.use_gt_range).set_by_result_name(self.exp_dir + "/" + self.result_name, self.db_obj).build()  ### 直接用 自己指定好的 result_name
            print("Reload: %s ok~~" % (self.result_obj.result_read_dir))
        else:
            self.result_obj = Result_builder().set_exp_obj_use_gt_range(self.use_gt_range).set_by_exp(self).build()  ### exp在train時 要自動建新的 result，才不會覆蓋到之前訓練的result，Result_builder()需要 db_obj 和 exp本身的describe_mid/end

            ### Auto_fill_result_name_Write 檔案
            if(self.phase == "train"):
                ### 定位 path
                code_exe_dir = "\\".join( self.code_exe_path.split("\\")[:-1] )  ### 舉例：'f:\\kong_model2\\Exps_7_v3\\doc3d\\I_to_M_Gk3_no_pad\\pyr_Tcrop256_pad20_jit15\\pyr_3s\\L3\\step09_3side_L3.py' 只取 'f:\\kong_model2\\Exps_7_v3\\doc3d\\I_to_M_Gk3_no_pad\\pyr_Tcrop256_pad20_jit15\\pyr_3s\\L3'
                auto_fill_result_name_file_path = f"{code_exe_dir}/Auto_fill_result_name_{self.machine_ip}.txt"
                ### 要存的 current_exp_result_name， 即 auto_fill_result_name
                auto_fill_result_name = self.result_obj.result_name.split("/")[-1]  ### 舉例：'Exps_7_v3/doc3d/I_to_M_Gk3_no_pad/pyr_Tcrop256_pad20_jit15/pyr_3s/L3/L3_ch032_bl_pyr__1s4__2s3__3s2-20220503_104724'， 只取 'L3_ch032_bl_pyr__1s4__2s3__3s2-20220503_104724'
                with open(auto_fill_result_name_file_path, "a+") as f:
                    f.write(auto_fill_result_name + "\n")
                    print(f"auto_fill_result_name_file_path: {auto_fill_result_name_file_path}, add '{auto_fill_result_name}' finish ~~~")


        ### 2.data，在這邊才建立而不在step6_b 就先建好是因為 如果沒有設定 img_resize時，需要參考 model_name 來決定如何 resize， 所以才寫在 exp 裡面
        if(self.img_resize is None):
            print("沒有設定 img_resize， 就自動根據 db_obj 裡面設定的 h, w 和 model_obj 裡面的 model_name 來自動指定大小囉")        

            # print("doing tf_data resize according model_name")
            # print("self.db_obj.h = ", self.db_obj.h)
            # print("self.db_obj.w = ", self.db_obj.w)
            # print("math.ceil(self.db_obj.h / 128) * 128 = ", math.ceil(self.db_obj.h / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
            # print("math.ceil(self.db_obj.w / 128) * 128 = ", math.ceil(self.db_obj.w / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
            # if  ("unet" in self.model_obj.model_name.value):
            #     self.img_resize = (math.ceil(self.db_obj.h / 128) * 128 , math.ceil(self.db_obj.w / 128) * 128)  ### 128的倍數，且要是gt_img的兩倍大喔！
            # elif("rect" in self.model_obj.model_name.value or "justG" in self.model_obj.model_name.value):
            #     self.img_resize = (math.ceil(self.db_obj.h / 4) * 4, math.ceil(self.db_obj.w / 4) * 4)  ### dis_img(in_img的大小)的大小且要是4的倍數
            self.img_resize = (self.db_obj.h, self.db_obj.w)
            print("自動設定 img_resize 的結果為：", self.img_resize)

        self.tf_data = tf_Data_builder().set_basic(self.db_obj, batch_size=self.batch_size, train_shuffle=self.train_shuffle).set_data_use_range(use_in_range=self.use_in_range, use_gt_range=self.use_gt_range).set_img_resize(self.img_resize).build_by_db_get_method().build()  ### tf_data 抓資料

        ##### 好像變成 it 設定專區， 那就順勢變成 it設定專區 吧～
        ### (算出來的1) 1個epoch內 會 更新 幾次
        self.one_ep_iters = self.tf_data.train_amount // self.batch_size  if(self.tf_data.train_amount % self.batch_size) == 0 else self.tf_data.train_amount // self.batch_size + 1
        ### (算出來的2) 整個exp 總共會  更新 幾次
        self.total_iters = self.epochs * self.one_ep_iters

        ### 知道total_iter後 即可知道 half iter 為多少， 如果 it_down_step設定half 這邊就可以直接指定給他囉～
        if(self.it_down_step == "half"): self.it_down_step = self.total_iters // 2
        ### 防呆， 如果有用it 的概念 但忘記設定 it_show_time_fq， 就直接設定為 it_see_fq， 這樣在存圖時， 就可以順便看看時間囉！
        if(self.it_see_fq is not None and self.it_show_time_fq is None): self.it_show_time_fq = self.it_see_fq
        ### (算出來的3) 計算 1個epoch 裡面 共有幾個 iter_see
        if(self.it_see_fq is not None):
            self.it_sees_amo_in_one_epoch = self.one_ep_iters // self.it_see_fq if(self.one_ep_iters % self.it_see_fq == 0 ) else self.one_ep_iters // self.it_see_fq + 1

        ### 3.model
        self.ckpt_read_manager  = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_read_dir,  max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
        self.ckpt_write_manager = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_write_dir, max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
        if( self.model_obj.discriminator is not None):
            self.ckpt_D_read_manager  = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt_D, directory=self.result_obj.ckpt_D_read_dir,  max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
            self.ckpt_D_write_manager = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt_D, directory=self.result_obj.ckpt_D_write_dir, max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
        if(reload_model):  ### 看需不需要reload model
            self.model_obj.ckpt.restore(self.ckpt_read_manager.latest_checkpoint)
            if(self.model_obj.discriminator is not None): self.model_obj.ckpt_D.restore(self.ckpt_D_read_manager.latest_checkpoint)
            self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()
            self.current_ep  = self.start_epoch
            ### 如果有設定 it_save_fq， 代表之前在train時 已經有 it資訊了(紀錄train到第幾個it)， 這邊就把他讀出來囉
            if(self.it_save_fq is not None):
                self.it_restart = self.model_obj.ckpt.iter_log.numpy()  ### 跳到第幾個it開始訓練 的概念
                self.Current_ep_it_setting_and_update_current_exp_it(value=self.it_restart)  ### 目前的it 指定成 上次的it
            print("Reload: %s Model ok~~ " % self.result_obj.result_read_dir)
            print("    start_epoch=%i" % self.start_epoch)
            if(self.it_save_fq is not None):
                print("    start_itor=%i" % self.it_restart)

        ####################################################################################################################
        ### 4.Loss_info, (5.save_code；train時才需要 loss_info_objs 和 把code存起來喔！test時不用～所以把存code部分拿進train裡囉)
        ###   loss_info_builders 在 exo build的時候就已經有指定一個了， 那個是要給 step12用的
        ###   train 用的 是要搭配 這裡才知道的 result 資訊(logs_read/write_dir) 後的 loss_info_builders
        loss_info_objs = []
        for loss_info_builder in self.loss_info_builders:
            loss_info_builder = loss_info_builder.copy()  ### copy完後，新的 loss_info_builders 更新他的 logs_dir～ 因為有copy 所以 不會 loss_info_obj 都是相同的 logs_read/write_dir 的問題啦！
            loss_info_builder.set_logs_dir(self.result_obj.logs_read_dir, self.result_obj.logs_write_dir)  ### 所以 loss_info_builders 要 根據 result資訊(logs_read/write_dir) 先更新一下    
            loss_info_objs.append(loss_info_builder.build())  ### 上面 logs_read/write_dir 後 更新 就可以建出 loss_info_objs 囉！
        self.loss_info_objs = loss_info_objs

        # for loss_info_builder in self.loss_info_builders:
        #     loss_info_builder.set_logs_dir(self.result_obj.logs_read_dir, self.result_obj.logs_write_dir)  ### 所以 loss_info_builders 要 根據 result資訊(logs_read/write_dir) 先更新一下    
        #     self.loss_info_objs.append(loss_info_builder.build())  ### 上面 logs_read/write_dir 後 更新 就可以建出 loss_info_objs 囉！

        # if(type(self.loss_info_builders) == type([])):
        #     for loss_info_builder in self.loss_info_builders:
        #         loss_info_builder.set_logs_dir(self.result_obj.logs_read_dir, self.result_obj.logs_write_dir)  ### 所以 loss_info_builders 要 根據 result資訊(logs_read/write_dir) 先更新一下    
        #         self.loss_info_objs.append(loss_info_builder.build())  ### 上面 logs_read/write_dir 後 更新 就可以建出 loss_info_objs 囉！
        # else:
        #     self.loss_info_builders.set_logs_dir(self.result_obj.logs_read_dir, self.result_obj.logs_write_dir)  ### 所以 loss_info_builders 要 根據 result資訊(logs_read/write_dir) 先更新一下
        #     self.loss_info_objs = [self.loss_info_builders.build()]  ### 上面 logs_read/write_dir 後 更新 就可以建出 loss_info_objs 囉！


    def train_reload(self):
        self.train(reload_result=True, reload_model=True)

    def train(self, reload_result=False, reload_model=False):
        self.exp_init(reload_result, reload_model)
        self.step0_save_code()  ### training 才需要把source code存起來，所以從exp_init移下來囉
        ################################################################################################################################################
        ### 第三階段：train 和 test
        ###  training 的部分 ###################################################################################################
        ###     以下的概念就是，每個模型都有自己的 generate_results 和 train_step，根據model_name 去各別import 各自的 function過來用喔！
        self.total_start_time = time.time()

        ### 多這 這段if 是因為 unet 有move_map的部分，所以要多做以下操作 把 move_map相關會用到的東西存起來
        if("unet" in self.model_obj.model_name.value and "flow" not in self.model_obj.model_name.value):
            from kong_util.util import get_max_db_move_xy
            self.model_obj.ckpt.max_train_move.assign(self.tf_data.max_train_move)  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            self.model_obj.ckpt.min_train_move.assign(self.tf_data.min_train_move)  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            max_db_move_x, max_db_move_y = get_max_db_move_xy(db_dir=self.db_obj.category, db_name=self.db_obj.db_name)  ### g生成的結果 做 apply_rec_move用
            self.model_obj.ckpt.max_db_move_x.assign(max_db_move_x)  ### 在test時 rec_img需要，所以要存起來
            self.model_obj.ckpt.max_db_move_y.assign(max_db_move_y)  ### 在test時 rec_img需要，所以要存起來
            self.ckpt_write_manager.save()
            print("save ok ~~~~~~~~~~~~~~~~~")

        # self.tf_data.train_db_combine = self.tf_data.train_db_combine.take(20)  ### debug用

        self.current_ep = self.start_epoch   ### 因為 epoch 的狀態 在 train 前後是不一樣的，所以需要用一個變數來記住，就用這個current_epoch來記錄囉！
        for epoch in range(self.start_epoch, self.epochs):
            ### 設定 it 資訊
            it_train_amount = self.one_ep_iters  ### 一個epoch內應該要 迭代幾次， 但有可能是 reload的情況， 會從中間的 iter開始迭代
            # self.current_ep_it  = 10000  ### debug用， 從中間切入看 lr 的狀況
            # it_train_amount -= 10000  ### debug用，跳過前 it_restart 個訓練資料

            ### 處理 it reload 的狀況
            if(self.it_save_fq is not None and self.it_restart is not None):
                self.Current_ep_it_setting_and_update_current_exp_it(value=self.it_restart)  ### 目前的it 指定成 上次的it
                it_train_amount -= self.it_restart  ### 跳過前 it_restart 個訓練資料， 所以 it_train_amount -= 這樣子
                self.it_restart = 0                 ### 功成身退， 設回 0 ， 減了他也沒影響， 這樣子在下個 epoch 時 才不會又 跳過前面的 iter
            ################################
            ###############################################################################################################################
            ###    step0 紀錄epoch開始訓練的時間
            self.ep_start_timestamp = time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime())
            self.ep_start_time = time.time()
            self.it_start_timestamp = time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime())
            self.it_start_time = time.time()
            print("Epoch: ", self.current_ep, "start at", self.ep_start_timestamp)
            ###############################################################################################################################
            ###    step0 設定learning rate
            self.LR_setting()

            ###############################################################################################################################
            if(self.current_ep == 0): print("Initializing Model~~~")  ### sample的時候就會initial model喔！
            ###############################################################################################################################
            ###     step1 用來看目前訓練的狀況
            if(self.current_ep == 0 or (self.current_ep % self.ep_see_fq == 0) ):
                self.train_step1_see_current_img(phase="train", training=False, postprocess=False, npz_save=False)   ### 介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～
            ###############################################################################################################################
            ### 以上 current_ep = epoch   ### 代表還沒訓練(Before train)
            ###     step2 訓練
            if(self.model_obj.discriminator is not None):
                ''' 超重要！ 初始化graph， 必須要走過所有運算流程才行(包含 gradient 和 apply_gradient)， 所以 把 train_step.D_training, G_training 都設True 喔！ '''
                ### 先把要用的物件都抓出來
                init_graph_finished = self.model_obj.train_step.init_graph_finished
                if(init_graph_finished == 0):
                    init_graph_combine = self.tf_data.train_db_combine.take(1)
                    for (_, train_in_pre, _, train_gt_pre, _) in (init_graph_combine):
                        print("==================== init_graph ====================")
                        self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=self.loss_info_objs, D_training=True, G_training=True)
                    self.model_obj.train_step.init_graph_finished = 1
                    print("==================== init_graph_finish ====================")

                ### 先把要用的物件都抓出來
                D_train_amount     = self.model_obj.train_step.D_train_amount
                G_train_amount     = self.model_obj.train_step.G_train_amount
                D_train_many_diff  = self.model_obj.train_step.D_train_many_diff
                G_train_many_diff  = self.model_obj.train_step.G_train_many_diff
                DG_train_many_diff = self.model_obj.train_step.DG_train_many_diff

                ### D,G 根據自己要train的次數， 建立各自 的 dataset
                if  (D_train_many_diff is False): D_train_db_combine = self.tf_data.train_db_combine                         ### 訓練多次時用 same資料
                elif(D_train_many_diff is True) : D_train_db_combine = self.tf_data.train_db_combine.repeat(D_train_amount)  ### 訓練多次時用 diff資料
                if  (G_train_many_diff is False): G_train_db_combine = self.tf_data.train_db_combine                         ### 訓練多次時用 same資料
                elif(G_train_many_diff is True) : G_train_db_combine = self.tf_data.train_db_combine.repeat(G_train_amount)  ### 訓練多次時用 diff資料
                DG_train_db_combine = self.tf_data.train_db_combine  ### DG 都用相同的資料 實用的dataset
                ### 參考： https://www.tensorflow.org/guide/data#basic_mechanics
                D_iter = iter(D_train_db_combine)
                G_iter = iter(G_train_db_combine)
                DG_iter = iter(DG_train_db_combine)

                for go_iter in trange(len(self.tf_data.train_db_combine.take(it_train_amount))):
                    if(DG_train_many_diff is True):
                        ''' 訓練D '''
                        if(D_train_many_diff is False): (_, train_in_pre, _, train_gt_pre, _) = next(D_iter)  ### 訓練多次時用 same資料， 取資料時機再for 外面
                        for _ in range(D_train_amount):
                            if(epoch == 0 and go_iter == 0): print("train D")  ### 確認寫得對不對
                            if(D_train_many_diff is True): (_, train_in_pre, _, train_gt_pre, _) = next(D_iter)  ### 訓練多次時用 diff資料， 取資料時機再for 裡面
                            # plt.imshow(train_in_pre[0][..., :3])
                            # plt.show()
                            self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=self.loss_info_objs, D_training=True, G_training=False)

                        ''' 訓練G '''
                        if(G_train_many_diff is False): (_, train_in_pre, _, train_gt_pre, _) = next(G_iter)  ### 訓練多次時用 same資料， 取資料時機再for 外面
                        for _ in range(G_train_amount):
                            if(epoch == 0 and go_iter == 0): print("train G")  ### 確認寫得對不對
                            if(G_train_many_diff is True): (_, train_in_pre, _, train_gt_pre, _) = next(G_iter)  ### 訓練多次時用 diff資料， 取資料時機再for 裡面
                            # plt.imshow(train_in_pre[0][..., :3])
                            # plt.show()
                            self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=self.loss_info_objs, D_training=False, G_training=True)
                    else:
                        (_, train_in_pre, _, train_gt_pre, _) = next(DG_iter)  ### DG 都用相同的資料
                        ''' 訓練D '''
                        for _ in range(D_train_amount):
                            if(epoch == 0 and go_iter == 0): print("train D")  ### 確認寫得對不對
                            # plt.imshow(train_in_pre[0][..., :3])
                            # plt.show()
                            self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=self.loss_info_objs, D_training=True, G_training=False)
                        ''' 訓練G '''
                        for _ in range(G_train_amount):
                            if(epoch == 0 and go_iter == 0): print("train G")  ### 確認寫得對不對
                            # plt.imshow(train_in_pre[0][..., :3])
                            # plt.show()
                            self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=self.loss_info_objs, D_training=False, G_training=True)

                    self.Current_ep_it_setting_and_update_current_exp_it(value= self.current_ep_it + 1)  ### +1 代表 after_rain 的意思
                    ### iter 看要不要 存圖、設定lr、儲存模型 (思考後覺得要在 after_train做)
                    self.current_it_See_result_or_set_LR_or_Save_Model()
                    # if( self.current_ep_it % 10 == 0): break   ### debug用，看subprocess成不成功
            else:
                for _, train_in_pre, _, train_gt_pre, name in tqdm(self.tf_data.train_db_combine.take(it_train_amount)):
                    ### train
                    # print("%06i" % it, name)  ### debug用
                    self.model_obj.train_step(model_obj=self.model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=self.loss_info_objs)
                    self.Current_ep_it_setting_and_update_current_exp_it(value= self.current_ep_it + 1)  ### +1 代表 after_rain 的意思
                    self.current_ex_it = self.current_ep * self.one_ep_iters + self.current_ep_it  ### 目前的it 在 exp 當中的哪裡
                    ### iter 看要不要 存圖、設定lr、儲存模型 (思考後覺得要在 after_train做)
                    self.current_it_See_result_or_set_LR_or_Save_Model()
                    # if(self.current_ep_it % 10 == 0): break   ### debug用，看subprocess成不成功

            self.current_ep += 1  ### 超重要！別忘了加呀！ 因為進到下個epoch了
            self.Current_ep_it_setting_and_update_current_exp_it(value= 0)  ### 超重要！別忘了加呀！ 因為進到下個epoch了， 所以iter變回0囉
            ### 以下 current_ep = epoch + 1   ### +1 代表訓練完了！變成下個epoch了！
            ###############################################################
            ###     step3 整個epoch 的 loss 算平均，存進tensorboard
            self.train_step3_Loss_info_save_loss()  ### 把他加進去 train_step1_see_current_img 裡面試試看， 所以這邊就不用做了喔， 結果還是拿出來了， 因為忘記 test, train_final_see 不需要 看 loss 呀~~
            ###############################################################################################################################
            ###     step4 儲存模型 (checkpoint) the model every "epoch_save_freq" epochs
            if (self.current_ep) % self.epoch_save_freq == 0:
                self.train_step4_save_model()
                # # print("save epoch_log :", current_ep)
                # self.model_obj.ckpt.epoch_log.assign(self.current_ep)  ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                # if(self.it_save_fq is not None):  self.model_obj.ckpt.iter_log.assign(self.current_ep_it)  ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                # self.ckpt_write_manager.save()
                # if(self.model_obj.discriminator is not None): self.ckpt_D_write_manager.save()
                # print("save ok ~~~~~~~~~~~~~~~~~")
            ###############################################################################################################################
            ###    step5 紀錄、顯示 訓練相關的時間
            self.train_step5_show_time()
            # break  ### debug用，看subprocess成不成功
            ###############################################################################################################################
            ###    step6 同步 result_write_dir 和 result_read_dir
            if(Result_Write_Dir != Result_Read_Dir):  ### 如果要train_reload 或 test  且當 read/write 資料夾位置不一樣時， write完的結果 copy 一份 放回read， 才有 東西 read 喔！
                print("○同步 cost_time.txt, logs_dir, train_code_dir")
                Syn_write_to_read_dir(write_dir=self.result_obj.result_write_dir,     read_dir=self.result_obj.result_read_dir,     build_new_dir=False, print_msg=False)
                Syn_write_to_read_dir(write_dir=self.result_obj.logs_write_dir,       read_dir=self.result_obj.logs_read_dir,       build_new_dir=False, print_msg=False)
                Syn_write_to_read_dir(write_dir=self.result_obj.train_code_write_dir, read_dir=self.result_obj.train_code_read_dir, build_new_dir=False, copy_sub_dir=True, print_msg=False)
                print("●完成～")
                if (self.current_ep) % self.epoch_save_freq == 0:
                    print("○同步 ckpt_dir")
                    Syn_write_to_read_dir(write_dir=self.result_obj.ckpt_write_dir, read_dir=self.result_obj.ckpt_read_dir, build_new_dir=True, print_msg=False)  ### 只想存最新的model，所以 build_new_dir 設 True
                    print("●完成～")

            if(self.current_ep == self.epoch_stop): break   ### 想要有lr 下降，但又不想train滿 中途想離開就 設 epcoh_stop 囉！

        ### 最後train完 記得也要看結果喔！
        self.train_step1_see_current_img(phase="train", training=False, postprocess=False, npz_save=False)   ### 介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～

    def current_it_See_result_or_set_LR_or_Save_Model(self):
        ### 執行 see, 存loss, show_time
        if(self.it_see_fq is not None):
            if( self.current_ep_it % self.it_see_fq == 0):  ### or ### 後來覺得 ep內 最後一個 it 不要存see， 因為 我目前的設定是 ep 是 Before_train存see， it 是 After_train存see， ep內最後一個 it 如果 after存 會跟 ep的before 重複存了！ 目前 doc3d 只train 1個 ep 只會重複一次沒問題， 之後多train epochs 就會多重複 epochs 次囉！
                # self.current_ep_it == self.one_ep_iters):   ### ep內最後一個 it 我希望要存see
                self.train_step1_see_current_img(phase="train", training=False, postprocess=False, npz_save=False)   ### 介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～
                self.train_step3_Loss_info_save_loss()

        ### 設定　lr
        self.LR_setting()

        ### 儲存model
        if(self.it_save_fq is not None):
            if (self.current_ep_it % self.it_save_fq == 0 or
                self.current_ex_it == self.total_iters):   ### 整個exp的最後一個 it 我希望要存
                self.train_step4_save_model()

        ### 顯示 時間
        if(self.it_show_time_fq is not None):
            if( self.current_ep_it % self.it_show_time_fq == 0):
                self.train_step5_show_time()
                self.it_start_time      = time.time()  ### 重設 it 時間
                self.it_start_timestamp = time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime())

    def Current_ep_it_setting_and_update_current_exp_it(self, value):
        self.current_ep_it = value
        self.current_ex_it = self.current_ep * self.one_ep_iters + self.current_ep_it  ### 目前的it 在 exp 當中的哪裡


    def LR_setting(self):
        ### 如果沒有 it下降， 就用ep下降， 直接縣性下降即可( 在epoch_down_step之後， 過一個epoch 下降一次 LR)
        if(self.it_down_step is None):
            self.lr_current = self.lr_start if self.current_ep < self.epoch_down_step else self.lr_start * (self.epochs - self.current_ep) / (self.epochs - self.epoch_down_step)

        ### it 下降比較複雜點， 需要階梯式的下降( 在it_down_step之後， 需要保持 it_down_fq 個 it 其 LR 數值要一樣， 過了 it_down_fq 再下降)
        elif(self.it_down_step is not None):
            if(self.it_down_fq is not None):
                it_down_amount = (self.total_iters - self.it_down_step) // self.it_down_fq  ### LR總共會下降幾次， 用//就好， 如果有餘數就給 最後一個 it_down
                lr_down_hole_value = self.lr_start / it_down_amount  ### LR 一次下降多少 單位
                lr_down_half_value = lr_down_hole_value / 2          ### LR 一次下降多少 單位 的一半

                current_it_down_id = (self.current_ex_it - self.it_down_step) // self.it_down_fq  ### 計算 current_it_down_id 身處在 哪個下降區間
                lr_down_value = 0  ### LR要下降的多少
                if(current_it_down_id >= 0):  ### >=0 代表 current_ex_it 有過 it_down_step 可以開始下降了
                    if(current_it_down_id == 0): lr_down_value = lr_down_half_value  ### 第1個區間 下降 半個單位
                    else                       : lr_down_value = lr_down_half_value + lr_down_hole_value * current_it_down_id  ### 之後的區間 承接第1個區間 的半個單位， 都下降一個單位
                self.lr_current = self.lr_start - lr_down_value
                
                # if( self.current_ep_it % self.it_down_fq == 0):  ### 觸碰到 it_sown_fq 才下調 lr， 才可以階梯式的下降喔
                #     self.lr_current = self.lr_start if self.current_ep_it < self.it_down_step else self.lr_start * (self.total_iters - self.current_ep_it) / (self.total_iters - self.it_down_step)
                #     print("lr_current:", self.lr_current)
            else:
                self.it_down_fq = self.it_see_fq
                print(f"可能忘記設定 self.it_down_fq 了， 所以 lr 目前都不會下降喔！ 所以自動防呆幫你設定成 self.it_see_fq 喔， 數值為 self.it_down_fq={self.it_see_fq}")

        ### LR apply 進 optimizer_G 或 optimizer_D
        self.model_obj.optimizer_G.lr = self.lr_current
        if(self.model_obj.optimizer_D is not None): self.model_obj.optimizer_D.lr = self.lr_current

    def testing(self, add_loss=False, bgr2rgb=False, knpy_save=False):
        print("self.result_obj.test_write_dir", self.result_obj.test_write_dir)
        test_combine_db = [
            self.tf_data.test_in_db.ord.batch(1)       .take(self.tf_data.test_amount),
            self.tf_data.test_in_db.pre .batch(1)      .take(self.tf_data.test_amount),
            self.tf_data.test_gt_db.ord.batch(1)       .take(self.tf_data.test_amount),
            self.tf_data.test_gt_db.pre.batch(1)       .take(self.tf_data.test_amount),
            self.tf_data.test_name_db.ord.batch(1)     .take(self.tf_data.test_amount),
        ]
        if(self.db_obj.have_rec_hope):         test_combine_db += [self.tf_data.rec_hope_test_db.ord.batch(1)      .take(self.tf_data.test_amount)]
        if(self.db_obj.have_DewarpNet_result): test_combine_db += [self.tf_data.DewarpNet_result_test.ord.batch(1) .take(self.tf_data.DewarpNet_result_test_amount)]

        for test_index, datas in enumerate(tqdm(zip(*test_combine_db))):
            rec_hope         = None
            DewarpNet_result = None
            if  (len(datas) == 5): test_in, test_in_pre, test_gt, test_gt_pre, test_name,                            = datas
            elif(len(datas) == 6): test_in, test_in_pre, test_gt, test_gt_pre, test_name, rec_hope                   = datas
            elif(len(datas) == 7): test_in, test_in_pre, test_gt, test_gt_pre, test_name, rec_hope, DewarpNet_result = datas

            test_name = test_name[0].numpy().decode("utf-8")
            # self.model_obj.generate_tests(self.model_obj.generator, test_name, test_in, test_in_pre, test_gt, test_gt_pre, rec_hope=rec_hope, current_ep=self.current_ep, exp_obj=self, training=False, add_loss=False, bgr2rgb=False)
            self.model_obj.generate_sees  (self.model_obj, "test", test_index, test_name, test_in, test_in_pre, test_gt, test_gt_pre, rec_hope, DewarpNet_result, self, training=False, see_reset_init=True, postprocess=True, npz_save=True, knpy_save=knpy_save)
        Save_as_jpg          (ord_dir  =self.result_obj.test_write_dir, dst_dir =self.result_obj.test_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完
        if(self.result_obj.test_write_dir != self.result_obj.test_read_dir):
            Syn_write_to_read_dir(write_dir=self.result_obj.test_write_dir, read_dir=self.result_obj.test_read_dir , build_new_dir=False, print_msg=False, copy_sub_dir=True)

    def train_step1_see_current_img(self, phase="train", training=False, see_reset_init=False, postprocess=False, npz_save=False):
        """
        current_ep      目前 model 正處在 被更新了幾次epoch 的狀態
        training：      可以設定 bn 的動作為 train 還是 test，當batch_size=1時，設為True可以模擬IN，設False圖會壞掉！也可由此得知目前的任務是適合用BN的
                        介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～
        see_reset_init：是給 test_see 用的，有時候製作 fake_exp 的時候，只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要 重建一份see，這時see_reset_init要設True就會重建一下囉
        """
        # sample_start_time = time.time()
        see_combine_db = [
            self.tf_data.see_in_db.ord.batch(1)       .take(self.tf_data.see_amount),
            self.tf_data.see_in_db.pre .batch(1)      .take(self.tf_data.see_amount),
            self.tf_data.see_gt_db.ord.batch(1)       .take(self.tf_data.see_amount),
            self.tf_data.see_gt_db.pre.batch(1)       .take(self.tf_data.see_amount),
            self.tf_data.see_name_db.ord.batch(1)     .take(self.tf_data.see_amount),
        ]
        if(self.db_obj.have_rec_hope):         see_combine_db += [self.tf_data.rec_hope_see_db.ord.batch(1)      .take(self.tf_data.see_amount)]
        if(self.db_obj.have_DewarpNet_result): see_combine_db += [self.tf_data.DewarpNet_result_see.ord.batch(1) .take(self.tf_data.DewarpNet_result_see_amount)]

        for see_index, datas in enumerate(tqdm(zip(*see_combine_db))):
            rec_hope         = None
            DewarpNet_result = None
            if  (len(datas) == 5): see_in, see_in_pre, see_gt, see_gt_pre, see_name,                            = datas
            elif(len(datas) == 6): see_in, see_in_pre, see_gt, see_gt_pre, see_name, rec_hope                   = datas
            elif(len(datas) == 7): see_in, see_in_pre, see_gt, see_gt_pre, see_name, rec_hope, DewarpNet_result = datas

            see_name = see_name[0].numpy().decode("utf-8")
            if  ("unet"  in self.model_obj.model_name.value and
                 "flow"  not in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj     , phase, see_index, see_name, see_in, see_in_pre, see_gt, see_gt_pre, rec_hope, DewarpNet_result, self.tf_data.max_train_move, self.tf_data.min_train_move, self.result_obj.result_write_dir, self, see_reset_init, postprocess=postprocess, npz_save=npz_save, knpy_save=False)  ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif("flow"  in self.model_obj.model_name.value):     self.model_obj.generate_sees(self.model_obj     , phase, see_index, see_name, see_in, see_in_pre, see_gt, see_gt_pre, rec_hope, DewarpNet_result, self, training, see_reset_init, postprocess=postprocess, npz_save=npz_save, knpy_save=False)
            elif("rect"  in self.model_obj.model_name.value):     self.model_obj.generate_sees(self.model_obj.rect, phase, see_index, see_name, see_in, see_in_pre, see_gt, see_gt_pre, rec_hope, DewarpNet_result, self, training, see_reset_init, postprocess=postprocess, npz_save=npz_save, knpy_save=False)
            elif("justG" in self.model_obj.model_name.value):     self.model_obj.generate_sees(self.model_obj     , phase, see_index, see_name, see_in, see_in_pre, see_gt, see_gt_pre, rec_hope, DewarpNet_result, self, training, see_reset_init, postprocess=postprocess, npz_save=npz_save, knpy_save=False)

        # self.result_obj.save_all_single_see_as_matplot_visual_multiprocess() ### 不行這樣搞，對當掉！但可以分開用別的python執行喔～
        # print("sample all see time:", time.time()-sample_start_time)


    def train_step3_Loss_info_save_loss(self):
        if(self.board_create_flag is False):
            for loss_info_obj in self.loss_info_objs:  ### 多output 的 case
                # print("loss_info_obj.logs_write_dir:", loss_info_obj.logs_write_dir)  ###debug用
                loss_info_obj.summary_writer = tf.summary.create_file_writer(loss_info_obj.logs_write_dir)  ### 建tensorboard，這會自動建資料夾喔！所以不用 Check_dir_exist... 之類的，注意 只有第一次 要建立tensorboard喔！
            self.board_create_flag = True

        for loss_info_obj in self.loss_info_objs:  ### 多output 的 case
            with loss_info_obj.summary_writer.as_default():
                for loss_name, loss_containor in loss_info_obj.loss_containors.items():  ### 單個output 裡的 多loss 的 case
                    ### tensorboard
                    tf.summary.scalar(loss_name, loss_containor.result(), step=self.current_ep_it)
                    tf.summary.scalar("lr", self.lr_current, step=self.current_ex_it)

                    ### 自己另存成 npy
                    loss_value = loss_containor.result().numpy()
                    if( os.path.isfile( f"{loss_info_obj.logs_write_dir}/{loss_name}.npy" ) is False):  ### 第一次 直接把值存成np.array
                        loss_npy_array = np.array( [None] * (self.total_iters + 1))  ### 我最末尾還會在做一個， 頭 尾 都做 所以要 +1
                        loss_npy_array[self.current_ex_it] = loss_value
                        np.save(loss_info_obj.logs_write_dir + "/" + loss_name, np.array(loss_npy_array))

                    else:  ### 第二次後，先把np.array先讀出來append值後 再存進去
                        loss_npy_array = np.load(f"{loss_info_obj.logs_write_dir}/{loss_name}.npy", allow_pickle=True)  ### logs_read/write_dir 這較特別！因為這是在 "training 過程中執行的 read" ，  我們想read 的 npy_loss 在train中 是使用  logs_write_dir 來存， 所以就要去 logs_write_dir 來讀囉！ 所以這邊 np.load 裡面適用 logs_write_dir 是沒問題的！
                        loss_npy_array[self.current_ex_it] = loss_value
                        np.save(loss_info_obj.logs_write_dir + "/" + loss_name, np.array(loss_npy_array))
                        # print(loss_npy_array)

        ###    reset tensorboard 的 loss紀錄容器
        for loss_info_obj in self.loss_info_objs:
            for loss_containor in loss_info_obj.loss_containors.values():
                loss_containor.reset_states()
            ###############################################################
            loss_info_obj.see_loss_during_train(self.epochs)  ### 把 loss資訊 用 matplot畫出來
            ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了
            # loss_info_obj.Draw_loss_during_train(epoch, self.epochs)  ### 在 train step1 generate_see裡已經把see的 matplot_visual圖畫出來了，再把 loss資訊加進去

    def train_step4_save_model(self):
        # print("save epoch_log :", current_ep)
        self.model_obj.ckpt.epoch_log.assign(self.current_ep)  ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
        if(self.it_save_fq is not None):
            self.model_obj.ckpt.iter_log.assign(self.current_ep_it)  ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
        self.ckpt_write_manager.save()
        if(self.model_obj.discriminator is not None): self.ckpt_D_write_manager.save()
        print("Save Model finish ~~~~~~~~~~~~~~~~~")

    def train_step5_show_time(self):
        print("current exp:", self.result_obj.result_read_dir)
        epoch_cost_time = time.time() - self.ep_start_time
        total_cost_time = time.time() - self.total_start_time
        it_fq_cost_time = 0
        if(self.it_show_time_fq is not None):
            if( self.current_ep_it % self.it_show_time_fq == 0 or self.current_ep_it == self.one_ep_iters):  ### ep內最後一個 it 我也希望要顯示it time
                it_fq_cost_time = time.time() - self.it_start_time
                ### 計算 it_sees_cur_in_one_epoch
                it_sees_cur_in_one_epoch      = self.current_ep_it // self.it_see_fq
                it_sees_cur_in_one_epoch_frac = self.current_ep_it % self.it_see_fq
                if(it_sees_cur_in_one_epoch_frac == 0): self.it_sees_cur_in_one_epoch = it_sees_cur_in_one_epoch
                else                                  : self.it_sees_cur_in_one_epoch = it_sees_cur_in_one_epoch + 1

        show_time_string = ""
        show_time_string +=  self.phase + "\n"
        show_time_string += 'epoch %i start at:%s, %s, %s' % (self.current_ep, self.ep_start_timestamp, self.machine_ip, self.machine_user) + "\n"
        show_time_string += 'epoch %i cost time:%.2f'      % (self.current_ep, epoch_cost_time) + "\n"
        show_time_string += "batch cost time:%.2f average" % (epoch_cost_time / self.one_ep_iters) + "\n"
        show_time_string += "total cost time:%s"           % (time_util(total_cost_time)) + "\n"
        show_time_string += "esti total time:%s"           % (time_util(epoch_cost_time * self.epochs)) + "\n"
        show_time_string += "esti least time:%s"           % (time_util(epoch_cost_time * (self.epochs - (self.current_ep + 1)))) + "\n"
        if(it_fq_cost_time != 0):
            show_time_string += "  " + "in this epoch, the it cost time:" + "\n"
            show_time_string += "    " + "it_sees: %i / %i"           % (self.it_sees_cur_in_one_epoch, self.it_sees_amo_in_one_epoch)+ "\n"
            show_time_string += "    " + "it %i ~ %i start at: %s"   % ((self.current_ep_it - self.it_show_time_fq), self.current_ep_it, self.it_start_timestamp)+ "\n"
            show_time_string += "    " + 'it_fq %i cost time: %.2f'  % (self.it_show_time_fq, it_fq_cost_time)  + "\n"
            show_time_string += "    " + 'it_avg    cost time: %.3f' % (it_fq_cost_time / self.it_show_time_fq) + "\n"
            show_time_string += "    " + "current  cost time:%s"   % (time_util(total_cost_time)) + "\n"
            show_time_string += "    " + 'it esti total time:%s'   %  time_util( int(  self.one_ep_iters                        / self.it_show_time_fq * it_fq_cost_time) ) + "\n"
            show_time_string += "    " + 'it esti least time:%s'   %  time_util( int( (self.one_ep_iters - self.current_ep_it)  / self.it_show_time_fq * it_fq_cost_time) ) + "\n"
            show_time_string += "\n"

        ### 存到 記事簿
        with open(self.result_obj.result_write_dir + "/" + "cost_time.txt", "a") as f: f.write(show_time_string)
        ### 在 cmd 也 show出來
        print(show_time_string, end="")


    def train_run_final_see(self):
        self.exp_init(reload_result=True, reload_model=True)
        self.train_step1_see_current_img(phase="train", training=False, see_reset_init=True, postprocess=True, npz_save=True)  ### 有時候製作 fake_exp 的時候 ， 只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要reset一下
        ### 雖然 先jpg 再crop 後的結果 佔的空間比較大， 但可以確保 不管重複跑幾次 crop 都不會壞掉喔！ 而這個 train_run_final_see 很容易需要重跑， 所以就將就一下吧～～
        Save_as_jpg          (self.result_obj.result_write_dir, self.result_obj.result_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完
        Find_ltrd_and_crop   (self.result_obj.result_write_dir, self.result_obj.result_write_dir, padding=15, search_amount=2, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        if(self.result_obj.result_write_dir != self.result_obj.result_read_dir):
            Syn_write_to_read_dir(write_dir=self.result_obj.result_write_dir, read_dir=self.result_obj.result_read_dir, build_new_dir=False, print_msg=False, copy_sub_dir=True)
        print("test see finish")

    # def test_see(self):
    #     """
    #     用最後儲存的 Model 來產生see～
    #     也常常拿來 reset in/gt see 喔！
    #     想設定 testing 時的 bn 使用的 training arg 的話， 麻煩用 exp.exp_bn_see_arg 來指定， 因為要用test_see 就要先建exp， 就統一寫在exp裡 個人覺得比較連貫， 因此 就不另外開一個 arg 給 test_see 用囉！
    #     """
    #     self.exp_init(reload_result=True, reload_model=True)
    #     self.train_step1_see_current_img(phase="test", training=False, see_reset_init=True, postprocess=True, npz_save=True)  ### 有時候製作 fake_exp 的時候 ， 只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要reset一下
    #     Syn_write_to_read_dir(write_dir=self.result_obj.result_write_dir, read_dir=self.result_obj.result_read_dir, build_new_dir=False, print_msg=False, copy_sub_dir=True)
    #     print("test see finish")

    def test(self, test_db_name="test", knpy_save=False):  ### 精神不好先暫時用 flow_mask flag 來區別 跟 flow 做不同的動作
        """
        """
        self.db_builder.reset_test_db_name(test_db_name)
        self.exp_init(reload_result=True, reload_model=True)
        self.testing(knpy_save=knpy_save)  ### 有時候製作 fake_exp 的時候 ， 只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要reset一下
        print("test finish")

    def board_rebuild(self):
        self.exp_init(reload_result=True, reload_model=False)
        for loss_info_obj in self.loss_info_objs:
            loss_info_obj.use_npy_rebuild_justG_tensorboard_loss(self, dst_dir=self.result_obj.logs_write_dir)
        print("board_rebuild finish")
        print("")

    def copy_ckpt(self):
        import shutil
        # Check_dir_exist_and_build_new_dir(self.result_obj.ckpt_write_dir)
        shutil.copytree(self.result_obj.ckpt_read_dir, self.result_obj.ckpt_write_dir)

    def Gather_test_SSIM_LD(self):
        def ax_dot_text(ax, values):
            for go_v, value in enumerate(values):
                ax.annotate( text="%.1f" %  value,            ### 顯示的文字
                    xy=(go_v, value),                         ### 要標註的目標點
                    xytext=( 0 , 5),                          ### 顯示的文字放哪裡
                    textcoords='offset points',               ### 目前東西放哪裡的坐標系用什麼
                    # arrowprops=dict(arrowstyle="->",          ### 畫箭頭的資訊
                    #                 connectionstyle= "arc3", )
                                    )
                ax.plot((go_v, go_v), (0, value), c="gray", alpha=0.5)

        def draw(ax, values, text, xmin, xmax, ymin, ymax):
            amount = len(values)
            mean = np.mean(values)

            x_ticks = range(amount)
            ax.plot(x_ticks, values         , label=f"{text}s")
            ax.plot(x_ticks, [mean] * amount, label=f"{text}_mean")

            ax.set_title(f"{text}s_mean:{mean}")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(list(x_ticks), rotation = 90)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.legend()
            ax_dot_text(ax, values)

        ### 去各個 test料夾 蒐集出 lds 和 ssims
        lds = []
        ssims = []
        for test in self.result_obj.tests:
            ld_path   = test.metric_read_dir + "/" + "LDs.npy"
            ssim_path = test.metric_read_dir + "/" + "SSIMs.npy"
            ld   = np.load(ld_path)
            ssim = np.load(ssim_path)
            ld   = ld  [-1]  ### -1 是取 final 的概念，因為如果是用 see， 可能會有很多個 epoch 的 LD/SSIM 數值喔！ 取-1 test也適用。
            ssim = ssim[-1]  ### -1 是取 final 的概念，因為如果是用 see， 可能會有很多個 epoch 的 LD/SSIM 數值喔！ 取-1 test也適用。

            lds  .append(ld)
            ssims.append(ssim)
        lds   = np.array(lds)
        ssims = np.array(ssims)
        lds_mean   = np.mean(lds)
        ssims_mean = np.mean(ssims)
        amount = len(lds)

        Check_dir_exist_and_build(self.result_obj.test_write_dir)

        lds_mean_path   = self.result_obj.test_write_dir + "/" + "LDs_mean"
        ssims_mean_path = self.result_obj.test_write_dir + "/" + "SSIMs_mean"
        np.save(lds_mean_path, lds_mean)
        np.save(ssims_mean_path, ssims_mean)

        ld_rotate_126_path   = self.result_obj.tests[126].metric_read_dir + "/rotate/" + "LDs.npy"
        ssim_rotate_126_path = self.result_obj.tests[126].metric_read_dir + "/rotate/" + "SSIMs.npy"
        ld_rotate_127_path   = self.result_obj.tests[127].metric_read_dir + "/rotate/" + "LDs.npy"
        ssim_rotate_127_path = self.result_obj.tests[127].metric_read_dir + "/rotate/" + "SSIMs.npy"

        if(os.path.isfile(ld_rotate_126_path   ) and
           os.path.isfile(ssim_rotate_126_path ) and
           os.path.isfile(ld_rotate_127_path   ) and
           os.path.isfile(ssim_rotate_127_path )  ):
            ld_rotate_126   = np.load(ld_rotate_126_path)
            ld_rotate_127   = np.load(ld_rotate_127_path)
            ssim_rotate_126 = np.load(ssim_rotate_126_path)
            ssim_rotate_127 = np.load(ssim_rotate_127_path)

            lds_rotate = lds.copy()
            ssims_rotate = ssims.copy()
            lds_rotate[126] = ld_rotate_126
            lds_rotate[127] = ld_rotate_127
            ssims_rotate[126] = ssim_rotate_126
            ssims_rotate[127] = ssim_rotate_127

            lds_rotate_mean   = np.mean(lds_rotate)
            ssims_rotate_mean = np.mean(ssims_rotate)

            lds_rotate_mean_path   = self.result_obj.test_write_dir + "/" + "rotate_LDs_mean"
            ssims_rotate_mean_path = self.result_obj.test_write_dir + "/" + "rotate_SSIMs_mean"
            np.save(lds_rotate_mean_path, lds_rotate_mean)
            np.save(ssims_rotate_mean_path, ssims_rotate_mean)

            ### lds 和 ssims 寫成 txt
            ld_ssim_rotate_value_txt_path    = self.result_obj.test_write_dir + "/" + "rotate_LD_SSIM_value.txt"
            with open(ld_ssim_rotate_value_txt_path, "w") as f:
                f.write(f"mean\n")
                f.write(f"ld_mean      : {lds_rotate_mean      }\n")
                f.write(f"ssim_mean    : {ssims_rotate_mean    }\n")

                for go_val in range(amount):
                    ssim    =  ssims_rotate   [go_val]
                    ld      =  lds_rotate     [go_val]
                    f.write(f"  current_id : {go_val}\n")
                    f.write(f"    my_rec_ssim    : {ssim    }\n")
                    f.write(f"    my_rec_ld      : {ld      }\n")

            ### lds 和 ssims matplot視覺化
            ld_ssim_rotate_value_visual_path = self.result_obj.test_write_dir + "/" + "rotate_LD_SSIM_value_visual"
            import matplotlib.pyplot as plt
            canvas_size = 3
            nrows = 2
            ncols = 1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(canvas_size * ncols * 6.3, canvas_size * nrows))
            draw(ax=ax[0], values=lds_rotate  , text="LD_rotate"  , xmin=0, xmax=129, ymin=0, ymax=50)
            draw(ax=ax[1], values=ssims_rotate, text="SSIM_rotate", xmin=0, xmax=129, ymin=0, ymax= 1)
            fig.tight_layout()
            # plt.show()
            plt.savefig(ld_ssim_rotate_value_visual_path)


        ### lds 和 ssims 寫成 txt
        ld_ssim_value_txt_path    = self.result_obj.test_write_dir + "/" + "LD_SSIM_value.txt"
        with open(ld_ssim_value_txt_path, "w") as f:
            f.write(f"mean\n")
            f.write(f"ld_mean      : {lds_mean      }\n")
            f.write(f"ssim_mean    : {ssims_mean    }\n")

            for go_val in range(amount):
                ssim    =  ssims   [go_val]
                ld      =  lds     [go_val]
                f.write(f"  current_id : {go_val}\n")
                f.write(f"    my_rec_ssim    : {ssim    }\n")
                f.write(f"    my_rec_ld      : {ld      }\n")

        ### writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
        ld_ssim_value_tboard_write_path    = self.result_obj.test_write_dir + "/" + "LD_SSIM_tboard"
        ld_ssim_value_tboard_read_path     = self.result_obj.test_read_dir  + "/" + "LD_SSIM_tboard"
        writer = tf.summary.create_file_writer(ld_ssim_value_tboard_write_path)
        with writer.as_default():
            for go_val in range(amount):
                tf.summary.scalar("SSIMs", ssims[go_val], step=go_val)
                tf.summary.scalar("LDs"  , lds  [go_val], step=go_val)
                if(go_val > 0):
                    tf.summary.scalar("mean_SSIM", np.mean(ssims[:go_val]), step=go_val)
                    tf.summary.scalar("mean_LD"  , np.mean(lds  [:go_val]), step=go_val)

            if(os.path.isfile(ld_rotate_126_path   ) and
                os.path.isfile(ssim_rotate_126_path ) and
                os.path.isfile(ld_rotate_127_path   ) and
                os.path.isfile(ssim_rotate_127_path )  ):
                for go_val in range(amount):
                    tf.summary.scalar("rotate_SSIMs", ssims_rotate[go_val], step=go_val)
                    tf.summary.scalar("rotate_LDs"  , lds_rotate  [go_val], step=go_val)
                    if(go_val > 0):
                        tf.summary.scalar("rotate_mean_SSIM", np.mean(ssims_rotate[:go_val]), step=go_val)
                        tf.summary.scalar("rotate_mean_LD"  , np.mean(lds_rotate  [:go_val]), step=go_val)


        ### lds 和 ssims matplot視覺化
        ld_ssim_value_visual_path = self.result_obj.test_write_dir + "/" + "LD_SSIM_value_visual"
        import matplotlib.pyplot as plt
        canvas_size = 3
        nrows = 2
        ncols = 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(canvas_size * ncols * 6.3, canvas_size * nrows))
        draw(ax=ax[0], values=lds  , text="LD"  , xmin=0, xmax=129, ymin=0, ymax=50)
        draw(ax=ax[1], values=ssims, text="SSIM", xmin=0, xmax=129, ymin=0, ymax= 1)
        fig.tight_layout()
        # plt.show()
        plt.savefig(ld_ssim_value_visual_path)

        ### 同步 test_write / test_read
        if(self.result_obj.test_write_dir != self.result_obj.test_read_dir):
            Syn_write_to_read_dir(write_dir=self.result_obj.test_write_dir , read_dir=self.result_obj.test_read_dir , build_new_dir=False, print_msg=True, copy_sub_dir=False)  ### 這個function寫出來的東西只有在 test_write_dir， 沒有寫進 sub_dir 所以不需要copy sub_dir
            Syn_write_to_read_dir(write_dir=ld_ssim_value_tboard_write_path, read_dir=ld_ssim_value_tboard_read_path, build_new_dir=False, print_msg=True, copy_sub_dir=True )  ### 注意 tensorboard 是寫成資料夾， 所以copy_sub_dir 要 設 True 喔！
        print("Gather_test_SSIM_LD finish")


    def run(self):
        self.machine_ip   = socket.gethostbyname(socket.gethostname())  ### 取得 本機 IP   給 train_step5_show_time 紀錄
        self.machine_user = getpass.getuser()                           ### 取得 本機 User 給 train_step5_show_time 紀錄
        if  (self.phase == "train"):          self.train()
        elif(self.phase == "train_reload"):   self.train_reload()
        elif(self.phase == "train_run_final_see"): self.train_run_final_see()
        elif(self.phase == "train_indicate"): pass  ### 待完成Z
        elif("test" in self.phase):
            # if(self.phase == "test_see"): self.test_see()
            if  (self.phase == "test_see"  ): self.test(test_db_name="see"      , knpy_save=False)
            elif(self.phase == "test_train"): self.test(test_db_name="train"    , knpy_save=False)
            else:                             self.test(test_db_name=self.phase , knpy_save=False)  ### 精神不好先暫時用 flow_mask flag 來區別 跟 flow 做不同的動作
        elif(self.phase == "board_rebuild"):  self.board_rebuild()
        elif(self.phase == "copy_ckpt"): self.copy_ckpt()
        elif(self.phase.lower() == "ok"): pass      ### 不做事情，只是個標記而以這樣子
        else: print("ㄘㄋㄇㄉ phase 打錯字了拉~~~")

