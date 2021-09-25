import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import time
from step0_access_path import Syn_write_to_read_dir, result_read_path, result_write_path
from step06_a_datas_obj import *
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

start_time = time.time()

class Experiment():
    def step0_save_code(self):
        '''
        把 step.py 和 kong_util資料夾 存一份進result
        '''
        import shutil
        from build_dataset_combine import Check_dir_exist_and_build
        code_dir = self.result_obj.train_code_write_dir  ### 定位出 result存code的目的地
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
        self.phase         = "train"
        self.db_builder    = None
        self.db_obj        = None
        self.model_builder = None
        self.model_obj     = None
        self.loss_info_builder = None
        self.loss_info_obj = None
        self.exp_dir      = None
        self.describe_mid = None
        self.describe_end = "try_try_try_enum"
        self.result_name  = None
        self.result_obj   = None
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
        self.ckpt_read_manager  = None
        self.ckpt_write_manager = None

################################################################################################################################################
################################################################################################################################################
    def exp_init(self, reload_result=False, reload_model=False):  ### 共作四件事： 1.result, 2.data, 3.model(reload), 4.Loss_info
        ### 0.真的建立出 model_obj， 在這之前都還是 KModel_builder喔！ 會寫這麼麻煩是為了 想實現 "真的用到的時候再建構出來！" 這樣才省時間！
        self.model_obj = self.model_builder.build()
        self.db_obj = self.db_builder.build()
        # print("self.model_obj", self.model_obj)  ### 檢查 KModel 有沒有正確的建立出來

        ### 1.result
        if(reload_result):
            # print("self.exp_dir:", self.exp_dir)
            # print("self.result_name:", self.result_name)
            self.result_obj   = Result_builder().set_by_result_name(self.exp_dir + "/" + self.result_name, self.in_use_range, self.gt_use_range).build()  ### 直接用 自己指定好的 result_name
            print("Reload: %s ok~~" % (self.result_obj.result_name))
        else: self.result_obj = Result_builder().set_by_exp(self).build()  ### exp在train時 要自動建新的 result，才不會覆蓋到之前訓練的result，Result_builder()需要 db_obj 和 exp本身的describe_mid/end
        ### 2.data，在這邊才建立而不在step6_b 就先建好是因為 要參考 model_name 來決定如何 resize 喔！
        self.tf_data      = tf_Data_builder().set_basic(self.db_obj, batch_size=self.batch_size, train_shuffle=self.train_shuffle).set_data_use_range(in_use_range=self.in_use_range, gt_use_range=self.gt_use_range).set_img_resize(self.model_obj.model_name).build_by_db_get_method().build()  ### tf_data 抓資料
        ### 3.model
        self.ckpt_read_manager  = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_read_dir,  max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
        self.ckpt_write_manager = tf.train.CheckpointManager(checkpoint=self.model_obj.ckpt, directory=self.result_obj.ckpt_write_dir, max_to_keep=1)  ###step4 建立checkpoint manager 設定最多存2份
        if(reload_model):  ### 看需不需要reload model
            self.model_obj.ckpt.restore(self.ckpt_read_manager.latest_checkpoint)
            self.start_epoch = self.model_obj.ckpt.epoch_log.numpy()
            print("Reload: %s Model ok~~ start_epoch=%i" % (self.result_obj.result_name, self.start_epoch))

        ####################################################################################################################
        ### 4.Loss_info, (5.save_code；train時才需要 loss_info_obj 和 把code存起來喔！test時不用～所以把存code部分拿進train裡囉)
        ###   loss_info_builder 在 exo build的時候就已經有指定一個了， 那個是要給 step12用的
        ###   train 用的 是要搭配 這裡才知道的 result 資訊(logs_read/write_dir) 後的 loss_info_builder
        self.loss_info_builder.set_logs_dir(self.result_obj.logs_read_dir, self.result_obj.logs_write_dir)  ### 所以 loss_info_builder 要 根據 result資訊(logs_read/write_dir) 先更新一下
        self.loss_info_obj = self.loss_info_builder.build()  ### 上面 logs_read/write_dir 後 更新 就可以建出 loss_info_obj 囉！


    def train_reload(self):
        self.train(reload_result=True, reload_model=True)

    def train(self, reload_result=False, reload_model=False):
        self.exp_init(reload_result, reload_model)
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
            self.ckpt_write_manager.save()
            print("save ok ~~~~~~~~~~~~~~~~~")

        current_epoch = self.start_epoch   ### 因為 epoch 的狀態 在 train 前後是不一樣的，所以需要用一個變數來記住，就用這個current_epoch來記錄囉！
        for epoch in range(self.start_epoch, self.epochs):
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
                self.ckpt_write_manager.save()
                print("save ok ~~~~~~~~~~~~~~~~~")
            ###############################################################################################################################
            ###    step5 紀錄、顯示 訓練相關的時間
            self.train_step5_show_time(current_epoch, e_start, total_start, epoch_start_timestamp)
            # break  ### debug用，看subprocess成不成功
            ###############################################################################################################################
            ###    step6 同步 result_write_dir 和 result_read_dir
            if(result_write_path != result_read_path):  ### 如果要train_reload 或 test  且當 read/write 資料夾位置不一樣時， write完的結果 copy 一份 放回read， 才有 東西 read 喔！
                print("○同步 cost_time.txt, logs_dir, train_code_dir")
                Syn_write_to_read_dir(write_dir=self.result_obj.result_write_dir,     read_dir=self.result_obj.result_read_dir,     build_new_dir=False, print_msg=False)
                Syn_write_to_read_dir(write_dir=self.result_obj.logs_write_dir,       read_dir=self.result_obj.logs_read_dir,       build_new_dir=False, print_msg=False)
                Syn_write_to_read_dir(write_dir=self.result_obj.train_code_write_dir, read_dir=self.result_obj.train_code_read_dir, build_new_dir=False, copy_sub_dir=True, print_msg=False)
                print("●完成～")
                if (current_epoch) % self.epoch_save_freq == 0:
                    print("○同步 ckpt_dir")
                    Syn_write_to_read_dir(write_dir=self.result_obj.ckpt_write_dir, read_dir=self.result_obj.ckpt_read_dir, build_new_dir=True, print_msg=False)  ### 只想存最新的model，所以 build_new_dir 設 True
                    print("●完成～")

            if(current_epoch == self.epoch_stop): break   ### 想要有lr 下降，但又不想train滿 中途想離開就 設 epcoh_stop 囉！

        ### 最後train完 記得也要看結果喔！
        self.train_step1_see_current_img(current_epoch, training=self.exp_bn_see_arg)   ### 介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～

    def testing(self, current_epoch, add_loss=False, bgr2rgb=False, flow_mask=False):
        from build_dataset_combine import Check_dir_exist_and_build_new_dir,  method1
        from matplot_fig_ax_util import Matplot_single_row_imgs
        from flow_bm_util import use_flow_to_get_bm, use_bm_to_rec_img
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=4)
        fig2, ax2 = plt.subplots(nrows=1, ncols=3)
        fig3, ax3 = plt.subplots(nrows=1, ncols=3)
        fig_bm, ax_bm = plt.subplots(nrows=1, ncols=2)

        print("self.result_obj.test_dir", self.result_obj.test_dir)
        Check_dir_exist_and_build_new_dir(self.result_obj.test_dir)
        test_in     = self.tf_data.test_in_db
        test_in_pre = self.tf_data.test_in_db_pre
        test_gt     = self.tf_data.test_gt_db

        flows = []
        for test_index, (test_in, test_in_pre, test_gt, test_gt_pre) in enumerate(tqdm(self.tf_data.test_db_combine)):
            # print("test_index~~~~~~~~~~~~~~~~", test_index)
            # print("test_in.shape", test_in.shape)
            # print("test_in_pre.shape", test_in_pre.shape)
            # print("test_gt.shape", test_gt.shape)
            # print("test_gt_pre.shape", test_gt_pre.shape)
            if  (flow_mask is True):
                in_img    = test_in[0].numpy()   ### HWC 和 tensor -> numpy
                pred_mask      = self.model_obj.generate_results(model_G=self.model_obj.generator, in_img_pre=test_in_pre, gt_use_range=self.gt_use_range)  ### BHWC
                pred_mask      = pred_mask[0].numpy()   ### HWC 和 tensor -> numpy
                gt_mask    = test_gt[0][0].numpy()

                single_row_imgs = Matplot_single_row_imgs(
                                        imgs      =[ in_img ,   pred_mask ,        gt_mask],    ### 把要顯示的每張圖包成list
                                        img_titles=["in_img", "pred_mask", "gt_mask"],    ### 把每張圖要顯示的字包成list
                                        fig_title ="test_%04i, epoch=%04i" % (test_index, current_epoch),   ### 圖上的大標題
                                        add_loss  =add_loss,
                                        bgr2rgb   =bgr2rgb)
                single_row_imgs.Draw_img()
                single_row_imgs.Save_fig(dst_dir=self.result_obj.test_dir, epoch=current_epoch, epoch_name="test_%04i" % test_index)  ### 如果沒有要接續畫loss，就可以存了喔！

                pass
            elif(flow_mask is False):
                in_img    = test_in[0].numpy()   ### HWC 和 tensor -> numpy
                ax[0].imshow(test_in[0])

                flow           = self.model_obj.generate_results(model_G=self.model_obj.generator, in_img_pre=test_in_pre, gt_use_range=self.gt_use_range)  ### BHWC
                flow           = flow[0].numpy()   ### HWC 和 tensor -> numpy
                # flow           = flow[..., ::-1]
                flow[..., 1]   = 1 - flow[..., 1]  ### y 上下 flip
                if(self.gt_use_range == "-1~1"): flow = (flow + 1) / 2   ### 如果 gt_use_range 是 -1~1 記得轉回 0~1
                print(" flow.shape", flow.shape)
                print(" flow.min()", flow.min())
                print(" flow.max()", flow.max())
                flow_v    = method1(flow[..., 1], flow[..., 2])  ### [..., ::-1] * 255. ### 如果用opencv存，才需要rgb->bgr 和 range:0~255
                ax[1].imshow(flow_v)
                ax2[0].imshow(flow[..., 0])
                ax2[1].imshow(flow[..., 1])
                ax2[2].imshow(flow[..., 2])

                gt_flow        = test_gt[0].numpy()   ### HWC 和 tensor -> numpy
                gt_flow_visual = method1(gt_flow[..., 2], gt_flow[..., 1])
                ax[2].imshow(gt_flow_visual)
                ax3[0].imshow(gt_flow[..., 0])
                ax3[1].imshow(gt_flow[..., 1])
                ax3[2].imshow(gt_flow[..., 2])
                print(" gt_flow.min()", gt_flow.min())
                print(" gt_flow.max()", gt_flow.max())


                valid_mask_pix_amount = (flow[..., 0] >= 0.99).astype(np.int).sum()
                total_pix_amount = flow.shape[0] * flow.shape[1]
                # print("valid_mask_pix_amount / total_pix_amount:", valid_mask_pix_amount / total_pix_amount)
                if( valid_mask_pix_amount / total_pix_amount > 0.25):
                    print("flow.shape", flow.shape)
                    print("type(flow)", type(flow))
                    plt.show()
                    print("valid_mask_pix_amount / total_pix_amount~~~~~~~~~~~~~~~~~~~~~~", valid_mask_pix_amount / total_pix_amount)
                    bm  = use_flow_to_get_bm(flow, flow_scale=768)
                    print("finish bm")
                    ax_bm[0].imshow(bm[..., 0])
                    ax_bm[1].imshow(bm[..., 1])
                    # plt.show()
                    plt.close()
                    # rec = use_bm_to_rec_img(bm, flow_scale=768, dis_img=in_img)
                    rec = use_bm_to_rec_img(bm, flow_scale=768, dis_img=test_in_pre[0].numpy())
                else:
                    bm  = np.zeros(shape=(768, 768, 2))
                    rec = np.zeros(shape=(768, 768, 3))

                # if(gt_flow.sum() > 0):
                #     gt_bm  = use_flow_to_get_bm(gt_flow, flow_scale=768)
                #     gt_rec = use_bm_to_rec_img(gt_bm, flow_scale=768, dis_img=in_img)
                # else:
                #     gt_bm  = np.zeros(shape=(768, 768, 2))
                #     gt_rec = np.zeros(shape=(768, 768, 3))

                # bm_visual  = method1(bm[...,0], bm[...,1]*-1)
                # gt_bm_visual = method1(gt_bm[...,0], gt_bm[...,1]*-1)
                single_row_imgs = Matplot_single_row_imgs(
                                        imgs      =[ in_img ,   flow_v ,        rec],    ### 把要顯示的每張圖包成list
                                        img_titles=["in_img", "pred_flow_v", "pred_rec"],    ### 把每張圖要顯示的字包成list
                                        fig_title ="test_%04i, epoch=%04i" % (test_index, current_epoch),   ### 圖上的大標題
                                        add_loss  =add_loss,
                                        bgr2rgb   =bgr2rgb)
                single_row_imgs.Draw_img()
                single_row_imgs.Save_fig(dst_dir=self.result_obj.test_dir, epoch=current_epoch, epoch_name="test_%04i" % test_index)  ### 如果沒有要接續畫loss，就可以存了喔！



    def train_step1_see_current_img(self, epoch, training=False, see_reset_init=False):
        """
        epoch：         目前 model 正處在 被更新了幾次epoch 的狀態
        training：      可以設定 bn 的動作為 train 還是 test，當batch_size=1時，設為True可以模擬IN，設False圖會壞掉！也可由此得知目前的任務是適合用BN的
                        介面目前的設計雖然規定一定要丟 training 這個參數， 但其實我底層在實作時 也會視情況 不需要 training 就不會用到喔，像是 IN 拉，所以如果是 遇到使用 IN 的generator，這裡的 training 亂丟 None也沒問題喔～因為根本不會用他這樣～
        see_reset_init：是給 test_see 用的，有時候製作 fake_exp 的時候，只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要 重建一份see，這時see_reset_init要設True就會重建一下囉
        """
        # sample_start_time = time.time()

        for see_index, (test_in, test_in_pre, test_gt, rec_hope_pre) in enumerate(tqdm(zip(self.tf_data.see_in_db          .take(self.tf_data.see_amount),
                                                                                           self.tf_data.see_in_db_pre      .take(self.tf_data.see_amount),
                                                                                           self.tf_data.see_gt_db          .take(self.tf_data.see_amount),
                                                                                           self.tf_data.rec_hope_see_db_pre.take(self.tf_data.see_amount)))):
            if  ("unet"  in self.model_obj.model_name.value and
                 "flow"  not in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in, test_in_pre, test_gt, rec_hope_pre, self.tf_data.max_train_move, self.tf_data.min_train_move, epoch, self.result_obj.result_write_dir, result_obj, see_reset_init)  ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif("flow"  in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in, test_in_pre, test_gt, rec_hope_pre, epoch, self.result_obj, training, see_reset_init)
            elif("rect"  in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.rect.generator, see_index, test_in, test_in_pre, test_gt, rec_hope_pre, epoch, self.result_obj, see_reset_init)
            elif("justG" in self.model_obj.model_name.value): self.model_obj.generate_sees(self.model_obj.generator     , see_index, test_in, test_in_pre, test_gt, rec_hope_pre, epoch, self.result_obj, see_reset_init)

        # self.result_obj.save_all_single_see_as_matplot_visual_multiprocess() ### 不行這樣搞，對當掉！但可以分開用別的python執行喔～
        # print("sample all see time:", time.time()-sample_start_time)

    def train_step3_Loss_info_save_loss(self, epoch):
        if(self.board_create_flag is False):
            self.loss_info_obj.summary_writer = tf.summary.create_file_writer(self.loss_info_obj.logs_write_dir)  ### 建tensorboard，這會自動建資料夾喔！所以不用 Check_dir_exist... 之類的，注意 只有第一次 要建立tensorboard喔！
            self.board_create_flag = True

        with self.loss_info_obj.summary_writer.as_default():
            for loss_name, loss_containor in self.loss_info_obj.loss_containors.items():
                ### tensorboard
                tf.summary.scalar(loss_name, loss_containor.result(), step=epoch)
                tf.summary.scalar("lr", self.lr_current, step=epoch)

                ### 自己另存成 npy
                loss_value = loss_containor.result().numpy()
                if(epoch == 1):  ### 第一次 直接把值存成np.array
                    np.save(self.result_obj.logs_write_dir + "/" + loss_name, np.array(loss_value.reshape(1)))

                else:  ### 第二次後，先把np.array先讀出來append值後 再存進去
                    loss_array = np.load(self.result_obj.logs_write_dir + "/" + loss_name + ".npy")  ### logs_read/write_dir 這較特別！因為這是在 "training 過程中執行的 read" ，  我們想read 的 npy_loss 在train中 是使用  logs_write_dir 來存， 所以就要去 logs_write_dir 來讀囉！ 所以這邊 np.load 裡面適用 logs_write_dir 是沒問題的！
                    loss_array = loss_array[:epoch]   ### 這是為了防止 如果程式在 step3,4之間中斷 這種 loss已經存完 但 model還沒存 的狀況，loss 會比想像中的多一步，所以加這行防止這種情況發生喔
                    loss_array = np.append(loss_array, loss_value)
                    np.save(self.result_obj.logs_write_dir + "/" + loss_name, np.array(loss_array))
                    # print(loss_array)

        ###    reset tensorboard 的 loss紀錄容器
        for loss_containor in self.loss_info_obj.loss_containors.values():
            loss_containor.reset_states()
        ###############################################################
        self.loss_info_obj.see_loss_during_train(self.epochs)  ### 把 loss資訊 用 matplot畫出來
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
        with open(self.result_obj.result_write_dir + "/" + "cost_time.txt", "a") as f:
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
        self.exp_init(reload_result=True, reload_model=True)
        self.train_step1_see_current_img(self.start_epoch, training=self.exp_bn_see_arg, see_reset_init=True)  ### 有時候製作 fake_exp 的時候 ， 只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要reset一下
        print("test see finish")

    def test(self, flow_mask=True):  ### 精神不好先暫時用 flow_mask flag 來區別 跟 flow 做不同的動作
        """
        """
        self.exp_init(reload_result=True, reload_model=True)
        self.testing(self.start_epoch, flow_mask=flow_mask)  ### 有時候製作 fake_exp 的時候 ， 只會複製 ckpt, log, ... ，see 不會複製過來，所以會需要reset一下
        print("test finish")

    def board_rebuild(self):
        self.exp_init(reload_result=True, reload_model=False)
        self.loss_info_obj.use_npy_rebuild_justG_tensorboard_loss(self, dst_dir=self.result_obj.logs_write_dir)
        print("board_rebuild finish")
        print("")

    def run(self):
        self.machine_ip   = socket.gethostbyname(socket.gethostname())  ### 取得 本機 IP   給 train_step5_show_time 紀錄
        self.machine_user = getpass.getuser()                           ### 取得 本機 User 給 train_step5_show_time 紀錄
        if  (self.phase == "train"):          self.train()
        elif(self.phase == "train_reload"):   self.train_reload()
        elif(self.phase == "test_see"):       self.test_see()
        elif(self.phase == "board_rebuild"):  self.board_rebuild()
        elif(self.phase == "test"):           self.test()  ### 精神不好先暫時用 flow_mask flag 來區別 跟 flow 做不同的動作
        elif(self.phase == "train_indicate"): pass  ### 待完成Z
        elif(self.phase.lower() == "ok"): pass      ### 不做事情，只是個標記而以這樣子
        else: print("ㄘㄋㄇㄉ phase 打錯字了拉~~~")



class Exp_builder():
    def __init__(self, exp=None):
        if(exp is None):
            self.exp = Experiment()
        else: self.exp = exp

    def set_com(self, machine="127.35"): return self  ### 只是單純讓我自己能直接看到而已，懶得去翻 cost_time.txt

    def set_basic(self, phase, db_builder, model_builder, loss_info_builder, exp_dir=".", describe_mid=None, describe_end=None, result_name=None):
        self.exp.phase = phase
        self.exp.db_builder = db_builder
        self.exp.model_builder = model_builder
        self.exp.loss_info_builder = loss_info_builder
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
        if(epoch_down_step is None): self.exp.epoch_down_step = epochs // 2
        else: self.exp.epoch_down_step = epoch_down_step
        self.exp.epoch_stop = epoch_stop
        self.exp.start_epoch = 0
        self.exp.exp_bn_see_arg = exp_bn_see_arg
        return self

    def set_train_in_gt_use_range(self, in_use_range="0~1", gt_use_range="0~1"):
        self.exp.in_use_range = in_use_range
        self.exp.gt_use_range = gt_use_range
        return self

    ### 整理後發現好像沒用到就註解起來囉～
    # def set_train_args_reload(self, result_name):
    #     self.exp.phase = "train_reload"
    #     self.result_name = result_name
    #     return self

    def set_result_name(self, result_name):
        self.exp.result_name = result_name
        return self

    def build(self, result_name=None):
        '''
        這邊先建好 result_obj 的話就可以給 step11, step12 用喔，
        且 因為我有寫 在train時 會自動建新的 result，所以不用怕 用到這邊 給 step11, step12 建立的 default result_obj 囉！

        也補上 建好result 馬上設定 loss_info_obj 拉，這樣 step11, step12 也能用了！
        '''
        if(self.exp.result_name is not None):
            # print("1.result_name", result_name, ", self.exp.gt_use_range~~~~~~~~~~~~~~~~~~~~~~~~~", self.exp.gt_use_range)  ### 追蹤see的建立過程
            self.exp.result_obj    = Result_builder().set_by_result_name(self.exp.exp_dir + "/" + self.exp.result_name, self.exp.in_use_range, self.exp.gt_use_range).build()  ### 直接用 自己指定好的 result_name


            ### 寫兩行的話 比較好打註解，寫一行其實也可以下面有補充～～
            self.exp.loss_info_builder = self.exp.loss_info_builder.copy()                                      ### 要做copy的動作， 才不會每個 exp_builder 都用到相同的 loss_info_builder 導致 建出相同的 loss_info_obj
            self.exp.loss_info_builder = self.exp.loss_info_builder.set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir)  ### copy完後，新的 loss_info_builder 更新他的 logs_dir～ 因為有copy 所以 不會 loss_info_obj 都是相同的 logs_read/write_dir 的問題啦！
            ### 補充：如果寫一行的話：
            # self.exp.loss_info_builder = self.exp.loss_info_builder.set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir).copy()  ### 先後copy() 都沒差

            ### copy完後，新的 loss_info_builder 更新他的 logs_dir～ 因為有copy 所以 不會 loss_info_obj 都是相同的 logs_read/write_dir 的問題啦！
            # self.exp.loss_info_builder.set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir)  ### copy完後，新的 loss_info_builder 更新他的 logs_dir～ 因為有copy 所以 不會 loss_info_obj 都是相同的 logs_read/write_dir 的問題啦！
            # self.exp.loss_info_obj = self.exp.loss_info_builder.build()  ### 這裡就要build了喔！為了給 step12 用拉！

            # self.exp.loss_info_obj = Loss_info_builder(self.exp.loss_info_obj, in_obj_copy=True).set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir).build()  ### 上面定位出 logs_read/write_dir 後 更新 loss_info_obj， in_obj_copy 記得要設True，原因寫在 Loss_info_builde 裡面喔
            # print("self.exp.loss_info_obj.logs_read_dir", self.exp.loss_info_obj.logs_read_dir)
            # print("self.exp.loss_info_obj.logs_write_dir", self.exp.loss_info_obj.logs_write_dir)
            # print()  ### 追蹤see的建立過程
            print(f"Experiment_builder build finish, can use {self.exp.exp_dir}")
        return self.exp
##########################################################################################################################################
### 5_1_GD_Gmae136_epoch700
# os_book_1532_rect_mae1 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_1_1", describe_end="1532data_mae1_127.28").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_1_2", describe_end="1532data_mae1_127.35").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mae6 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_1_3", describe_end="1532data_mae1_127.51").set_train_args(epochs=700).set_result_name(result_name="")

##########################################################################################################################################
### 5_2_GD_vs_justG
# os_book_1532_rect_D_05  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_2_2", describe_end="1532data_D_0.5_128.245").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_D_025 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_2_3", describe_end="1532data_D_0.25_127.35").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_D_01  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect, describe_mid="5_2_4", describe_end="1532data_D_0.1_127.28").set_train_args(epochs=700).set_result_name(result_name="")

##########################################################################################################################################
### 5_3_just_G_136920 ### 目前mae部分還是需要手動調(20200626)
# os_book_1532_justG_mae1 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_1", describe_end="1532data_mae1_127.28").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_2", describe_end="1532data_mae3_127.51").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mae6 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_3", describe_end="1532data_mae6_128.246").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mae9 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG, describe_mid="5_3_4", describe_end="1532data_mae9_127.35").set_train_args(epochs=700).set_result_name(result_name="")

##########################################################################################################################################
### 5_4_just_G_a_bigger  ### 目前其他 smaller, smaller2 的高度 400, 300 都要手動去調喔 resize大小喔！
# os_book_1532_justG_mae3_big      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_big, justG, describe_mid="5_4_1", describe_end="1532data_mae3_big_127.35").set_train_args(epochs=700).set_result_name(result_name="type7b_h500_w332_real_os_book-20200615_030658-justG-1532data_mae3_big_127.35")
# os_book_1532_justG_mae3_smaller  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_big, justG, describe_mid="5_4_3", describe_end="1532data_mae3_big_127.35").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mae3_smaller2 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_big, justG, describe_mid="5_4_4", describe_end="1532data_mae3_big_127.35").set_train_args(epochs=700).set_result_name(result_name="")

##########################################################################################################################################
### 5_5_focus_GD_vs_G
# os_book_1532_rect_mae3_focus     = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect , describe_mid="5_5_2", describe_end="1532data_mae3_focus_127.35").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mae3_focus    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, justG, describe_mid="5_5_4", describe_end="1532data_mae3_focus_127.35").set_train_args(epochs=700).set_result_name(result_name="")


##########################################################################################################################################
### 5_6_a_400_page
# os_book_400_rect_mae3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_400data, rect, describe_mid="5_6_1", describe_end="400data_mae3_127.35").set_train_args(epochs=2681).set_result_name(result_name="")
# os_book_800_rect_mae3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_800data, rect, describe_mid="no time to train", describe_end="800data_mae3_127.35").set_train_args(epochs=1341).set_result_name(result_name="")
# os_book_400_justG_mae3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_400data, justG, describe_mid="5_6_2", describe_end="400data_justG_mae3_127.28").set_train_args(epochs=2681).set_result_name(result_name="")

##########################################################################################################################################
### 5_7_first_k7_vs_k3
# os_book_1532_rect_firstk3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_firstk3 , describe_mid="5_7_2", describe_end="127.246").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_firstk3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_firstk3, describe_mid="5_7_4", describe_end="128.246").set_train_args(epochs=700).set_result_name(result_name="")

##########################################################################################################################################
### 5_8a_GD_mrf
# os_book_1532_rect_mrf7          = Exp_builder().set_com("127.48" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf7         , describe_mid="5_8a_2", describe_end="mrf7")         .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf79         = Exp_builder().set_com("128.245").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf79        , describe_mid="5_8a_3", describe_end="mrf79")        .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_replace_mrf7  = Exp_builder().set_com("127.35" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_replace_mrf7 , describe_mid="5_8a_4", describe_end="replace_mrf7") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_replace_mrf79 = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_replace_mrf79, describe_mid="5_8a_5", describe_end="replace_mrf79").set_train_args(epochs=700).set_result_name(result_name="")

### 5_8b_G_mrf
########################################################### 08b2
# os_book_1532_justG_mrf7          = Exp_builder().set_com("128.245").set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf7         , describe_mid="5_8b_2" , describe_end="mrf7").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf7_k3       = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf7_k3      , describe_mid="5_8b_2b", describe_end="mrf7_k3" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf5_k3       = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf5_k3      , describe_mid="5_8b_2c", describe_end="mrf5_k3" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf3_k3       = Exp_builder().set_com("127.48" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf3_k3      , describe_mid="5_8b_2d", describe_end="mrf3_k3" ).set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08b3
# os_book_1532_justG_mrf79         = Exp_builder().set_com("128.245").set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf79        , describe_mid="5_8b_3" , describe_end="mrf79").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf79_k3      = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf79_k3     , describe_mid="5_8b_3b", describe_end="mrf79_k3").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf57_k3      = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf57_k3     , describe_mid="5_8b_3c", describe_end="mrf57_k3").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf35_k3      = Exp_builder().set_com("127.35" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf35_k3     , describe_mid="5_8b_3d", describe_end="mrf35_k3" ).set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08b4
# os_book_1532_justG_mrf_replace7  = Exp_builder().set_com("127.40" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace7 , describe_mid="5_8b_4" , describe_end="mrf_replace7").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace5  = Exp_builder().set_com("127.35" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace5 , describe_mid="5_8b_4b", describe_end="mrf_replace5").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace3  = Exp_builder().set_com("127.48" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace3 , describe_mid="5_8b_4c", describe_end="mrf_replace3").set_train_args(epochs=700).set_result_name(result_name="")
########################################################### 08b5
# os_book_1532_justG_mrf_replace79 = Exp_builder().set_com("127.55" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace79, describe_mid="5_8b_5" , describe_end="mrf_replace79").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace75 = Exp_builder().set_com("127.55" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace75, describe_mid="5_8b_5b", describe_end="mrf_replace75").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace35 = Exp_builder().set_com("127.28" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace35, describe_mid="5_8b_5c", describe_end="mrf_replace35").set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08c
# os_book_1532_justG_mrf135_k3      = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c1", describe_end="Gk3mrf135" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf357_k3      = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c2", describe_end="Gk3mrf357" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf3579_k3     = Exp_builder().set_com("127.28" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c3", describe_end="Gk3mrf3579").set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08d
# os_book_1532_rect_mrf35_Gk3_DnoC_k4   = Exp_builder().set_com("127.55" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf35_Gk3_DnoC_k4    , describe_mid="5_8d1", describe_end="Gmrf35"  ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf135_Gk3_DnoC_k4  = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf135_Gk3_DnoC_k4   , describe_mid="5_8d2", describe_end="Gmrf135" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf357_Gk3_DnoC_k4  = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf357_Gk3_DnoC_k4   , describe_mid="5_8d3", describe_end="Gmrf357" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf3579_Gk3_DnoC_k4 = Exp_builder().set_com("127.28" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf3579_Gk3_DnoC_k4  , describe_mid="5_8d4", describe_end="Gmrf3579").set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 09a
# os_book_1532_rect_Gk4_D_concat_k3    = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_concat_k3   , describe_mid="5_9a_2", describe_end="Gk4_D_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk4_D_no_concat_k4 = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_no_concat_k4, describe_mid="5_9a_3", describe_end="Gk4_D_no_concat_k4").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk4_D_no_concat_k3 = Exp_builder().set_com("127.28" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_no_concat_k3, describe_mid="5_9a_4", describe_end="Gk4_D_no_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 09b
# os_book_1532_rect_Gk3_D_concat_k4    = Exp_builder().set_com("no com").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_concat_k4   , describe_mid="5_9b_1", describe_end="Gk3_D_concat_k4") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_D_concat_k3    = Exp_builder().set_com("no com").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_concat_k3   , describe_mid="5_9b_2", describe_end="Gk3_D_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_D_no_concat_k4 = Exp_builder().set_com("127.55").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_no_concat_k4   , describe_mid="5_9b_3", describe_end="Gk3_D_no_concat_k4") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_D_no_concat_k3 = Exp_builder().set_com("127.48").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_no_concat_k3   , describe_mid="5_9b_4", describe_end="Gk3_D_no_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 10
### 5_10_GD_D_train1_G_train_135
### 舊版，如果要重train記得改資料庫喔(拿掉focus)！
# os_book_1532_rect_mae3_focus_G03D01 = Exp_builder().set_com("127.35").set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect, describe_mid="5_9_2", describe_end="1532data_mae3_focus_G03D01").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mae3_focus_G05D01 = Exp_builder().set_com("127.35").set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect, describe_mid="5_9_3", describe_end="1532data_mae3_focus_G05D01").set_train_args(epochs=700).set_result_name(result_name="")

# ### 新版
# os_book_1532_rect_Gk3_train3_Dk4_no_concat = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_train3_Dk4_no_concat, describe_mid="5_10_2", describe_end="Gk3_train3_Dk4_no_concat").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_train5_Dk4_no_concat = Exp_builder().set_com("no com" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_train5_Dk4_no_concat, describe_mid="5_10_3", describe_end="Gk3_train5_Dk4_no_concat").set_train_args(epochs=700).set_result_name(result_name="")
# ########################################################### 11
# os_book_1532_Gk3_no_res             = Exp_builder().set_com("127.51" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_fk3_no_res            , describe_mid="5_11_1", describe_end="justG_fk3_no_res").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_no_res_D_no_concat = Exp_builder().set_com("127.28" ).set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_fk3_no_res_D_no_concat, describe_mid="5_11_2", describe_end="rect_fk3_no_res_D_no_concat").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_no_res_mrf357      = Exp_builder().set_com("128.246").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_mrf357_no_res     , describe_mid="5_11_3", describe_end="Gk3_mrf357_no_res").set_train_args(epochs=700).set_result_name(result_name="")

# ########################################################### 12
# exp_dir12 = "5_12_resb_num"
# os_book_1532_Gk3_resb00 = Exp_builder().set_com("finish").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb00 , exp_dir=exp_dir12, describe_mid="5_12_1", describe_end="Gk3_resb00" ) .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb01 = Exp_builder().set_com("127.48").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb01 , exp_dir=exp_dir12, describe_mid="5_12_2", describe_end="Gk3_resb01" ) .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb03 = Exp_builder().set_com("127.35").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb03 , exp_dir=exp_dir12, describe_mid="5_12_3", describe_end="Gk3_resb03" ) .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb05 = Exp_builder().set_com("no com").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb05 , exp_dir=exp_dir12, describe_mid="5_12_4", describe_end="Gk3_resb05") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb07 = Exp_builder().set_com("no com").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb07 , exp_dir=exp_dir12, describe_mid="5_12_5", describe_end="Gk3_resb07") .set_train_args(epochs=700).set_result_name(result_name="")
# # os_book_1532_Gk3_resb09 ### 原本已經訓練過了，但為了確保沒train錯，還是建了resb_09來train看看囉
# os_book_1532_Gk3_resb09 = Exp_builder().set_com("finish") .set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb09 , exp_dir=exp_dir12, describe_mid="5_12_6",   describe_end="Gk3_resb09") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb11 = Exp_builder().set_com("127.55") .set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb11 , exp_dir=exp_dir12, describe_mid="5_12_7",   describe_end="Gk3_resb11") .set_train_args(epochs=700).set_result_name(result_name="")
# ### 13
# os_book_1532_Gk3_resb15 = Exp_builder().set_com("127.28") .set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb15 , exp_dir=exp_dir12, describe_mid="5_12_7_3", describe_end="Gk3_resb15") .set_train_args(epochs=700).set_result_name(result_name="")
# ### 17
# os_book_1532_Gk3_resb20 = Exp_builder().set_com("128.244").set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb20 , exp_dir=exp_dir12, describe_mid="5_12_8",   describe_end="Gk3_resb20") .set_train_args(epochs=700).set_result_name(result_name="")

# ########################################################### 12
# exp_dir13 = "5_13_coord_conv"
# os_book_1532_justGk3_coord_conv        = Exp_builder().set_com("127.35").set_basic("train", type7b_h500_w332_real_os_book_1532data, justGk3_coord_conv        , exp_dir=exp_dir13, describe_mid="5_13_1", describe_end="coord_conv") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justGk3_mrf357_coord_conv = Exp_builder().set_com("127.28").set_basic("train", type7b_h500_w332_real_os_book_1532data, justGk3_mrf357_coord_conv , exp_dir=exp_dir13, describe_mid="5_13_2", describe_end="mrf357_coord_conv") .set_train_args(epochs=700).set_result_name(result_name="")


### 測試subprocessing 有沒有用
# blender_os_book_flow_unet_epoch002 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch2, G_mae_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch002") .set_train_args(batch_size=30, epochs=1).set_result_name(result_name="")
# blender_os_book_flow_unet_epoch003 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch3, G_mae_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch003") .set_train_args(epochs=3).set_result_name(result_name="")
# blender_os_book_flow_unet_epoch004 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch4, G_mae_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch004") .set_train_args(epochs=4).set_result_name(result_name="")

### 測試 怎麼樣設定 multiprocess 才較快
# testest     = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_IN_ch64, G_mae_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1_4_e060", describe_end="testest"         ).set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-testest")      ### copy from ch64_in_epoch060
# testest_big = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_IN_ch64, G_mae_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1_4_e700", describe_end="testest_big"     ).set_train_args(epochs=700, exp_bn_see_arg=None).set_train_in_gt_use_range(in_use_range="0~1", gt_use_range="0~1").set_result_name(result_name="type8_blender_os_book-testest_big")  ### copy from ch64_in_epoch700


import sys
if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_a_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        # blender_os_book_flow_unet_new_shuf_ch008_fake.build().run()
        # rect_fk3_ch64_tfIN_resb_ok9_epoch500.build().run()
        # rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.build().run()
        # concat_A.build().run()
        # flow_rect_7_level_fk7.build().run()
        # flow_rect_7_level_fk3.build().run()
        # flow_rect_2_level_fk3.build().run()
        # unet_2l.build().run()
        # unet_7l.build().run()
        # unet_7l_skip_use_add.build().run()
        # unet_8l.build().run()
        # rect_2_level_fk3_ReLU.build().run()
        # ch64_in_epoch500_sigmoid.build().run()
        # in_th_mo_th_gt_th.build().run()
        # t3_in_01_mo_th_gt_th_mae.build().run()
        # unet_IN_7l_2to4noC.build().run()
        # unet_IN_7l_skip_use_cnn1_NO_relu.build().run()
        # unet_IN_7l_skip_use_cnn1_USEsigmoid.build().run()
        # test1.build().run()
        # unet_IN_7l_2to3noC_e100.build().run()
        # ch64_in_sk_cSE_e060_wrong.build().run()
        # ch64_in_sk_sSE_e060.build().run()
        # ch64_in_sk_scSE_e060_wrong.build().run()
        # ch64_in_cnnNoBias_epoch060.build().run()
        # in_new_ch004_ep060.build().run()
        # testest.build().run()
        # mask_ch032_tanh_mae_ep060.build().run()
        # mask_ch032_sigmoid_bce_ep060.build().run()
        # mask_have_bg_ch032_sigmoid_bce_ep060.build().run()
        # mask_have_bg_ch128_sigmoid_bce_ep060.build().run()
        mask_h_bg_ch128_sig_bce_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_a_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
