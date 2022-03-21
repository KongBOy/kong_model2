from step10_b1_exp_obj_load_and_train_and_test import Experiment
from step11_b_result_obj_builder import Result_builder

from step0_access_path import Change_name_used_Result_Read_Dirs
import os

class Exp_builder():
    def __init__(self, exp=None):
        if(exp is None):
            self.exp = Experiment()
        else: self.exp = exp

    def set_basic(self, phase, db_builder, model_builder, loss_info_builders, exp_dir=".", code_exe_path=".", describe_mid=None, describe_end=None, result_name=None):
        self.exp.phase = phase
        self.exp.db_builder = db_builder
        self.exp.model_builder = model_builder
        if(type(loss_info_builders) == type([])): self.exp.loss_info_builders = loss_info_builders
        else:                                     self.exp.loss_info_builders = [loss_info_builders]
        self.exp.exp_dir = exp_dir
        self.exp.code_exe_path = code_exe_path
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

    def set_train_in_gt_use_range(self, use_in_range, use_gt_range):
        self.exp.use_in_range = use_in_range
        self.exp.use_gt_range = use_gt_range
        return self

    ### 整理後發現好像沒用到就註解起來囉～
    # def set_train_args_reload(self, result_name):
    #     self.exp.phase = "train_reload"
    #     self.result_name = result_name
    #     return self

    def set_result_name(self, result_name):
        self.exp.result_name = result_name
        # print("0.set_result_name", result_name)  ### 追蹤see的建立過程
        return self

    def set_multi_model_reload_exp_builders_dict(self, **kwargs):
        self.exp.multi_model_by_history = True
        self.exp.multi_model_reload_exp_builders_dict = kwargs
        return self

    def build_exp_temporary(self):
        '''
        這邊先建好 result_obj 的話就可以給 step11, step12 用喔，
        且 因為我有寫 在train時 會自動建新的 result，所以不用怕 用到這邊 給 step11, step12 建立的 default result_obj 囉！

        也補上 建好result 馬上設定 loss_info_obj 拉，這樣 step11, step12 也能用了！
        '''
        if(self.exp.result_name is not None):
            ''' 建 虛擬的 result_obj ，這樣就不用 exp_init() 就能使用 resul_obj裡面的東西囉'''
            if("test" in self.exp.phase):
                if  ("test_see"   in self.exp.phase): self.exp.db_builder.reset_test_db_name("see")
                elif("test_train" in self.exp.phase): self.exp.db_builder.reset_test_db_name("train")
                else:                                 self.exp.db_builder.reset_test_db_name(self.exp.phase)

            self.db_obj = self.exp.db_builder.build()
            # print("1.result_name", self.exp.result_name, ", self.exp.use_gt_range~~~~~~~~~~~~~~~~~~~~~~~~~", self.exp.use_gt_range)  ### 追蹤see的建立過程
            self.exp.result_obj    = Result_builder().set_exp_obj_use_gt_range(self.exp.use_gt_range).set_by_result_name(self.exp.exp_dir + "/" + self.exp.result_name, self.db_obj).build()  ### 直接用 自己指定好的 result_name

            ''' 建 虛擬的 loss_info_obj ，這樣就不用 exp_init() 就能使用 loss_info_obj裡面的東西囉'''
            loss_info_objs = []
            for loss_info_builder in self.exp.loss_info_builders:
                loss_info_builder = loss_info_builder.copy()  ### copy完後，新的 loss_info_builders 更新他的 logs_dir～ 因為有copy 所以 不會 loss_info_obj 都是相同的 logs_read/write_dir 的問題啦！
                loss_info_builder.set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir)  ### 所以 loss_info_builders 要 根據 result資訊(logs_read/write_dir) 先更新一下
                loss_info_objs.append(loss_info_builder.build())  ### 上面 logs_read/write_dir 後 更新 就可以建出 loss_info_objs 囉！
            self.exp.loss_info_objs = loss_info_objs

            ### 補充：如果寫一行的話：
            # self.exp.loss_info_builders = self.exp.loss_info_builders.set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir).copy()  ### 先後copy() 都沒差

            # self.exp.loss_info_builders.set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir)  ### copy完後，新的 loss_info_builders 更新他的 logs_dir～ 因為有copy 所以 不會 loss_info_obj 都是相同的 logs_read/write_dir 的問題啦！
            # self.exp.loss_info_obj = self.exp.loss_info_builders.build()  ### 這裡就要build了喔！為了給 step12 用拉！

            # self.exp.loss_info_obj = Loss_info_builder(self.exp.loss_info_obj, in_obj_copy=True).set_logs_dir(self.exp.result_obj.logs_read_dir, self.exp.result_obj.logs_write_dir).build()  ### 上面定位出 logs_read/write_dir 後 更新 loss_info_obj， in_obj_copy 記得要設True，原因寫在 Loss_info_builde 裡面喔
            # print("self.exp.loss_info_obj.logs_read_dir", self.exp.loss_info_obj.logs_read_dir)
            # print("self.exp.loss_info_obj.logs_write_dir", self.exp.loss_info_obj.logs_write_dir)
            # print()  ### 追蹤see的建立過程


    def build(self):
        self.build_exp_temporary()
        print(f"Experiment_builder build finish, can use {self.exp.exp_dir}")
        return self.exp
    ###########################################################################################################################################################################################################
    ###########################################################################################################################################################################################################

    def _change_result_name_final_rename(self, result_name_ord, result_name_dst, run_change=False, print_msg=False):
        exp_dir = self.exp.exp_dir
        for Result_Read_Dir in Change_name_used_Result_Read_Dirs:  ### 我目前有四個存資料的地方， 6T, 4T, 2T, 400GB 的硬碟這樣子
            result_path_ord = Result_Read_Dir  + f"result/{exp_dir}/" + result_name_ord
            result_path_dst = Result_Read_Dir  + f"result/{exp_dir}/" + result_name_dst
            if(run_change):
                if(os.path.isdir(result_path_ord)):
                    os.rename(result_path_ord, result_path_dst)
                    if(print_msg):
                        print(f"{result_path_ord} rename to")
                        print(f"{result_path_dst} finish~~")
                else:
                    print(f"result_path_ord: {result_path_ord} 已經不存在，也許之前已經改過名字， 現在又重複改到了， 因此不執行改名")

        self.exp.result_name = result_name_dst  ### 有可能連續的 .change前面都不 run_change， 最後才run_change， 就需要更新目前的 result_name 才可以達到接續 change_name 的效果喔

    def _get_result_name_basic_v1(self):
        ''' v1： 0: db_name, 1: describe_mid, 2: timestamp,  3: model_name,   4: describe_end '''
        result_name_ord = self.exp.result_name
        result_name_components = result_name_ord.split("-")
        db_category  = result_name_components[0]
        describe_mid = result_name_components[1]
        timestamp    = result_name_components[2]
        model_name   = result_name_components[3]
        describe_end = result_name_components[4]
        return result_name_ord, db_category, describe_mid, timestamp, model_name, describe_end

    def _get_result_name_basic_v2(self):
        ''' v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp '''
        result_name_ord = self.exp.result_name
        result_name_components = result_name_ord.split("-")
        # print("result_name_components", result_name_components)
        db_category  = result_name_components[0]
        describe_mid = result_name_components[1]
        model_name    = result_name_components[2]
        describe_end   = result_name_components[3]
        timestamp = result_name_components[4]
        return result_name_ord, db_category, describe_mid, model_name, describe_end, timestamp

    ##############################################################################################################################
    def change_result_name_v1_to_v2(self, run_change=False, print_msg=False):
        self.build_exp_temporary()
        '''
        v1： 0: db_name, 1: describe_mid, 2: timestamp,  3: model_name,   4: describe_end
        v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp

        使用方法就是 在 step10a.py 裡面 直接 在 exp_builder 後面 .change_result_name_v1_to_v2() 之後 案 F5 就可以跑了喔

        根據 step11_b_result_obj_builder 的 set_by_exp 裡的 _get_result_name_by_exp 決定的喔
            舉例：6_mask_unet/5_5b_ch032_bce_s001_100_sobel_k5_s001_100/type8_blender_os_book-5b_bce_s001_sobel_k5_s001-20211104_150928-flow_unet-mask_h_bg_ch032_sig_L6_ep060
        '''
        result_name_ord, db_category, describe_mid, timestamp, model_name, describe_end = self._get_result_name_basic_v1()
        result_name_v1 = f"{db_category}-{describe_mid}-{timestamp}-{model_name}-{describe_end}"  ### v1： 0: db_name, 1: describe_mid, 2: timestamp,  3: model_name,   4: describe_end
        result_name_v2 = f"{db_category}-{describe_mid}-{model_name}-{describe_end}-{timestamp}"  ### v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp
        self._change_result_name_final_rename(result_name_v1, result_name_v2, run_change=run_change, print_msg=print_msg)
        return self

    ##############################################################################################################################
    def change_result_name_v2_Remove_os_book(self, run_change=False, print_msg=False):
        self.build_exp_temporary()
        ''' 使用方法就是 在 step10a.py 裡面 直接 在 exp_builder 後面 .change_result_name_v1_to_v2() 之後 案 F5 就可以跑了喔'''
        result_name_ord, db_category, describe_mid, model_name, describe_end, timestamp = self._get_result_name_basic_v2()
        db_category_prone = db_category.replace("type8_blender_os_book", "type8_blender")
        result_name_dst = f"{db_category_prone}-{describe_mid}-{model_name}-{describe_end}-{timestamp}"  ### v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp
        self._change_result_name_final_rename(result_name_ord, result_name_dst, run_change=run_change, print_msg=print_msg)
        return self

    def change_result_name_v2_Remove_describe_end_loss(self, run_change=False, print_msg=False):
        self.build_exp_temporary()
        ''' 使用方法就是 在 step10a.py 裡面 直接 在 exp_builder 後面 .change_result_name_v1_to_v2() 之後 案 F5 就可以跑了喔'''
        result_name_ord, db_category, describe_mid, model_name, describe_end, timestamp = self._get_result_name_basic_v2()

        describe_end_elements = describe_end.split("_")
        loss_describe = self.exp.loss_info_objs[0].loss_describe
        # print("loss_describe:", loss_describe)

        for loss_describe_element in loss_describe.split("_"):
            # print("loss_describe_element:", loss_describe_element)
            # print("describe_end_elements:", describe_end_elements)
            if(loss_describe_element in describe_end_elements): describe_end_elements.remove(loss_describe_element)
            # print("describe_end_elements:", describe_end_elements)
        describe_end_prone = "_".join(describe_end_elements)
        result_name_dst = f"{db_category}-{describe_mid}-{model_name}-{describe_end_prone}-{timestamp}"  ### v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp

        self._change_result_name_final_rename(result_name_ord, result_name_dst, run_change=run_change, print_msg=print_msg)
        return self

    def change_result_name_v2_Describe_end_use_Uniform_model_name(self, run_change=False, print_msg=False):
        ''' 使用方法就是 在 step10a.py 裡面 直接 在 exp_builder 後面 .change_result_name_v1_to_v2() 之後 案 F5 就可以跑了喔'''
        self.build_exp_temporary()
        result_name_ord, db_category, describe_mid, model_name, describe_end, timestamp = self._get_result_name_basic_v2()

        uniform_model_name = self.exp.model_builder.kong_model.model_describe
        describe_end = uniform_model_name
        print("uniform_model_name:", uniform_model_name)
        result_name_dst = f"{db_category}-{describe_mid}-{model_name}-{describe_end}-{timestamp}"  ### v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp

        self._change_result_name_final_rename(result_name_ord, result_name_dst, run_change=run_change, print_msg=print_msg)
        return self

    def change_result_name_v2_Describe_end_use_New_Describe_end(self, run_change=False, print_msg=False):
        ''' 使用方法就是 在 step10a.py 裡面 直接 在 exp_builder 後面 .change_result_name_v1_to_v2() 之後 案 F5 就可以跑了喔'''
        self.build_exp_temporary()
        result_name_ord, db_category, describe_mid, model_name, describe_end, timestamp = self._get_result_name_basic_v2()

        new_describe_end = self.exp.describe_end
        describe_end = new_describe_end
        print("new_describe_end:", new_describe_end)
        result_name_dst = f"{db_category}-{describe_mid}-{model_name}-{describe_end}-{timestamp}"  ### v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp

        self._change_result_name_final_rename(result_name_ord, result_name_dst, run_change=run_change, print_msg=print_msg)
        return self

    def change_result_name_v2_to_v3_Remove_describe_mid_model_name(self, run_change=False, print_msg=False):
        self.build_exp_temporary()
        result_name_components = self.exp.result_name.split("-")  ### v2： 0: db_name, 1: describe_mid, 2: model_name, 3: describe_end, 4: timestamp

        result_name_ord = "-".join(result_name_components)
        del result_name_components[1:3]
        result_name_dst = "-".join(result_name_components)        ### v3： 0: db_name, 1: describe_end, 2: timestamp
        self._change_result_name_final_rename(result_name_ord, result_name_dst, run_change=run_change, print_msg=print_msg)
        return self

    def change_result_name_v3_to_v4_Remove_db_name(self, run_change=False, print_msg=False):
        self.build_exp_temporary()
        result_name_components = self.exp.result_name.split("-")  ### v3： 0: db_name, 1: describe_end, 2: timestamp

        result_name_ord = "-".join(result_name_components)
        print("result_name_ord", result_name_ord)
        if("type8_blender" in result_name_components or "ch032" == result_name_components[0]): del result_name_components[0:1]
        result_name_dst = "-".join(result_name_components)        ### v4： 0: describe_end, 2: timestamp
        self._change_result_name_final_rename(result_name_ord, result_name_dst, run_change=run_change, print_msg=print_msg)
        return self

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
# os_book_1532_rect_mrf7          = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf7         , describe_mid="5_8a_2", describe_end="mrf7")         .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf79         = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf79        , describe_mid="5_8a_3", describe_end="mrf79")        .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_replace_mrf7  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_replace_mrf7 , describe_mid="5_8a_4", describe_end="replace_mrf7") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_replace_mrf79 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_replace_mrf79, describe_mid="5_8a_5", describe_end="replace_mrf79").set_train_args(epochs=700).set_result_name(result_name="")

### 5_8b_G_mrf
########################################################### 08b2
# os_book_1532_justG_mrf7          = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf7         , describe_mid="5_8b_2" , describe_end="mrf7").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf7_k3       = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf7_k3      , describe_mid="5_8b_2b", describe_end="mrf7_k3" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf5_k3       = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf5_k3      , describe_mid="5_8b_2c", describe_end="mrf5_k3" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf3_k3       = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf3_k3      , describe_mid="5_8b_2d", describe_end="mrf3_k3" ).set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08b3
# os_book_1532_justG_mrf79         = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf79        , describe_mid="5_8b_3" , describe_end="mrf79").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf79_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf79_k3     , describe_mid="5_8b_3b", describe_end="mrf79_k3").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf57_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf57_k3     , describe_mid="5_8b_3c", describe_end="mrf57_k3").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf35_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf35_k3     , describe_mid="5_8b_3d", describe_end="mrf35_k3" ).set_train_args(epochs=700).set_result_name(result_name="")

##################################################
# os_book_1532_justG_mrf_replace7  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace7 , describe_mid="5_8b_4" , describe_end="mrf_replace7").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace5  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace5 , describe_mid="5_8b_4b", describe_end="mrf_replace5").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace3  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace3 , describe_mid="5_8b_4c", describe_end="mrf_replace3").set_train_args(epochs=700).set_result_name(result_name="")
########################################################### 08b5
# os_book_1532_justG_mrf_replace79 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace79, describe_mid="5_8b_5" , describe_end="mrf_replace79").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace75 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace75, describe_mid="5_8b_5b", describe_end="mrf_replace75").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf_replace35 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf_replace35, describe_mid="5_8b_5c", describe_end="mrf_replace35").set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08c
# os_book_1532_justG_mrf135_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c1", describe_end="Gk3mrf135" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf357_k3      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c2", describe_end="Gk3mrf357" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justG_mrf3579_k3     = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_mrf357_k3     , describe_mid="5_8c3", describe_end="Gk3mrf3579").set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 08d
# os_book_1532_rect_mrf35_Gk3_DnoC_k4   = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf35_Gk3_DnoC_k4    , describe_mid="5_8d1", describe_end="Gmrf35"  ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf135_Gk3_DnoC_k4  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf135_Gk3_DnoC_k4   , describe_mid="5_8d2", describe_end="Gmrf135" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf357_Gk3_DnoC_k4  = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf357_Gk3_DnoC_k4   , describe_mid="5_8d3", describe_end="Gmrf357" ).set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mrf3579_Gk3_DnoC_k4 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_mrf3579_Gk3_DnoC_k4  , describe_mid="5_8d4", describe_end="Gmrf3579").set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 09a
# os_book_1532_rect_Gk4_D_concat_k3    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_concat_k3   , describe_mid="5_9a_2", describe_end="Gk4_D_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk4_D_no_concat_k4 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_no_concat_k4, describe_mid="5_9a_3", describe_end="Gk4_D_no_concat_k4").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk4_D_no_concat_k3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk4_D_no_concat_k3, describe_mid="5_9a_4", describe_end="Gk4_D_no_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 09b
# os_book_1532_rect_Gk3_D_concat_k4    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_concat_k4   , describe_mid="5_9b_1", describe_end="Gk3_D_concat_k4") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_D_concat_k3    = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_concat_k3   , describe_mid="5_9b_2", describe_end="Gk3_D_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_D_no_concat_k4 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_no_concat_k4   , describe_mid="5_9b_3", describe_end="Gk3_D_no_concat_k4") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_D_no_concat_k3 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_D_no_concat_k3   , describe_mid="5_9b_4", describe_end="Gk3_D_no_concat_k3") .set_train_args(epochs=700).set_result_name(result_name="")

########################################################### 10
### 5_10_GD_D_train1_G_train_135
### 舊版，如果要重train記得改資料庫喔(拿掉focus)！
# os_book_1532_rect_mae3_focus_G03D01 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect, describe_mid="5_9_2", describe_end="1532data_mae3_focus_G03D01").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_mae3_focus_G05D01 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data_focus, rect, describe_mid="5_9_3", describe_end="1532data_mae3_focus_G05D01").set_train_args(epochs=700).set_result_name(result_name="")

# ### 新版
# os_book_1532_rect_Gk3_train3_Dk4_no_concat = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_train3_Dk4_no_concat, describe_mid="5_10_2", describe_end="Gk3_train3_Dk4_no_concat").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_rect_Gk3_train5_Dk4_no_concat = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_Gk3_train5_Dk4_no_concat, describe_mid="5_10_3", describe_end="Gk3_train5_Dk4_no_concat").set_train_args(epochs=700).set_result_name(result_name="")
# ########################################################### 11
# os_book_1532_Gk3_no_res             = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justG_fk3_no_res            , describe_mid="5_11_1", describe_end="justG_fk3_no_res").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_no_res_D_no_concat = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, rect_fk3_no_res_D_no_concat, describe_mid="5_11_2", describe_end="rect_fk3_no_res_D_no_concat").set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_no_res_mrf357      = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_mrf357_no_res     , describe_mid="5_11_3", describe_end="Gk3_mrf357_no_res").set_train_args(epochs=700).set_result_name(result_name="")

# ########################################################### 12
# exp_dir12 = "5_12_resb_num"
# os_book_1532_Gk3_resb00 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb00 , exp_dir=exp_dir12, describe_mid="5_12_1", describe_end="Gk3_resb00" ) .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb01 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb01 , exp_dir=exp_dir12, describe_mid="5_12_2", describe_end="Gk3_resb01" ) .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb03 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb03 , exp_dir=exp_dir12, describe_mid="5_12_3", describe_end="Gk3_resb03" ) .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb05 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb05 , exp_dir=exp_dir12, describe_mid="5_12_4", describe_end="Gk3_resb05") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb07 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb07 , exp_dir=exp_dir12, describe_mid="5_12_5", describe_end="Gk3_resb07") .set_train_args(epochs=700).set_result_name(result_name="")
# # os_book_1532_Gk3_resb09 ### 原本已經訓練過了，但為了確保沒train錯，還是建了resb_09來train看看囉
# os_book_1532_Gk3_resb09 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb09 , exp_dir=exp_dir12, describe_mid="5_12_6",   describe_end="Gk3_resb09") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_Gk3_resb11 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb11 , exp_dir=exp_dir12, describe_mid="5_12_7",   describe_end="Gk3_resb11") .set_train_args(epochs=700).set_result_name(result_name="")
# ### 13
# os_book_1532_Gk3_resb15 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb15 , exp_dir=exp_dir12, describe_mid="5_12_7_3", describe_end="Gk3_resb15") .set_train_args(epochs=700).set_result_name(result_name="")
# ### 17
# os_book_1532_Gk3_resb20 = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, Gk3_resb20 , exp_dir=exp_dir12, describe_mid="5_12_8",   describe_end="Gk3_resb20") .set_train_args(epochs=700).set_result_name(result_name="")

# ########################################################### 12
# exp_dir13 = "5_13_coord_conv"
# os_book_1532_justGk3_coord_conv        = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justGk3_coord_conv        , exp_dir=exp_dir13, describe_mid="5_13_1", describe_end="coord_conv") .set_train_args(epochs=700).set_result_name(result_name="")
# os_book_1532_justGk3_mrf357_coord_conv = Exp_builder().set_basic("train", type7b_h500_w332_real_os_book_1532data, justGk3_mrf357_coord_conv , exp_dir=exp_dir13, describe_mid="5_13_2", describe_end="mrf357_coord_conv") .set_train_args(epochs=700).set_result_name(result_name="")


### 測試subprocessing 有沒有用
# blender_os_book_flow_unet_epoch002 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch2, G_mae_s001_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch002") .set_train_args(batch_size=30, epochs=1).set_result_name(result_name="")
# blender_os_book_flow_unet_epoch003 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch3, G_mae_s001_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch003") .set_train_args(epochs=3).set_result_name(result_name="")
# blender_os_book_flow_unet_epoch004 = Exp_builder().set_basic("train", type8_blender_os_book_768, flow_unet_epoch4, G_mae_s001_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1", describe_end="epoch004") .set_train_args(epochs=4).set_result_name(result_name="")

### 測試 怎麼樣設定 multiprocess 才較快
# testest     = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_IN_ch64, G_mae_s001_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1_4_e060", describe_end="testest"         ).set_train_args(epochs= 60, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-testest")      ### copy from ch64_in_epoch060
# testest_big = Exp_builder().set_basic("test_see", type8_blender_os_book_768, flow_unet_IN_ch64, G_mae_s001_loss_info_builder, exp_dir=exp_dir14, describe_mid="5_14_1_4_e700", describe_end="testest_big"     ).set_train_args(epochs=700, exp_bn_see_arg=None).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="type8_blender_os_book-testest_big")  ### copy from ch64_in_epoch700


import sys
if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        # blender_os_book_flow_unet_new_shuf_ch008_fake.build().run()
        # rect_fk3_ch64_tfIN_resb_ok9_epoch500.build().run()
        # rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.build().run()
        # concat_A.build().run()
        # flow_rect_7_level_fk7.build().run()
        # flow_rect_7_level_fk3.build().run()
        # flow_rect_2_level_fk3.build().run()
        # unet_2l.build().run()
        # unet_L7.build().run()
        # unet_L7_skip_use_add.build().run()
        # unet_8l.build().run()
        # rect_2_level_fk3_ReLU.build().run()
        # ch64_in_epoch500_sigmoid.build().run()
        # in_th_mo_th_gt_th.build().run()
        # t3_in_01_mo_th_gt_th_mae.build().run()
        # unet_IN_L7_2to4noC.build().run()
        # unet_IN_L7_skip_use_cnn1_NO_relu.build().run()
        # unet_IN_L7_skip_use_cnn1_USEsigmoid.build().run()
        # test1.build().run()
        # unet_IN_L7_2to3noC_e100.build().run()
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

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
