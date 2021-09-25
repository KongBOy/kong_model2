from step0_access_path import JPG_QUALITY, CORE_AMOUNT_NPY_TO_NPZ, CORE_AMOUNT_BM_REC_VISUAL, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name, move_dir_certain_file, method2
from matplot_fig_ax_util import Matplot_single_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from flow_bm_util import use_flow_to_get_bm, use_bm_to_rec_img
from video_from_img import Video_combine_from_dir
from multiprocess_util import multi_processing_interface
from multiprocessing import Process, Manager

import cv2
import time
import numpy as np
from tqdm import tqdm
import os

sys.path.append("SIFT_dev/SIFTflow")
import matplotlib.pyplot as plt  ### debug用
from   matplotlib.gridspec import GridSpec
import datetime
# import pdb

'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual
'''

class See_info:
    '''
    See 是 最直接、最基本 model output的東西，在各個model裡面都應該有寫 自己的 generate_see
    而這邊只是 讀取 training 過程中生成的 See 這樣子囉~~
    '''
    def __init__(self, result_read_dir, result_write_dir, see_name):
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.see_name = see_name

        self.result_read_dir  = result_read_dir
        self.result_write_dir = result_write_dir

        self.see_read_dir  = self.result_read_dir  + "/" + self.see_name
        self.see_write_dir = self.result_write_dir + "/" + self.see_name

        self.in_use_range = "0~1"
        self.gt_use_range = "0~1"

    def get_see_base_info(self):
        ''' 我有把 see_file_amount 這個attr拿掉囉！ 因為 應用的時候需要一直 -1, -2, -3 很煩， -1, -2, -3 分別代表什麼數字都直接定義清楚這樣子拉！'''
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.see_jpg_names            = get_dir_certain_file_name(self.see_read_dir, certain_word=".jpg")
        self.see_in_img_path          = self.see_read_dir + "/" + self.see_jpg_names[0]
        self.see_gt_flow_v_path       = self.see_read_dir + "/" + self.see_jpg_names[1]
        self.see_rec_hope_path        = self.see_read_dir + "/" + self.see_jpg_names[2]

        ### 因為 See_flow_visual 和 See_bm_rec 要用到， 所以從 See_flow_visual 提升上來囉！
        self.see_flow_epoch_jpg_names      = get_dir_certain_file_name(self.see_read_dir, certain_word="epoch", certain_ext=".jpg")
        self.see_flow_epoch_jpg_read_paths = [self.see_read_dir + "/" + epoch_jpg_name for epoch_jpg_name in self.see_flow_epoch_jpg_names]  ### 沒有 write_paths，因為這是 predict_flow_visual， 是從model 訓練過程產生的， 後處理不會產生！ 就是不會做ewrite的動作囉！就不用write_path拉！
        self.see_flow_epoch_jpg_amount     = len(self.see_flow_epoch_jpg_names)

        ### 不確定合不合理， 目前就先暫時用 see_flow_epoch_jpg_names -1(去掉epoch0) 來代表 現在已經train了幾個epochs囉！ 即 trained_epochs， see_flow_epoch_jpg_names 用了覺得不合理再換吧～
        self.trained_epoch       = self.see_flow_epoch_jpg_amount - 1  ### 去掉epoch0
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
class See_npy_to_npz(See_info):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_npy_to_npz, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.old1_see_npz_read_dir   = self.see_read_dir
        self.old1_see_npz_write_dir  = self.see_write_dir
        self.old2_see_npz_read_dir   = self.see_read_dir  + "/npz"
        self.old2_see_npz_write_dir  = self.see_write_dir + "/npz"

        ### 給 npy轉npz 用的
        self.see_npy_read_dir      = self.see_read_dir
        self.see_npy_write_dir     = self.see_write_dir

        ### 給 bm_rec 用的
        self.see_npz_read_dir       = self.see_read_dir  + "/1_npz"
        self.see_npz_write_dir      = self.see_write_dir + "/1_npz"

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_npz_dir(print_msg=True)

    def Change_npz_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_npz_dir, Current See:{self.see_name}")
        # move_dir_certain_file(self.old1_see_npz_read_dir,  certain_word="epoch", certain_ext=".npz", dst_dir=self.see_npz_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old1_see_npz_write_dir, certain_word="epoch", certain_ext=".npz", dst_dir=self.see_npz_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old1_see_npz_read_dir,  certain_word="0b-gt", certain_ext=".npz", dst_dir=self.see_npz_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old1_see_npz_write_dir, certain_word="0b-gt", certain_ext=".npz", dst_dir=self.see_npz_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_see_npz_read_dir,  certain_word="epoch", certain_ext=".npz", dst_dir=self.see_npz_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old2_see_npz_write_dir, certain_word="epoch", certain_ext=".npz", dst_dir=self.see_npz_write_dir, print_msg=print_msg)

    ### 給自己 npy轉npz 用的
    def get_npy_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.see_npy_names            = get_dir_certain_file_name(self.see_npy_read_dir, certain_word=".npy")
        self.see_npy_read_paths       = [self.see_npy_read_dir  + "/" + npy_name for npy_name in self.see_npy_names]  ### 沒有 write_paths，因為式 npy轉npz， 不會有寫npy的動作， 雖然下面的 compare 會寫一點npy， 但也因為 有用 .replace() 所以用 see_npy_name.replace() 較保險這樣子！
        self.see_npy_amount           = len(self.see_npy_read_paths)

    ### 給下一步 bm_rec 用的
    def get_npz_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        ### 有包含 gt_flow 的 list喔！ 第一個 放的是 gt_flow
        print("self.see_npz_read_dir~~~~~~~~~~~~~~~~~", self.see_npz_read_dir)
        self.see_npz_names            = get_dir_certain_file_name(self.see_npz_read_dir, certain_word=".npz")
        self.flow_gt_npz_path         = self.see_npz_read_dir + "/" + self.see_npz_names[0]

        ### 不包含 gt_flow 的 list喔！
        self.see_npz_epoch_names      = get_dir_certain_file_name(self.see_npz_read_dir, certain_word="epoch", certain_ext=".npz")
        self.see_npz_epoch_read_paths = [self.see_npz_read_dir + "/" + epoch_npz_name for epoch_npz_name in self.see_npz_epoch_names]    ### 沒有 write_paths，同上 ，既然已經沒有 self.see_npz_write_paths， 當然更不會有 self.see_npz_epoch_write_paths 拉！

        self.see_npz_epoch_amount     = len(self.see_npz_epoch_read_paths)

    def Npy_to_npz(self, single_see_core_amount=8, see_print_msg=False):   ### 因為有刪東西的動作，覺得不要multiprocess比較安全~~
        """
        把 See 資料夾內的.npy改存成.npz，存完會把.npy刪除喔～
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Npy_to_npz, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立資料夾：不能 build_new_dir，因為原本的 .npy 因為容量太大 ， 轉完 .npz， 最後會把.npy刪掉！ 因此如果 第二次以上執行 就不會有.npy了 無法重建.npz！ 所以不能把 .npz資料夾刪掉重建喔！
        Check_dir_exist_and_build(self.see_npz_write_dir)

        ### See_method 第二部分：取得see資訊
        self.get_npy_info()


        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(self.see_npy_amount > 0):
            if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
                self._npy_to_npz(start_index=0, amount=self.see_npy_amount)
            elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
                multi_processing_interface(core_amount=CORE_AMOUNT_NPY_TO_NPZ, task_amount=self.see_npy_amount, task=self._npy_to_npz, print_msg=see_print_msg)  ### 因為和 bm_rec 的動作包一起， 外面指定的 single_see_core_amount 是比較適合 bm_rec 的， 所以 npy_to_npz 就用 在 step0 統一指定的 CORE數囉！
            else:
                print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理～沒事情就空白拉

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.see_write_dir != self.see_read_dir):  ### 因為接下去的任務需要 此任務的結果， 如果 read/write 資料夾位置不一樣， write完的結果 copy 一份 放回read， 才能讓接下去的動作 有 東西 read 喔！
            Syn_write_to_read_dir(write_dir=self.see_npz_write_dir, read_dir=self.see_npz_read_dir, build_new_dir=False)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Npy_to_npz, Current See:{self.see_name}, cost time:{time.time() - start_time}")

    ### See_method 第三部分：主要做的事情在這裡
    def _npy_to_npz(self, start_index, amount):
        for see_npy_name in tqdm(self.see_npy_names[start_index:start_index + amount]):  ### 因為有用 .replace()， 對see_npy_name.replace() 較保險， 所以這邊用 see_npy_name 而不用 see_npy_path！
            npy = np.load(self.see_read_dir + "/" + see_npy_name)
            np.savez_compressed(self.see_npz_write_dir + "/" + see_npy_name.replace(".npy", ".npz"), npy)
            os.remove(self.see_read_dir + "/" + see_npy_name)
            # print(self.see_read_dir + "/" + see_npy_name, "delete ok")
            # npz = np.load(self.see_read_dir + "/" + see_npy_name.replace(".npy", ".npz"))  ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
            # print((npy - npz["arr_0"]).sum())                                              ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
        ### 不要想 邊生圖 邊 npy轉npz了，原因寫在 _Draw_bm_rec_matplot_visual 上面

    def npy_to_npz_comapre(self):
        self.get_npy_info()

        ### load_3_load_50_npy
        start_time = time.time()
        for go_name, see_npy_path in enumerate(self.see_npy_read_paths):
            np.load(see_npy_path)   ### 344 MB
        load_3_time = time.time() - start_time
        print("load_3_load_50_npy ok")

        ### save_3_save_50_npy
        npys = []
        for go_name, see_npy_path in enumerate(self.see_npy_read_paths):
            npys.append(np.load(see_npy_path))   ### 344 MB
        start_time = time.time()
        for go_name, npy in enumerate(npys):
            np.save(self.see_npy_paths[go_name], npy)
        save_3_time = time.time() - start_time
        print("save_3_save_50_npy ok")

        ### save_2_save_50_npz
        start_time = time.time()
        for go_name, npy in enumerate(npys):
            np.savez_compressed(self.see_write_dir + "/" + self.see_npy_names[go_name].replace(".npy", ""), npy)
        save_2_time = time.time() - start_time
        print("save_2_save_50_npz ok")

        ### load_2_load_50_npz
        start_time = time.time()
        for go_name, see_npy_name in enumerate(self.see_npy_names):
            np.load(self.see_read_dir + "/" + see_npy_name.replace(".npy", ".npz"))   ### 344 MB
        load_2_time = time.time() - start_time
        print("load_2_load_50_npz ok")

        ### save_1_save_1_npz_conatin_50npy
        start_time = time.time()
        np.savez_compressed(self.see_write_dir + "/" + "000_try_npz", np.array(npys))
        save_1_time = time.time() - start_time
        print("save_1_save_1_npz_conatin_50npy ok")

        ### load_1_load_1_npz_conatin_50npy
        start_time = time.time()
        big_npz = np.load(self.see_read_dir + "/" + "000_try_npz.npz")
        print(big_npz["arr_0"].shape)   ### 小心！要有使用他，才會真的去load資料喔！
        load_1_time = time.time() - start_time
        print("load_1_load_1_npz_conatin_50npy ok")
        print("")
        os.remove(self.see_read_dir + "/" + "000_try_npz.npz")  ### 只是用來看讀取寫入速度而已，沒有真的要用，所以測試完後記得要刪掉喔！


        print("save_1_save_1_npz_conatin_50npy:", save_1_time)
        print("save_2_save_50_npz:", save_2_time)
        print("save_3_save_50_npy:", save_3_time)
        print("load_1_load_1_npz_conatin_50npy:", load_1_time)
        print("load_2_load_50_npz:", load_2_time)
        print("load_3_load_50_npy:", load_3_time)
        """
        save_1_save_1_npz_conatin_50npy: 7.14643120765686
        save_2_save_50_npz: 7.625092267990112
        save_3_save_50_npy: 0.1186835765838623 ---------------->用這個
        load_1_load_1_npz_conatin_50npy: 1.0990545749664307
        load_2_load_50_npz: 0.2214348316192627 ---------------->用這個
        load_3_load_50_npy: 0.31119656562805176
        """
        print("finish")
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
class See_flow_visual(See_info):
    """
    See_flow_visual 是用來視覺化 See 的物件，因此這個Class我覺得也應該要設計成 training 中可以被使用的這樣子囉
      所以要看的東西就是簡單的：
        單純的input, 單純的output, 單純的gt
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_flow_visual, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.old_matplot_visual_read_dir   = self.see_read_dir  + "/matplot_visual"
        self.old_matplot_visual_write_dir  = self.see_write_dir + "/matplot_visual"

        self.matplot_visual_read_dir  = self.see_read_dir  + "/1_flow_matplot_visual"
        self.matplot_visual_write_dir = self.see_write_dir + "/1_flow_matplot_visual"

        self.single_row_imgs_during_train = None  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_flow_matplot_dir(print_msg=True)

    def Change_flow_matplot_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_flow_matplot_dir, Current See:{self.see_name}")
        # move_dir_certain_file(self.old_matplot_visual_read_dir,  certain_word="flow_epoch", certain_ext=".jpg", dst_dir=self.matplot_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_visual_write_dir, certain_word="flow_epoch", certain_ext=".jpg", dst_dir=self.matplot_visual_write_dir, print_msg=print_msg)

    ### 因為  See_bm_rec 要用到， 所以從 See_flow_visual 提升上去 See_info囉！
    # def get_flow_info(self):
        # self.see_flow_epoch_jpg_names      = get_dir_certain_file_name(self.see_read_dir, certain_word="epoch", certain_ext=".jpg")
        # self.see_flow_epoch_jpg_read_paths = [self.see_read_dir + "/" + epoch_jpg_name for epoch_jpg_name in self.see_flow_epoch_jpg_names]  ### 沒有 write_paths， 同上
        # self.see_flow_epoch_jpg_amount     = len(self.see_flow_epoch_jpg_names)

    ###############################################################################################
    ###############################################################################################
    ### 主要做的事情，此fun會給 save_as_matplot_visual_during/after train 使用
    def _Draw_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img = cv2.imread(self.see_in_img_path)            ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_gt_flow_v_path )            ### 要記得see的第二張存的是 輸出的gt影像
        img    = cv2.imread(self.see_flow_epoch_jpg_read_paths[epoch])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img ,   img ,      gt_img],    ### 把要顯示的每張圖包成list
                                img_titles=["in_img", "out_img", "gt_img"],    ### 把每張圖要顯示的字包成list
                                fig_title ="flow_epoch=%04i" % epoch,   ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)
        single_row_imgs.Draw_img()
        return single_row_imgs

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def save_as_matplot_visual_during_train(self, epoch, show_msg=False, bgr2rgb=False):  ### 訓練中，一張張 生成matplot_visual(這裡不能後處理，因為後處理需要全局的see_file，這裡都單張單張的會出問題)
        Check_dir_exist_and_build(self.matplot_visual_write_dir)
        start_time = time.time()
        # if(epoch==0):
        #     Check_dir_exist_and_build_new_dir(self.matplot_visual_write_dir)      ### 建立 存結果的資料夾
        self.get_see_base_info()  ### 每次執行都要 update喔！ 取得result內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.single_row_imgs_during_train = self._Draw_matplot_visual(epoch, add_loss=True, bgr2rgb=bgr2rgb)  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！
        if(show_msg): print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing save_as_matplot_visual_during_train, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

    ###############################################################################################
    def draw_loss_at_see_during_train(self, epoch, epochs):
        Check_dir_exist_and_build(self.matplot_visual_write_dir)  ### 以防matplot_visual資料夾被刪掉，要生圖找不到資料夾
        self.single_row_imgs_during_train.Draw_ax_loss_during_train(self.single_row_imgs_during_train.ax[-1, 1], self.see_read_dir + "/../logs", epoch, epochs)
        self.single_row_imgs_during_train.Save_fig(dst_dir=self.matplot_visual_write_dir, epoch=epoch)

    ###############################################################################################
    ###############################################################################################
    def Save_as_matplot_visual(self, add_loss=False, bgr2rgb=False, single_see_core_amount=8, see_print_msg=False):
        """
        Save_as_matplot_visual(_after_train) 最後想試試看 省掉他 會不會影響我的理解
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_matplot_visual, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.matplot_visual_write_dir)      ### 建立 存結果的資料夾

        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_matplot_visual_after_train(0, self.see_flow_epoch_jpg_amount, add_loss=add_loss, bgr2rgb=bgr2rgb)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.matplot_visual_write_dir, self.matplot_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.matplot_visual_write_dir, self.matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            ### see內的任務 有切 multiprocess
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_flow_epoch_jpg_amount, task=self._draw_matplot_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=see_print_msg)
            ### 後處理讓結果更小 但 又不失視覺品質，多核心版(core_amount 在 step0 裡調)
            Find_ltrd_and_crop (self.matplot_visual_write_dir, self.matplot_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.matplot_visual_write_dir, self.matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        Video_combine_from_dir(self.matplot_visual_write_dir, self.matplot_visual_write_dir)          ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.matplot_visual_write_dir != self.matplot_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.matplot_visual_write_dir, read_dir=self.matplot_visual_read_dir, build_new_dir=True)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Save_as_matplot_visual, Current See:{self.see_name}, cost_time:{time.time() - start_time}")
        print("")

    ### See_method 第三部分：主要做的事情在這裡
    ### 訓練後，可以走訪所有see_file 並重新產生 matplot_visual
    def _draw_matplot_visual_after_train(self, start_epoch, epoch_amount, add_loss=False, bgr2rgb=False):
        """
        有可能畫完主圖 還要再畫 loss，所以多這個method，多做的事情都在這裡處理
        處理完後就 Save_fig 囉！
        """
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            single_row_imgs = self._Draw_matplot_visual(go_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)
            if(add_loss)   : single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.ax[-1, 1], self.see_read_dir + "/../logs", go_epoch, min_epochs=self.see_flow_epoch_jpg_amount, ylim=0.04)
            single_row_imgs.Save_fig(dst_dir=self.matplot_visual_write_dir, epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
class See_bm_rec(See_npy_to_npz):
    """
    See_bm_rec 是用來 把 模型生成的 See 裡面的 flow，去生成 bm, rec，順便也視覺化出來這樣子囉
       所以裡面會有：
          bm_visual資料夾
          rec_visual資料夾
          順便視覺化一下(input, output, output_gt, rec, rec_gt)，bm視覺化好像有點問題先跳過ˊ口ˋ
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_bm_rec, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.old_matplot_bm_rec_visual_read_dir   = self.see_read_dir  + "/matplot_bm_rec_visual"
        self.old_matplot_bm_rec_visual_write_dir  = self.see_write_dir + "/matplot_bm_rec_visual"
        self.old_bm_visual_read_dir               = self.see_read_dir  + "/matplot_bm_rec_visual/bm_visual"
        self.old_bm_visual_write_dir              = self.see_write_dir + "/matplot_bm_rec_visual/bm_visual"
        self.old_rec_visual_read_dir              = self.see_read_dir  + "/matplot_bm_rec_visual/rec_visual"
        self.old_rec_visual_write_dir             = self.see_write_dir + "/matplot_bm_rec_visual/rec_visual"

        self.bm_rec_matplot_visual_read_dir   = self.see_read_dir  + "/2_bm_rec_matplot_visual"
        self.bm_rec_matplot_visual_write_dir  = self.see_write_dir + "/2_bm_rec_matplot_visual"
        self.bm_visual_read_dir               = self.see_read_dir  + "/2_bm_rec_matplot_visual/bm_visual"
        self.bm_visual_write_dir              = self.see_write_dir + "/2_bm_rec_matplot_visual/bm_visual"
        self.rec_visual_read_dir              = self.see_read_dir  + "/2_bm_rec_matplot_visual/rec_visual"
        self.rec_visual_write_dir             = self.see_write_dir + "/2_bm_rec_matplot_visual/rec_visual"

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_bm_rec_dir(print_msg=True)


    def Change_bm_rec_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_bm_rec_dir, Current See:{self.see_name}")
        # move_dir_certain_file(self.old_matplot_bm_rec_visual_read_dir,  certain_word="epoch", certain_ext=".jpg", dst_dir=self.bm_rec_matplot_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_bm_rec_visual_write_dir, certain_word="epoch", certain_ext=".jpg", dst_dir=self.bm_rec_matplot_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_bm_rec_visual_read_dir,  certain_word="combine", certain_ext=".avi", dst_dir=self.bm_rec_matplot_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_bm_rec_visual_write_dir, certain_word="combine", certain_ext=".avi", dst_dir=self.bm_rec_matplot_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_bm_visual_read_dir,   certain_word="bm_epoch",  certain_ext=".jpg", dst_dir=self.bm_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_bm_visual_write_dir,  certain_word="bm_epoch",  certain_ext=".jpg", dst_dir=self.bm_visual_write_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_bm_visual_read_dir,   certain_word="combine",   certain_ext=".avi", dst_dir=self.bm_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_bm_visual_write_dir,  certain_word="combine",   certain_ext=".avi", dst_dir=self.bm_visual_write_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_bm_visual_read_dir,   certain_word="bm_gt",     certain_ext=".jpg", dst_dir=self.bm_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_bm_visual_write_dir,  certain_word="bm_gt",     certain_ext=".jpg", dst_dir=self.bm_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_rec_visual_read_dir,  certain_word="rec_epoch", certain_ext=".jpg", dst_dir=self.rec_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_rec_visual_write_dir, certain_word="rec_epoch", certain_ext=".jpg", dst_dir=self.rec_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_rec_visual_read_dir,  certain_word="rec_gt",    certain_ext=".jpg", dst_dir=self.rec_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_rec_visual_write_dir, certain_word="rec_gt",    certain_ext=".jpg", dst_dir=self.rec_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_rec_visual_read_dir,  certain_word="combine",   certain_ext=".avi", dst_dir=self.rec_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_rec_visual_write_dir, certain_word="combine",   certain_ext=".avi", dst_dir=self.rec_visual_write_dir, print_msg=print_msg)


    ### 給下一步 metric 用的
    def get_bm_rec_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.bm_names  = get_dir_certain_file_name(self.bm_visual_read_dir , certain_word="bm_epoch", certain_ext=".jpg")
        self.bm_read_paths  = [self.bm_visual_read_dir + "/" + name for name in self.bm_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.rec_names = get_dir_certain_file_name(self.rec_visual_read_dir, certain_word="rec_epoch", certain_ext=".jpg")
        self.rec_read_paths = [self.rec_visual_read_dir + "/" + name for name in self.rec_names]  ### 沒有 write_path， 因為 bm_rec 只需要指定 write_dir 即可寫入資料夾

        self.see_rec_amount = len(self.rec_names)

    ###############################################################################################
    ###############################################################################################
    def Save_as_bm_rec_matplot_visual(self,   ### 訓練後，可以走訪所有see_file 並重新產生 bm_rec_matplot_visual
                                      add_loss=False,
                                      bgr2rgb =False,
                                      single_see_core_amount=CORE_AMOUNT_BM_REC_VISUAL,
                                      see_print_msg=False):
        """
        save_as_bm_rec_matplot_visual(_after_train) 最後想試試看 省掉他 會不會影響我的理解
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_bm_rec_matplot_visual, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.bm_rec_matplot_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果
        Check_dir_exist_and_build_new_dir(self.bm_visual_write_dir)              ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果
        Check_dir_exist_and_build_new_dir(self.rec_visual_write_dir)             ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果

        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.get_npz_info()
        print("here~~~~~~~~~~~~~~~~~~~~~", self.see_npz_epoch_amount)

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_bm_rec_matplot_visual_after_train(0, self.see_npz_epoch_amount, add_loss, bgr2rgb)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_npz_epoch_amount, task=self._draw_bm_rec_matplot_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=see_print_msg)
            ### 後處理讓結果更小 但 又不失視覺品質，多核心版(core_amount 在 step0 裡調)
            Find_ltrd_and_crop (self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        video_processes = []
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.see_read_dir, self.see_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉###
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.rec_visual_write_dir, self.rec_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        for video_p in video_processes: video_p.start()
        for video_p in video_processes: video_p.join()

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        ###   記得順序是 先同步父 再 同步子 喔！
        if(self.bm_rec_matplot_visual_write_dir != self.bm_rec_matplot_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.bm_rec_matplot_visual_write_dir, read_dir=self.bm_rec_matplot_visual_read_dir, build_new_dir=True)  ### 父
            Syn_write_to_read_dir(write_dir=self.bm_visual_write_dir,             read_dir=self.bm_visual_read_dir,             build_new_dir=True)  ### 子
            Syn_write_to_read_dir(write_dir=self.rec_visual_write_dir,            read_dir=self.rec_visual_read_dir,            build_new_dir=True)  ### 子

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Save_as_bm_rec_matplot_visual, Current See:{self.see_name}, cost time:{time.time() - start_time}")
        print("")

    ### See_method 第三部分：主要做的事情在這裡
    def _draw_bm_rec_matplot_visual_after_train(self, start_epoch, epoch_amount, add_loss, bgr2rgb):
        """
        有可能畫完主圖 還要再畫 loss，所以多這個method，多做的事情都在這裡處理
        處理完後就 Save_fig 囉！
        """
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            single_row_imgs = self._Draw_bm_rec_matplot_visual(go_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)
            single_row_imgs.Save_fig(dst_dir=self.bm_rec_matplot_visual_write_dir, epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    ### See_method 第三部分a
    ###     我覺得先把 npy 轉成 npz 再來生圖比較好，不要在這邊 邊生圖 邊轉 npz，覺得的原因如下：
    ###         1.這樣這裡做的事情太多了~~
    ###         2.npy轉npz 我會把 npy刪掉，但這樣第二次執行時 self.see_npy_names 就會是空的，還要寫if來判斷何時讀 npy, npz ，覺得複雜~
    def _Draw_bm_rec_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img    = cv2.imread(self.see_in_img_path)          ### 要記得see的jpg第一張存的是 輸入的in影像
        flow_v    = cv2.imread(self.see_flow_epoch_jpg_read_paths[epoch])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        gt_flow_v = cv2.imread(self.see_gt_flow_v_path )          ### 要記得see0的jpg第二張存的是 輸出的gt影像

        # print("2. see gt_use_range=", self.gt_use_range)
        # start_time = time.time()
        bm, rec, gt_bm, gt_rec = self._get_bm_rec_and_gt_bm_gt_rec(epoch=epoch, dis_img=in_img)  ### 做一次 大約 1~2 秒
        # print("self._get_bm_rec_and_gt_bm_gt_rec cost time:", time.time() - start_time)

        # bm_visual  = method1(bm[...,0], bm[...,1]*-1)
        # gt_bm_visual = method1(gt_bm[...,0], gt_bm[...,1]*-1)
        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img ,   flow_v ,   gt_flow_v, rec, gt_rec],    ### 把要顯示的每張圖包成list
                                img_titles=["in_img", "pred_flow_v", "gt_flow_v", "pred_rec", "gt_rec"],    ### 把每張圖要顯示的字包成list
                                fig_title ="epoch=%04i" % epoch,   ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)
        single_row_imgs.Draw_img()

        ### 單獨存大張 bm，有空再弄
        ### 單獨存大張 rec：
        if(epoch <= 3): cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_gt.jpg", gt_rec)  ### 存大張gt，gt只要存一次即可，所以加個if這樣子，<=3是因為 bm_rec 懶的寫防呆 是從 第四個epoch才開始做~~，要不然epoch==2 就行囉！，所以目前gt會存兩次拉但時間應該多一咪咪而以先這樣吧~~
        cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_epoch=%04i.jpg" % epoch, rec)     ### 存大張rec

        return single_row_imgs

    ### See_method 第三部分b
    def _get_bm_rec_and_gt_bm_gt_rec(self, epoch, dis_img):
        ### pred flow part
        flow          = np.load(self.see_npz_epoch_read_paths[epoch])["arr_0"]  ### see資料夾 內的flow 該epoch產生的flow 讀出來，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        flow [..., 1] = 1 - flow[..., 1]
        bm, rec = self._use_flow_to_rec(dis_img=dis_img, flow=flow)

        ### gt flow part
        gt_flow            = np.load(self.flow_gt_npz_path)["arr_0"]       ### 要記得see的npz 第一張存的是 gt_flow 喔！   ，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        gt_flow   [..., 1] = 1 - gt_flow[..., 1]
        gt_bm, gt_rec = self._use_flow_to_rec(dis_img=dis_img, flow=gt_flow)
        return bm, rec, gt_bm, gt_rec

    ### See_method 第三部分c
    def _use_flow_to_rec(self, dis_img, flow):
        if(self.gt_use_range == "-1~1"): flow = (flow + 1) / 2   ### 如果 gt_use_range 是 -1~1 記得轉回 0~1
        h, w = flow.shape[:2]
        total_pix_amount = h * w
        valid_mask_pix_amount = (flow[..., 0] >= 0.99).astype(np.int).sum()
        # print("valid_mask_pix_amount / total_pix_amount:", valid_mask_pix_amount / total_pix_amount)
        if( valid_mask_pix_amount / total_pix_amount > 0.28):
            bm  = use_flow_to_get_bm(flow, flow_scale=h)
            rec = use_bm_to_rec_img (bm  , flow_scale=h, dis_img=dis_img)
        else:
            bm  = np.zeros(shape=(h, w, 2))
            rec = np.zeros(shape=(h, w, 3))
        return bm, rec


    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ### 好像都沒用到，先註解起來吧～再看看要不要刪掉
    # def Save_as_bm_rec_matplot_visual_at_certain_epoch(self, epoch, add_loss=False, bgr2rgb=False):   ### 訓練後，對"指定"epoch的 see結果 產生 bm_rec_matplot_visual
    #     print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_bm_rec_matplot_visual_at_certain_epoch, Current See:{self.see_name}, at_certain_epoch:{epoch}")
    #     start_time = time.time()
    #     Check_dir_exist_and_build(self.see_write_dir)
    #     Check_dir_exist_and_build(self.bm_rec_matplot_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
    #     Check_dir_exist_and_build(self.bm_visual_write_dir)              ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
    #     Check_dir_exist_and_build(self.rec_write_visual_dir)             ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
    #     self.get_see_base_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量

    #     ### 防呆一下
    #     current_final_epoch = self.see_file_amount - 3   ### epochs是 epoch總數，要減掉：in_img, gt_img 和 epoch0
    #     if(epoch <= current_final_epoch):
    #         single_row_imgs = self._Draw_bm_rec_matplot_visual(epoch, add_loss, bgr2rgb)
    #         single_row_imgs.Save_fig(dst_dir=self.bm_rec_matplot_visual_write_dir, epoch=epoch)  ### 如果沒有要接續畫loss，就可以存了喔！
    #         ### 後處理讓結果更小 但 又不失視覺品質
    #         # Find_ltrd_and_crop(self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 兩次以上有危險可能會 crop錯喔！所以就不crop了~
    #         Save_as_jpg(self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], multiprocess=True, core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
    #         print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_bm_rec_matplot_visual_at_certain_epoch, Current See:{self.see_name}, at_certain_epoch:{epoch}, cost_time:{time.time() - start_time}")
    #     else:
    #         print("epoch=%i 超過目前exp的epoch數目囉！有可能是還沒train完see還沒產生到該epoch 或者 是輸入的epoch數 超過 epochs囉！" % epoch)
    #         print("Save_as_bm_rec_matplot_visual_at_certain_epoch不做事情拉~")

#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
class See_rec_metric(See_bm_rec):
    """
    我把它繼承See_bm_rec 的概念是要做完 See_bm_rec 後才能做 See_rec_metric 喔！
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_rec_metric, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """

        self.old1_metric_read_dir  = self.see_read_dir  + "/matplot_metric_visual"
        self.old1_metric_write_dir = self.see_write_dir + "/matplot_metric_visual"
        self.old2_metric_read_dir  = self.see_read_dir  + "/metric"
        self.old2_metric_write_dir = self.see_write_dir + "/metric"
        self.old3_metric_read_dir  = self.see_read_dir  + "/2_metric"
        self.old3_metric_write_dir = self.see_write_dir + "/2_metric"

        self.old1_metric_ld_color_read_dir  = self.see_read_dir  + "/metric"
        self.old1_metric_ld_color_write_dir = self.see_write_dir + "/metric"
        self.old2_metric_ld_color_read_dir  = self.see_read_dir  + "/metric/ld_color"
        self.old2_metric_ld_color_write_dir = self.see_write_dir + "/metric/ld_color"
        self.old3_metric_ld_color_read_dir  = self.see_read_dir  + "/2_metric/ld_color"
        self.old3_metric_ld_color_write_dir = self.see_write_dir + "/2_metric/ld_color"
        self.old_ld_gray_read_dir   = self.see_read_dir  + "/metric/ld_gray"
        self.old_ld_gray_write_dir  = self.see_write_dir + "/metric/ld_gray"
        self.old2_ld_gray_read_dir   = self.see_read_dir  + "/2_metric/ld_gray"
        self.old2_ld_gray_write_dir  = self.see_write_dir + "/2_metric/ld_gray"

        self.old_matplot_metric_visual_read_dir  = self.see_read_dir  + "/matplot_metric_visual"
        self.old_matplot_metric_visual_write_dir = self.see_write_dir + "/matplot_metric_visual"
        self.old2_matplot_metric_visual_read_dir  = self.see_read_dir  + "/2_matplot_metric_visual"
        self.old2_matplot_metric_visual_write_dir = self.see_write_dir + "/2_matplot_metric_visual"


        self.metrec_read_dir  = self.see_read_dir  + "/3_metric"
        self.metrec_write_dir = self.see_write_dir + "/3_metric"
        self.metric_ld_color_read_dir  = self.see_read_dir  + "/3_metric/ld_color"
        self.metric_ld_color_write_dir = self.see_write_dir + "/3_metric/ld_color"
        self.metric_ld_gray_read_dir   = self.see_read_dir  + "/3_metric/ld_gray"
        self.metric_ld_gray_write_dir  = self.see_write_dir + "/3_metric/ld_gray"
        self.metric_ld_matplot_read_dir   = self.see_read_dir  + "/3_metric/ld_matplot"
        self.metric_ld_matplot_write_dir  = self.see_write_dir + "/3_metric/ld_matplot"
        self.metric_im1_read_dir   = self.see_read_dir  + "/3_metric/im1"
        self.metric_im1_write_dir  = self.see_write_dir + "/3_metric/im1"
        self.metric_im2_read_dir   = self.see_read_dir  + "/3_metric/im2"
        self.metric_im2_write_dir  = self.see_write_dir + "/3_metric/im2"


        self.matplot_metric_visual_read_dir  = self.see_read_dir  + "/3_matplot_metric_visual"
        self.matplot_metric_visual_write_dir = self.see_write_dir + "/3_matplot_metric_visual"

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_metric_dir(print_msg=True)


    def Change_metric_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_metric_dir, Current See:{self.see_name}")
        # start_time = time.time()

        # move_dir_certain_file(self.old1_metric_read_dir,  certain_word=".npy", dst_dir=self.metrec_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old1_metric_write_dir, certain_word=".npy", dst_dir=self.metrec_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_metric_read_dir,  certain_word=".npy", dst_dir=self.metrec_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old2_metric_write_dir, certain_word=".npy", dst_dir=self.metrec_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old3_metric_read_dir,  certain_word=".npy", dst_dir=self.metrec_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old3_metric_write_dir, certain_word=".npy", dst_dir=self.metrec_write_dir, print_msg=print_msg)

        # move_dir_certain_file(self.old1_metric_ld_color_read_dir,  certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_color_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old1_metric_ld_color_write_dir, certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_color_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_metric_ld_color_read_dir,  certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_color_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old2_metric_ld_color_write_dir, certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_color_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old3_metric_ld_color_read_dir,  certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_color_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old3_metric_ld_color_write_dir, certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_color_write_dir, print_msg=print_msg)

        # move_dir_certain_file(self.old_ld_gray_read_dir,  certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_gray_read_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_ld_gray_write_dir, certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_gray_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_ld_gray_read_dir,  certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_gray_read_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_ld_gray_write_dir, certain_word="ld_epoch", certain_ext=".jpg", dst_dir=self.metric_ld_gray_write_dir, print_msg=print_msg)

        # move_dir_certain_file(self.old_matplot_metric_visual_read_dir,  certain_word="epoch", certain_ext=".jpg", dst_dir=self.matplot_metric_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_metric_visual_write_dir, certain_word="epoch", certain_ext=".jpg", dst_dir=self.matplot_metric_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_metric_visual_read_dir,  certain_word="combine",      certain_ext=".avi", dst_dir=self.matplot_metric_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_matplot_metric_visual_write_dir, certain_word="combine",      certain_ext=".avi", dst_dir=self.matplot_metric_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_matplot_metric_visual_read_dir,  certain_word="epoch", certain_ext=".jpg", dst_dir=self.matplot_metric_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old2_matplot_metric_visual_write_dir, certain_word="epoch", certain_ext=".jpg", dst_dir=self.matplot_metric_visual_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_matplot_metric_visual_read_dir,  certain_word="combine",      certain_ext=".avi", dst_dir=self.matplot_metric_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old2_matplot_metric_visual_write_dir, certain_word="combine",      certain_ext=".avi", dst_dir=self.matplot_metric_visual_write_dir, print_msg=print_msg)

        # print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Change_metric_dir, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

    ### 給下一步 metric_visual 用的
    def get_metric_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        # self.metric_names  = get_dir_certain_file_name(self.metrec_read_dir , certain_word="metric_epoch", certain_ext=".jpg")
        # self.metric_read_paths  = [self.matplot_metric_visual_read_dir + "/" + name for name in self.metric_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.ld_color_visual_names       = get_dir_certain_file_name(self.metric_ld_color_read_dir , certain_word="ld_epoch", certain_ext=".jpg")
        self.ld_color_visual_read_path   = [self.metric_ld_color_read_dir + "/" + name for name in self.ld_color_visual_names]  ### 沒有 write_path， 因為 ld_visual 只需要指定 write_dir 即可寫入資料夾

        # self.see_file_amount = len(self.metric_names)

    ###############################################################################################
    ###############################################################################################
    def Calculate_SSIM_LD(self, single_see_core_amount=8,
                                see_print_msg=False):
        """
        覺得還是要用 path 的方式 在 matlab 裡面 用 imread(path)，
        path 的方式：8秒左右
        python內np.array 傳給 matlab：30秒左右

        假設：
            1.都要 compress_all 完以後
            2.並把結果都 集中到一起
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Calculate_SSIM_LD, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build(self.metrec_write_dir)           ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～
        Check_dir_exist_and_build(self.metric_ld_color_write_dir)  ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～
        Check_dir_exist_and_build(self.metric_ld_gray_write_dir)   ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～
        Check_dir_exist_and_build(self.metric_ld_matplot_write_dir)   ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～
        Check_dir_exist_and_build(self.metric_im1_write_dir)   ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～
        Check_dir_exist_and_build(self.metric_im2_write_dir)   ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～

        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()  ### 暫時寫這邊，到時應該要拉出去到result_level，要不然每做一次就要重新更新一次，但不用這麼頻繁，只需要一開始更新一次即可
        self.get_bm_rec_info()   ### 暫時寫這邊，到時應該要拉出去到result_level，要不然每做一次就要重新更新一次，但不用這麼頻繁，只需要一開始更新一次即可

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        with Manager() as manager:  ### 設定在 multiprocess 裡面 共用的 list
            ### multiprocess 內的 global 的 list， share memory 的概念了，就算不multiprocess 也可以用喔！ 不過記得如果要在with外用， 要先轉回list() 就是了！
            SSIMs = manager.list()  # []的概念
            LDs   = manager.list()  # []的概念

            if  (single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
                self._do_matlab_SSIM_LD(0, self.see_rec_amount, SSIMs, LDs)
                Save_as_jpg        (self.metric_ld_matplot_write_dir, self.metric_ld_matplot_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
            elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
                multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_rec_amount, task=self._do_matlab_SSIM_LD, task_args=[SSIMs, LDs], print_msg=see_print_msg)
                Save_as_jpg        (self.metric_ld_matplot_write_dir, self.metric_ld_matplot_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
            else:
                print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

            # SSIMs = list(SSIMs)  ### share memory 的list 轉回 python list
            # LDs   = list(LDs)    ### share memory 的list 轉回 python list
            SSIMs = sorted(SSIMs, key=lambda ssim : ssim[0])  ### 有轉 list 的功效喔！所以上面兩行可省！
            LDs   = sorted(LDs,   key=lambda LD   : LD[0])    ### 有轉 list 的功效喔！所以上面兩行可省！

        ### See_method 第四部分：後處理，去除 在 share memory 的 index 後 存成 npy
        SSIMs = np.array(SSIMs)[:, 1]
        LDs   = np.array(LDs)[:, 1]
        # print("SSIMs", SSIMs)
        # print("LDs", LDs)
        np.save(f"{self.metrec_write_dir}/SSIMs", SSIMs)
        np.save(f"{self.metrec_write_dir}/LDs",   LDs)

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        ###   記得順序是 先同步父 再 同步子 喔！
        if(self.metrec_write_dir != self.metrec_read_dir):  ### 因為接下去的任務需要 此任務的結果， 如果 read/write 資料夾位置不一樣， write完的結果 copy 一份 放回read， 才能讓接下去的動作 有 東西 read 喔！
            Syn_write_to_read_dir(write_dir=self.metrec_write_dir,          read_dir=self.metrec_read_dir,          build_new_dir=True)  ### 父
            Syn_write_to_read_dir(write_dir=self.metric_ld_color_write_dir, read_dir=self.metric_ld_color_read_dir, build_new_dir=True)  ### 子
            Syn_write_to_read_dir(write_dir=self.metric_ld_gray_write_dir,  read_dir=self.metric_ld_gray_read_dir,  build_new_dir=True)  ### 子

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Calculate_SSIM_LD, Current See:{self.see_name}, cost_time:{time.time() - start_time}")
        print("")

    ### See_method 第三部分：主要做的事情在這裡
    def _do_matlab_SSIM_LD(self, start_epoch, epoch_amount, SSIMs, LDs):
        from kong_use_evalUnwarp_sucess import use_DewarpNet_eval

        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            ### rec_GT 要怎麼轉成 rec_pred
            path1 = self.rec_read_paths[go_epoch]  ### bm_rec_matplot_visual/rec_visual/rec_epoch=0000.jpg
            path2 = self.see_rec_hope_path     ### 0c-rec_hope.jpg
            # path2 = self.rec_read_paths[-1]        ### bm_rec_matplot_visual/rec_visual/rec_gt.jpg

            ### rec_pred 要怎麼轉成 rec_GT
            # path1 = self.see_rec_hope_path     ### 0c-rec_hope.jpg
            # path2 = self.rec_read_paths[go_epoch]  ### bm_rec_matplot_visual/rec_visual/rec_epoch=0000.jpg

            # print("path1~~~~~~~~~~~~", path1)
            # print("path2~~~~~~~~~~~~", path2)

            ord_dir = os.getcwd()                            ### step1 紀錄 目前的主程式資料夾
            os.chdir("SIFT_dev/SIFTflow")                    ### step2 跳到 SIFTflow資料夾裡面
            [SSIM, LD, vx, vy, d, im1, im2] = use_DewarpNet_eval(path1, path2)  ### step3 執行 SIFTflow資料夾裡面 的 kong_use_evalUnwarp_sucess.use_DewarpNet_eval 來執行 kong_evalUnwarp_sucess.m
            os.chdir(ord_dir)                                ### step4 跳回 主程式資料夾

            # fig, ax = plt.subplots(nrows=1, ncols=2)
            # rec_img    = cv2.imread(path1)
            # rec_gt_img = cv2.imread(path2)
            # ax[0].imshow(rec_img)
            # ax[1].imshow(rec_gt_img)
            # plt.show()
            # plt.close()

            single_row_imgs = Matplot_single_row_imgs(
                        imgs      =[d],    ### 把要顯示的每張圖包成list
                        img_titles=[],
                        fig_title ="",   ### 圖上的大標題
                        pure_img  =True,
                        add_loss  =False,
                        bgr2rgb   =False)
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(self.metric_ld_matplot_write_dir, epoch=go_epoch, epoch_name="ld_epoch")
            # print("d.max()~~~~~~~~~~", d.max())  ### 目前手動看 大概就是 epoch=0000 會很大 剩下epoch都很小， 然後epoch=0000 大概都40幾， 所以我設50囉！
            # plt.show()

            ld_visual = method2(vx, vy, color_shift=3)  ### 因為等等是 直接用 cv2 直接寫，所以不用 bgr2rgb喔！

            cv2.imwrite(self.metric_ld_color_write_dir + "/ld_epoch=%04i.jpg" % go_epoch, ld_visual)
            cv2.imwrite(self.metric_ld_gray_write_dir  + "/ld_epoch=%04i.jpg" % go_epoch, d.astype(np.uint8))
            cv2.imwrite(self.metric_im1_write_dir  + "/im1_epoch=%04i.jpg" % go_epoch, (im1 * 255).astype(np.uint8))
            cv2.imwrite(self.metric_im2_write_dir  + "/im1_epoch=%04i.jpg" % go_epoch, (im2 * 255).astype(np.uint8))

            # print(go_epoch, SSIM, LD)
            SSIMs.append((go_epoch, SSIM))
            LDs  .append((go_epoch, LD))

    ###############################################################################################
    ###############################################################################################
    def Visual_SSIM_LD(self, add_loss=False,
                             bgr2rgb =False,
                             single_see_core_amount=8,
                             see_print_msg=False):

        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Visual_SSIM_LD, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build_new_dir(self.matplot_metric_visual_write_dir)  ### 一定要build_new_dir ，才不會有 "中斷後重新執行 或 第二次執行"時 .jpg 和 .png 混再一起 擾亂了 Find_ltrd_and_crop 喔！

        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()
        self.get_bm_rec_info()
        self.get_metric_info()

        SSIMs = np.load(f"{self.metrec_read_dir}/SSIMs.npy")
        LDs   = np.load(f"{self.metrec_read_dir}/LDs.npy")

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._visual_SSIM_LD(0, self.see_rec_amount, SSIMs, LDs, add_loss=add_loss, bgr2rgb=bgr2rgb)
            Find_ltrd_and_crop     (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg            (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_rec_amount, task=self._visual_SSIM_LD, task_args=[SSIMs, LDs, add_loss, bgr2rgb], print_msg=see_print_msg)
            Find_ltrd_and_crop     (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg            (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        Video_combine_from_dir (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir)


        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.matplot_metric_visual_write_dir != self.matplot_metric_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.matplot_metric_visual_write_dir, read_dir=self.matplot_metric_visual_read_dir, build_new_dir=True)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Visual_SSIM_LD, Current See:{self.see_name}, cost_time:{time.time() - start_time}")
        print("")

    ### See_method 第三部分：主要做的事情在這裡
    def _visual_SSIM_LD(self, start_epoch, epoch_amount, SSIMs, LDs, add_loss=False, bgr2rgb=False):
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            path1 = self.rec_read_paths[go_epoch]  ### bm_rec_matplot_visual/rec_visual/rec_epoch=0000.jpg
            path2 = self.see_rec_hope_path     ### 0c-rec_hope.jpg
            SSIM = SSIMs[go_epoch]
            LD   = LDs  [go_epoch]

            in_img     = cv2.imread(self.see_in_img_path)
            rec_img    = cv2.imread(path1)
            rec_gt_img = cv2.imread(path2)
            ld_visual  = cv2.imread(self.ld_color_visual_read_path[go_epoch])
            single_row_imgs = Matplot_single_row_imgs(
                        imgs      =[in_img,   rec_img ,   rec_gt_img],    ### 把要顯示的每張圖包成list
                        img_titles=[ "in_img", "rec"    , "gt_rec"],    ### 把每張圖要顯示的字包成list
                        fig_title ="epoch=%04i, SSIM=%.2f, LD=%.2f" % (go_epoch, SSIM, LD),   ### 圖上的大標題
                        add_loss  =False,
                        bgr2rgb   =bgr2rgb)
            if(add_loss):
                ld_visual = ld_visual[..., ::-1]  ### opencv -> matplot
                ### step1 先架構好 整張圖的骨架
                single_row_imgs.step1_add_row_col(add_where="add_row", merge=True)
                single_row_imgs.step1_add_row_col(add_where="add_col", merge=True, grid_ratio=1.9)

                ### step2 把圖都畫上去
                single_row_imgs.Draw_img()
                single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.merged_ax_list[0], self.metrec_read_dir, go_epoch, min_epochs=self.see_rec_amount, ylim=25)
                single_row_imgs.merged_ax_list[1].imshow(ld_visual)

                ### step3 重新規劃一下 各個圖 要顯示的 大小比例
                gs_bass = GridSpec(single_row_imgs.fig_row_amount, single_row_imgs.fig_col_amount, width_ratios=[1, 1, 1, 2], height_ratios=[1, 1])
                for go_r, r_ax in enumerate(single_row_imgs.ax):
                    for go_c, r_c_ax in enumerate(r_ax):
                        # print(f"gs_bass[{go_c}].get_position(single_row_imgs.fig)", gs_bass[go_r, go_c].get_position(single_row_imgs.fig))  ### 可以看到 目前使用的規格的範圍 和 其 對應到 single_row_imgs.fig 上 框出的box是在哪裡
                        r_c_ax.set_position(gs_bass[go_c].get_position(single_row_imgs.fig))    ### 根據目前的圖(single_row_imgs.fig)， 重新規劃一下 各個圖 要顯示的 大小比例

                # print("gs_bass[1, :3]", gs_bass[1, :3].get_position(single_row_imgs.fig))  ### 可以看到 目前使用的規格的範圍 和 其 對應到 single_row_imgs.fig 上 框出的box是在哪裡
                # print("gs_bass[:, 3 ]", gs_bass[:, 3 ].get_position(single_row_imgs.fig))  ### 可以看到 目前使用的規格的範圍 和 其 對應到 single_row_imgs.fig 上 框出的box是在哪裡
                single_row_imgs.merged_ax_list[0].set_position(gs_bass[1, :3].get_position(single_row_imgs.fig))   ### 根據目前的圖(single_row_imgs.fig)， 重新規劃一下 各個圖 要顯示的 大小比例
                single_row_imgs.merged_ax_list[1].set_position(gs_bass[:, 3 ].get_position(single_row_imgs.fig))   ### 根據目前的圖(single_row_imgs.fig)， 重新規劃一下 各個圖 要顯示的 大小比例

            # plt.show()
            single_row_imgs.Save_fig(dst_dir=self.matplot_metric_visual_write_dir, epoch=go_epoch, epoch_name="metric_epoch")  ### 話完圖就可以存了喔！



class See(See_flow_visual, See_rec_metric):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See, self).__init__(result_read_dir, result_write_dir, see_name)



if(__name__ == "__main__"):
    from step0_access_path import result_read_path, result_write_path
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_3b_4-20210306_231628-flow_unet-ch32_bn_16", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_005-train")
    # try_npy_to_npz.npy_to_npz_comapre()
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)

    test_see = See(result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-testest", result_write_dir=result_write_path + "result/5_14_flow_unet/type8_blender_os_book-testest", see_name="see_001-real")
    test_see.Calculate_SSIM_LD(epoch=0, single_see_core_amount=8)
