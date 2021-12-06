from step11_a0_see_base import See_info

from step0_access_path import JPG_QUALITY, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_names, move_dir_certain_file
from matplot_fig_ax_util import Matplot_single_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir
from multiprocess_util import multi_processing_interface

import cv2
import time
from tqdm import tqdm

import datetime
# import pdb

'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''


class See_flow_visual(See_info):
    """
    See_flow_visual 是用來視覺化 See 的物件，因此這個Class我覺得也應該要設計成 training 中可以被使用的這樣子囉
      所以要看的東西就是簡單的：
        單純的input, 單純的output, 單純的gt
    """
    def __init__(self, result_obj, see_name):
        super(See_flow_visual, self).__init__(result_obj, see_name)
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.old_flow_matplot_visual_read_dir   = self.see_read_dir  + "/matplot_visual"
        self.old_flow_matplot_visual_write_dir  = self.see_write_dir + "/matplot_visual"

        self.flow_matplot_visual_read_dir  = self.see_read_dir  + "/1_flow_matplot_visual"
        self.flow_matplot_visual_write_dir = self.see_write_dir + "/1_flow_matplot_visual"
        self.flow_v_read_dir  = self.see_read_dir
        self.flow_v_write_dir = self.see_write_dir

        self.single_row_imgs_during_train = None  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_flow_matplot_dir(print_msg=True)

    def Change_flow_matplot_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_flow_matplot_dir, Current See:{self.see_name}")
        # move_dir_certain_file(self.old_flow_matplot_visual_read_dir,  certain_word="flow_epoch", certain_ext=".jpg", dst_dir=self.flow_matplot_visual_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old_flow_matplot_visual_write_dir, certain_word="flow_epoch", certain_ext=".jpg", dst_dir=self.flow_matplot_visual_write_dir, print_msg=print_msg)

    ### 因為  See_bm_rec 要用到， 所以從 See_flow_visual 提升上去 See_info囉！ 結果還是拉回來了， 因為覺得要用的時候再抓， 否則像只 predict mask 的狀況，就沒有flow拉！
    def get_flow_info(self):
        self.gt_flow_jpg_path   = self.get_path_savely(self.see_read_dir, certain_word="gt_flow", certain_ext=".jpg")
        self.rec_hope_path        = self.flow_v_write_dir + "/" + get_dir_certain_file_names(self.see_read_dir, certain_word="rec_hope")[0]

        self.flow_ep_jpg_names      = get_dir_certain_file_names(self.see_read_dir, certain_word="epoch", certain_word2="flow", certain_ext=".jpg")
        self.flow_ep_jpg_read_paths = [self.see_read_dir + "/" + epoch_name for epoch_name in self.flow_ep_jpg_names]  ### 沒有 write_paths， 同上
        self.flow_ep_jpg_amount     = len(self.flow_ep_jpg_names)


        self.trained_epoch       = self.flow_ep_jpg_amount - 1  ### 去掉epoch0

    ###############################################################################################
    ###############################################################################################
    ### 主要做的事情，此fun會給 save_as_matplot_visual_during/after train 使用
    def _Draw_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img = cv2.imread(self.in_img_path)                    ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.gt_flow_jpg_path )              ### 要記得see的第二張存的是 輸出的gt影像
        img    = cv2.imread(self.flow_ep_jpg_read_paths[epoch])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
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
        Check_dir_exist_and_build(self.flow_matplot_visual_write_dir)
        start_time = time.time()
        # if(epoch==0):
        #     Check_dir_exist_and_build_new_dir(self.flow_matplot_visual_write_dir)      ### 建立 存結果的資料夾
        self.get_see_base_info()  ### 每次執行都要 update喔！ 取得result內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.single_row_imgs_during_train = self._Draw_matplot_visual(epoch, add_loss=True, bgr2rgb=bgr2rgb)  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！
        if(show_msg): print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing save_as_matplot_visual_during_train, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

    ###############################################################################################
    def draw_loss_at_see_during_train(self, epoch, epochs):
        Check_dir_exist_and_build(self.flow_matplot_visual_write_dir)  ### 以防matplot_visual資料夾被刪掉，要生圖找不到資料夾
        self.single_row_imgs_during_train.Draw_ax_loss_during_train(self.single_row_imgs_during_train.ax[-1, 1], self.see_read_dir + "/../logs", epoch, epochs)
        self.single_row_imgs_during_train.Save_fig(dst_dir=self.flow_matplot_visual_write_dir, name="epoch", epoch=epoch)

    ###############################################################################################
    ###############################################################################################
    def Save_as_matplot_visual(self, add_loss=False, bgr2rgb=False, single_see_core_amount=8, see_print_msg=False, **args):
        """
        Save_as_matplot_visual(_after_train) 最後想試試看 省掉他 會不會影響我的理解
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_matplot_visual, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.flow_matplot_visual_write_dir)      ### 建立 存結果的資料夾

        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.get_flow_info()

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_matplot_visual_after_train(0, self.flow_ep_jpg_amount, add_loss=add_loss, bgr2rgb=bgr2rgb)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.flow_matplot_visual_write_dir, self.flow_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.flow_matplot_visual_write_dir, self.flow_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            ### see內的任務 有切 multiprocess
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.flow_ep_jpg_amount, task=self._draw_matplot_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=see_print_msg)
            ### 後處理讓結果更小 但 又不失視覺品質，多核心版(core_amount 在 step0 裡調)
            Find_ltrd_and_crop (self.flow_matplot_visual_write_dir, self.flow_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.flow_matplot_visual_write_dir, self.flow_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        Video_combine_from_dir(self.flow_matplot_visual_write_dir, self.flow_matplot_visual_write_dir)          ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.flow_matplot_visual_write_dir != self.flow_matplot_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.flow_matplot_visual_write_dir, read_dir=self.flow_matplot_visual_read_dir, build_new_dir=True)

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
            if(add_loss)   : single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.ax[-1, 1], self.see_read_dir + "/../logs", go_epoch, min_epochs=self.flow_ep_jpg_amount, ylim=0.04)
            single_row_imgs.Save_fig(dst_dir=self.flow_matplot_visual_write_dir, name="epoch", epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

