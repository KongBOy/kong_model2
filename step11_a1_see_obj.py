from step0_access_path import JPG_QUALITY, CORE_AMOUNT, CORE_AMOUNT_NPY_TO_NPZ, CORE_AMOUNT_BM_REC_VISUAL, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name, remove_dir_certain_file_name
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
from kong_use_evalUnwarp_sucess import use_DewarpNet_eval
import matplotlib.pyplot as plt  ### debug用
import datetime
# import pdb

class See_info:
    '''
    See 是 最直接、最基本 model output的東西，在各個model裡面都應該有寫 自己的 generate_see
    而這邊只是 讀取 training 過程中生成的 See 這樣子囉~~
    '''
    def __init__(self, result_read_dir, result_write_dir, see_name):
        self.result_read_dir = result_read_dir
        self.result_write_dir = result_write_dir

        self.see_name = see_name
        self.see_read_dir = self.result_read_dir + "/" + self.see_name
        self.see_write_dir = self.result_write_dir + "/" + self.see_name

        self.see_jpg_names = None
        self.see_npy_names = None
        self.see_file_amount = None

        self.in_use_range = "0~1"
        self.gt_use_range = "0~1"

    def get_see_dir_info(self):
        self.see_jpg_names   = get_dir_certain_file_name(self.see_read_dir, certain_word=".jpg")
        self.see_jpg_paths = [self.see_read_dir + "/" + jpg_name for jpg_name in self.see_jpg_names]

        self.see_epoch_jpg_names = get_dir_certain_file_name(self.see_read_dir, certain_word="epoch", certain_ext=".jpg")
        self.see_epoch_jpg_paths = [self.see_read_dir + "/" + epoch_jpg_name for epoch_jpg_name in self.see_epoch_jpg_names]
        self.see_file_amount     = len(self.see_epoch_jpg_names)

        self.see_npy_names   = get_dir_certain_file_name(self.see_read_dir, certain_word=".npy")
        self.see_npy_read_paths  = [self.see_read_dir  + "/" + npy_name for npy_name in self.see_npy_names]  ### 沒有 write_paths，因為式 npy轉npz， 不會有寫npy的動作， 雖然下面的 compare 會寫一點npy， 但也因為 有用 .replace() 所以用 see_npy_name.replace() 較保險這樣子！


    def save_as_jpg(self):  ### 後來看覺得好像有點多餘
        Check_dir_exist_and_build(self.see_write_dir)
        Save_as_jpg(self.see_write_dir, self.see_write_dir, delete_ord_file=True)

    def save_as_avi(self):  ### 後來看覺得好像有點多餘
        Check_dir_exist_and_build(self.see_write_dir)
        Video_combine_from_dir(self.see_read_dir, self.see_write_dir, "0-combine_jpg_tail_long.avi", tail_long=True)


class See_visual(See_info):
    """
    See_visual 是用來視覺化 See 的物件，因此這個Class我覺得也應該要設計成 training 中可以被使用的這樣子囉
      所以要看的東西就是簡單的：
        單純的input, 單純的output, 單純的gt
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_visual, self).__init__(result_read_dir, result_write_dir, see_name)
        self.matplot_visual_read_dir  = self.see_read_dir  + "/matplot_visual"
        self.matplot_visual_write_dir = self.see_write_dir + "/matplot_visual"

        ### 不確定要不要，因為在initial就做這麼多事情好嗎~~會不會容易出錯哩~~
        ### 覺得還是不要比較好，要使用到的時候再建立，要不然有時候在analyze只是想要result_obj而已，結果又把see資料夾又重建了一次
        # Check_dir_exist_and_build(self.see_write_dir)
        # self.get_see_dir_info()   ### 好像只有在 analyze時會用到！所以用到的時候再抓就好囉！
        self.single_row_imgs_during_train = None  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！

    ###############################################################################################
    ###############################################################################################
    ### 主要做的事情，此fun會給 save_as_matplot_visual_during/after train 使用
    def _Draw_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img = cv2.imread(self.see_jpg_paths[0])            ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_jpg_paths[1])            ### 要記得see的第二張存的是 輸出的gt影像
        img    = cv2.imread(self.see_epoch_jpg_paths[epoch])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img ,   img ,      gt_img],    ### 把要顯示的每張圖包成list
                                img_titles=["in_img", "out_img", "gt_img"],    ### 把每張圖要顯示的字包成list
                                fig_title ="epoch=%04i" % epoch,   ### 圖上的大標題
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
        self.get_see_dir_info()  ### 每次執行都要 update喔！ 取得result內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.single_row_imgs_during_train = self._Draw_matplot_visual(epoch, add_loss=True, bgr2rgb=bgr2rgb)  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！
        if(show_msg): print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing save_as_matplot_visual_during_train, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

    ###############################################################################################
    def draw_loss_at_see_during_train(self, epoch, epochs):
        Check_dir_exist_and_build(self.matplot_visual_write_dir)  ### 以防matplot_visual資料夾被刪掉，要生圖找不到資料夾
        self.single_row_imgs_during_train.Draw_ax_loss_during_train(self.single_row_imgs_during_train.ax[-1, 1], self.see_read_dir + "/../logs", epoch, epochs)
        self.single_row_imgs_during_train.Save_fig(dst_dir=self.matplot_visual_write_dir, epoch=epoch)

    ###############################################################################################
    ###############################################################################################
    ### 訓練後，可以走訪所有see_file 並重新產生 matplot_visual
    ### See_method 第三部分：主要做的事情在這裡
    def _draw_matplot_visual_after_train(self, start_epoch, epoch_amount, add_loss=False, bgr2rgb=False):
        """
        有可能畫完主圖 還要再畫 loss，所以多這個method，多做的事情都在這裡處理
        處理完後就 Save_fig 囉！
        """
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            single_row_imgs = self._Draw_matplot_visual(go_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)
            if(add_loss)   : single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.ax[-1, 1], self.see_read_dir + "/../logs", go_epoch, min_epochs=self.see_file_amount, ylim=0.04)
            single_row_imgs.Save_fig(dst_dir=self.matplot_visual_write_dir, epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

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
        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_matplot_visual_after_train(0, self.see_file_amount, add_loss=add_loss, bgr2rgb=bgr2rgb)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.matplot_visual_write_dir, self.matplot_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.matplot_visual_write_dir, self.matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            ### see內的任務 有切 multiprocess
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_file_amount, task=self._draw_matplot_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=see_print_msg)
            ### 後處理讓結果更小 但 又不失視覺品質，多核心版(core_amount 在 step0 裡調)
            Find_ltrd_and_crop (self.matplot_visual_write_dir, self.matplot_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.matplot_visual_write_dir, self.matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        Video_combine_from_dir(self.matplot_visual_write_dir, self.matplot_visual_write_dir)          ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.matplot_visual_write_dir != self.matplot_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.matplot_visual_write_dir, read_dir=self.matplot_visual_read_dir)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Save_as_matplot_visual, Current See:{self.see_name}, cost_time:{time.time() - start_time}")
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################


class See_bm_rec(See_info):
    """
    See_bm_rec 是用來 把 模型生成的 See 裡面的 flow，去生成 bm, rec，順便也視覺化出來這樣子囉
       所以裡面會有：
          bm_visual資料夾
          rec_visual資料夾
          順便視覺化一下(input, output, output_gt, rec, rec_gt)，bm視覺化好像有點問題先跳過ˊ口ˋ
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_bm_rec, self).__init__(result_read_dir, result_write_dir, see_name)

        self.matplot_bm_rec_visual_read_dir   = self.see_read_dir  + "/matplot_bm_rec_visual"
        self.matplot_bm_rec_visual_write_dir  = self.see_write_dir + "/matplot_bm_rec_visual"
        self.bm_visual_read_dir               = self.see_read_dir  + "/matplot_bm_rec_visual/bm_visual"
        self.bm_visual_write_dir              = self.see_write_dir + "/matplot_bm_rec_visual/bm_visual"
        self.rec_visual_read_dir              = self.see_read_dir  + "/matplot_bm_rec_visual/rec_visual"
        self.rec_visual_write_dir             = self.see_write_dir + "/matplot_bm_rec_visual/rec_visual"
        self.bm_names  = None
        self.bm_paths  = None
        self.rec_names = None
        self.rec_paths = None

    ###############################################################################################
    ###############################################################################################
    def get_bm_rec_info(self):
        self.bm_names  = get_dir_certain_file_name(self.bm_visual_read_dir , certain_word="bm_epoch", certain_ext=".jpg")
        self.bm_paths  = [self.bm_visual_read_dir + "/" + name for name in self.bm_names]
        self.rec_names = get_dir_certain_file_name(self.rec_visual_read_dir, certain_word="rec_epoch", certain_ext=".jpg")
        self.rec_paths = [self.rec_visual_read_dir + "/" + name for name in self.rec_names]

        self.see_file_amount = len(self.rec_names)

    ###############################################################################################
    ###############################################################################################
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

    def _get_bm_rec_and_gt_bm_gt_rec(self, epoch, dis_img):
        ### pred flow part
        flow          = np.load(self.see_epoch_npz_read_paths[epoch])["arr_0"]  ### see資料夾 內的flow 該epoch產生的flow 讀出來，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        flow [..., 1] = 1 - flow[..., 1]
        bm, rec = self._use_flow_to_rec(dis_img=dis_img, flow=flow)

        ### gt flow part
        gt_flow            = np.load(self.see_npz_read_paths[0])["arr_0"]       ### 要記得see的npz 第一張存的是 gt_flow 喔！   ，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        gt_flow   [..., 1] = 1 - gt_flow[..., 1]
        gt_bm, gt_rec = self._use_flow_to_rec(dis_img=dis_img, flow=gt_flow)
        return bm, rec, gt_bm, gt_rec

    ### 我覺得先把 npy 轉成 npz 再來生圖比較好，不要在這邊 邊生圖 邊轉 npz，覺得的原因如下：
    ###     1.這樣這裡做的事情太多了~~
    ###     2.npy轉npz 我會把 npy刪掉，但這樣第二次執行時 self.see_npy_names 就會是空的，還要寫if來判斷何時讀 npy, npz ，覺得複雜~
    def _Draw_matplot_bm_rec_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img    = cv2.imread(self.see_jpg_paths[0])          ### 要記得see的jpg第一張存的是 輸入的in影像
        flow_v    = cv2.imread(self.see_epoch_jpg_paths[epoch])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        gt_flow_v = cv2.imread(self.see_jpg_paths[1])          ### 要記得see0的jpg第二張存的是 輸出的gt影像

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

    ### See_method 第三部分：主要做的事情在這裡
    def _draw_matplot_bm_rec_visual_after_train(self, start_epoch, epoch_amount, add_loss, bgr2rgb):
        """
        有可能畫完主圖 還要再畫 loss，所以多這個method，多做的事情都在這裡處理
        處理完後就 Save_fig 囉！
        """
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            single_row_imgs = self._Draw_matplot_bm_rec_visual(go_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)
            single_row_imgs.Save_fig(dst_dir=self.matplot_bm_rec_visual_write_dir, epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！


    def Save_as_matplot_bm_rec_visual(self,   ### 訓練後，可以走訪所有see_file 並重新產生 matplot_bm_rec_visual
                                      add_loss=False,
                                      bgr2rgb =False,
                                      single_see_core_amount=CORE_AMOUNT_BM_REC_VISUAL,
                                      see_print_msg=False):
        """
        save_as_matplot_bm_rec_visual(_after_train) 最後想試試看 省掉他 會不會影響我的理解
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_matplot_bm_rec_visual, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.matplot_bm_rec_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果
        Check_dir_exist_and_build_new_dir(self.bm_visual_write_dir)              ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果
        Check_dir_exist_and_build_new_dir(self.rec_visual_write_dir)             ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果

        ### See_method 第二部分：取得see資訊
        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.get_npz_info()

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_matplot_bm_rec_visual_after_train(0, self.see_file_amount, add_loss, bgr2rgb)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_file_amount, task=self._draw_matplot_bm_rec_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=see_print_msg)
            ### 後處理讓結果更小 但 又不失視覺品質，多核心版(core_amount 在 step0 裡調)
            Find_ltrd_and_crop (self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        video_processes = []
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.see_read_dir, self.see_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉###
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.rec_visual_write_dir, self.rec_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        for video_p in video_processes: video_p.start()
        for video_p in video_processes: video_p.join()

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.matplot_bm_rec_visual_write_dir != self.matplot_bm_rec_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.matplot_bm_rec_visual_write_dir, read_dir=self.matplot_bm_rec_visual_read_dir)
            Syn_write_to_read_dir(write_dir=self.bm_visual_write_dir,             read_dir=self.bm_visual_read_dir)
            Syn_write_to_read_dir(write_dir=self.rec_visual_write_dir,            read_dir=self.rec_visual_read_dir)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Save_as_matplot_bm_rec_visual, Current See:{self.see_name}, cost time:{time.time() - start_time}")

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    ### 好像都沒用到，先註解起來吧～再看看要不要刪掉
    # def Save_as_matplot_bm_rec_visual_at_certain_epoch(self, epoch, add_loss=False, bgr2rgb=False):   ### 訓練後，對"指定"epoch的 see結果 產生 matplot_bm_rec_visual
    #     print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_matplot_bm_rec_visual_at_certain_epoch, Current See:{self.see_name}, at_certain_epoch:{epoch}")
    #     start_time = time.time()
    #     Check_dir_exist_and_build(self.see_write_dir)
    #     Check_dir_exist_and_build(self.matplot_bm_rec_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
    #     Check_dir_exist_and_build(self.bm_visual_write_dir)              ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
    #     Check_dir_exist_and_build(self.rec_write_visual_dir)             ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
    #     self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量

    #     ### 防呆一下
    #     current_final_epoch = self.see_file_amount - 3   ### epochs是 epoch總數，要減掉：in_img, gt_img 和 epoch0
    #     if(epoch <= current_final_epoch):
    #         single_row_imgs = self._Draw_matplot_bm_rec_visual(epoch, add_loss, bgr2rgb)
    #         single_row_imgs.Save_fig(dst_dir=self.matplot_bm_rec_visual_write_dir, epoch=epoch)  ### 如果沒有要接續畫loss，就可以存了喔！
    #         ### 後處理讓結果更小 但 又不失視覺品質
    #         # Find_ltrd_and_crop(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 兩次以上有危險可能會 crop錯喔！所以就不crop了~
    #         Save_as_jpg(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], multiprocess=True, core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
    #         print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_matplot_bm_rec_visual_at_certain_epoch, Current See:{self.see_name}, at_certain_epoch:{epoch}, cost_time:{time.time() - start_time}")
    #     else:
    #         print("epoch=%i 超過目前exp的epoch數目囉！有可能是還沒train完see還沒產生到該epoch 或者 是輸入的epoch數 超過 epochs囉！" % epoch)
    #         print("Save_as_matplot_bm_rec_visual_at_certain_epoch不做事情拉~")

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

class See_npy_to_npz(See_info):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_bm_rec, self).__init__(result_read_dir, result_write_dir, see_name)

    def get_npz_info(self):
        self.see_npz_names            = get_dir_certain_file_name(self.see_read_dir, certain_word=".npz")
        self.see_npz_read_paths       = [self.see_read_dir  + "/" + npz_name for npz_name in self.see_npz_names]  ### 沒有 write_paths，因為有用 .replace() 所以用 see_npy_name.replace() 較保險這樣子！

        self.see_epoch_npz_names      = get_dir_certain_file_name(self.see_read_dir, certain_word="epoch", certain_ext=".npz")
        self.see_epoch_npz_read_paths = [self.see_read_dir + "/" + epoch_npz_name for epoch_npz_name in self.see_epoch_npz_names]    ### 沒有 write_paths，同上 ，既然已經沒有 self.see_npz_write_paths， 當然更不會有 self.see_epoch_npz_write_paths 拉！

        self.see_file_amount = len(self.see_epoch_npz_read_paths)

    def npy_to_npz_comapre(self):
        self.get_see_dir_info()

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

    def Npy_to_npz(self, single_see_core_amount=8, see_print_msg=False):   ### 因為有刪東西的動作，覺得不要multiprocess比較安全~~
        """
        把 See 資料夾內的.npy改存成.npz，存完會把.npy刪除喔～
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Npy_to_npz, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立資料夾：不能 build_new_dir，因為目前 npy, npz 是 和 visual 存在同個資料夾 沒分開！ build_new_dir 會把 visual 全刪光， 所以 下面有一行 os.remove 做完會把 .npy刪掉
        ###     且 目前是跟 see_dir 共用資料夾， 也不用build_dir 喔！

        ### See_method 第二部分：取得see資訊
        self.get_see_dir_info()


        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(len(self.see_npy_names) > 0):
            if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
                self._npy_to_npz(start_index=0, amount=len(self.see_npy_names))
            elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
                multi_processing_interface(core_amount=single_see_core_amount, task_amount=len(self.see_npy_names), task=self._npy_to_npz, print_msg=see_print_msg)
            else:
                print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理～沒事情就空白拉

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.see_write_dir != self.see_read_dir):  ### 因為接下去的任務需要 此任務的結果， 如果 read/write 資料夾位置不一樣， write完的結果 copy 一份 放回read， 才能讓接下去的動作 有 東西 read 喔！
            Syn_write_to_read_dir(write_dir=self.see_write_dir, read_dir=self.see_read_dir)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Npy_to_npz, Current See:{self.see_name}, cost time:{time.time() - start_time}")

    ### See_method 第三部分：主要做的事情在這裡
    def _npy_to_npz(self, start_index, amount):
        for see_npy_name in tqdm(self.see_npy_names[start_index:start_index + amount]):  ### 因為有用 .replace()， 對see_npy_name.replace() 較保險， 所以這邊用 see_npy_name 而不用 see_npy_path！
            npy = np.load(self.see_read_dir + "/" + see_npy_name)
            np.savez_compressed(self.see_write_dir + "/" + see_npy_name.replace(".npy", ".npz"), npy)
            os.remove(self.see_read_dir + "/" + see_npy_name)
            # print(self.see_read_dir + "/" + see_npy_name, "delete ok")
            # npz = np.load(self.see_read_dir + "/" + see_npy_name.replace(".npy", ".npz"))  ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
            # print((npy - npz["arr_0"]).sum())                                              ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
        ### 不要想 邊生圖 邊 npy轉npz了，原因寫在 _Draw_matplot_bm_rec_visual 上面



class See_rec_metric(See_bm_rec):
    """
    我把它繼承See_bm_rec 的概念是要做完 See_bm_rec 後才能做 See_rec_metric 喔！
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_rec_metric, self).__init__(result_read_dir, result_write_dir, see_name)

        self.matplot_metric_read_dir  = self.see_read_dir  + "/metric"
        self.matplot_metric_write_dir = self.see_write_dir + "/metric"
        self.matplot_metric_visual_read_dir  = self.see_read_dir  + "/matplot_metric_visual"
        self.matplot_metric_visual_write_dir = self.see_write_dir + "/matplot_metric_visual"
        self.metric_names = None
        self.metric_paths = None

    ###############################################################################################
    ###############################################################################################
    def get_metric_info(self):
        self.metric_names  = get_dir_certain_file_name(self.bm_visual_read_dir , certain_word="metric_epoch", certain_ext=".jpg")
        self.metric_paths  = [self.matplot_metric_visual_read_dir + "/" + name for name in self.metric_names]

        self.see_file_amount = len(self.metric_names)

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
        Check_dir_exist_and_build(self.matplot_metric_write_dir)  ### 不build new_dir 是因為 覺德 算一次的時間太長了ˊ口ˋ 怕不小心操作錯誤就要重算～

        ### See_method 第二部分：取得see資訊
        self.get_see_dir_info()  ### 暫時寫這邊，到時應該要拉出去到result_level，要不然每做一次就要重新更新一次，但不用這麼頻繁，只需要一開始更新一次即可
        self.get_bm_rec_info()   ### 暫時寫這邊，到時應該要拉出去到result_level，要不然每做一次就要重新更新一次，但不用這麼頻繁，只需要一開始更新一次即可

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        with Manager() as manager:  ### 設定在 multiprocess 裡面 共用的 list
            ### multiprocess 內的 global 的 list， share memory 的概念了，就算不multiprocess 也可以用喔！ 不過記得如果要在with外用， 要先轉回list() 就是了！
            SSIMs = manager.list()  # []的概念
            LDs   = manager.list()  # []的概念

            if  (single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
                self._do_matlab_SSIM_LD(0, self.see_file_amount, SSIMs, LDs)
            elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
                multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_file_amount, task=self._do_matlab_SSIM_LD, task_args=[SSIMs, LDs], print_msg=see_print_msg)
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
        np.save(f"{self.matplot_metric_write_dir}/SSIMs", SSIMs)
        np.save(f"{self.matplot_metric_write_dir}/LDs",   LDs)

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.matplot_metric_write_dir != self.matplot_metric_read_dir):  ### 因為接下去的任務需要 此任務的結果， 如果 read/write 資料夾位置不一樣， write完的結果 copy 一份 放回read， 才能讓接下去的動作 有 東西 read 喔！
            Syn_write_to_read_dir(write_dir=self.matplot_metric_write_dir, read_dir=self.matplot_metric_read_dir)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Calculate_SSIM_LD, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

    ### See_method 第三部分：主要做的事情在這裡
    def _do_matlab_SSIM_LD(self, start_epoch, epoch_amount, SSIMs, LDs):
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            path1 = self.rec_paths[go_epoch]  ### matplot_bm_rec_visual/rec_visual/rec_epoch=0000.jpg
            path2 = self.see_jpg_paths[2]     ### 0c-rec_hope.jpg
            # print(path1)
            # print(path2)

            ord_dir = os.getcwd()                            ### step1 紀錄 目前的主程式資料夾
            os.chdir("SIFT_dev/SIFTflow")                    ### step2 跳到 SIFTflow資料夾裡面
            [[SSIM, LD]] = use_DewarpNet_eval(path1, path2)  ### step3 執行 SIFTflow資料夾裡面 的 kong_use_evalUnwarp_sucess.use_DewarpNet_eval 來執行 kong_evalUnwarp_sucess.m
            os.chdir(ord_dir)                                ### step4 跳回 主程式資料夾

            # fig, ax = plt.subplots(nrows=1, ncols=2)
            # rec_img    = cv2.imread(path1)
            # rec_gt_img = cv2.imread(path2)
            # ax[0].imshow(rec_img)
            # ax[1].imshow(rec_gt_img)
            # plt.show()
            # plt.close()

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
        self.get_see_dir_info()
        self.get_bm_rec_info()
        SSIMs = np.load(f"{self.matplot_metric_read_dir}/SSIMs.npy")
        LDs   = np.load(f"{self.matplot_metric_read_dir}/LDs.npy")

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._visual_SSIM_LD(0, self.see_file_amount, SSIMs, LDs, add_loss=add_loss, bgr2rgb=bgr2rgb)
            Find_ltrd_and_crop     (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg            (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.see_file_amount, task=self._visual_SSIM_LD, task_args=[SSIMs, LDs, add_loss, bgr2rgb], print_msg=see_print_msg)
            Find_ltrd_and_crop     (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg            (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        Video_combine_from_dir (self.matplot_metric_visual_write_dir, self.matplot_metric_visual_write_dir)


        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.matplot_metric_visual_write_dir != self.matplot_metric_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.matplot_metric_visual_write_dir, read_dir=self.matplot_metric_visual_read_dir)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish _visual_SSIM_LD, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

    ### See_method 第三部分：主要做的事情在這裡
    def _visual_SSIM_LD(self, start_epoch, epoch_amount, SSIMs, LDs, add_loss=False, bgr2rgb=False):
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            path1 = self.rec_paths[go_epoch]  ### matplot_bm_rec_visual/rec_visual/rec_epoch=0000.jpg
            path2 = self.see_jpg_paths[2]     ### 0c-rec_hope.jpg
            SSIM = SSIMs[go_epoch]
            LD   = LDs  [go_epoch]

            in_img     = cv2.imread(self.see_jpg_paths[0])
            rec_img    = cv2.imread(path1)
            rec_gt_img = cv2.imread(path2)
            single_row_imgs = Matplot_single_row_imgs(
                        imgs      =[in_img,   rec_img ,   rec_gt_img],    ### 把要顯示的每張圖包成list
                        img_titles=[ "in_img", "rec"    , "gt_rec"],    ### 把每張圖要顯示的字包成list
                        fig_title ="epoch=%04i, SSIM=%.2f, LD=%.2f" % (go_epoch, SSIM, LD),   ### 圖上的大標題
                        add_loss  =add_loss,
                        bgr2rgb   =bgr2rgb)
            single_row_imgs.Draw_img()
            if(add_loss)   : single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.ax[-1, 1], self.matplot_metric_read_dir, go_epoch, min_epochs=self.see_file_amount, ylim=25)
            single_row_imgs.Save_fig(dst_dir=self.matplot_metric_visual_write_dir, epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！



class See(See_visual, See_npy_to_npz, See_rec_metric):
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
