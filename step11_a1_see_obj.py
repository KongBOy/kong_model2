from step0_access_path import JPG_QUALITY, CORE_AMOUNT

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name, Matplot_single_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from flow_bm_util import use_flow_to_get_bm, use_bm_to_rec_img
from video_from_img import Video_combine_from_dir

import cv2
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import pdb

class See_info:
    '''
    See 是 最直接、最基本 model output的東西，在各個model裡面都應該有寫 自己的 generate_see
    而這邊只是 讀取 training 過程中生成的 See 這樣子囉~~
    '''
    def __init__(self, result_dir, see_name):
        self.result_dir = result_dir
        self.see_name = see_name
        self.see_dir = self.result_dir + "/" + self.see_name

        self.see_jpg_names = None
        self.see_npy_names = None
        self.see_file_amount = None

    def get_see_dir_info(self):
        self.see_jpg_names = get_dir_certain_file_name(self.see_dir, ".jpg")
        self.see_npy_names = get_dir_certain_file_name(self.see_dir, ".npy")
        self.see_file_amount = len(self.see_jpg_names)
        # self.matplot_visual_dir = self.see_dir + "/matplot_visual"
        # Check_dir_exist_and_build(self.matplot_visual_dir)

    def save_as_jpg(self):  ### 後來看覺得好像有點多餘
        Check_dir_exist_and_build(self.see_dir)
        Save_as_jpg(self.see_dir, self.see_dir, delete_ord_file=True)

    def save_as_avi(self):  ### 後來看覺得好像有點多餘
        Check_dir_exist_and_build(self.see_dir)
        Video_combine_from_dir(self.see_dir, self.see_dir, "0-combine_jpg_tail_long.avi", tail_long=True)



class See_visual(See_info):
    """
    See_visual 是用來視覺化 See 的物件，因此這個Class我覺得也應該要設計成 training 中可以被使用的這樣子囉
      所以要看的東西就是簡單的：
        單純的input, 單純的output, 單純的gt
    """
    def __init__(self, result_dir, see_name):
        super(See_visual, self).__init__(result_dir, see_name)
        self.matplot_visual_dir = self.see_dir + "/matplot_visual"

        ### 不確定要不要，因為在initial就做這麼多事情好嗎~~會不會容易出錯哩~~
        ### 覺得還是不要比較好，要使用到的時候再建立，要不然有時候在analyze只是想要result_obj而已，結果又把see資料夾又重建了一次
        # Check_dir_exist_and_build(self.see_dir)
        # self.get_see_dir_info()   ### 好像只有在 analyze時會用到！所以用到的時候再抓就好囉！
        self.single_row_imgs_during_train = None  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！

    ###############################################################################################
    ###############################################################################################
    ### 主要做的事情，此fun會給 save_as_matplot_visual_during/after train 使用
    def _Draw_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img = cv2.imread(self.see_dir + "/" + self.see_jpg_names[0])       ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_dir + "/" + self.see_jpg_names[1])       ### 要記得see的第二張存的是 輸出的gt影像
        img = cv2.imread(self.see_dir + "/" + self.see_jpg_names[epoch + 2])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
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
        Check_dir_exist_and_build(self.matplot_visual_dir)
        start_time = time.time()
        # if(epoch==0):
        #     Check_dir_exist_and_build_new_dir(self.matplot_visual_dir)      ### 建立 存結果的資料夾
        self.get_see_dir_info()  ### 每次執行都要 update喔！ 取得result內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.single_row_imgs_during_train = self._Draw_matplot_visual(epoch, add_loss=True, bgr2rgb=bgr2rgb)  ### 要給train的step3畫loss，所以提升成see的attr才能讓外面存取囉！
        if(show_msg): print(f"processing {self.see_name}, cost_time:{time.time() - start_time}")

    ###############################################################################################
    def draw_loss_at_see_during_train(self, epoch, epochs):
        Check_dir_exist_and_build(self.matplot_visual_dir)  ### 以防matplot_visual資料夾被刪掉，要生圖找不到資料夾
        self.single_row_imgs_during_train.Draw_ax_loss_during_train(self.single_row_imgs_during_train.ax[-1, 1], self.see_dir + "/../logs", epoch, epochs )
        self.single_row_imgs_during_train.Save_fig(dst_dir=self.matplot_visual_dir, epoch=epoch)

    ###############################################################################################
    ###############################################################################################
    ### 訓練後，可以走訪所有see_file 並重新產生 matplot_visual
    def _draw_matplot_visual_after_train(self, start_img, img_amount, add_loss):
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 2):  ### 第三張 才開始存 epoch影像喔！
                epoch = go_img - 2  ### 第三張 開始才是 epoch影像喔！所以epoch的數字 是go_img-2
                single_row_imgs = self._Draw_matplot_visual(epoch, add_loss)
                if(add_loss)   : single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.ax[-1, 1], self.see_dir + "/../logs", epoch, min_epochs=self.see_file_amount - 2)  ### 如果要畫loss，去呼叫Draw_ax_loss 並輸入 ax 進去畫，還有最後面的參數，是輸入 epochs！所以要-2！
                single_row_imgs.Save_fig(dst_dir=self.matplot_visual_dir, epoch=epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    def _draw_matplot_visual_after_train_multiprocess(self, add_loss, core_amount=CORE_AMOUNT, task_amount=600, print_msg=False):  ### 以 see內的任務 當單位來切
        print("processing %s" % self.see_name)
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._draw_matplot_visual_after_train, task_args=[add_loss], print_msg=print_msg)


    def save_as_matplot_visual_after_train(self, add_loss=False, single_see_multiprocess=True, print_msg=False):  ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時(像是result裡面的save_all_single_see_as_matplot_visual_multiprocess)，傳參數時這要設為false喔！
        print(f"doing {self.see_name} save_as_matplot_visual_after_train")
        start_time = time.time()
        # matplot_visual_dir = self.see_dir + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(self.see_dir)
        Check_dir_exist_and_build_new_dir(self.matplot_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        if(single_see_multiprocess): self._draw_matplot_visual_after_train_multiprocess(add_loss, core_amount=CORE_AMOUNT, task_amount=self.see_file_amount, print_msg=print_msg)  ### 以 see內的任務 當單位來切，task_amount輸入self.see_file_amount是對的！不用-2變epoch喔！
        else: self._draw_matplot_visual_after_train(0, self.see_file_amount, add_loss)

        ### 後處理讓結果更小 但 又不失視覺品質
        Find_ltrd_and_crop(self.matplot_visual_dir, self.matplot_visual_dir, padding=15, search_amount=10)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(self.matplot_visual_dir, self.matplot_visual_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])  ### matplot圖存完是png，改存成jpg省空間
        # Video_combine_from_dir(self.matplot_visual_dir, self.matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        print("cost_time:", time.time() - start_time)
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
    def __init__(self, result_dir, see_name):
        super(See_bm_rec, self).__init__(result_dir, see_name)

        self.matplot_bm_rec_visual_dir = self.see_dir + "/matplot_bm_rec_visual"
        self.bm_visual_dir             = self.see_dir + "/matplot_bm_rec_visual/bm_visual"
        self.rec_visual_dir            = self.see_dir + "/matplot_bm_rec_visual/rec_visual"
        self.bm_names  = None
        self.bm_paths  = None
        self.rec_names = None
        self.rec_paths = None

    ###############################################################################################
    ###############################################################################################
    def get_bm_rec_info(self):
        self.bm_names  = get_dir_certain_file_name(self.bm_visual_dir , ".jpg")
        self.bm_paths  = [self.bm_visual_dir + "/" + name for name in self.bm_names]
        self.rec_names = get_dir_certain_file_name(self.rec_visual_dir, ".jpg")
        self.rec_paths = [self.rec_visual_dir + "/" + name for name in self.rec_names]

        self.see_file_amount = len(self.rec_names)
        # self.matplot_visual_dir = self.see_dir + "/matplot_visual"
        # Check_dir_exist_and_build(self.matplot_visual_dir)

    ###############################################################################################
    ###############################################################################################
    def _Draw_matplot_bm_rec_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img    = cv2.imread(self.see_dir + "/" + self.see_jpg_names[0])          ### 要記得see的第一張存的是 輸入的in影像
        gt_flow_v = cv2.imread(self.see_dir + "/" + self.see_jpg_names[1])          ### 要記得see的第二張存的是 輸出的gt影像
        flow_v    = cv2.imread(self.see_dir + "/" + self.see_jpg_names[epoch + 2])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        gt_flow   = np.load(self.see_dir + "/" + self.see_npy_names[0])
        flow      = np.load(self.see_dir + "/" + self.see_npy_names[epoch + 1])
        # breakpoint()
        bm  = use_flow_to_get_bm(flow, flow_scale=768)
        rec = use_bm_to_rec_img(bm, flow_scale=768, dis_img=in_img)
        if(gt_flow.sum() > 0):
            gt_bm  = use_flow_to_get_bm(gt_flow, flow_scale=768)
            gt_rec = use_bm_to_rec_img(gt_bm, flow_scale=768, dis_img=in_img)
        else:
            gt_bm  = np.zeros(shape=(768, 768, 2))
            gt_rec = np.zeros(shape=(768, 768, 3))

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
        if(epoch <= 3): cv2.imwrite(self.rec_visual_dir + "/" + "rec_gt.jpg", gt_rec)  ### 存大張gt，gt只要存一次即可，所以加個if這樣子，<=3是因為 bm_rec 懶的寫防呆 是從 第四個epoch才開始做~~，要不然epoch==2 就行囉！，所以目前gt會存兩次拉但時間應該多一咪咪而以先這樣吧~~
        cv2.imwrite(self.rec_visual_dir + "/" + "rec_epoch=%04i.jpg" % epoch, rec)     ### 存大張rec
        return single_row_imgs

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def _draw_matplot_bm_rec_visual_after_train(self, start_img, img_amount, add_loss, bgr2rgb):
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 3):        ### 第四張 才開始存 epoch影像喔！相當於epoch1才開始存，因為epoch0太差了沒寫防呆會出錯，目前乾脆先直接跳過有空再寫防呆。
                epoch = go_img - 2  ### 第三張 開始才是 epoch影像喔！所以epoch的數字 是go_img-2
                single_row_imgs = self._Draw_matplot_bm_rec_visual(epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)
                single_row_imgs.Save_fig(dst_dir=self.matplot_bm_rec_visual_dir, epoch=epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    def _draw_matplot_bm_rec_visual_after_train_multiprocess(self, add_loss, bgr2rgb, core_amount=CORE_AMOUNT, task_amount=600, print_msg=False):  ### 以 see內的任務 當單位來切
        print("processing %s" % self.see_name)
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._draw_matplot_bm_rec_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=print_msg)

    def save_as_matplot_bm_rec_visual_after_train(self,   ### 訓練後，可以走訪所有see_file 並重新產生 matplot_bm_rec_visual
                                           add_loss=False,
                                           bgr2rgb =False,
                                           single_see_multiprocess=True,
                                           print_msg=False):  ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時(像是result裡面的save_all_single_see_as_matplot_bm_rec_visual_multiprocess)，傳參數時這要設為false喔！
        print(f"doing {self.see_name} save_as_matplot_bm_rec_visual_after_train")
        start_time = time.time()
        # matplot_bm_rec_visual_dir = self.see_dir + "/" + "matplot_bm_rec_visual" ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(self.see_dir)
        Check_dir_exist_and_build_new_dir(self.matplot_bm_rec_visual_dir)      ### 建立 存結果的資料夾
        Check_dir_exist_and_build_new_dir(self.bm_visual_dir)      ### 建立 存結果的資料夾
        Check_dir_exist_and_build_new_dir(self.rec_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        if(single_see_multiprocess): self._draw_matplot_bm_rec_visual_after_train_multiprocess(add_loss, bgr2rgb, core_amount=CORE_AMOUNT, task_amount=self.see_file_amount, print_msg=print_msg)  ### see內的任務 當單位來切，task_amount輸入self.see_file_amount是對的！不用-2變epoch喔！
        else: self._draw_matplot_bm_rec_visual_after_train(0, self.see_file_amount, add_loss, bgr2rgb)

        ### 後處理讓結果更小 但 又不失視覺品質
        Find_ltrd_and_crop(self.matplot_bm_rec_visual_dir, self.matplot_bm_rec_visual_dir, padding=15, search_amount=10)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(self.matplot_bm_rec_visual_dir, self.matplot_bm_rec_visual_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])  ### matplot圖存完是png，改存成jpg省空間
        # Video_combine_from_dir(self.matplot_bm_rec_visual_dir, self.matplot_bm_rec_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        print("cost_time:", time.time() - start_time)
    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
class See(See_visual, See_bm_rec):
    def __init__(self, result_dir, see_name):
        super(See, self).__init__(result_dir, see_name)
