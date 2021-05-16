from step0_access_path import JPG_QUALITY, CORE_AMOUNT, CORE_AMOUNT_NPY_TO_NPZ, CORE_AMOUNT_BM_REC_VISUAL, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name, Matplot_single_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from flow_bm_util import use_flow_to_get_bm, use_bm_to_rec_img
from video_from_img import Video_combine_from_dir
from multiprocess_util import multi_processing_interface
from multiprocessing import Process

import cv2
import time
import numpy as np
from tqdm import tqdm
import os


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
        self.see_dir = self.result_read_dir + "/" + self.see_name
        self.see_write_dir = self.result_write_dir + "/" + self.see_name

        self.see_jpg_names = None
        self.see_npy_names = None
        self.see_file_amount = None

        self.in_use_range = "0~1"
        self.gt_use_range = "0~1"

    def get_see_dir_info(self):
        self.see_jpg_names = get_dir_certain_file_name(self.see_dir, ".jpg")
        self.see_npy_names = get_dir_certain_file_name(self.see_dir, ".npy")
        self.see_npz_names = get_dir_certain_file_name(self.see_dir, ".npz")
        self.see_jpg_paths = [self.see_dir + "/" + jpg_name for jpg_name in self.see_jpg_names]
        self.see_npy_paths = [self.see_dir + "/" + npy_name for npy_name in self.see_npy_names]
        self.see_npz_paths = [self.see_dir + "/" + npz_name for npz_name in self.see_npz_names]
        self.see_file_amount = len(self.see_jpg_names)

    def save_as_jpg(self):  ### 後來看覺得好像有點多餘
        Check_dir_exist_and_build(self.see_write_dir)
        Save_as_jpg(self.see_write_dir, self.see_write_dir, delete_ord_file=True)

    def save_as_avi(self):  ### 後來看覺得好像有點多餘
        Check_dir_exist_and_build(self.see_write_dir)
        Video_combine_from_dir(self.see_dir, self.see_write_dir, "0-combine_jpg_tail_long.avi", tail_long=True)



class See_visual(See_info):
    """
    See_visual 是用來視覺化 See 的物件，因此這個Class我覺得也應該要設計成 training 中可以被使用的這樣子囉
      所以要看的東西就是簡單的：
        單純的input, 單純的output, 單純的gt
    """
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_visual, self).__init__(result_read_dir, result_write_dir, see_name)
        self.matplot_visual_dir = self.see_write_dir + "/matplot_visual"

        ### 不確定要不要，因為在initial就做這麼多事情好嗎~~會不會容易出錯哩~~
        ### 覺得還是不要比較好，要使用到的時候再建立，要不然有時候在analyze只是想要result_obj而已，結果又把see資料夾又重建了一次
        # Check_dir_exist_and_build(self.see_write_dir)
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
        if(show_msg): print(f"See level: doing save_as_matplot_visual_during_train, Current See:{self.see_name}, cost_time:{time.time() - start_time}")

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
                current_epoch = go_img - 2  ### 第三張 開始才是 epoch影像喔！所以epoch的數字 是go_img-2
                single_row_imgs = self._Draw_matplot_visual(current_epoch, add_loss)
                if(add_loss)   : single_row_imgs.Draw_ax_loss_after_train(single_row_imgs.ax[-1, 1], self.see_dir + "/../logs", current_epoch, min_epochs=self.see_file_amount - 2)  ### 如果要畫loss，去呼叫Draw_ax_loss 並輸入 ax 進去畫，還有最後面的參數，是輸入 epochs！所以要-2！
                single_row_imgs.Save_fig(dst_dir=self.matplot_visual_dir, epoch=current_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    def _draw_matplot_visual_after_train_multiprocess(self, add_loss, core_amount=CORE_AMOUNT, task_amount=600, print_msg=False):  ### 以 see內的任務 當單位來切
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._draw_matplot_visual_after_train, task_args=[add_loss], print_msg=print_msg)


    def save_as_matplot_visual_after_train(self, add_loss=False, single_see_multiprocess=True, print_msg=False):  ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時(像是result裡面的save_all_single_see_as_matplot_visual_multiprocess)，傳參數時這要設為false喔！
        print(f"See level: doing save_as_matplot_visual_after_train, Current See:{self.see_name}")
        start_time = time.time()
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.matplot_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        if(single_see_multiprocess):
            ### see內的任務 有切 multiprocess
            self._draw_matplot_visual_after_train_multiprocess(add_loss, core_amount=CORE_AMOUNT, task_amount=self.see_file_amount, print_msg=print_msg)  ### 以 see內的任務 當單位來切，task_amount輸入self.see_file_amount是對的！不用-2變epoch喔！
            ### 後處理讓結果更小 但 又不失視覺品質
            Find_ltrd_and_crop(self.matplot_visual_dir, self.matplot_visual_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg(self.matplot_visual_dir, self.matplot_visual_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:  ### see內的任務 不切 multiprocess，和上面幾乎一樣，只差 call 沒 multiprocess 的 method 和 core_amount 指定1
            self._draw_matplot_visual_after_train(0, self.see_file_amount, add_loss)
            ### 後處理讓結果更小 但 又不失視覺品質
            Find_ltrd_and_crop(self.matplot_visual_dir, self.matplot_visual_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg(self.matplot_visual_dir, self.matplot_visual_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間

        # Video_combine_from_dir(self.matplot_visual_dir, self.matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        print(f"See level: doing save_as_matplot_visual_after_train, Current See:{self.see_name}, cost_time:{time.time() - start_time}")
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

        self.matplot_bm_rec_visual_read_dir   = self.see_dir       + "/matplot_bm_rec_visual"
        self.matplot_bm_rec_visual_write_dir  = self.see_write_dir + "/matplot_bm_rec_visual"
        self.bm_visual_read_dir               = self.see_dir       + "/matplot_bm_rec_visual/bm_visual"
        self.bm_visual_write_dir              = self.see_write_dir + "/matplot_bm_rec_visual/bm_visual"
        self.rec_visual_read_dir              = self.see_dir       + "/matplot_bm_rec_visual/rec_visual"
        self.rec_visual_write_dir             = self.see_write_dir + "/matplot_bm_rec_visual/rec_visual"
        self.bm_names  = None
        self.bm_paths  = None
        self.rec_names = None
        self.rec_paths = None

    ###############################################################################################
    ###############################################################################################
    def get_bm_rec_info(self):
        self.bm_names  = get_dir_certain_file_name(self.bm_visual_read_dir , ".jpg")
        self.bm_paths  = [self.bm_visual_read_dir + "/" + name for name in self.bm_names]
        self.rec_names = get_dir_certain_file_name(self.rec_visual_read_dir, ".jpg")
        self.rec_paths = [self.rec_visual_read_dir + "/" + name for name in self.rec_names]

        self.see_file_amount = len(self.rec_names)

    ###############################################################################################
    ###############################################################################################
    ### 我覺得先把 npy 轉成 npz 再來生圖比較好，不要在這邊 邊生圖 邊轉 npz，覺得的原因如下：
    ###     1.這樣這裡做的事情太多了~~
    ###     2.npy轉npz 我會把 npy刪掉，但這樣第二次執行時 self.see_npy_names 就會是空的，還要寫if來判斷何時讀 npy, npz ，覺得複雜~
    def _Draw_matplot_bm_rec_visual(self, epoch, add_loss=False, bgr2rgb=False):
        in_img    = cv2.imread(self.see_dir + "/" + self.see_jpg_names[0])          ### 要記得see的jpg第一張存的是 輸入的in影像
        gt_flow_v = cv2.imread(self.see_dir + "/" + self.see_jpg_names[1])          ### 要記得see0的jpg第二張存的是 輸出的gt影像
        flow_v    = cv2.imread(self.see_dir + "/" + self.see_jpg_names[epoch + 2])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        gt_flow   = np.load(self.see_dir + "/" + self.see_npz_names[0])["arr_0"]          ### 要記得see的npz 第一張存的是 gt_flow 喔！   ，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        flow      = np.load(self.see_dir + "/" + self.see_npz_names[epoch + 1])["arr_0"]  ### see資料夾 內的flow 該epoch產生的flow 讀出來，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        # gt_flow[..., 1] = 1 - gt_flow[..., 1]
        flow      [..., 1] = 1 - flow[..., 1]
        gt_flow   [..., 1] = 1 - gt_flow[..., 1]

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=1, ncols=4)
        # print(gt_flow.min())
        # print(gt_flow.max())
        # ax[0].imshow(gt_flow[..., 0])
        # ax[1].imshow(gt_flow[..., 1])
        # ax[2].imshow(gt_flow_y_inv[..., 0])
        # ax[3].imshow(gt_flow_y_inv[..., 1])
        # plt.show()
        # print("2. see gt_use_range=", self.gt_use_range)
        if(self.gt_use_range == "-1~1"): flow = (flow + 1) / 2   ### 如果 gt_use_range 是 -1~1 記得轉回 0~1

        # breakpoint()

        ### predict flow part
        valid_mask_pix_amount = (flow[..., 0] >= 0.99).astype(np.int).sum()
        total_pix_amount = flow.shape[0] * flow.shape[1]
        # print("valid_mask_pix_amount / total_pix_amount:", valid_mask_pix_amount / total_pix_amount)
        if( valid_mask_pix_amount / total_pix_amount > 0.28):
            bm  = use_flow_to_get_bm(flow, flow_scale=768)
            rec = use_bm_to_rec_img(bm, flow_scale=768, dis_img=in_img)
        else:
            bm  = np.zeros(shape=(768, 768, 2))
            rec = np.zeros(shape=(768, 768, 3))


        # import matplotlib.pyplot as plt
        # gt_valid_mask_pix_amount = (gt_flow[..., 0] >= 0.99).astype(np.int).sum()
        # gt_total_pix_amount = gt_flow.shape[0] * flow.shape[1]
        # print("gt_valid_mask_pix_amount", gt_valid_mask_pix_amount)
        # print("gt_total_pix_amount", gt_total_pix_amount)
        # print("gt_valid_mask_pix_amount / gt_total_pix_amount:", gt_valid_mask_pix_amount / gt_total_pix_amount)
        # plt.imshow(gt_flow)
        # plt.show()
        if(gt_flow[..., 0].sum() > 0):
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
        if(epoch <= 3): cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_gt.jpg", gt_rec)  ### 存大張gt，gt只要存一次即可，所以加個if這樣子，<=3是因為 bm_rec 懶的寫防呆 是從 第四個epoch才開始做~~，要不然epoch==2 就行囉！，所以目前gt會存兩次拉但時間應該多一咪咪而以先這樣吧~~
        cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_epoch=%04i.jpg" % epoch, rec)     ### 存大張rec

        # del in_img
        # del gt_flow_v
        # del flow_v
        # del gt_flow
        # del bm
        # del rec
        # del gt_bm
        # del gt_rec
        return single_row_imgs

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################
    def _draw_matplot_bm_rec_visual_after_train(self, start_img, img_amount, add_loss, bgr2rgb):
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 2):        ### 已經有用msdk寫防呆了，所以可以從 第三張開始做囉！
                current_epoch = go_img - 2  ### 第三張 開始才是 epoch影像喔！所以epoch的數字 是go_img-2
                single_row_imgs = self._Draw_matplot_bm_rec_visual(current_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)
                single_row_imgs.Save_fig(dst_dir=self.matplot_bm_rec_visual_write_dir, epoch=current_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    def _draw_matplot_bm_rec_visual_after_train_multiprocess(self, add_loss, bgr2rgb, core_amount=CORE_AMOUNT_BM_REC_VISUAL, task_amount=600, print_msg=False):  ### 以 see內的任務 當單位來切
        start_time = time.time()
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._draw_matplot_bm_rec_visual_after_train, task_args=[add_loss, bgr2rgb], print_msg=print_msg)
        print("draw_matplot_bm_rec_visual_after_train cost_time:", time.time() - start_time)

    def save_as_matplot_bm_rec_visual_after_train(self,   ### 訓練後，可以走訪所有see_file 並重新產生 matplot_bm_rec_visual
                                           add_loss=False,
                                           bgr2rgb =False,
                                           single_see_multiprocess=True,
                                           print_msg=False):  ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時(像是result裡面的save_all_single_see_as_matplot_bm_rec_visual_multiprocess)，傳參數時這要設為false喔！
        print(f"See level: doing save_as_matplot_bm_rec_visual_after_train, Current See:{self.see_name}")
        start_time = time.time()
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.matplot_bm_rec_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果
        Check_dir_exist_and_build_new_dir(self.bm_visual_write_dir)              ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果
        Check_dir_exist_and_build_new_dir(self.rec_visual_write_dir)             ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果

        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        if(single_see_multiprocess):
            ### see內的任務 有切 multiprocess
            self._draw_matplot_bm_rec_visual_after_train_multiprocess(add_loss, bgr2rgb, core_amount=CORE_AMOUNT_BM_REC_VISUAL, task_amount=self.see_file_amount, print_msg=print_msg)  ### see內的任務 當單位來切，task_amount輸入self.see_file_amount是對的！不用-2變epoch喔！
            ### 後處理讓結果更小 但 又不失視覺品質
            Find_ltrd_and_crop(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            ### see內的任務 不切 multiprocess，和上面幾乎一樣，只差 call 沒 multiprocess 的 method 和 core_amount 指定1
            self._draw_matplot_bm_rec_visual_after_train(0, self.see_file_amount, add_loss, bgr2rgb)
            ### 後處理讓結果更小 但 又不失視覺品質
            Find_ltrd_and_crop(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間

        ### 存 video
        video_processes = []
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.see_dir, self.see_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉###
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.rec_visual_write_dir, self.rec_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        for video_p in video_processes: video_p.start()
        for video_p in video_processes: video_p.join()

        print(f"See level: doing save_as_matplot_bm_rec_visual_after_train, Current See:{self.see_name}, cost time:{time.time() - start_time}")

    def save_as_matplot_bm_rec_visual_after_train_at_certain_epoch(self, epoch, add_loss=False, bgr2rgb=False):   ### 訓練後，對"指定"epoch的 see結果 產生 matplot_bm_rec_visual
        print(f"See level: doing save_as_matplot_bm_rec_visual_after_train_at_certain_epoch, Current See:{self.see_name}, at_certain_epoch:{epoch}")
        start_time = time.time()
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build(self.matplot_bm_rec_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
        Check_dir_exist_and_build(self.bm_visual_write_dir)              ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
        Check_dir_exist_and_build(self.rec_write_visual_dir)             ### 建立 存結果的資料夾，如果存在 也不需要刪掉重建喔，執行這個通常都是某個epoch有問題想重建，所以不需要把其他epoch的東西也刪掉這樣子
        self.get_see_dir_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量

        ### 防呆一下
        current_final_epoch = self.see_file_amount - 3   ### epochs是 epoch總數，要減掉：in_img, gt_img 和 epoch0
        if(epoch <= current_final_epoch):
            single_row_imgs = self._Draw_matplot_bm_rec_visual(epoch, add_loss, bgr2rgb)
            single_row_imgs.Save_fig(dst_dir=self.matplot_bm_rec_visual_write_dir, epoch=epoch)  ### 如果沒有要接續畫loss，就可以存了喔！
            ### 後處理讓結果更小 但 又不失視覺品質
            # Find_ltrd_and_crop(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 兩次以上有危險可能會 crop錯喔！所以就不crop了~
            Save_as_jpg(self.matplot_bm_rec_visual_write_dir, self.matplot_bm_rec_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], multiprocess=True, core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
            print(f"See level: doing save_as_matplot_bm_rec_visual_after_train_at_certain_epoch, Current See:{self.see_name}, at_certain_epoch:{epoch}, cost_time:{time.time() - start_time}")
        else:
            print("epoch=%i 超過目前exp的epoch數目囉！有可能是還沒train完see還沒產生到該epoch 或者 是輸入的epoch數 超過 epochs囉！" % epoch)
            print("save_as_matplot_bm_rec_visual_after_train_at_certain_epoch不做事情拉~")

    ###############################################################################################
    ###############################################################################################
    ###############################################################################################

class See_try_npy_to_npz(See_info):
    def npy_to_npz_comapre(self):
        self.get_see_dir_info()

        ### load_3_load_50_npy
        start_time = time.time()
        for go_name, see_npy_name in enumerate(self.see_npy_names):
            np.load(self.see_dir + "/" + see_npy_name)   ### 344 MB
        load_3_time = time.time() - start_time
        print("load_3_load_50_npy ok")

        ### save_3_save_50_npy
        npys = []
        for go_name, see_npy_name in enumerate(self.see_npy_names):
            npys.append(np.load(self.see_dir + "/" + see_npy_name))   ### 344 MB
        start_time = time.time()
        for go_name, npy in enumerate(npys):
            np.save(self.see_write_dir + "/" + self.see_npy_names[go_name], npy)
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
            np.load(self.see_dir + "/" + see_npy_name.replace(".npy", ".npz"))   ### 344 MB
        load_2_time = time.time() - start_time
        print("load_2_load_50_npz ok")

        ### save_1_save_1_npz_conatin_50npy
        start_time = time.time()
        np.savez_compressed(self.see_write_dir + "/" + "000_try_npz", np.array(npys))
        save_1_time = time.time() - start_time
        print("save_1_save_1_npz_conatin_50npy ok")

        ### load_1_load_1_npz_conatin_50npy
        start_time = time.time()
        big_npz = np.load(self.see_dir + "/" + "000_try_npz.npz")
        print(big_npz["arr_0"].shape)   ### 小心！要有使用他，才會真的去load資料喔！
        load_1_time = time.time() - start_time
        print("load_1_load_1_npz_conatin_50npy ok")
        print("")
        os.remove(self.see_dir + "/" + "000_try_npz.npz")  ### 只是用來看讀取寫入速度而已，沒有真的要用，所以測試完後記得要刪掉喔！


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

    ### 不要想 邊生圖 邊 npy轉npz了，原因寫在 _Draw_matplot_bm_rec_visual 上面
    # def single_npy_to_npz_by_path(self, npy_path):
    #     npy = np.load(npy_path)
    #     np.savez_compressed(npy_path.replace(".npy", ".npz"), npy)
    #     os.remove(self.see_dir + "/" + npy_path)
    def _npy_to_npz(self, start_index, amount):
        for see_npy_name in tqdm(self.see_npy_names[start_index:start_index + amount]):
            npy = np.load(self.see_dir + "/" + see_npy_name)
            np.savez_compressed(self.see_write_dir + "/" + see_npy_name.replace(".npy", ".npz"), npy)
            os.remove(self.see_dir + "/" + see_npy_name)
            # print(self.see_dir + "/" + see_npy_name, "delete ok")
            # npz = np.load(self.see_dir + "/" + see_npy_name.replace(".npy", ".npz"))  ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
            # print((npy - npz["arr_0"]).sum())                                         ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！

    def _npy_to_npz_multiprocess(self, core_amount=CORE_AMOUNT_NPY_TO_NPZ, task_amount=600, print_msg=False):
        print("processing %s" % self.see_name)
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._npy_to_npz, print_msg=print_msg)


    def all_npy_to_npz(self, multiprocess=False, print_msg=False):   ### 因為有刪東西的動作，覺得不要multiprocess比較安全~~
        """
        把 See 資料夾內的.npy改存成.npz，存完會把.npy刪除喔～
        """
        print(f"See level: doing all_npy_to_npz, Current See:{self.see_name}")
        start_time = time.time()
        self.get_see_dir_info()
        if(len(self.see_npy_names) > 0):
            if(multiprocess):
                self._npy_to_npz_multiprocess(core_amount=CORE_AMOUNT_NPY_TO_NPZ, task_amount=len(self.see_npy_names))
            else:
                self._npy_to_npz(start_index=0, amount=len(self.see_npy_names))

                # for see_npy_name in tqdm(self.see_npy_names):
                #     npy = np.load(self.see_dir + "/" + see_npy_name)
                #     np.savez_compressed(self.see_write_dir + "/" + see_npy_name.replace(".npy", ".npz"), npy)
                #     os.remove(self.see_dir + "/" + see_npy_name)
        print(f"See level: doing all_npy_to_npz, Current See:{self.see_name}, cost time:{time.time() - start_time}")




class See(See_visual, See_bm_rec, See_try_npy_to_npz):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See, self).__init__(result_read_dir, result_write_dir, see_name)


if(__name__ == "__main__"):
    from step0_access_path import result_read_path
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_3b_4-20210306_231628-flow_unet-ch32_bn_16", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_005-train")
    # try_npy_to_npz.npy_to_npz_comapre()
    # try_npy_to_npz.all_npy_to_npz(multiprocess=True)
    # try_npy_to_npz.all_npy_to_npz(multiprocess=True)
