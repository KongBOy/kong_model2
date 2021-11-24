from step11_a0_see_base import See_info

from step0_access_path import JPG_QUALITY, CORE_AMOUNT_BM_REC_VISUAL, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_names, move_dir_certain_file
from matplot_fig_ax_util import Matplot_single_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from flow_bm_util import use_flow_to_get_bm, use_bm_to_rec_img
from video_from_img import Video_combine_from_dir
from multiprocess_util import multi_processing_interface
from multiprocessing import Process

import cv2
import time
import numpy as np
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
        self.bm_names  = get_dir_certain_file_names(self.bm_visual_read_dir , certain_word="bm_epoch", certain_ext=".jpg")
        self.bm_read_paths  = [self.bm_visual_read_dir + "/" + name for name in self.bm_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.rec_names = get_dir_certain_file_names(self.rec_visual_read_dir, certain_word="rec_epoch", certain_ext=".jpg")
        self.rec_read_paths = [self.rec_visual_read_dir + "/" + name for name in self.rec_names]  ### 沒有 write_path， 因為 bm_rec 只需要指定 write_dir 即可寫入資料夾

        self.rec_gt_name = get_dir_certain_file_names(self.rec_visual_read_dir, certain_word="gt", certain_ext=".jpg")
        self.rec_gt_path = self.rec_visual_read_dir + "/" + self.rec_gt_name[0]  ### 沒有 write_path， 因為 bm_rec 只需要指定 write_dir 即可寫入資料夾

        self.see_rec_amount = len(self.rec_read_paths)
        self.trained_epoch  = self.see_rec_amount - 1  ### 去掉epoch0

    ###############################################################################################
    ###############################################################################################
    def Save_as_bm_rec_matplot_visual(self,   ### 訓練後，可以走訪所有see_file 並重新產生 bm_rec_matplot_visual
                                      add_loss=False,
                                      bgr2rgb =False,
                                      single_see_core_amount=CORE_AMOUNT_BM_REC_VISUAL,
                                      see_print_msg=False,
                                      jump_to=0):
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
        self.get_flow_info()
        # print("here~~~~~~~~~~~~~~~~~~~~~", self.npz_epoch_amount)

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_bm_rec_matplot_visual_after_train(0, self.npz_epoch_amount, add_loss, bgr2rgb, jump_to)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.bm_rec_matplot_visual_write_dir, self.bm_rec_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.npz_epoch_amount, task=self._draw_bm_rec_matplot_visual_after_train, task_args=[add_loss, bgr2rgb, jump_to], print_msg=see_print_msg)
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
    def _draw_bm_rec_matplot_visual_after_train(self, start_epoch, epoch_amount, add_loss, bgr2rgb, jump_to):
        """
        有可能畫完主圖 還要再畫 loss，所以多這個method，多做的事情都在這裡處理
        處理完後就 Save_fig 囉！
        """
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            if(go_epoch < jump_to): continue
            # print("current go_epoch:", go_epoch, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            single_row_imgs = self._Draw_bm_rec_matplot_visual(go_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb, jump_to=jump_to)
            single_row_imgs.Save_fig(dst_dir=self.bm_rec_matplot_visual_write_dir, epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    ### See_method 第三部分a
    ###     我覺得先把 npy 轉成 npz 再來生圖比較好，不要在這邊 邊生圖 邊轉 npz，覺得的原因如下：
    ###         1.這樣這裡做的事情太多了~~
    ###         2.npy轉npz 我會把 npy刪掉，但這樣第二次執行時 self.npy_names 就會是空的，還要寫if來判斷何時讀 npy, npz ，覺得複雜~
    def _Draw_bm_rec_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False, jump_to=0):
        in_img    = cv2.imread(self.in_img_path)                    ### 要記得see的jpg第一張存的是 輸入的in影像
        flow_v    = cv2.imread(self.flow_ep_jpg_read_paths[epoch])  ### see資料夾 內的影像 該epoch產生的影像 讀出來
        gt_flow_v = cv2.imread(self.gt_flow_jpg_path )              ### 要記得see0的jpg第二張存的是 輸出的gt影像

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
        if(epoch <= jump_to): cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_gt.jpg", gt_rec)  ### 存大張gt，gt只要存一次即可，所以加個if這樣子，<=3是因為 bm_rec 懶的寫防呆 是從 第四個epoch才開始做~~，要不然epoch==2 就行囉！，所以目前gt會存兩次拉但時間應該多一咪咪而以先這樣吧~~
        # cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_gt.jpg", gt_rec)                  ### 存大張gt，雖然只要 gt只要存一次即可 同上可以寫個if只存一次， 但也沒省多少時間， 都存算了！ 以防 哪天從中途的epoch開始跑結果沒存到 rec_gt
        cv2.imwrite(self.rec_visual_write_dir + "/" + "rec_epoch=%04i.jpg" % epoch, rec)     ### 存大張rec

        return single_row_imgs

    ### See_method 第三部分b
    def _get_bm_rec_and_gt_bm_gt_rec(self, epoch, dis_img):
        ### pred flow part
        flow          = np.load(self.npz_epoch_read_paths[epoch])["arr_0"]  ### see資料夾 內的flow 該epoch產生的flow 讀出來，npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        flow [..., 1] = 1 - flow[..., 1]
        bm, rec = self._use_flow_to_rec(dis_img=dis_img, flow=flow)
        # print("dis_img.max():", dis_img.max())
        # print("flow.max():", flow.max())
        # print("rec.max():", rec.max())
        # print("bm.max():", bm.max())

        # from step08_b_use_G_generate import flow_or_coord_visual_op
        # y_coord = (flow[..., 1] * 255.).astype(np.uint8)
        # x_coord = (flow[..., 2] * 255.).astype(np.uint8)
        # bm_visual = flow_or_coord_visual_op(bm)
        # flow_visual = flow_or_coord_visual_op(flow)
        # cv2.imshow("y_coord", y_coord)
        # cv2.imshow("x_coord", x_coord)
        # cv2.imshow("flow_visual", flow_visual.astype(np.uint8))
        # cv2.imshow("bm_visual", bm_visual.astype(np.uint8))
        # cv2.waitKey()
        # '''
        # dis_img.max(): 255
        # flow.max(): 1.0
        # rec.max(): 0
        # bm.max(): 0.81640625
        # '''
        ### gt flow part
        gt_flow            = np.load(self.flow_gt_npz_path)["arr_0"]       ### npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        if("real" in self.see_name):  ### 因為 see-real 沒有gt_flow， 本來全黑沒問題， 不過我後來有玩 fake_see 不是全黑就會出問題， 所以乾脆 see-real 就不要處理
            h, w = gt_flow.shape[:2]
            gt_bm  = np.zeros(shape=(h, w, 2))
            gt_rec = np.zeros(shape=(h, w, 3))
        else:
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
        if( valid_mask_pix_amount / total_pix_amount > 0.20):
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
