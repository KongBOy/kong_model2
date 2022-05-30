from step11_a0_see_base import See_info

from step0_access_path import JPG_QUALITY, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

import sys
sys.path.append("kong_util")
from kong_util.util import get_dir_certain_file_names, move_dir_certain_file, method2
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs
from kong_util.build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from kong_util.video_from_img import Video_combine_from_dir
from kong_util.multiprocess_util import multi_processing_interface
from multiprocessing import Manager

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

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''



#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################################################################
class See_rec_metric(See_info):
    """
    我把它繼承See_bm_rec 的概念是要做完 See_bm_rec 後才能做 See_rec_metric 喔！
    """
    def __init__(self, result_obj, see_name):
        super(See_rec_metric, self).__init__(result_obj, see_name)
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
        # self.metric_names  = get_dir_certain_file_names(self.metrec_read_dir , certain_word="metric_epoch", certain_ext=".jpg")
        # self.metric_read_paths  = [self.matplot_metric_visual_read_dir + "/" + name for name in self.metric_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.ld_color_visual_names       = get_dir_certain_file_names(self.metric_ld_color_read_dir , certain_word="ld_epoch", certain_ext=".jpg")
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
            path2 = self.rec_hope_path     ### 0c-rec_hope.jpg
            # path2 = self.rec_read_paths[-1]        ### bm_rec_matplot_visual/rec_visual/rec_gt.jpg

            ### rec_pred 要怎麼轉成 rec_GT
            # path1 = self.rec_hope_path     ### 0c-rec_hope.jpg
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
            single_row_imgs.Save_fig(self.metric_ld_matplot_write_dir, name="ld_epoch", epoch=go_epoch)
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
            path2 = self.rec_hope_path     ### 0c-rec_hope.jpg
            SSIM = SSIMs[go_epoch]
            LD   = LDs  [go_epoch]

            in_img     = cv2.imread(self.in_img_path)
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
            single_row_imgs.Save_fig(dst_dir=self.matplot_metric_visual_write_dir, name="metric_epoch", epoch=go_epoch)  ### 話完圖就可以存了喔！
