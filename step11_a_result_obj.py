from step0_access_path import JPG_QUALITY


import sys 
sys.path.append("kong_util")
from util import get_dir_certain_file_name, matplot_visual_single_row_imgs, matplot_visual_multi_row_imgs, draw_loss_util
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir

import cv2
import time 
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

class See:
    def __init__(self, result_dir, see_name):
        self.result_dir=result_dir
        self.see_name=see_name

        self.see_dir = self.result_dir + "/" + self.see_name
        self.see_file_names = None
        self.see_file_amount = None

        ### 不確定要不要，因為在initial就做這麼多事情好嗎~~會不會容易出錯哩~~
        Check_dir_exist_and_build(self.see_dir)
        self.get_see_file_info()  

    def get_see_file_info(self):
        self.see_file_names = get_dir_certain_file_name(self.see_dir, ".jpg")
        self.see_file_amount = len(self.see_file_names)

    def save_as_jpg(self):
        Save_as_jpg(self.see_dir, self.see_dir, delete_ord_file=True)

    def save_as_avi(self):
        Video_combine_from_dir(self.see_dir, self.see_dir, "0-combine_jpg_tail_long.avi", tail_long=True)

    ###############################################################################################
    ###############################################################################################
    def _Draw_matplot_visual(self, epoch, matplot_visual_dir, add_loss=False):
        in_img = cv2.imread(self.see_dir + "/" + self.see_file_names[0]) ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_dir + "/" + self.see_file_names[1]) ### 要記得see的第二張存的是 輸出的gt影像
        img = cv2.imread(self.see_dir + "/" + self.see_file_names[epoch+2]) ### see資料夾 內的影像 該epoch產生的影像 讀出來 
        fig, ax = matplot_visual_single_row_imgs(img_titles=["in_img", "out_img", "gt_img"],  ### 把每張圖要顯示的字包成list 
                                               imgs      = [ in_img ,   img ,      gt_img],      ### 把要顯示的每張圖包成list
                                               fig_title = "epoch=%04i"%epoch,   ### 圖上的大標題
                                               add_loss  = add_loss)

        if(add_loss): fig, ax = draw_loss_util(fig, ax[-1,1], self.see_dir+"/../logs", epoch, self.see_file_amount-2)
        plt.savefig(matplot_visual_dir+"/"+"epoch=%04i"%epoch )
        plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！

        # return fig, ax

    ###############################################################################################
    ###############################################################################################
    def save_as_matplot_visual_during_train(self, epoch, show_msg=False): ### 訓練中，一張張 生成matplot_visual(這裡不能後處理，因為後處理需要全局的see_file，這裡都單張單張的會出問題)
        start_time = time.time()
        matplot_visual_dir = self.see_dir + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        if(epoch==0):Check_dir_exist_and_build_new_dir(matplot_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_file_info() ### 取得 結果內的 某個see資料夾 內的所有影像 檔名
        self._Draw_matplot_visual(epoch, matplot_visual_dir)
        if(show_msg): print(f"processing {self.see_name}, cost_time:{time.time() - start_time}")
    ###############################################################################################
    ###############################################################################################
    def save_as_matplot_visual_after_train(self,   ### 訓練後，可以走訪所有see_file 並重新產生 matplot_visual
                                           add_loss = False,
                                           single_see_multiprocess=True): ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時(像是result裡面的save_all_single_see_as_matplot_visual_multiprocess)，傳參數時這要設為false喔！
        print(f"doing {self.see_name} save_as_matplot_visual_after_train")
        start_time = time.time()
        matplot_visual_dir = self.see_dir + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(matplot_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_file_info() ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        if(single_see_multiprocess):self._draw_matplot_visual_after_train_multiprocess(matplot_visual_dir, add_loss, core_amount=8, task_amount=self.see_file_amount)
        else:self._draw_matplot_visual_after_train(0, self.see_file_amount, matplot_visual_dir, add_loss)

        ### 後處理讓結果更小 但 又不失視覺品質
        Find_ltrd_and_crop(matplot_visual_dir, matplot_visual_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_visual_dir,matplot_visual_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_visual_dir, matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def _draw_matplot_visual_after_train_multiprocess(self, matplot_visual_dir, add_loss, core_amount=8, task_amount=600):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._draw_matplot_visual_after_train, task_args=[matplot_visual_dir, add_loss])

    def _draw_matplot_visual_after_train(self, start_img, img_amount, matplot_visual_dir, add_loss):
        print("processing %s"%self.see_name)
        for go_img in tqdm(range(start_img, start_img+img_amount)):
            if(go_img>=2): ### 第三張 才開始存 epoch影像喔！
                epoch = go_img-2  ### 第三張 才開始存 epoch影像喔！所以epoch的數字 是go_img-2
                fig, ax = self._Draw_matplot_visual(epoch, matplot_visual_dir, add_loss)
    ###############################################################################################
    ###############################################################################################



class Result:
    # def __init__(self, result_name=None, r_describe=None):
    def __init__(self):
        ### train的時候用的
        self.result_name = None
        self.result_dir  = None
        self.ckpt_dir = None
        self.logs_dir = None
        self.sees_ver = None
        self.sees = None
        self.see_amount = None
        self.see_file_amount = None 
        
        ### after train的時候才用的
        self.ana_plot_title = None ### 這是給matplot用的title

    # def rename_see1_to_see2(self):
    #     for go_see in range(self.see_amount):
    #         if(os.path.isdir(self.sees1[go_see].see_dir)):
    #             print("rename_ord:", self.sees1[go_see].see_dir)
    #             print("rename_dst:", self.sees2[go_see].see_dir)
    #             os.rename(self.sees1[go_see].see_dir, self.sees2[go_see].see_dir)

    def save_single_see_as_matplot_visual(self, see_num, add_loss=False, single_see_multiprocess=True):
        print(f"current result:{self.result_name}")
        self.sees[see_num].save_as_matplot_visual_after_train(add_loss, single_see_multiprocess)
            
    def save_all_single_see_as_matplot_visual(self, start_index, amount, add_loss=False, single_see_multiprocess=True):
        for see_num in tqdm(range(start_index, start_index+amount)):
            self.save_single_see_as_matplot_visual(see_num, add_loss, single_see_multiprocess)

        ### 測完成功可刪
        # for see in self.sees[start_index: start_index+amount]:
        #     see.save_as_matplot_visual_after_train(single_see_multiprocess)

    def save_all_single_see_as_matplot_visual_multiprocess(self, add_loss=False):
        print(f"doing {self.result_name}")
        from util import multi_processing_interface
        single_see_multiprocess = False  ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！
        multi_processing_interface(core_amount=8 ,task_amount=self.see_amount, task=self.save_all_single_see_as_matplot_visual, task_args=[add_loss, single_see_multiprocess])

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def save_multi_see_as_matplot_visual(self, see_nums, save_name, add_loss=False):
        ### 防呆 ### 這很重要喔！因為 row 只有一個時，matplot的ax的維度只有一維，但我的操作都兩維 會出錯！所以要切去一維的method喔！
        if(len(see_nums) == 1):
            print("因為 see_nums 的數量只有一個，自動切換成 single 的 method 囉～")
            self.save_single_see_as_matplot_visual(see_nums[0])
            return  
        ###############################################################################################
        start_time = time.time()
        matplot_multi_see_dir = self.result_dir + "/" + save_name ### 結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(matplot_multi_see_dir)          ### 建立 存結果的資料夾

        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for see_num in see_nums:
            in_imgs.append(cv2.imread(self.sees[see_num].see_dir + "/" + self.sees[see_num].see_file_names[0]))
            gt_imgs.append(cv2.imread(self.sees[see_num].see_dir + "/" + self.sees[see_num].see_file_names[1]))

        ### 抓 第一row的 要顯示的 titles
        titles = ["in_img", self.ana_plot_title, "gt_img"]
        r_c_titles = [titles] ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs
        print("doing save_multi_see_as_matplot_visual")
        self._draw_multi_see_multiprocess(see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss, core_amount=8, task_amount=self.see_file_amount)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(matplot_multi_see_dir, matplot_multi_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_multi_see_dir, matplot_multi_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_multi_see_dir, matplot_multi_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def _draw_multi_see_multiprocess(self, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False, core_amount=8, task_amount=600):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._Draw_multi_see, task_args=[see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss])

    def _Draw_multi_see(self, start_img, img_amount, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False ):
        print("doing analyze_col_results_multi_see")
        for go_img in tqdm(range(start_img, start_img+img_amount)):
            if(go_img >=2):
                epoch = go_img-2
                r_c_imgs = []
                for go_see_num, see_num in enumerate(see_nums):
                    c_imgs = [in_imgs[go_see_num]]
                    c_imgs.append(cv2.imread(self.sees[see_num].see_dir + "/" + self.sees[see_num].see_file_names[go_img]))
                    c_imgs += [gt_imgs[go_see_num]]
                    r_c_imgs.append(c_imgs)
                fig, ax = matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles, 
                                              rows_cols_imgs   = r_c_imgs,
                                              fig_title        = "epoch=%04i"%epoch,   ### 圖上的大標題
                                              bgr2rgb          = True,
                                              add_loss         = add_loss)

                if(add_loss): fig, ax = draw_loss_util(fig, ax[-1,1], self.logs_dir, epoch, self.see_file_amount-2)
                plt.savefig(matplot_multi_see_dir+"/"+"epoch=%04i"%epoch )
                plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！
    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################



if(__name__=="__main__"):
    from step11_b_result_obj_builder import Result_builder
    import matplotlib.pyplot as plt

    ## Result 的 各method測試：
    os_book = Result_builder().set_by_result_name("5_just_G_mae1369/type7b_h500_w332_real_os_book-20200525_225555-just_G-1532data_mae9_127.51").set_ana_plot_title("mae9").build()
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31],"train_rd")
    # os_book.save_single_see_as_matplot_visual(see_num=0)
    
    # os_book.sees[0].save_as_matplot_visual_after_train(add_loss=True, single_see_multiprocess=True)
    # os_book.save_single_see_as_matplot_visual( see_num=0, add_loss=True, single_see_multiprocess=True)
    # os_book.save_all_single_see_as_matplot_visual_multiprocess( add_loss=True)
    os_book.save_multi_see_as_matplot_visual([0,1,2], save_name="test_lt", add_loss=True)