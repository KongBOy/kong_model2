import cv2
import time 
import os
from tqdm import tqdm

import sys 
sys.path.append("kong_util")
from step0_access_path import access_path, JPG_QUALITY
from util import get_dir_certain_file_name, matplot_visual_single_row_imgs, matplot_visual_multi_row_imgs, draw_loss_util, Matplot_single_row_imgs, Matplot_multi_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir

import matplotlib.pyplot as plt

class Result_analyzer:
    def __init__(self, ana_describe):
        self.ana_describe = ana_describe
        self.analyze_dir = access_path + "analyze_dir"+"/"+self.ana_describe ### 例如 .../data_dir/analyze_dir/testtest
        Check_dir_exist_and_build(self.analyze_dir)
    
    ########################################################################################################################################
    def _temp_c_results_see1_update_to_see2_and_get_see_file_names(self, c_results):
        ### 暫時的update see1~see2
        for result in c_results:
            result.rename_see1_to_see2()
            for see in result.sees:
                see.get_see_file_names()

    def _temp_r_c_results_update_see1_to_see2_and_get_see_file_names(self, r_c_results):
        for c_results in r_c_results:
            self._temp_c_results_see1_update_to_see2_and_get_see_file_names(c_results)
########################################################################################################################################
########################################################################################################################################
class Col_results_analyzer(Result_analyzer):
    def __init__(self, ana_describe, col_results):
        super().__init__(ana_describe)
        self.c_results = col_results
        self.c_min_see_file_amount = self.get_c_min_see_file_amount()
        self.c_min_train_epochs = self.c_min_see_file_amount -2  ### 從see_file_name數量來推估 目前result被訓練幾個epoch，-2是減去 in_img, gt_img 兩個檔案

    def get_c_min_see_file_amount(self):
        see_file_amounts = []
        for result in self.c_results:
            see_file_amounts.append(result.see_file_amount)
        return min(see_file_amounts)
    ########################################################################################################################################
    ### 單一row，同see
    def _Draw_col_results_single_see(self, start_img, img_amount, see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss=False):
        # print("doing analyze_col_results_multi_see")
        for go_img in tqdm(range(start_img, start_img+img_amount)):
            if(go_img >=2):
                epoch = go_img-2
                
                c_imgs   = [in_imgs]
                for result in self.c_results: c_imgs.append(cv2.imread(result.sees[see_num].see_dir + "/" + result.sees[see_num].see_file_names[go_img]))
                c_imgs += [gt_imgs]

                single_row_imgs = Matplot_single_row_imgs(
                                imgs      = c_imgs,      ### 把要顯示的每張圖包成list
                                img_titles= c_titles,  ### 把每張圖要顯示的字包成list 
                                fig_title = "epoch=%04i"%epoch,   ### 圖上的大標題
                                add_loss  = add_loss)
                single_row_imgs.Draw_img()
                if(add_loss):
                    for go_result, result in enumerate(self.c_results):
                        single_row_imgs.Draw_ax_loss_after_train(ax=single_row_imgs.ax[-1, go_result+1], logs_dir=result.logs_dir, cur_epoch=epoch, min_epochs=self.c_min_train_epochs)
                single_row_imgs.Save_fig( dst_dir=analyze_see_dir, epoch=epoch)
                

    def _draw_col_results_single_see_multiprocess(self, see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss=False, core_amount=8, task_amount=100):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._Draw_col_results_single_see, task_args= [see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss] )
    
    def analyze_col_results_single_see(self, see_num, add_loss=False, single_see_multiprocess=True, single_see_core_amount=8): ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時，傳參數時這要設為false
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + self.c_results[0].sees[see_num].see_name  ### (可以再想想好名字！)分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(analyze_see_dir)                                       ### 建立 存結果的資料夾


        ### 暫時的update see1~see2
        # self._temp_r_c_results_update_see1_to_see2_and_get_see_file_names(self.c_results)
        
        ### 抓 in/gt imgs
        in_imgs = cv2.imread(self.c_results[0].sees[see_num].see_dir + "/" + self.c_results[0].sees[see_num].see_file_names[0])
        gt_imgs = cv2.imread(self.c_results[0].sees[see_num].see_dir + "/" + self.c_results[0].sees[see_num].see_file_names[1])

        ### 抓 要顯示的 titles
        c_titles = ["in_img"]
        for result in self.c_results: c_titles.append(result.ana_plot_title)
        c_titles += ["gt_img"]

        ### 抓  要顯示的imgs 並且畫出來
        if(single_see_multiprocess): self._draw_col_results_single_see_multiprocess(see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss, core_amount=single_see_core_amount, task_amount=self.c_min_see_file_amount)
        else: self._Draw_col_results_single_see(0, self.c_min_see_file_amount, see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss)

        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)
              
    def analyze_col_results_all_single_see(self,start_see, see_amount, add_loss=False, single_see_multiprocess=False, single_see_core_amount=8):
        for go_see in range(start_see, start_see + see_amount):
            self.analyze_col_results_single_see(go_see, add_loss, single_see_multiprocess=single_see_multiprocess, single_see_core_amount=single_see_core_amount) ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！

    def analyze_col_results_all_single_see_multiprocess(self, add_loss=False, core_amount=8, task_amount=32, single_see_multiprocess=False, single_see_core_amount=8):
        if(single_see_multiprocess==False):  ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！
            from util import multi_processing_interface
            multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self.analyze_col_results_all_single_see, task_args=[add_loss, False])
        else:  ### 如果大任務不開multi core，就給小任務開拉！
            self.analyze_col_results_all_single_see(start_see=0, see_amount=32, add_loss=add_loss, single_see_multiprocess=True, single_see_core_amount=single_see_core_amount)

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    def _Draw_col_results_multi_see(self, start_img, img_amount, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss=False ):
        for go_img in tqdm(range(start_img, start_img+img_amount)):
            if(go_img >=2):
                epoch = go_img-2

                r_c_imgs = []
                for go_see_num, see_num in enumerate(see_nums):
                    c_imgs   = [in_imgs[go_see_num]]
                    for result in self.c_results: c_imgs.append(cv2.imread(result.sees[see_num].see_dir + "/" + result.sees[see_num].see_file_names[go_img]))
                    c_imgs += [gt_imgs[go_see_num]]
                    r_c_imgs.append(c_imgs)

                multi_row_imgs = Matplot_multi_row_imgs(
                                              rows_cols_imgs   = r_c_imgs,
                                              rows_cols_titles = r_c_titles, 
                                              fig_title        = "epoch=%04i"%epoch,   ### 圖上的大標題
                                              bgr2rgb          = True,
                                              add_loss         = add_loss)
                multi_row_imgs.Draw_img()
                if(add_loss): 
                    for go_result, result in enumerate(self.c_results):
                        multi_row_imgs.Draw_ax_loss_after_train(ax=multi_row_imgs.ax[-1, go_result+1], logs_dir=result.logs_dir, cur_epoch=epoch, min_epochs=self.c_min_train_epochs)
                multi_row_imgs.Save_fig(dst_dir=analyze_see_dir, epoch=epoch)
        
                # fig, ax = matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles, 
                #                               rows_cols_imgs   = r_c_imgs,
                #                               fig_title        = "epoch=%04i"%epoch,   ### 圖上的大標題
                #                               bgr2rgb          = True,
                #                               add_loss         = add_loss)
                # if(add_loss): 
                #     for go_result, result in enumerate(self.c_results):
                #         draw_loss_util(fig, ax[-1,go_result+1], result.logs_dir, epoch, self.c_min_see_file_amount-2)
                # plt.savefig(analyze_see_dir+"/"+"epoch=%04i"%epoch )
                # plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！

    def _draw_col_results_multi_see_multiprocess(self, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss=False, core_amount=8, task_amount=100):
        print("use multiprocess")
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._Draw_col_results_multi_see, task_args= [see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss] )


    ### 同col同result，同row同see
    def analyze_col_results_multi_see(self, see_nums, save_name, add_loss=False, multiprocess=True):
        print(f"doing analyze_col_results_multi_see, save_name={save_name}")
        ### 防呆 ### 這很重要喔！因為 row 只有一個時，matplot的ax的維度只有一維，但我的操作都兩維 會出錯！所以要切去一維的method喔！
        if(len(see_nums) == 1):
            print("因為 see_nums 的數量只有一個，自動切換成 single 的 method 囉～")
            self.analyze_col_results_single_see(see_nums[0], add_loss)
            return  
        ###############################################################################################
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + save_name  ### (可以再想想好名字！)分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(analyze_see_dir)                                  ### 建立 存結果的資料夾
        print(f"save to {analyze_see_dir}")
        
        ### 暫時的update see1~see2
        # self._temp_c_results_see1_update_to_see2_and_get_see_file_names(self.c_results)
        
        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for see_num in see_nums:
            in_imgs.append(cv2.imread(self.c_results[0].sees[see_num].see_dir + "/" + self.c_results[0].sees[see_num].see_file_names[0]))
            gt_imgs.append(cv2.imread(self.c_results[0].sees[see_num].see_dir + "/" + self.c_results[0].sees[see_num].see_file_names[1]))

        ### 抓 第一row的 要顯示的 titles
        c_titles = ["in_img"]
        for result in self.c_results: c_titles.append(result.ana_plot_title)
        c_titles += ["gt_img"]
        r_c_titles = [c_titles] ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs 並且畫出來
        if(multiprocess):     self._draw_col_results_multi_see_multiprocess( see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss, core_amount=8, task_amount=self.c_min_see_file_amount)
        else:self._Draw_col_results_multi_see(0, self.c_min_see_file_amount, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 50]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    


### 目前小任務還沒有切multiprocess喔！
class Row_col_results_analyzer(Result_analyzer):
    def __init__(self, ana_describe, row_col_results):
        super().__init__( ana_describe)
        self.r_c_results = row_col_results
        self.r_c_min_see_file_amount = self.get_r_c_min_see_file_amount()
    
    def get_r_c_min_see_file_amount(self):
        see_file_amounts = []
        for row_results in self.r_c_results:
            for result in row_results:
                see_file_amounts.append(result.see_file_amount)
        return min(see_file_amounts)
    ########################################################################################################################################
    ### 各row各col 皆 不同result，但全部都看相同某個see；這analyzer不會有 multi_see 的method喔！因為row被拿去show不同的result了，就沒有空間給multi_see拉，所以參數就不用 single_see_multiprocess囉！
    def _draw_row_col_results_single_see(self, start_img, img_amount, see_num, r_c_titles, analyze_see_dir ):
        ### 要記得see的第一張存的是 輸入的in影像，第二張存的是 輸出的gt影像
        ### 因為是certain_see → 所有的result看的是相同see，所以所有result的in/gt都一樣喔！乾脆就抓最左上角result的in/gt就好啦！
        in_img = cv2.imread(self.r_c_results[0][0].sees[see_num].see_dir + "/" + self.r_c_results[0][0].sees[see_num].see_file_names[0] )  ### 第一張：in_img
        gt_img = cv2.imread(self.r_c_results[0][0].sees[see_num].see_dir + "/" + self.r_c_results[0][0].sees[see_num].see_file_names[1] )  ### 第二張：gt_img
        # for go_img in tqdm(range(self.r_c_min_see_file_amount)):
        for go_img in tqdm(range(start_img, start_img+img_amount)):
            if(go_img >=2):
                epoch = go_img-2
                # print("see_num=", see_num, "go_img=", go_img)
                r_c_imgs   = [] ### r_c_imgs   抓出所要要顯示的圖   ，然後要記得每個row的第一張要放in_img，最後一張要放gt_img喔！
                for row_results in self.r_c_results:
                    c_imgs   = [in_img]   ### 每個row的第一張要放in_img
                    for result in row_results: ### 抓出一個row的 img 和 title
                        c_imgs.append( cv2.imread( result.sees[see_num].see_dir + "/" + result.sees[see_num].see_file_names[go_img] ))
                    c_imgs += [gt_img]      ### 每個row的最後一張要放gt_img
                    r_c_imgs.append(c_imgs)
                ###########################################################################################################
                matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles, 
                                              rows_cols_imgs   = r_c_imgs,
                                              fig_title        ="epoch=%04i"%epoch,   ### 圖上的大標題
                                              dst_dir          = analyze_see_dir, 
                                              file_name        ="epoch=%04i"%epoch,
                                              bgr2rgb          = True )

    def _draw_row_col_results_single_see_multiprocess(self, see_num, r_c_titles, analyze_see_dir, core_amount=8, task_amount=100):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._draw_row_col_results_single_see, task_args= [see_num, r_c_titles, analyze_see_dir] )


    def analyze_row_col_results_single_see(self, see_num):
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + self.r_c_results[0][0].sees[see_num].see_name ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(analyze_see_dir)                                          ### 建立 存結果的資料夾
        
        ### 暫時的update see1~see2
        # self._temp_r_c_results_update_see1_to_see2_and_get_see_file_names(self.r_c_results)

        ### 抓 每row 每col 各不同result的 要顯示的 titles
        r_c_titles = [] ### r_c_titles 抓出所有要顯示的標題 ，然後要記得每個row的第一張要放in_img，最後一張要放gt_img喔！
        for row_results in self.r_c_results:
            c_titles = ["in_img"] ### 每個row的第一張要放in_img
            for result in row_results: ### 抓出一個row的 img 和 title
                c_titles.append(result.ana_plot_title)
            c_titles += ["gt_img"]  ### 每個row的最後一張要放gt_img
            r_c_titles.append(c_titles)


        print("processing see_num:", see_num)
        ### 抓 每row 每col 各不同result的 要顯示的imgs 並且畫出來
        ### 注意，這analyzer不會有 multi_see 的method喔！因為row被拿去show不同的result了，就沒有空間給 multi_see拉，所以不用寫if/else 來 限制 multi_see時 single_see_multiprocess 要設False這樣子～
        self._draw_row_col_results_single_see_multiprocess(see_num, r_c_titles, analyze_see_dir, core_amount=8, task_amount=self.r_c_min_see_file_amount)

        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def analyze_row_col_results_all_single_see(self,start_see, see_amount):
        for go_see in range(start_see, start_see + see_amount):
            self.analyze_row_col_results_single_see(go_see)
    
    def analyze_row_col_results_all_single_see_multiprocess(self, core_amount=8, task_amount=32):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self.analyze_row_col_results_all_single_see)


if(__name__=="__main__"):
    from step11_b_result_obj_builder import Result_builder
    from step11_c2_instance_organize import *

    ### Result_analyzer 的 各method測試：
    # try_c_result_analyzer = Col_results_analyzer( ana_describe="try_c_result_analyzer", col_results=[os_book_justG_mae1, os_book_justG_mae3, os_book_justG_mae6, os_book_justG_mae9, os_book_justG_mae20 ])
    # try_c_result_analyzer.analyze_col_results_single_see(0, add_loss=True, single_see_multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_single_see(0, add_loss=True, single_see_multiprocess=True)
    # try_c_result_analyzer.analyze_col_results_all_single_see(start_see=0, see_amount=1, add_loss=True)
    # try_c_result_analyzer.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # try_c_result_analyzer.analyze_col_results_multi_see([16], "train_lt", add_loss = False, multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = False, multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True, multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True, multiprocess=True)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)

    
    ##################################################################################################################
    # os_book_G_D_mae136 = Col_results_analyzer(ana_describe="5_1_GD_Gmae136_epoch700", col_results=[os_book_rect_1532_mae1, os_book_rect_1532_mae3, os_book_rect_1532_mae6, os_book_rect_1532_mae6_noD])
    # os_book_G_D_mae136.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([ 2, 3], "test_lt",  add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([ 6, 7], "test_rt",  add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([10,11], "test_ld",  add_loss = True)
    # os_book_G_D_mae136.analyze_col_results_multi_see([12,13], "test_rd",  add_loss = True)


    ##################################################################################################################
    # os_book_rect_D025目前太少，之後再加進去
    # os_book_G_D_vs_justG = Col_results_analyzer(ana_describe="5_2_GD_vs_justG", col_results=[os_book_rect_D10, os_book_rect_D05, os_book_rect_D01, os_book_rect_D00, os_book_rect_D00_justG])
    # os_book_G_D_vs_justG.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_G_D_vs_justG.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)


    ##################################################################################################################
    # os_book_1532data_mae136920 = Col_results_analyzer(ana_describe="5_3_justG_136920", 
    #                                     col_results=[os_book_justG_mae1,
    #                                                 os_book_justG_mae3,
    #                                                 os_book_justG_mae6,
    #                                                 os_book_justG_mae9,
    #                                                 os_book_justG_mae20
    #                                                 ])
    # os_book_1532data_mae136920.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([ 2, 3], "test_lt" , add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([ 6, 7], "test_rt" , add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([10,11], "test_ld" , add_loss = True)
    # os_book_1532data_mae136920.analyze_col_results_multi_see([12,13], "test_rd" , add_loss = True)


    ##################################################################################################################
    ### 4.放大縮小的比較 norm/ bigger
    # os_book_justG_bigger = Col_results_analyzer(ana_describe="5_4_justG_a_bigger", col_results=[os_book_justG_normal, os_book_justG_bigger])
    # os_book_justG_bigger.analyze_col_results_all_single_see_multiprocess(add_loss=True,single_see_multiprocess=True, single_see_core_amount=12)
    # os_book_justG_bigger.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_justG_bigger.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)

    ##################################################################################################################
    ### 4.放大縮小的比較 bigger_wrong/norm/small/small2
    # os_book_justG_big_small = Col_results_analyzer(ana_describe="5_4_justG_b_bigger_smaller", col_results=[os_book_justG_bigger, os_book_justG_normal, os_book_justG_smaller, os_book_justG_smaller2])
    # os_book_justG_big_small.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # os_book_justG_big_small.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_justG_big_small.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)

    ##################################################################################################################
    ### 5.focus
    # os_book_focus_GD_vs_G = Col_results_analyzer(ana_describe="5_5_focus_GD_vs_G", col_results=[os_book_rect_nfocus, os_book_rect_focus, os_book_justG_nfocus, os_book_justG_focus])
    # os_book_focus_GD_vs_G.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_focus_GD_vs_G.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)


    ##################################################################################################################
    ### 6.400 vs 1532
    # os_book_400 = Col_results_analyzer(ana_describe="5_6_a_400_page", col_results=[os_book_400_rect, os_book_400_justG])
    # os_book_400.analyze_col_results_all_single_see_multiprocess(add_loss=True, single_see_multiprocess=True, single_see_core_amount=10)
    # os_book_400.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_400.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)

    ##################################################################################################################
    ### 6.400 vs 1532
    # os_book_400_vs_1532 = Col_results_analyzer(ana_describe="5_6_b_400_vs_1532_page", col_results=[os_book_400_rect, os_book_400_justG, os_book_1532_rect, os_book_1532_justG])
    # os_book_400_vs_1532.analyze_col_results_all_single_see_multiprocess(add_loss=True, single_see_multiprocess=True, single_see_core_amount=10)
    # os_book_400_vs_1532.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_400_vs_1532.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)

    ##################################################################################################################
    ### 7.第一層 k7 vs k3
    # os_book_first_k7_vs_k3 = Col_results_analyzer(ana_describe="5_7_first_k7_vs_k3", col_results=[os_book_GD_first_k7, os_book_GD_first_k3, os_book_G_first_k7, os_book_G_first_k3 ])
    # os_book_first_k7_vs_k3.analyze_col_results_all_single_see_multiprocess(add_loss=True, single_see_multiprocess=True, single_see_core_amount=12)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_first_k7_vs_k3.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)

    ##################################################################################################################
    ### 8a.GD + mrf比較
    # os_book_GD_mrf = Col_results_analyzer(ana_describe="5_8a_GD_mrf", col_results=[os_book_GD_no_mrf, os_book_GD_mrf_79, os_book_GD_mrf_replace79])
    # os_book_GD_mrf = Col_results_analyzer(ana_describe="5_8a_GD_mrf_all", col_results=[os_book_GD_no_mrf, os_book_GD_mrf_7, os_book_GD_mrf_79, os_book_GD_mrf_replace7, os_book_GD_mrf_replace79])
    # os_book_GD_mrf.analyze_col_results_all_single_see_multiprocess(add_loss=True,single_see_multiprocess=True, single_see_core_amount=12)
    # os_book_GD_mrf.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_GD_mrf.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)

    ##################################################################################################################
    ### 8b.G + mrf比較
    # os_book_G_mrf = Col_results_analyzer(ana_describe="5_8b_G_mrf", col_results=[os_book_G_no_mrf,  os_book_G_mrf_79, os_book_G_mrf_replace79])
    os_book_G_mrf = Col_results_analyzer(ana_describe="5_8b_G_mrf_all", col_results=[os_book_G_no_mrf, os_book_G_mrf_7,  os_book_G_mrf_79, os_book_G_mrf_replace7, os_book_G_mrf_replace79])
    os_book_G_mrf.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_G_mrf.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)
    # os_book_G_mrf.analyze_col_results_all_single_see_multiprocess(add_loss=True,single_see_multiprocess=True, single_see_core_amount=12)


    ##################################################################################################################
    ### 9.GD，D保持train 1次，G train 1,3,5次 比較 
    # os_book_mrf = Col_results_analyzer(ana_describe="5_9_GD_D", col_results=[os_book_D1G1, os_book_D1G3, os_book_D1G5])
    # os_book_mrf.analyze_col_results_all_single_see_multiprocess(add_loss=True, core_amount=2)
    # os_book_mrf.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # os_book_mrf.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)


    # os_book_400.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    ### Row_col_results_analyzer 的使用範例
    # try_r_c_ana = Row_col_results_analyzer(ana_describe="try_row_col_results", 
    #                         row_col_results = [
    #                                     [os_book_justG_mae1 , os_book_justG_mae3],
    #                                     [os_book_justG_mae20, os_book_justG_mae6]
    #                                             ])
    # try_r_c_ana.analyze_row_col_results_single_see(2)




    # have_bg_gt_gray_mae3  = Result("1_bg_&_gt_color/type5d-real_have_see-have_bg-gt_gray3ch_20200428-152656_model5_rect2", r_describe="have_bg_gt_gray")
    # have_bg_gt_color_mae3 = Result("1_bg_&_gt_color/type5d-real_have_see-have_bg-gt_color_20200428-153059_model5_rect2"  , r_describe="have_bg_gt_color")
    # no_bg_gt_color_mae3   = Result("1_bg_&_gt_color/type5c-real_have_see-no_bg-gt-color_20200428-132611_model5_rect2"    , r_describe="no_bg_gt_color")
    # no_bg_gt_gray_mae3    = Result("1_bg_&_gt_color/type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2"  , r_describe="no_bg_gt_gray")

    # no_mrf_mae1 = Result("2_no_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200506-064552_model5_rect2" , r_describe="no_mrf_mae1")
    # no_mrf_mae3 = Result("2_no_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2" , r_describe="no_mrf_mae3")
    # no_mrf_mae6 = Result("2_no_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200506-065346_model5_rect2" , r_describe="no_mrf_mae6")
    
    # mrf_7_9_1   = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-190344_model6_mrf_rect2" , r_describe="mrf_7_9_mae1")
    # mrf_7_9_3   = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200429-145226_model6_mrf_rect2" , r_describe="mrf_7_9_mae3")
    # mrf_7_9_6   = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231036_model6_mrf_rect2" , r_describe="mrf_7_9_mae6")
    # mrf_7_11_1  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-190955_model6_mrf_rect2" , r_describe="mrf_7_11_mae1")
    # mrf_7_11_3  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200429-150505_model6_mrf_rect2" , r_describe="mrf_7_11_mae3")
    # mrf_7_11_6  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231336_model6_mrf_rect2" , r_describe="mrf_7_11_mae6")
    # mrf_9_11_1  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-190837_model6_mrf_rect2" , r_describe="mrf_9_11_mae1")
    # mrf_9_11_3  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200429-145548_model6_mrf_rect2" , r_describe="mrf_9_11_mae3")
    # mrf_9_11_6  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231249_model6_mrf_rect2" , r_describe="mrf_9_11_mae6")
    # mrf_13579_1 = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-191110_model6_mrf_rect2" , r_describe="mrf_13579_mae1")
    # mrf_13579_3 = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200428-154149_model6_mrf_rect2" , r_describe="mrf_13579_mae3")
    # mrf_13579_6 = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231530_model6_mrf_rect2" , r_describe="mrf_13579_mae6")


    # mrf_replace7_use7   = Result("4_mrf_replace7/type5c-real_have_see-no_bg-gt-gray3ch_20200507-105001_model6_mrf_rect2" , r_describe="replace7_use7")
    # mrf_replace7_use5_7 = Result("4_mrf_replace7/type5c-real_have_see-no_bg-gt-gray3ch_20200507-105739_model6_mrf_rect2" , r_describe="replace7_use5+7")
    # mrf_replace7_use7_9 = Result("4_mrf_replace7/type5c-real_have_see-no_bg-gt-gray3ch_20200507-110022_model6_mrf_rect2" , r_describe="replace7_use7+9")
    
    #################################################################################################################################
    ### 分析 bg 和 gt_color
    # bg_and_gt_color_results = [have_bg_gt_gray_mae3, have_bg_gt_color_mae3, no_bg_gt_color_mae3, no_bg_gt_gray_mae3]
    # bg_and_gt_color_analyze = Result_analyzer("bg_and_gt_color_analyze")

    ### 覺得好像可以省，因為看multi的比較方便ˊ口ˋ 不過single還是有好處：可以放很大喔！覺得有空再生成好了～
    # # bg_and_gt_color_analyze.analyze_col_results_all_single_see_multiprocess(bg_and_gt_color_results)  ### 覺得好像可以省，因為看multi的比較方便ˊ口ˋ

    ### 這裡是在測試 col_results_multi_see 一次看多少row比較好，結論是4
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [14,15],          "test_str_2img")  
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [12,14,15],       "test_str_3img") ### 覺得3不錯！
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [11,12,14,15],    "test_str_4img") ### 覺得4不錯！且可看lt,rt,ld,rd
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [11,12,13,14,15], "test_str_5img")
    
    ### 一次看多see
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [ 1, 2, 4, 5], "test_img")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [ 6, 7, 9,10], "test_lin")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [11,12,14,15], "test_str")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [17,18,19,20], "train_img")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [22,23,24,25], "train_lin")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see(bg_and_gt_color_results, [28,29,30,31], "train_str")


    #################################################################################################################################
    ### 分析 mrf loss mae的比例
    # mrf_r_c_results = [   
    #                       [no_mrf_mae1, mrf_7_9_1, mrf_7_11_1, mrf_9_11_1, mrf_13579_1],
    #                       [no_mrf_mae3, mrf_7_9_3, mrf_7_11_3, mrf_9_11_3, mrf_13579_3],
    #                       [no_mrf_mae6, mrf_7_9_6, mrf_7_11_6, mrf_9_11_6, mrf_13579_6]
    #                   ]
    # mrf_loss_analyze = Result_analyzer(ana_describe="mrf_loss_analyze")
    # mrf_loss_analyze.analyze_row_col_results_all_single_see_multiprocess(mrf_r_c_results)
    
    #################################################################################################################################
    ### 分析 mrf 取代 第一層7
    # mrf_replace7_results = [no_mrf_mae3, mrf_replace7_use7, mrf_replace7_use5_7, mrf_replace7_use7_9]
    # mrf_replace7_analyze = Result_analyzer("mrf_replace7_analyze")
    
    # ## 覺得好像可以省，因為看multi的比較方便ˊ口ˋ 不過single還是有好處：可以放很大喔！覺得有空再生成好了～
    # # mrf_replace7_analyze.analyze_col_results_all_single_see_multiprocess(mrf_replace7_results)
    
    # ## 一次看多see
    # mrf_replace7_analyze.analyze_col_results_multi_see(mrf_replace7_results, [ 1, 2, 4, 5], "test_img")
    # mrf_replace7_analyze.analyze_col_results_multi_see(mrf_replace7_results, [ 6, 7, 9,10], "test_lin")
    # mrf_replace7_analyze.analyze_col_results_multi_see(mrf_replace7_results, [11,12,14,15], "test_str")
    # mrf_replace7_analyze.analyze_col_results_multi_see(mrf_replace7_results, [17,18,19,20], "train_img")
    # mrf_replace7_analyze.analyze_col_results_multi_see(mrf_replace7_results, [22,23,24,25], "train_lin")
    # mrf_replace7_analyze.analyze_col_results_multi_see(mrf_replace7_results, [28,29,30,31], "train_str")
    #################################################################################################################################
    ### 分析 os_book
    # os_book_results = [os_book]
    # os_book_analyze = Result_analyzer("os_book")
    
    # os_book.save_all_single_see_as_matplot_visual_multiprocess( )
    
    ## 一次看多see
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 0, 1, 2, 3], "test_lt")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 4, 5, 6, 7], "test_rt")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 4, 5, 6, 7], "test_rt")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 6, 7, 9,10], "test_lin")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [11,12,14,15], "test_str")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [17,18,19,20], "train_img")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [22,23,24,25], "train_lin")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [28,29,30,31], "train_str")


    #########################################################################################################
    # mrf_c_results = [mrf_7_9_1, mrf_7_11_1, mrf_9_11_1, mrf_13579_1]
    # try_c_result_multi_see = Result_analyzer(ana_describe="try_c_result_multi_see")
    # try_c_result_multi_see.analyze_col_results_multi_see(mrf_c_results, [1,3,5], "see_1_3_5_jpg_then_crop")
    # try_c_result_multi_see.analyze_col_results_multi_see(mrf_c_results, [1,3,5], "see_1_3_5_jpg")
    # try_c_result_multi_see.analyze_col_results_single_see(mrf_c_results, 1)