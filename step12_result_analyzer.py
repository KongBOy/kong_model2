import cv2
import time
from tqdm import tqdm

import sys
sys.path.append("kong_util")
from step0_access_path import analyze_access_path, JPG_QUALITY, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from matplot_fig_ax_util import Matplot_single_row_imgs, Matplot_multi_row_imgs
from build_dataset_combine import Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop, Save_as_jpg
from video_from_img import Video_combine_from_dir


class Result_analyzer:
    def __init__(self, ana_describe):
        self.ana_describe = ana_describe
        self.analyze_dir = analyze_access_path + "analyze_dir" + "/" + self.ana_describe  ### 例如 .../data_dir/analyze_dir/testtest
        Check_dir_exist_and_build(self.analyze_dir)

    ########################################################################################################################################
    def _c_results_get_see_dir_info(self, c_results):
        """
        update 一下 所有 result 的 sees， 這裡是 c_results 的所有 results
        """
        for result in c_results:
            for see in result.sees:
                see.get_see_dir_info()

    def _r_c_results_get_see_dir_info(self, r_c_results):
        """
        update 一下 所有 result 的 sees， 這裡是 r_c_results 的所有 results， 會 一個個 r 去呼叫 _c_results_get_see_dir_info 來 處理 c_results 的 sees 喔！
        """
        for c_results in r_c_results:
            self._c_results_get_see_dir_info(c_results)
########################################################################################################################################
########################################################################################################################################
class Col_results_analyzer(Result_analyzer):
    def __init__(self, ana_describe, col_results):
        super().__init__(ana_describe)
        self.c_results = col_results
        self.c_min_see_file_amount = self.get_c_min_see_file_amount()
        self.c_min_train_epochs = self.c_min_see_file_amount - 2  ### 從see_file_name數量來推估 目前result被訓練幾個epoch，-2是減去 in_img, gt_img 兩個檔案

    def get_c_min_see_file_amount(self):
        see_file_amounts = []
        for result in self.c_results:
            ### 執行step12以前應該就要確保 see 已經生成完畢， 這樣子的假設下每個see都是一樣多檔案喔，所以就挑第一個拿他的see_file_amount就好囉～
            result.sees[0].get_see_dir_info()  ### 先去把 sees[0] 的資訊更新
            see_file_amounts.append(result.sees[0].see_file_amount)   ### 再把 sees[0]的 see_file_amount 抓出來
        return min(see_file_amounts)

    ########################################################################################################################################
    ### 單一row，同see
    def _Draw_col_results_single_see(self, start_img, img_amount, see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss=False):
        # print("doing analyze_col_results_multi_see")
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 2):
                epoch = go_img - 2

                c_imgs   = [in_imgs]
                for result in self.c_results: c_imgs.append(cv2.imread(result.sees[see_num].see_read_dir + "/" + result.sees[see_num].see_jpg_names[go_img]))
                c_imgs += [gt_imgs]

                single_row_imgs = Matplot_single_row_imgs(
                                imgs      =c_imgs,      ### 把要顯示的每張圖包成list
                                img_titles=c_titles,    ### 把每張圖要顯示的字包成list
                                fig_title ="epoch=%04i" % epoch,   ### 圖上的大標題
                                add_loss  =add_loss)
                single_row_imgs.Draw_img()
                if(add_loss):
                    for go_result, result in enumerate(self.c_results):
                        single_row_imgs.Draw_ax_loss_after_train(ax=single_row_imgs.ax[-1, go_result + 1], logs_read_dir=result.logs_read_dir, cur_epoch=epoch, min_epochs=self.c_min_train_epochs)
                single_row_imgs.Save_fig(dst_dir=analyze_see_dir, epoch=epoch)


    def _draw_col_results_single_see_multiprocess(self, see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss=False, core_amount=8, task_amount=100):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._Draw_col_results_single_see, task_args=[see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss])

    def analyze_col_results_single_see(self, see_num, add_loss=False, single_see_multiprocess=True, single_see_core_amount=8):  ### single_see_multiprocess 預設是true，然後要記得在大任務multiprocess時，傳參數時這要設為false
        print(f"{self.ana_describe} doing analyze_col_results_single_see, see_num:{see_num}, multiprocess:{single_see_multiprocess}")
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + self.c_results[0].sees[see_num].see_name  ### (可以再想想好名字！)分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(analyze_see_dir)                                       ### 建立 存結果的資料夾


        ### 在使用 所有 result 前， 要記得先去 update 一下 他們的 sees 喔！
        self._c_results_get_see_dir_info(self.c_results)

        ### 抓 in/gt imgs
        in_imgs = cv2.imread(self.c_results[0].sees[see_num].see_read_dir + "/" + self.c_results[0].sees[see_num].see_jpg_names[0])
        gt_imgs = cv2.imread(self.c_results[0].sees[see_num].see_read_dir + "/" + self.c_results[0].sees[see_num].see_jpg_names[1])

        ### 抓 要顯示的 titles
        c_titles = ["in_img"]
        for result in self.c_results: c_titles.append(result.ana_describe)
        c_titles += ["gt_img"]

        ### 抓  要顯示的imgs 並且畫出來
        if(single_see_multiprocess): self._draw_col_results_single_see_multiprocess(see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss, core_amount=single_see_core_amount, task_amount=self.c_min_see_file_amount)
        else: self._Draw_col_results_single_see(0, self.c_min_see_file_amount, see_num, in_imgs, gt_imgs, c_titles, analyze_see_dir, add_loss)

        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def analyze_col_results_all_single_see(self, start_see, see_amount, add_loss=False, single_see_multiprocess=False, single_see_core_amount=8):
        for go_see in range(start_see, start_see + see_amount):
            self.analyze_col_results_single_see(go_see, add_loss=add_loss, single_see_multiprocess=single_see_multiprocess, single_see_core_amount=single_see_core_amount)  ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！

    def analyze_col_results_all_single_see_multiprocess(self, add_loss=False, core_amount=8, task_amount=32, single_see_multiprocess=False, single_see_core_amount=8):
        if(single_see_multiprocess is False):  ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！
            from util import multi_processing_interface
            multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self.analyze_col_results_all_single_see, task_args=[add_loss, False])
        else:  ### 如果大任務不開multi core，就給小任務開拉！
            self.analyze_col_results_all_single_see(start_see=0, see_amount=32, add_loss=add_loss, single_see_multiprocess=True, single_see_core_amount=single_see_core_amount)

    ########################################################################################################################################
    ########################################################################################################################################
    ########################################################################################################################################
    def _Draw_col_results_multi_see(self, start_img, img_amount, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss=False):
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 2):
                epoch = go_img - 2

                r_c_imgs = []
                for go_see_num, see_num in enumerate(see_nums):
                    c_imgs   = [in_imgs[go_see_num]]
                    for result in self.c_results: c_imgs.append(cv2.imread(result.sees[see_num].see_read_dir + "/" + result.sees[see_num].see_jpg_names[go_img]))
                    c_imgs += [gt_imgs[go_see_num]]
                    r_c_imgs.append(c_imgs)

                multi_row_imgs = Matplot_multi_row_imgs(
                                              rows_cols_imgs   = r_c_imgs,
                                              rows_cols_titles = r_c_titles,
                                              fig_title        = "epoch=%04i" % epoch,   ### 圖上的大標題
                                              bgr2rgb          = True,
                                              add_loss         = add_loss)
                multi_row_imgs.Draw_img()
                if(add_loss):
                    for go_result, result in enumerate(self.c_results):
                        multi_row_imgs.Draw_ax_loss_after_train(ax=multi_row_imgs.ax[-1, go_result + 1], logs_read_dir=result.logs_read_dir, cur_epoch=epoch, min_epochs=self.c_min_train_epochs)
                multi_row_imgs.Save_fig(dst_dir=analyze_see_dir, epoch=epoch)

    def _draw_col_results_multi_see_multiprocess(self, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss=False, core_amount=8, task_amount=100):
        print("use multiprocess")
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._Draw_col_results_multi_see, task_args=[see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss])

    ### 同col同result，同row同see
    def analyze_col_results_multi_see(self, see_nums, save_name, add_loss=False, multiprocess=True):
        print(f"{self.ana_describe} doing analyze_col_results_multi_see, save_name={save_name}, multiprocess:{multiprocess}")
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

        ### 在使用 所有 result 前， 要記得先去 update 一下 他們的 sees 喔！
        self._c_results_get_see_dir_info(self.c_results)

        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for see_num in see_nums:
            in_imgs.append(cv2.imread(self.c_results[0].sees[see_num].see_read_dir + "/" + self.c_results[0].sees[see_num].see_jpg_names[0]))
            gt_imgs.append(cv2.imread(self.c_results[0].sees[see_num].see_read_dir + "/" + self.c_results[0].sees[see_num].see_jpg_names[1]))

        ### 抓 第一row的 要顯示的 titles
        c_titles = ["in_img"]
        for result in self.c_results: c_titles.append(result.ana_describe)
        c_titles += ["gt_img"]
        r_c_titles = [c_titles]  ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs 並且畫出來
        if(multiprocess):       self._draw_col_results_multi_see_multiprocess(see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss, core_amount=8, task_amount=self.c_min_see_file_amount)
        else: self._Draw_col_results_multi_see(0, self.c_min_see_file_amount, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, add_loss)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 50], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)




### 目前小任務還沒有切multiprocess喔！
class Row_col_results_analyzer(Result_analyzer):
    def __init__(self, ana_describe, row_col_results):
        super().__init__(ana_describe)
        self.r_c_results = row_col_results
        self.r_c_min_see_file_amount = self.get_r_c_min_see_file_amount()

    def get_r_c_min_see_file_amount(self):
        see_file_amounts = []
        for row_results in self.r_c_results:
            for result in row_results:
                ### 執行step12以前應該就要確保 see 已經生成完畢， 這樣子的假設下每個see都是一樣多檔案喔，所以就挑第一個拿他的see_file_amount就好囉～
                result.sees[0].get_see_dir_info()  ### 先去把 sees[0] 的資訊更新
                see_file_amounts.append(result.sees[0].see_file_amount)   ### 再把 sees[0]的 see_file_amount 抓出來
        return min(see_file_amounts)

    ########################################################################################################################################
    ### 各row各col 皆 不同result，但全部都看相同某個see；這analyzer不會有 multi_see 的method喔！因為row被拿去show不同的result了，就沒有空間給multi_see拉，所以參數就不用 single_see_multiprocess囉！
    def _draw_row_col_results_single_see(self, start_img, img_amount, see_num, r_c_titles, analyze_see_dir):
        ### 要記得see的第一張存的是 輸入的in影像，第二張存的是 輸出的gt影像
        ### 因為是certain_see → 所有的result看的是相同see，所以所有result的in/gt都一樣喔！乾脆就抓最左上角result的in/gt就好啦！
        in_img = cv2.imread(self.r_c_results[0][0].sees[see_num].see_read_dir + "/" + self.r_c_results[0][0].sees[see_num].see_jpg_names[0])  ### 第一張：in_img
        gt_img = cv2.imread(self.r_c_results[0][0].sees[see_num].see_read_dir + "/" + self.r_c_results[0][0].sees[see_num].see_jpg_names[1])  ### 第二張：gt_img
        # for go_img in tqdm(range(self.r_c_min_see_file_amount)):
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 2):
                epoch = go_img - 2
                # print("see_num=", see_num, "go_img=", go_img)
                r_c_imgs   = []  ### r_c_imgs   抓出所要要顯示的圖   ，然後要記得每個row的第一張要放in_img，最後一張要放gt_img喔！
                for row_results in self.r_c_results:
                    c_imgs   = [in_img]   ### 每個row的第一張要放in_img
                    for result in row_results:  ### 抓出一個row的 img 和 title
                        c_imgs.append(cv2.imread(result.sees[see_num].see_read_dir + "/" + result.sees[see_num].see_jpg_names[go_img]))
                    c_imgs += [gt_img]      ### 每個row的最後一張要放gt_img
                    r_c_imgs.append(c_imgs)
                ###########################################################################################################
                # rows_cols_titles, rows_cols_imgs, fig_title="epoch=1005", bgr2rgb=True, add_loss=False

                row_col_imgs = Matplot_multi_row_imgs(
                                              rows_cols_imgs   = r_c_imgs,
                                              rows_cols_titles = r_c_titles,
                                              fig_title        = "epoch=%04i" % epoch,   ### 圖上的大標題
                                              bgr2rgb          = True,
                                              add_loss         = False)
                row_col_imgs.Draw_img()
                row_col_imgs.Save_fig(dst_dir=analyze_see_dir, epoch=epoch)

    def _draw_row_col_results_single_see_multiprocess(self, see_num, r_c_titles, analyze_see_dir, core_amount=8, task_amount=100):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._draw_row_col_results_single_see, task_args=[see_num, r_c_titles, analyze_see_dir])


    def analyze_row_col_results_single_see(self, see_num, single_see_multiprocess=False, single_see_core_amount=8):
        print(f"{self.ana_describe} doing analyze_row_col_results_single_see, see_num:{see_num}, multiprocess:{single_see_multiprocess}")
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + self.r_c_results[0][0].sees[see_num].see_name  ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(analyze_see_dir)                                          ### 建立 存結果的資料夾

        ### 在使用 所有 result 前， 要記得先去 update 一下 他們的 sees 喔！
        self._r_c_results_get_see_dir_info(self.r_c_results)

        ### 抓 每row 每col 各不同result的 要顯示的 titles
        r_c_titles = []  ### r_c_titles 抓出所有要顯示的標題 ，然後要記得每個row的第一張要放in_img，最後一張要放gt_img喔！
        for row_results in self.r_c_results:
            c_titles = ["in_img"]  ### 每個row的第一張要放in_img
            for result in row_results:  ### 抓出一個row的 img 和 title
                c_titles.append(result.ana_describe)
            c_titles += ["gt_img"]  ### 每個row的最後一張要放gt_img
            r_c_titles.append(c_titles)


        print("processing see_num:", see_num)
        ### 抓 每row 每col 各不同result的 要顯示的imgs 並且畫出來
        ### 注意，這analyzer不會有 multi_see 的method喔！因為row被拿去show不同的result了，就沒有空間給 multi_see拉，所以不用寫if/else 來 限制 multi_see時 single_see_multiprocess 要設False這樣子～
        if(single_see_multiprocess): self._draw_row_col_results_single_see_multiprocess(see_num, r_c_titles, analyze_see_dir, core_amount=single_see_core_amount, task_amount=self.r_c_min_see_file_amount)
        else: self._draw_row_col_results_single_see(0, self.r_c_min_see_file_amount, see_num, r_c_titles, analyze_see_dir)

        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def analyze_row_col_results_all_single_see(self, start_see, see_amount, single_see_multiprocess=False, single_see_core_amount=8):
        for go_see in range(start_see, start_see + see_amount):
            self.analyze_row_col_results_single_see(go_see, single_see_multiprocess=single_see_multiprocess, single_see_core_amount=single_see_core_amount)

    def analyze_row_col_results_all_single_see_multiprocess(self, core_amount=8, task_amount=32, single_see_multiprocess=False, single_see_core_amount=8):
        from util import multi_processing_interface
        if(single_see_multiprocess):
            self.analyze_row_col_results_all_single_see(16, int(task_amount / 2), single_see_multiprocess=single_see_multiprocess, single_see_core_amount=single_see_core_amount)
            self.analyze_row_col_results_all_single_see(0, int(task_amount / 2), single_see_multiprocess=single_see_multiprocess, single_see_core_amount=single_see_core_amount)
        else:
            multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self.analyze_row_col_results_all_single_see)

def check_analyze(analyze_objs):
    for analyze_obj in analyze_objs:
        analyze_obj.analyze_col_results_single_see(see_num=31, add_loss=True, single_see_multiprocess=True, single_see_core_amount=12)

def doing_analyze_2page(analyze_obj):
    # analyze_obj.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([20,23], "train_rt", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([24,25], "train_ld", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([30,31], "train_rd", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([ 2, 3], "test_lt", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([ 6, 7], "test_rt", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([10,11], "test_ld", add_loss = True)
    # analyze_obj.analyze_col_results_multi_see([12,13], "test_rd", add_loss = True)
    analyze_obj.analyze_col_results_all_single_see_multiprocess(add_loss=True, single_see_multiprocess=True, single_see_core_amount=10)



class Bm_Rec_exps_analyze(Result_analyzer):
    def __init__(self, ana_describe, exps):
        super().__init__(ana_describe)
        self.exps = exps

    def _build_analyze_see_bm_rec_dir(self, see_num):
        analyze_see_dir = self.analyze_dir + "/" + self.exps[0].result_obj.sees[see_num].see_name  ### (可以再想想好名字！)分析結果存哪裡定位出來，上面是analyze_see_dir
        analyze_see_bm_dir  = analyze_see_dir + "/" + "bm"       ### 定出 存結果的資料夾
        analyze_see_rec_dir = analyze_see_dir + "/" + "rec"      ### 定出 存結果的資料夾
        Check_dir_exist_and_build_new_dir(analyze_see_dir)       ### 建立 存結果的資料夾
        Check_dir_exist_and_build_new_dir(analyze_see_bm_dir)    ### 建立 存結果的資料夾
        Check_dir_exist_and_build_new_dir(analyze_see_rec_dir)   ### 建立 存結果的資料夾
        return analyze_see_bm_dir, analyze_see_rec_dir


    def single_see_certain_rec_analyze(self, see_num, epoch):
        analyze_see_bm_dir, analyze_see_rec_dir = self._build_analyze_see_bm_rec_dir(see_num)  ### 定出 存結果的資料夾

        for exp in self.exps:
            see = exp.result_obj.sees[see_num]      ### 先抓出 要用的物件
            analyze_describe = result.ana_describe  ### 先抓出 要用的物件，補充一下在step11_c.py 可以自己設定設定每個 每個result的 ana_describe 喔！

            see.get_bm_rec_info()  ### 抓 result/see_.../matplot_bm_rec_visual/bm_visual 和 rec_visual 的 nemas, paths
            analyze_see_rec_final_path = analyze_see_rec_dir + "/" + analyze_describe + ".jpg"  ### 定出存哪：rec_final_path
            analyze_see_rec_gt_path    = analyze_see_rec_dir + "/" + "rec_gt" + ".jpg"          ### 定出存哪：rec_gt_path
            rec_gt    = cv2.imread(see.rec_paths[epoch])  ### 讀圖，倒數第一張 是 gt
            rec_final = cv2.imread(see.rec_paths[epoch])  ### 讀圖，倒數第二張 是 最後一個epoch
            # print(result.sees[see_num].rec_paths[-2])                   ### debug用
            cv2.imwrite(analyze_see_rec_final_path, rec_final)          ### 根據上面定出的位置存圖
            cv2.imwrite(analyze_see_rec_gt_path   , rec_gt)             ### 根據上面定出的位置存圖
        return self

    def single_see_final_rec_analyze(self, see_num):
        print(f"Bm_Rec_exps_analyze, doing single_see_final_rec_analyze, analyzing see_num:{see_num}")
        analyze_see_bm_dir, analyze_see_rec_dir = self._build_analyze_see_bm_rec_dir(see_num)  ### 定出 存結果的資料夾

        for exp in self.exps:
            see = exp.result_obj.sees[see_num]          ### 先抓出 要用的物件
            analyze_describe = exp.result_obj.ana_describe  ### 先抓出 要用的物件，補充一下在step11_c.py 可以自己設定設定每個 每個result的 ana_describe 喔！

            see.get_bm_rec_info()  ### 抓 result/see_.../matplot_bm_rec_visual/bm_visual 和 rec_visual 的 nemas, paths
            analyze_see_rec_final_path = analyze_see_rec_dir + "/" + analyze_describe + ".jpg"  ### 定出存哪：rec_final_path
            analyze_see_rec_gt_path    = analyze_see_rec_dir + "/" + "rec_gt" + ".jpg"          ### 定出存哪：rec_gt_path
            rec_gt    = cv2.imread(see.rec_paths[-1])  ### 讀圖，倒數第一張 是 gt
            rec_final = cv2.imread(see.rec_paths[-2])  ### 讀圖，倒數第二張 是 最後一個epoch
            # print(see.rec_paths[-2])                   ### debug用
            cv2.imwrite(analyze_see_rec_final_path, rec_final)          ### 根據上面定出的位置存圖
            cv2.imwrite(analyze_see_rec_gt_path   , rec_gt)             ### 根據上面定出的位置存圖
        return self

    def all_single_see_final_rec_analyze(self):
        print(f"Bm_Rec_exps_analyze, doing all_single_see_final_rec_analyze, analyzing {self.ana_describe}")
        start_time = time.time()
        for see_num in range(self.exps[0].result_obj.see_amount):
            self.single_see_final_rec_analyze(see_num=see_num)
        print(f"Bm_Rec_exps_analyze, doing all_single_see_final_rec_analyze, analyzing {self.ana_describe}, cost time:{time.time() - start_time}")
        return self

    def analyze_tensorboard(self, reset_board_dir=False):
        analyze_board_dir = self.analyze_dir + "/" + "boards"  ### 分析結果存哪裡定位出來 例如 D:\0 data_dir\analyze_dir\5_14-bm_rec-2_1-ch_results\boards
        if(reset_board_dir): Check_dir_exist_and_build_new_dir(analyze_board_dir)    ### 一開始雖然覺得下面較好，但實際用起來發現我也不會存結論在board資料夾，而是直接寫在這裡，所以就多個reset_flag，讓我可控制要不要刪掉上次的board資料夾囉！
        else:                Check_dir_exist_and_build(analyze_board_dir)           ### 建立 存結果的資料夾，目前覺的外層的這個 不用 build_new_dir，這樣才可以存筆記在裡面，要小心的是 如果 exps 有刪掉某個exp，就不會自動刪掉囉！
        for exp in self.exps:
            analyze_board_ana_dir = analyze_board_dir + "/" + exp.result_obj.ana_describe   ### D:\0 data_dir\analyze_dir\5_14-bm_rec-2_1-ch_results\boards\exp.result_obj.ana_describe
            exp.loss_info_obj = exp.loss_info_builder.build()
            print("exp.loss_info_builder.loss_info_obj.logs_read_dir ~~~~~~~ ", exp.loss_info_builder.loss_info_obj.logs_read_dir)
            print("exp.loss_info_builder.loss_info_obj.logs_write_dir ~~~~~~~", exp.loss_info_builder.loss_info_obj.logs_write_dir)
            print("exp.loss_info_obj.logs_read_dir ~~~~~~~ " , exp.loss_info_obj.logs_read_dir)
            print("exp.loss_info_obj.logs_write_dir ~~~~~~~", exp.loss_info_obj.logs_write_dir)
            exp.loss_info_obj.use_npy_rebuild_justG_tensorboard_loss(exp=exp, dst_dir=analyze_board_ana_dir)
        return self


if(__name__ == "__main__"):
    from step11_c_exp_grouping import  *

    ana_title = "05_14-bm_rec-"
    # Bm_Rec_exps_analyze(ana_title + "0_1-epoch_old_shuf_exps",     epoch_old_shuf_exps)    .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "0_2-epoch_new_shuf_exps",     epoch_new_shuf_exps)    .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "0_3-epoch_old_new_shuf_exps", epoch_old_new_shuf_exps).all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "0_4-ch_old_shuf_exps",        ch_old_shuf_exps)       .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "0_5-ch_new_shuf_exps",        ch_new_shuf_exps)       .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "0_6-ch_old_new_shuf_exps",    ch_old_new_shuf_exps)   .all_single_see_final_rec_analyze().analyze_tensorboard()

    # Bm_Rec_exps_analyze(ana_title + "1_1-epoch_exps",                      epoch_exps)                     .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "1_1b-epoch_exps_in_300_500",           epoch300_500_exps)               .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "2_1-ch_exps",                         ch_exps)                        .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "2_2-epoch_no_down_vs_ch_exps",        epoch_no_down_vs_ch_exps)       .all_single_see_final_rec_analyze().analyze_tensorboard()

    # Bm_Rec_exps_analyze(ana_title + "3_1-bn_ch64_exps_bn_see_arg_T",       bn_ch64_exps_bn_see_arg_T)      .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "3_2-bn_ch32_exps_bn_see_arg_T",       bn_ch32_exps_bn_see_arg_T)      .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "3_3-bn_ch64_exps_bn_see_arg_F_and_T", bn_ch64_exps_bn_see_arg_F_and_T).all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "3_4-bn_ch32_exps_bn_see_arg_F_and_T", bn_ch32_exps_bn_see_arg_F_and_T).all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "4_1a-bn_in_size1_exps",                bn_in_size1_exps)               .all_single_see_final_rec_analyze().analyze_tensorboard()

    # Bm_Rec_exps_analyze(ana_title + "4_2-bn_in_sizen_exps",                bn_in_sizen_exps)               .all_single_see_final_rec_analyze().analyze_tensorboard()

    # Bm_Rec_exps_analyze(ana_title + "5_1-in_concat_AB",                    in_concat_AB)                   .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "6_1-unet_layers",                     unet_layers)                    .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "7a_1-unet_skip_use_add",               unet_skip_use_add)              .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "7a_2-unet_skip_use_concat_vs_add",     unet_skip_use_concat_vs_add)    .all_single_see_final_rec_analyze().analyze_tensorboard()

    # Bm_Rec_exps_analyze(ana_title + "7b-unet_skip_noC",                      unet_skip_noC)                     .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "7c-unet_skip_use_cnn",                  unet_skip_use_cnn)                 .all_single_see_final_rec_analyze().analyze_tensorboard(reset_board_dir=True)

    # Bm_Rec_exps_analyze(ana_title + "8_1-unet_range_mae",                  unet_range_mae)                 .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "8_2-unet_range_mae_good",             unet_range_mae_good)            .all_single_see_final_rec_analyze().analyze_tensorboard()
    # Bm_Rec_exps_analyze(ana_title + "8_3-unet_range_mae_ok",               unet_range_mae_ok)              .all_single_see_final_rec_analyze().analyze_tensorboard()

    # Bm_Rec_exps_analyze(ana_title + "9_1-rect_layers_right_relu",        rect_layers_right_relu)           .all_single_see_final_rec_analyze()


    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ### Result_analyzer 的 各method測試：
    # try_c_result_analyzer = Col_results_analyzer(ana_describe="try_c_result_analyzer", col_results=[os_book_justG_mae1, os_book_justG_mae3, os_book_justG_mae6, os_book_justG_mae9, os_book_justG_mae20 ])
    # try_c_result_analyzer.analyze_col_results_single_see(0, add_loss=True, single_see_multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_single_see(0, add_loss=True, single_see_multiprocess=True)
    # try_c_result_analyzer.analyze_col_results_all_single_see(start_see=0, see_amount=1, add_loss=True)
    # try_c_result_analyzer.analyze_col_results_all_single_see_multiprocess(add_loss=True)
    # try_c_result_analyzer.analyze_col_results_multi_see([16], "train_lt", add_loss = False, multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = False, multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True, multiprocess=False)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True, multiprocess=True)
    # try_c_result_analyzer.analyze_col_results_multi_see([16,19], "train_lt", add_loss = True)


    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    # os_book_G_D_mae136 = Col_results_analyzer(ana_describe="5_1_GD_Gmae136_epoch700", col_results=[os_book_rect_1532_mae1, os_book_rect_1532_mae3, os_book_rect_1532_mae6, os_book_rect_1532_mae6_noD]); doing_analyze_2page(os_book_G_D_mae136); doing_analyze_2page(os_book_G_D_mae136)
    ##################################################################################################################
    # os_book_rect_D025目前太少，之後再加進去
    # os_book_G_D_vs_justG = Col_results_analyzer(ana_describe="5_2_GD_vs_justG", col_results=[os_book_rect_D10, os_book_rect_D05, os_book_rect_D01, os_book_rect_D00, os_book_rect_D00_justG]); doing_analyze_2page(os_book_G_D_vs_justG)
    ##################################################################################################################
    # os_book_1532data_mae136920 = Col_results_analyzer(ana_describe="5_3_justG_136920", col_results=[os_book_justG_mae1, os_book_justG_mae3, os_book_justG_mae6, os_book_justG_mae9, os_book_justG_mae20]); doing_analyze_2page(os_book_1532data_mae136920)
    ##################################################################################################################
    ### 5_4.放大縮小的比較 norm/ bigger
    # os_book_justG_bigger = Col_results_analyzer(ana_describe="5_4_justG_a_bigger", col_results=[os_book_justG_normal, os_book_justG_bigger]); doing_analyze_2page(os_book_justG_bigger)
    ##################################################################################################################
    ### 5_4.放大縮小的比較 bigger_wrong/norm/small/small2
    # os_book_justG_big_small = Col_results_analyzer(ana_describe="5_4_justG_b_bigger_smaller", col_results=[os_book_justG_bigger, os_book_justG_normal, os_book_justG_smaller, os_book_justG_smaller2]); doing_analyze_2page(os_book_justG_big_small)
    ##################################################################################################################
    ### 5_5.focus
    # os_book_focus_GD_vs_G = Col_results_analyzer(ana_describe="5_5_focus_GD_vs_G", col_results=[os_book_rect_nfocus, os_book_rect_focus, os_book_justG_nfocus, os_book_justG_focus]); doing_analyze_2page(os_book_focus_GD_vs_G)
    ##################################################################################################################
    ### 5_6.400 vs 1532
    # os_book_400 = Col_results_analyzer(ana_describe="5_6_a_400_page", col_results=[os_book_400_rect, os_book_400_justG]); doing_analyze_2page(os_book_400)
    ##################################################################################################################
    ### 5_6.400 vs 1532
    # os_book_400_vs_1532 = Col_results_analyzer(ana_describe="5_6_b_400_vs_1532_page", col_results=[os_book_400_rect, os_book_400_justG, os_book_1532_rect, os_book_1532_justG]); doing_analyze_2page(os_book_400_vs_1532)
    ##################################################################################################################
    ### 5_7.第一層 k7 vs k3
    # os_book_first_k7_vs_k3 = Col_results_analyzer(ana_describe="5_7_first_k7_vs_k3", col_results=[os_book_GD_first_k7, os_book_GD_first_k3, os_book_G_first_k7, os_book_G_first_k3 ]); doing_analyze_2page(os_book_first_k7_vs_k3)
    ##################################################################################################################
    ### 5_8a.GD + mrf比較
    # os_book_GD_mrf     = Col_results_analyzer(ana_describe="5_8a_GD_mrf"    , col_results=[os_book_GD_no_mrf, os_book_GD_mrf_79, os_book_GD_mrf_replace79]); doing_analyze_2page(os_book_GD_mrf)
    # os_book_GD_mrf_all = Col_results_analyzer(ana_describe="5_8a_GD_mrf_all", col_results=[os_book_GD_no_mrf, os_book_GD_mrf_7 , os_book_GD_mrf_79, os_book_GD_mrf_replace7, os_book_GD_mrf_replace79]); doing_analyze_2page(os_book_GD_mrf_all)
    ##################################################################################################################
    ### 5_8b._1 G + mrf比較
    # os_book_G_mrf                      = Col_results_analyzer    (ana_describe="5_8b_1_G_mrf"                , col_results=[os_book_G_no_mrf, os_book_G_mrf_79, os_book_G_mrf_replace79]); doing_analyze_2page(os_book_G_mrf)
    # os_book_G_mrf_all                  = Col_results_analyzer    (ana_describe="5_8b_1_G_mrf_all"            , col_results=[os_book_G_no_mrf, os_book_G_mrf_7,  os_book_G_mrf_79, os_book_G_mrf_replace7, os_book_G_mrf_replace79]); doing_analyze_2page(os_book_G_mrf_all)

    # os_book_8c_Gmrf_3branch            = Col_results_analyzer    (ana_describe="5_8c_Gmrf_3branch",  col_results=[os_book_G_mrf_135 ,  os_book_G_mrf_357 , os_book_G_mrf_3579 ]); doing_analyze_2page(os_book_8c_Gmrf_3branch )
    # os_book_8d_GDmrf_3branch           = Col_results_analyzer    (ana_describe="5_8d_GDmrf_3branch", col_results=[os_book_GD_mrf_135,  os_book_GD_mrf_357, os_book_GD_mrf_3579]); doing_analyze_2page(os_book_8d_GDmrf_3branch)
    # os_book_5_8cd                      = Row_col_results_analyzer(ana_describe="5_8cd", row_col_results=[[os_book_G_mrf_135,  os_book_G_mrf_357, os_book_G_mrf_3579],
    #                                                                                                      [os_book_GD_mrf_135,  os_book_GD_mrf_357, os_book_GD_mrf_3579]])
    # os_book_5_8cd.analyze_row_col_results_all_single_see_multiprocess(single_see_multiprocess=True, single_see_core_amount=10)

    # os_book_8b_2_G_mrf753_k3                  = Col_results_analyzer    (ana_describe="5_8b_2_G_mrf753_k3"          , col_results=[os_book_G_no_mrf_firstk3, os_book_G_mrf_7_firstk3,  os_book_G_mrf_5_firstk3,  os_book_G_mrf_3_firstk3 ]); doing_analyze_2page(os_book_8b_2_G_mrf753_k3)
    # os_book_8b_3_G_mrf97_75_53                = Col_results_analyzer    (ana_describe="5_8b_3_G_mrf97_75_53"        , col_results=[os_book_G_no_mrf_firstk3, os_book_G_mrf_79_firstk3, os_book_G_mrf_57_firstk3, os_book_G_mrf_35_firstk3]); doing_analyze_2page(os_book_8b_3_G_mrf97_75_53)
    # os_book_8b_4_G_mrf_replace753             = Col_results_analyzer    (ana_describe="5_8b_4_G_mrf_replace753"     , col_results=[os_book_G_no_mrf,  os_book_G_mrf_replace7,  os_book_G_mrf_replace5,  os_book_G_mrf_replace3 ]); doing_analyze_2page(os_book_8b_4_G_mrf_replace753)
    # os_book_8b_5_G_mrf_replace97_75_53        = Col_results_analyzer    (ana_describe="5_8b_5_G_mrf_replace97_75_53", col_results=[os_book_G_no_mrf,  os_book_G_mrf_replace79, os_book_G_mrf_replace57, os_book_G_mrf_replace35]); doing_analyze_2page(os_book_8b_5_G_mrf_replace97_75_53)
    # os_book_8b_6a_Gmrf7_1_2branch_try_replace = Col_results_analyzer    (ana_describe="5_8b_6a_Gmrf7_1_2branch_try_replace", col_results=[os_book_G_mrf_7_firstk3,  os_book_G_mrf_79_firstk3, os_book_G_mrf_replace7, os_book_G_mrf_replace79]); doing_analyze_2page(os_book_8b_6a_Gmrf7_1_2branch_try_replace)
    # os_book_8b_6b_Gmrf5_1_2branch_try_replace = Col_results_analyzer    (ana_describe="5_8b_6b_Gmrf5_1_2branch_try_replace", col_results=[os_book_G_mrf_5_firstk3,  os_book_G_mrf_57_firstk3, os_book_G_mrf_replace5, os_book_G_mrf_replace57]); doing_analyze_2page(os_book_8b_6b_Gmrf5_1_2branch_try_replace)
    # os_book_8b_6c_Gmrf3_1_2branch_try_replace = Col_results_analyzer    (ana_describe="5_8b_6c_Gmrf3_1_2branch_try_replace", col_results=[os_book_G_mrf_3_firstk3,  os_book_G_mrf_35_firstk3, os_book_G_mrf_replace3, os_book_G_mrf_replace35]); doing_analyze_2page(os_book_8b_6c_Gmrf3_1_2branch_try_replace)
    # os_book_5_8b_7_all                        = Row_col_results_analyzer(ana_describe="5_8b_7_all", row_col_results=[[os_book_G_mrf_7_firstk3,  os_book_G_mrf_79_firstk3, os_book_G_mrf_replace7, os_book_G_mrf_replace79],
    #                                                                                                                   [os_book_G_mrf_5_firstk3,  os_book_G_mrf_57_firstk3, os_book_G_mrf_replace5, os_book_G_mrf_replace57],
    #                                                                                                                   [os_book_G_mrf_3_firstk3,  os_book_G_mrf_35_firstk3, os_book_G_mrf_replace3, os_book_G_mrf_replace35]])
    # os_book_5_8b_7_all.analyze_row_col_results_all_single_see_multiprocess(single_see_multiprocess=True, single_see_core_amount=8)

    ##################################################################################################################
    ### 5_9a.Gk7D，D_concat_try and k3 or k4
    # os_book_Gk7D_concat_try_and_k3_k4          = Col_results_analyzer(ana_describe="5_9a_Gk7D_D_concat_try_and_k3,4"         , col_results=[os_book_Gk7_D_concat_k4, os_book_Gk7_D_concat_k3, os_book_Gk7_D_no_concat_k4, os_book_Gk7_D_no_concat_k3]);                   doing_analyze_2page(os_book_Gk7D_concat_try_and_k3_k4)
    # os_book_Gk7D_concat_try_and_k3_k4_and_no_D = Col_results_analyzer(ana_describe="5_9a_Gk7D_D_concat_try_and_k3,4_and_no_D", col_results=[os_book_Gk7_D_concat_k4, os_book_Gk7_D_concat_k3, os_book_Gk7_D_no_concat_k4, os_book_Gk7_D_no_concat_k3, os_book_Gk7_no_D]); doing_analyze_2page(os_book_Gk7D_concat_try_and_k3_k4_and_no_D)

    ### 5_9b.Gk3D，D_concat_try and k3 or k4
    # os_book_Gk3D_concat_try_and_k3_k4          = Col_results_analyzer(ana_describe="5_9b_Gk3D_D_concat_try_and_k3,4"         , col_results=[os_book_Gk3_D_concat_k4, os_book_Gk3_D_concat_k3, os_book_Gk3_D_no_concat_k4, os_book_Gk3_D_no_concat_k3]);                   doing_analyze_2page(os_book_Gk3D_concat_try_and_k3_k4)
    # os_book_Gk3D_concat_try_and_k3_k4_and_no_D = Col_results_analyzer(ana_describe="5_9b_Gk3D_D_concat_try_and_k3,4_and_no_D", col_results=[os_book_Gk3_D_concat_k4, os_book_Gk3_D_concat_k3, os_book_Gk3_D_no_concat_k4, os_book_Gk3_D_no_concat_k3, os_book_Gk3_no_D]); doing_analyze_2page(os_book_Gk3D_concat_try_and_k3_k4_and_no_D)

    ##################################################################################################################
    ### 5_10.GD，D保持train 1次，G train 1,3,5次 比較
    # os_book_D1Gmany                     = Col_results_analyzer(ana_describe="5_10_GD_D", col_results=[os_book_D1G1, os_book_D1G3, os_book_D1G5]); doing_analyze_2page(os_book_D1Gmany)
    # os_book_GD_D_train1_G_train_135     = Col_results_analyzer(ana_describe="5_10_GD_D_train1_G_train_135"  , col_results=[os_book_D1G1 , os_book_D1G3, os_book_D1G5 ]); doing_analyze_2page(os_book_GD_D_train1_G_train_135)
    ##################################################################################################################
    # os_book_11a_G_res_try                  = Col_results_analyzer(ana_describe="5_11a_G_res_try"               , col_results=[os_book_Gk3_res              , os_book_Gk3_no_res               ]); doing_analyze_2page(os_book_11a_G_res_try)
    # os_book_11b_G_mrf357_res_try           = Col_results_analyzer(ana_describe="5_11b_G_mrf357_res_try"          , col_results=[os_book_G_mrf_357_res        , os_book_G_mrf_357_no_res         ]); doing_analyze_2page(os_book_11b_G_mrf357_res_try)
    # os_book_11c_Gk3_Dk4_no_concat_res_try  = Col_results_analyzer(ana_describe="5_11c_Gk3_Dk4_no_concat_res_try" , col_results=[os_book_Gk3_Dk4_no_concat_res, os_book_Gk3_Dk4_no_concat_no_res ]); doing_analyze_2page(os_book_11c_Gk3_Dk4_no_concat_res_try)
    ##################################################################################################################
    # os_book_12_Gk3_resb_num           = Col_results_analyzer(ana_describe="5_12_Gk3_resb_num"           , col_results=[os_book_Gk3_resb0 , os_book_Gk3_resb1, os_book_Gk3_resb7, os_book_Gk3_resb9, os_book_Gk3_resb20]); doing_analyze_2page(os_book_12_Gk3_resb_num)
    # os_book_12_Gk3_resb_num_11_512ep  = Col_results_analyzer(ana_describe="5_12_Gk3_resb_num_11_512ep"  , col_results=[os_book_Gk3_resb0 , os_book_Gk3_resb1, os_book_Gk3_resb7, os_book_Gk3_resb9, os_book_Gk3_resb11,  os_book_Gk3_resb20 ]); doing_analyze_2page(os_book_12_Gk3_resb_num_11_512ep)

    ##################################################################################################################
    # os_book_13a_Gk3_coord_conv            = Col_results_analyzer(ana_describe="5_13a_Gk3_coord_conv"            , col_results=[os_book_Gk3_no_coord_conv           , os_book_Gk3_coord_conv_first, os_book_Gk3_coord_conv_first_end, os_book_Gk3_coord_conv_all]); doing_analyze_2page(os_book_13a_Gk3_coord_conv)
    # os_book_13b_Gk3_mrf357_coord_conv     = Col_results_analyzer(ana_describe="5_13b_Gk3_mrf357_coord_conv"    , col_results=[os_book_Gk3_mrf_357_no_coord_conv   , os_book_Gk3_mrf357_coord_conv_first, os_book_Gk3_mrf357_coord_conv_all]); doing_analyze_2page(os_book_13b_Gk3_mrf357_coord_conv)



    ########################################################################################################################################################################
    ########################################################################################################################################################################
    # check_analyze([
    #     os_book_8b_2_G_mrf753_k3,
    #     os_book_8b_3_G_mrf97_75_53,
    #     os_book_8b_4_G_mrf_replace753,
    #     os_book_8b_5_G_mrf_replace97_75_53,
    #     os_book_8b_6a_Gmrf7_1_2branch_try_replace,
    #     os_book_8b_6b_Gmrf5_1_2branch_try_replace,
    #     os_book_8b_6c_Gmrf3_1_2branch_try_replace])
    ########################################################################################################################################################################
    ########################################################################################################################################################################


    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ### 分析 bg 和 gt_color
    # bg_and_gt_color_analyze = Col_results_analyzer(ana_describe="4_1-bg_and_gt_color_analyze", col_results=bg_and_gt_color_results)  #  ; doing_analyze_2page(os_book_G_D_mae136); doing_analyze_2page(os_book_G_D_mae136)

    ### 覺得好像可以省，因為看multi的比較方便ˊ口ˋ 不過single還是有好處：可以放很大喔！覺得有空再生成好了～
    # bg_and_gt_color_analyze.analyze_col_results_all_single_see_multiprocess(bg_and_gt_color_results)  ### 覺得好像可以省，因為看multi的比較方便ˊ口ˋ

    ### 這裡是在測試 col_results_multi_see 一次看多少row比較好，結論是4
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([14, 15],             "test_str_2img")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([12, 14, 15],         "test_str_3img")  ### 覺得3不錯！
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([11, 12, 14, 15],     "test_str_4img")  ### 覺得4不錯！且可看lt,rt,ld,rd
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([11, 12, 13, 14, 15], "test_str_5img")

    ### 一次看多see
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([ 1,  2,  4,  5], "test_img")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([ 6,  7,  9, 10], "test_lin")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([11, 12, 14, 15], "test_str")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([17, 18, 19, 20], "train_img")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([22, 23, 24, 25], "train_lin")
    # bg_and_gt_color_analyze.analyze_col_results_multi_see([28, 29, 30, 31], "train_str")


    #################################################################################################################################
    ### 分析 mrf loss mae的比例
    # mrf_loss_analyze = Row_col_results_analyzer(ana_describe="4_3-mrf_loss_analyze", row_col_results=mrf_r_c_results)  #  ; doing_analyze_2page(os_book_G_D_mae136); doing_analyze_2page(os_book_G_D_mae136)
    # mrf_loss_analyze.analyze_row_col_results_all_single_see_multiprocess()

    #################################################################################################################################
    ### 分析 mrf 取代 第一層7
    # mrf_replace7_analyze = Col_results_analyzer(ana_describe="4_4-mrf_replace7_analyze", col_results=mrf_replace7_results)  #  ; doing_analyze_2page(os_book_G_D_mae136); doing_analyze_2page(os_book_G_D_mae136)
    ### 覺得好像可以省，因為看multi的比較方便ˊ口ˋ 不過single還是有好處：可以放很大喔！覺得有空再生成好了～
    # mrf_replace7_analyze.analyze_col_results_all_single_see_multiprocess(mrf_replace7_results)

    # ## 一次看多see
    # mrf_replace7_analyze.analyze_col_results_multi_see([ 1,  2,  4,  5], "test_img")
    # mrf_replace7_analyze.analyze_col_results_multi_see([ 6,  7,  9, 10], "test_lin")
    # mrf_replace7_analyze.analyze_col_results_multi_see([11, 12, 14, 15], "test_str")
    # mrf_replace7_analyze.analyze_col_results_multi_see([17, 18, 19, 20], "train_img")
    # mrf_replace7_analyze.analyze_col_results_multi_see([22, 23, 24, 25], "train_lin")
    # mrf_replace7_analyze.analyze_col_results_multi_see([28, 29, 30, 31], "train_str")
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    ### 這個好像刪掉找不到了
    ### 分析 os_book
    # os_book_results = [os_book]
    # os_book_analyze = Result_analyzer("os_book")

    # os_book.save_all_single_see_as_matplot_visual_multiprocess()

    ## 一次看多see
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 0, 1, 2, 3], "test_lt")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 4, 5, 6, 7], "test_rt")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 4, 5, 6, 7], "test_rt")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [ 6, 7, 9,10], "test_lin")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [11,12,14,15], "test_str")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [17,18,19,20], "train_img")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [22,23,24,25], "train_lin")
    # os_book_analyze.analyze_col_results_multi_see(os_book_results, [28,29,30,31], "train_str")
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################
    #################################################################################################################################



    #########################################################################################################
    ### Col_results_analyzer 的使用範例
    # try_c_results = [mrf_7_9_1, mrf_7_11_1, mrf_9_11_1, mrf_13579_1]
    # try_c_result_multi_see = Col_results_analyzer(ana_describe="try_c_result_multi_see", col_results=try_c_results)  #  ; doing_analyze_2page(os_book_G_D_mae136); doing_analyze_2page(os_book_G_D_mae136)

    # try_c_result_multi_see.analyze_col_results_multi_see([1, 3, 5], "see_1_3_5_jpg_then_crop")
    # try_c_result_multi_see.analyze_col_results_multi_see([1, 3, 5], "see_1_3_5_jpg")
    # try_c_result_multi_see.analyze_col_results_single_see(1)
    ##################################################################################################################
    ### Row_col_results_analyzer 的使用範例
    # try_r_c_ana = Row_col_results_analyzer(ana_describe="try_row_col_results",row_col_results = [[os_book_justG_mae1 , os_book_justG_mae3],
    #                                                                                              [os_book_justG_mae20, os_book_justG_mae6] ])
    # try_r_c_ana.analyze_row_col_results_single_see(2)
