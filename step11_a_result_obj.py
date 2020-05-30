import cv2
import time 
import os
from tqdm import tqdm

import sys 
sys.path.append("kong_util")
from step0_access_path import access_path
from step06_a_datas_obj import DB_C
from util import get_dir_certain_file_name, matplot_visual_single_row_imgs, matplot_visual_multi_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir

class See:
    def __init__(self, result_dir, see_name):
        self.result_dir=result_dir
        self.see_name=see_name

        self.see_dir = self.result_dir + "/" + self.see_name
        self.see_file_names = None
        self.see_file_amount = None

        ### 不確定要不要，因為在initial就做這麼多事情好嗎~~會不會容易出錯哩~~
        Check_dir_exist_and_build(self.see_dir)
        self.get_see_file_names()  

    def get_see_file_names(self):
        self.see_file_names = get_dir_certain_file_name(self.see_dir, ".jpg")
        self.see_file_amount = len(self.see_file_names)

    def save_as_jpg(self):
        Save_as_jpg(self.see_dir, self.see_dir, delete_ord_file=True)

    def save_as_avi(self):
        Video_combine_from_dir(self.see_dir, self.see_dir, "0-combine_jpg_tail_long.avi", tail_long=True)

    def save_as_matplot_visual_during_train(self, epoch, show_msg=False):
        start_time = time.time()
        matplot_visual_dir = self.see_dir + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        if(epoch==0):Check_dir_exist_and_build_new_dir(matplot_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_file_names() ### 取得 結果內的 某個see資料夾 內的所有影像 檔名
        
        in_img = cv2.imread(self.see_dir + "/" + self.see_file_names[0]) ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_dir + "/" + self.see_file_names[1]) ### 要記得see的第二張存的是 輸出的gt影像
        img = cv2.imread(self.see_dir + "/" + self.see_file_names[epoch+2]) ### see資料夾 內的影像 該epoch產生的影像 讀出來 

        matplot_visual_single_row_imgs(img_titles=["in_img", "out_img", "gt_img"],  ### 把每張圖要顯示的字包成list 
                                        imgs      =[ in_img ,   img    ,  gt_img],      ### 把要顯示的每張圖包成list
                                        fig_title ="epoch=%04i"%epoch,   ### 圖上的大標題
                                        dst_dir   =matplot_visual_dir,   ### 圖存哪
                                        file_name ="epoch=%04i"%epoch,   ### 檔名
                                        )

        # Find_ltrd_and_crop(matplot_visual_dir, matplot_visual_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        # Save_as_jpg(matplot_visual_dir,matplot_visual_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        # Video_combine_from_dir(matplot_visual_dir, matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        if(show_msg): print(f"processing {self.see_name}, cost_time:{time.time() - start_time}")

    def save_as_matplot_visual_after_train(self):
        start_time = time.time()
        matplot_visual_dir = self.see_dir + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(matplot_visual_dir)      ### 建立 存結果的資料夾

        self.get_see_file_names() ### 取得 結果內的 某個see資料夾 內的所有影像 檔名
        print("processing %s"%self.see_name)
        in_img = cv2.imread(self.see_dir + "/" + self.see_file_names[0]) ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_dir + "/" + self.see_file_names[1]) ### 要記得see的第二張存的是 輸出的gt影像
        for go_img, certain_see_file_name in enumerate(tqdm(self.see_file_names)):
            if(go_img>=2): ### 第三張 才開始存 epoch影像喔！
                epoch = go_img-2  ### 第三張 才開始存 epoch影像喔！所以epoch的數字 是go_img-2
                img = cv2.imread(self.see_dir + "/" + certain_see_file_name)        ### see資料夾 內的影像 讀出來                
                matplot_visual_single_row_imgs(img_titles=["in_img", "out_img", "gt_img"],  ### 把每張圖要顯示的字包成list 
                                               imgs      =[ in_img ,   img    ,  gt_img],      ### 把要顯示的每張圖包成list
                                               fig_title ="epoch=%04i"%epoch,   ### 圖上的大標題
                                               dst_dir   =matplot_visual_dir,   ### 圖存哪
                                               file_name ="epoch=%04i"%epoch)   ### 檔名

        Find_ltrd_and_crop(matplot_visual_dir, matplot_visual_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_visual_dir,matplot_visual_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_visual_dir, matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)



class Result_init_builder:
    def __init__(self, result=None):
        if(result is None):
            self.result = Result()
        else:
            self.result = result 
        
    def build(self):
        return self.result

class Result_sees_builder(Result_init_builder):
    def _build_sees(self, sees_ver):
        if  (sees_ver == "sees_ver1"):
            self.result.sees = [  See(self.result.result_dir, "see-%03i"%see_num) for see_num in range(32) ]
        elif(sees_ver == "sees_ver2"):
            self.result.sees = [  See(self.result.result_dir, "see_000-test_emp"), 
                           See(self.result.result_dir, "see_001-test_img"),See(self.result.result_dir, "see_002-test_img"),See(self.result.result_dir, "see_003-test_img"),See(self.result.result_dir, "see_004-test_img"),See(self.result.result_dir, "see_005-test_img"),
                           See(self.result.result_dir, "see_006-test_lin"),See(self.result.result_dir, "see_007-test_lin"),See(self.result.result_dir, "see_008-test_lin"),See(self.result.result_dir, "see_009-test_lin"),See(self.result.result_dir, "see_010-test_lin"),
                           See(self.result.result_dir, "see_011-test_str"),See(self.result.result_dir, "see_012-test_str"),See(self.result.result_dir, "see_013-test_str"),See(self.result.result_dir, "see_014-test_str"),See(self.result.result_dir, "see_015-test_str"),
                           See(self.result.result_dir, "see_016-train_emp"),
                           See(self.result.result_dir, "see_017-train_img"),See(self.result.result_dir, "see_018-train_img"),See(self.result.result_dir, "see_019-train_img"),See(self.result.result_dir, "see_020-train_img"),See(self.result.result_dir, "see_021-train_img"),
                           See(self.result.result_dir, "see_022-train_lin"),See(self.result.result_dir, "see_023-train_lin"),See(self.result.result_dir, "see_024-train_lin"),See(self.result.result_dir, "see_025-train_lin"),See(self.result.result_dir, "see_026-train_lin"),
                           See(self.result.result_dir, "see_027-train_str"),See(self.result.result_dir, "see_028-train_str"),See(self.result.result_dir, "see_029-train_str"),See(self.result.result_dir, "see_030-train_str"),See(self.result.result_dir, "see_031-train_str")]
        elif(sees_ver == "sees_ver3"):
            self.result.sees = [  See(self.result.result_dir, "see_000-test_lt1"),See(self.result.result_dir, "see_001-test_lt2"),See(self.result.result_dir, "see_002-test_lt3"),See(self.result.result_dir, "see_003-test_lt4"),
                            See(self.result.result_dir, "see_004-test_rt1"),See(self.result.result_dir, "see_005-test_rt2"),See(self.result.result_dir, "see_006-test_rt3"),See(self.result.result_dir, "see_007-test_rt4"),
                            See(self.result.result_dir, "see_008-test_ld1"),See(self.result.result_dir, "see_009-test_ld2"),See(self.result.result_dir, "see_010-test_ld3"),See(self.result.result_dir, "see_011-test_ld4"),
                            See(self.result.result_dir, "see_012-test_rd1"),See(self.result.result_dir, "see_013-test_rd2"),See(self.result.result_dir, "see_014-test_rd3"),See(self.result.result_dir, "see_015-test_rd4"),
                            See(self.result.result_dir, "see_016-train_lt1"),See(self.result.result_dir, "see_017-train_lt2"),See(self.result.result_dir, "see_018-train_lt3"),See(self.result.result_dir, "see_019-train_lt4"),
                            See(self.result.result_dir, "see_020-train_rt1"),See(self.result.result_dir, "see_021-train_rt2"),See(self.result.result_dir, "see_022-train_rt3"),See(self.result.result_dir, "see_023-train_rt4"),
                            See(self.result.result_dir, "see_024-train_ld1"),See(self.result.result_dir, "see_025-train_ld2"),See(self.result.result_dir, "see_026-train_ld3"),See(self.result.result_dir, "see_027-train_ld4"),
                            See(self.result.result_dir, "see_028-train_rd1"),See(self.result.result_dir, "see_029-train_rd2"),See(self.result.result_dir, "see_030-train_rd3"),See(self.result.result_dir, "see_031-train_rd4")]
    
        self.result.see_amount = len(self.result.sees)
        self.result.see_file_amount = self.result.sees[0].see_file_amount ### 應該是每個see都一樣多檔案，所以就挑第一個拿他的see_file_amount就好囉～


class Result_train_builder(Result_sees_builder):
    ###     3b.用result_name 裡面的 DB_CATEGORY 來決定sees_ver
    def _use_result_name_find_sees_ver(self):
        db_c = self.result.result_name.split("/")[-1].split("-")[0] ### "/"是為了抓底層資料夾，"-"是為了抓 DB_CATEGORY
        sees_ver = ""
        if  (db_c in [DB_C.type5c_real_have_see_no_bg_gt_color_gray3ch.value, 
                      DB_C.type5d_real_have_see_have_bg_gt_color_gray3ch.value,
                      DB_C.type6_h_384_w_256_smooth_curl_fold_and_page.value  ]): sees_ver="sees_ver2"
        elif(db_c in [DB_C.type7_h472_w304_real_os_book.value, 
                      DB_C.type7b_h500_w332_real_os_book.value]): sees_ver="sees_ver3"
        else: sees_ver = "sees_ver1"
        return sees_ver

    ### 設定方式二：直接給 result_name來設定( result_name格式可以參考 _get_result_name_by_exp )
    def set_by_result_name(self, result_name):
        ### 3a.用result_name 來設定ckpt, logs 的資料夾
        self.result.result_name = result_name ### 如果他被包在某個資料夾，該資料夾也算名字喔！ex：5_just_G_mae1369/type7b_h500_w332_real_os_book-20200525_225555-just_G-1532data_mae9_127.35_copy
        self.result.result_dir  = access_path + "result/" + result_name
        self.result.ckpt_dir = self.result.result_dir + "/ckpt"
        self.result.logs_dir = self.result.result_dir + "/logs"

        ### 3b.用result_name 來決定sees_ver，之後再去建立sees
        self.result.sees_ver = self._use_result_name_find_sees_ver()
        self._build_sees(self.result.sees_ver)
        return self

    ###     1.用 exp 資訊來 決定 result_name
    def _get_result_name_by_exp(self, exp):
        import datetime
        ### 自動決定 result_name，再去做進一步設定
        ### 大概長這樣 type1_h=256,w=256_complex_"describe_mid"_20200328-215330_model5_rect2_"describe_end"
        result_name_element = [exp.db_obj.category.value]
        if(exp.describe_mid is not None): result_name_element += [exp.describe_mid]
        result_name_element += [datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), exp.model_obj.model_name.value]
        if(exp.describe_end is not None): result_name_element += [exp.describe_end]
        return "-".join(result_name_element)### result資料夾，裡面放checkpoint和tensorboard資料夾

    ### 設定方式一：用exp_obj來設定
    def set_by_exp(self, exp):
        ### 1.用 exp 資訊來 決定 result_name
        self.result.result_name = self._get_result_name_by_exp(exp)

        ### 2.決定好 result_name 後，用result_name來設定Result
        self.set_by_result_name(self.result.result_name)
        return self

class Result_plot_builder(Result_train_builder):
    def set_plot_title(self, plot_title):
        self.result.plot_title = plot_title
        return self

class Result_builder(Result_plot_builder):pass

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
        self.plot_title = None ### 這是給matplot用的title

    # def rename_see1_to_see2(self):
    #     for go_see in range(self.see_amount):
    #         if(os.path.isdir(self.sees1[go_see].see_dir)):
    #             print("rename_ord:", self.sees1[go_see].see_dir)
    #             print("rename_dst:", self.sees2[go_see].see_dir)
    #             os.rename(self.sees1[go_see].see_dir, self.sees2[go_see].see_dir)
    def save_single_see_as_matplot_visual(self, see_num):
        self.sees[see_num].save_as_matplot_visual_after_train()
            
    def save_all_single_see_as_matplot_visual(self, start_index, amount):
        print("start_index", start_index)
        for see in self.sees[start_index: start_index+amount]:
            see.save_as_matplot_visual_after_train()

    def save_all_single_see_as_matplot_visual_multiprocess(self):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=8 ,task_amount=self.see_amount, task=self.save_all_single_see_as_matplot_visual)

    ##############################################################################################################################
    def save_multi_see_as_matplot_visual(self, see_nums, save_name):
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
        titles = ["in_img", self.plot_title, "gt_img"]
        r_c_titles = [titles] ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs
        print("doing save_multi_see_as_matplot_visual")
        self._draw_multi_see_multiprocess(see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, core_amount=8, task_amount=self.see_file_amount)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(matplot_multi_see_dir, matplot_multi_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_multi_see_dir, matplot_multi_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_multi_see_dir, matplot_multi_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def _draw_multi_see(self, start_img, img_amount, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir ):
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
                matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles, 
                                              rows_cols_imgs   = r_c_imgs,
                                              fig_title        ="epoch=%04i"%epoch,   ### 圖上的大標題
                                              dst_dir          = matplot_multi_see_dir, 
                                              file_name        ="epoch=%04i"%epoch,
                                              bgr2rgb          =True)

    def _draw_multi_see_multiprocess(self, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, core_amount=8, task_amount=600):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._draw_multi_see, task_args=[see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir])


if(__name__=="__main__"):
    ## Result 的 各method測試：
    os_book = Result_builder().set_by_result_name("5_just_G_mae1369/type7b_h500_w332_real_os_book-20200525_225555-just_G-1532data_mae9_127.35_copy").set_plot_title("mae9").build()
    os_book.save_multi_see_as_matplot_visual([29, 30, 31],"train_rd")
    os_book.save_single_see_as_matplot_visual(see_num=12)
