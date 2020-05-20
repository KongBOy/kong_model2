import cv2
import time 
import os
from tqdm import tqdm

import sys 
sys.path.append("kong_util")
from step0_access_path import access_path
from util import get_dir_certain_file_name, matplot_visual_single_row_imgs, matplot_visual_multi_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir

class See:
    def __init__(self, result_dir, see_name):
        self.result_dir=result_dir
        self.see_name=see_name

        self.see_dir = self.result_dir + "/" + self.see_name
        # self.see_file_names = self.get_see_file_names()

    def get_see_file_names(self):
        self.see_file_names = get_dir_certain_file_name(self.see_dir, ".jpg")

    def save_as_jpg(self):
        Save_as_jpg(self.see_dir, self.see_dir, delete_ord_file=True)

    def save_as_avi(self):
        Video_combine_from_dir(self.see_dir, self.see_dir, "0-combine_jpg_tail_long.avi", tail_long=True)

    def save_as_matplot_visual(self):
        start_time = time.time()
        matplot_visual_dir = self.see_dir + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(matplot_visual_dir)               ### 建立 存結果的資料夾

        self.get_see_file_names() ### 取得 結果內的 某個see資料夾 內的所有影像 檔名
        print("processing %s"%self.see_name)
        in_img = cv2.imread(self.see_dir + "/" + self.see_file_names[0]) ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_dir + "/" + self.see_file_names[1]) ### 要記得see的第二張存的是 輸出的gt影像
        for go_img, certain_see_file_name in enumerate(tqdm(self.see_file_names)):
            if(go_img>=2): ### 第三張 才開始存 epoch影像喔！
                print(".",end="")                 ### 顯示進度用
                if(go_img+1) % 100 == 0: print()  ### 顯示進度用

                epoch = go_img-2  ### 第三張 才開始存 epoch影像喔！所以epoch的數字 是go_img-2
                img = cv2.imread(self.see_dir + "/" + certain_see_file_name)        ### see資料夾 內的影像 讀出來                
                matplot_visual_single_row_imgs(img_titles=["in_img", "out_img", "gt_img"],  ### 把每張圖要顯示的字包成list 
                                               imgs      =[  in_img ,      img   , gt_img],      ### 把要顯示的每張圖包成list
                                               fig_title ="epoch=%04i"%epoch,   ### 圖上的大標題
                                               dst_dir   =matplot_visual_dir,   ### 圖存哪
                                               file_name ="epoch=%04i"%epoch)   ### 檔名

        Find_ltrd_and_crop(matplot_visual_dir, matplot_visual_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_visual_dir,matplot_visual_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_visual_dir, matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)



class Result:
    def __init__(self, result_name, r_describe=None):
        if(r_describe is None):self.result_dir = access_path + "result/" + result_name
        else                  :self.result_dir = access_path + "result/" + result_name + "-" + r_describe
        self.ckpt_dir = self.result_dir + "/ckpt"
        self.logs_dir = self.result_dir + "/logs"
        
        self.r_describe = r_describe
        self.see_dirs = [self.result_dir + "/" + "see-%03i"%see_num for see_num in range(32) ]
        self.sees1 = [  See(self.result_dir, "see-%03i"%see_num) for see_num in range(32) ]
        self.sees2 = [  See(self.result_dir, "see_000-test_emp"), 
                        See(self.result_dir, "see_001-test_img"),
                        See(self.result_dir, "see_002-test_img"),
                        See(self.result_dir, "see_003-test_img"),
                        See(self.result_dir, "see_004-test_img"),
                        See(self.result_dir, "see_005-test_img"),
                        See(self.result_dir, "see_006-test_lin"),
                        See(self.result_dir, "see_007-test_lin"),
                        See(self.result_dir, "see_008-test_lin"),
                        See(self.result_dir, "see_009-test_lin"),
                        See(self.result_dir, "see_010-test_lin"),
                        See(self.result_dir, "see_011-test_str"),
                        See(self.result_dir, "see_012-test_str"),
                        See(self.result_dir, "see_013-test_str"),
                        See(self.result_dir, "see_014-test_str"),
                        See(self.result_dir, "see_015-test_str"),
                        See(self.result_dir, "see_016-train_emp"),
                        See(self.result_dir, "see_017-train_img"),
                        See(self.result_dir, "see_018-train_img"),
                        See(self.result_dir, "see_019-train_img"),
                        See(self.result_dir, "see_020-train_img"),
                        See(self.result_dir, "see_021-train_img"),
                        See(self.result_dir, "see_022-train_lin"),
                        See(self.result_dir, "see_023-train_lin"),
                        See(self.result_dir, "see_024-train_lin"),
                        See(self.result_dir, "see_025-train_lin"),
                        See(self.result_dir, "see_026-train_lin"),
                        See(self.result_dir, "see_027-train_str"),
                        See(self.result_dir, "see_028-train_str"),
                        See(self.result_dir, "see_029-train_str"),
                        See(self.result_dir, "see_030-train_str"),
                        See(self.result_dir, "see_031-train_str")]
        self.see_amount = len(self.sees2)
        
    @staticmethod
    def new_from_result_name(result_name):
        ### 第零階段：決定result, logs, ckpt 存哪裡 並 把source code存起來s
        ###    train_reload和test 根據 "result_name"，直接去放result的資料夾複製囉！
        return Result(result_name=result_name)

    @staticmethod
    def new_from_experiment(exp):
        import datetime
        ### 大概長這樣 type1_h=256,w=256_complex_"describe_mid"_20200328-215330_model5_rect2_"describe_end"
        result_name_element = [exp.db_obj.category.value]
        if(exp.describe_mid is not None): result_name_element += [exp.describe_mid]
        result_name_element += [datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), exp.model_obj.model_name.value]
        if(exp.describe_end is not None): result_name_element += [exp.describe_end]

        result_name = "_".join(result_name_element)### result資料夾，裡面放checkpoint和tensorboard資料夾
        return Result(result_name=result_name)
    


    def rename_see1_to_see2(self):
        for go_see in range(self.see_amount):
            if(os.path.isdir(self.sees1[go_see].see_dir)):
                print("rename_ord:", self.sees1[go_see].see_dir)
                print("rename_dst:", self.sees2[go_see].see_dir)
                os.rename(self.sees1[go_see].see_dir, self.sees2[go_see].see_dir)
            
    def save_all_see_as_matplot_visual(self, start_index, amount):
        print("start_index", start_index)
        for see in self.sees2[start_index: start_index+amount]:
            see.save_as_matplot_visual()

    def save_all_see_as_matplot_visual_multiprocess(self):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=8 ,task_amount=self.see_amount, task=self.save_all_see_as_matplot_visual)

    
class Result_analyzer:
    def __init__(self, r_describe=""):
        self.r_describe = r_describe
        self.analyze_dir = access_path + "analyze_dir"+"/"+self.r_describe ### 例如 .../data_dir/analyze_dir/testtest
        Check_dir_exist_and_build(self.analyze_dir)
    
    ########################################################################################################################################
    def _temp_c_results_see1_update_to_see2_and_get_see_file_names(self, c_results):
        ### 暫時的update see1~see2
        for result in c_results:
            result.rename_see1_to_see2()
            for see in result.sees2:
                see.get_see_file_names()

    
    def _temp_r_c_results_update_see1_to_see2_and_get_see_file_names(self, r_c_results):
        for c_results in r_c_results:
            self._temp_c_results_see1_update_to_see2_and_get_see_file_names(c_results)

    ########################################################################################################################################
    ### 單一row，同see
    def analyze_col_results_single_see(self, c_results, see_num):
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + c_results[0].sees2[see_num].see_name  ### (可以再想想好名字！)分析結果存哪裡定位出來
        Check_dir_exist_and_build(analyze_see_dir)                                       ### 建立 存結果的資料夾

        ### 暫時的update see1~see2
        self._temp_r_c_results_update_see1_to_see2_and_get_see_file_names(c_results)
        
        ### 抓 in/gt imgs
        in_imgs = cv2.imread(c_results[0].sees2[see_num].see_dir + "/" + c_results[0].sees2[see_num].see_file_names[0])
        gt_imgs = cv2.imread(c_results[0].sees2[see_num].see_dir + "/" + c_results[0].sees2[see_num].see_file_names[1])

        ### 抓 要顯示的 titles
        c_titles = ["in_img"]
        for result in c_results: c_titles.append(result.r_describe)
        c_titles += ["gt_img"]

        ### 抓  要顯示的imgs
        print("doing analyze_col_results_multi_see")
        for go_img in tqdm(range(600)):
            if(go_img >=2):
                epoch = go_img-2
                
                c_imgs   = [in_imgs]
                for result in c_results: c_imgs.append(cv2.imread(result.sees2[see_num].see_dir + "/" + result.sees2[see_num].see_file_names[go_img]))
                c_imgs += [gt_imgs]
                matplot_visual_single_row_imgs(img_titles=c_titles,
                                               imgs=c_imgs, 
                                               fig_title        ="epoch=%04i"%epoch,   ### 圖上的大標題
                                               dst_dir          = analyze_see_dir, 
                                               file_name        ="epoch=%04i"%epoch,
                                               bgr2rgb          = True)
        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def analyze_col_results_all_single_see(self,start_see, see_amount, c_results):
        for go_see in range(start_see, start_see + see_amount):
            self.analyze_col_results_single_see(c_results, go_see)

    def analyze_col_results_all_single_see_multiprocess(self, c_results, core_amount=8, task_amount=32):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self.analyze_col_results_all_single_see, task_args=[c_results])

    ########################################################################################################################################
    ### 同col同result，同row同see
    def _draw_col_results_multi_see(self, start_img, img_amount, c_results, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir ):
        print("doing analyze_col_results_multi_see")
        for go_img in tqdm(range(start_img, start_img+img_amount)):
            if(go_img >=2):
                epoch = go_img-2

                r_c_imgs = []
                for go_see_num, see_num in enumerate(see_nums):
                    c_imgs   = [in_imgs[go_see_num]]
                    for result in c_results: c_imgs.append(cv2.imread(result.sees2[see_num].see_dir + "/" + result.sees2[see_num].see_file_names[go_img]))
                    c_imgs += [gt_imgs[go_see_num]]
                    r_c_imgs.append(c_imgs)
                matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles, 
                                              rows_cols_imgs   = r_c_imgs,
                                              fig_title        ="epoch=%04i"%epoch,   ### 圖上的大標題
                                              dst_dir          = analyze_see_dir, 
                                              file_name        ="epoch=%04i"%epoch,
                                              bgr2rgb          =True)

    def _draw_col_results_multi_see_multiprocess(self, c_results, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, core_amount=8, task_amount=600):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self._draw_col_results_multi_see, task_args=( c_results, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir))

    def analyze_col_results_multi_see(self, c_results, see_nums, save_name):
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + save_name  ### (可以再想想好名字！)分析結果存哪裡定位出來
        Check_dir_exist_and_build(analyze_see_dir)                                  ### 建立 存結果的資料夾
        
        ### 暫時的update see1~see2
        self._temp_c_results_see1_update_to_see2_and_get_see_file_names(c_results)
        
        
        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for see_num in see_nums:
            in_imgs.append(cv2.imread(c_results[0].sees2[see_num].see_dir + "/" + c_results[0].sees2[see_num].see_file_names[0]))
            gt_imgs.append(cv2.imread(c_results[0].sees2[see_num].see_dir + "/" + c_results[0].sees2[see_num].see_file_names[1]))

        ### 抓 第一row的 要顯示的 titles
        c_titles = ["in_img"]
        for result in c_results: c_titles.append(result.r_describe)
        c_titles += ["gt_img"]
        r_c_titles = [c_titles] ### 還是包成r_c_titles的形式喔！

        ### 抓 row/col 要顯示的imgs
        print("doing analyze_col_results_multi_see")
        self._draw_col_results_multi_see_multiprocess(c_results, see_nums, in_imgs, gt_imgs, r_c_titles, analyze_see_dir, core_amount=8, task_amount=600)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)


    ########################################################################################################################################
    ### 各row各col 皆 不同result，但全部都看相同某個see
    def analyze_row_col_results_certain_see(self, r_c_results, see_num):
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + r_c_results[0][0].sees2[see_num].see_name ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(analyze_see_dir)                                          ### 建立 存結果的資料夾
        
        ### 暫時的update see1~see2
        self._temp_r_c_results_update_see1_to_see2_and_get_see_file_names(r_c_results)

        print("processing see_num:", see_num)
        ### 要記得see的第一張存的是 輸入的in影像，第二張存的是 輸出的gt影像
        ### 因為是certain_see → 所有的result看的是相同see，所以所有result的in/gt都一樣喔！乾脆就抓最左上角result的in/gt就好啦！
        in_img = cv2.imread(r_c_results[0][0].sees2[see_num].see_dir + "/" + r_c_results[0][0].sees2[see_num].see_file_names[0] )  ### 第一張：in_img
        gt_img = cv2.imread(r_c_results[0][0].sees2[see_num].see_dir + "/" + r_c_results[0][0].sees2[see_num].see_file_names[1] )  ### 第二張：gt_img
        for go_img in tqdm(range(600)):
            if(go_img >=2):
                epoch = go_img-2
                # print("see_num=", see_num, "go_img=", go_img)
                r_c_imgs   = [] ### r_c_imgs   抓出所要要顯示的圖   ，然後要記得每個row的第一張要放in_img，最後一張要放gt_img喔！
                r_c_titles = [] ### r_c_titles 抓出所有要顯示的標題 ，然後要記得每個row的第一張要放in_img，最後一張要放gt_img喔！
                for row_results in r_c_results:
                    c_imgs   = [in_img]   ### 每個row的第一張要放in_img
                    c_titles = ["in_img"] ### 每個row的第一張要放in_img
                    for result in row_results: ### 抓出一個row的 img 和 title
                        c_imgs.append( cv2.imread( result.sees2[see_num].see_dir + "/" + result.sees2[see_num].see_file_names[go_img] ))
                        c_titles.append(result.r_describe)
                    c_imgs += [gt_img]      ### 每個row的最後一張要放gt_img
                    c_titles += ["gt_img"]  ### 每個row的最後一張要放gt_img
                    r_c_imgs.append(c_imgs)
                    r_c_titles.append(c_titles)
                ###########################################################################################################
                matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles, 
                                              rows_cols_imgs   = r_c_imgs,
                                              fig_title        ="epoch=%04i"%epoch,   ### 圖上的大標題
                                              dst_dir          = analyze_see_dir, 
                                              file_name        ="epoch=%04i"%epoch,
                                              bgr2rgb          = True )

        Find_ltrd_and_crop(analyze_see_dir, analyze_see_dir, padding=15, search_amount=10) ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(analyze_see_dir, analyze_see_dir,delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, 40]) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def analyze_row_col_results_sees(self,start_see, see_amount,  r_c_results):
        for go_see in range(start_see, start_see + see_amount):
            self.analyze_row_col_results_certain_see(r_c_results, go_see)
    
    def analyze_row_col_results_all_sees_multiprocess(self, r_c_results, core_amount=8, task_amount=32):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount ,task_amount=task_amount, task=self.analyze_row_col_results_sees, task_args=(r_c_results))
        

if(__name__=="__main__"):
    have_bg_gt_gray_mae3  = Result("1_bg_&_gt_color/type5d-real_have_see-have_bg-gt_gray3ch_20200428-152656_model5_rect2", r_describe="have_bg_gt_gray")
    have_bg_gt_color_mae3 = Result("1_bg_&_gt_color/type5d-real_have_see-have_bg-gt_color_20200428-153059_model5_rect2"  , r_describe="have_bg_gt_color")
    no_bg_gt_color_mae3   = Result("1_bg_&_gt_color/type5c-real_have_see-no_bg-gt-color_20200428-132611_model5_rect2"    , r_describe="no_bg_gt_color")
    no_bg_gt_gray_mae3    = Result("1_bg_&_gt_color/type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2"  , r_describe="no_bg_gt_gray")

    no_mrf_mae1 = Result("2_no_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200506-064552_model5_rect2" , r_describe="no_mrf_mae1")
    no_mrf_mae3 = Result("2_no_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2" , r_describe="no_mrf_mae3")
    no_mrf_mae6 = Result("2_no_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200506-065346_model5_rect2" , r_describe="no_mrf_mae6")
    
    mrf_7_9_1   = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-190344_model6_mrf_rect2" , r_describe="mrf_7_9_mae1")
    mrf_7_9_3   = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200429-145226_model6_mrf_rect2" , r_describe="mrf_7_9_mae3")
    mrf_7_9_6   = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231036_model6_mrf_rect2" , r_describe="mrf_7_9_mae6")
    mrf_7_11_1  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-190955_model6_mrf_rect2" , r_describe="mrf_7_11_mae1")
    mrf_7_11_3  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200429-150505_model6_mrf_rect2" , r_describe="mrf_7_11_mae3")
    mrf_7_11_6  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231336_model6_mrf_rect2" , r_describe="mrf_7_11_mae6")
    mrf_9_11_1  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-190837_model6_mrf_rect2" , r_describe="mrf_9_11_mae1")
    mrf_9_11_3  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200429-145548_model6_mrf_rect2" , r_describe="mrf_9_11_mae3")
    mrf_9_11_6  = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231249_model6_mrf_rect2" , r_describe="mrf_9_11_mae6")
    mrf_13579_1 = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200504-191110_model6_mrf_rect2" , r_describe="mrf_13579_mae1")
    mrf_13579_3 = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200428-154149_model6_mrf_rect2" , r_describe="mrf_13579_mae3")
    mrf_13579_6 = Result("3_mrf_mae136/type5c-real_have_see-no_bg-gt-gray3ch_20200501-231530_model6_mrf_rect2" , r_describe="mrf_13579_mae6")


    mrf_replace7_use7   = Result("4_mrf_replace7/type5c-real_have_see-no_bg-gt-gray3ch_20200507-105001_model6_mrf_rect2" , r_describe="replace7_use7")
    mrf_replace7_use5_7 = Result("4_mrf_replace7/type5c-real_have_see-no_bg-gt-gray3ch_20200507-105739_model6_mrf_rect2" , r_describe="replace7_use5+7")
    mrf_replace7_use7_9 = Result("4_mrf_replace7/type5c-real_have_see-no_bg-gt-gray3ch_20200507-110022_model6_mrf_rect2" , r_describe="replace7_use7+9")


    ### 把 result內的 matplot_visual 壓小
    compress_results = [ 
                    # have_bg_gt_gray_mae3, 
                    # have_bg_gt_color_mae3, 
                    # no_bg_gt_color_mae3, 
                    # no_bg_gt_gray_mae3,
                    # no_mrf_mae1, 
                    # no_mrf_mae3, 
                    # no_mrf_mae6,
                    # mrf_7_9_1,
                    # mrf_7_9_3,
                    # mrf_7_9_6,
                    # mrf_7_11_1,
                    # mrf_7_11_3,
                    # mrf_7_11_6,
                    # mrf_9_11_1,
                    # mrf_9_11_3,
                    # mrf_9_11_6,
                    # mrf_13579_1,
                    # mrf_13579_3,
                    # mrf_13579_6,
                    mrf_replace7_use7,
                    mrf_replace7_use5_7,
                    mrf_replace7_use7_9
                    ]

    # for result in compress_results:
    #     print("now_doing", result.r_describe)
    #     result.rename_see1_to_see2()
    #     result.save_all_see_as_matplot_visual_multiprocess()
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
    mrf_r_c_results = [   
                          [no_mrf_mae1, mrf_7_9_1, mrf_7_11_1, mrf_9_11_1, mrf_13579_1],
                          [no_mrf_mae3, mrf_7_9_3, mrf_7_11_3, mrf_9_11_3, mrf_13579_3],
                          [no_mrf_mae6, mrf_7_9_6, mrf_7_11_6, mrf_9_11_6, mrf_13579_6]
                      ]
    mrf_loss_analyze = Result_analyzer(r_describe="mrf_loss_analyze")
    mrf_loss_analyze.analyze_row_col_results_all_sees_multiprocess(mrf_r_c_results)
    
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


    #########################################################################################################
    # mrf_c_results = [mrf_7_9_1, mrf_7_11_1, mrf_9_11_1, mrf_13579_1]
    # try_c_result_multi_see = Result_analyzer(r_describe="try_c_result_multi_see")
    # try_c_result_multi_see.analyze_col_results_multi_see(mrf_c_results, [1,3,5], "see_1_3_5_jpg_then_crop")
    # try_c_result_multi_see.analyze_col_results_multi_see(mrf_c_results, [1,3,5], "see_1_3_5_jpg")
    # try_c_result_multi_see.analyze_col_results_single_see(mrf_c_results, 1)
    

    # l,t,r,d = Find_db_left_top_right_down(try_c_result_multi_see.analyze_dir + "/" +"analyze_col_results_multi_see")
    # print(l,t,r,d)
    # Find_ltrd_and_crop(try_c_result_multi_see.analyze_dir + "/" +"analyze_col_results_multi_see", try_c_result_multi_see.analyze_dir + "/" +"analyze_col_results_multi_see", 15, search_amount=10)
