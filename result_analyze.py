import sys 
sys.path.append("kong_util")
from util import get_dir_certain_file_name, matplot_visual_one_row_imgs
from build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build
from video_from_img import Video_combine_from_dir
from step0_access_path import access_path
import cv2
import time 




class Result:
    def __init__(self, result_dir_name, describe):
        self.result_dir = access_path + "result/" + result_dir_name
        self.describe = describe
        self.see_dirs = [self.result_dir + "/" + "see-%03i"%see_num for see_num in range(32) ]
        self.ckpt_dir = self.result_dir + "/ckpt"
        self.logs_dir = self.result_dir + "/logs"

    def get_certain_see_file_names(self, see_num):
        return get_dir_certain_file_name(self.see_dirs[see_num], ".jpg")

    def save_see_as_jpg(self, see_num):
        Save_as_jpg(self.see_dirs[see_num], self.see_dirs[see_num],delete_ord_file=True)
    
    def save_see_as_avi(self, see_num):
        Video_combine_from_dir(self.see_dirs[see_num], self.see_dirs[see_num], "0-combine_jpg_tail_long.avi", tail_long=True)

    def save_all_see_as_avi(self):
        for go_see in range(32): Video_combine_from_dir(self.see_dirs[go_see], self.see_dirs[go_see], "0-combine_jpg_tail_long.avi", tail_long=True)

    def save_see_as_matplot_visual(self, see_num):
        start_time = time.time()
        matplot_visual_dir = self.see_dirs[see_num] + "/" + "matplot_visual" ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(matplot_visual_dir)                        ### 建立 存結果的資料夾

        certain_see_file_names = self.get_certain_see_file_names(see_num) ### 取得 第一組結果內的 第see-num個 資料夾 內的所有影像 檔名
        print("processing see_num:", see_num)
        in_img = cv2.imread(self.see_dirs[see_num] + "/" + certain_see_file_names[0]) ### 要記得see的第一張存的是 輸入的in影像
        gt_img = cv2.imread(self.see_dirs[see_num] + "/" + certain_see_file_names[1]) ### 要記得see的第二張存的是 輸出的gt影像
        for go_img, certain_see_file_name in enumerate(certain_see_file_names):
            if(go_img>=2): ### 第三張 才開始存 epoch影像喔！
                print(".",end="")                 ### 顯示進度用
                if(go_img+1) % 100 == 0: print()  ### 顯示進度用

                epoch = go_img-2  ### 第三張 才開始存 epoch影像喔！所以epoch的數字 是go_img-2
                img = cv2.imread(self.see_dirs[see_num] + "/" + certain_see_file_name)        ### 第see-num個 資料夾 內的影像 讀出來                
                matplot_visual_one_row_imgs(img_titles=["dis_img", self.describe, "gt_img"],  ### 把每張圖要顯示的字包成list 
                                            imgs      =[  in_img ,      img   , gt_img],      ### 把要顯示的每張圖包成list
                                            fig_title ="epoch=%04i"%epoch,   ### 圖上的大標題
                                            dst_dir   =matplot_visual_dir,   ### 圖存哪
                                            file_name ="epoch=%04i"%epoch)   ### 檔名

        Save_as_jpg(matplot_visual_dir,matplot_visual_dir,delete_ord_file=True) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_visual_dir, matplot_visual_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    def save_all_see_as_matplot_visual(self, start_see=0):
        for go_see in range(start_see, 32):
            self.save_see_as_matplot_visual(go_see)


class Result_analyzer:
    def __init__(self, describe=""):
        self.describe = describe
        self.analyze_dir = access_path + "analyze_dir"+"/"+self.describe ### 例如 .../data_dir/analyze_dir/testtest
        Check_dir_exist_and_build(self.analyze_dir)

    ### 如果這還看不懂的話，就先去看 Result.save_see_as_matplot_visual()吧，那邊是最簡的寫法囉，這也是從那擴增來的
    def analyze_2_result_certain_see(self, result1, result2, see_num):
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + "%03i"%see_num ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(analyze_see_dir)                ### 建立 存結果的資料夾

        result1_certain_see_file_names = result1.get_certain_see_file_names(see_num) ### 取得 第一組結果內的 第see-num個 資料夾 內的所有影像 檔名
        result2_certain_see_file_names = result2.get_certain_see_file_names(see_num) ### 取得 第二組結果內的 第see-num個 資料夾 內的所有影像 檔名

        print("processing see_num:", see_num)
        in_img = cv2.imread(result1.see_dirs[see_num] + "/" + result1_certain_see_file_names[0]) ### 要記得see的第一張存的是 輸入的in影像，補充一下這裡用result2_img也沒差，因為go_see是一樣的，代表各result輸入都一樣
        gt_img = cv2.imread(result1.see_dirs[see_num] + "/" + result1_certain_see_file_names[1]) ### 要記得see的第二張存的是 輸出的gt影像，補充一下這裡用result2_img也沒差，因為go_see是一樣的，代表各result輸入都一樣
        for go_img, (result1_certain_see_file_name, result2_certain_see_file_name) in enumerate(zip(result1_certain_see_file_names, result2_certain_see_file_names)):
            if(go_img>=2): ### 第三張 才開始存 epoch影像喔！
                print(".",end="")               ### 顯示進度用
                if(go_img+1) % 100 == 0: print()  ### 顯示進度用

                epoch = go_img-2  ### 第三張 才開始存 epoch影像喔！所以epoch的數字 是go_img-2
                result1_img = cv2.imread(result1.see_dirs[see_num] + "/" + result1_certain_see_file_name) ### 把第一組結果的 第see-num個 資料夾 內的影像 讀出來
                result2_img = cv2.imread(result2.see_dirs[see_num] + "/" + result2_certain_see_file_name) ### 把第二組結果的 第see-num個 資料夾 內的影像 讀出來
                
                matplot_visual_one_row_imgs(img_titles=["dis_img", result1.describe, result2.describe, "gt_img"],  ### 把每張圖要顯示的字包成list 
                                            imgs      =[  in_img , result1_img        , result2_img   , gt_img],    ### 把要顯示的每張圖包成list
                                            fig_title ="epoch=%04i"%epoch,   ### 圖上的大標題
                                            dst_dir   =analyze_see_dir,      ### 圖存哪
                                            file_name ="epoch=%04i"%epoch)   ### 檔名

        Save_as_jpg(analyze_see_dir,analyze_see_dir,delete_ord_file=True) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)
        
    def analyze_2_result_all_see(self, result1, result2, start_see = 0):
        for go_see in range(start_see, 32):
            self.analyze_2_result_certain_see(result1, result2, go_see)


    ### 這是很general的寫法，如果沒辦法直接看懂先去看 analyze_2_result_certain_see，我是從那裡改成general的！
    def analyze_results_certain_see(self, results, see_num):
        start_time = time.time()
        analyze_see_dir = self.analyze_dir + "/" + "%03i"%see_num ### 分析結果存哪裡定位出來
        Check_dir_exist_and_build(analyze_see_dir)                ### 建立 存結果的資料夾

        results_certain_see_file_names = [] ### result1, result2, ... 各個resultX的 certain_see_file_names都存起來
        for result in results:
            results_certain_see_file_names.append( result.get_certain_see_file_names(see_num))

        print("processing see_num:", see_num)
        in_img = cv2.imread(results[0].see_dirs[see_num] + "/" + results_certain_see_file_names[0][0] )  ### 要記得see的第一張存的是 輸入的in影像，補充一下這裡用results[x]都沒差，因為go_see是一樣的，代表各result輸入都一樣，只是要注意results_certain_see_file_names[x][0]的第一個[]數字要對應到就是了！
        gt_img = cv2.imread(results[0].see_dirs[see_num] + "/" + results_certain_see_file_names[0][1] )  ### 要記得see的第二張存的是 輸出的gt影像，補充一下這裡用results[x]都沒差，因為go_see是一樣的，代表各result輸入都一樣，只是要注意results_certain_see_file_names[x][0]的第一個[]數字要對應到就是了！
        for go_img, results_certain_see_file_name in enumerate( zip( *results_certain_see_file_names)): ### 如果沒辦法直接看懂先去看 analyze_2_result_certain_see
            # print(go_img, results_certain_see_file_name) ### debug看一下而已
            if(go_img>=2): ### 第三張 才開始存 epoch影像喔！
                print(".",end="")                 ### 顯示進度用
                if (go_img+1) % 100 == 0: print() ### 顯示進度用

                epoch = go_img-2 ### 第三張 才開始存 epoch影像喔！所以epoch的數字 是go_img-2
                results_img = [] ### 把要顯示的每張圖 讀出來 且 包成list
                for go_result, result_certain_see_file_name in enumerate(results_certain_see_file_name):
                    results_img.append( cv2.imread(results[go_result].see_dirs[see_num] + "/" + result_certain_see_file_name ) ) ### 把 第go_result組結果 的 第see_num個 資料夾 內的影像讀出來
                img_titles = [ result.describe for result in results ]      ### 把每張圖要顯示的字包成list 
                matplot_visual_one_row_imgs(img_titles=["dis_img", *img_titles  ,"gt_img"],  ### 頭 加一張 dis_img，尾 加一張 gt_img
                                            imgs      =[ in_img  , *results_img , gt_img],   ### 頭 加一張 dis_img，尾 加一張 gt_img
                                            fig_title ="epoch=%04i"%epoch,   ### 圖上的大標題
                                            dst_dir   =analyze_see_dir,      ### 圖存哪
                                            file_name ="epoch=%04i"%epoch)   ### 檔名
            
        Save_as_jpg(analyze_see_dir,analyze_see_dir,delete_ord_file=True) ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(analyze_see_dir, analyze_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)

    ### 這是很general的寫法，如果沒辦法直接看懂先去看 analyze_2_result_certain_see，我是從那裡改成general的！
    def analyze_results_all_see(self, results,  start_see = 0):
        for go_see in range(start_see, 32):
            self.analyze_results_certain_see(results, go_see)


result1 = Result("type5d-real_have_see-have_bg-gt_color_20200428-153059_model5_rect2"  , describe="have_bg-gt_color")
result2 = Result("type5d-real_have_see-have_bg-gt_gray3ch_20200428-152656_model5_rect2", describe="have_bg-gt_gray")
result3 = Result("type5c-real_have_see-no_bg-gt-color_20200428-132611_model5_rect2"    , describe="no_bg-gt_color")
result4 = Result("type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2"  , describe="no_bg-gt_gray")
analyze1 = Result_analyzer(describe="pure_rect2-bg_effect")
# analyze1.analyze_2_result_certain_see(result1, result2, 0)
# analyze1.analyze_2_result_all_see(result1, result2)
# analyze1.analyze_results_certain_see([result1, result2, result3, result4], 0)
# analyze1.analyze_results_all_see([result1, result2, result3, result4],start_see=1)

# result1.save_see_as_avi(see_num=0)
# result1.save_see_as_matplot_visual(see_num=0)
result1.save_all_see_as_matplot_visual()
# result1.save_all_see_as_avi()
### 把 see 000~031 都做成影片
# from video_from_img import Video_combine_from_imgs, Video_combine_from_dir
# from build_dataset_combine import Save_as_jpg
# for i in range(32):
#     ord_dir = r"F:\Users\Lin_server\Desktop\0 data_dir\result\type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2\see-%03i\matplot_visual"%i
#     Save_as_jpg(ord_dir, ord_dir,delete_ord_file=True)
#     Video_combine_from_dir(ord_dir, ord_dir, "combine_jpg.avi")



# import numpy as np 
# from util import matplot_visual_one_row_imgs
# import cv2

# ord_dir = r"F:\Users\Lin_server\Desktop\0 data_dir\result\type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2\see-%03i"%(0)
# from util import get_dir_img
# imgs = get_dir_img(ord_dir, float_return=False)

# img1 = np.ones(shape=(500,404,3), dtype = np.uint8)
# img2 = np.ones(shape=(472,304,3), dtype = np.uint8)*125
# img3 = np.ones(shape=(384,256,3), dtype = np.uint8)*125
# titles = ["distorted_img","distorted_img"]
# imgs = [imgs[0],imgs[1],imgs[2]]

# # cv2.imshow("123", img1)
# # cv2.imshow("456", img2)
# matplot_visual_one_row_imgs(titles, imgs)
# cv2.waitKey()
