from step11_a0_see_base import See_info

from step0_access_path import JPG_QUALITY, CORE_AMOUNT_WM_VISUAL, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

from step08_b_use_G_generate_0_util import WcM_01_visual_op

from kong_util.util import get_dir_certain_file_names, move_dir_certain_file
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs, Matplot_multi_row_imgs
from kong_util.wc_util import WM_3d_plot
from kong_util.build_dataset_combine import Save_as_jpg, Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from kong_util.multiprocess_util import multi_processing_interface
from kong_util.video_from_img import Video_combine_from_dir
from multiprocessing import Process


from tqdm import tqdm
import cv2
import numpy as np
import os
import time

import datetime
import pdb

'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''

class See_wc(See_info):
    def __init__(self, result_obj, see_name):
        super(See_wc, self).__init__(result_obj, see_name)
        """
        __init__：放 Dir：..._read_dir
                          ..._write_dir
        """
        self.old1_wc_read_dir   = self.see_read_dir
        self.old1_wc_write_dir  = self.see_write_dir
        ##########################################################################################
        # self.wc_read_dir      = self.see_read_dir  + "/1_wc"  ### 之後也許東西多起來會需要獨立資料夾， 先寫起來放
        # self.wc_write_dir     = self.see_write_dir + "/1_wc"  ### 之後也許東西多起來會需要獨立資料夾， 先寫起來放
        self.wc_read_dir      = self.see_read_dir
        self.wc_write_dir     = self.see_write_dir

        self.wx_read_dir      = self.see_read_dir
        self.wx_write_dir     = self.see_write_dir
        self.wy_read_dir      = self.see_read_dir
        self.wy_write_dir     = self.see_write_dir
        self.wz_read_dir      = self.see_read_dir
        self.wz_write_dir     = self.see_write_dir

        ### 給 bm_rec 用的
        self.WM_npz_read_dir       = self.see_read_dir  + "/1_npz"
        self.WM_npz_write_dir      = self.see_write_dir + "/1_npz"

        self.WM_matplot_visual_read_dir   = self.see_read_dir  + "/2_WM_matplot_visual"
        self.WM_matplot_visual_write_dir  = self.see_write_dir + "/2_WM_matplot_visual"
        self.WM_3D_matplot_visual_read_dir   = self.see_read_dir  + "/2_WM_matplot_visual/3D_visual"
        self.WM_3D_matplot_visual_write_dir  = self.see_write_dir + "/2_WM_matplot_visual/3D_visual"

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_wc_dir(print_msg=True)

    ### 先寫起來放著
    def Change_wc_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_wc_dir, Current See:{self.see_name}")
        move_dir_certain_file(self.old1_wc_read_dir,  certain_word="epoch", certain_ext=".jpg", dst_dir=self.wc_read_dir,  print_msg=print_msg)
        move_dir_certain_file(self.old1_wc_write_dir, certain_word="epoch", certain_ext=".jpg", dst_dir=self.wc_write_dir, print_msg=print_msg)

    ### 給下一步 bm_rec 用的
    def get_wc_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.wc_names                              = get_dir_certain_file_names(self.wc_read_dir , certain_word="epoch", certain_word2="wc",              certain_ext=".jpg", print_msg=False)
        if(len(self.wc_names) == 0): self.wc_names = get_dir_certain_file_names(self.wc_read_dir , certain_word="epoch", certain_word2="W_visual",        certain_ext=".jpg", print_msg=False)
        if(len(self.wc_names) == 0): self.wc_names = get_dir_certain_file_names(self.wc_read_dir , certain_word="epoch", certain_word2="W_w_Mgt_visual",  certain_ext=".jpg", print_msg=False)
        if(len(self.wc_names) == 0): self.wc_names = get_dir_certain_file_names(self.wc_read_dir , certain_word="epoch", certain_word2="W_w_M_visual",    certain_ext=".jpg", print_msg=False)
        
        self.wx_names                              = get_dir_certain_file_names(self.wx_read_dir , certain_word="epoch", certain_word2="wx",              certain_ext=".jpg", print_msg=False)
        if(len(self.wx_names) == 0): self.wx_names = get_dir_certain_file_names(self.wx_read_dir , certain_word="epoch", certain_word2="Wx_visual",       certain_ext=".jpg", print_msg=False)
        if(len(self.wx_names) == 0): self.wx_names = get_dir_certain_file_names(self.wx_read_dir , certain_word="epoch", certain_word2="Wx_w_Mgt_visual", certain_ext=".jpg", print_msg=False)
        if(len(self.wx_names) == 0): self.wx_names = get_dir_certain_file_names(self.wx_read_dir , certain_word="epoch", certain_word2="Wx_w_M_visual",   certain_ext=".jpg", print_msg=False)
        self.wy_names                              = get_dir_certain_file_names(self.wy_read_dir , certain_word="epoch", certain_word2="wy",              certain_ext=".jpg", print_msg=False)
        if(len(self.wy_names) == 0): self.wy_names = get_dir_certain_file_names(self.wy_read_dir , certain_word="epoch", certain_word2="Wy_visual",       certain_ext=".jpg", print_msg=False)
        if(len(self.wy_names) == 0): self.wy_names = get_dir_certain_file_names(self.wy_read_dir , certain_word="epoch", certain_word2="Wy_w_Mgt_visual", certain_ext=".jpg", print_msg=False)
        if(len(self.wy_names) == 0): self.wy_names = get_dir_certain_file_names(self.wy_read_dir , certain_word="epoch", certain_word2="Wy_w_M_visual",   certain_ext=".jpg", print_msg=False)
        self.wz_names                              = get_dir_certain_file_names(self.wz_read_dir , certain_word="epoch", certain_word2="wz",              certain_ext=".jpg", print_msg=False)
        if(len(self.wz_names) == 0): self.wz_names = get_dir_certain_file_names(self.wz_read_dir , certain_word="epoch", certain_word2="Wz_visual",       certain_ext=".jpg", print_msg=False)
        if(len(self.wz_names) == 0): self.wz_names = get_dir_certain_file_names(self.wz_read_dir , certain_word="epoch", certain_word2="Wz_w_Mgt_visual", certain_ext=".jpg", print_msg=False)
        if(len(self.wz_names) == 0): self.wz_names = get_dir_certain_file_names(self.wz_read_dir , certain_word="epoch", certain_word2="Wz_w_M_visual",   certain_ext=".jpg", print_msg=False)

        self.WM_npz_all_names   = get_dir_certain_file_names(self.WM_npz_read_dir , certain_word=".", certain_ext=".npz", print_msg=False)
        self.WM_npz_epoch_names = get_dir_certain_file_names(self.WM_npz_read_dir , certain_word="epoch", certain_ext=".npz", print_msg=False)  ### 目前沒用到，先寫著備用
        self.WM_npz_gt_name     = self.get_name_savely(self.WM_npz_read_dir, certain_word="gt_W", certain_ext=".npz", print_msg=False)

        self.temp_M_name        = self.get_name_savely(self.see_read_dir, certain_word="gt_mask", certain_ext=".jpg", print_msg=False)  ### 等之後 改寫 I_to_Wxyz 多把 Mask concat 到 W 後面， 這個就不需要囉
        self.temp_M_path        = f"{self.see_read_dir}/{self.temp_M_name}"                                            ### 等之後 改寫 I_to_Wxyz 多把 Mask concat 到 W 後面， 這個就不需要囉

        self.WM_3D_epoch_names  = get_dir_certain_file_names(self.WM_3D_matplot_visual_read_dir , certain_word="epoch", certain_ext=".jpg", print_msg=False)


        self.wc_read_paths  = [self.wc_read_dir + "/" + name for name in self.wc_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！
        self.wx_read_paths  = [self.wx_read_dir + "/" + name for name in self.wx_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！
        self.wy_read_paths  = [self.wy_read_dir + "/" + name for name in self.wy_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！
        self.wz_read_paths  = [self.wz_read_dir + "/" + name for name in self.wz_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.WM_npz_all_read_paths   = [self.WM_npz_read_dir + "/" + name for name in self.WM_npz_all_names]    ### 目前還沒用到～　所以也沒有寫 write_path 囉！
        self.WM_npz_epoch_read_paths = [self.WM_npz_read_dir + "/" + name for name in self.WM_npz_epoch_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！
        self.WM_npz_gt_path          = self.get_path_savely(self.WM_npz_read_dir, certain_word="gt_W", certain_ext=".npz")
        self.WM_3D_epoch_read_path   = [self.WM_3D_matplot_visual_read_dir + "/" + name for name in self.WM_3D_epoch_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.WM_npz_all_amount   = len(self.WM_npz_all_read_paths)
        self.WM_npz_epoch_amount = len(self.WM_npz_epoch_read_paths)

        self.wc_amount = len(self.wc_read_paths)
        self.trained_epoch  = self.wc_amount - 1  ### 去掉epoch0
        # print("self.wc_read_paths", self.wc_read_paths)

    def Save_as_WM_matplot_visual(self,   ### 訓練後，可以走訪所有see_file 並重新產生 bm_rec_matplot_visual
                                      add_loss=False,
                                      bgr2rgb =False,
                                      single_see_core_amount=CORE_AMOUNT_WM_VISUAL,
                                      see_print_msg=False,
                                      jump_to=0,
                                      **args):
        """
        Save_as_WM_matplot_visual(_after_train) 最後想試試看 省掉他 會不會影響我的理解
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Save_as_WM_matplot_visual, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立 存結果的資料夾
        Check_dir_exist_and_build(self.see_write_dir)
        Check_dir_exist_and_build_new_dir(self.WM_matplot_visual_write_dir)     ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果，且也可以確保 Find_ltrd_and_crop 第二次以上執行不會壞掉
        Check_dir_exist_and_build_new_dir(self.WM_3D_matplot_visual_write_dir)  ### 建立 存結果的資料夾，如果存在 要 刪掉重建，確保生成的都是新的結果，且也可以確保 Find_ltrd_and_crop 第二次以上執行不會壞掉

        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.get_npz_info()
        self.get_wc_info()

        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
            self._draw_WM_matplot_visual(0, self.WM_npz_epoch_amount, add_loss, bgr2rgb, jump_to)
            ### 後處理讓結果更小 但 又不失視覺品質，單核心版
            Find_ltrd_and_crop (self.WM_matplot_visual_write_dir, self.WM_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=1)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.WM_matplot_visual_write_dir, self.WM_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=1)  ### matplot圖存完是png，改存成jpg省空間
        elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
            multi_processing_interface(core_amount=single_see_core_amount, task_amount=self.WM_npz_epoch_amount, task=self._draw_WM_matplot_visual, task_args=[add_loss, bgr2rgb, jump_to], print_msg=see_print_msg)
            ### 後處理讓結果更小 但 又不失視覺品質，多核心版(core_amount 在 step0 裡調)
            Find_ltrd_and_crop (self.WM_matplot_visual_write_dir, self.WM_matplot_visual_write_dir, padding=15, search_amount=10, core_amount=CORE_AMOUNT_FIND_LTRD_AND_CROP)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
            Save_as_jpg        (self.WM_matplot_visual_write_dir, self.WM_matplot_visual_write_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY], core_amount=CORE_AMOUNT_SAVE_AS_JPG)  ### matplot圖存完是png，改存成jpg省空間
        else:
            print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理 存 video
        video_processes = []
        # video_processes.append(Process( target=Video_combine_from_dir, args=(self.see_read_dir, self.see_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.WM_matplot_visual_write_dir, self.WM_matplot_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉###
        video_processes.append(Process( target=Video_combine_from_dir, args=(self.WM_3D_matplot_visual_write_dir, self.WM_3D_matplot_visual_write_dir) ) )  ### 存成jpg後 順便 把所有圖 串成影片，覺得好像還沒好到需要看影片，所以先註解掉之後有需要再打開囉###
        for video_p in video_processes: video_p.start()
        for video_p in video_processes: video_p.join()

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        ###   記得順序是 先同步父 再 同步子 喔！
        if(self.WM_matplot_visual_write_dir != self.WM_matplot_visual_read_dir):
            Syn_write_to_read_dir(write_dir=self.WM_matplot_visual_write_dir, read_dir=self.WM_matplot_visual_read_dir, build_new_dir=True)  ### 父

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Save_as_WM_matplot_visual, Current See:{self.see_name}, cost time:{time.time() - start_time}")
        print("")

    ### See_method 第三部分：主要做的事情在這裡
    def _draw_WM_matplot_visual(self, start_epoch, epoch_amount, add_loss, bgr2rgb, jump_to):
        """
        有可能畫完主圖 還要再畫 loss，所以多這個method，多做的事情都在這裡處理
        處理完後就 Save_fig 囉！
        """
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            if(go_epoch < jump_to): continue
            # print("current go_epoch:", go_epoch, "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            matplot_imgs = self._Draw_WM_matplot_visual(go_epoch, add_loss=add_loss, bgr2rgb=bgr2rgb, jump_to=jump_to)
            matplot_imgs.Save_fig(dst_dir=self.WM_matplot_visual_write_dir, name="epoch", epoch=go_epoch)  ### 如果沒有要接續畫loss，就可以存了喔！

    ### See_method 第三部分a
    ###     我覺得先把 npy 轉成 npz 再來生圖比較好，不要在這邊 邊生圖 邊轉 npz，覺得的原因如下：
    ###         1.這樣這裡做的事情太多了~~
    ###         2.npy轉npz 我會把 npy刪掉，但這樣第二次執行時 self.npy_names 就會是空的，還要寫if來判斷何時讀 npy, npz ，覺得複雜~
    def _Draw_WM_matplot_visual(self, epoch, add_loss=False, bgr2rgb=False, jump_to=0):
        ### 不知道
        # WM_range = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0.2, "zmax": 1}  ### zmin 用 0.2 是因為 直接看結果覺得下面空空的浪費
        ### 柱狀
        # WM_range = {"xmin": -0.08075158, "xmax": 0.07755918, "ymin": -0.13532962, "ymax": 0.1357405, "zmin": 0.0, "zmax": 0.039187048}  ### zmin 用 0.2 是因為 直接看結果覺得下面空空的浪費
        # W_min = -0.135329262
        # W_max =  0.1357405
        ### doc3d hole norm
        # WM_range = {"xmin": -1.2280148, "xmax": 1.2387834, "ymin": -1.2410645, "ymax": 1.2485291, "zmin": -0.67187124, "zmax": 0.63452387}  ### zmin 用 0.2 是因為 直接看結果覺得下面空空的浪費
        # W_min = -1.2410645
        # W_max =  1.2485291
        ### doc3d ch_norm
        WM_range = {"xmin": -1.2280148, "xmax": 1.2387834, "ymin": -1.2410645, "ymax": 1.2485291, "zmin": -0.67187124, "zmax": 0.63452387}  ### zmin 用 0.2 是因為 直接看結果覺得下面空空的浪費
        W_min = np.array( [ WM_range["zmin"], WM_range["ymin"], WM_range["xmin"] ] ).reshape(1, 1, 3)
        W_max = np.array( [ WM_range["zmax"], WM_range["ymax"], WM_range["xmax"] ] ).reshape(1, 1, 3)

        in_img     = cv2.imread(self.dis_img_path)
        WM_01      = np.load(self.WM_npz_epoch_read_paths[epoch])["arr_0"]  ### npz的讀法要["arr_0"]，因為我存npz的時候沒給key_value，預設就 arr_0 囉！
        WM_01_gt   = np.load(self.WM_npz_gt_path)["arr_0"]
        h, w, c = WM_01.shape           ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        WM_01_gt = WM_01_gt[:h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！

        WM_back    = WM_01 * (W_max - W_min) + W_min
        WM_gt_back = WM_01_gt * (W_max - W_min) + W_min
        if(WM_01.shape[2] == 3):
            M = cv2.imread(self.temp_M_path)[..., 0:1]
            M = M[:h, :w, :]            ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
            M = np.where(M > 0.95, 1, 0)
            WM_back = np.concatenate((WM_back, M), axis=-1)
            WM_gt_back = np.concatenate((WM_gt_back, M), axis=-1)

        WM_3D_path    = f'{self.WM_3D_matplot_visual_write_dir}/{self.WM_npz_epoch_names[epoch].replace(".npz", ".jpg")}'
        WM_3D_gt_path = f'{self.WM_3D_matplot_visual_write_dir}/{self.WM_npz_gt_name           .replace(".npz", ".jpg")}'
        WM_3d_plot(WM_back, savefig=True, save_path=WM_3D_path, **WM_range)
        if(os.path.isfile(WM_3D_gt_path) is False): WM_3d_plot(WM_gt_back, savefig=True, save_path=WM_3D_gt_path, **WM_range)

        W_visual,   Wx_visual,   Wy_visual,   Wz_visual   = WcM_01_visual_op(WM_01,    out_ch3=True)
        Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = WcM_01_visual_op(WM_01_gt, out_ch3=True)

        imgs = [ [in_img,     in_img,      in_img,      in_img],
                 [W_visual,   Wx_visual,   Wy_visual,   Wz_visual  ],
                 [Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual]]
        img_titles = [  ["in_img",     "pred_3D",     "gt_3D",       ""],
                        ["W_visual",   "Wx_visual",   "Wy_visual",   "Wz_visual"  ],
                        ["Wgt_visual", "Wxgt_visual", "Wygt_visual", "Wzgt_visual"] ]

                        
        matplot_imgs = Matplot_multi_row_imgs(
                                rows_cols_imgs=imgs,
                                rows_cols_titles=img_titles,
                                fig_title="epoch=%04i" % epoch,
                                add_loss=add_loss,
                                bgr2rgb=bgr2rgb)
        matplot_imgs.Draw_img()
        WM_3d_plot(WM_back,    fig=matplot_imgs.fig, ax=matplot_imgs.ax[0], ax_r=0, ax_c=1, ax_rows=3, **WM_range)
        WM_3d_plot(WM_gt_back, fig=matplot_imgs.fig, ax=matplot_imgs.ax[0], ax_r=0, ax_c=2, ax_rows=3, **WM_range)
        matplot_imgs.ax[0, 3].remove()

        return matplot_imgs