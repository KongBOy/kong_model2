from step0_access_path import JPG_QUALITY, CORE_AMOUNT

import sys
sys.path.append("kong_util")
from matplot_fig_ax_util import Matplot_multi_row_imgs
from build_dataset_combine import Save_as_jpg,  Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir

import cv2
import time
from tqdm import tqdm

# import matplotlib.pyplot as plt
# import pdb

from multiprocess_util import multi_processing_interface

import datetime

class Result:
    # def __init__(self, result_name=None, r_describe=None):
    def __init__(self):
        ### train的時候用的
        self.result_name = None
        self.result_read_dir  = None
        self.result_write_dir = None
        self.ckpt_read_dir = None
        self.ckpt_write_dir = None
        self.logs_read_dir  = None
        self.logs_write_dir = None
        self.sees_ver = None
        self.sees = None
        self.see_amount = None
        self.test_dir = None

        self.in_use_range = "0~1"
        self.gt_use_range = "0~1"

        ### after train的時候才用的
        self.ana_describe = None  ### 這是給matplot用的title

    # def rename_see1_to_see2(self):
    #     for go_see in range(self.see_amount):
    #         if(os.path.isdir(self.sees1[go_see].see_read_dir)):
    #             print("rename_ord:", self.sees1[go_see].see_read_dir)
    #             print("rename_dst:", self.sees2[go_see].see_read_dir)
    #             os.rename(self.sees1[go_see].see_read_dir, self.sees2[go_see].see_read_dir)

    ### 在train step3的時候 才會做這個動作，在那個階段，看的應該是result_obj，所以Draw_loss才寫在Rsult而不寫在See囉
    def Draw_loss_during_train(self, epoch, epochs):
        for see in self.sees:
            see.draw_loss_at_see_during_train(epoch, epochs)

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def save_multiple_single_see_as_matplot_visual(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=7, see_print_msg=False, see_core_amount=7, result_print_msg=False):
        """
        see_core_amount == 1 , single_see_core_amount == 1：單核心跑， 單個see 依序跑， see內的多個任務 依序跑
        see_core_amount == 1 , single_see_core_amount  > 1：多核心跑， 單個see 同時跑， see內的多個任務 同時跑
        see_core_amount  > 1 , single_see_core_amount == 1：多核心跑， 多個see 同時跑， see內的多個任務 依序跑
        see_core_amount  > 1 , single_see_core_amount  > 1：多核心跑， 多個see 同時跑， see內的多個任務 同時跑
        """
        result_start = time.time()
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: doing save_multiple_single_see_as_matplot_visual, Current Result:{self.result_name}")

        if  (see_core_amount == 1):
            """
            see_core_amount == 1 , single_see_core_amount == 1：單核心跑， 單個see 依序跑， see內的多個任務 依序跑
            see_core_amount == 1 , single_see_core_amount  > 1：多核心跑， 單個see 同時跑， see內的多個任務 同時跑
            see內 當單位 切 multiprocess

            如果 see_file_amount少 這好像比較快
            """
            self._save_multiple_single_see_as_matplot_visual(start_see, see_amount, add_loss=add_loss, bgr2rgb=bgr2rgb, single_see_core_amount=single_see_core_amount, see_print_msg=see_print_msg)
        elif(see_core_amount  > 1):
            multi_processing_interface(core_amount=see_core_amount, task_amount=see_amount , task=self._save_multiple_single_see_as_matplot_visual, task_start_index=start_see, task_args=[add_loss, bgr2rgb, single_see_core_amount, see_print_msg], print_msg=result_print_msg)

            """
            see_core_amount  > 1 , single_see_core_amount == 1：多核心跑， 多個see 同時跑， see內的多個任務 依序跑
            see_core_amount  > 1 , single_see_core_amount  > 1：多核心跑， 多個see 同時跑， see內的多個任務 同時跑
            以 整個see 當單位 切 multiprocess
            """
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), "Result level: finish save_multiple_single_see_as_matplot_visual")
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), "Result level: cost_time=", time.time() - result_start)

    def _save_multiple_single_see_as_matplot_visual(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=7, see_print_msg=False):
        ### 用 for 迴圈 依序 跑 單個 see， see內 當單位 切 multiprocess
        for see_num in range(start_see, start_see + see_amount):  ### 防呆，以防直接使用 save_all_single_see_as_matplot_visual 時 start_index 設的比0大 但 amount 設成 see_amount 或 純粹不小心算錯數字(要算準start_index + amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
            self.sees[see_num].save_as_matplot_visual_after_train(add_loss=add_loss, bgr2rgb=bgr2rgb, single_see_core_amount=single_see_core_amount, see_print_msg=see_print_msg)

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def save_multiple_single_see_as_matplot_bm_rec_visual(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=8, see_print_msg=False, see_core_amount=7, result_print_msg=False):
        """
        同上
        """
        result_start = time.time()
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: doing save_multiple_single_see_as_matplot_bm_rec_visual, Current Result:{self.result_name}")

        if  (see_core_amount == 1):
            """
            同上

            如果 see_file_amount少 這好像比較快
            """
            self._save_multiple_single_see_as_matplot_bm_rec_visual(start_see, see_amount, add_loss=add_loss, bgr2rgb=bgr2rgb, single_see_core_amount=single_see_core_amount, see_print_msg=see_print_msg)
        elif(see_core_amount  > 1):
            multi_processing_interface(core_amount=see_core_amount, task_amount=see_amount , task=self._save_multiple_single_see_as_matplot_bm_rec_visual, task_start_index=start_see, task_args=[add_loss, bgr2rgb, single_see_core_amount, see_print_msg], print_msg=result_print_msg)
            """
            同上

            如果 see_file_amount多 這好像比較快
            建議 see_core_amount=7, single_see_core_amount=1

            已經在127.28證實 以  see  為單位比較快, 大約 1549(不含 npy to npz), 設定為：bm/rec core 13, crop 和 jpg core 皆 1
                            以 see內 為單位比較慢, 大約 1973(不含 npy to npz)

            雖然 以 see 為單位比較快 ， 但要使用的記憶體超大 50GB 左右 ，
                127.28 最好別同時做其他事情(training, 開google 之類的)
                127.35 跑不大起來, 或者core設超大：bm/rec core 20~30 之類的

            目前覺得不建議使用，因為以sees內的see當單位來切，覺得有點沒效率
            不過真的試過以後，效率其實還不錯！
            但記憶體會爆的問題還是在，可能只適合在記憶體大的電腦跑這樣子
            """
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), "Result level: finish save_multiple_single_see_as_matplot_bm_rec_visual")
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), "Result level: cost_time=", time.time() - result_start)

    ### 寫成 start_see + see_amount 真的麻，但是 這是為了 multiprocess 的設計才這樣寫的，只能忍一下囉～
    def _save_multiple_single_see_as_matplot_bm_rec_visual(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=8, see_print_msg=False):
        """
        用 for 迴圈 依序 跑 單個 see， see內 當單位 切 multiprocess
        """
        for see_num in range(start_see, start_see + see_amount):
            if(see_num < self.see_amount):  ### 防呆，以防直接使用 _save_multiple_single_see_as_matplot_bm_rec_visual 時 start_see 設的比0大 但 see_amount 設成 self.see_amount 或 純粹不小心算錯數字(要算準start_see + see_amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
                self.sees[see_num].all_npy_to_npz(multiprocess=True)
                self.sees[see_num].save_as_matplot_bm_rec_visual_after_train(add_loss, bgr2rgb, single_see_core_amount=single_see_core_amount, see_print_msg=see_print_msg)
        print("")

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def calculate_multiple_single_see_SSIM_LD(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=8, see_print_msg=False, see_core_amount=7, result_print_msg=False):
        """
        同上
        """
        result_start = time.time()
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: doing calculate_multiple_single_see_SSIM_LD, Current Result:{self.result_name}")
        if  (see_core_amount == 1):
            """
            同上

            推薦使用：see_core_amount=1, single_see_core_amount=7
            """
            self._calculate_multiple_single_see_SSIM_LD(start_see, see_amount, add_loss=add_loss, bgr2rgb=bgr2rgb, single_see_core_amount=single_see_core_amount, see_print_msg=see_print_msg)
        elif(see_core_amount  > 1):
            """
            同上

            如果要用建議 see_core_amount=7, single_see_core_amount=1
            如果在epoch數小的時候可使用，
            如果在epoch數大時不建議用，因為 matlab.engine不知為何 很容易當掉，這邊當一個core  感覺很容易就全當，也不知當哪個core，很麻煩
            """
            multi_processing_interface(core_amount=see_core_amount, task_amount=see_amount , task=self._calculate_multiple_single_see_SSIM_LD, task_start_index=start_see, task_args=[add_loss, bgr2rgb, single_see_core_amount, see_print_msg], print_msg=result_print_msg)
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), "Result level: finish calculate_multiple_single_see_SSIM_LD")
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), "Result level: cost_time=", time.time() - result_start)

    ### 寫成 start_see + see_amount 真的麻，但是 這是為了 multiprocess 的設計才這樣寫的，只能忍一下囉～
    def _calculate_multiple_single_see_SSIM_LD(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=8, see_print_msg=False):
        """
        用 for 迴圈 依序 跑 單個 see， see內 當單位 切 multiprocess
        """
        # print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: doing calculate_multiple_single_see_SSIM_LD_combine, Current Result:{self.result_name}")
        for see_num in range(start_see, start_see + see_amount):
            if(see_num < self.see_amount):  ### 防呆，以防直接使用 calculate_multiple_single_see_SSIM_LD 時 start_see 設的比0大 但 see_amount 設成 self.see_amount 或 純粹不小心算錯數字(要算準start_see + see_amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
                self.sees[see_num].Calculate_SSIM_LD(add_loss=add_loss, bgr2rgb=bgr2rgb, single_see_core_amount=single_see_core_amount, see_print_msg=see_print_msg)

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################
    def save_single_see_as_matplot_bm_rec_visual_at_certain_epoch(self, see_num, epoch, add_loss=False, bgr2rgb=False):
        """
        單個 see 裡面的 certain_epoch 重做 matplot_bm_rec_visual
        """
        if(see_num < self.see_amount):  ### 防呆，以防直接使用 save_all_single_see_as_matplot_visual 時 start_index 設的比0大 但 amount 設成 see_amount 或 純粹不小心算錯數字(要算準start_index + amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
            print(f"current result:{self.result_name}")
            self.sees[see_num].all_npy_to_npz(multiprocess=True)
            self.sees[see_num].save_as_matplot_bm_rec_visual_after_train_at_certain_epoch(epoch, add_loss, bgr2rgb)

    def save_all_single_see_as_matplot_bm_rec_visual_at_certain_epoch(self, epoch, add_loss=False, bgr2rgb=False):
        """
        所有 see 裡面的 certain_epoch 重做 matplot_bm_rec_visual
        """
        for see_num in self.see_amount:
            self.save_single_see_as_matplot_bm_rec_visual_at_certain_epoch(see_num, epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)


    def save_all_single_see_as_matplot_bm_rec_visual_at_final_epoch(self, add_loss=False, bgr2rgb=False):
        """
        所有 see 裡面的 final_epoch 重做 matplot_bm_rec_visual
        """
        for see_num in range(self.see_amount):
            self.sees[see_num].get_see_dir_info()  ### 定位出current_final_epochs前，先更新一下 see_dir_info
            current_final_epochs = self.sees[see_num].see_file_amount - 3   ### 定位出 current_final_epochs，current的意思是可能目前還沒train完也可以用，epochs是 epoch總數，要減掉：in_img, gt_img 和 epoch0
            self.save_single_see_as_matplot_bm_rec_visual_at_certain_epoch(see_num, current_final_epochs, add_loss=add_loss, bgr2rgb=bgr2rgb)

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def _Draw_multi_see(self, start_epoch, epoch_amount, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False):  ### 因為需要看多see，所以提升到Result才的到多see喔！
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            r_c_imgs = []
            for go_see_num, see_num in enumerate(see_nums):
                c_imgs = [in_imgs[go_see_num]]
                c_imgs.append(cv2.imread(self.sees[see_num].see_read_dir + "/" + self.sees[see_num].see_epoch_jpg_names[go_epoch]))
                c_imgs += [gt_imgs[go_see_num]]
                r_c_imgs.append(c_imgs)

            multi_row_imgs = Matplot_multi_row_imgs(
                                rows_cols_imgs=r_c_imgs,
                                rows_cols_titles=r_c_titles,
                                fig_title ="epoch=%04i" % go_epoch,   ### 圖上的大標題,
                                bgr2rgb   =True,
                                add_loss  =add_loss)

            multi_row_imgs.Draw_img()
            if(add_loss): multi_row_imgs.Draw_ax_loss_after_train(multi_row_imgs.ax[-1, 1], self.logs_read_dir, go_epoch, self.sees[see_num].see_file_amount - 2)
            multi_row_imgs.Save_fig(dst_dir=matplot_multi_see_dir, epoch=go_epoch)


    def _draw_multi_see_multiprocess(self, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False, core_amount=CORE_AMOUNT, task_amount=600, print_msg=False):
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._Draw_multi_see, task_args=[see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss], print_msg=print_msg)

    ### 好像比較少用到
    def save_multi_see_as_matplot_visual(self, see_nums, save_name, add_loss=False, multiprocess=True, see_print_msg=False):
        print(f"doing save_multi_see_as_matplot_visual, save_name is {save_name}")
        ### 防呆 ### 這很重要喔！因為 row 只有一個時，matplot的ax的維度只有一維，但我的操作都兩維 會出錯！所以要切去一維的method喔！
        if(len(see_nums) == 1):
            print("因為 see_nums 的數量只有一個，自動切換成 single 的 method 囉～")
            self.save_multiple_single_see_as_matplot_visual(see_nums[0], add_loss=add_loss)
            return
        ###############################################################################################
        start_time = time.time()
        matplot_multi_see_dir = self.result_write_dir + "/" + save_name  ### 結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(matplot_multi_see_dir)   ### 建立 存結果的資料夾

        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for see_num in see_nums:
            in_imgs.append(cv2.imread(self.sees[see_num].see_read_dir + "/" + self.sees[see_num].see_jpg_names[0]))
            gt_imgs.append(cv2.imread(self.sees[see_num].see_read_dir + "/" + self.sees[see_num].see_jpg_names[1]))

        ### 抓 第一row的 要顯示的 titles
        titles = ["in_img", self.ana_describe, "gt_img"]
        r_c_titles = [titles]  ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs
        if(multiprocess): self._draw_multi_see_multiprocess(see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss, core_amount=CORE_AMOUNT, task_amount=self.sees[see_num].see_file_amount, print_msg=print_msg)
        else: self._Draw_multi_see(0, self.see_file_amount, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(matplot_multi_see_dir, matplot_multi_see_dir, padding=15, search_amount=10)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_multi_see_dir, matplot_multi_see_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])  ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_multi_see_dir, matplot_multi_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)


    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################



if(__name__ == "__main__"):
    from step11_b_result_obj_builder import Result_builder

    ############################################################################################################################################
    ### matplot visual  待更新，有需要的時候再弄就好了 反正用法有寫在 step10b囉！
    ############################################################################################################################################
    ## Result 的 各method測試：
    ### 單loss 的情況
    # os_book = Result_builder().set_by_result_name("5_justG_mae1369/type7b_h500_w332_real_os_book-20200525_225555-justG-1532data_mae9_127.51").set_ana_describe("mae9").build()
    # save_multiple_single_see_as_matplot_visual(self, start_see, see_amount, add_loss=False, bgr2rgb=False, single_see_core_amount=7, print_msg=False, see_core_amount=7):
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=True,  bgr2rgb=True, single_see_core_amount=1, pring_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=False, bgr2rgb=True, single_see_core_amount=1, pring_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=True,  bgr2rgb=True, single_see_core_amount=8, pring_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=False, bgr2rgb=True, single_see_core_amount=8, pring_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=True,  bgr2rgb=True, single_see_core_amount=2, pring_msg=False, see_core_amount=7)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=False, bgr2rgb=True, single_see_core_amount=2, pring_msg=False, see_core_amount=7)
    ### 更新到這就懶的更新了，有需要再繼續更新下去吧 反正用法都寫在 setp10b 了！


    # os_book.save_multi_see_as_matplot_visual([29],"train_rd", add_loss=False, multiprocess=False) ### 看會不會自動跳轉
    # os_book.save_multi_see_as_matplot_visual([29],"train_rd", add_loss=True, multiprocess=False)    ### 看會不會自動跳轉
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=False, multiprocess=False)
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=True, multiprocess=False)
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=True, multiprocess=True)

    ### 看多 loss 的情況
    # os_book_lots_loss = Result_builder().set_by_result_name("5_rect_mae136/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35").set_ana_describe("see_lots_loss").build()
    # os_book_lots_loss.save_multiple_single_see_as_matplot_visual(see_num=0, add_loss=True)


    ############################################################################################################################################
    ### bm rec  待更新，有需要的時候再弄就好了 反正用法有寫在 step10b囉！
    ############################################################################################################################################
    blender_os_book = Result_builder().set_by_result_name("5_14_flow_unet/type8_blender_os_book-5_14_1-20210228_144200-flow_unet-epoch050_try_npz").set_ana_describe("blender").build()
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=0, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=0, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=False)
    blender_os_book.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True)

    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=5, add_loss=False, bgr2rgb=True)   ### 如果失敗就單個跑吧~~
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=6, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=7, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=8, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=9, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=10, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=11, add_loss=False, bgr2rgb=True)
    # print(dir(blender_os_book.sees[0]))
    # print(blender_os_book.sees[0].matplot_visual_dir)
