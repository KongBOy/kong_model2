from step0_access_path import JPG_QUALITY, CORE_AMOUNT

import sys
sys.path.append("kong_util")
from util import Matplot_multi_row_imgs
from build_dataset_combine import Save_as_jpg,  Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from video_from_img import Video_combine_from_dir

import cv2
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import pdb



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
        self.ana_plot_title = None  ### 這是給matplot用的title

    # def rename_see1_to_see2(self):
    #     for go_see in range(self.see_amount):
    #         if(os.path.isdir(self.sees1[go_see].see_dir)):
    #             print("rename_ord:", self.sees1[go_see].see_dir)
    #             print("rename_dst:", self.sees2[go_see].see_dir)
    #             os.rename(self.sees1[go_see].see_dir, self.sees2[go_see].see_dir)

    ### 在train step3的時候 才會做這個動作，在那個階段，看的應該是result_obj，所以Draw_loss才寫在Rsult而不寫在See囉
    def Draw_loss_during_train(self, epoch, epochs):
        for see in self.sees:
            see.draw_loss_at_see_during_train(epoch, epochs)

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def save_single_see_as_matplot_visual(self, see_num, add_loss=False, single_see_multiprocess=True):
        if(see_num < self.see_amount):  ### 防呆，以防直接使用 save_all_single_see_as_matplot_visual 時 start_index 設的比0大 但 amount 設成 see_amount 或 純粹不小心算錯數字(要算準start_index + amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
            print(f"current result:{self.result_name}")
            self.sees[see_num].save_as_matplot_visual_after_train(add_loss, single_see_multiprocess)

    def save_all_single_see_as_matplot_visual(self, start_index, amount, add_loss=False, single_see_multiprocess=True):  ### 以 see內的任務 當單位來切(如果single_see_multiprocess==True的話)
        for see_num in tqdm(range(start_index, start_index + amount)):
            self.save_single_see_as_matplot_visual(see_num, add_loss, single_see_multiprocess)

    def save_all_single_see_as_matplot_visual_multiprocess(self, add_loss=False):  ### 以 sees 的 see當單位來切
        """
        目前覺得不建議使用，因為以sees內的see當單位來切，覺得有點沒效率
        """
        print(f"doing {self.result_name}")
        from util import multi_processing_interface
        single_see_multiprocess = False  ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！
        multi_processing_interface(core_amount=CORE_AMOUNT, task_amount=self.see_amount, task=self.save_all_single_see_as_matplot_visual, task_args=[add_loss, single_see_multiprocess])

    ##############################################################################################################################
    ##############################################################################################################################
    def save_single_see_as_matplot_bm_rec_visual(self, see_num, add_loss=False, bgr2rgb=False, single_see_multiprocess=True, print_msg=False):
        if(see_num < self.see_amount):  ### 防呆，以防直接使用 save_all_single_see_as_matplot_visual 時 start_index 設的比0大 但 amount 設成 see_amount 或 純粹不小心算錯數字(要算準start_index + amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
            print(f"current result:{self.result_name}")
            self.sees[see_num].all_npy_to_npz(multiprocess=True)
            self.sees[see_num].save_as_matplot_bm_rec_visual_after_train(add_loss, bgr2rgb, single_see_multiprocess, print_msg=print_msg)

    def save_all_single_see_as_matplot_bm_rec_visual(self, start_index, amount, add_loss=False, bgr2rgb=False, single_see_multiprocess=True, print_msg=False):  ### 以 see內的任務 當單位來切(如果single_see_multiprocess==True的話)
        for see_num in tqdm(range(start_index, start_index + amount)):  ### 這裡寫 start_index + amount 是為了 multiprocess 的格式！
            self.save_single_see_as_matplot_bm_rec_visual(see_num, add_loss, bgr2rgb, single_see_multiprocess, print_msg=print_msg)

    def save_all_single_see_as_matplot_bm_rec_visual_multiprocess(self, add_loss=False, bgr2rgb=False, print_msg=False):  ### 以 sees 的 see當單位來切
        """
        目前覺得不建議使用，因為以sees內的see當單位來切，覺得有點沒效率
        """
        print(f"doing {self.result_name}")
        from util import multi_processing_interface
        single_see_multiprocess = False  ### 注意！大任務已經分給多core了，小任務不能再切分給多core囉！要不然會當掉！
        multi_processing_interface(core_amount=CORE_AMOUNT, task_amount=self.see_amount, task=self.save_all_single_see_as_matplot_bm_rec_visual, task_args=[add_loss, bgr2rgb, single_see_multiprocess], print_msg=print_msg)

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    def _Draw_multi_see(self, start_img, img_amount, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False):  ### 因為需要看多see，所以提升到Result才的到多see喔！
        for go_img in tqdm(range(start_img, start_img + img_amount)):
            if(go_img >= 2):
                epoch = go_img - 2
                r_c_imgs = []
                for go_see_num, see_num in enumerate(see_nums):
                    c_imgs = [in_imgs[go_see_num]]
                    c_imgs.append(cv2.imread(self.sees[see_num].see_dir + "/" + self.sees[see_num].see_jpg_names[go_img]))
                    c_imgs += [gt_imgs[go_see_num]]
                    r_c_imgs.append(c_imgs)

                multi_row_imgs = Matplot_multi_row_imgs(
                                    rows_cols_imgs=r_c_imgs,
                                    rows_cols_titles=r_c_titles,
                                    fig_title ="epoch=%04i" % epoch,   ### 圖上的大標題,
                                    bgr2rgb   =True,
                                    add_loss  =add_loss)

                multi_row_imgs.Draw_img()
                if(add_loss): multi_row_imgs.Draw_ax_loss_after_train(multi_row_imgs.ax[-1, 1], self.logs_dir, epoch, self.see_file_amount - 2)
                multi_row_imgs.Save_fig(dst_dir=matplot_multi_see_dir, epoch=epoch)

                # fig, ax = matplot_visual_multi_row_imgs(rows_cols_titles = r_c_titles,
                #                               rows_cols_imgs   = r_c_imgs,
                #                               fig_title        = "epoch=%04i"%epoch,   ### 圖上的大標題
                #                               bgr2rgb          = True,
                #                               add_loss         = add_loss)
                # if(add_loss): fig, ax = draw_loss_util(fig, ax[-1,1], self.logs_dir, epoch, self.see_file_amount-2)
                # plt.savefig(matplot_multi_see_dir+"/"+"epoch=%04i"%epoch )
                # plt.close()  ### 一定要記得關喔！要不然圖開太多會當掉！

    def _draw_multi_see_multiprocess(self, see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False, core_amount=CORE_AMOUNT, task_amount=600, print_msg=False):
        from util import multi_processing_interface
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._Draw_multi_see, task_args=[see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss], print_msg=print_msg)

    ### 好像比較少用到
    def save_multi_see_as_matplot_visual(self, see_nums, save_name, add_loss=False, multiprocess=True, print_msg=False):
        print(f"doing save_multi_see_as_matplot_visual, save_name is {save_name}")
        ### 防呆 ### 這很重要喔！因為 row 只有一個時，matplot的ax的維度只有一維，但我的操作都兩維 會出錯！所以要切去一維的method喔！
        if(len(see_nums) == 1):
            print("因為 see_nums 的數量只有一個，自動切換成 single 的 method 囉～")
            self.save_single_see_as_matplot_visual(see_nums[0], add_loss=add_loss)
            return
        ###############################################################################################
        start_time = time.time()
        matplot_multi_see_dir = self.result_dir + "/" + save_name  ### 結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(matplot_multi_see_dir)   ### 建立 存結果的資料夾

        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for see_num in see_nums:
            in_imgs.append(cv2.imread(self.sees[see_num].see_dir + "/" + self.sees[see_num].see_jpg_names[0]))
            gt_imgs.append(cv2.imread(self.sees[see_num].see_dir + "/" + self.sees[see_num].see_jpg_names[1]))

        ### 抓 第一row的 要顯示的 titles
        titles = ["in_img", self.ana_plot_title, "gt_img"]
        r_c_titles = [titles]  ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs
        if(multiprocess): self._draw_multi_see_multiprocess(see_nums, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss, core_amount=CORE_AMOUNT, task_amount=self.see_file_amount, print_msg=print_msg)
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
    import matplotlib.pyplot as plt

    ## Result 的 各method測試：
    ### 單loss 的情況
    # os_book = Result_builder().set_by_result_name("5_justG_mae1369/type7b_h500_w332_real_os_book-20200525_225555-justG-1532data_mae9_127.51").set_ana_plot_title("mae9").build()
    # os_book.save_single_see_as_matplot_visual(see_num=0, add_loss=False, single_see_multiprocess=False)
    # os_book.save_single_see_as_matplot_visual(see_num=0, add_loss=True, single_see_multiprocess=False)
    # os_book.save_single_see_as_matplot_visual(see_num=0, add_loss=False, single_see_multiprocess=True)
    # os_book.save_single_see_as_matplot_visual(see_num=0, add_loss=True, single_see_multiprocess=True)
    # os_book.save_all_single_see_as_matplot_visual_multiprocess( add_loss=True)

    # os_book.save_multi_see_as_matplot_visual([29],"train_rd", add_loss=False, multiprocess=False) ### 看會不會自動跳轉
    # os_book.save_multi_see_as_matplot_visual([29],"train_rd", add_loss=True, multiprocess=False)    ### 看會不會自動跳轉
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=False, multiprocess=False)
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=True, multiprocess=False)
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=True, multiprocess=True)

    ### 看多 loss 的情況
    # os_book_lots_loss = Result_builder().set_by_result_name("5_rect_mae136/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35").set_ana_plot_title("see_lots_loss").build()
    # os_book_lots_loss.save_single_see_as_matplot_visual(see_num=0, add_loss=True, single_see_multiprocess=True)


    ############################################################################################################################################
    blender_os_book = Result_builder().set_by_result_name("5_14_flow_unet/type8_blender_os_book-5_14_1-20210228_144200-flow_unet-epoch050_try_npz").set_ana_plot_title("blender").build()
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=0, add_loss=False, bgr2rgb=True, single_see_multiprocess=False)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=0, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # blender_os_book.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=False)
    blender_os_book.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)

    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=5, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)   ### 如果失敗就單個跑吧~~
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=6, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=7, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=8, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=9, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=10, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # blender_os_book.save_single_see_as_matplot_bm_rec_visual(see_num=11, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
    # print(dir(blender_os_book.sees[0]))
    # print(blender_os_book.sees[0].matplot_visual_dir)
