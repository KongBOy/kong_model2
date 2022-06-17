from step0_access_path import JPG_QUALITY, CORE_AMOUNT, CORE_AMOUNT_NPY_TO_NPZ
from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from kong_util.matplot_fig_ax_util import Matplot_multi_row_imgs
from kong_util.build_dataset_combine import Save_as_jpg,  Check_dir_exist_and_build_new_dir, Find_ltrd_and_crop
from kong_util.video_from_img import Video_combine_from_dir
from kong_util.multiprocess_util import multi_processing_interface

import cv2
import time
from tqdm import tqdm

# import matplotlib.pyplot as plt
# import pdb


import datetime

class Result:
    # def __init__(self, result_name=None, r_describe=None):
    def __init__(self):
        ### train的時候用的
        self.exp_obj_use_gt_range = None  ### 好像 跑multiprocess 的時候 會因為 exp_obj 裡面的 model_obj 的 train_step 無法被pickle 無法平行處理， 所以就不要整個 exp_obj 傳進來， 只傳需要的 use_gt_range 進來就好囉！
        self.result_name = None
        self.result_read_dir  = None
        self.result_write_dir = None
        self.ckpt_read_dir = None
        self.ckpt_write_dir = None
        self.ckpt_D_read_dir = None
        self.ckpt_D_write_dir = None
        self.logs_read_dir  = None
        self.logs_write_dir = None
        self.train_code_read_dir  = None
        self.train_code_write_dir = None
        self.sees_ver = None
        self.sees = None
        self.see_amount = None

        self.test_read_dir = None
        self.test_write_dir = None
        self.test_db_name = None
        self.tests = None
        self.test_amount = None
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
    def result_do_all_single_see(self, used_what_see, **args):
        '''
        used_see 目前可輸入: 'see' 或 'test'
        '''
        if  (used_what_see == "see" ): task_amount = self.see_amount
        elif(used_what_see == "test"): task_amount = self.test_amount
        else:
            print(f"result_do_single_see 的 used_what_see 輸入錯誤， 目前輸入{used_what_see}, 目前只支援 'see' 或 'test'")
            exit()

        self.result_do_some_single_see(used_what_see, start_id=0, task_amount=task_amount, **args)

    def result_do_some_single_see(self, used_what_see, start_id, task_amount, **args):
        '''
        used_see 目前可輸入: 'see' 或 'test'
        '''
        if  (used_what_see == "see" ): task_amount = self.see_amount
        elif(used_what_see == "test"): task_amount = self.test_amount
        else:
            print(f"result_do_single_see 的 used_what_see 輸入錯誤， 目前輸入{used_what_see}, 目前只支援 'see' 或 'test'")
            exit()

        """
        step1： Result的功能是把 See 裡面寫的 專門給單個see 的 method ，用for 走訪所有See 然後都執行一次，
        step2： Result也可以 multiprocess，就是把 See當單位 來做multiprocess了，See 裡面本身就有 multiprocess 處理了
        所以可以  多個see 同時跑， see內的多個任務 同時跑喔！

        args舉例大概長相：args == {
            see_method_name        : str
            add_loss               : bool
            bgr2rgb                : bool

            single_see_core_amount : int
            see_print_msg          : bool
            see_core_amount       : int
            result_print_msg       : bool
        }
        method舉例怎麼呼叫：result_do_all_single_see(used_see, stard_id, task_amount, see_method_name=str, add_loss=bool, bgr2rgb=bool, single_see_core_amount=int, see_print_msg=bool, see_core_amount=int, result_print_msg=bool)
            see_method_name 目前以下選擇：
                Save_as_matplot_visual
                Save_as_bm_rec_matplot_visual
                Calculate_SSIM_LD
        """
        if( "see_core_amount"        not in args.keys()): args["see_core_amount"       ] = 1
        if( "single_see_core_amount" not in args.keys()): args["single_see_core_amount"] = 1
        if( "see_print_msg"          not in args.keys()): args["see_print_msg"         ] = False
        if( "result_print_msg"       not in args.keys()): args["result_print_msg"      ] = False
        result_start = time.time()
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: doing {args['see_method_name']}, Current Result:{self.result_read_dir}")
        """
        step1： see_core_amount == 1 , single_see_core_amount == 1：單核心跑， 單個see 依序跑， see內的多個任務 依序跑
        step1： see_core_amount == 1 , single_see_core_amount  > 1：多核心跑， 單個see 依序跑， see內的多個任務 同時跑
        step2： see_core_amount  > 1 , single_see_core_amount == 1：多核心跑， 多個see 同時跑， see內的多個任務 依序跑
        step2： see_core_amount  > 1 , single_see_core_amount  > 1：多核心跑， 多個see 同時跑， see內的多個任務 同時跑
        """

        args["used_what_see"] = used_what_see  ### 配合 multiprocess 傳參數的方式， 把 used_what_see 加進去
        if  (args["see_core_amount"] == 1):
            """ step1
            see_core_amount == 1 , single_see_core_amount == 1：單核心跑， 單個see 依序跑， see內的多個任務 依序跑
            see_core_amount == 1 , single_see_core_amount  > 1：多核心跑， 單個see 依序跑， see內的多個任務 同時跑
            see內 當單位 切 multiprocess
            """
            self._result_do_some_single_see(start_id, task_amount, args)
        elif(args["see_core_amount"]  > 1):
            """ step2
            see_core_amount  > 1 , single_see_core_amount == 1：多核心跑， 多個see 同時跑， see內的多個任務 依序跑
            see_core_amount  > 1 , single_see_core_amount  > 1：多核心跑， 多個see 同時跑， see內的多個任務 同時跑
            以 整個see 當單位 切 multiprocess
            """
            multi_processing_interface(core_amount=args["see_core_amount"], task_amount=task_amount , task=self._result_do_some_single_see, task_start_index=start_id, task_args=[args], print_msg=args["result_print_msg"])
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: {args['see_method_name']} finish")
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Result level: {args['see_method_name']}, cost_time = {time.time() - result_start}")

    ### 寫成 start_see + see_amount 真的麻，但是 這是為了 multiprocess 的設計才這樣寫的，只能忍一下囉～
    def _result_do_some_single_see(self, start_see, task_amount, args):
        ### result_do_single_see 也是開放外面可以呼叫， 這樣 use_what_see 就變成 位置參數了， 所以配合位置參數， 把 used_what_see 抓出來 等等丟位置參數， 記得也要把 關鍵字args裡面的 used_what_see 刪掉喔， 要不燃就會 丟兩次參數了
        used_what_see = args["used_what_see"]   ### 把 used_what_see 抓出來 等等丟位置參數
        del args["used_what_see"]               ### 記得也要把 關鍵字args裡面的 used_what_see 刪掉喔， 要不燃就會 丟兩次參數了
        """
        用 for 迴圈 依序 跑 單個 see， see內任務 當單位 切 multiprocess
        """
        for used_see_id in tqdm(range(start_see, start_see + task_amount)):
            # if(used_see_id < task_amount):  ### 防呆，以防直接使用 calculate_multiple_single_see_SSIM_LD 時 start_see 設的比0大 但 see_amount 設成 self.see_amount 或 純粹不小心算錯數字(要算準start_see + see_amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
            self.result_do_single_see(used_what_see, used_see_id, **args)


    def result_do_single_see(self, used_what_see, used_see_id, **args):
        '''
        used_see 目前可輸入: 'see' 或 'test'
        '''
        if  (used_what_see == "see" ): used_see = self.sees
        elif(used_what_see == "test"): used_see = self.tests
        else:
            print(f"result_do_single_see 的 used_what_see 輸入錯誤， 目前輸入{used_what_see}, 目前只支援 'see' 或 'test'")
            exit()

        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Current Result:{self.result_read_dir}, doing { used_see[used_see_id].see_name}")  ### 這邊也顯示就不用每次都要往上滾很久才知道 現在正處在哪個 result裡囉！
        see_start = time.time()
        if(args["see_method_name"] == "Npy_to_npz"):
            used_see[used_see_id].Npy_to_npz        (**args)

        if(args["see_method_name"] == "Save_as_flow_matplot_visual"):
            used_see[used_see_id].Save_as_matplot_visual        (**args)

        if(args["see_method_name"] == "Save_as_bm_rec_matplot_visual"):
            used_see[used_see_id].Npy_to_npz                    (**args)
            used_see[used_see_id].Save_as_bm_rec_matplot_visual (**args)
            """
            如果 see_file_amount少，建議 多個see 同時跑， see內的多個任務 同時跑：see_core_amount=7, single_see_core_amount=1
            如果 see_file_amount多，建議 單個see 依序跑， see內的多個任務 同時跑：see_core_amount=1, single_see_core_amount=8之類的

            已經在127.28證實處理 500 epoch 左右的case 以  see  為單位比較快, 大約 1549(不含 npy to npz), 設定為：bm/rec core 13, crop 和 jpg core 皆 1
                                                    以 see內 為單位比較慢, 大約 1973(不含 npy to npz)

            雖然 以 see 為單位比較快 ， 但要使用的記憶體超大 50GB 左右 ，
                127.28 最好別同時做其他事情(training, 開google 之類的)
                127.35 跑不大起來, 或者core設超大：bm/rec core 20~30 之類的

            目前覺得不建議使用，因為以sees內的see當單位來切，覺得有點沒效率
            不過真的試過以後，效率其實還不錯！
            但記憶體會爆的問題還是在，可能只適合在記憶體大的電腦跑這樣子
            """
        if(args["see_method_name"] == "Save_as_WM_matplot_visual"):
            used_see[used_see_id].Npy_to_npz               (**args)
            used_see[used_see_id].Save_as_WM_matplot_visual(**args)

        if(args["see_method_name"] == "Calculate_and_Visual_SSIM_LD"):
            used_see[used_see_id].Calculate_SSIM_LD (single_see_core_amount=args["single_see_core_amount"], see_print_msg=args["see_print_msg"])
            used_see[used_see_id].Visual_SSIM_LD    (add_loss=args["add_loss"], bgr2rgb=args["bgr2rgb"], single_see_core_amount=args["single_see_core_amount"], see_print_msg=args["see_print_msg"])
            """
            推薦使用 單個see 依序跑， see內的多個任務 同時跑：see_core_amount=1, single_see_core_amount=4之類的
            不推使用 多個see 同時跑， see內的多個任務 同時跑：因為 matlab.engine不知為何 很容易當掉，這邊當一個core  感覺很容易就全當，也不知當哪個core，很麻煩
                如果硬要 多see同時做跑 建議 see_core_amount=7, single_see_core_amount=1
                如果在epoch數小的時候免強可使用
            """
        if(args["see_method_name"] == "Calculate_SSIM_LD"):
            used_see[used_see_id].Calculate_SSIM_LD (single_see_core_amount=args["single_see_core_amount"], see_print_msg=args["see_print_msg"])

        if(args["see_method_name"] == "Visual_SSIM_LD"):
            used_see[used_see_id].Visual_SSIM_LD    (add_loss=args["add_loss"], bgr2rgb=args["bgr2rgb"], single_see_core_amount=args["single_see_core_amount"], see_print_msg=args["see_print_msg"])

        if(args["see_method_name"] == "Wrong_flow_rename_to_coord"):
            used_see[used_see_id].Wrong_flow_rename_to_coord()


        ### 如果 單核心跑， 單個see 依序跑，顯示這個 see_cost_time 才有用喔！ 如果是 多核心跑， 多個see同時跑， 分配完see 馬上就會到這行， 不會記錄到 see 花的時間喔！
        if(args["see_core_amount"] == 1):
            print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"Current Result:{self.result_read_dir}, doing { used_see[used_see_id].see_name} finish, cost_time:{time.time() - see_start}")  ### 這邊也顯示就不用每次都要往上滾很久才知道 現在正處在哪個 result裡囉！
            print("")

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################
    ### 好像都沒用到，先註解起來吧～再看看要不要刪掉
    # def save_single_see_as_bm_rec_matplot_visual_at_certain_epoch(self, used_see_id, epoch, add_loss=False, bgr2rgb=False):
    #     """
    #     單個 see 裡面的 certain_epoch 重做 bm_rec_matplot_visual
    #     """
    #     if(used_see_id < self.see_amount):  ### 防呆，以防直接使用 save_all_single_see_as_matplot_visual 時 start_index 設的比0大 但 amount 設成 see_amount 或 純粹不小心算錯數字(要算準start_index + amount 真的麻煩，但是 這是為了 multiprocess 的設計才這樣寫的，只能權衡一下囉)
    #         print(f"current result:{self.result_read_dir}")
    #         self.sees[used_see_id].Npy_to_npz(multiprocess=True)
    #         self.sees[used_see_id].Save_as_bm_rec_matplot_visual_at_certain_epoch(epoch, add_loss, bgr2rgb)

    # def save_all_single_see_as_bm_rec_matplot_visual_at_certain_epoch(self, epoch, add_loss=False, bgr2rgb=False):
    #     """
    #     所有 see 裡面的 certain_epoch 重做 bm_rec_matplot_visual
    #     """
    #     for used_see_id in self.see_amount:
    #         self.save_single_see_as_bm_rec_matplot_visual_at_certain_epoch(used_see_id, epoch, add_loss=add_loss, bgr2rgb=bgr2rgb)


    # def save_all_single_see_as_bm_rec_matplot_visual_at_final_epoch(self, add_loss=False, bgr2rgb=False):
    #     """
    #     所有 see 裡面的 final_epoch 重做 bm_rec_matplot_visual
    #     """
    #     for used_see_id in range(self.see_amount):
    #         self.sees[used_see_id].get_see_base_info()  ### 定位出current_final_epochs前，先更新一下 see_dir_info
    #         current_final_epochs = self.sees[used_see_id].see_file_amount - 3   ### 定位出 current_final_epochs，current的意思是可能目前還沒train完也可以用，epochs是 epoch總數，要減掉：in_img, gt_img 和 epoch0
    #         self.save_single_see_as_bm_rec_matplot_visual_at_certain_epoch(used_see_id, current_final_epochs, add_loss=add_loss, bgr2rgb=bgr2rgb)

    ##############################################################################################################################
    ##############################################################################################################################
    ##############################################################################################################################
    ### 好像都沒用到，先註解起來吧～再看看要不要 把功能用其他方式實現出來 再刪掉
    '''
    def _Draw_multi_see(self, start_epoch, epoch_amount, used_see_ids, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False):  ### 因為需要看多see，所以提升到Result才的到多see喔！
        for go_epoch in tqdm(range(start_epoch, start_epoch + epoch_amount)):
            r_c_imgs = []
            for go_used_see_id, used_see_id in enumerate(used_see_ids):
                c_imgs = [in_imgs[go_used_see_id]]
                c_imgs.append(cv2.imread(self.sees[used_see_id].see_read_dir + "/" + self.sees[used_see_id].flow_ep_jpg_names[go_epoch]))
                c_imgs += [gt_imgs[go_used_see_id]]
                r_c_imgs.append(c_imgs)

            multi_row_imgs = Matplot_multi_row_imgs(
                                rows_cols_imgs=r_c_imgs,
                                rows_cols_titles=r_c_titles,
                                fig_title ="epoch=%04i" % go_epoch,   ### 圖上的大標題,
                                bgr2rgb   =True,
                                add_loss  =add_loss)

            multi_row_imgs.Draw_img()
            if(add_loss): multi_row_imgs.Draw_ax_loss_after_train(multi_row_imgs.ax[-1, 1], self.logs_read_dir, go_epoch, self.sees[used_see_id].see_file_amount - 2)
            multi_row_imgs.Save_fig(dst_dir=matplot_multi_see_dir, name="epoch", epoch=go_epoch)


    def _draw_multi_see_multiprocess(self, used_see_ids, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss=False, core_amount=CORE_AMOUNT, task_amount=600, print_msg=False):
        multi_processing_interface(core_amount=core_amount, task_amount=task_amount, task=self._Draw_multi_see, task_args=[used_see_ids, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss], print_msg=print_msg)

    ### 好像比較少用到
    def save_multi_see_as_matplot_visual(self, used_see_ids, save_name, add_loss=False, multiprocess=True, see_print_msg=False):
        print(f"doing save_multi_see_as_matplot_visual, save_name is {save_name}")
        ### 防呆 ### 這很重要喔！因為 row 只有一個時，matplot的ax的維度只有一維，但我的操作都兩維 會出錯！所以要切去一維的method喔！
        if(len(used_see_ids) == 1):
            print("因為 used_see_ids 的數量只有一個，自動切換成 single 的 method 囉～")
            self.save_multiple_single_see_as_matplot_visual(used_see_ids[0], add_loss=add_loss)
            return
        ###############################################################################################
        start_time = time.time()
        matplot_multi_see_dir = self.result_write_dir + "/" + save_name  ### 結果存哪裡定位出來
        Check_dir_exist_and_build_new_dir(matplot_multi_see_dir)   ### 建立 存結果的資料夾

        ### 抓 各row的in/gt imgs
        in_imgs = []
        gt_imgs = []
        for used_see_id in used_see_ids:
            in_imgs.append(cv2.imread(self.sees[used_see_id].in_img_path))
            gt_imgs.append(cv2.imread(self.sees[used_see_id].gt_flow_jpg_path))

        ### 抓 第一row的 要顯示的 titles
        titles = ["in_img", self.ana_describe, "gt_img"]
        r_c_titles = [titles]  ### 還是包成r_c_titles的形式喔！因為 matplot_visual_multi_row_imgs 當初寫的時候是包成 r_c_titles

        ### 抓 row/col 要顯示的imgs
        if(multiprocess): self._draw_multi_see_multiprocess(used_see_ids, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss, core_amount=CORE_AMOUNT, task_amount=self.sees[used_see_id].see_file_amount, print_msg=print_msg)
        else: self._Draw_multi_see(0, self.see_file_amount, used_see_ids, in_imgs, gt_imgs, r_c_titles, matplot_multi_see_dir, add_loss)

        ### 後處理，讓資料變得 好看 且 更小 並 串成影片
        Find_ltrd_and_crop(matplot_multi_see_dir, matplot_multi_see_dir, padding=15, search_amount=10)  ### 有實驗過，要先crop完 再 壓成jpg 檔案大小才會變小喔！
        Save_as_jpg(matplot_multi_see_dir, matplot_multi_see_dir, delete_ord_file=True, quality_list=[cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])  ### matplot圖存完是png，改存成jpg省空間
        Video_combine_from_dir(matplot_multi_see_dir, matplot_multi_see_dir)          ### 存成jpg後 順便 把所有圖 串成影片
        print("cost_time:", time.time() - start_time)
    '''

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
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=True,  bgr2rgb=True, single_see_core_amount=1, print_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=False, bgr2rgb=True, single_see_core_amount=1, print_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=True,  bgr2rgb=True, single_see_core_amount=8, print_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=False, bgr2rgb=True, single_see_core_amount=8, print_msg=False, see_core_amount=1)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=True,  bgr2rgb=True, single_see_core_amount=2, print_msg=False, see_core_amount=7)
    # os_book.save_multiple_single_see_as_matplot_visual(start_see=0, see_amount=1, add_loss=False, bgr2rgb=True, single_see_core_amount=2, print_msg=False, see_core_amount=7)
    ### 更新到這就懶的更新了，有需要再繼續更新下去吧 反正用法都寫在 setp10b 了！


    # os_book.save_multi_see_as_matplot_visual([29],"train_rd", add_loss=False, multiprocess=False) ### 看會不會自動跳轉
    # os_book.save_multi_see_as_matplot_visual([29],"train_rd", add_loss=True, multiprocess=False)    ### 看會不會自動跳轉
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=False, multiprocess=False)
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=True, multiprocess=False)
    # os_book.save_multi_see_as_matplot_visual([29, 30, 31], "train_rd", add_loss=True, multiprocess=True)

    ### 看多 loss 的情況
    # os_book_lots_loss = Result_builder().set_by_result_name("5_rect_mae136/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35").set_ana_describe("see_lots_loss").build()
    # os_book_lots_loss.save_multiple_single_see_as_matplot_visual(used_see_id=0, add_loss=True)


    ############################################################################################################################################
    ### bm rec  待更新，有需要的時候再弄就好了 反正用法有寫在 step10b囉！
    ############################################################################################################################################
    # blender_os_book = Result_builder().set_by_result_name("5_14_flow_unet/type8_blender_os_book-5_14_1-20210228_144200-flow_unet-epoch050_try_npz").set_ana_describe("blender").build()
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=0, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=0, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_all_single_see_as_bm_rec_matplot_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=False)
    # blender_os_book.save_all_single_see_as_bm_rec_matplot_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True)

    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=5, add_loss=False, bgr2rgb=True)   ### 如果失敗就單個跑吧~~
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=6, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=7, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=8, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=9, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=10, add_loss=False, bgr2rgb=True)
    # blender_os_book.save_single_see_as_bm_rec_matplot_visual(used_see_id=11, add_loss=False, bgr2rgb=True)
    # print(dir(blender_os_book.sees[0]))
    # print(blender_os_book.sees[0].matplot_visual_dir)
