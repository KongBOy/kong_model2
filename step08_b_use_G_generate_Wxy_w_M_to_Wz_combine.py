from cv2 import Sobel
import numpy as np
import cv2

import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs, Matplot_multi_row_imgs
from kong_util.flow_bm_util import check_flow_quality_then_I_w_F_to_R


from step08_b_use_G_generate_0_util import Use_G_generate, Value_Range_Postprocess_to_01, W_01_visual_op, C_01_concat_with_M_to_F_and_get_F_visual, C_01_and_C_01_w_M_to_F_and_visualize

import matplotlib.pyplot as plt
import os
import pdb

class Wyx_w_M_to_Wz(Use_G_generate):
    def __init__(self, focus=False, tight_crop=None, sobel=None, sobel_only=False):
        super(Wyx_w_M_to_Wz, self).__init__()
        self.focus = focus
        self.tight_crop = tight_crop
        self.sobel      = sobel
        self.sobel_only = sobel_only

    def doing_things(self):
        current_ep    = self.exp_obj.current_ep
        current_ep_it = self.exp_obj.current_ep_it
        it_see_fq     = self.exp_obj.it_see_fq
        if(it_see_fq is None): ep_it_string = "epoch%03i"        %  current_ep
        else                 : ep_it_string = "epoch%03i_it%06i" % (current_ep, current_ep_it)
        current_time = self.exp_obj.current_time
        if  (self.phase == "train"): used_sees = self.exp_obj.result_obj.sees
        elif(self.phase == "test"):  used_sees = self.exp_obj.result_obj.tests

        private_write_dir    = used_sees[self.index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        public_write_dir     = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        '''
        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''

        ''' 重新命名 讓我自己比較好閱讀'''
        WM_and_I_in_ord = self.in_ord  ### 3024, 3024
        WM_and_I_in_pre = self.in_pre  ###  512, 512 或 448, 448
        # WM_in_ord       = WM_and_I_in_ord[0]  ### 沒用到註解起來
        WM_in_pre       = WM_and_I_in_pre[0]
        dis_img_ord     = WM_and_I_in_ord[1]
        dis_img_pre     = WM_and_I_in_pre[1]
        # WM_gt_ord       = self.gt_ord  ### 沒用到註解起來
        WM_gt_pre       = self.gt_pre
        rec_hope        = self.rec_hope

        ''' tight crop '''
        if(self.tight_crop is not None):
            Mgt_pre_for_crop = WM_gt_pre[..., 3:4]

            # Wgt_w_Mgt    , _ = self.tight_crop(Wgt_w_Mgt     , Mgt_pre_for_crop)  ### 沒用到
            WM_in_pre, _ = self.tight_crop(WM_in_pre , Mgt_pre_for_crop)
            WM_gt_pre, _ = self.tight_crop(WM_gt_pre , Mgt_pre_for_crop)

            ##### dis_img_ord 在 tight_crop 要用 dis_img_pre 來反推喔！
            ### 取得 crop 之前的大小
            ord_h, ord_w = dis_img_ord.shape[1:3]    ### BHWC， 取 HW, 3024, 3024
            pre_h, pre_w = dis_img_pre.shape[1:3]    ### BHWC， 取 HW,  512,  512 或 448, 448 之類的
            ### 算出 ord 和 pre 之間的比例
            ratio_h_p2o  = ord_h / pre_h  ### p2o 是 pre_to_ord 的縮寫
            ratio_w_p2o  = ord_w / pre_w  ### p2o 是 pre_to_ord 的縮寫
            ### 對 pre 做 crop
            dis_img_pre, pre_boundary = self.tight_crop(dis_img_pre  , Mgt_pre_for_crop)
            ### 根據比例 放大回來 crop 出 ord
            ord_l_pad    = np.round(pre_boundary["l_pad_slice"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_r_pad    = np.round(pre_boundary["r_pad_slice"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_t_pad    = np.round(pre_boundary["t_pad_slice"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_d_pad    = np.round(pre_boundary["d_pad_slice"].numpy() * ratio_h_p2o).astype(np.int32)
            dis_img_ord  = dis_img_ord[:, ord_t_pad : ord_d_pad , ord_l_pad : ord_r_pad , :]  ### BHWC

            # self.tight_crop.reset_jit()  ### 注意 test 的時候我們不用 random jit 囉！


        ''' use_model '''
        Wyx_pre            = WM_in_pre[..., 1:3]
        Mgt_pre            = WM_gt_pre[..., 3:4]  ### 模擬一下 之後的 Wyx 是從 model_out 來的， 可能會需要 * M
        Wyx_pre_with_M_pre = Wyx_pre * Mgt_pre    ### 模擬一下 之後的 Wyx 是從 model_out 來的， 可能會需要 * M
        ### 沒有用 sobel 就跟以前一樣， 丟 Wyx_pre_with_M_pre
        if  (self.sobel is None):
            in_data = Wyx_pre_with_M_pre
        ### 有用 sobel 的話， 看是要 1.只丟sobel 還是 2. 跟Wxy一起丟
        elif(self.sobel is not None):
            Sob_Wyx_Gx, Sob_Wyx_Gy = self.sobel.Calculate_sobel_edges(Wyx_pre, Mask=Mgt_pre)
            Sob_Wyx_Gxy = np.concatenate([Sob_Wyx_Gx, Sob_Wyx_Gy], axis=-1)  ### 舉例：model_output_raw  ==  C_pre_raw
            ### 1.只丟 sobel 的結果
            if(self.sobel_only is True):
                in_data = Sob_Wyx_Gxy
            ### 2.要把 sobel 和 Wyx 一起丟進去
            else:
                Wyx_and_Sob_Wyx_Gxy = np.concatenate([Wyx_pre_with_M_pre, Sob_Wyx_Gxy], axis=-1)  ### 舉例：model_output_raw  ==  C_pre_raw
                in_data = Wyx_and_Sob_Wyx_Gxy

        Wz_raw_pre = self.model_obj.generator(in_data, training=self.training)

        ''' IN/GT Sobel 視覺化部分 '''
        ##### IN
        in_sob_objs = []
        if(self.sobel is not None): in_sob_objs.append(self.sobel)
        else                      : in_sob_objs.append(Sobel_MAE(sobel_kernel_size=15, sobel_kernel_scale=1, stride=1, erose_M=True))  ### 如果 input 沒有使用 sobel， 就自己建一個出來
        Win_yx_Gx_visual, Win_yx_Gy_visual = in_sob_objs[0].Calculate_sobel_edges(Wyx_pre, Mask=Mgt_pre)
        Win_y_Gx_visual = Win_yx_Gx_visual[0, ..., 0:1].numpy()
        Win_x_Gx_visual = Win_yx_Gx_visual[0, ..., 1:2].numpy()
        Win_y_Gy_visual = Win_yx_Gy_visual[0, ..., 0:1].numpy()
        Win_x_Gy_visual = Win_yx_Gy_visual[0, ..., 1:2].numpy()

        ##### GT
        ### 也想看一下 model_out 丟進去 sobel 後的結果長什麼樣子
        loss_info_obj = self.exp_obj.loss_info_objs[0]
        loss_fun_dict = loss_info_obj.loss_funs_dict
        gt_sob_objs = []
        for loss_name, func_obj in loss_fun_dict.items():
            if("sobel" in loss_name):
                gt_sob_objs.append(func_obj)

        ### 如果 gt_loss 沒有使用 sobel， 就自己建一個出來
        if(len(gt_sob_objs) == 0):
            gt_sob_objs.append(Sobel_MAE(sobel_kernel_size=5, sobel_kernel_scale=1, stride=1, erose_M=True))

        ### 把 所有的 model_out 套用上 相對應的 sobel
        Wz_raw_Gx, Wz_raw_Gy = gt_sob_objs[0].Calculate_sobel_edges(Wz_raw_pre)
        Wz_raw_Gx = Wz_raw_Gx[0].numpy()
        Wz_raw_Gy = Wz_raw_Gy[0].numpy()
        ''' Sobel end '''''''''''''''''''''

        ### 後處理 Output (Wz_raw_pre)
        Wz_raw_01 = Value_Range_Postprocess_to_01(Wz_raw_pre, self.exp_obj.use_gt_range)
        Wz_raw_01 = Wz_raw_01[0].numpy()
        ### 順便處理一下gt
        WM_gt_pre = WM_gt_pre[0].numpy()  ### 這個還沒轉numpy喔， 記得轉
        WM_gt_01  = Value_Range_Postprocess_to_01(WM_gt_pre, self.exp_obj.use_gt_range)
        ''''''''''''''''''''''''''''''''''''''''''''''''
        ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        h, w, c = Wz_raw_01.shape
        Mgt_pre = Mgt_pre [0].numpy()
        Mgt_pre = Mgt_pre [:h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！

        ### 視覺化 Output pred (Wz)
        if(self.focus is False):
            Wz_visual = (Wz_raw_01 * 255).astype(np.uint8)
        else:
            Wz_raw_visual   = (Wz_raw_01 * 255).astype(np.uint8)

            Wz_w_Mgt_01     =  Wz_raw_01 * Mgt_pre
            Wz_w_Mgt_visual = (Wz_w_Mgt_01 * 255).astype(np.uint8)

            ''' Sobel部分：raw 乘完M 後 Sobel 部分 ，  這部分是只有 fucus is True 才需要 '''
            ### 也想看一下 model_out 丟進去 sobel 後的結果長什麼樣子
            ### 把 所有的 model_out 套用上 相對應的 sobel
            for go_sob, sob_obj in enumerate(gt_sob_objs):
                if(go_sob == 0): W_z_w_M_Gx_visual, W_z_w_M_Gy_visual = sob_obj.Calculate_sobel_edges(Wz_raw_pre, Mask=Mgt_pre[np.newaxis, ...])
                W_z_w_M_Gx_visual = W_z_w_M_Gx_visual[0].numpy()
                W_z_w_M_Gy_visual = W_z_w_M_Gy_visual[0].numpy()
            ''' Sobel end '''''''''''''''''''''

        ### 視覺化 Output gt (Wgt)
        Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = W_01_visual_op(WM_gt_01)
        ''''''''''''''''''''''''''''''''''''''''''''''''
        ### 視覺化 Input (Wyx)
        WM_in_pre = WM_in_pre[0].numpy()
        WM_in_01  = Value_Range_Postprocess_to_01(WM_in_pre, self.exp_obj.use_in_range)
        Win_visual, Wxin_visual, Wyin_visual, Wzin_visual = W_01_visual_op(WM_in_01)

        ### 視覺化 Input (Sob_Wyx_Gxy)
        if(self.sobel is not None):
            Sob_Wy_Gx = Sob_Wyx_Gx[:, :h, :w, 0:1]  ### Wy 其 x方向的梯度， [:, :h, :w, ] 是因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！ ### (Sob_Wyx_Gx - Sob_Wyx_Gx.min()) / (Sob_Wyx_Gx.max() - Sob_Wyx_Gx.min()) * Mgt_pre
            Sob_Wx_Gx = Sob_Wyx_Gx[:, :h, :w, 1:2]  ### Wx 其 x方向的梯度， [:, :h, :w, ] 是因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！ ### (Sob_Wyx_Gx - Sob_Wyx_Gx.min()) / (Sob_Wyx_Gx.max() - Sob_Wyx_Gx.min()) * Mgt_pre
            Sob_Wy_Gy = Sob_Wyx_Gy[:, :h, :w, 0:1]  ### Wy 其 y方向的梯度， [:, :h, :w, ] 是因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！ ### (Sob_Wyx_Gx - Sob_Wyx_Gx.min()) / (Sob_Wyx_Gx.max() - Sob_Wyx_Gx.min()) * Mgt_pre
            Sob_Wx_Gy = Sob_Wyx_Gy[:, :h, :w, 1:2]  ### Wx 其 y方向的梯度， [:, :h, :w, ] 是因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！ ### (Sob_Wyx_Gx - Sob_Wyx_Gx.min()) / (Sob_Wyx_Gx.max() - Sob_Wyx_Gx.min()) * Mgt_pre
            Sob_Wy_Gx_fig, _ = self.sobel.Visualize_sobel_result(Sob_Wy_Gx)
            Sob_Wx_Gx_fig, _ = self.sobel.Visualize_sobel_result(Sob_Wx_Gx)
            Sob_Wy_Gy_fig, _ = self.sobel.Visualize_sobel_result(Sob_Wy_Gy)
            Sob_Wx_Gy_fig, _ = self.sobel.Visualize_sobel_result(Sob_Wx_Gy)

        ### 視覺化 Mgt_pre
        Mgt_visual = (Mgt_pre * 255).astype(np.uint8)

        ### 視覺化 輔助人看的 Input (I)
        dis_img_ord = dis_img_ord[0].numpy()
        rec_hope  = rec_hope[0].numpy()

        ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
        if(self.bgr2rgb):
            dis_img_ord  = dis_img_ord[:, :, ::-1]
            rec_hope     = rec_hope[:, :, ::-1]

        ### DB in/gt 對照的東西
        if(current_ep == 0 or self.see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
            ### IN 部分
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      dis_img_ord)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a2-Wyx_w_Mgt(in_img).jpg", Win_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wx_w_Mgt.jpg", Wxin_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a4-Wy_w_Mgt.jpg", Wyin_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a5-Wz_w_Mgt.jpg", Wzin_visual)

            if(self.sobel is not None):
                Sob_Wx_Gx_fig.savefig(private_write_dir + "/" + "0a_u1a3_2-Sob_Wx_Gx_w_Mgt.png"); plt.close()  ### 記得用完要把圖關掉
                Sob_Wx_Gy_fig.savefig(private_write_dir + "/" + "0a_u1a3_3-Sob_Wx_Gy_w_Mgt.png"); plt.close()  ### 記得用完要把圖關掉
                Sob_Wy_Gx_fig.savefig(private_write_dir + "/" + "0a_u1a4_2-Sob_Wy_Gx_w_Mgt.png"); plt.close()  ### 記得用完要把圖關掉
                Sob_Wy_Gy_fig.savefig(private_write_dir + "/" + "0a_u1a4_3-Sob_Wy_Gy_w_Mgt.png"); plt.close()  ### 記得用完要把圖關掉

            ### GT 部分
            if(self.npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_W", WM_gt_01)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_W", WM_gt_01)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",  Wgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg", Wxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg", Wygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg", Wzgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)

        ### Model Out
        if(self.focus is False):
            if(self.npz_save is False): np.save            (private_write_dir + "/" + f"{ep_it_string}-u1b1-W", Wz_raw_01)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + f"{ep_it_string}-u1b1-W", Wz_raw_01)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b1-Wz_visual.jpg", Wz_visual)

        else:
            if(self.npz_save is False): np.save            (private_write_dir + "/" + f"{ep_it_string}-u1b1-W_w_Mgt", Wz_w_Mgt_01)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + f"{ep_it_string}-u1b1-W_w_Mgt", Wz_w_Mgt_01)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b8-Wz_raw_visual.jpg"  , Wz_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b9-Wz_w_Mgt_visual.jpg", Wz_w_Mgt_visual)

        if(self.postprocess):
            current_see_name = self.fname.split(".")[0]   # used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            if(self.focus is False):
                imgs        = [[ dis_img_ord , Wxin_visual,  Wyin_visual,  Wz_visual , Wzgt_visual]]
                img_titles  = [["dis_img_ord",    "Wx_in",     "Wy_in",     "Wz_pred",   "Wzgt"]]

                imgs       += [[Win_x_Gx_visual, Win_x_Gy_visual, Win_y_Gx_visual, Win_y_Gy_visual, Wz_raw_Gx, Wz_raw_Gy]]
                img_titles += [["Win_x_Gx_visual", "Win_x_Gy_visual", "Win_y_Gx_visual", "Win_y_Gy_visual", "Wz_raw_Gx", "Wz_raw_Gy"]]
            else:
                imgs        = [[ dis_img_ord ,  Mgt_visual, Wxin_visual, Wyin_visual,  Wz_raw_visual, Wz_w_Mgt_visual,  Wzgt_visual]]
                img_titles  = [["dis_img_ord",   "Mgt",       "Wx_in",       "Wy_in",     "Wz_raw",      "Wz_w_Mgt",       "Wzgt"]]

                imgs       += [[Win_x_Gx_visual, Win_x_Gy_visual, Win_y_Gx_visual, Win_y_Gy_visual, Wz_raw_Gx, Wz_raw_Gy, W_z_w_M_Gx_visual, W_z_w_M_Gy_visual]]
                img_titles += [["Win_x_Gx_visual", "Win_x_Gy_visual", "Win_y_Gx_visual", "Win_y_Gy_visual", "Wz_raw_Gx", "Wz_raw_Gy", "W_z_w_M_Gx_visual", "W_z_w_M_Gy_visual"]]

                # [W_z_w_M_Gx_visual, W_z_w_M_Gy_visual]
                # imgs       += [[Win_x_Gx_visual, Win_x_Gy_visual, Win_y_Gx_visual, Win_y_Gy_visual]]
                # img_titles += 
                # [Wz_raw_Gx, Wz_raw_Gy]

            single_row_imgs = Matplot_multi_row_imgs(
                                    rows_cols_imgs   = imgs,         ### 把要顯示的每張圖包成list
                                    rows_cols_titles = img_titles,               ### 把每張圖要顯示的字包成list
                                    fig_title = "%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss  = self.add_loss,
                                    bgr2rgb   = self.bgr2rgb,
                                    fix_size  =(256, 256))  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            print("save to:", self.exp_obj.result_obj.test_write_dir)

            if(self.phase == "test"):
                pass
