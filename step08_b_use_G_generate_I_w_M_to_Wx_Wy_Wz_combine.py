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

class I_w_M_to_W(Use_G_generate):
    def __init__(self, separate_out=False, focus=False, tight_crop=None):
        super(I_w_M_to_W, self).__init__()
        self.separate_out = separate_out
        self.focus = focus
        self.tight_crop = tight_crop

    def doing_things(self):
        current_ep   = self.exp_obj.current_ep
        current_it   = self.exp_obj.current_it
        it_see_fq   = self.exp_obj.it_see_fq
        if(it_see_fq is None): ep_it_string = "epoch%03i"        %  current_ep
        else                  : ep_it_string = "epoch%03i_it%06i" % (current_ep, current_it)
        current_time = self.exp_obj.current_time
        if  (self.phase == "train"): used_sees = self.exp_obj.result_obj.sees
        elif(self.phase == "test"):  used_sees = self.exp_obj.result_obj.tests

        private_write_dir    = used_sees[self.index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        public_write_dir     = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        '''
        gt_mask_coord[0] 為 mask  (1, h, w, 1)
        gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x

        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''

        ''' 重新命名 讓我自己比較好閱讀'''
        dis_img_ord        = self.in_ord  ### 3024, 3024
        dis_img_pre        = self.in_pre  ###  512, 512 或 448, 448
        Wgt_w_Mgt          = self.gt_ord
        Wgt_w_Mgt_pre      = self.gt_pre
        rec_hope           = self.rec_hope

        ''' tight crop '''
        if(self.tight_crop is not None):
            Mgt_pre_for_crop   = Wgt_w_Mgt_pre[..., 0:1]

            # Wgt_w_Mgt    , _ = self.tight_crop(Wgt_w_Mgt     , Mgt_pre_for_crop)  ### 沒用到
            Wgt_w_Mgt_pre, _ = self.tight_crop(Wgt_w_Mgt_pre , Mgt_pre_for_crop)

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
        Mgt_pre          = Wgt_w_Mgt_pre[..., 3:4]
        Wgt_pre          = Wgt_w_Mgt_pre[..., 0:3]
        I_pre_with_M_pre = dis_img_pre * Mgt_pre

        if(self.separate_out is False):
            W_raw_pre = self.model_obj.generator(I_pre_with_M_pre, training=self.training)
        else:
            Wz_raw_pre, Wy_raw_pre, Wx_raw_pre = self.model_obj.generator(I_pre_with_M_pre, training=self.training)

            W_raw_pre  = np.concatenate([Wz_raw_pre, Wy_raw_pre, Wx_raw_pre], axis=-1)  ### tensor 會自動轉 numpy

        ### 後處理 Output (W_raw_pre)
        W_raw_01 = Value_Range_Postprocess_to_01(W_raw_pre, self.exp_obj.use_gt_range)
        W_raw_01 = W_raw_01[0].numpy()
        ### 順便處理一下gt
        Wgt_pre = Wgt_pre[0].numpy()
        Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, self.exp_obj.use_gt_range)
        ''''''''''''
        ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        h, w, c = W_raw_01.shape
        Mgt_pre = Mgt_pre [0].numpy()
        Mgt_pre = Mgt_pre [:h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！

        ### 視覺化 Output pred (W)
        if(self.focus is False):
            W_visual,   Wx_visual,   Wy_visual,   Wz_visual   = W_01_visual_op(W_raw_01)
        else:
            W_w_Mgt_01 = W_raw_01 * Mgt_pre
            W_raw_visual,   Wx_raw_visual,   Wy_raw_visual,   Wz_raw_visual   = W_01_visual_op(W_raw_01)
            W_w_Mgt_visual, Wx_w_Mgt_visual, Wy_w_Mgt_visual, Wz_w_Mgt_visual = W_01_visual_op(W_w_Mgt_01)

        ### 視覺化 Output gt (Wgt)
        Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = W_01_visual_op(Wgt_01)
        ''''''''''''''''''''''''''''''''''''''''''''''''
        ### 視覺化 Input (I)
        dis_img_ord = dis_img_ord[0].numpy()
        rec_hope  = rec_hope[0].numpy()

        ### 視覺化 Input (I_w_M)
        I_w_M_01  = Value_Range_Postprocess_to_01(I_pre_with_M_pre, self.exp_obj.use_gt_range)
        I_w_M_01 = I_w_M_01[0].numpy()
        I_w_M_visual = (I_w_M_01 * 255).astype(np.uint8)

        ### 視覺化 Mgt_pre
        Mgt_visual = (Mgt_pre * 255).astype(np.uint8)


        ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
        if(self.bgr2rgb):
            dis_img_ord  = dis_img_ord[:, :, ::-1]
            rec_hope     = rec_hope[:, :, ::-1]
            I_w_M_visual = I_w_M_visual[:, :, ::-1]

        if(current_ep == 0 or self.see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      dis_img_ord)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt(in_img).jpg", I_w_M_visual)

            if(self.npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_W", Wgt_01)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_W", Wgt_01)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",  Wgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg", Wxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg", Wygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg", Wzgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)

        if(self.focus is False):
            if(self.npz_save is False): np.save            (private_write_dir + "/" + f"{ep_it_string}-u1b1-W", W_raw_01)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + f"{ep_it_string}-u1b1-W", W_raw_01)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b2-W_visual.jpg" , W_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b3-Wx_visual.jpg", Wx_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b4-Wy_visual.jpg", Wy_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b5-Wz_visual.jpg", Wz_visual)

        else:
            if(self.npz_save is False): np.save            (private_write_dir + "/" + f"{ep_it_string}-u1b1-W_w_Mgt", W_w_Mgt_01)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + f"{ep_it_string}-u1b1-W_w_Mgt", W_w_Mgt_01)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b2-W_raw_visual.jpg"   , W_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b3-W_w_Mgt_visual.jpg" , W_w_Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b4-Wx_raw_visual.jpg"  , Wx_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b5-Wx_w_Mgt_visual.jpg", Wx_w_Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b6-Wy_raw_visual.jpg"  , Wy_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b7-Wy_w_Mgt_visual.jpg", Wy_w_Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b8-Wz_raw_visual.jpg"  , Wz_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b9-Wz_w_Mgt_visual.jpg", Wz_w_Mgt_visual)

        if(self.postprocess):
            current_see_name = used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            if(self.focus is False):
                imgs       = [ dis_img_ord ,   W_visual , Wgt_visual]
                img_titles = ["dis_img_ord", "Wpred",   "Wgt"]
            else:
                imgs       = [ dis_img_ord ,  Mgt_visual, I_w_M_visual,  W_raw_visual, W_w_Mgt_visual,  Wgt_visual]
                img_titles = ["dis_img_ord", "Mgt",       "I_with_M",    "W_raw",      "W_w_Mgt",       "Wgt"]

            single_row_imgs = Matplot_single_row_imgs(
                                    imgs      = imgs,         ### 把要顯示的每張圖包成list
                                    img_titles= img_titles,               ### 把每張圖要顯示的字包成list
                                    fig_title = "%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss  = self.add_loss,
                                    bgr2rgb   = self.bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            print("save to:", self.exp_obj.result_obj.test_write_dir)

            if(self.phase == "test"):
                ### W_01 back to W then + M
                gt_min = self.exp_obj.db_obj.db_gt_range.min
                gt_max = self.exp_obj.db_obj.db_gt_range.max

                if(self.focus is False): W = W_raw_01 * (gt_max - gt_min) + gt_min
                else:                    W = W_w_Mgt_01 * (gt_max - gt_min) + gt_min
                if(self.exp_obj.db_obj.get_method.value == "build_by_in_I_gt_W_hole_norm_then_mul_M_right"): W = W * Mgt_pre
                WM = np.concatenate([W, Mgt_pre], axis=-1)
                ### 確認寫得對不對
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(W_01)
                # ax[1].imshow(W - gt_min)
                # print(W.max())
                # print(W.min())
                # plt.show()

                ### 定位出 存檔案的位置
                gather_WM_npy_dir  = f"{public_write_dir}/pred_WM_{self.phase}-{current_time}/WM_npy_then_npz"
                gather_WM_knpy_dir = f"{public_write_dir}/pred_WM_{self.phase}-{current_time}/WM_knpy"
                Check_dir_exist_and_build(gather_WM_npy_dir)
                Check_dir_exist_and_build(gather_WM_knpy_dir)

                ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
                WM_npy_path  = f"{gather_WM_npy_dir}/{current_see_name}_pred.npy"
                WM_knpy_path = f"{gather_WM_knpy_dir}/{current_see_name}_pred.knpy"
                np.save(WM_npy_path, WM)
                Save_npy_path_as_knpy(WM_npy_path, WM_knpy_path)

                ### .npy刪除(因為超占空間) 改存 .npz
                np.savez_compressed(WM_npy_path.replace(".npy", ".npz"), WM)
                os.remove(WM_npy_path)

                # breakpoint()
