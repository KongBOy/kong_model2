import numpy as np
import cv2

import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs, Matplot_multi_row_imgs
from kong_util.flow_bm_util import check_flow_quality_then_I_w_F_to_R


from step08_b_use_G_generate_0_util import Use_G_generate_Interface, Value_Range_Postprocess_to_01, WcM_01_visual_op, C_01_concat_with_M_to_F_and_get_F_visual, C_01_and_C_01_w_M_to_F_and_visualize

import matplotlib.pyplot as plt
import os
import pdb

class W_w_M_to_Cx_Cy(Use_G_generate_Interface):
    def __init__(self, separate_out=False, focus=False, tight_crop=None, remove_in_bg=True):
        super(W_w_M_to_Cx_Cy, self).__init__()
        self.separate_out = separate_out
        self.focus = focus
        self.tight_crop = tight_crop
        self.remove_in_bg = remove_in_bg
        if(self.tight_crop is not None): self.tight_crop.jit_scale = 0  ### 防呆 test 的時候我們不用 random jit 囉！

    def doing_things(self):
        current_ep   = self.exp_obj.current_ep
        current_ep_it   = self.exp_obj.current_ep_it
        it_see_fq   = self.exp_obj.it_see_fq
        if(it_see_fq is None): ep_it_string = "epoch%03i"        %  current_ep
        else                  : ep_it_string = "epoch%03i_it%06i" % (current_ep, current_ep_it)
        current_time = self.exp_obj.current_time
        if  (self.phase == "train"): used_sees = self.exp_obj.result_obj.sees
        elif(self.phase == "test"):  used_sees = self.exp_obj.result_obj.tests
        # print("sees_sees~~~~~~~~~~~~~~~~~~", exp_obj.result_obj.sees)
        # print("tests_sees~~~~~~~~~~~~~~~~~", exp_obj.result_obj.tests)
        private_write_dir     = used_sees[self.index].see_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        private_rec_write_dir = used_sees[self.index].rec_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        public_write_dir      = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        # print("private_rec_write_dir:", private_rec_write_dir)
        '''
        in_WM[0][0]： wc
        in_WM[1][0]： dis_img
            第一個[  ]是 取 wc 或 dis_img， 
            第二個[ ] 是 取 batch

        in_WM_pre[..., 3:4] 為 M (1, h, w, 1)
        in_WM_pre[..., 0:3] 為 W (1, h, w, 3) 先z 再y 再x

        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''

        ''' 重新命名 讓我自己比較好閱讀'''
        in_WM_and_dis_img     = self.in_ord
        in_WM_and_dis_img_pre = self.in_pre
        Mgt_C_ord             = self.gt_ord
        Mgt_C_pre             = self.gt_pre
        rec_hope              = self.rec_hope
        dis_img_ord           = in_WM_and_dis_img[1]      ### 3024, 3024
        dis_img_pre           = in_WM_and_dis_img_pre[1]  ###  512,  512
        in_WM_pre             = in_WM_and_dis_img_pre[0]

        ''' tight crop '''
        if(self.tight_crop is not None):
            Mgt_pre_for_crop  = Mgt_C_pre[..., 0:1]

            in_WM_pre, _ = self.tight_crop(in_WM_pre , Mgt_pre_for_crop)
            Mgt_C_ord, _ = self.tight_crop(Mgt_C_ord , Mgt_pre_for_crop)
            Mgt_C_pre, _ = self.tight_crop(Mgt_C_pre , Mgt_pre_for_crop)

            ##### dis_img_ord 在 tight_crop 要用 dis_img_pre 來反推喔！
            ### 取得 crop 之前的大小
            ord_h, ord_w = dis_img_ord.shape[1:3]    ### BHWC， 取 HW, 3024, 3024
            pre_h, pre_w = dis_img_pre.shape[1:3]    ### BHWC， 取 HW,  512,  512 或 448, 448 之類的
            ### 算出 ord 和 pre 之間的比例
            ratio_h_p2o  = ord_h / pre_h  ### p2o 是 pre_to_ord 的縮寫
            ratio_w_p2o  = ord_w / pre_w  ### p2o 是 pre_to_ord 的縮寫
            ### 對 pre 做 crop
            dis_img_pre_croped_resized, pre_boundary = self.tight_crop(dis_img_pre  , Mgt_pre_for_crop)  ### 可以看一下 丟進去model 的img 長什麼樣子
            ### 根據比例 放大回來 crop 出 ord， 這是在rec的時候才會用到， 現在 Wxyz_to_Cxy 要做 rec 就會用到了喔！
            ord_l_pad    = np.round(pre_boundary["l_pad_slice"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_r_pad    = np.round(pre_boundary["r_pad_slice"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_t_pad    = np.round(pre_boundary["t_pad_slice"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_d_pad    = np.round(pre_boundary["d_pad_slice"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_l_out_amo = np.round(pre_boundary["l_out_amo"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_t_out_amo = np.round(pre_boundary["t_out_amo"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_r_out_amo = np.round(pre_boundary["r_out_amo"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_d_out_amo = np.round(pre_boundary["d_out_amo"].numpy() * ratio_h_p2o).astype(np.int32)
            dis_img_ord_croped_not_accurate   = np.pad(dis_img_ord.numpy(), ( (0, 0), (ord_t_out_amo, ord_d_out_amo), (ord_l_out_amo, ord_r_out_amo), (0, 0)  ))  ### BHWC
            dis_img_ord_croped_not_accurate   = dis_img_ord_croped_not_accurate[:, ord_t_pad : ord_d_pad , ord_l_pad : ord_r_pad , :]  ### BHWC

        ### 視覺化一下
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(5 * 3, 5))
        # ax[0].imshow(Mgt_C_pre[0, ..., 0:1])
        # ax[1].imshow(Mgt_C_pre[0, ..., 1:2])
        # ax[2].imshow(Mgt_C_pre[0, ..., 2:3])
        # plt.tight_layout()
        # plt.show()

        ''' use_model '''
        W_pre   = in_WM_pre[..., 0:3]
        Mgt_pre = in_WM_pre[..., 3:4]
        if(self.remove_in_bg): W_pre_W_M_pre = W_pre * Mgt_pre
        else                 : W_pre_W_M_pre = W_pre
        # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 5))
        # ax[0, 0].imshow(W_pre[0])
        # ax[0, 1].imshow(Mgt_pre[0])
        # ax[0, 2].imshow(W_pre_W_M_pre[0])

        if(self.separate_out is False):
            C_raw_pre = self.model_obj.generator(W_pre_W_M_pre, training=self.training)
            # ax[1, 0].imshow(C_raw_pre[0, ..., 0])
            # ax[1, 1].imshow(C_raw_pre[0, ..., 1])
            # fig.tight_layout()
            # plt.show()
        else:
            Cx_raw_pre, Cy_raw_pre = self.model_obj.generator(W_pre_W_M_pre, training=self.training)
            # ax[1, 0].imshow(Cy_raw_pre[0])
            # ax[1, 1].imshow(Cx_raw_pre[0])
            # fig.tight_layout()
            # plt.show()
            C_raw_pre = np.concatenate([Cy_raw_pre, Cx_raw_pre], axis=-1)  ### tensor 會自動轉 numpy

        ''' Sobel 部分 '''
        ### 也想看一下 model_out 丟進去 sobel 後的結果長什麼樣子
        sob_objs = []
        sob_objs_len = len(sob_objs)
        for go_l, loss_info_obj in enumerate(self.exp_obj.loss_info_objs):  ### 走訪   所有 loss_info_objs
            loss_fun_dict = loss_info_obj.loss_funs_dict                    ### 把   目前的 loss_info_obj  的 loss_fun_dict 抓出來
            for loss_name, func_obj in loss_fun_dict.items():               ### 走訪 目前的 loss_info_obj  的 當中的所有 loss
                if("sobel" in loss_name): sob_objs.append(func_obj)         ### 如果 其中有使用到 sobel， 把它append 進去 sob_objs

            ### 如果 gt_loss 沒有使用 sobel， 就自己建一個出來
            if(sob_objs_len == len(sob_objs)):        ### 如果 sob_objs 的長度沒變， 代表 目前的loss_info_obj 當中的 gt_loss 沒有用 sobel
                from step10_a1_loss import Sobel_MAE  ### 自己建一個
                sob_objs.append(Sobel_MAE(sobel_kernel_size=5, sobel_kernel_scale=1, stride=1, erose_M=True))
            ### 更新一下 目前的 sob_objs 的 長度
            sob_objs_len = len(sob_objs)

        ### 把 所有的 model_out 套用上 相對應的 sobel
        if(self.separate_out is False):
            Cyx_raw_Gx, Cyx_raw_Gy = sob_objs[0].Calculate_sobel_edges(C_raw_pre)
            Cy_raw_Gx = Cyx_raw_Gx[..., 1]
            Cx_raw_Gx = Cyx_raw_Gx[..., 0]
            Cy_raw_Gy = Cyx_raw_Gy[..., 1]
            Cx_raw_Gy = Cyx_raw_Gy[..., 0]
            if(self.focus):
                Cxy_w_M_Gx, Cxy_w_M_Gy = sob_objs[0].Calculate_sobel_edges(C_raw_pre, Mask=Mgt_pre)
                Cx_w_M_Gx = Cxy_w_M_Gx[..., 0]
                Cy_w_M_Gx = Cxy_w_M_Gx[..., 1]
                Cx_w_M_Gy = Cxy_w_M_Gy[..., 0]
                Cy_w_M_Gy = Cxy_w_M_Gy[..., 1]
        else:
            for go_sob, sob_obj in enumerate(sob_objs):
                if(go_sob == 0): Cy_raw_Gx, Cy_raw_Gy = sob_obj.Calculate_sobel_edges(Cy_raw_pre)
                if(go_sob == 1): Cx_raw_Gx, Cx_raw_Gy = sob_obj.Calculate_sobel_edges(Cx_raw_pre)
                if(self.focus):
                    ''' raw 乘完M 後 Sobel 部分 ，  這部分是只有 fucus is True 才需要 '''
                    ### 也想看一下 model_out 丟進去 sobel 後的結果長什麼樣子
                    ### 把 所有的 model_out 套用上 相對應的 sobel
                    for go_sob, sob_obj in enumerate(sob_objs):
                        if(go_sob == 0): Cx_w_M_Gx, Cx_w_M_Gy = sob_obj.Calculate_sobel_edges(Cx_raw_pre, Mask=Mgt_pre)
                        if(go_sob == 1): Cy_w_M_Gx, Cy_w_M_Gy = sob_obj.Calculate_sobel_edges(Cy_raw_pre, Mask=Mgt_pre)

        Cy_raw_Gx = Cy_raw_Gx[0].numpy()
        Cx_raw_Gx = Cx_raw_Gx[0].numpy()
        Cy_raw_Gy = Cy_raw_Gy[0].numpy()
        Cx_raw_Gy = Cx_raw_Gy[0].numpy()
        if(self.focus):
            Cx_w_M_Gx = Cx_w_M_Gx[0].numpy()
            Cy_w_M_Gx = Cy_w_M_Gx[0].numpy()
            Cx_w_M_Gy = Cx_w_M_Gy[0].numpy()
            Cy_w_M_Gy = Cy_w_M_Gy[0].numpy()
        ''' Sobel end '''''''''''''''''''''''''''''''''''''''


        ### 後處理 Output (C_raw_pre)
        C_raw = Value_Range_Postprocess_to_01(C_raw_pre, self.exp_obj.use_gt_range)
        C_raw = C_raw[0]
        ### 順便處理一下gt
        Cgt_pre = Mgt_C_pre[0, ..., 1:3].numpy()
        Cgt_01 = Value_Range_Postprocess_to_01(Cgt_pre, self.exp_obj.use_gt_range)
        ''''''''''''
        # ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
        h, w, c = C_raw.shape
        Mgt = Mgt_C_pre[0, ..., 0:1].numpy()
        # Mgt = Mgt [:h, :w, :]  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！

        ### 視覺化 Output pred (F)
        if(self.focus is False):
            F,   F_visual,   Cx_visual,   Cy_visual   = C_01_concat_with_M_to_F_and_get_F_visual(C_raw,  Mgt)
            F_visual   = F_visual  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
        else:
            F_raw , F_raw_visual , Cx_raw_visual , Cy_raw_visual, F_w_Mgt,   F_w_Mgt_visual,   Cx_w_Mgt_visual,   Cy_w_Mgt_visual = C_01_and_C_01_w_M_to_F_and_visualize(C_raw, Mgt)
            F_raw_visual   = F_raw_visual   [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            F_w_Mgt_visual = F_w_Mgt_visual [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！


        ### 視覺化 Output gt (Fgt)
        Fgt, Fgt_visual, Cxgt_visual, Cygt_visual = C_01_concat_with_M_to_F_and_get_F_visual(Cgt_01, Mgt)
        Fgt_visual = Fgt_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
        ''''''''''''''''''''''''''''''''''''''''''''''''
        ### 視覺化 Input (W)
        W_01 = Value_Range_Postprocess_to_01(W_pre)
        W_01 = W_01[0].numpy()
        W_visual, Wx_visual, Wy_visual, Wz_visual  = WcM_01_visual_op(W_01)

        ### 視覺化 Input (W_w_M)
        W_w_M_01 = Value_Range_Postprocess_to_01(W_pre_W_M_pre)
        W_w_M_01 = W_w_M_01[0].numpy()
        W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual  = WcM_01_visual_op(W_w_M_01)

        ### 視覺化 Mgt_pre
        Mgt_visual = (Mgt_pre[0].numpy() * 255).astype(np.uint8)

        ### 這個是給後處理用的 dis_img_ord
        dis_img_ord = dis_img_ord [0].numpy()
        dis_img_ord_croped_not_accurate = dis_img_ord_croped_not_accurate[0]
        rec_hope    = rec_hope    [0].numpy()

        ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
        if(self.bgr2rgb):
            rec_hope    = rec_hope  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
            Fgt_visual  = Fgt_visual[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
            dis_img_ord = dis_img_ord   [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
            dis_img_ord_croped_not_accurate = dis_img_ord_croped_not_accurate[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch

            if(self.focus is False):
                F_visual       = F_visual  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
            else:
                F_raw_visual   = F_raw_visual  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
                F_w_Mgt_visual = F_w_Mgt_visual  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch


        if(current_ep == 0 or self.see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_rec_write_dir)    ### 建立 放輔助檔案 的資料夾
            ###################
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",          dis_img_ord)  ### 存 dis_img_ord 沒錯， 這樣子做 tight_crop才正確 不是存 dis_img_ord_croped_not_accurate 喔！ 因為本身已經做過一次tight_crop了， 這樣子再做tight_crop 就多做一次囉～
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_W_01.jpg",         W_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_Wx_01.jpg",        Wx_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_Wy_01.jpg",        Wy_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_Wz_01.jpg",        Wz_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a2-gt_mask.jpg",          Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a3-W_w_Mgt(in_img).jpg",  W_w_M_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wx_w_Mgt.jpg", Wx_w_M_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wy_w_Mgt.jpg", Wy_w_M_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wz_w_Mgt.jpg", Wz_w_M_visual)

            if(self.npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_b_gt_flow", Fgt)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_b_gt_flow", Fgt)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_b_gt_flow.jpg", Fgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_b_gt_Cx.jpg",   Cxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_b_gt_Cy.jpg",   Cygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)

        if(self.focus is False):
            ###################
            if(self.npz_save is False): np.save            (private_write_dir + "/" + f"{ep_it_string}-u1b1_flow", F)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + f"{ep_it_string}-u1b1_flow", F)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b2_flow.jpg", F_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b3_Cx.jpg"  , Cx_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b4_Cy.jpg"  , Cy_visual)

        else:
            if(self.npz_save is False): np.save            (private_write_dir + "/" + f"{ep_it_string}-u1b1_F_w_Mgt", F_w_Mgt)          ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + f"{ep_it_string}-u1b1_F_w_Mgt", F_w_Mgt)          ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b2_F_raw.jpg"   , F_raw_visual)     ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b3_F_w_Mgt.jpg" , F_w_Mgt_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b4_Cx_raw.jpg"  , Cx_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b5_Cx_w_Mgt.jpg", Cx_w_Mgt_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b6_Cy_raw.jpg"  , Cy_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}-u1b7_Cy_w_Mgt.jpg", Cy_w_Mgt_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～


        if(self.model_obj.discriminator is not None):
            C_pre_w_Mgt = (C_raw_pre * Mgt)  ### 1, 512, 512, 2
            Cgt_pre     =  Cgt_pre[np.newaxis, ...]           ### 1, 512, 512, 2

            fake_score = self.model_obj.discriminator(C_pre_w_Mgt).numpy()[0]
            real_score = self.model_obj.discriminator(Cgt_pre).numpy()[0]

            if(self.model_obj.train_step.BCE_use_mask):
                import tensorflow as tf
                from kong_util.Disc_and_receptive_field_util import get_receptive_filed_feature_length, tf_M_resize_then_erosion_by_kong, get_receptive_field_mask

                ''' GAN 的 Discriminator 有用 Mask 的 Case '''
                ### 把 Mask 縮小到 跟 D_out 一樣的大小
                ###     這邊在取得大小， 這邊其實可以直接抓 fake_score的shape， 不過想看看自己模擬的 receptive field 寫得對不對， 所以就用自己寫的function來抓大小囉～
                kernel_size = self.model_obj.discriminator.kernel_size
                strides     = self.model_obj.discriminator.strides
                layer       = self.model_obj.discriminator.depth_level
                receptive_filed_feature_length = get_receptive_filed_feature_length(kernel_size, strides, layer, ord_len=dis_img_pre_croped_resized.shape[0])

                ### 模擬訓練中怎麼縮小， 這邊就怎麼縮小， 可以參考 step10_loss 裡的 BCE loss 喔～
                BCE_Mask_type = self.model_obj.train_step.BCE_Mask_type.lower()
                if  (BCE_Mask_type == "erosion"):  M_used = tf_M_resize_then_erosion_by_kong(Mgt_pre, resize_h=receptive_filed_feature_length, resize_w=receptive_filed_feature_length)
                elif(BCE_Mask_type == "bicubic"):  M_used = tf.image.resize(Mgt_pre, size=(receptive_filed_feature_length, receptive_filed_feature_length), method=tf.image.ResizeMethod.BICUBIC)
                elif(BCE_Mask_type == "area"   ):  M_used = tf.image.resize(Mgt_pre, size=(receptive_filed_feature_length, receptive_filed_feature_length), method=tf.image.ResizeMethod.AREA)
                elif(BCE_Mask_type == "nearest"):  M_used = tf.image.resize(Mgt_pre, size=(receptive_filed_feature_length, receptive_filed_feature_length), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                ### BHWC -> HWC
                M_used = M_used[0]

                ### 取得 Mask 在原影像 的 receptive_filed_mask
                receptive_filed_mask   = get_receptive_field_mask(kernel_size=kernel_size, strides=strides, layer=layer, img_shape=dis_img_pre_croped_resized.shape, Mask=M_used, vmin=0.5)  ### return HWC， vmin=0.5 是為了等等相乘時留一點透明度

                ### D_out 跟 縮小M 相乘， 原影像 跟 原影像的receptive_filed_mask 相乘
                fake_score_w_M         = fake_score * M_used
                real_score_w_M         = real_score * M_used
                F_w_Mgt_visual_w_RFM   = (F_w_Mgt_visual * receptive_filed_mask).astype(np.uint8)
                Fgt_visual_w_RFM       = (Fgt_visual     * receptive_filed_mask).astype(np.uint8)
                # if(bgr2rgb):
                #     F_w_Mgt_visual_w_RFM = F_w_Mgt_visual_w_RFM[..., ::-1]
                #     Fgt_visual_w_RFM     = Fgt_visual_w_RFM[..., ::-1]


                single_row_imgs = Matplot_multi_row_imgs(
                                rows_cols_imgs   = [ [F_w_Mgt_visual ,  fake_score , fake_score_w_M, F_w_Mgt_visual_w_RFM] , [Fgt_visual,     real_score,   real_score_w_M,  Fgt_visual_w_RFM]],   ### 把要顯示的每張圖包成list
                                rows_cols_titles = [ ["F_w_Mgt_visual", "fake_score", "fake_score_w_M", "F_w_Mgt_visual_w_RFM"], ["F_w_Mgt_visual_w_RFM", "Fgt_visual", "real_score", "real_score_w_M", "Fgt_visual_w_RFM"]],  ### 把每張圖要顯示的字包成list
                                fig_title        = f"{ep_it_string}-Discriminator",  ### 圖上的大標題
                                add_loss         = self.add_loss,
                                bgr2rgb          = self.bgr2rgb,   ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                                where_colorbar   = [[None, True, True, None], [None, True, True, None]],
                                w_same_as_first  = True,
                                one_ch_vmin = 0,
                                one_ch_vmax = 1)

                ### 1 row 的版本， 覺得太長
                # single_row_imgs = Matplot_single_row_imgs(
                #                         imgs      =[ F_w_Mgt_visual ,  fake_score , fake_score_w_M, F_w_Mgt_visual_w_RFM,    Fgt_visual,     real_score,   real_score_w_M,  Fgt_visual_w_RFM],   ### 把要顯示的每張圖包成list
                #                         img_titles=["F_w_Mgt_visual", "fake_score", "fake_score_w_M", "F_w_Mgt_visual_w_RFM", "Fgt_visual", "real_score", "real_score_w_M", "Fgt_visual_w_RFM"],  ### 把每張圖要顯示的字包成list
                #                         fig_title =f"{ep_it_string}-Discriminator",  ### 圖上的大標題
                #                         add_loss  =add_loss,
                #                         bgr2rgb   =bgr2rgb,    ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                #                         where_colorbar = [None, True, True, None, None, True, True, None],
                #                         w_same_as_first=True)

            else:
                ''' GAN 的 Discriminator 沒用 Mask 的 Case '''
                single_row_imgs = Matplot_single_row_imgs(
                            imgs            = [ F_w_Mgt_visual ,  fake_score ,  Fgt_visual,   real_score],   ### 把要顯示的每張圖包成list
                            img_titles      = ["F_w_Mgt_visual", "fake_score", "Fgt_visual", "real_score"],  ### 把每張圖要顯示的字包成list
                            fig_title       = f"{ep_it_string}-Discriminator",  ### 圖上的大標題
                            add_loss        = self.add_loss,
                            bgr2rgb         = self.bgr2rgb,  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                            where_colorbar  = [None, True, None, True],
                            w_same_as_first = True,
                            one_ch_vmin=0,
                            one_ch_vmax=1)

            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=private_write_dir, name=f"{ep_it_string}-u1b8_Disc")  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            # print("save to:", private_write_dir)
            # breakpoint()

        if(self.postprocess):
            current_see_name = self.fname.split(".")[0]   # used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾

            bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=dis_img_ord_croped_not_accurate, flow=F_w_Mgt)
            '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
            cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)

            if(self.focus is False):
                imgs       = [ W_visual ,   Mgt_visual , W_w_M_visual,  F_visual ,    rec,   rec_hope]     ### 把要顯示的每張圖包成list
                img_titles = ["W_01",        "Mgt",        "W_w_M",     "pred_F", "pred_rec", "rec_hope"]  ### 把每張圖要顯示的字包成list

                ''' Sobel 部分 '''
                ### 也想看一下 model_out 丟進去 sobel 後的結果長什麼樣子
                imgs       += [[Cx_raw_Gx, Cx_raw_Gy, Cy_raw_Gx, Cy_raw_Gy]]
                img_titles += [["Cx_Gx"  , "Cx_Gy"  , "Cy_Gx"  , "Cy_Gy"  ]]

            else:
                imgs       = [[ W_visual ,   Mgt_visual , W_w_M_visual,  F_raw_visual, F_w_Mgt_visual,    rec,      rec_hope]]         ### 把要顯示的每張圖包成list
                img_titles = [["W_01",        "Mgt",        "W_w_M",   "F_raw_visual", "F_w_Mgt_visual",     "pred_rec", "rec_hope"]]  ### 把每張圖要顯示的字包成list

                ''' Sobel 部分 '''
                ### 也想看一下 model_out 丟進去 sobel 後的結果長什麼樣子
                imgs       += [[ Cx_w_M_Gx ,  Cx_w_M_Gy ,  Cy_w_M_Gx ,  Cy_w_M_Gy ]]
                imgs       += [[ Cx_raw_Gx ,  Cx_raw_Gy ,  Cy_raw_Gx ,  Cy_raw_Gy ]]
                img_titles += [["Cx_w_M_Gx", "Cx_w_M_Gy", "Cy_w_M_Gx", "Cy_w_M_Gy"]]
                img_titles += [["Cx_raw_Gx", "Cx_raw_Gy", "Cy_raw_Gx", "Cy_raw_Gy"]]

            single_row_imgs = Matplot_multi_row_imgs(
                                    rows_cols_imgs   = imgs,
                                    rows_cols_titles = img_titles,
                                    fig_title        = "%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss         = self.add_loss,
                                    bgr2rgb          = self.bgr2rgb,  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                                    fix_size         = (500, 500))
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            print("save to:", self.exp_obj.result_obj.test_write_dir)
