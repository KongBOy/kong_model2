import numpy as np
import cv2
from step0_access_path import kong_model2_dir
from step08_b_use_G_generate_0_util import Use_G_generate_Interface, Value_Range_Postprocess_to_01, WcM_01_visual_op, W_01_concat_with_M_to_WM_and_get_W_visual, W_01_and_W_01_w_M_to_WM_and_visualize, C_01_and_C_01_w_M_to_F_and_visualize, C_01_concat_with_M_to_F_and_get_F_visual, W_visual_like_DewarpNet
from kong_util.flow_bm_util import check_flow_quality_then_I_w_F_to_R, use_flow_to_rec

from kong_util.util import method2
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs, Matplot_multi_row_imgs

import matplotlib.pyplot as plt
import os
import pdb

class I_w_M_to_W_to_C(Use_G_generate_Interface):
    def __init__(self, separate_out=False, focus=False, tight_crop=None, remove_in_bg=True):
        super(I_w_M_to_W_to_C, self).__init__()
        self.separate_out = separate_out
        self.focus = focus
        self.tight_crop = tight_crop
        self.remove_in_bg = remove_in_bg

        self.DewarpNet_ssims  = []
        self.DewarpNet_lds    = []
        self.my_rec_ssims     = []
        self.my_rec_lds       = []

    def save_SSIM_LD_mean(self, dst_dir):
        amount = len(self.DewarpNet_ssims)

        DewarpNet_ssim_mean = np.mean(np.array(self.DewarpNet_ssims))
        DewarpNet_ld_mean   = np.mean(np.array(self.DewarpNet_lds  ))
        my_rec_ssim_mean    = np.mean(np.array(self.my_rec_ssims   ))
        my_rec_ld_mean      = np.mean(np.array(self.my_rec_lds     ))

        dst_path = dst_dir + "/" + "0_SSIM_LD_mean.txt"
        with open(dst_path, "w") as f:
            f.write(f"mean\n")
            f.write(f"DewarpNet_ssim_mean : {DewarpNet_ssim_mean }\n")
            f.write(f"DewarpNet_ld_mean   : {DewarpNet_ld_mean   }\n")
            f.write(f"my_rec_ssim_mean    : {my_rec_ssim_mean    }\n")
            f.write(f"my_rec_ld_mean      : {my_rec_ld_mean      }\n")

            for go_val in range(amount):
                DewarpNet_ssim =  self.DewarpNet_ssims[go_val]
                DewarpNet_ld   =  self.DewarpNet_lds  [go_val]
                my_rec_ssim    =  self.my_rec_ssims   [go_val]
                my_rec_ld      =  self.my_rec_lds     [go_val]
                f.write(f"  current_id : {go_val}\n")
                f.write(f"    DewarpNet_ssim : {DewarpNet_ssim }\n")
                f.write(f"    DewarpNet_ld   : {DewarpNet_ld   }\n")
                f.write(f"    my_rec_ssim    : {my_rec_ssim    }\n")
                f.write(f"    my_rec_ld      : {my_rec_ld      }\n")

    def doing_things(self):
        current_ep    = self.exp_obj.current_ep
        current_ep_it = self.exp_obj.current_ep_it
        it_see_fq     = self.exp_obj.it_see_fq
        if(it_see_fq is None): ep_it_string = "epoch%03i"        %  current_ep
        else                 : ep_it_string = "epoch%03i_it%06i" % (current_ep, current_ep_it)
        current_time  = self.exp_obj.current_time
        if  (self.phase == "train"): used_sees = self.exp_obj.result_obj.sees
        elif(self.phase == "test"):  used_sees = self.exp_obj.result_obj.tests

        private_write_dir     = used_sees[self.index].see_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        private_rec_write_dir = used_sees[self.index].rec_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        ''' 想做 by_flow 的話打開註解'''
        # private_rec_by_flow_write_dir = used_sees[self.index].rec_by_flow_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        private_npz_write_dir = used_sees[self.index].npz_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        public_write_dir      = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        '''
        gt_ord/pre 的 [0]第一個是 取 wc, [1] 是取 flow, 第二個[0]是取 batch， 這次試試看用 M 不用 M_pre

        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''

        ''' 重新命名 讓我自己比較好閱讀'''
        dis_img_ord = self.in_ord
        dis_img_pre = self.in_pre
        Wgt_ord = self.gt_ord [0]
        Wgt_pre = self.gt_pre [0]
        Fgt_ord = self.gt_ord [1]
        Fgt_pre = self.gt_pre [1]
        DewarpNet_result = self.DewarpNet_result

        rec_hope = self.rec_hope

        ''' tight crop '''''''''''''''''''''''''''''''''''''''
        if(self.tight_crop is not None):
            Mgt_pre_for_crop   = Wgt_pre[..., 3:4]

            Wgt_pre, _ = self.tight_crop(Wgt_pre , Mgt_pre_for_crop)  ### 給 test concat 用
            Fgt_pre, _ = self.tight_crop(Fgt_pre , Mgt_pre_for_crop)

            ##### dis_img_ord 在 tight_crop 要用 dis_img_pre 來反推喔！
            ### 取得 crop 之前的大小
            ord_h, ord_w = dis_img_ord.shape[1:3]    ### BHWC， 取 HW, 3024, 3024
            pre_h, pre_w = dis_img_pre.shape[1:3]    ### BHWC， 取 HW,  512,  512 或 448, 448 之類的
            ### 算出 ord 和 pre 之間的比例
            ratio_h_p2o  = ord_h / pre_h  ### p2o 是 pre_to_ord 的縮寫
            ratio_w_p2o  = ord_w / pre_w  ### p2o 是 pre_to_ord 的縮寫
            ### 對 pre 做 crop
            dis_img_pre_croped_resized, pre_boundary = self.tight_crop(dis_img_pre  , Mgt_pre_for_crop)  ### 可以看一下 丟進去model 的img 長什麼樣子
            ### 根據比例 放大回來 crop 出 ord， 這是在rec的時候才會用到， 現在 I_w_M_to_Wxyz_to_Cxy 要做 rec 就會用到了喔！
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
            # self.tight_crop.reset_jit()  ### 注意 test 的時候我們不用 random jit 囉！
        ''' tight crop end '''''''''''''''''''''''''''''''''


        ''' use model '''''''''''''''''''''''''''''''''''''''
        Mgt     = Wgt_ord[..., 3:4]  ### [0]第一個是 取 wc, [1] 是取 flow, 第二個[0]是取 batch， 這次試試看用 M 不用 M_pre
        Mgt_pre = Wgt_pre[..., 3:4]  ### 但是思考了一下，因為我現在focus train， 頁面外面是 灰色的， 如果 頁面外面 Mask 外面 有一大片 微小的值 會不會 GG呢 ??? 好像改回 Mgt_pre 比較安全???
        used_M   = Mgt_pre

        if(self.remove_in_bg): I_pre_w_M = dis_img_pre_croped_resized * used_M  ### 這次試試看用 M 不用 M_pre
        else                 : I_pre_w_M = dis_img_pre_croped_resized

        if(self.separate_out is False):
            W_pre_raw, C_pre_raw = self.model_obj(I_pre_w_M, Mask=used_M, training=self.training)
        else:
            Wz_pre_raw, Wy_pre_raw, Wx_pre_raw, Cx_pre_raw, Cy_pre_raw = self.model_obj.generator(I_pre_w_M, Mask=used_M, training=self.training)
            W_pre_raw = np.concatenate((Wz_pre_raw, Wy_pre_raw, Wx_pre_raw), axis=-1)
            C_pre_raw = np.concatenate((Cy_pre_raw, Cx_pre_raw), axis=-1)

        ### 後處理 Output (W_raw_pre)
        W_01_raw = Value_Range_Postprocess_to_01(W_pre_raw, self.exp_obj.use_gt_range)
        C_01_raw = Value_Range_Postprocess_to_01(C_pre_raw, self.exp_obj.use_gt_range)
        W_01_raw = W_01_raw[0]
        C_01_raw = C_01_raw[0]
        used_M   = used_M[0].numpy()


        ''' Sobel 部分 '''''''''''''''''''''''''''''''''''''''
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
            for go_sob, sob_obj in enumerate(sob_objs):
                if(go_sob == 0): Wzyx_raw_Gx, Wzyx_raw_Gy = sob_obj.Calculate_sobel_edges(W_pre_raw)
                if(go_sob == 1): Cxy_raw_Gx , Cxy_raw_Gy  = sob_obj.Calculate_sobel_edges(C_pre_raw)
            Wz_raw_Gx = Wzyx_raw_Gx[..., 0]
            Wy_raw_Gx = Wzyx_raw_Gx[..., 1]
            Wx_raw_Gx = Wzyx_raw_Gx[..., 2]
            Wz_raw_Gy = Wzyx_raw_Gy[..., 0]
            Wy_raw_Gy = Wzyx_raw_Gy[..., 1]
            Wx_raw_Gy = Wzyx_raw_Gy[..., 2]
            Cy_raw_Gx = Cxy_raw_Gx [..., 1]
            Cx_raw_Gx = Cxy_raw_Gx [..., 0]
            Cy_raw_Gy = Cxy_raw_Gy [..., 1]
            Cx_raw_Gy = Cxy_raw_Gy [..., 0]
            if(self.focus):
                if(go_sob == 0): Wzyx_w_M_Gx, Wzyx_w_M_Gy = sob_obj.Calculate_sobel_edges(W_pre_raw, Mask=used_M[np.newaxis, ...])
                if(go_sob == 1): Cxy_w_M_Gx , Cxy_w_M_Gy  = sob_obj.Calculate_sobel_edges(C_pre_raw, Mask=used_M[np.newaxis, ...])
                Wz_w_M_Gx = Wzyx_w_M_Gx[..., 0]
                Wy_w_M_Gx = Wzyx_w_M_Gx[..., 1]
                Wx_w_M_Gx = Wzyx_w_M_Gx[..., 2]
                Wz_w_M_Gy = Wzyx_w_M_Gy[..., 0]
                Wy_w_M_Gy = Wzyx_w_M_Gy[..., 1]
                Wx_w_M_Gy = Wzyx_w_M_Gy[..., 2]
                Cy_w_M_Gx = Cxy_w_M_Gx [..., 1]
                Cx_w_M_Gx = Cxy_w_M_Gx [..., 0]
                Cy_w_M_Gy = Cxy_w_M_Gy [..., 1]
                Cx_w_M_Gy = Cxy_w_M_Gy [..., 0]
        else:
            for go_sob, sob_obj in enumerate(sob_objs):
                if(go_sob == 0): Wz_raw_Gx, Wz_raw_Gy = sob_obj.Calculate_sobel_edges(Wz_pre_raw)
                if(go_sob == 1): Wy_raw_Gx, Wy_raw_Gy = sob_obj.Calculate_sobel_edges(Wy_pre_raw)
                if(go_sob == 2): Wx_raw_Gx, Wx_raw_Gy = sob_obj.Calculate_sobel_edges(Wx_pre_raw)
                if(go_sob == 3): Cy_raw_Gx, Cy_raw_Gy = sob_obj.Calculate_sobel_edges(Cy_pre_raw)
                if(go_sob == 4): Cx_raw_Gx, Cx_raw_Gy = sob_obj.Calculate_sobel_edges(Cx_pre_raw)
            if(self.focus):
                for go_sob, sob_obj in enumerate(sob_objs):
                    if(go_sob == 0): Wz_w_M_Gx, Wz_w_M_Gy = sob_obj.Calculate_sobel_edges(Wz_pre_raw, Mask=used_M[np.newaxis, ...])
                    if(go_sob == 1): Wy_w_M_Gx, Wy_w_M_Gy = sob_obj.Calculate_sobel_edges(Wy_pre_raw, Mask=used_M[np.newaxis, ...])
                    if(go_sob == 2): Wx_w_M_Gx, Wx_w_M_Gy = sob_obj.Calculate_sobel_edges(Wx_pre_raw, Mask=used_M[np.newaxis, ...])
                    if(go_sob == 3): Cy_w_M_Gx, Cy_w_M_Gy = sob_obj.Calculate_sobel_edges(Cy_pre_raw, Mask=used_M[np.newaxis, ...])
                    if(go_sob == 4): Cx_w_M_Gx, Cx_w_M_Gy = sob_obj.Calculate_sobel_edges(Cx_pre_raw, Mask=used_M[np.newaxis, ...])
        Wz_raw_Gx = Wz_raw_Gx[0].numpy()
        Wz_raw_Gy = Wz_raw_Gy[0].numpy()
        Wy_raw_Gx = Wy_raw_Gx[0].numpy()
        Wy_raw_Gy = Wy_raw_Gy[0].numpy()
        Wx_raw_Gx = Wx_raw_Gx[0].numpy()
        Wx_raw_Gy = Wx_raw_Gy[0].numpy()
        Cy_raw_Gx = Cy_raw_Gx[0].numpy()
        Cy_raw_Gy = Cy_raw_Gy[0].numpy()
        Cx_raw_Gx = Cx_raw_Gx[0].numpy()
        Cx_raw_Gy = Cx_raw_Gy[0].numpy()

        if(self.focus):
            Wz_w_M_Gx = Wz_w_M_Gx[0].numpy()
            Wz_w_M_Gy = Wz_w_M_Gy[0].numpy()
            Wy_w_M_Gx = Wy_w_M_Gx[0].numpy()
            Wy_w_M_Gy = Wy_w_M_Gy[0].numpy()
            Wx_w_M_Gx = Wx_w_M_Gx[0].numpy()
            Wx_w_M_Gy = Wx_w_M_Gy[0].numpy()
            Cy_w_M_Gx = Cy_w_M_Gx[0].numpy()
            Cy_w_M_Gy = Cy_w_M_Gy[0].numpy()
            Cx_w_M_Gx = Cx_w_M_Gx[0].numpy()
            Cx_w_M_Gy = Cx_w_M_Gy[0].numpy()
        ''' Sobel end '''''''''''''''''''''''''''''''''''''''


        ''' use_model end '''''''''''''''''''''''''''''''''''''''

        '''model in visualize'''
        I_01_w_M  = Value_Range_Postprocess_to_01(I_pre_w_M, self.exp_obj.use_gt_range)
        I_w_M_visual = (I_01_w_M[0].numpy() * 255).astype(np.uint8)
        Mgt_visual   = (used_M * 255).astype(np.uint8)

        '''model out visualize'''  ### 後處理： 拿掉 batch 和 弄成01 和 轉成 numpy
        if(self.focus is False):
            W_raw_c_M, W_raw_visual, Wx_raw_visual,   Wy_raw_visual,   Wz_raw_visual  = W_01_concat_with_M_to_WM_and_get_W_visual(W_01_raw, used_M)
            F_raw    , F_raw_visual, Cx_raw_visual,   Cy_raw_visual                   = C_01_concat_with_M_to_F_and_get_F_visual (C_01_raw, used_M)

        else:
            W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual, W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual = W_01_and_W_01_w_M_to_WM_and_visualize(W_01_raw, used_M)
            F_raw    , F_raw_visual, Cx_raw_visual, Cy_raw_visual,                F_w_Mgt,   F_w_M_visual, Cx_w_M_visual, Cy_w_M_visual                = C_01_and_C_01_w_M_to_F_and_visualize (C_01_raw, used_M)

        '''model gt visualize'''
        Wgt_pre = Wgt_pre[0].numpy()
        Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, self.exp_obj.use_gt_range)
        Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = WcM_01_visual_op(Wgt_01)

        Cgt_pre = Fgt_pre[0, ..., 1:3].numpy()
        Cgt_01  = Value_Range_Postprocess_to_01(Cgt_pre, self.exp_obj.use_gt_range)
        Fgt, Fgt_visual, Cxgt_visual, Cygt_visual = C_01_concat_with_M_to_F_and_get_F_visual(Cgt_01, used_M)

        ''' model postprocess輔助 visualize'''
        dis_img_ord         = dis_img_ord[0].numpy()
        dis_img_ord_croped_not_accurate = dis_img_ord_croped_not_accurate[0]
        dis_img_pre         = (dis_img_pre[0].numpy() * 255).astype(np.uint8)
        dis_img_pre_croped_resized_visual  = (dis_img_pre_croped_resized[0].numpy() * 255).astype(np.uint8)  ### 可以看一下 丟進去model 的img 長什麼樣子
        rec_hope            = rec_hope[0].numpy()
        DewarpNet_result    = DewarpNet_result[0].numpy()

        '''cv2 bgr 與 rgb 的調整'''
        ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
        if(self.bgr2rgb):
            dis_img_ord        = dis_img_ord       [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            dis_img_ord_croped_not_accurate   = dis_img_ord_croped_not_accurate  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            dis_img_pre_croped_resized_visual = dis_img_pre_croped_resized_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            rec_hope           = rec_hope          [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            DewarpNet_result   = DewarpNet_result  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            I_w_M_visual       = I_w_M_visual      [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

        if(current_ep == 0 or self.see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_npz_write_dir)    ### 建立 放輔助檔案 的資料夾
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      dis_img_ord)  ### 存 dis_img_ord 沒錯， 這樣子做 tight_crop才正確 不是存 dis_img_ord_croped_not_accurate 喔！ 因為本身已經做過一次tight_crop了， 這樣子再做tight_crop 就多做一次囉～
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img_ord_croped_not_accurate.jpg", dis_img_ord_croped_not_accurate)    ### 可以看一下 丟進去model 的img 對應原始size 長什麼樣子
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img_pre_croped_resized.jpg"     , dis_img_pre_croped_resized_visual)  ### 可以看一下 丟進去model 的img 長什麼樣子
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt(in_img).jpg", I_w_M_visual)

            if(self.npz_save is False): np.save            (private_write_dir     + "/" + "0b_u1b1-gt_W", Wgt_01)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + "0b_u1b1-gt_W", Wgt_01)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",  Wgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg", Wxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg", Wygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg", Wzgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.png",   rec_hope)

            if(self.npz_save is False): np.save            (private_write_dir     + "/" + "0b_u2b1-gt_b_Fgt", Fgt)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + "0b_u2b1-gt_b_Fgt", Fgt)
            cv2.imwrite(private_write_dir + "/" + "0b_u2b2-gt_b_Fgt.jpg", Fgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u2b3-gt_b_Cxgt.jpg",   Cxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u2b4-gt_b_Cygt.jpg",   Cygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)
        if(self.focus):
            if(self.npz_save is False): np.save            (private_write_dir     + "/" + f"{ep_it_string}_u1b1-W_w_Mgt", W_w_M_c_M)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + f"{ep_it_string}_u1b1-W_w_Mgt", W_w_M_c_M)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b2-W_raw_visual.jpg" , W_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b3-W_w_M_visual.jpg" , W_w_M_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b4-Wx_raw_visual.jpg", Wx_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b5-Wx_w_M_visual.jpg", Wx_w_M_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b6-Wy_raw_visual.jpg", Wy_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b7-Wy_w_M_visual.jpg", Wy_w_M_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b8-Wz_raw_visual.jpg", Wz_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b9-Wz_w_M_visual.jpg", Wz_w_M_visual)

            if(self.npz_save is False): np.save            (private_write_dir     + "/" + f"{ep_it_string}_u2b1-F_w_Mgt", F_w_Mgt)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + f"{ep_it_string}_u2b1-F_w_Mgt", F_w_Mgt)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b2-F_raw.jpg"   , F_raw_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b3-F_w_Mgt.jpg" , F_w_M_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b4-Cx_raw.jpg"  , Cx_raw_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b5-Cx_w_Mgt.jpg", Cx_w_M_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b6-Cy_raw.jpg"  , Cy_raw_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b7-Cy_w_Mgt.jpg", Cy_w_M_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
        else:
            if(self.npz_save is False): np.save            (private_write_dir     + "/" + f"{ep_it_string}_u1b1-W_w_Mgt", W_raw_c_M)  ### 有空在整理這個要怎麼寫
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + f"{ep_it_string}_u1b1-W_w_Mgt", W_raw_c_M)  ### 有空在整理這個要怎麼寫
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b2-W_raw_visual.jpg" , W_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b4-Wx_raw_visual.jpg", Wx_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b6-Wy_raw_visual.jpg", Wy_raw_visual)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u1b8-Wz_raw_visual.jpg", Wz_raw_visual)

            if(self.npz_save is False): np.save            (private_write_dir     + "/" + f"{ep_it_string}_u2b1-F_w_Mgt", F_raw)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + f"{ep_it_string}_u2b1-F_w_Mgt", F_raw)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b2-F_raw.jpg"   , F_raw_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b4-Cx_raw.jpg"  , Cx_raw_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u2b6-Cy_raw.jpg"  , Cy_raw_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

        if(self.postprocess):
            use_what_flow = None
            if(self.focus): use_what_flow = F_w_Mgt
            else          : use_what_flow = F_raw

            Check_dir_exist_and_build(private_rec_write_dir)          ### 建立 放輔助檔案 的資料夾
            ''' 想做 by_flow 的話打開註解'''
            # Check_dir_exist_and_build(private_rec_by_flow_write_dir)  ### 建立 放輔助檔案 的資料夾
            current_see_name = self.fname.split(".")[0]   # used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=dis_img_ord_croped_not_accurate, flow=use_what_flow)
            ''' 想做 by_flow 的話打開註解'''
            # rec_by_flow   = use_flow_to_rec(dis_img=dis_img_ord_croped_not_accurate, flow=use_what_flow)

            ### 給 ppt 用的
            h, w = bm.shape[:2]
            bm_mask = np.ones(shape=(h, w, 1))
            # canvas_size = 5
            # nrows = 1
            # ncols = 4
            # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(canvas_size * ncols, canvas_size * nrows))
            # ax[0].imshow(bm[..., 0])
            # ax[1].imshow(bm[..., 1])
            # plt.show()
            bm_visual = np.concatenate( (bm[..., ::-1], bm_mask), axis=-1)
            bm_visual = (bm_visual * 255).astype(np.uint8)
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u3_ppt-bm.jpg", bm_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u3_ppt-W_raw_visual.jpg" , W_visual_like_DewarpNet(W_raw_visual, used_M))
            cv2.imwrite(private_write_dir + "/" + f"{ep_it_string}_u3_ppt-W_w_M_visual.jpg" , W_visual_like_DewarpNet(W_w_M_visual, used_M))

            '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
            # gt_bm, gt_rec = check_F_quality_then_I_w_F_to_R(dis_img=dis_img_pre_croped_resized_visual, F=Fgt)
            cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.png" % current_ep, rec)
            ''' 想做 by_flow 的話打開註解'''
            # cv2.imwrite(private_rec_by_flow_write_dir + "/" + "rec_epoch=%04i.png" % current_ep, rec_by_flow)

            if(self.focus is False):
                r_c_imgs   = [ [dis_img_pre_croped_resized_visual , Mgt_visual     , I_w_M_visual  , rec             , rec_hope ],
                               [W_raw_visual       , Wx_raw_visual , Wy_raw_visual , Wz_raw_visual , ],
                               [Wx_raw_Gx          , Wx_raw_Gy     , Wy_raw_Gx     , Wy_raw_Gy     , Wz_raw_Gx    , Wz_raw_Gy     ],
                               [F_raw_visual       , Cx_raw_visual , Cy_raw_visual , ],
                               [Cx_raw_Gx          , Cx_raw_Gy     , Cy_raw_Gx     , Cy_raw_Gy] ]
                r_c_titles = [ ["dis_img"          , "Mgt"         , "I_with_M"    , "rec"         , "rec_hope"],
                               ["W_raw"            , "Wx_raw"      , "Wy_raw"      , "Wz_raw"      , ],
                               ["Wx_raw_Gx"        , "Wx_raw_Gy"   , "Wy_raw_Gx"   , "Wy_raw_Gy"   , "Wz_raw_Gx"  , "Wz_raw_Gy"     ],
                               ["F_raw"            , "Cx_raw"      , "Cy_raw"      , ],
                               ["Cx_raw_Gx"        , "Cx_raw_Gy"   , "Cy_raw_Gx"   , "Cy_raw_Gy"] ]

            else:
                r_c_imgs   = [ [dis_img_pre_croped_resized_visual , Mgt_visual     , I_w_M_visual  , rec             , rec_hope ],
                               [Wgt_visual         , W_raw_visual  , W_w_M_visual  , Wx_raw_visual , Wx_w_M_visual , Wy_raw_visual, Wy_w_M_visual , Wz_raw_visual, Wz_w_M_visual ],
                               [Wx_w_M_Gx          , Wx_w_M_Gy     , Wy_w_M_Gx     , Wy_w_M_Gy     , Wz_w_M_Gx    , Wz_w_M_Gy     ],
                               [Fgt_visual         , F_raw_visual  , F_w_M_visual  , Cx_raw_visual , Cx_w_M_visual , Cy_raw_visual, Cy_w_M_visual , ],
                               [Cx_w_M_Gx          , Cx_w_M_Gy     , Cy_w_M_Gx     , Cy_w_M_Gy] ]
                r_c_titles = [ ["dis_img"          , "Mgt"         , "I_with_M"    , "rec"         , "rec_hope"],
                               ["Wgt"              , "W_raw"       , "W_w_M"       , "Wx_raw"      , "Wx_w_M"      , "Wy_raw"        , "Wy_w_M"        , "Wz_raw"     , "Wz_w_M"      ],
                               ["Wx_w_M_Gx"        , "Wx_w_M_Gy"   , "Wy_w_M_Gx"   , "Wy_w_M_Gy"   , "Wz_w_M_Gx"   , "Wz_w_M_Gy"     ],
                               ["Fgt"              , "F_raw"       , "F_w_M"       , "Cx_raw"      , "rec_hope"    , "Cx_w_M_visual" , "Cy_w_M_visual" , ],
                               ["Cx_w_M_Gx"        , "Cx_w_M_Gy"   , "Cy_w_M_Gx"   , "Cy_w_M_Gy"] ]

            single_row_imgs = Matplot_multi_row_imgs(
                                    rows_cols_imgs   = r_c_imgs,         ### 把要顯示的每張圖包成list
                                    rows_cols_titles = r_c_titles,               ### 把每張圖要顯示的字包成list
                                    fig_title        = "%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss         = self.add_loss,
                                    bgr2rgb          = self.bgr2rgb,
                                    fix_size         =(400, 400))  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            print("save to:", self.exp_obj.result_obj.test_write_dir)

            ### W_01 back to W then + M
            gt_min = self.exp_obj.db_obj.db_gt_range.min
            gt_max = self.exp_obj.db_obj.db_gt_range.max
            W_w_M_c_M[..., 0:3] = W_w_M_c_M[..., 0:3] * (gt_max - gt_min) + gt_min
            if(self.exp_obj.db_obj.get_method.value == "build_by_in_I_gt_W_hole_norm_then_mul_M_right"): W_w_M_c_M[..., 0:3] = W_w_M_c_M[..., 0:3] * used_M

            if(self.phase == "test" and self.knpy_save is True):
                ### 定位出 存檔案的位置
                gather_WM_npy_dir  = f"{public_write_dir}/gather_WM_{self.phase}-{current_time}/WM_npy_then_npz"
                gather_WM_knpy_dir = f"{public_write_dir}/gather_WM_{self.phase}-{current_time}/WM_knpy"
                Check_dir_exist_and_build(gather_WM_npy_dir)
                Check_dir_exist_and_build(gather_WM_knpy_dir)

                ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
                WM_npy_path  = f"{gather_WM_npy_dir}/{current_see_name}_pred.npy"
                WM_knpy_path = f"{gather_WM_knpy_dir}/{current_see_name}_pred.knpy"
                np.save(WM_npy_path, W_w_M_c_M)
                Save_npy_path_as_knpy(WM_npy_path, WM_knpy_path)

                ### .npy刪除(因為超占空間) 改存 .npz
                np.savez_compressed(WM_npy_path.replace(".npy", ".npz"), W_w_M_c_M)
                os.remove(WM_npy_path)

                ### rec放一起
                gather_rec_dir  = f"{public_write_dir}/gather_Rec_{self.phase}-{current_time}"
                gather_rec_path = f"{gather_rec_dir}/{current_see_name}_pred.jpg"
                Check_dir_exist_and_build(gather_rec_dir)
                cv2.imwrite(gather_rec_path, rec)

                # breakpoint()
