import numpy as np
import cv2

from step08_b_use_G_generate_0_util import Use_G_generate, Value_Range_Postprocess_to_01, WcM_01_visual_op, W_01_and_W_01_w_M_to_WM_and_visualize, C_01_and_C_01_w_M_to_F_and_visualize, C_01_concat_with_M_to_F_and_get_F_visual
from kong_util.flow_bm_util import check_flow_quality_then_I_w_F_to_R

from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs, Matplot_multi_row_imgs

import matplotlib.pyplot as plt
import os
import pdb

class I_w_M_to_W_to_C(Use_G_generate):
    def __init__(self, separate_out=False, focus=False, tight_crop=None):
        super(I_w_M_to_W_to_C, self).__init__()
        self.separate_out = separate_out
        self.focus = focus
        self.tight_crop = tight_crop

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
        private_npz_write_dir = used_sees[self.index].npz_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        public_write_dir      = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        '''
        gt_ord/pre 的 [0]第一個是 取 wc, [1] 是取 flow, 第二個[0]是取 batch， 這次試試看用 M 不用 M_pre

        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''

        ''' 重新命名 讓我自己比較好閱讀'''
        I_pre   = self.in_pre
        Wgt_ord = self.gt_ord [0]
        Wgt_pre = self.gt_pre [0]
        Fgt_ord = self.gt_ord [1]
        Fgt_pre = self.gt_pre [1]

        rec_hope = self.rec_hope

        ''' use model '''''''''''''''''''''
        Mgt     = self.Wgt_ord[0][..., 3:4]  ### [0]第一個是 取 wc, [1] 是取 flow, 第二個[0]是取 batch， 這次試試看用 M 不用 M_pre
        Mgt_pre = self.Wgt_pre[0][..., 3:4]  ### 但是思考了一下，因為我現在focus train， 頁面外面是 灰色的， 如果 頁面外面 Mask 外面 有一大片 微小的值 會不會 GG呢 ??? 好像改回 Mgt_pre 比較安全???
        used_M   = Mgt_pre

        I_pre_w_M = I_pre * used_M  ### 這次試試看用 M 不用 M_pre
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
        ''' use_model end '''''''''''''''''''''''''''''''''''''''

        '''model in visualize'''
        I_01_w_M  = Value_Range_Postprocess_to_01(I_pre_w_M, self.exp_obj.use_gt_range)
        I_w_M_visual = (I_01_w_M[0].numpy() * 255).astype(np.uint8)
        Mgt_visual   = (used_M[0].numpy() * 255).astype(np.uint8)

        '''model out visualize'''  ### 後處理： 拿掉 batch 和 弄成01 和 轉成 numpy
        if(self.focus is False):
            W_visual,   Wx_visual,   Wy_visual,   Wz_visual   = WcM_01_visual_op(W_01_raw)
            F,           F_visual,   Cx_visual,   Cy_visual   = C_01_concat_with_M_to_F_and_get_F_visual(C_01_raw,  Mgt)
            F_visual   = F_visual  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

        else:
            W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual, W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual = W_01_and_W_01_w_M_to_WM_and_visualize(W_01_raw, used_M)
            F_raw    , F_raw_visual, Cx_raw_visual, Cy_raw_visual, F_w_Mgt,   F_w_M_visual,   Cx_w_M_visual,   Cy_w_M_visual   = C_01_and_C_01_w_M_to_F_and_visualize(C_01_raw, used_M)

        '''model gt visualize'''
        Wgt_pre = Wgt_pre[0].numpy()
        Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, self.exp_obj.use_gt_range)
        Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = WcM_01_visual_op(Wgt_01)

        Cgt_pre = Fgt_pre[0, ..., 1:3].numpy()
        Cgt_01  = Value_Range_Postprocess_to_01(Cgt_pre, self.exp_obj.use_gt_range)
        Fgt, Fgt_visual, Cxgt_visual, Cygt_visual = C_01_concat_with_M_to_F_and_get_F_visual(Cgt_01, used_M)

        ''' model postprocess輔助 visualize'''
        dis_img_visual  = (I_pre[0].numpy() * 255).astype(np.uint8)
        rec_hope = rec_hope[0].numpy()

        '''cv2 bgr 與 rgb 的調整'''
        ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
        if(self.bgr2rgb):
            dis_img_visual = dis_img_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            rec_hope = rec_hope[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            I_w_M_visual = I_w_M_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

        if(current_ep == 0 or self.see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_rec_write_dir)    ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_npz_write_dir)    ### 建立 放輔助檔案 的資料夾
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      dis_img_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt(in_img).jpg", I_w_M_visual)

            if(self.npz_save is False): np.save            (private_write_dir     + "/" + "0b_u1b1-gt_W", Wgt_01)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + "0b_u1b1-gt_W", Wgt_01)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",  Wgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg", Wxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg", Wygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg", Wzgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)

            if(self.npz_save is False): np.save            (private_write_dir     + "/" + "0b_u2b1-gt_b_Fgt", Fgt)
            if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + "0b_u2b1-gt_b_Fgt", Fgt)
            cv2.imwrite(private_write_dir + "/" + "0b_u2b2-gt_b_Fgt.jpg", Fgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u2b3-gt_b_Cxgt.jpg",   Cxgt_visual)
            cv2.imwrite(private_write_dir + "/" + "0b_u2b4-gt_b_Cygt.jpg",   Cygt_visual)
            cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)

        if(self.npz_save is False): np.save            (private_write_dir     + "/" + "epoch_%04i_u1b1-W_w_Mgt" % ep_it_string, W_w_M_c_M)
        if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + "epoch_%04i_u1b1-W_w_Mgt" % ep_it_string, W_w_M_c_M)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2-W_raw_visual.jpg"    % ep_it_string, W_raw_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3-W_w_M_visual.jpg"    % ep_it_string, W_w_M_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4-Wx_raw_visual.jpg"   % ep_it_string, Wx_raw_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b5-Wx_w_M_visual.jpg"   % ep_it_string, Wx_w_M_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b6-Wy_raw_visual.jpg"   % ep_it_string, Wy_raw_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b7-Wy_w_M_visual.jpg"   % ep_it_string, Wy_w_M_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b8-Wz_raw_visual.jpg"   % ep_it_string, Wz_raw_visual)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b9-Wz_w_M_visual.jpg"   % ep_it_string, Wz_w_M_visual)

        if(self.npz_save is False): np.save            (private_write_dir     + "/" + "epoch_%04i_u2b1-F_w_Mgt" % ep_it_string, F_w_Mgt)
        if(self.npz_save is True ): np.savez_compressed(private_npz_write_dir + "/" + "epoch_%04i_u2b1-F_w_Mgt" % ep_it_string, F_w_Mgt)
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b2-F_raw.jpg"    % ep_it_string, F_raw_visual)     ### 把 生成的 F_visual 存進相對應的資料夾
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b3-F_w_Mgt.jpg"  % ep_it_string, F_w_M_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b4-Cx_raw.jpg"   % ep_it_string, Cx_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b5-Cx_w_Mgt.jpg" % ep_it_string, Cx_w_M_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b6-Cy_raw.jpg"   % ep_it_string, Cy_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
        cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b7-Cy_w_Mgt.jpg" % ep_it_string, Cy_w_M_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

        if(self.postprocess):
            current_see_name = self.fname.split(".")[0]   # used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=dis_img_visual, flow=F_w_Mgt)
            '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
            # gt_bm, gt_rec = check_F_quality_then_I_w_F_to_R(dis_img=dis_img_visual, F=Fgt)
            cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)

            if(self.focus is False):
                r_c_imgs   = [ [dis_img_visual , Mgt_visual        , I_w_M_visual  , rec             , rec_hope ],
                               [W_visual       , Wx_visual         , Wy_visual     , Wz_visual       , ],
                               [F_visual       , Cx_visual         , Cy_visual     , ] ]
                r_c_titles = [ ["dis_img"      , "Mgt"             , "I_with_M"    , "rec"           , "rec_hope"],
                               ["W"            , "Wx"              , "Wy"          , "Wz"            ,    ],
                               ["F"            , "Cx"              , "Cy"          , ] ]

            else:
                r_c_imgs   = [ [dis_img_visual , Mgt_visual        , I_w_M_visual  , rec             , rec_hope ],
                               [Wgt_visual     , W_raw_visual      , W_w_M_visual  , Wx_raw_visual   , Wx_w_M_visual , Wy_raw_visual, Wy_w_M_visual , Wz_raw_visual, Wz_w_M_visual ],
                               [Fgt_visual     , F_raw_visual      , F_w_M_visual  , Cx_raw_visual   , Cx_w_M_visual , Cy_raw_visual, Cy_w_M_visual ] ]
                r_c_titles = [ ["dis_img"      , "Mgt"             , "I_with_M"    , "rec"           , "rec_hope"],
                               ["Wgt"          , "W_raw"           , "W_w_M"       , "Wx_raw"        , "Wx_w_M"      , "Wy_raw"     , "Wy_w_M"      , "Wz_raw"     , "Wz_w_M"      ],
                               ["Fgt"          , "F_raw"           , "F_w_M"       , "Cx_raw"        , "rec_hope"      , "Cx_w_M_visual" , "Cy_w_M_visual" ] ]

            single_row_imgs = Matplot_multi_row_imgs(
                                    rows_cols_imgs   = r_c_imgs,         ### 把要顯示的每張圖包成list
                                    rows_cols_titles = r_c_titles,               ### 把每張圖要顯示的字包成list
                                    fig_title        = "%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss         = self.add_loss,
                                    bgr2rgb          = self.bgr2rgb,
                                    one_ch_vmin      = 0,
                                    one_ch_vmax      = 255)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
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