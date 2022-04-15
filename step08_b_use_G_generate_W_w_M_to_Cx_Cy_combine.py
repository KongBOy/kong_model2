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

class W_w_M_to_Cx_Cy(Use_G_generate):
    def __init__(self, focus=False, tight_crop=None):
        super(W_w_M_to_Cx_Cy, self).__init__()
        self.tight_crop = tight_crop
        self.focus = focus
        if(self.tight_crop is not None): self.tight_crop.jit_scale = 0  ### 防呆 test 的時候我們不用 random jit 囉！

    def doing_things(self):
        current_ep   = self.exp_obj.current_ep
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
        in_WM_pre[..., 3:4] 為 M (1, h, w, 1)
        in_WM_pre[..., 0:3] 為 W (1, h, w, 3) 先z 再y 再x

        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''

        ''' 重新命名 讓我自己比較好閱讀'''
        in_WM             = self.in_ord
        in_WM_pre         = self.in_pre
        Mgt_C             = self.gt_ord
        Mgt_C_pre         = self.gt_pre
        rec_hope          = self.rec_hope

        if(self.tight_crop is not None):
            Mgt_pre = Mgt_C_pre[..., 0:1]

            in_WM     = self.tight_crop(in_WM     , Mgt_pre)
            in_WM_pre = self.tight_crop(in_WM_pre , Mgt_pre)
            Mgt_C     = self.tight_crop(Mgt_C     , Mgt_pre)
            Mgt_C_pre = self.tight_crop(Mgt_C_pre , Mgt_pre)
            rec_hope  = self.tight_crop(rec_hope  , Mgt_pre)
            # self.tight_crop.reset_jit()  ### 注意 test 的時候我們不用 random jit 囉！

        ### 這個是給後處理用的 dis_img
        dis_img  = in_WM[1][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        rec_hope = rec_hope[0].numpy()

        ''' use_model '''
        W_pre   = in_WM_pre[..., 0:3]
        Mgt_pre = in_WM_pre[..., 3:4]
        W_pre_W_M_pre = W_pre * Mgt_pre
        Cx_raw_pre, Cy_raw_pre = self.model_obj.generator(W_pre_W_M_pre, training=self.training)
        ''''''''''''
        ### visualize W_pre
        W_01 = Value_Range_Postprocess_to_01(W_pre)
        W_01 = W_01[0].numpy()
        W_visual, Wx_visual, Wy_visual, Wz_visual  = W_01_visual_op(W_01)

        ### visualize Mgt_pre
        Mgt_visual = (Mgt_pre[0].numpy() * 255).astype(np.uint8)

        ### visualize W_pre_W_M_pre
        W_w_M_01 = Value_Range_Postprocess_to_01(W_pre_W_M_pre)
        W_w_M_01 = W_w_M_01[0].numpy()
        W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual  = W_01_visual_op(W_w_M_01)

        ### Cx_pre, Cy_pre postprocess and visualize
        ### postprocess
        C_raw_pre = np.concatenate([Cy_raw_pre, Cx_raw_pre], axis=-1)  ### tensor 會自動轉 numpy
        C_raw = Value_Range_Postprocess_to_01(C_raw_pre, self.exp_obj.use_gt_range)
        C_raw = C_raw[0]
        Cgt_pre = Mgt_C_pre[0, ..., 1:3].numpy()
        Cgt_01 = Value_Range_Postprocess_to_01(Cgt_pre, self.exp_obj.use_gt_range)

        Mgt = Mgt_C_pre[0, ..., 0:1].numpy()

        ### Cx_pre, Cy_pre postprocess and visualize
        ### postprocess
        # C_pre = np.concatenate([Cy_pre, Cx_pre], axis=-1)  ### tensor 會自動轉 numpy
        # C = Value_Range_Postprocess_to_01(C_pre, exp_obj.use_gt_range)
        # C = C[0]
        # Cgt = Fgt[0, ..., 1:3].numpy()

        if(self.focus is False):
            F,   F_visual,   Cx_visual,   Cy_visual   = C_01_concat_with_M_to_F_and_get_F_visual(C_raw,  Mgt)
            Fgt, Fgt_visual, Cxgt_visual, Cygt_visual = C_01_concat_with_M_to_F_and_get_F_visual(Cgt_01, Mgt)
            F_visual   = F_visual  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            Fgt_visual = Fgt_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
        else:
            F_raw , F_raw_visual , Cx_raw_visual , Cy_raw_visual, F_w_Mgt,   F_w_Mgt_visual,   Cx_w_Mgt_visual,   Cy_w_Mgt_visual = C_01_and_C_01_w_M_to_F_and_visualize(C_raw, Mgt)
            Fgt   , Fgt_visual   , Cxgt_visual   , Cygt_visual  = C_01_concat_with_M_to_F_and_get_F_visual(Cgt_01, Mgt)
            F_raw_visual   = F_raw_visual  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            F_w_Mgt_visual = F_w_Mgt_visual  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
            Fgt_visual     = Fgt_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

        if(current_ep == 0 or self.see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_rec_write_dir)    ### 建立 放輔助檔案 的資料夾
            ###################
            cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",          dis_img)
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
            if(self.npz_save is False): np.save            (private_write_dir + "/" + "epoch_%04i_u1b1_flow" % current_ep, F)
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + "epoch_%04i_u1b1_flow" % current_ep, F)
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2_flow.jpg" % current_ep, F_visual)
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3_Cx.jpg"   % current_ep, Cx_visual)
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4_Cy.jpg"   % current_ep, Cy_visual)

        else:
            if(self.npz_save is False): np.save            (private_write_dir + "/" + "epoch_%04i_u1b1_F_w_Mgt" % current_ep, F_w_Mgt)          ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            if(self.npz_save is True ): np.savez_compressed(private_write_dir + "/" + "epoch_%04i_u1b1_F_w_Mgt" % current_ep, F_w_Mgt)          ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2_F_raw.jpg"    % current_ep, F_raw_visual)     ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3_F_w_Mgt.jpg"  % current_ep, F_w_Mgt_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4_Cx_raw.jpg"   % current_ep, Cx_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b5_Cx_w_Mgt.jpg" % current_ep, Cx_w_Mgt_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b6_Cy_raw.jpg"   % current_ep, Cy_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
            cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b7_Cy_w_Mgt.jpg" % current_ep, Cy_w_Mgt_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～


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
                receptive_filed_feature_length = get_receptive_filed_feature_length(kernel_size, strides, layer, ord_len=dis_img.shape[0])

                ### 模擬訓練中怎麼縮小， 這邊就怎麼縮小， 可以參考 step10_loss 裡的 BCE loss 喔～
                BCE_Mask_type = self.model_obj.train_step.BCE_Mask_type.lower()
                if  (BCE_Mask_type == "erosion"):  M_used = tf_M_resize_then_erosion_by_kong(Mgt_pre, resize_h=receptive_filed_feature_length, resize_w=receptive_filed_feature_length)
                elif(BCE_Mask_type == "bicubic"):  M_used = tf.image.resize(Mgt_pre, size=(receptive_filed_feature_length, receptive_filed_feature_length), method=tf.image.ResizeMethod.BICUBIC)
                elif(BCE_Mask_type == "area"   ):  M_used = tf.image.resize(Mgt_pre, size=(receptive_filed_feature_length, receptive_filed_feature_length), method=tf.image.ResizeMethod.AREA)
                elif(BCE_Mask_type == "nearest"):  M_used = tf.image.resize(Mgt_pre, size=(receptive_filed_feature_length, receptive_filed_feature_length), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                ### BHWC -> HWC
                M_used = M_used[0]

                ### 取得 Mask 在原影像 的 receptive_filed_mask
                receptive_filed_mask   = get_receptive_field_mask(kernel_size=kernel_size, strides=strides, layer=layer, img_shape=dis_img.shape, Mask=M_used, vmin=0.5)  ### return HWC， vmin=0.5 是為了等等相乘時留一點透明度

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
                                fig_title        = "epoch_%04i_Discriminator" % current_ep,  ### 圖上的大標題
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
                #                         fig_title ="epoch_%04i_Discriminator" % current_ep,  ### 圖上的大標題
                #                         add_loss  =add_loss,
                #                         bgr2rgb   =bgr2rgb,    ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                #                         where_colorbar = [None, True, True, None, None, True, True, None],
                #                         w_same_as_first=True)

            else:
                ''' GAN 的 Discriminator 沒用 Mask 的 Case '''
                single_row_imgs = Matplot_single_row_imgs(
                            imgs            = [ F_w_Mgt_visual ,  fake_score ,  Fgt_visual,   real_score],   ### 把要顯示的每張圖包成list
                            img_titles      = ["F_w_Mgt_visual", "fake_score", "Fgt_visual", "real_score"],  ### 把每張圖要顯示的字包成list
                            fig_title       = "epoch_%04i_Discriminator" % current_ep,  ### 圖上的大標題
                            add_loss        = self.add_loss,
                            bgr2rgb         = self.bgr2rgb,  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                            where_colorbar  = [None, True, None, True],
                            w_same_as_first = True,
                            one_ch_vmin=0,
                            one_ch_vmax=1)

            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=private_write_dir, name="epoch_%04i_u1b8_Disc" % current_ep)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            # print("save to:", private_write_dir)
            # breakpoint()

        if(self.postprocess):
            current_see_name = used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=dis_img, flow=F_w_Mgt)
            '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
            cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)

            if(self.focus is False):
                imgs       = [ W_visual ,   Mgt_visual , W_w_M_visual,  F_visual ,    rec,   rec_hope],    ### 把要顯示的每張圖包成list
                img_titles = ["W_01",        "Mgt",        "W_w_M",     "pred_F", "pred_rec", "rec_hope"], ### 把每張圖要顯示的字包成list
            else:
                imgs       = [ W_visual ,   Mgt_visual , W_w_M_visual,  F_raw_visual, F_w_Mgt_visual,    rec,      rec_hope],         ### 把要顯示的每張圖包成list
                img_titles = ["W_01",        "Mgt",        "W_w_M",   "F_raw_visual", "F_w_Mgt_visual",     "pred_rec", "rec_hope"],  ### 把每張圖要顯示的字包成list

            single_row_imgs = Matplot_single_row_imgs(
                                    imgs       = imgs,
                                    img_titles = img_titles,
                                    fig_title  = "%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss   = self.add_loss,
                                    bgr2rgb    = self.bgr2rgb,  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
                                    w_same_as_first=True)
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
            print("save to:", self.exp_obj.result_obj.test_write_dir)
