import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from step08_b_use_G_generate_0_util import Use_G_generate_Interface

import matplotlib.pyplot as plt
import os
import pdb

def M_save_as_fake_F_and_WcM(gather_base_dir, current_see_name, M):
    '''
    M 要 h,w,c 喔
    '''

    h, w = M.shape[:2]
    ### 準備存 fake_F
    fake_C = np.zeros(shape=(h, w, 2), dtype=np.float32)
    fake_F = np.concatenate((M, fake_C), axis=-1)
    fake_F = fake_F.astype(np.float32)

    ### 定位出 存檔案的位置
    gather_fake_F_npy_dir  = gather_base_dir + "/1_uv-1_npy_then_npz"
    gather_fake_F_knpy_dir = gather_base_dir + "/1_uv-3_knpy"
    Check_dir_exist_and_build(gather_fake_F_npy_dir)
    Check_dir_exist_and_build(gather_fake_F_knpy_dir)

    ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
    fake_F_npy_path  = f"{gather_fake_F_npy_dir}/{current_see_name}.npy"
    fake_F_knpy_path = f"{gather_fake_F_knpy_dir}/{current_see_name}.knpy"
    np.save(fake_F_npy_path, fake_F)
    Save_npy_path_as_knpy(fake_F_npy_path, fake_F_knpy_path)
    print("fake_F_npy_path     :", fake_F_npy_path)
    print("fake_F_knpy_path    :", fake_F_knpy_path)

    ### .npy刪除(因為超占空間) 改存 .npz
    np.savez_compressed(fake_F_npy_path.replace(".npy", ".npz"), fake_F)
    os.remove(fake_F_npy_path)
    ###############################################################################
    ### 準備存 fake_W_w_M (我是覺得不用存 W 了， 因為已經包含再 W_w_M 裡面了)
    fake_W = np.zeros(shape=(h, w, 3), dtype=np.float32)
    fale_W_w_M = np.concatenate((fake_W, M), axis=-1)
    fale_W_w_M = fale_W_w_M.astype(np.float32)

    ### 定位出 存檔案的位置
    gather_fale_W_w_M_npy_dir  = gather_base_dir + "/2_wc-4_W_w_M_npy_then_npz"
    gather_fale_W_w_M_knpy_dir = gather_base_dir + "/2_wc-5_W_w_M_knpy"
    Check_dir_exist_and_build(gather_fale_W_w_M_npy_dir)
    Check_dir_exist_and_build(gather_fale_W_w_M_knpy_dir)

    ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
    fale_W_w_M_npy_path  = f"{gather_fale_W_w_M_npy_dir}/{current_see_name}.npy"
    fale_W_w_M_knpy_path = f"{gather_fale_W_w_M_knpy_dir}/{current_see_name}.knpy"
    np.save(fale_W_w_M_npy_path, fale_W_w_M)
    Save_npy_path_as_knpy(fale_W_w_M_npy_path, fale_W_w_M_knpy_path)
    print("fale_W_w_M_npy_path :", fale_W_w_M_npy_path)
    print("fale_W_w_M_knpy_path:", fale_W_w_M_knpy_path)

    ### .npy刪除(因為超占空間) 改存 .npz
    np.savez_compressed(fale_W_w_M_npy_path.replace(".npy", ".npz"), fale_W_w_M)
    os.remove(fale_W_w_M_npy_path)

def M_save_as_Trimap(gather_base_dir, current_see_name, Mask, dil_ksize, ero_ksize, status, dis_img, dil_comment, ero_comment):
    ### 這段可以看一下 官方給的 dilate/erose 的 kernel 大概可以長怎樣
    # canvas_size = 5
    # nrows = 1
    # ncols = 3
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(canvas_size * ncols, canvas_size * nrows))

    # RECT_kernel    = cv2.getStructuringElement(cv2.MORPH_RECT   , (11, 11))  ### 全1
    # CROSS_kernel   = cv2.getStructuringElement(cv2.MORPH_CROSS  , (11, 11))  ### 十字架
    # ELLIPSE_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  ### 橢圓形

    # ax[0].imshow(RECT_kernel   , vmin=0, vmax=1)
    # ax[1].imshow(CROSS_kernel  , vmin=0, vmax=1)
    # ax[2].imshow(ELLIPSE_kernel, vmin=0, vmax=1)
    # fig.tight_layout()
    # plt.show()

    '''
    目前的status: ord/pre/255
    '''

    tri_M_bin   = np.where(Mask > 0.5, 1, 0).astype(np.float32)
    if(dil_ksize > 0):
        dil_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT   , (dil_ksize, dil_ksize))  ### 全1
        tri_dilated = cv2.dilate(tri_M_bin, dil_kernel, iterations=1) * 255
    else: tri_dilated = tri_M_bin.copy() * 255

    if(ero_ksize > 0):
        ero_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT   , (ero_ksize, ero_ksize))  ### 全1
        tri_erosed  = cv2.erode (tri_M_bin, ero_kernel, iterations=1) * 255
    else: tri_erosed = tri_M_bin.copy()  * 255

    ### 
    tri_M = tri_dilated.copy()
    tri_M[ ( (tri_dilated == 255) & (tri_erosed == 0) ) ] = 128
    tri_M = tri_M.astype(np.uint8)

    ### 定位出 存檔案的位置
    gather_trimap_msk  = gather_base_dir + f"/3_trimap_{status}_dil_k{dil_comment}__ero_k{ero_comment}/msk"
    gather_trimap_tri  = gather_base_dir + f"/3_trimap_{status}_dil_k{dil_comment}__ero_k{ero_comment}/tri"
    gather_trimap_img  = gather_base_dir + f"/3_trimap_{status}_dil_k{dil_comment}__ero_k{ero_comment}/img"
    Check_dir_exist_and_build(gather_trimap_msk)
    Check_dir_exist_and_build(gather_trimap_tri)
    Check_dir_exist_and_build(gather_trimap_img)
    tri_msk_path = f"{gather_trimap_msk}/{current_see_name}.png"
    tri_map_path = f"{gather_trimap_tri}/{current_see_name}.png"
    tri_img_path = f"{gather_trimap_img}/{current_see_name}.png"

    ### 存起來
    cv2.imwrite(tri_msk_path, (Mask * 255).astype(np.uint8))
    cv2.imwrite(tri_map_path, tri_M)
    cv2.imwrite(tri_img_path, dis_img)

    return tri_M


class I_to_M(Use_G_generate_Interface):
    def __init__(self, tight_crop=None):
        super(I_to_M, self).__init__()
        self.tight_crop = tight_crop
        if(self.tight_crop is not None): self.tight_crop.jit_scale = 0  ### 防呆 test 的時候我們不用 random jit 囉！

    def doing_things(self):
        current_ep = self.exp_obj.current_ep
        current_ep_it   = self.exp_obj.current_ep_it
        it_see_fq   = self.exp_obj.it_see_fq
        if(it_see_fq is None): ep_it_string = "epoch%03i"        %  current_ep
        else                  : ep_it_string = "epoch%03i_it%06i" % (current_ep, current_ep_it)
        current_time = self.exp_obj.current_time
        if  (self.phase == "train"): used_sees = self.exp_obj.result_obj.sees
        elif(self.phase == "test"):  used_sees = self.exp_obj.result_obj.tests
        private_write_dir      = used_sees[self.index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        private_mask_write_dir = used_sees[self.index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
        public_write_dir       = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        # print('public_write_dir:', public_write_dir)

        ''' 重新命名 讓我自己比較好閱讀'''
        dis_img_ord       = self.in_ord
        dis_img_pre       = self.in_pre
        gt_mask_coord     = self.gt_ord
        gt_mask_coord_pre = self.gt_pre
        rec_hope          = self.rec_hope

        if(self.tight_crop is not None):
            gt_mask_pre = gt_mask_coord_pre[..., 0:1]
            gt_mask_coord    , _ = self.tight_crop(gt_mask_coord, gt_mask_pre)
            # gt_mask_coord_pre, _ = self.tight_crop(gt_mask_coord_pre, gt_mask_pre)  ### 沒用到
            # fig, ax = plt.subplots(nrows=1, ncols=2)
            # ax[0].imshow( gt_mask_pre[0])
            # ax[1].imshow( gt_mask_coord[0, ..., 0:1])
            # plt.show()
            # self.tight_crop.reset_jit()  ### 注意 test 的時候我們不用 random jit 囉！

            ##### dis_img_ord 在 tight_crop 要用 dis_img_pre 來反推喔！
            ### 取得 crop 之前的大小
            ord_h, ord_w = dis_img_ord.shape[1:3]    ### BHWC， 取 HW, 3024, 3024
            pre_h, pre_w = dis_img_pre.shape[1:3]    ### BHWC， 取 HW,  512,  512 或 448, 448 之類的
            ### 算出 ord 和 pre 之間的比例
            ratio_h_p2o  = ord_h / pre_h  ### p2o 是 pre_to_ord 的縮寫
            ratio_w_p2o  = ord_w / pre_w  ### p2o 是 pre_to_ord 的縮寫
            ### 對 pre 做 crop
            dis_img_pre_croped_resized, pre_boundary = self.tight_crop(dis_img_pre  , gt_mask_pre)  ### 可以看一下 丟進去model 的img 長什麼樣子
            ### 根據比例 放大回來 crop 出 ord，目前這個case好像用不到，這是在rec的時候才會用到，如果是要 建 dataset 的話 要存 dis_img_ord 才對喔
            ord_l_pad    = np.round(pre_boundary["l_pad_slice"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_r_pad    = np.round(pre_boundary["r_pad_slice"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_t_pad    = np.round(pre_boundary["t_pad_slice"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_d_pad    = np.round(pre_boundary["d_pad_slice"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_l_out_amo = np.round(pre_boundary["l_out_amo"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_t_out_amo = np.round(pre_boundary["t_out_amo"].numpy() * ratio_w_p2o).astype(np.int32)
            ord_r_out_amo = np.round(pre_boundary["r_out_amo"].numpy() * ratio_h_p2o).astype(np.int32)
            ord_d_out_amo = np.round(pre_boundary["d_out_amo"].numpy() * ratio_h_p2o).astype(np.int32)
            dis_img_ord_croped_not_accurate = np.pad(dis_img_ord.numpy(), ( (0, 0), (ord_t_out_amo, ord_d_out_amo), (ord_l_out_amo, ord_r_out_amo), (0, 0)  ))  ### BHWC， not_accurate 意思是可能會差1個pixel， 因為 乘完 ratio_p2o 視作四捨五入 
            dis_img_ord_croped_not_accurate = dis_img_ord_croped_not_accurate[:, ord_t_pad : ord_d_pad , ord_l_pad : ord_r_pad , :]  ### BHWC， not_accurate 意思是可能會差1個pixel， 因為 乘完 ratio_p2o 視作四捨五入 
            ### 可能還是會差 1個 pixel 之類的

        ''' use_model '''
        M_pre = self.model_obj.generator(dis_img_pre_croped_resized, training=self.training)
        M_pre = M_pre[0].numpy()
        M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
        M_visual = (M * 255).astype(np.uint8)

        '''
        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''
        dis_img_ord_croped_not_accurate  =  dis_img_ord_croped_not_accurate[0].astype(np.uint8)             ### 目前這個case好像用不到，這是在rec的時候才會用到，如果是要 建 dataset 的話 要存 dis_img_ord 才對喔， not_accurate 意思是可能會差1個pixel， 因為 乘完 ratio_p2o 視作四捨五入 
        dis_img_pre_croped_resized       = (dis_img_pre_croped_resized[0].numpy() * 255 ).astype(np.uint8)  ### 可以看一下 丟進去model 的img 長什麼樣子
        dis_img_ord                      = (dis_img_ord       [0].numpy()       ).astype(np.uint8)
        dis_img_pre                      = (dis_img_pre       [0].numpy() * 255 ).astype(np.uint8)
        Mgt_visual                       = (gt_mask_coord[0, ..., 0:1].numpy() * 255).astype(np.uint8)
        rec_hope                         = rec_hope[0].numpy()
        # plt.figure()
        # plt.imshow(Mgt_visual)
        # plt.show()
        # print("Mgt_visual.dtype:", Mgt_visual.dtype)
        # print("Mgt_visual.shape:", Mgt_visual.shape)
        # print("Mgt_visual.max():", Mgt_visual.numpy().max())
        # print("Mgt_visual.min():", Mgt_visual.numpy().min())

        if(self.bgr2rgb):
            dis_img_ord_croped_not_accurate = dis_img_ord_croped_not_accurate[:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
            dis_img_pre_croped_resized = dis_img_pre_croped_resized[:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
            dis_img_ord        = dis_img_ord       [:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
            dis_img_pre        = dis_img_pre       [:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
            rec_hope           = rec_hope          [:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr

        ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        if(current_ep == 0 or self.see_reset_init):
            Check_dir_exist_and_build(private_write_dir)
            Check_dir_exist_and_build(private_mask_write_dir)
            cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img(in_img).jpg", dis_img_ord)  ### 存 dis_img_ord 沒錯， 這樣子做 tight_crop才正確 不是存 dis_img_ord_croped_not_accurate 喔！ 因為本身已經做過一次tight_crop了， 這樣子再做tight_crop 就多做一次囉～
            cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img_pre_croped_resized.jpg", dis_img_pre_croped_resized)  ### 可以看一下 丟進去model 的img 長什麼樣子
            cv2.imwrite(private_write_dir  + "/" + "0b_u1b1-gt_mask.jpg", Mgt_visual)
        cv2.imwrite(    private_mask_write_dir + "/" + f"{ep_it_string}-u1b1_mask.jpg", M_visual)

        if(self.postprocess):
            current_see_name = self.fname.split(".")[0]   # used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾

            from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs
            imgs       = [ dis_img_ord_croped_not_accurate,   dis_img_pre_croped_resized,   M_visual , Mgt_visual]
            img_titles = ["dis_img_ord_croped_not_accurate",  "dis_img_pre_croped_resized",   "M",   "Mgt_visual"]

            single_row_imgs = Matplot_single_row_imgs(
                                    imgs      =imgs,         ### 把要顯示的每張圖包成list
                                    img_titles=img_titles,   ### 把每張圖要顯示的字包成list
                                    fig_title ="%s, epoch=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss  =self.add_loss,
                                    bgr2rgb   =self.bgr2rgb)
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 如果沒有要接續畫loss，就可以存了喔！

            '''
            Fake_F 的部分
            '''
            if(self.phase == "test" and self.knpy_save is True):
                db_h = self.exp_obj.db_obj.h
                db_w = self.exp_obj.db_obj.w

                if(self.tight_crop is None):
                    M = cv2.resize(M, (db_w, db_h))  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 所以要 back 回 pre的原始大小喔！
                    M = M.reshape(db_h, db_w, 1)     ### 把 ch加回來
                else:
                    M = self.tight_crop.croped_back(M, pre_boundary, back_w=448, back_h=448)

                gather_mask_dir   = public_write_dir + "/pred_mask"
                Check_dir_exist_and_build(gather_mask_dir)
                cv2.imwrite(f"{gather_mask_dir}/{current_see_name}.jpg", M_visual)

                print("")
                ###############################################################################
                ### 定位出 存檔案的Base位置
                gather_base_dir = public_write_dir + f"/Gather_Pred-{current_time}"
                Check_dir_exist_and_build(gather_base_dir)

                ###############################################################################
                ### 準備存 dis_img
                gather_dis_img_ord_dir_w_Crop  = gather_base_dir + "/0_dis_img_ord_w_Crop"  ### 目前這個case好像用不到，這是在rec的時候才會用到，如果是要 建 dataset 的話 要存 dis_img_ord 才對喔， not_accurate 意思是可能會差1個pixel， 因為 乘完 ratio_p2o 視作四捨五入 
                gather_dis_img_pre_dir_w_Crop  = gather_base_dir + "/0_dis_img_pre_w_Crop"  ### 可以看一下 丟進去model 的img 長什麼樣子
                gather_dis_img_ord_dir         = gather_base_dir + "/0_dis_img"
                gather_dis_img_pre_dir         = gather_base_dir + "/0_dis_img_pre"
                Check_dir_exist_and_build(gather_dis_img_ord_dir_w_Crop)
                Check_dir_exist_and_build(gather_dis_img_pre_dir_w_Crop)
                Check_dir_exist_and_build(gather_dis_img_ord_dir)
                Check_dir_exist_and_build(gather_dis_img_pre_dir)
                dis_img_ord_w_Crop_path  = f"{gather_dis_img_ord_dir_w_Crop}/{current_see_name}.{self.exp_obj.db_obj.in_format}"
                dis_img_pre_w_Crop_path  = f"{gather_dis_img_pre_dir_w_Crop}/{current_see_name}.{self.exp_obj.db_obj.in_format}"
                dis_img_ord_path         = f"{gather_dis_img_ord_dir}/{current_see_name}.{self.exp_obj.db_obj.in_format}"
                dis_img_pre_path         = f"{gather_dis_img_pre_dir}/{current_see_name}.{self.exp_obj.db_obj.in_format}"
                cv2.imwrite(dis_img_ord_w_Crop_path, dis_img_ord_croped_not_accurate)  ### 目前這個case好像用不到，這是在rec的時候才會用到，如果是要 建 dataset 的話 要存 dis_img_ord 才對喔， not_accurate 意思是可能會差1個pixel， 因為 乘完 ratio_p2o 視作四捨五入 
                cv2.imwrite(dis_img_pre_w_Crop_path, dis_img_pre_croped_resized)
                cv2.imwrite(dis_img_ord_path      , dis_img_ord)
                cv2.imwrite(dis_img_pre_path      , dis_img_pre)
                ###############################################################################
                ### 準備存 rec_hope
                gather_rec_hope_dir  = gather_base_dir + "/0_rec_hope"
                Check_dir_exist_and_build(gather_rec_hope_dir)
                rec_hope_path  = f"{gather_rec_hope_dir}/{current_see_name}.{self.exp_obj.db_obj.in_format}"
                cv2.imwrite(rec_hope_path, rec_hope)
                ###############################################################################
                ### 存 fake_F 和 fake_WcM
                M_save_as_fake_F_and_WcM(gather_base_dir, current_see_name, M)
                ###############################################################################
                ### 存 trimap
                dil_ksize_pre = 45
                ero_ksize_pre =  1
                dil_ksize_ord = int(dil_ksize_pre * ratio_h_p2o)
                ero_ksize_ord = int(ero_ksize_pre * ratio_h_p2o)
                dil_ksize_255 = int(dil_ksize_pre * 255 / 448)
                ero_ksize_255 = int(ero_ksize_pre * 255 / 448)
                tri_M_pre       = M.copy()
                tri_M_ord       = cv2.resize(tri_M_pre, (ord_w, ord_h))
                tri_M_255       = cv2.resize(tri_M_pre, (255  , 255))
                tri_M_ord = M_save_as_Trimap(gather_base_dir=gather_base_dir, current_see_name=current_see_name, Mask=tri_M_ord, dil_ksize=dil_ksize_ord, ero_ksize=ero_ksize_ord, status="ord", dis_img=dis_img_ord                        , dil_comment=dil_ksize_pre, ero_comment=ero_ksize_pre)  ### comment 統一用pre才對， 因為我是用pre當基準， 要不然數字會一直變動
                tri_M_pre = M_save_as_Trimap(gather_base_dir=gather_base_dir, current_see_name=current_see_name, Mask=tri_M_pre, dil_ksize=dil_ksize_pre, ero_ksize=ero_ksize_pre, status="pre", dis_img=dis_img_pre                        , dil_comment=dil_ksize_pre, ero_comment=ero_ksize_pre)  ### comment 統一用pre才對， 因為我是用pre當基準， 要不然數字會一直變動
                tri_M_255 = M_save_as_Trimap(gather_base_dir=gather_base_dir, current_see_name=current_see_name, Mask=tri_M_255, dil_ksize=dil_ksize_255, ero_ksize=ero_ksize_255, status="255", dis_img=cv2.resize(dis_img_ord, (255, 255)), dil_comment=dil_ksize_pre, ero_comment=ero_ksize_pre)  ### comment 統一用pre才對， 因為我是用pre當基準， 要不然數字會一直變動
                ###############################################################################


######################################################################################################################################################################################################
######################################################################################################################################################################################################
def use_model(model_obj, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    M_pre = model_obj.generator(in_img_pre, training=training)
    M_pre = M_pre[0].numpy()
    M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
    M_visual = (M * 255).astype(np.uint8)
    return M, M_visual

def I_Generate_M_see(model_obj, phase, index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "train"): used_sees = exp_obj.result_obj.sees
    elif(phase == "test"):  used_sees = exp_obj.result_obj.tests
    private_write_dir      = used_sees[index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_mask_write_dir = used_sees[index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    public_write_dir       = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    # print('public_write_dir:', public_write_dir)
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    in_img    = in_img[0].numpy()
    gt_mask   = (gt_mask_coord[0, ..., 0:1].numpy() * 255).astype(np.uint8)
    # print("gt_mask.dtype:", gt_mask.dtype)
    # print("gt_mask.shape:", gt_mask.shape)
    # print("gt_mask.max():", gt_mask.numpy().max())
    # print("gt_mask.min():", gt_mask.numpy().min())

    pred_mask, pred_mask_visual = use_model(model_obj, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)

    # print("bgr2rgb", bgr2rgb)
    if(bgr2rgb): in_img = in_img[:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr

    if(current_ep == 0 or see_reset_init):                                              ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)                                   ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(private_mask_write_dir)                                  ### 建立 model生成的結果 的資料夾
        cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img(in_img).jpg", in_img)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(private_write_dir  + "/" + "0b_u1b1-gt_mask.jpg", gt_mask)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(    private_mask_write_dir + "/" + "epoch_%04i_u1b1_mask.jpg" % current_ep, pred_mask_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs
        imgs = [ in_img ,   pred_mask_visual , gt_mask]
        img_titles = ["in_img", "pred_mask", "gt_mask"]

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =imgs,         ### 把要顯示的每張圖包成list
                                img_titles=img_titles,               ### 把每張圖要顯示的字包成list
                                fig_title ="%s, epoch=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 如果沒有要接續畫loss，就可以存了喔！

        '''
        Fake_F 的部分
        '''
        if(phase == "test"):
            gather_mask_dir   = public_write_dir + "/pred_mask"
            Check_dir_exist_and_build(gather_mask_dir)
            cv2.imwrite(f"{gather_mask_dir}/{current_see_name}.jpg", pred_mask_visual)

            h, w = pred_mask.shape[:2]
            fake_name = current_see_name.split(".")[0]
            print("")
            ###############################################################################
            fake_C = np.zeros(shape=(h, w, 2), dtype=np.float32)
            fake_F = np.concatenate((pred_mask, fake_C), axis=-1)
            fake_F = fake_F.astype(np.float32)

            ### 定位出 存檔案的位置
            gather_fake_F_dir = public_write_dir + "/pred_mask/fake_F"
            gather_fake_F_npy_dir  = gather_fake_F_dir + "/1 npy_then_npz"
            gather_fake_F_knpy_dir = gather_fake_F_dir + "/2 knpy"
            Check_dir_exist_and_build(gather_fake_F_dir)
            Check_dir_exist_and_build(gather_fake_F_npy_dir)
            Check_dir_exist_and_build(gather_fake_F_knpy_dir)

            ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
            fake_F_npy_path  = f"{gather_fake_F_npy_dir}/{fake_name}.npy"
            fake_F_knpy_path = f"{gather_fake_F_knpy_dir}/{fake_name}.knpy"
            np.save(fake_F_npy_path, fake_F)
            Save_npy_path_as_knpy(fake_F_npy_path, fake_F_knpy_path)
            print("fake_F_npy_path :", fake_F_npy_path)
            print("fake_F_knpy_path:", fake_F_knpy_path)

            ### .npy刪除(因為超占空間) 改存 .npz
            np.savez_compressed(fake_F_npy_path.replace(".npy", ".npz"), fake_F)
            os.remove(fake_F_npy_path)
            ###############################################################################
            fake_W = np.zeros(shape=(h, w, 3), dtype=np.float32)
            fake_W = np.concatenate((fake_W, pred_mask), axis=-1)
            fake_W = fake_W.astype(np.float32)

            ### 定位出 存檔案的位置
            gather_fake_W_dir = public_write_dir + "/pred_mask/fake_W"
            gather_fake_W_npy_dir  = gather_fake_W_dir + "/1 npy"
            gather_fake_W_knpy_dir = gather_fake_W_dir + "/2 knpy"
            Check_dir_exist_and_build(gather_fake_W_dir)
            Check_dir_exist_and_build(gather_fake_W_npy_dir)
            Check_dir_exist_and_build(gather_fake_W_knpy_dir)

            ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
            fake_W_npy_path  = f"{gather_fake_W_npy_dir}/{fake_name}.npy"
            fake_W_knpy_path = f"{gather_fake_W_knpy_dir}/{fake_name}.knpy"
            np.save(fake_W_npy_path, fake_W)
            Save_npy_path_as_knpy(fake_W_npy_path, fake_W_knpy_path)
            print("fake_W_npy_path :", fake_W_npy_path)
            print("fake_W_knpy_path:", fake_W_knpy_path)

            ### .npy刪除(因為超占空間) 改存 .npz
            np.savez_compressed(fake_W_npy_path.replace(".npy", ".npz"), fake_W)
            os.remove(fake_W_npy_path)
