import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys

from step08_b_use_G_generate_0_util import Value_Range_Postprocess_to_01, W_01_visual_op
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
import datetime
import pdb
import os

####################################################################################################
def use_model(model_obj, _1, in_img_pre, _3, Wgt_w_Mgt_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    Mgt_pre = Wgt_w_Mgt_pre[..., 3:4]
    Wgt_pre = Wgt_w_Mgt_pre[..., 0:3]
    I_pre_with_M_pre = in_img_pre * Mgt_pre

    Wz_raw_pre, Wy_raw_pre, Wx_raw_pre = model_obj.generator(I_pre_with_M_pre, training=training)

    ### 後處理： 拿掉 batch 和 弄成01 和 轉成 numpy
    Wz_raw_pre = Wz_raw_pre[0].numpy()
    Wy_raw_pre = Wy_raw_pre[0].numpy()
    Wx_raw_pre = Wx_raw_pre[0].numpy()
    W_raw_pre  = np.concatenate([Wz_raw_pre, Wy_raw_pre, Wx_raw_pre], axis=-1)
    W_raw_01 = Value_Range_Postprocess_to_01(W_raw_pre, use_gt_range)

    Wgt_pre = Wgt_pre[0].numpy()
    Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, use_gt_range)

    Mgt_pre = Mgt_pre[0].numpy()

    I_w_M_01  = Value_Range_Postprocess_to_01(I_pre_with_M_pre, use_gt_range)
    I_w_M_01 = I_w_M_01[0].numpy()
    return W_raw_01, I_w_M_01, Wgt_01, Mgt_pre

def I_w_M_Gen_Wx_Wy_Wz_focus_to_W_see(model_obj, phase, index, in_img, in_img_pre, _3, Wgt_w_Mgt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "train"): used_sees = exp_obj.result_obj.sees
    elif(phase == "test"):  used_sees = exp_obj.result_obj.tests
    private_write_dir    = used_sees[index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x

    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    # plt.imshow(in_img[0])
    # plt.show()
    in_img   = in_img  [0].numpy()
    rec_hope = rec_hope[0].numpy()

    W_raw_01, I_w_M_01, Wgt_01, Mgt_pre = use_model(model_obj, None, in_img_pre, None, Wgt_w_Mgt_pre, exp_obj.use_gt_range, training=training)
    W_w_Mgt_01 = W_raw_01 * Mgt_pre


    W_raw_visual,   Wx_raw_visual,   Wy_raw_visual,   Wz_raw_visual   = W_01_visual_op(W_raw_01)
    W_w_Mgt_visual, Wx_w_Mgt_visual, Wy_w_Mgt_visual, Wz_w_Mgt_visual = W_01_visual_op(W_w_Mgt_01)
    Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = W_01_visual_op(Wgt_01)
    # print("Wgt_visual", Wgt_visual.max())
    # print("Wgt_visual", Wgt_visual.min())
    Mgt_visual = (Mgt_pre * 255).astype(np.uint8)
    I_w_M_visual = (I_w_M_01 * 255).astype(np.uint8)
    ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
    if(bgr2rgb):
        in_img = in_img[:, :, ::-1]
        rec_hope = rec_hope[:, :, ::-1]
        I_w_M_visual = I_w_M_visual[:, :, ::-1]
    # print("W_01.shape:          ", W_01.shape)
    # print("W_visual.shape:   ", W_visual.shape)
    # print("Wgt.shape:       ", Wgt.shape)
    # print("Wgt_visual.shape:", Wgt_visual.shape)

    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      in_img)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt(in_img).jpg", I_w_M_visual)

        if(npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_W", Wgt_01)
        if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_W", Wgt_01)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",  Wgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg", Wxgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg", Wygt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg", Wzgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)
    if(npz_save is False): np.save            (private_write_dir + "/" + "epoch_%04i_u1b1-W_w_Mgt" % current_ep, W_w_Mgt_01)
    if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "epoch_%04i_u1b1-W_w_Mgt" % current_ep, W_w_Mgt_01)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2-W_raw_visual.jpg"    % current_ep, W_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3-W_w_Mgt_visual.jpg"  % current_ep, W_w_Mgt_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4-Wx_raw_visual.jpg"   % current_ep, Wx_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b5-Wx_w_Mgt_visual.jpg" % current_ep, Wx_w_Mgt_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b6-Wy_raw_visual.jpg"   % current_ep, Wy_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b7-Wy_w_Mgt_visual.jpg" % current_ep, Wy_w_Mgt_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b8-Wz_raw_visual.jpg"   % current_ep, Wz_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b9-Wz_w_Mgt_visual.jpg" % current_ep, Wz_w_Mgt_visual)

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        imgs       = [ in_img ,  Mgt_visual, I_w_M_visual,  W_raw_visual, W_w_Mgt_visual,  Wgt_visual]
        img_titles = ["in_img", "Mgt",       "I_with_M",    "W_raw",      "W_w_Mgt",       "Wgt"]

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =imgs,         ### 把要顯示的每張圖包成list
                                img_titles=img_titles,               ### 把每張圖要顯示的字包成list
                                fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)

        if(phase == "test"):
            ### W_01 back to W then + M
            gt_min = exp_obj.db_obj.db_gt_range.min
            gt_max = exp_obj.db_obj.db_gt_range.max
            W = W_w_Mgt_01 * (gt_max - gt_min) + gt_min
            if(exp_obj.db_obj.get_method.value == "in_dis_gt_wc_try_mul_M"): W = W * Mgt_pre
            WM = np.concatenate([W, Mgt_pre], axis=-1)
            ### 確認寫得對不對
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(W_01)
            # ax[1].imshow(W - gt_min)
            # print(W.max())
            # print(W.min())
            # plt.show()

            ### 定位出 存檔案的位置
            gather_WM_npy_dir  = f"{public_write_dir}/pred_WM_{phase}-{current_time}/WM_npy_then_npz"
            gather_WM_knpy_dir = f"{public_write_dir}/pred_WM_{phase}-{current_time}/WM_knpy"
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
