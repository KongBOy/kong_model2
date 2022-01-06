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

####################################################################################################
def I_w_M_Gen_Wx_Wy_Wz_to_W(model_G, _1, in_img_pre, _3, Wgt_w_Mgt_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    Mgt_pre = Wgt_w_Mgt_pre[..., 3:4]
    Wgt_pre = Wgt_w_Mgt_pre[..., 0:3]
    I_pre_with_M_pre = in_img_pre * Mgt_pre

    Wz_pre, Wy_pre, Wx_pre = model_G(I_pre_with_M_pre, training=training)

    ### 後處理： 拿掉 batch 和 弄成01 和 轉成 numpy
    Wz_pre = Wz_pre[0].numpy()
    Wy_pre = Wy_pre[0].numpy()
    Wx_pre = Wx_pre[0].numpy()
    W_pre  = np.concatenate([Wz_pre, Wy_pre, Wx_pre], axis=-1)
    W_01 = Value_Range_Postprocess_to_01(W_pre, use_gt_range)

    Wgt_pre = Wgt_pre[0].numpy()
    Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, use_gt_range)

    Mgt_pre = Mgt_pre[0].numpy()

    I_w_M_01  = Value_Range_Postprocess_to_01(I_pre_with_M_pre, use_gt_range)
    I_w_M_01 = I_w_M_01[0].numpy()
    return W_01, I_w_M_01, Wgt_01, Mgt_pre

def I_w_M_Gen_Wx_Wy_Wz_to_W_see(model_G, phase, index, in_img, in_img_pre, _3, Wgt_w_Mgt_pre, rec_hope=None, current_ep=0, exp_obj=None, training=True, see_reset_init=True, postprocess=False, add_loss=False, bgr2rgb=True):
    if  (phase == "see"):  used_sees = exp_obj.result_obj.sees
    elif(phase == "test"): used_sees = exp_obj.result_obj.tests
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

    W_01, I_w_M_01, Wgt_01, Mgt_pre = I_w_M_Gen_Wx_Wy_Wz_to_W(model_G, None, in_img_pre, None, Wgt_w_Mgt_pre, exp_obj.use_gt_range, training=training)


    W_visual,   Wx_visual,   Wy_visual,   Wz_visual   = W_01_visual_op(W_01)
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
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_img.jpg",      in_img)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a2-gt_mask.jpg",      Mgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a3-in_img_w_Mgt.jpg", I_w_M_visual)

        np.save    (private_write_dir + "/" + "0b_u1b1-gt_W",             Wgt_01)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",         Wgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg",         Wxgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg",         Wygt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg",         Wzgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",           rec_hope)
    np.save(    private_write_dir + "/" + "epoch_%04i_u1b1-W"             % current_ep, W_01)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2-W_visual.jpg"  % current_ep, W_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3-Wx_visual.jpg" % current_ep, Wx_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4-Wy_visual.jpg" % current_ep, Wy_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b5-Wz_visual.jpg" % current_ep, Wz_visual)


    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        imgs       = [ in_img ,   W_visual , Wgt_visual]
        img_titles = ["in_img", "Wpred",   "Wgt"]

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =imgs,         ### 把要顯示的每張圖包成list
                                img_titles=img_titles,               ### 把每張圖要顯示的字包成list
                                fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)

        ### W_01 back to W then + M
        gt_min = exp_obj.db_obj.db_gt_range.min
        gt_max = exp_obj.db_obj.db_gt_range.min
        W = W_01 * (gt_max - gt_min) + gt_min
        WM = np.concatenate([W, Mgt_pre], axis=-1)

        gather_WM_npy_dir  = f"{public_write_dir}/pred_WM_{phase}/WM_npy"
        gather_WM_knpy_dir = f"{public_write_dir}/pred_WM_{phase}/WM_knpy"
        Check_dir_exist_and_build(gather_WM_npy_dir)
        Check_dir_exist_and_build(gather_WM_knpy_dir)

        WM_npy_path  = f"{gather_WM_npy_dir}/{current_see_name}_pred.npy"
        WM_knpy_path = f"{gather_WM_knpy_dir}/{current_see_name}_pred.knpy"
        np.save(WM_npy_path, WM)
        Save_npy_path_as_knpy(WM_npy_path, WM_knpy_path)
        # breakpoint()
