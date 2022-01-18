import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys

from step08_b_use_G_generate_0_util import Value_Range_Postprocess_to_01, W_01_visual_op, W_01_and_W_01_w_M_to_WM_and_visualize, C_01_and_C_01_w_M_to_F_and_visualize, C_01_concat_with_M_to_F_and_get_F_visual
from flow_bm_util import check_flow_quality_then_I_w_F_to_R

sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
import datetime
import pdb

####################################################################################################
# def use_model(model_G, in_dis, in_dis_pre, gt_WM_F, gt_WM_F_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～

#     return W_raw_01, I_w_M_01, Wgt_01, Mgt_pre

def I_w_M_Gen_Wx_Wy_Wz_focus_Gen_Cx_Cy_focus_to_F_see(model_G, phase, index, in_dis, in_dis_pre, gt_WM_F, gt_WM_F_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "see"):  used_sees = exp_obj.result_obj.sees
    elif(phase == "test"): used_sees = exp_obj.result_obj.tests
    private_write_dir    = used_sees[index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x

    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    ### 這個是給後處理用的 dis_img
    '''use model'''
    I_pre     = in_dis_pre[0]
    Mgt       = gt_WM_F[0][..., 3:4]  ### [0]第一個是 取 wc, [1] 是取 flow, 第二個[0]是取 batch， 這次試試看用 M 不用 M_pre
    I_pre_w_M = I_pre * Mgt  ### 這次試試看用 M 不用 M_pre

    W_pre_raw, W_pre_w_M, C_pre_raw, C_pre_w_M = model_G(I_pre_w_M, Mask=Mgt, training=training)
    ''''''
    ### 後處理： 拿掉 batch 和 弄成01 和 轉成 numpy
    # W_01_w_M = Value_Range_Postprocess_to_01(W_pre_w_M, exp_obj.use_gt_range)
    # W_01_w_M = W_01_w_M[0].numpy()
    # W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual  = W_01_visual_op(W_01_w_M)

    # W_01_raw = Value_Range_Postprocess_to_01(W_pre_raw, exp_obj.use_gt_range)
    # W_01_raw = W_01_raw.numpy()
    # Wx_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual = W_01_visual_op(W_01_raw)

    # ### C_w_M
    # C_01_w_M = Value_Range_Postprocess_to_01(C_pre_w_M, exp_obj.use_gt_range)
    # C_01_w_M = C_01_w_M.numpy()


    # ### C_raw
    # C_01_raw = Value_Range_Postprocess_to_01(C_pre_raw, exp_obj.use_gt_range)
    # C_01_raw = C_01_raw.numpy()

    '''model in visualize'''
    I_01_w_M  = Value_Range_Postprocess_to_01(I_pre_w_M, exp_obj.use_gt_range)
    I_01_w_M = I_01_w_M[0].numpy()
    I_w_M_visual = (I_01_w_M * 255).astype(np.uint8)

    Mgt = Mgt[0].numpy()
    Mgt_visual = (Mgt * 255).astype(np.uint8)

    '''model out visualize'''
    W_01_raw = Value_Range_Postprocess_to_01(W_pre_raw, exp_obj.use_gt_range)
    W_01_raw = W_01_raw[0].numpy()
    W_raw_c_M, W_raw_visual, Wx_raw_visual, Wy_raw_visual, Wz_raw_visual, W_w_M_c_M, W_w_M_visual, Wx_w_M_visual, Wy_w_M_visual, Wz_w_M_visual = W_01_and_W_01_w_M_to_WM_and_visualize(W_01_raw, Mgt)

    C_01_raw = Value_Range_Postprocess_to_01(C_pre_raw, exp_obj.use_gt_range)
    C_01_raw = C_01_raw[0].numpy()
    F_raw, F_raw_visual, Cx_raw_visual, Cy_raw_visual, F_w_Mgt,   F_w_Mgt_visual,   Cx_w_Mgt_visual,   Cy_w_Mgt_visual   = C_01_and_C_01_w_M_to_F_and_visualize(C_01_raw, Mgt)

    '''model gt visualize'''
    Wgt_pre = gt_WM_F_pre[0][0].numpy()
    Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, exp_obj.use_gt_range)
    Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = W_01_visual_op(Wgt_01)

    Cgt_pre = gt_WM_F_pre[1][0, ..., 1:3].numpy()
    Cgt_01  = Value_Range_Postprocess_to_01(Cgt_pre, exp_obj.use_gt_range)
    Fgt, Fgt_visual, Cxgt_visual, Cygt_visual = C_01_concat_with_M_to_F_and_get_F_visual(Cgt_01, Mgt)

    ''' model postprocess輔助 visualize'''
    dis_img  = in_dis[0].numpy()
    rec_hope = rec_hope[0].numpy()

    '''cv2 bgr 與 rgb 的調整'''
    ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
    if(bgr2rgb):
        dis_img = dis_img[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
        rec_hope = rec_hope[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
        I_w_M_visual = I_w_M_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      dis_img)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt(in_img).jpg", I_w_M_visual)

        np.save    (private_write_dir + "/" + "0b_u1b1-gt_W",      Wgt_01)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_W.jpg",  Wgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg", Wxgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg", Wygt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg", Wzgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)

        np.save    (private_write_dir + "/" + "0b_u2b1-gt_b_Fgt",     Fgt)
        cv2.imwrite(private_write_dir + "/" + "0b_u2b2-gt_b_Fgt.jpg", Fgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u2b3-gt_b_Cxgt.jpg",   Cxgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u2b4-gt_b_Cygt.jpg",   Cygt_visual)
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)

    np.save(    private_write_dir + "/" + "epoch_%04i_u1b1-W_w_Mgt"             % current_ep, W_w_M_c_M)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2-W_raw_visual.jpg"    % current_ep, W_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3-W_w_M_visual.jpg"    % current_ep, W_w_M_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4-Wx_raw_visual.jpg"   % current_ep, Wx_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b5-Wx_w_M_visual.jpg"   % current_ep, Wx_w_M_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b6-Wy_raw_visual.jpg"   % current_ep, Wy_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b7-Wy_w_M_visual.jpg"   % current_ep, Wy_w_M_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b8-Wz_raw_visual.jpg"   % current_ep, Wz_raw_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b9-Wz_w_M_visual.jpg"   % current_ep, Wz_w_M_visual)

    np.save(    private_write_dir + "/" + "epoch_%04i_u2b1_F_w_Mgt"      % current_ep, F_w_Mgt)          ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b2_F_raw.jpg"    % current_ep, F_raw_visual)     ### 把 生成的 F_visual 存進相對應的資料夾
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b3_F_w_Mgt.jpg"  % current_ep, F_w_Mgt_visual)   ### 把 生成的 F_visual 存進相對應的資料夾
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b4_Cx_raw.jpg"   % current_ep, Cx_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b5_Cx_w_Mgt.jpg" % current_ep, Cx_w_Mgt_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b6_Cy_raw.jpg"   % current_ep, Cy_raw_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b7_Cy_w_Mgt.jpg" % current_ep, Cy_w_Mgt_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

    # if(postprocess):
    #     current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
    #     bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=dis_img, flow=F_w_Mgt)

    #     imgs       = [ dis_img ,  Mgt_visual, I_w_M_visual,  W_raw_visual, W_w_M_visual,  Wgt_visual,  F_raw_visual, F_w_Mgt_visual,    rec,      rec_hope]
    #     img_titles = ["dis_img", "Mgt",       "I_with_M",    "W_raw",      "W_w_M",       "Wgt",   "F_raw_visual", "F_w_Mgt_visual",     "pred_rec", "rec_hope" ]

    #     single_row_imgs = Matplot_single_row_imgs(
    #                             imgs      =imgs,         ### 把要顯示的每張圖包成list
    #                             img_titles=img_titles,               ### 把每張圖要顯示的字包成list
    #                             fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
    #                             add_loss  =add_loss,
    #                             bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
    #     single_row_imgs.Draw_img()
    #     single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
    #     print("save to:", exp_obj.result_obj.test_write_dir)

    #     ### W_01 back to W then + M
    #     gt_min = exp_obj.db_obj.db_gt_range.min
    #     gt_max = exp_obj.db_obj.db_gt_range.max
    #     W_w_M_c_M[..., 0:3] = W_w_M_c_M[..., 0:3] * (gt_max - gt_min) + gt_min
    #     if(exp_obj.db_obj.get_method.value == "in_dis_gt_wc_try_mul_M"): W_w_M_c_M[..., 0:3] = W_w_M_c_M[..., 0:3] * Mgt

    #     gather_WM_npy_dir  = f"{public_write_dir}/pred_WM_{phase}-{current_time}/WM_npy"
    #     gather_WM_knpy_dir = f"{public_write_dir}/pred_WM_{phase}-{current_time}/WM_knpy"
    #     Check_dir_exist_and_build(gather_WM_npy_dir)
    #     Check_dir_exist_and_build(gather_WM_knpy_dir)

    #     WM_npy_path  = f"{gather_WM_npy_dir}/{current_see_name}_pred.npy"
    #     WM_knpy_path = f"{gather_WM_knpy_dir}/{current_see_name}_pred.knpy"
    #     np.save(WM_npy_path, W_w_M_c_M)
    #     Save_npy_path_as_knpy(WM_npy_path, WM_knpy_path)
    #     # breakpoint()
