import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys

from step08_b_use_G_generate_0_util import Value_Range_Postprocess_to_01, W_01_visual_op, C_with_M_to_F_and_get_F_visual
from flow_bm_util import check_flow_quality_then_I_w_F_to_R

sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
import datetime
import pdb

####################################################################################################
def use_model(model_G, in_WM_pre, training):
    W_pre   = in_WM_pre[..., 0:3]
    Mgt_pre = in_WM_pre[..., 3:4]
    W_pre_W_M_pre = W_pre * Mgt_pre
    Cx_pre, Cy_pre = model_G(W_pre_W_M_pre, training=training)

    return W_pre, Mgt_pre, W_pre_W_M_pre, Cx_pre, Cy_pre

def W_w_M_Gen_Cx_Cy_see(model_G, phase, index, in_WM, in_WM_pre, Fgt, Fgt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "see"):  used_sees = exp_obj.result_obj.sees
    elif(phase == "test"): used_sees = exp_obj.result_obj.tests
    private_write_dir    = used_sees[index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_rec_write_dir = used_sees[index].rec_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    # print("private_rec_write_dir:", private_rec_write_dir)
    '''
    in_WM_pre[..., 3:4] 為 M (1, h, w, 1)
    in_WM_pre[..., 0:3] 為 W (1, h, w, 3) 先z 再y 再x

    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    # plt.imshow(in_img[0])
    # plt.show()
    rec_hope  = rec_hope[0].numpy()

    W_pre, Mgt_pre, W_pre_W_M_pre, Cx_pre, Cy_pre = use_model(model_G, in_WM_pre, training)

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
    C_pre = np.concatenate([Cy_pre, Cx_pre], axis=-1)  ### tensor 會自動轉 numpy
    C = Value_Range_Postprocess_to_01(C_pre, exp_obj.use_gt_range)
    C = C[0]
    Cgt = Fgt[0, ..., 1:3].numpy()
    Cx_visual = (Cx_pre[0].numpy() * 255).astype(np.uint8)
    Cy_visual = (Cy_pre[0].numpy() * 255).astype(np.uint8)
    Cxgt_visual = (Cgt[..., 1:2] * 255).astype(np.uint8)
    Cygt_visual = (Cgt[..., 0:1] * 255).astype(np.uint8)

    Mgt = Fgt[0, ..., 0:1].numpy()
    F,   F_visual   = C_with_M_to_F_and_get_F_visual(C, Mgt)
    Fgt, Fgt_visual = C_with_M_to_F_and_get_F_visual(Cgt, Mgt)
    F_visual   = F_visual  [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
    Fgt_visual = Fgt_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！


    ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
    if(bgr2rgb):
        rec_hope   = rec_hope  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        F_visual   = F_visual  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        Fgt_visual = Fgt_visual[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch

    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_W_01.jpg",         W_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_Wx_01.jpg",        Wx_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_Wy_01.jpg",        Wy_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-ord_Wz_01.jpg",        Wz_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a2-gt_mask.jpg",          Mgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a3-W_w_Mgt(in_img).jpg",  W_w_M_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wx_w_Mgt(in_img).jpg", Wx_w_M_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wy_w_Mgt(in_img).jpg", Wy_w_M_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a3-Wz_w_Mgt(in_img).jpg", Wz_w_M_visual)

        np.save    (private_write_dir + "/" + "0b_u1b1-gt_b_gt_flow",     Fgt)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_b_gt_flow.jpg", Fgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_b_gt_Cx.jpg",   Cxgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_b_gt_Cy.jpg",   Cygt_visual)
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)
    np.save(    private_write_dir + "/" + "epoch_%04i_u1b1_flow"     % current_ep, F)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2_flow.jpg" % current_ep, F_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3_Cx.jpg"   % current_ep, Cx_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4_Cy.jpg"   % current_ep, Cy_visual)


    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        dis_img = in_WM[0][0].numpy()  ### [0]第一個是 取 dis_img， 第二個[0]是取 batch
        bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=dis_img, flow=F)
        '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
        cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ W_visual ,   Mgt_visual , W_w_M_visual,  F_visual ,    rec,   rec_hope],    ### 把要顯示的每張圖包成list
                                img_titles=["W_01",        "Mgt",        "W_w_M",     "pred_F", "pred_rec", "rec_hope"], ### 把每張圖要顯示的字包成list
                                fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)
