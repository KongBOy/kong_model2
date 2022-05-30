import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import Check_dir_exist_and_build
from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
from step08_b_use_G_generate_0_util import WcM_01_visual_op, Value_Range_Postprocess_to_01

######################################################################################################################################################################################################
def wc_visual_op(wc):
    wc = wc.astype(np.float32)
    wc = (wc - wc.min()) / (wc.max() - wc.min() + 0.000001) * 255.
    wc_2d_v = wc.astype(np.uint8)
    return wc_2d_v

####################################################################################################
def I_Generate_W(model_obj, _1, in_img_pre, _3, Wgt_w_Mgt_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    W_pre = model_obj.generator(in_img_pre, training=training)
    W_pre = W_pre[0, ..., 0:3].numpy()
    W_01 = Value_Range_Postprocess_to_01(W_pre, use_gt_range)

    Wgt_pre = Wgt_w_Mgt_pre[0, ..., 0:3].numpy()
    Wgt_01  = Value_Range_Postprocess_to_01(Wgt_pre, use_gt_range)
    return W_01, Wgt_01

def I_Generate_W_see(model_obj, phase, index, in_img, in_img_pre, _3, Wgt_w_Mgt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
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
    if(bgr2rgb):
        in_img = in_img[:, :, ::-1]
        rec_hope = rec_hope[:, :, ::-1]

    W_01, Wgt_01    = I_Generate_W(model_obj, None, in_img_pre, None, Wgt_w_Mgt_pre, exp_obj.use_gt_range, training=training)

    W_visual,   Wx_visual,   Wy_visual,   Wz_visual   = WcM_01_visual_op(W_01)
    Wgt_visual, Wxgt_visual, Wygt_visual, Wzgt_visual = WcM_01_visual_op(Wgt_01)
    # print("wc.shape:          ", wc.shape)
    # print("wc_visual.shape:   ", wc_visual.shape)
    # print("gt_wc.shape:       ", gt_wc.shape)
    # print("gt_wc_visual.shape:", gt_wc_visual.shape)

    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img(in_img).jpg", in_img)             ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張

        if(npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_wc", Wgt_01)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_wc", Wgt_01)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Wx.jpg",  Wxgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Wy.jpg",  Wygt_visual)
        cv2.imwrite(private_write_dir + "/" + "0b_u1b5-gt_Wz.jpg",  Wzgt_visual)       
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)           ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    if(npz_save is False): np.save            (private_write_dir + "/" + "epoch_%04i_u1b1-W" % current_ep, W_01)
    if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "epoch_%04i_u1b1-W" % current_ep, W_01)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2-W_visual.jpg"  % current_ep, W_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3-Wx_visual.jpg" % current_ep, Wx_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4-Wy_visual.jpg" % current_ep, Wy_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b5-Wz_visual.jpg" % current_ep, Wz_visual)

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs
        imgs = [ in_img ,   W_visual , Wgt_visual]
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
