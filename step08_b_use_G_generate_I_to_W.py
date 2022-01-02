import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt

######################################################################################################################################################################################################
def wc_visual_op(wc):
    wc = wc.astype(np.float32)
    wc = (wc - wc.min()) / (wc.max() - wc.min() + 0.000001) * 255.
    wc_2d_v = wc.astype(np.uint8)
    return wc_2d_v

####################################################################################################
def I_Generate_W(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    wc = model_G(in_img_pre, training=training)
    wc = wc[0].numpy()
    return wc

def I_Generate_W_see(model_G, phase, index, in_img, in_img_pre, gt_wc, _4, rec_hope=None, epoch=0, exp_obj=None, training=True, see_reset_init=True, postprocess=False, add_loss=False, bgr2rgb=True):
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
    if(bgr2rgb):
        in_img = in_img[:, :, ::-1]
        rec_hope = rec_hope[:, :, ::-1]

    wc    = I_Generate_W(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    gt_wc = gt_wc[0, ..., :3].numpy()

    wc_visual    = wc_visual_op(wc)
    gt_wc_visual = wc_visual_op(gt_wc)
    # print("wc.shape:          ", wc.shape)
    # print("wc_visual.shape:   ", wc_visual.shape)
    # print("gt_wc.shape:       ", gt_wc.shape)
    # print("gt_wc_visual.shape:", gt_wc_visual.shape)

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-in_img.jpg", in_img)             ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        np.save    (private_write_dir + "/" + "0b_u1b1-gt_wc",      gt_wc)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_wc.jpg",  gt_wc_visual)                            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope)           ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    np.save(    private_write_dir + "/" + "epoch_%04i_u1b1_wc"            % epoch, wc)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2_wc_visual.jpg" % epoch, wc_visual)                  ### 把 生成的 flow_visual 存進相對應的資料夾


    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        from matplot_fig_ax_util import Matplot_single_row_imgs
        imgs = [ in_img ,   wc_visual , gt_wc_visual]
        img_titles = ["in_img", "Wpred",   "Wgt"]

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =imgs,         ### 把要顯示的每張圖包成list
                                img_titles=img_titles,               ### 把每張圖要顯示的字包成list
                                fig_title ="%s, epoch=%04i" % (current_see_name, int(epoch)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 如果沒有要接續畫loss，就可以存了喔！