import numpy as np
import cv2

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
from step08_b_use_G_generate_0_util import Value_Range_Postprocess_to_01
######################################################################################################################################################################################################
def I_w_Mgt_to_Cy(model_G, _1, in_img_pre, _3, gt_mask_coord_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    '''
    這邊model 生成的是 ch2 的 coord， 要再跟 mask concate 後才會變成 ch3 的 flow 喔！
    '''
    gt_mask_pre  = gt_mask_coord_pre[..., 0:1]
    I_pre_w_M = in_img_pre * gt_mask_pre

    cy_pre      = model_G(I_pre_w_M, training=training)
    cy_pre = cy_pre[0].numpy()
    cy = Value_Range_Postprocess_to_01(cy_pre, use_gt_range)

    I_w_M_visual = (I_pre_w_M[0].numpy() * 255.).astype(np.uint8)
    return cy, I_w_M_visual

def I_w_Mgt_to_Cy_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, gt_mask_coord_pre, rec_hope=None, exp_obj=None, training=True, bgr2rgb=True):
    in_img    = in_img[0].numpy()
    cy, I_w_M_visual = I_w_Mgt_to_Cy(model_G, None, in_img_pre, None, gt_mask_coord_pre, exp_obj.use_gt_range, training=training)
    gt_mask  = gt_mask_coord[0, ..., 0:1]
    gt_cy    = gt_mask_coord[0, ..., 1:2]

    gt_mask_visual = (gt_mask.numpy() * 255).astype(np.uint8)
    gt_cy_visual   = (gt_cy.numpy() * 255).astype(np.uint8)
    Cy_visual      = (cy * 255).astype(np.uint8)

    rec_hope = rec_hope[0].numpy()
    if(bgr2rgb):
        in_img   = in_img  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        I_w_M_visual = I_w_M_visual[:, :, ::-1]
        rec_hope = rec_hope[:, :, ::-1]

    return in_img, I_w_M_visual, gt_mask_visual, Cy_visual, gt_cy_visual, rec_hope

def I_w_Mgt_to_Cy_see(model_G, phase, index, in_img, in_img_pre, gt_mask_coord, gt_mask_coord_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "see"):  used_sees = exp_obj.result_obj.sees
    elif(phase == "test"): used_sees = exp_obj.result_obj.tests
    private_write_dir     = used_sees[index].see_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    in_img, I_w_M_visual, gt_mask_visual, Cy_visual, gt_cy_visual, rec_hope = I_w_Mgt_to_Cy_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, gt_mask_coord_pre, rec_hope=rec_hope, exp_obj=exp_obj, training=training, bgr2rgb=bgr2rgb)
    if(current_ep == 0 or see_reset_init):          ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      in_img)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      gt_mask_visual)               ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt(in_img).jpg", I_w_M_visual)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張

        ''' 覺得 u1b 不用寫 mask， 因為 unet1 又沒有 output mask！ '''
        cv2.imwrite(private_write_dir + "/" + "0b_u1b1-gt_a_gt_Cy.jpg",    gt_cy_visual)                 ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)                     ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b1-Cy.jpg"   % current_ep, Cy_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        in_img, I_w_M_visual, gt_mask_visual, Cy_visual, gt_cy_visual, rec_hope = I_w_Mgt_to_Cy_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, gt_mask_coord_pre, rec_hope=rec_hope, exp_obj=exp_obj, training=training, bgr2rgb=bgr2rgb)

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img , gt_mask_visual, I_w_M_visual , gt_cy_visual ,    Cy_visual,   ],    ### 把要顯示的每張圖包成list
                                img_titles=["in_img",    "gt_mask",  "I_w_M",          "gt_cy",        "pred_cy"],    ### 把每張圖要顯示的字包成list
                                fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)
