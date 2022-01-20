
import numpy as np
import cv2

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
from flow_bm_util import check_flow_quality_then_I_w_F_to_R
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
from step08_b_use_G_generate_0_util import Value_Range_Postprocess_to_01, C_01_concat_with_M_to_F_and_get_F_visual

######################################################################################################################################################################################################
def I_w_Mgt_Gen_Cx_Cy_to_C(model_G, _1, in_img_pre, _3, gt_mask_coord_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    gt_mask_pre  = gt_mask_coord_pre[..., 0:1]
    I_pre_with_M = in_img_pre * gt_mask_pre

    Cx_pre, Cy_pre = model_G(I_pre_with_M, training=training)  ### 沒辦法當初設定成這樣子train， 就只能繼續保持這樣子了，要不然以前train好的東西 不能繼續用下去 QQ
    Cx_pre = Cx_pre[0].numpy()
    Cy_pre = Cy_pre[0].numpy()
    C_pre = np.concatenate([Cy_pre, Cx_pre], axis=-1)
    C = Value_Range_Postprocess_to_01(C_pre, use_gt_range)

    I_w_M_visual = (I_pre_with_M[0].numpy() * 255.).astype(np.uint8)
    return C, I_w_M_visual

def I_w_Mgt_Gen_Cx_Cy_to_C_w_Mgt_to_F_basic_data(model_G, in_img, in_img_pre, gt_mask_coord_pre, rec_hope=None, exp_obj=None, training=True, bgr2rgb=True):
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
                                但 plt 存圖是rgb， 所以存圖不用轉ch， 把 bgr2rgb設False喔！
    '''
    in_img   = in_img[0].numpy()
    coord, I_w_M_visual   = I_w_Mgt_Gen_Cx_Cy_to_C(model_G, None, in_img_pre, None, gt_mask_coord_pre, exp_obj.use_gt_range, training=training)
    gt_mask  = gt_mask_coord_pre[0, ..., 0:1].numpy()
    gt_coord = gt_mask_coord_pre[0, ..., 1:3].numpy()
    Mgt_visual = (gt_mask * 255).astype(np.uint8)
    flow,    flow_visual   , Cx_visual,   Cy_visual   = C_01_concat_with_M_to_F_and_get_F_visual(coord,    gt_mask)
    gt_flow, gt_flow_visual, Cxgt_visual, Cygt_visual = C_01_concat_with_M_to_F_and_get_F_visual(gt_coord, gt_mask)
    flow_visual    = flow_visual   [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
    gt_flow_visual = gt_flow_visual[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
    rec_hope       = rec_hope[0].numpy()
    if(bgr2rgb):
        in_img         = in_img        [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        I_w_M_visual   = I_w_M_visual[:, :, ::-1]
        rec_hope       = rec_hope      [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        flow_visual    = flow_visual   [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        gt_flow_visual = gt_flow_visual[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
    return in_img, I_w_M_visual, flow, flow_visual, Mgt_visual, gt_flow_visual, gt_flow, Cx_visual, Cy_visual, Cxgt_visual, Cygt_visual, rec_hope

def I_w_Mgt_Gen_Cx_Cy_to_C_with_Mgt_to_F_see(model_G, phase, index, in_img, in_img_pre, _3, gt_mask_coord_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "train"): used_sees = exp_obj.result_obj.sees
    elif(phase == "test"):  used_sees = exp_obj.result_obj.tests
    private_write_dir     = used_sees[index].see_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_rec_write_dir = used_sees[index].rec_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    '''
    gt_mask_coord_pre[0] 為 mask  (1, h, w, 1)
    gt_mask_coord_pre[1] 為 coord (1, h, w, 2) 先y 在x

    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr
    in_img, I_w_M_visual, flow, flow_visual, Mgt_visual, gt_flow_visual, gt_flow, Cx_visual, Cy_visual, Cxgt_visual, Cygt_visual, rec_hope = I_w_Mgt_Gen_Cx_Cy_to_C_w_Mgt_to_F_basic_data(model_G, in_img, in_img_pre, gt_mask_coord_pre, rec_hope, exp_obj=exp_obj, training=training, bgr2rgb=bgr2rgb)

    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(private_rec_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img.jpg",      in_img)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a1-gt_mask.jpg",      Mgt_visual)
        cv2.imwrite(private_write_dir + "/" + "0a_u1a2-dis_img_w_Mgt((in_img)).jpg", I_w_M_visual)

        ''' 覺得 u1b 不用寫 mask， 因為 unet1 又沒有 output mask！ '''
        if(npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_b_gt_flow",     gt_flow)                        ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_b_gt_flow",     gt_flow)                        ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_b_gt_flow.jpg", gt_flow_visual)                 ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_b_gt_Cx.jpg",   Cxgt_visual)                    ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_b_gt_Cy.jpg",   Cygt_visual)                    ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",          rec_hope)                       ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    if(npz_save is False): np.save            (private_write_dir + "/" + "epoch_%04i_u1b1_flow"            % current_ep, flow)         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "epoch_%04i_u1b1_flow"            % current_ep, flow)         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2_flow.jpg" % current_ep, flow_visual)  ### 把 生成的 flow_visual 存進相對應的資料夾
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3_Cx.jpg"          % current_ep, Cx_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4_Cy.jpg"          % current_ep, Cy_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾

        bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=in_img, flow=flow)
        '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
        # gt_bm, gt_rec = check_flow_quality_then_I_w_F_to_R(dis_img=in_img, flow=gt_flow)
        cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)
        # print("private_rec_write_dir:", private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep)

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img , Mgt_visual, I_w_M_visual,  flow_visual ,    rec,   rec_hope],  ### 把要顯示的每張圖包成list
                                img_titles=["in_img", "gt_mask",  "I_with_M",  "pred_flow_v", "pred_rec", "rec_hope"],  ### 把每張圖要顯示的字包成list
                                fig_title ="%s, epoch=%04i" % (current_see_name, int(current_ep)),              ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   = bgr2rgb)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)

