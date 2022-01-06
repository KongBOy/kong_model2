import numpy as np
import cv2

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
from flow_bm_util import check_flow_quality_then_I_w_F_to_R
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
from step08_b_use_G_generate_0_util import Value_Range_Postprocess_to_01, C_with_M_to_F_and_get_F_visual

def I_gen_M_w_I_gen_C(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    '''
    這邊model 生成的是 ch2 的 coord， 要再跟 mask concate 後才會變成 ch3 的 flow 喔！
    M        0~1
    M_pre    0~1
    M_visual 0~255

    I_pre    0~1 或 -1~1 看 use_in_range
    I        0~255
    I_visual 0~255

    C_pre    0~1 或 -1~1 看 use_gt_range
    C        0~1
    C_visual 寫在外面要用 method1，寫在外面是想跟 gt_F 一起寫看起來漂亮且好改
    '''
    M_pre, C_pre = model_G(in_img_pre, training=training)
    M_pre = M_pre[0].numpy()
    M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
    M_visual = (M * 255).astype(np.uint8)

    I_pre_with_M_pre = in_img_pre * M_pre
    I_with_M_visual = (I_pre_with_M_pre[0].numpy() * 255.).astype(np.uint8)

    C_pre = C_pre[0].numpy()
    C = Value_Range_Postprocess_to_01(C_pre, use_gt_range)

    return M, M_visual, I_pre_with_M_pre, I_with_M_visual, C

def I_gen_M_w_I_gen_C_w_M_to_F_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, rec_hope=None, exp_obj=None, training=True, bgr2rgb=True):
    in_img    = in_img[0].numpy()
    Mgt  = gt_mask_coord[0, ..., 0:1]
    Cgt  = gt_mask_coord[0, ..., 1:3].numpy()
    Cxgt_visual = (Cgt[..., 1:2] * 255).astype(np.uint8)
    Cygt_visual = (Cgt[..., 0:1] * 255).astype(np.uint8)
    M, M_visual, I_pre_with_M_pre, I_with_M_visual, C = I_gen_M_w_I_gen_C(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    Cx_visual = (C[..., 1:2] * 255).astype(np.uint8)
    Cy_visual = (C[..., 0:1] * 255).astype(np.uint8)
    Mgt_visual = (Mgt.numpy() * 255).astype(np.uint8)

    F,   F_visual   = C_with_M_to_F_and_get_F_visual(C  , M  )
    Fgt, Fgt_visual = C_with_M_to_F_and_get_F_visual(Cgt, Mgt)
    F_visual    = F_visual   [:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
    Fgt_visual  = Fgt_visual[:, :, ::-1]   ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

    rec_hope = rec_hope[0].numpy()
    if(bgr2rgb):
        in_img   = in_img  [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        I_with_M_visual = I_with_M_visual[:, :, ::-1]
        rec_hope = rec_hope[:, :, ::-1]
        F_visual    = F_visual   [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        Fgt_visual = Fgt_visual[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
    return in_img, M_visual, Mgt_visual, I_with_M_visual, F, F_visual, Fgt, Fgt_visual, Cx_visual, Cy_visual, Cxgt_visual, Cygt_visual, rec_hope

def I_gen_M_w_I_gen_C_w_M_to_F_see(model_G, phase, index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "see"):  used_sees = exp_obj.result_obj.sees
    elif(phase == "test"): used_sees = exp_obj.result_obj.tests
    private_write_dir     = used_sees[index].see_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_rec_write_dir = used_sees[index].rec_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾

    in_img, M_visual, Mgt_visual, I_with_M_visual, F, F_visual, Fgt, Fgt_visual, Cx_visual, Cy_visual, Cxgt_visual, Cygt_visual, rec_hope = I_gen_M_w_I_gen_C_w_M_to_F_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, rec_hope=rec_hope, exp_obj=exp_obj, training=training, bgr2rgb=bgr2rgb)
    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)    ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(private_rec_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(f"{private_write_dir}/0a_u1a-in_img.jpg",  in_img)

        cv2.imwrite(f"{private_write_dir}/0b_u1b-gt_mask.jpg", Mgt_visual)
        cv2.imwrite(f"{private_write_dir}/0b_u2b-gt_Cx.jpg",   Cxgt_visual)
        cv2.imwrite(f"{private_write_dir}/0b_u2b-gt_Cy.jpg",   Cygt_visual)
        np .save   (f"{private_write_dir}/0b_u2b-gt_flow.npy", Fgt)
        cv2.imwrite(f"{private_write_dir}/0b_u2b-gt_flow.jpg", Fgt_visual)
        cv2.imwrite(f"{private_write_dir}/0c-rec_hope.jpg",    rec_hope)

    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b-mask.jpg"  % current_ep, M_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2a-I_w_M.jpg" % current_ep, I_with_M_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b-Cx.jpg"    % current_ep, Cx_visual)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b-Cy.jpg"    % current_ep, Cy_visual)
    np .save   (private_write_dir + "/" + "epoch_%04i_u2b-flow.npy"  % current_ep, F)
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u2b-flow.jpg"  % current_ep, F_visual)

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=in_img, flow=F)
        '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
        # gt_bm, gt_rec = check_flow_quality_then_I_w_F_to_R(dis_img=in_img, flow=gt_flow)  ### 因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！
        cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img , M_visual,  Mgt_visual, I_with_M_visual , F_visual ,   Fgt_visual,     rec,       rec_hope ],    ### 把要顯示的每張圖包成list
                                img_titles=["in_img",    "Mask", "gt_Mask",  "I_with_M",     "pred_flow_v",  "gt_flow_v", "pred_rec", "rec_hope"],    ### 把每張圖要顯示的字包成list
                                fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)
