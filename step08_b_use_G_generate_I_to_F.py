import numpy as np
import cv2

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
from flow_bm_util import check_flow_quality_then_I_w_F_to_R
from matplot_fig_ax_util import Matplot_single_row_imgs

import matplotlib.pyplot as plt
from step08_b_use_G_generate_0_util import F_01_or_C_01_method1_visual_op, Value_Range_Postprocess_to_01

######################################################################################################################################################################################################
def I_Generate_F(model_obj, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    flow_pre = model_obj.generator(in_img_pre, training=training)
    flow_pre = flow_pre[0].numpy()
    flow = Value_Range_Postprocess_to_01(flow_pre, use_gt_range)
    return flow

def I_Gen_F_basic_data(model_obj, in_img, in_img_pre, gt_flow, rec_hope, exp_obj=None, training=True, bgr2rgb=True):
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
                                但 plt 存圖是rgb， 所以存圖不用轉ch， 把 bgr2rgb設False喔！
    '''
    in_img         = in_img[0].numpy()   ### HWC 和 tensor -> numpy
    flow           = I_Generate_F(model_obj, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    Cx_visual      = (flow[..., 2:3] * 255).astype(np.uint8)
    Cy_visual      = (flow[..., 1:2] * 255).astype(np.uint8)
    M_visual       = (flow[..., 0:1] * 255).astype(np.uint8)
    gt_flow        = gt_flow[0].numpy()   ### HWC 和 tensor -> numpy
    Cxgt_visual    = (gt_flow[..., 2:3] * 255).astype(np.uint8)
    Cygt_visual    = (gt_flow[..., 1:2] * 255).astype(np.uint8)
    Mgt_visual     = (gt_flow[..., 0:1] * 255).astype(np.uint8)
    rec_hope       = rec_hope[0].numpy()

    flow_visual    = F_01_or_C_01_method1_visual_op(flow)[:, :, ::-1]     ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！
    gt_flow_visual = F_01_or_C_01_method1_visual_op(gt_flow)[:, :, ::-1]  ### cv2 處理完 是 bgr， 但這裡都是用 tf2 rgb的角度來處理， 所以就模擬一下 轉乘 tf2 的rgb囉！

    if(bgr2rgb):
        in_img         = in_img        [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        rec_hope       = rec_hope      [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        flow_visual    = flow_visual   [:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
        gt_flow_visual = gt_flow_visual[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
    return in_img, flow, gt_flow, rec_hope, flow_visual, M_visual, Cx_visual, Cy_visual, gt_flow_visual, Mgt_visual, Cxgt_visual, Cygt_visual

def I_Generate_F_see(model_obj, phase, index, in_img, in_img_pre, gt_flow, _4, rec_hope, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "train"): used_sees = exp_obj.result_obj.sees
    elif(phase == "test"):  used_sees = exp_obj.result_obj.tests
    private_write_dir     = used_sees[index].see_write_dir          ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_rec_write_dir = used_sees[index].rec_visual_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    public_write_dir     = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    in_img, flow, gt_flow, rec_hope, flow_visual, M_visual, Cx_visual, Cy_visual, gt_flow_visual, Mgt_visual, Cxgt_visual, Cygt_visual = I_Gen_F_basic_data(model_obj, in_img, in_img_pre, gt_flow, rec_hope, exp_obj=exp_obj, training=training, bgr2rgb=bgr2rgb)

    if(current_ep == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)   ### 建立 see資料夾
        Check_dir_exist_and_build(private_rec_write_dir)   ### 建立 see資料夾
        cv2.imwrite(private_write_dir + "/" + "0a_u1a0-dis_img(in_img).jpg",  in_img)                    ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張

        if(npz_save is False): np.save            (private_write_dir + "/" + "0b_u1b1-gt_flow", gt_flow)                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "0b_u1b1-gt_flow", gt_flow)                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b2-gt_flow.jpg", gt_flow_visual)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b3-gt_Cx.jpg",   Cxgt_visual)                    ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0b_u1b4-gt_Cy.jpg",   Cygt_visual)                    ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(private_write_dir + "/" + "0c-rec_hope.jpg",     rec_hope)                  ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    if(npz_save is False): np.save            (private_write_dir + "/" + "epoch_%04i_u1b1_flow" % current_ep, flow)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    if(npz_save is True ): np.savez_compressed(private_write_dir + "/" + "epoch_%04i_u1b1_flow" % current_ep, flow)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b2_flow.jpg" % current_ep, flow_visual)                  ### 把 生成的 flow_visual 存進相對應的資料夾
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b3_Cx.jpg"   % current_ep, Cx_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(private_write_dir + "/" + "epoch_%04i_u1b4_Cy.jpg"   % current_ep, Cy_visual)    ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

    ### matplot_visual的部分，記得因為用 matplot 所以要 bgr轉rgb，但是因為有用matplot_visual_single_row_imgs，裡面會bgr轉rgb了，所以這裡不用轉囉！
    ### 這部分要記得做！在 train_step3 的 exp_obj.result_obj.Draw_loss_during_train(epoch, self.epochs) 才有畫布可以畫loss！
    ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了
    # exp_obj.result_obj.sees[see_index].save_as_matplot_visual_during_train(current_ep, bgr2rgb=True)

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        bm, rec       = check_flow_quality_then_I_w_F_to_R(dis_img=in_img, flow=flow)
        '''gt不能做bm_rec，因為 real_photo 沒有 C！ 所以雖然用 test_blender可以跑， 但 test_real_photo 會卡住， 因為 C 全黑！'''
        # gt_bm, gt_rec = check_flow_quality_then_I_w_F_to_R(dis_img=in_img, flow=gt_flow)

        cv2.imwrite(private_rec_write_dir + "/" + "rec_epoch=%04i.jpg" % current_ep, rec)
        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =[ in_img ,  flow_visual ,    rec,      rec_hope],    ### 把要顯示的每張圖包成list
                                img_titles=["in_img", "pred_flow_v", "pred_rec", "rec_hope"],    ### 把每張圖要顯示的字包成list
                                fig_title ="%s, current_ep=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)  ### 這裡會轉第2次bgr2rgb， 剛好轉成plt 的 rgb
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 這裡是轉第2次的bgr2rgb， 剛好轉成plt 的 rgb  ### 如果沒有要接續畫loss，就可以存了喔！
        print("save to:", exp_obj.result_obj.test_write_dir)
