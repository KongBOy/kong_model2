import sys
sys.path.append("kong_util")

import matplotlib.pyplot as plt
import cv2
from build_dataset_combine import Check_dir_exist_and_build, Save_as_jpg, method1
# from matplot_fig_ax_util import matplot_visual_single_row_imgs
import numpy as np
import tensorflow as tf

from step06_a_datas_obj import Range

### 用 網路 生成 影像
def I_Generate_R(model_G, _1, in_img_pre, _3, _4, use_gt_range):
    rect       = model_G(in_img_pre, training=True)  ### 把影像丟進去model生成還原影像
    # print("rect_back before max, min:", rect_back.numpy().max(), rect_back.numpy().min())  ### 測試 拉range 有沒有拉對
    if  (use_gt_range == Range(-1, 1)): rect_back  = ((rect[0].numpy() + 1) * 125).astype(np.uint8)   ### 把值從 -1~1轉回0~255 且 dtype轉回np.uint8
    elif(use_gt_range == Range( 0, 1)): rect_back  = ( rect[0].numpy() * 255).astype(np.uint8)   ### 把值從 -1~1轉回0~255 且 dtype轉回np.uint8
    # print("rect_back after max, min:", rect_back.numpy().max(), rect_back.numpy().min())  ### 測試 拉range 有沒有拉對
    return rect_back  ### 注意訓練model時是用tf來讀img，為rgb的方式訓練，所以生成的是rgb的圖喔！


### 這是一張一張進來的，沒有辦法跟 Result 裡面的 see 生成法合併，要的話就是把這裡matplot部分去除，用result裡的see生成matplot圖囉！
def I_Generate_R_see(model_G, see_index, in_img, in_img_pre, gt_img, _4, rec_hope, epoch=0, exp_obj=None, see_reset_init=False):
    rect_back = I_Generate_R(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range)

    see_write_dir  = exp_obj.result_obj.sees[see_index].see_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    plot_dir = see_write_dir + "/" + "matplot_visual"    ### 每個 see資料夾 內都有一個matplot_visual 存 in_img, rect, gt_img 併起來好看的結果

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)   ### 建立 see資料夾
        Check_dir_exist_and_build(plot_dir)  ### 建立 see資料夾/matplot_visual資料夾
        cv2.imwrite(see_write_dir + "/" + "0a-in_img.jpg", in_img[0].numpy())   ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_img.jpg", gt_img[0].numpy())  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i.jpg" % epoch, rect_back[:, :, ::-1])  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！

    ### matplot_visual的部分，記得因為用 matplot 所以要 bgr轉rgb，但是因為有用matplot_visual_single_row_imgs，裡面會bgr轉rgb了，所以這裡不用轉囉！
    ### 這部分要記得做！在 train_step3 的 exp_obj.result_obj.Draw_loss_during_train(epoch, self.epochs) 才有畫布可以畫loss！
    exp_obj.result_obj.sees[see_index].save_as_matplot_visual_during_train(epoch)

    # imgs = [in_img, rect_back, gt_img]  ### 把 in_img, rect_back, gt_img 包成list
    # titles = ['Input Image', 'rect Image', 'Ground Truth']  ### 設定 title要顯示的字
    # matplot_visual_single_row_imgs(img_titles=titles, imgs=imgs, fig_title="epoch_%04i"%epoch, dst_dir=plot_dir ,file_name="epoch=%04i"%epoch, bgr2rgb=False)
    # Save_as_jpg(plot_dir, plot_dir,delete_ord_file=True)   ### matplot圖存完是png，改存成jpg省空間



######################################################################################################################################################################################################
######################################################################################################################################################################################################
def wc_visual_op(wc):
    wc = wc.astype(np.float32)
    wc = (wc - wc.min()) / (wc.max() - wc.min() + 0.000001) * 255.
    wc_2d_v = wc.astype(np.uint8)
    return wc_2d_v

def flow_or_coord_visual_op(data):
    data_ch = data.shape[2]
    x_ind = 0
    y_ind = 0
    if  (data_ch == 3):
        '''
        mask: mask/y/x
        '''
        x_ind = 2
        y_ind = 1
    elif(data_ch == 2):
        '''
        coord: y/x
        '''
        x_ind = 1
        y_ind = 0
    return method1(data[..., x_ind], data[..., y_ind])[..., ::-1] * 255.
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def I_Generate_F(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    flow = model_G(in_img_pre, training=training)
    # print("flow before max, min:", flow.numpy().max(), flow.numpy().min())  ### 測試 拉range 有沒有拉對
    if(use_gt_range == Range(-1, 1)): flow = (flow + 1) / 2
    # print("flow after max, min:", flow.numpy().max(), flow.numpy().min())  ### 測試 拉range 有沒有拉對
    flow = flow[0].numpy()
    return flow

def I_Generate_F_see(model_G, see_index, in_img, in_img_pre, gt_flow, _4, rec_hope, epoch=0, exp_obj=None, training=True, see_reset_init=True):
    in_img   = in_img  [0].numpy()[:, :, ::-1]
    flow     = I_Generate_F(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    gt_flow  = gt_flow [0].numpy()
    rec_hope = rec_hope[0].numpy()[:, :, ::-1]

    flow_visual    = flow_or_coord_visual_op(flow)   .astype(np.uint8)
    gt_flow_visual = flow_or_coord_visual_op(gt_flow).astype(np.uint8)

    see_write_dir  = exp_obj.result_obj.sees[see_index].see_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)   ### 建立 see資料夾
        cv2.imwrite(see_write_dir + "/" + "0a-in_img.jpg",       in_img)                    ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_a_gt_flow.jpg", gt_flow_visual)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0c-rec_hope.jpg",     rec_hope)                  ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
        np.save(see_write_dir + "/" + "0b-gt_a_gt_flow",         gt_flow)                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    np.save(    see_write_dir + "/" + "epoch_%04i_a_flow"            % epoch, flow)         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_a_flow_visual.jpg" % epoch, flow_visual)  ### 把 生成的 flow_visual 存進相對應的資料夾
    # cv2.imwrite(see_write_dir + "/" + "epoch_%04i_b_in_rec_img.jpg" % epoch     , in_rec_img)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！

    ### matplot_visual的部分，記得因為用 matplot 所以要 bgr轉rgb，但是因為有用matplot_visual_single_row_imgs，裡面會bgr轉rgb了，所以這裡不用轉囉！
    ### 這部分要記得做！在 train_step3 的 exp_obj.result_obj.Draw_loss_during_train(epoch, self.epochs) 才有畫布可以畫loss！
    ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了
    # exp_obj.result_obj.sees[see_index].save_as_matplot_visual_during_train(epoch, bgr2rgb=True)
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def I_Generate_M(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    mask = model_G(in_img_pre, training=training)
    mask = mask[0].numpy()
    mask = (mask * 255).astype(np.uint8)
    return mask


def I_Generate_M_see(model_G, see_index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, epoch=0, exp_obj=None, training=True, see_reset_init=True):
    in_img  = in_img[0][:, :, ::-1].numpy()
    mask    = I_Generate_M(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    gt_mask = (gt_mask_coord[0][0].numpy() * 255).astype(np.uint8)
    # print("gt_mask.dtype:", gt_mask.dtype)
    # print("gt_mask.shape:", gt_mask.shape)
    # print("gt_mask.max():", gt_mask.numpy().max())
    # print("gt_mask.min():", gt_mask.numpy().min())

    see_write_dir  = exp_obj.result_obj.sees[see_index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    mask_write_dir = exp_obj.result_obj.sees[see_index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    print("mask_write_dir:", mask_write_dir)
    if(epoch == 0 or see_reset_init):                                              ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)                                   ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(mask_write_dir)                                  ### 建立 model生成的結果 的資料夾
        cv2.imwrite(see_write_dir  + "/" + "0a-in_img.jpg", in_img)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir  + "/" + "0b-gt_a_mask.bmp", gt_mask)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(    mask_write_dir + "/" + "epoch_%04i_a_mask.bmp" % epoch, mask)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

def I_Gen_M_test(model_G, test_index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, epoch=0, exp_obj=None, training=False, see_reset_init=True):
    pred_mask = I_Generate_M(model_G, None, in_img_pre, None, None, use_gt_range=exp_obj.use_gt_range)  ### BHWC
    pred_mask = pred_mask.numpy()   ### HWC 和 tensor -> numpy
    in_img    = in_img[0].numpy()   ### HWC 和 tensor -> numpy
    gt_mask   = test_gt[0][0].numpy()

    single_row_imgs = Matplot_single_row_imgs(
                            imgs      =[ in_img ,   pred_mask ,        gt_mask],    ### 把要顯示的每張圖包成list
                            img_titles=["in_img", "pred_mask", "gt_mask"],    ### 把每張圖要顯示的字包成list
                            fig_title ="test_%04i, epoch=%04i" % (test_index, current_epoch),   ### 圖上的大標題
                            add_loss  =add_loss,
                            bgr2rgb   =bgr2rgb)
    single_row_imgs.Draw_img()
    single_row_imgs.Save_fig(dst_dir=exp_obj.result_obj.test_dir, epoch=current_epoch, epoch_name="test_%04i" % test_index)  ### 如果沒有要接續畫loss，就可以存了喔！

    pass
######################################################################################################################################################################################################
######################################################################################################################################################################################################
def C_with_Mgt_to_F_and_get_F_visual(coord, gt_mask):
    flow        = np.concatenate([gt_mask, coord], axis=-1)  ### channel concate
    flow_visual = flow_or_coord_visual_op(flow).astype(np.uint8)
    return flow, flow_visual
####################################################################################################
def I_Generate_C(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    coord = model_G(in_img_pre, training=training)
    coord = coord[0].numpy()
    return coord

def I_Generate_C_with_Mgt_to_F_see(model_G, see_index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, epoch=0, exp_obj=None, training=True, see_reset_init=True):
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x
    '''
    # plt.imshow(in_img[0])
    # plt.show()
    in_img   = in_img[0][:, :, ::-1].numpy()
    coord    = I_Generate_C(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    gt_mask  = gt_mask_coord[0][0].numpy()
    gt_coord = gt_mask_coord[1][0].numpy()
    flow,    flow_visual    = C_with_Mgt_to_F_and_get_F_visual(coord,    gt_mask)
    gt_flow, gt_flow_visual = C_with_Mgt_to_F_and_get_F_visual(gt_coord, gt_mask)

    see_write_dir   = exp_obj.result_obj.sees[see_index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(see_write_dir + "/" + "0a-in_img.jpg",       in_img)             ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_a_gt_mask.jpg", (gt_mask * 255).astype(np.uint8))  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_a_gt_mask",     gt_mask)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_b_gt_flow.jpg", gt_flow_visual)                            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_b_gt_flow",     gt_flow)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0c-rec_hope.jpg",     rec_hope[0][:, :, ::-1].numpy())           ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    np.save(    see_write_dir + "/" + "epoch_%04i_a_flow"            % epoch, flow)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_a_flow_visual.jpg" % epoch, flow_visual)                  ### 把 生成的 flow_visual 存進相對應的資料夾

####################################################################################################
def I_Generate_W(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    wc = model_G(in_img_pre, training=training)
    wc = wc[0].numpy()
    return wc

def I_Generate_W_see(model_G, see_index, in_img, in_img_pre, gt_wc, _4, rec_hope=None, epoch=0, exp_obj=None, training=True, see_reset_init=True):
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x
    '''
    # plt.imshow(in_img[0])
    # plt.show()
    wc    = I_Generate_W(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)

    gt_wc = gt_wc[0].numpy()
    gt_wc = gt_wc[..., :3]

    wc_visual    = wc_visual_op(wc)
    gt_wc_visual = wc_visual_op(gt_wc)
    # print("wc.shape:          ", wc.shape)
    # print("wc_visual.shape:   ", wc_visual.shape)
    # print("gt_wc.shape:       ", gt_wc.shape)
    # print("gt_wc_visual.shape:", gt_wc_visual.shape)

    see_write_dir   = exp_obj.result_obj.sees[see_index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)    ### 建立 放輔助檔案 的資料夾
        cv2.imwrite(see_write_dir + "/" + "0a-in_img.jpg",       in_img[0][:, :, ::-1].numpy())             ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        # cv2.imwrite(see_write_dir + "/" + "0b-gt_a_gt_mask.jpg", (gt_mask.numpy() * 255).astype(np.uint8))  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        # np.save    (see_write_dir + "/" + "0b-gt_a_gt_mask",     gt_mask)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_b_gt_wc.jpg", gt_wc_visual)                            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_b_gt_wc",     gt_wc)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0c-rec_hope.jpg",   rec_hope[0][:, :, ::-1].numpy())           ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    np.save(    see_write_dir + "/" + "epoch_%04i_a_wc"            % epoch, wc)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_a_wc_visual.jpg" % epoch, wc_visual)                  ### 把 生成的 flow_visual 存進相對應的資料夾
######################################################################################################################################################################################################
def Mgt_Generate_C(model_G, _1, _2, _3, gt_mask_coord_pre, gt_use_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    '''
    這邊model 生成的是 ch2 的 coord， 要再跟 mask concate 後才會變成 ch3 的 flow 喔！
    '''
    gt_mask_pre  = gt_mask_coord_pre[0]
    gt_coord_pre = gt_mask_coord_pre[1]

    coord      = model_G(gt_mask_pre, training=training)
    # print("coord before max, min:", coord.numpy().max(), coord.numpy().min())  ### 測試 拉range 有沒有拉對
    if(gt_use_range == "-1~1"): coord = (coord + 1) / 2
    # print("coord after max, min:", coord.numpy().max(), coord.numpy().min())  ### 測試 拉range 有沒有拉對
    return coord


def Mgt_Generate_C_with_Mgt_to_F_see(model_G, see_index, in_img, _2, gt_mask_coord, gt_mask_coord_pre, rec_hope=None, epoch=0, exp_obj=None, training=True, see_reset_init=True):
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x
    '''
    coord     = Mgt_Generate_C(model_G, None, None, None, gt_mask_coord_pre, exp_obj.gt_use_range, training=training)
    coord     = coord[0]
    gt_mask  = gt_mask_coord[0][0]
    gt_coord = gt_mask_coord[1][0]
    flow,    flow_visual    = C_with_Mgt_to_F_and_get_F_visual(coord,    gt_mask)
    gt_flow, gt_flow_visual = C_with_Mgt_to_F_and_get_F_visual(gt_coord, gt_mask)

    see_write_dir  = exp_obj.result_obj.sees[see_index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    mask_write_dir = exp_obj.result_obj.sees[see_index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)    ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(mask_write_dir)   ### 建立 model生成的結果 的資料夾
        cv2.imwrite(see_write_dir + "/" + "0a1-in_img.jpg",      in_img[0][:, :, ::-1].numpy())             ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0a2-in_gt_mask.jpg",  (gt_mask.numpy() * 255).astype(np.uint8))  ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_a_gt_mask.jpg", (gt_mask.numpy() * 255).astype(np.uint8))  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_a_gt_mask",     gt_mask)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_b_gt_flow.jpg", gt_flow_visual)                            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_b_gt_flow",     gt_flow)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0c-rec_hope.jpg",     rec_hope[0][:, :, ::-1].numpy())           ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    np.save(    see_write_dir + "/" + "epoch_%04i_a_flow"            % epoch, flow)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_a_flow_visual.jpg" % epoch, flow_visual)                  ### 把 生成的 flow_visual 存進相對應的資料夾


######################################################################################################################################################################################################
def I_with_Mgt_Generate_C(model_G, _1, in_img_pre, _3, gt_mask_coord_pre, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    '''
    這邊model 生成的是 ch2 的 coord， 要再跟 mask concate 後才會變成 ch3 的 flow 喔！
    '''
    gt_mask_pre  = gt_mask_coord_pre[0]
    gt_coord_pre = gt_mask_coord_pre[1]
    I_with_M = in_img_pre * gt_mask_pre

    coord      = model_G(I_with_M, training=training)
    # print("coord before max, min:", coord.numpy().max(), coord.numpy().min())  ### 測試 拉range 有沒有拉對
    if(use_gt_range == Range(-1, 1)): coord = (coord + 1) / 2
    # print("coord after max, min:", coord.numpy().max(), coord.numpy().min())  ### 測試 拉range 有沒有拉對
    coord    = coord   [0].numpy()
    I_with_M = I_with_M[0].numpy()
    return coord, I_with_M


def I_with_Mgt_Generate_C_with_Mgt_to_F_see(model_G, see_index, in_img, in_img_pre, gt_mask_coord, gt_mask_coord_pre, rec_hope=None, epoch=0, exp_obj=None, training=True, see_reset_init=True):
    '''
    gt_mask_coord[0] 為 mask  (1, h, w, 1)
    gt_mask_coord[1] 為 coord (1, h, w, 2) 先y 在x
    '''
    coord, I_with_M = I_with_Mgt_Generate_C(model_G, None, in_img_pre, None, gt_mask_coord_pre, exp_obj.use_gt_range, training=training)
    gt_mask  = gt_mask_coord[0][0]
    gt_coord = gt_mask_coord[1][0]
    flow,    flow_visual    = C_with_Mgt_to_F_and_get_F_visual(coord,    gt_mask)
    gt_flow, gt_flow_visual = C_with_Mgt_to_F_and_get_F_visual(gt_coord, gt_mask)

    see_write_dir  = exp_obj.result_obj.sees[see_index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    mask_write_dir = exp_obj.result_obj.sees[see_index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)    ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(mask_write_dir)   ### 建立 model生成的結果 的資料夾
        cv2.imwrite(see_write_dir + "/" + "0a1-in_img_with_Mgt.jpg", (I_with_M[:, :, ::-1] * 255).astype(np.uint8))  ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0a2-in_gt_mask.jpg",  (gt_mask.numpy() * 255).astype(np.uint8))  ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_a_gt_mask.jpg", (gt_mask.numpy() * 255).astype(np.uint8))  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_a_gt_mask",     gt_mask)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_b_gt_flow.jpg", gt_flow_visual)                            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save    (see_write_dir + "/" + "0b-gt_b_gt_flow",     gt_flow)                                   ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0c-rec_hope.jpg",     rec_hope[0][:, :, ::-1].numpy())           ### 寫一張 rec_hope圖進去，hope 我 rec可以做到這麼好ˊ口ˋ，0c是為了保證自動排序會放在第三張
    np.save(    see_write_dir + "/" + "epoch_%04i_a_flow"            % epoch, flow)                         ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_a_flow_visual.jpg" % epoch, flow_visual)                  ### 把 生成的 flow_visual 存進相對應的資料夾
