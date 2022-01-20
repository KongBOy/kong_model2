import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy

import matplotlib.pyplot as plt
import os

######################################################################################################################################################################################################
######################################################################################################################################################################################################
def I_Generate_M(model_G, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    M_pre = model_G(in_img_pre, training=training)
    M_pre = M_pre[0].numpy()
    M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
    M_visual = (M * 255).astype(np.uint8)
    return M, M_visual

def I_Generate_M_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, exp_obj=None, training=True, bgr2rgb=False):
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
                                但 plt 存圖是rgb， 所以存圖不用轉ch， 把 bgr2rgb設False喔！
    '''
    in_img    = in_img[0].numpy()
    pred_mask, pred_mask_visual = I_Generate_M(model_G, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)
    gt_mask   = (gt_mask_coord[0, ..., 0:1].numpy() * 255).astype(np.uint8)

    if(bgr2rgb): in_img = in_img[:, :, ::-1]  ### tf2 讀出來是 rgb， 但cv2存圖是bgr， 所以記得要轉一下ch
    # print("gt_mask.dtype:", gt_mask.dtype)
    # print("gt_mask.shape:", gt_mask.shape)
    # print("gt_mask.max():", gt_mask.numpy().max())
    # print("gt_mask.min():", gt_mask.numpy().min())
    return in_img, pred_mask, pred_mask_visual, gt_mask

def I_Generate_M_see(model_G, phase, index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "train"): used_sees = exp_obj.result_obj.sees
    elif(phase == "test"):  used_sees = exp_obj.result_obj.tests
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    in_img, pred_mask, pred_mask_visual, gt_mask = I_Generate_M_basic_data(model_G, in_img, in_img_pre, gt_mask_coord, exp_obj, training, bgr2rgb=True)
    private_write_dir      = used_sees[index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_mask_write_dir = used_sees[index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    public_write_dir       = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    # print('public_write_dir:', public_write_dir)

    if(current_ep == 0 or see_reset_init):                                              ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)                                   ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(private_mask_write_dir)                                  ### 建立 model生成的結果 的資料夾
        cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img(in_img).jpg", in_img)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(private_write_dir  + "/" + "0b_u1b1-gt_mask.bmp", gt_mask)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(    private_mask_write_dir + "/" + "epoch_%04i_u1b1_mask.bmp" % current_ep, pred_mask_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

    if(postprocess):
        current_see_name = used_sees[index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
        from matplot_fig_ax_util import Matplot_single_row_imgs
        imgs = [ in_img ,   pred_mask_visual , gt_mask]
        img_titles = ["in_img", "pred_mask", "gt_mask"]

        single_row_imgs = Matplot_single_row_imgs(
                                imgs      =imgs,         ### 把要顯示的每張圖包成list
                                img_titles=img_titles,               ### 把每張圖要顯示的字包成list
                                fig_title ="%s, epoch=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                add_loss  =add_loss,
                                bgr2rgb   =bgr2rgb)
        single_row_imgs.Draw_img()
        single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 如果沒有要接續畫loss，就可以存了喔！

        '''
        Fake_F 的部分
        '''
        if(phase == "test"):
            gather_mask_dir   = public_write_dir + "/pred_mask"
            Check_dir_exist_and_build(gather_mask_dir)
            cv2.imwrite(f"{gather_mask_dir}/{current_see_name}.bmp", pred_mask_visual)

            h, w = pred_mask.shape[:2]
            fake_name = current_see_name.split(".")[0]
            print("")
            ###############################################################################
            fake_C = np.zeros(shape=(h, w, 2), dtype=np.float32)
            fake_F = np.concatenate((pred_mask, fake_C), axis=-1)
            fake_F = fake_F.astype(np.float32)

            ### 定位出 存檔案的位置
            gather_fake_F_dir = public_write_dir + "/pred_mask/fake_F"
            gather_fake_F_npy_dir  = gather_fake_F_dir + "/1 npy_then_npz"
            gather_fake_F_knpy_dir = gather_fake_F_dir + "/2 knpy"
            Check_dir_exist_and_build(gather_fake_F_dir)
            Check_dir_exist_and_build(gather_fake_F_npy_dir)
            Check_dir_exist_and_build(gather_fake_F_knpy_dir)

            ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
            fake_F_npy_path  = f"{gather_fake_F_npy_dir}/{fake_name}.npy"
            fake_F_knpy_path = f"{gather_fake_F_knpy_dir}/{fake_name}.knpy"
            np.save(fake_F_npy_path, fake_F)
            Save_npy_path_as_knpy(fake_F_npy_path, fake_F_knpy_path)
            print("fake_F_npy_path :", fake_F_npy_path)
            print("fake_F_knpy_path:", fake_F_knpy_path)

            ### .npy刪除(因為超占空間) 改存 .npz
            np.savez_compressed(fake_F_npy_path.replace(".npy", ".npz"), fake_F)
            os.remove(fake_F_npy_path)
            ###############################################################################
            fake_W = np.zeros(shape=(h, w, 3), dtype=np.float32)
            fake_W = np.concatenate((fake_W, pred_mask), axis=-1)
            fake_W = fake_W.astype(np.float32)

            ### 定位出 存檔案的位置
            gather_fake_W_dir = public_write_dir + "/pred_mask/fake_W"
            gather_fake_W_npy_dir  = gather_fake_W_dir + "/1 npy"
            gather_fake_W_knpy_dir = gather_fake_W_dir + "/2 knpy"
            Check_dir_exist_and_build(gather_fake_W_dir)
            Check_dir_exist_and_build(gather_fake_W_npy_dir)
            Check_dir_exist_and_build(gather_fake_W_knpy_dir)

            ### 存.npy(必須要！不能直接存.npz，因為轉.knpy是要他存成檔案後把檔案頭去掉才能變.knpy喔) 和 .knpy
            fake_W_npy_path  = f"{gather_fake_W_npy_dir}/{fake_name}.npy"
            fake_W_knpy_path = f"{gather_fake_W_knpy_dir}/{fake_name}.knpy"
            np.save(fake_W_npy_path, fake_W)
            Save_npy_path_as_knpy(fake_W_npy_path, fake_W_knpy_path)
            print("fake_W_npy_path :", fake_W_npy_path)
            print("fake_W_knpy_path:", fake_W_knpy_path)

            ### .npy刪除(因為超占空間) 改存 .npz
            np.savez_compressed(fake_W_npy_path.replace(".npy", ".npz"), fake_W)
            os.remove(fake_W_npy_path)
