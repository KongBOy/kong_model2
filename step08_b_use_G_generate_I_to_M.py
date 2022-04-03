import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy

import matplotlib.pyplot as plt
import os
import pdb

class Use_G_generate:
    def __init__(self):
        self.model_obj      = None
        self.phase          = None
        self.index          = None
        self.in_ord         = None
        self.in_pre         = None
        self.gt_ord         = None
        self.gt_pre         = None
        self.rec_hope       = None
        self.exp_obj        = None
        self.training       = None
        self.see_reset_init = None
        self.postprocess    = None
        self.npz_save       = None
        self.add_loss       = None
        self.bgr2rgb        = None

    def __call__(self, model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
        self.model_obj      = model_obj
        self.phase          = phase
        self.index          = index
        self.in_ord         = in_ord
        self.in_pre         = in_pre
        self.gt_ord         = gt_ord
        self.gt_pre         = gt_pre
        self.rec_hope       = rec_hope
        self.exp_obj        = exp_obj
        self.training       = training
        self.see_reset_init = see_reset_init
        self.postprocess    = postprocess
        self.npz_save       = npz_save
        self.add_loss       = add_loss
        self.bgr2rgb        = bgr2rgb
        self.doing_things()

    def doing_things(self):
        pass

# class Use_G_generate:
#     def __call__(self,model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
#         self.doing_things(model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False)

#     def doing_things(self,model_obj, phase, index, in_ord, in_pre, gt_ord, gt_pre, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=False):
#         ''' Not Implement'''
#         pass

def tight_crop(data, Mask, pad_size=20, resize=None):
    import tensorflow as tf
    '''
    目前的寫法連 batch 都考慮進去囉
    resize: [h, w]
    '''
    ### np.where( T/F_map 或 0/非0_array ) 可參考：https://numpy.org/doc/stable/reference/generated/numpy.where.html，
    ### np.where( T/F_map 或 0/非0_array ) 只有放一個參數的時候， 相當於 np.nonzero()， 如果放三個參數時， 為true的地方 填入 第二個參數值， 為False的地方 填入 第三個參數值,
    ### 不直接用 nonzero的原因是 tf 沒有 nonzero 但有 where， 為了 tf, numpy 都通用 這邊 numpy 就配合tf 用 where 囉～
    ### 但也要注意 tf.where( T/F_map 或 0/非0_array ) return 回來的東西 跟 np.where( T/F_map 或 0/非0_array ) 回傳的東西 shape 不大一樣喔
    ### np.where( T/F_map 或 0/非0_array ) -> tuple ( x非零indexs, y非零indexs, z非零indexs )
    ### tf.where( T/F_map 或 0/非0_array ) -> tensor.shape( 所有非零點的個數, 3 ( 非零xyz_index) )
    nonzero_map = Mask > 0
    ### numpy 寫法：
    # if  (len(Mask.shape) == 4): b_ind, y_ind, x_ind, c_ind = np.where(nonzero_map)
    # elif(len(Mask.shape) == 3): y_ind, x_ind, c_ind = np.where(nonzero_map)
    # elif(len(Mask.shape) == 2): y_ind, x_ind = np.where(nonzero_map)

    ### tf 寫法
    nonzero_index = tf.where(nonzero_map)
    if  (len(Mask.shape) == 4): col_id = 1  ### BHWC， H在第1個col
    elif(len(Mask.shape) == 3): col_id = 0  ### HWC ， H在第0個col
    elif(len(Mask.shape) == 2): col_id = 0  ### HWC ， H在第0個col
    y_ind = nonzero_index[:, col_id     : col_id + 1]
    x_ind = nonzero_index[:, col_id + 1 : col_id + 2]

    # x_min = x_ind.min()
    # x_max = x_ind.max()
    # y_min = y_ind.min()
    # y_max = y_ind.max()
    x_min = tf.reduce_min(x_ind)
    x_max = tf.reduce_max(x_ind)
    y_min = tf.reduce_min(y_ind)
    y_max = tf.reduce_max(y_ind)

    l_pad = x_min - pad_size
    r_pad = x_max + pad_size
    t_pad = y_min - pad_size
    d_pad = y_max + pad_size
    l_out = tf.constant(0, tf.int64)
    r_out = tf.constant(0, tf.int64)
    t_out = tf.constant(0, tf.int64)
    d_out = tf.constant(0, tf.int64)
    if  (len(data.shape) == 4): b, h, w, c = data.shape
    elif(len(data.shape) == 3): h, w, c = data.shape
    elif(len(data.shape) == 2): h, w = data.shape

    ### 先 pad 再 crop
    if(l_pad < 0): l_out = - l_pad
    if(t_pad < 0): t_out = - t_pad
    if(r_pad > w - 1): r_out = r_pad - (w - 1)
    if(d_pad > h - 1): d_out = d_pad - (h - 1)

    ### 看 pad 的範圍有沒有超過影像， 有的話就 pad
    if(l_out > 0 or r_out > 0  or t_out > 0 or d_out > 0):
        # if  (len(data.shape) == 4): data = np.pad(data, ( (0    ,     0), (t_out, d_out), (l_out, r_out), (    0,     0) ) , 'reflect')
        # elif(len(data.shape) == 3): data = np.pad(data, ( (t_out, d_out), (l_out, r_out), (    0,     0) )                 , 'reflect')
        # elif(len(data.shape) == 2): data = np.pad(data, ( (t_out, d_out), (l_out, r_out) )                                 , 'reflect')
        if  (len(data.shape) == 4): data = tf.pad(data, ( (0    ,     0), (t_out, d_out), (l_out, r_out), (    0,     0) ) , 'REFLECT')
        elif(len(data.shape) == 3): data = tf.pad(data, ( (t_out, d_out), (l_out, r_out), (    0,     0) )                 , 'REFLECT')
        elif(len(data.shape) == 2): data = tf.pad(data, ( (t_out, d_out), (l_out, r_out) )                                 , 'REFLECT')
    # breakpoint()

    ### 對pad完的 data 重新定位
    if(l_pad < 0): l_pad = tf.constant(0, tf.int64)
    if(r_pad < 0): r_pad = tf.constant(0, tf.int64)
    # l_pad = max(l_pad, 0)          ### l_pad, t_pad 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0
    # t_pad = max(t_pad, 0)          ### l_pad, t_pad 可能會被剪到 負的， 但index最小是0喔 ， 所以最小取0
    r_pad = r_pad + l_out + r_out  ### r_pad, d_pad 自己如果超過的話， 因為會pad出去， 所以要加上 超過的部分， 在來還要考慮如果 l_pad, t_pad 超出去的話， 因為index最小為0， 代表 左、上 超出去的部分 要補到 右、下 的部分， 所以要多加 l_out, t_out 喔！
    d_pad = d_pad + t_out + d_out  ### r_pad, d_pad 自己如果超過的話， 因為會pad出去， 所以要加上 超過的部分， 在來還要考慮如果 l_pad, t_pad 超出去的話， 因為index最小為0， 代表 左、上 超出去的部分 要補到 右、下 的部分， 所以要多加 l_out, t_out 喔！

    ### pad 完以後再 crop
    if  (len(data.shape) == 4): data = data[:, t_pad : d_pad + 1, l_pad : r_pad + 1, :]  ### BHWC
    elif(len(data.shape) == 3): data = data[t_pad : d_pad + 1, l_pad : r_pad + 1, :]     ### HWC
    elif(len(data.shape) == 2): data = data[t_pad : d_pad + 1, l_pad : r_pad + 1]        ### HW

    if(resize is not None): data = tf.image.resize(data, resize, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # breakpoint()
    return data


class I_to_M(Use_G_generate):
    def __init__(self, tight_crop=False, pad_size=20, resize=None):
        super(I_to_M, self).__init__()
        # self.tight_crop = tight_crop
        self.tight_crop = tight_crop

    def doing_things(self):
        current_ep = self.exp_obj.current_ep
        current_time = self.exp_obj.current_time
        if  (self.phase == "train"): used_sees = self.exp_obj.result_obj.sees
        elif(self.phase == "test"):  used_sees = self.exp_obj.result_obj.tests
        private_write_dir      = used_sees[self.index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
        private_mask_write_dir = used_sees[self.index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
        public_write_dir       = "/".join(used_sees[self.index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
        # print('public_write_dir:', public_write_dir)

        ''' 重新命名 讓我自己比較好閱讀'''
        in_img            = self.in_ord
        in_img_pre        = self.in_pre
        gt_mask_coord     = self.gt_ord
        gt_mask_coord_pre = self.gt_pre

        if(self.tight_crop):
            pad_size = 20
            reisze = (256, 256)

            gt_mask_pre = gt_mask_coord_pre[..., 0:1]

            in_img            = tight_crop(in_img, gt_mask_pre, pad_size, reisze)
            in_img_pre        = tight_crop(in_img_pre, gt_mask_pre, pad_size, reisze)
            gt_mask_coord     = tight_crop(gt_mask_coord, gt_mask_pre, pad_size, reisze)
            gt_mask_coord_pre = tight_crop(gt_mask_coord_pre, gt_mask_pre, pad_size, reisze)


        ''' use_model '''
        M_pre = self.model_obj.generator(in_img_pre, training=self.training)
        M_pre = M_pre[0].numpy()
        M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
        M_visual = (M * 255).astype(np.uint8)

        '''
        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''
        in_img    = in_img[0].numpy()
        gt_mask   = (gt_mask_coord[0, ..., 0:1].numpy() * 255).astype(np.uint8)
        # print("gt_mask.dtype:", gt_mask.dtype)
        # print("gt_mask.shape:", gt_mask.shape)
        # print("gt_mask.max():", gt_mask.numpy().max())
        # print("gt_mask.min():", gt_mask.numpy().min())

        if(self.bgr2rgb): in_img = in_img[:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr

        if(current_ep == 0 or self.see_reset_init):                                              ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)                                   ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_mask_write_dir)                                  ### 建立 model生成的結果 的資料夾
            cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img(in_img).jpg", in_img)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
            cv2.imwrite(private_write_dir  + "/" + "0b_u1b1-gt_mask.jpg", gt_mask)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(    private_mask_write_dir + "/" + "epoch_%04i_u1b1_mask.jpg" % current_ep, M_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

        if(self.postprocess):
            current_see_name = used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            from matplot_fig_ax_util import Matplot_single_row_imgs
            imgs = [ in_img.astype(np.uint8) ,   M_visual , gt_mask]
            img_titles = ["in_img", "M", "gt_mask"]

            single_row_imgs = Matplot_single_row_imgs(
                                    imgs      =imgs,         ### 把要顯示的每張圖包成list
                                    img_titles=img_titles,               ### 把每張圖要顯示的字包成list
                                    fig_title ="%s, epoch=%04i" % (current_see_name, int(current_ep)),  ### 圖上的大標題
                                    add_loss  =self.add_loss,
                                    bgr2rgb   =self.bgr2rgb)
            single_row_imgs.Draw_img()
            single_row_imgs.Save_fig(dst_dir=public_write_dir, name=current_see_name)  ### 如果沒有要接續畫loss，就可以存了喔！

            '''
            Fake_F 的部分
            '''
            if(self.phase == "test"):
                gather_mask_dir   = public_write_dir + "/pred_mask"
                Check_dir_exist_and_build(gather_mask_dir)
                cv2.imwrite(f"{gather_mask_dir}/{current_see_name}.jpg", M_visual)

                h, w = M.shape[:2]
                fake_name = current_see_name.split(".")[0]
                print("")
                ###############################################################################
                fake_C = np.zeros(shape=(h, w, 2), dtype=np.float32)
                fake_F = np.concatenate((M, fake_C), axis=-1)
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
                fake_W = np.concatenate((fake_W, M), axis=-1)
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



######################################################################################################################################################################################################
######################################################################################################################################################################################################
def use_model(model_obj, _1, in_img_pre, _3, _4, use_gt_range, training=False):  ### training 這個參數是為了 一開使 用BN ，為了那些exp 還能重現所以才保留，現在用 IN 完全不會使用到他這樣子拉～
    M_pre = model_obj.generator(in_img_pre, training=training)
    M_pre = M_pre[0].numpy()
    M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
    M_visual = (M * 255).astype(np.uint8)
    return M, M_visual

def I_Generate_M_see(model_obj, phase, index, in_img, in_img_pre, gt_mask_coord, _4, rec_hope=None, exp_obj=None, training=True, see_reset_init=True, postprocess=False, npz_save=False, add_loss=False, bgr2rgb=True):
    current_ep = exp_obj.current_ep
    current_time = exp_obj.current_time
    if  (phase == "train"): used_sees = exp_obj.result_obj.sees
    elif(phase == "test"):  used_sees = exp_obj.result_obj.tests
    private_write_dir      = used_sees[index].see_write_dir   ### 每個 see 都有自己的資料夾 存 in/gt 之類的 輔助檔案 ，先定出位置
    private_mask_write_dir = used_sees[index].mask_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    public_write_dir       = "/".join(used_sees[index].see_write_dir.replace("\\", "/").split("/")[:-1])  ### private 的上一層資料夾
    # print('public_write_dir:', public_write_dir)
    '''
    bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
    '''
    in_img    = in_img[0].numpy()
    gt_mask   = (gt_mask_coord[0, ..., 0:1].numpy() * 255).astype(np.uint8)
    # print("gt_mask.dtype:", gt_mask.dtype)
    # print("gt_mask.shape:", gt_mask.shape)
    # print("gt_mask.max():", gt_mask.numpy().max())
    # print("gt_mask.min():", gt_mask.numpy().min())

    pred_mask, pred_mask_visual = use_model(model_obj, None, in_img_pre, None, None, exp_obj.use_gt_range, training=training)

    # print("bgr2rgb", bgr2rgb)
    if(bgr2rgb): in_img = in_img[:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr

    if(current_ep == 0 or see_reset_init):                                              ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(private_write_dir)                                   ### 建立 放輔助檔案 的資料夾
        Check_dir_exist_and_build(private_mask_write_dir)                                  ### 建立 model生成的結果 的資料夾
        cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img(in_img).jpg", in_img)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(private_write_dir  + "/" + "0b_u1b1-gt_mask.jpg", gt_mask)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(    private_mask_write_dir + "/" + "epoch_%04i_u1b1_mask.jpg" % current_ep, pred_mask_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

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
            cv2.imwrite(f"{gather_mask_dir}/{current_see_name}.jpg", pred_mask_visual)

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
