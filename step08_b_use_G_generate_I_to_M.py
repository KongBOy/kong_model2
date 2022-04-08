import numpy as np
import cv2

from step06_a_datas_obj import Range

import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
from step08_b_use_G_generate_0_util import tight_crop

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


class I_to_M(Use_G_generate):
    def __init__(self, tight_crop=False, pad_size=20, resize=None):
        super(I_to_M, self).__init__()
        self.tight_crop = tight_crop
        self.pad_size = pad_size
        self.resize   = resize

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
            gt_mask_pre = gt_mask_coord_pre[..., 0:1]

            in_img            = tight_crop(in_img, gt_mask_pre, self.pad_size, self.resize)
            in_img_pre        = tight_crop(in_img_pre, gt_mask_pre, self.pad_size, self.resize)
            gt_mask_coord     = tight_crop(gt_mask_coord, gt_mask_pre, self.pad_size, self.resize)
            gt_mask_coord_pre = tight_crop(gt_mask_coord_pre, gt_mask_pre, self.pad_size, self.resize)


        ''' use_model '''
        M_pre = self.model_obj.generator(in_img_pre, training=self.training)
        M_pre = M_pre[0].numpy()
        M = M_pre  ### 因為 mask 要用 BCE， 所以Range 只可能 Range(0, 1)， 沒有其他可能， 所以不用做 postprocess M 就直接是 M_pre 囉
        M_visual = (M * 255).astype(np.uint8)

        '''
        bgr2rgb： tf2 讀出來是 rgb， 但 cv2 存圖是bgr， 所以此狀況記得要轉一下ch 把 bgr2rgb設True！
        '''
        in_img    = in_img[0].numpy()
        Mgt_visual   = (gt_mask_coord[0, ..., 0:1].numpy() * 255).astype(np.uint8)
        # print("Mgt_visual.dtype:", Mgt_visual.dtype)
        # print("Mgt_visual.shape:", Mgt_visual.shape)
        # print("Mgt_visual.max():", Mgt_visual.numpy().max())
        # print("Mgt_visual.min():", Mgt_visual.numpy().min())

        if(self.bgr2rgb): in_img = in_img[:, :, ::-1]  ### 這裡是轉第1次的bgr2rgb， 轉成cv2 的 bgr

        if(current_ep == 0 or self.see_reset_init):                                              ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
            Check_dir_exist_and_build(private_write_dir)                                   ### 建立 放輔助檔案 的資料夾
            Check_dir_exist_and_build(private_mask_write_dir)                                  ### 建立 model生成的結果 的資料夾
            cv2.imwrite(private_write_dir  + "/" + "0a_u1a0-dis_img(in_img).jpg", in_img)                ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
            cv2.imwrite(private_write_dir  + "/" + "0b_u1b1-gt_mask.jpg", Mgt_visual)            ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(    private_mask_write_dir + "/" + "epoch_%04i_u1b1_mask.jpg" % current_ep, M_visual)  ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～

        if(self.postprocess):
            current_see_name = used_sees[self.index].see_name.replace("/", "-")  ### 因為 test 會有多一層 "test_db_name"/test_001， 所以把 / 改成 - ，下面 Save_fig 才不會多一層資料夾
            from kong_util.matplot_fig_ax_util import Matplot_single_row_imgs
            imgs = [ in_img.astype(np.uint8) ,   M_visual , Mgt_visual]
            img_titles = ["in_img", "M", "Mgt_visual"]

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
                ### 先粗略寫， 有時間再來敢先趕meeting
                M = cv2.resize(M, (512, 512), interpolation=cv2.INTER_AREA)
                M = M.reshape(512, 512, 1)

                gather_mask_dir   = public_write_dir + "/pred_mask"
                Check_dir_exist_and_build(gather_mask_dir)
                cv2.imwrite(f"{gather_mask_dir}/{current_see_name}.jpg", M_visual)

                h, w = M.shape[:2]
                fake_name = current_see_name.split(".")[0]
                print("")
                ###############################################################################
                ### 準備存 fake_F
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
                ### 準備存 fake_W
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
