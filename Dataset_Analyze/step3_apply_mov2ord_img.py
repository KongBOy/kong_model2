import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
from kong_util.util import get_dir_moves, get_dir_imgs, method2, get_xy_map, get_max_db_move_xy_from_numpy, time_util
import numpy as np
import cv2
import time

def get_l_r_t_d(kernel_size):
    l = t = (kernel_size + 1) // 2 - 1
    r = d = kernel_size // 2 + 1
    return l, r, t, d

def search_mask_have_hole(dis_msk, hole_size=1, debug=False):
    pag_msk = dis_msk.copy()
    pag_msk[pag_msk > 1] = 1
    row, col = pag_msk.shape[:2]

    ### 用來過濾 前景的filter
    check_foreground_size = hole_size + 2 + 2 + 2
    ch_l, ch_r, ch_t, ch_d = get_l_r_t_d(check_foreground_size)
    check_foreground_ok = check_foreground_size ** 2 * 0.83  #0.91 ### 要超過check_foreground_ok才算事前景，沒超過就算背景

    ### 用來檢查 page上有沒有洞的filter
    check_page_size = hole_size + 2
    pa_l, pa_r, pa_t, pa_d = get_l_r_t_d(check_page_size)
    page     = check_page_size ** 2
    hole     = (hole_size) // 2 + 1
    check_page_ok  = page - hole   ### 要超過page_ok才算是完整頁面(沒有洞)，沒超過就算有洞

#    debug用，視覺化前景 和 偵測到哪邊有洞
    if(debug):
        hole_around_visual = np.zeros_like(pag_msk)
        foreground_visual = np.zeros_like(pag_msk)

    around_have_hole_amount = 0
    for go_row in range(row):
        for go_col in range(col):
            ### debug用，視覺化前景
            if(debug and pag_msk[go_row - ch_t:go_row + ch_d, go_col - ch_l:go_col + ch_r].sum() > check_foreground_ok):
                foreground_visual[go_row, go_col] += 1

            if((pag_msk[go_row - ch_t:go_row + ch_d, go_col - ch_l:go_col + ch_r].sum() > check_foreground_ok) and  ### 要確定是 頁面 還是 背景
               (pag_msk[go_row - pa_t:go_row + pa_d, go_col - pa_l:go_col + pa_r].sum() <= check_page_ok)):   ### 要超過page_ok才算沒有洞，沒超過就算有洞

                if(debug):
                    # print("row,col around have hole", go_row, go_col, go_col, go_row)
                    hole_around_visual[go_row, go_col] += 1  ## debug用，視覺化偵測到哪邊有洞
                around_have_hole_amount += 1
    print("around_have_hole_amount", around_have_hole_amount)

    if(debug):
        from step0_access_path import Data_Access_Dir
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3a-hole_around_visual.bmp" % (name), hole_around_visual * 70)
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3a-foreground_visual.bmp" % (name), foreground_visual * 70)

    return around_have_hole_amount


def get_dis_img_hw_from_db_max_move_xy(row, col, max_db_move_x, max_db_move_y):
    dis_h = int(np.around(max_db_move_y + row + max_db_move_y))  ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    dis_w = int(np.around(max_db_move_x + col + max_db_move_x))  ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    return dis_h, dis_w

### 仔細思考了以後，決定取精確好了！所以就先用float加完後再取int拉！
def get_dis_img_start_left_top(move_map, max_db_move_x, max_db_move_y):
    left = int(move_map[0, 0, 0] - int(move_map[0, 0, 0]) + max_db_move_x)  ### 我們在乎的是 小數點的部分 相加後有沒有進位，所以用 move_map-int(move_map)喔！
    top  = int(move_map[0, 0, 1] - int(move_map[0, 0, 1]) + max_db_move_y)  ### 我們在乎的是 小數點的部分 相加後有沒有進位，所以用 move_map-int(move_map)喔！
    return left, top

### 這步沒有辦法用整張array一起處理，因為msk要統計相同的點被重複家幾次，這就需要一個個pixel去跑才行??思考一下好像可以解決喔！
def apply_move(img, move_map, max_db_move_x=None, max_db_move_y=None, name="0", write_to_step3=False, return_base_xy=False, dst_dir="."):
    row, col = move_map.shape[:2]  ### row, col 由 step2產生的flow 決定
    img = cv2.resize(img, (col, row), interpolation=cv2.INTER_NEAREST)

    ksize = 3
    if(max_db_move_x is None and max_db_move_y is None):
        move_x = move_map[..., 0]
        max_db_move_x = abs(move_x.max())
        # move_x_min = abs(move_x.min())
        move_y = move_map[..., 1]
        max_db_move_y = abs(move_y.max())
        # move_y_min = abs(move_y.min())

    ### 初始化各個會用到的canvas
    # dis_h = int( np.around(max_db_move_y + row + max_db_move_y) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    # dis_w = int( np.around(max_db_move_x + col + max_db_move_x) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    dis_h, dis_w = get_dis_img_hw_from_db_max_move_xy(row, col, max_db_move_x, max_db_move_y)  ### function化，因為其他地方也用的到！
    dis_img  = np.zeros(shape=(dis_h, dis_w, 3), dtype=np.float64)
    rec_mov  = np.zeros(shape=(dis_h, dis_w, 2), dtype=np.float64)
    dis_msk  = np.zeros(shape=(dis_h, dis_w)   , dtype=np.uint8)

    ### 把原始影像扭曲，並取得 rec_mov
    for go_row in range(row):
        for go_col in range(col):
            dst_x = go_col + int(move_map[go_row, go_col, 0] + max_db_move_x)  ### 現在的起點是(max_db_move_x, max_db_move_y)，所以要位移一下
            dst_y = go_row + int(move_map[go_row, go_col, 1] + max_db_move_y)  ### 現在的起點是(max_db_move_x, max_db_move_y)，所以要位移一下

            dis_img[dst_y, dst_x, :] += img[go_row, go_col, :]
            rec_mov[dst_y, dst_x, :] += move_map[go_row, go_col, :] * -1
            dis_msk[dst_y, dst_x]   += 1

            # print(dst_y,dst_x)
            # cv2.imshow("img",img)
            # cv2.imshow("Mask.bmp",(dis_msk*70).astype(np.uint8))
            # cv2.imshow("dis_img",dis_img.astype(np.uint8))
            # cv2.waitKey()


    for go_row in range(dis_img.shape[0]):
        for go_col in range(dis_img.shape[1]):
            if dis_msk[go_row, go_col] > 1:   ### 扭曲的過程中可能有 多點移到相同的點會被加多次，把他跟加的次數除回來
                dis_img[go_row, go_col] = dis_img[go_row, go_col] / dis_msk[go_row, go_col]
                rec_mov[go_row, go_col] = rec_mov[go_row, go_col] / dis_msk[go_row, go_col]

    ### 視覺化
    if(write_to_step3):
        from step0_access_path import Data_Access_Dir
        move_map_visual = method2(move_map[..., 0], move_map[..., 1], color_shift=1)
        np.save(Data_Access_Dir + dst_dir + "/" + "%s-2-q" % (name), move_map)
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-1-I.bmp" % (name), img)
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-2-q.jpg" % (name), move_map_visual)
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3a2-I1.jpg" % (name), dis_img)
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3a3-Mask.bmp" % (name), dis_msk * 70)
        np.save    (Data_Access_Dir + dst_dir + "/" + "%s-3a3-Mask" % (name), dis_msk)

        gt_pad_img = np.zeros(shape=(dis_h, dis_w, 3), dtype=np.float64)
        left, top  = get_dis_img_start_left_top(move_map, max_db_move_x, max_db_move_y)
        gt_pad_img[top:top + row, left:left + col] = img
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-4-gt_ord_pad.bmp" % (name), gt_pad_img)

    ####################################################################################################################
    #### 扭曲影像 空洞的地方補起來
    search_mask_have_hole_count = 0
    while(search_mask_have_hole(dis_msk) > 0 and search_mask_have_hole_count < 10):
        dis_img_ref = dis_img.copy()  ### 要把 參考img 跟 處理中img 分開 喔！
        dis_msk_ref = dis_msk.copy()  ### 要把 參考msk 跟 處理中msk 分開 喔！
        rec_mov_ref = rec_mov.copy()  ### 要把 參考rec_mov 跟 處理中rec_mov 分開 喔！
        for go_row in range(dis_h):
            for go_col in range(dis_w):
                if(dis_msk_ref[go_row, go_col] == 0 and dis_msk_ref[go_row - 1:go_row + 2, go_col - 1:go_col + 2].sum() >=6 ):
                    # print(go_row,go_col)
                    l, r, t, d = get_l_r_t_d(ksize)
                    msk_sum = dis_msk_ref[go_row - t:go_row + d, go_col - l:go_col + r].sum()

                    dis_img[go_row,go_col,0] =  (dis_img_ref[go_row-t:go_row+d,go_col-l:go_col+r,0]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    dis_img[go_row,go_col,1] =  (dis_img_ref[go_row-t:go_row+d,go_col-l:go_col+r,1]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    dis_img[go_row,go_col,2] =  (dis_img_ref[go_row-t:go_row+d,go_col-l:go_col+r,2]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)

                    rec_mov[go_row,go_col,0] = (rec_mov_ref[go_row-t:go_row+d,go_col-l:go_col+r,0]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    rec_mov[go_row,go_col,1] = (rec_mov_ref[go_row-t:go_row+d,go_col-l:go_col+r,1]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)

                    dis_msk[go_row,go_col] += 1
        search_mask_have_hole_count += 1
    if(write_to_step3):
        from step0_access_path import Data_Access_Dir
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3a1-I1-patch.bmp" % (name), dis_img.astype(np.uint8))
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3a4-Mask-patch.bmp" % (name), dis_msk * 70)
        np.save(Data_Access_Dir + dst_dir + "/" + "%s-3a4-Mask-patch" % (name), dis_msk)

        np.save(Data_Access_Dir + dst_dir + "/" + "%s-3b-rec_mov_map" % (name), rec_mov.astype(np.float32))
        rec_mov_visual = method2(rec_mov[:, :, 0], rec_mov[:, :, 1], 1)
        cv2.imwrite(Data_Access_Dir + dst_dir + "/" + "%s-3b-rec_mov_visual.jpg" % (name), rec_mov_visual)
        print("dis_msk.max()", dis_msk.max())

    dis_img = dis_img.astype(np.uint8)  ### float64運算完記得 轉回 uint8 才能正常顯示喔！

    if(return_base_xy):
        return dis_img.copy(), rec_mov.copy(), max_db_move_x, max_db_move_y

    return dis_img.copy(), rec_mov.copy()


def apply_move_at_lots_process(start_index, amount, ord_imgs, ord_imgs_amount, move_maps, max_db_move_x, max_db_move_y, write_to_step3, dst_dir):
    start_time = time.time()
    for i, move_map in enumerate(move_maps[start_index:start_index + amount]):
        apply_start_time = time.time()                      ### 計算處理一張花的時間
        img = ord_imgs[np.random.randint(ord_imgs_amount)]  ### 從 ord_img裡隨便挑一張
        name = "%06i" % (i + start_index)                       ### 設定 流水號的檔名
        dis_img, rec_mov = apply_move(img, move_map, max_db_move_x=max_db_move_x, max_db_move_y=max_db_move_y, name=name, write_to_step3=write_to_step3, dst_dir=dst_dir)  ### ord_img 去apply扭曲
        print("%06i process 1 mesh cost time:" % (i + start_index), "%.3f" % (time.time() - apply_start_time), "total_time:", time_util(time.time() - start_time))
        print("")


def load_data_and_apply_move(ord_imgs_dir, move_maps_dir, dst_dir, start_index, write_to_step3=True):
    from step0_access_path import Data_Access_Dir

    start_time = time.time()

    ord_imgs        = get_dir_imgs(Data_Access_Dir + ord_imgs_dir)              ### 取得ord_imgs
    ord_imgs_amount = len(ord_imgs)                                             ### 取得ord_imgs個數，等等取隨機時用的到
    move_maps       = get_dir_moves(Data_Access_Dir + move_maps_dir)            ### 取得move_maps
    max_db_move_x, max_db_move_y = get_max_db_move_xy_from_numpy(move_maps)     ### 取得move_maps 的整體最大移動量 max_db_move_xy
    print("max_db_move_x", max_db_move_x, ", max_db_move_y", max_db_move_y)     ### 看一下 max_db_move_xy 是多少
    Check_dir_exist_and_build(Data_Access_Dir + dst_dir)                       ### 建立放結果的資料夾


    core_amount = 8
    amount = len(move_maps)
    split_amount = int(amount // core_amount)
    fract_amount = int(amount % core_amount)

    from multiprocessing import Process
    processes = []

    for i in range(core_amount):
        process_amount = split_amount
        if(i == (core_amount - 1) and (fract_amount != 0)): process_amount += fract_amount
        processes.append(Process(target=apply_move_at_lots_process, args=(split_amount * i, process_amount, ord_imgs, ord_imgs_amount, move_maps, max_db_move_x, max_db_move_y, write_to_step3, dst_dir)))
        print("registering process %02i dealing %04i data" % (i, process_amount))

    for process in processes:
        process.start()




if(__name__ == "__main__"):
    # src_dir     = "step2_build_flow_h=384,w=256_complex"
    # src_dir     = "step2_build_flow_h=384,w=256_complex+page"
    # src_dir     = "step2_build_flow_h=384,w=256_complex+page_more_like"
    # src_dir       = "step2_build_flow_h=384,w=256_smooth_curl+fold"
    # src_dir       = "step2_build_flow_h=384,w=256_page"
    # src_dir       = "step2_build_flow_h=384,w=256_prep"
    src_dir       = "step2_build_flow_h=384,w=256_smooth-curl+fold_and_page"
    ord_imgs_dir  = "step1_page"

    ### 這是用在 如果不小心中斷，可以用這設定從哪裡開始
    # start_index = 0
    # amount      = 2000

    ### 也可以用來切給多個python平行處理
    start_index = 0

    ################################################################################
    move_maps_dir = src_dir + "/" + "move_maps"
    dst_dir       = src_dir.replace("step2_build_flow", "step3_apply_flow")
    load_data_and_apply_move(ord_imgs_dir, move_maps_dir, dst_dir, start_index, write_to_step3=True)
