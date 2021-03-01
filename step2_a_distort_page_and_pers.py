import sys
sys.path.append("kong_util")
import time
import numpy as np
import matplotlib.pyplot as plt
from util import get_xy_map, Show_3d_scatter_along_xy, method2, Show_move_map_apply, time_util
from build_dataset_combine import Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir
from step0_access_path import data_access_path
from step3_apply_mov2ord_img import apply_move


### 這是觀察try_calcOpticalFlowFarneback的結果直接分析 perspective_y方向怎麼位移 的方法，找四次方程式，目前失敗了
def build_perspective_move_y(row, col):
    width  = col
    height = row
    x, y = get_xy_map(row, col)

    # 寫出係數矩陣 A
    A = np.array([
        [ 25**4,  25**3,  25**2,  25**1],
        [ 50**4,  50**3,  50**2,  50**1],
        [100**4, 100**3, 100**2, 100**1],
        [360**4, 360**3, 360**2, 360**1]
    ])
    # 寫出常數矩陣 B
    B = np.array([-8, 0, 13, -30]).reshape(4, 1)
    # 找出係數矩陣的反矩陣 A_inv
    A_inv = np.linalg.inv(A)
    # 將 A_inv 與 B 相乘，即可得到解答
    ans = A_inv.dot(B)
    print("ans", ans)



    ### 公式推倒過程看筆記
    a = ans[0]  #-0.0000604
    b = ans[1]  #0.2336
    c = ans[2]  #-0.864
    d = ans[3]
    z = a * y ** 4 + b * y ** 3 + c * y ** 2 + d * y ** 1
    return z

# row=360
# col=270
# move_y = build_perspective_move_y(row, col)



def build_perspective_move_map(row, col, ratio_col_t, ratio_row, ratio_col_d):
    '''
    # row：高度
    # col：寬度
    # ratio_col_t：上邊縮小的比率，可以用0.85
    # ratio_row  ：高度縮小的比率，可以用0.95
    # ratio_col_d：下邊縮小的比率，可以用0.95
    '''
    ### 注意這邊(0,0)是用圖的中心當原點，不是圖的左上角喔！
    x, y = get_xy_map(row, col)
    x = x - int(col / 2)
    y = y - int(row / 2)

    persp_row = row * ratio_row / 2  ### 下方 y方向的縮小比率
    persp_y   = y   * ratio_row      ### 下方 透射後的y座標


    ratio_col = ratio_col_t * (persp_row - persp_y) / (persp_row * 2)  +  \
                ratio_col_d * (persp_y  - (-int(row / 2) * ratio_row)) / (persp_row * 2)    ### x方向的縮小比率，這事會隨y而變化的，看筆記的圖才好理解喔！
    persp_x   = x * ratio_col  ### 透射後的x座標

    move_x = persp_x - x  ### 原x座標怎麼位移到透射
    move_y = persp_y - y  ### 原y座標怎麼位移到透射
    move_map = np.dstack((move_x, move_y))  ### 我move_map個格式就是shape=(row, col, 2)，所以拼起來一下囉！
    return move_map


def build_page_move_map(row, col, top_curl=27, down_curl=19, lr_shift=5):
    ### 備份一下，怕改錯，沒問題可刪囉
    # def build_page_move_y(row, col):
    #     width  = col
    #     height = row
    #     x, y = get_xy_map(row, col)

    #     ### 公式推倒過程看筆記
    #     A = (-19-27) / (height*(width/2)**2)
    #     m = A* (x-(width/2))**2
    #     B = (27-0)*4/(width**2)       ### 原本27-6，但覺得-0也可以，覺得原本-6應該是因為 拍照的誤差造成的 最小移動量要往上跑6，所以不要應該也是可以
    #     b = B*((x- (width/2))**2) + 0 ### 原本  +6，但覺得+0也可以，覺得原本+6應該是因為 拍照的誤差造成的 最小移動量要往上跑6，所以不要應該也是可以
    #     z = m*y +b
    #     return z

    def build_page_move_y(row, col, top_curl=27, down_curl=19):
        width  = col
        height = row
        x, y = get_xy_map(row, col)

        ### 公式推倒過程看筆記
        A = (-down_curl - top_curl) / (height * (width / 2) ** 2)
        m = A * (x - (width / 2)) ** 2
        B = (top_curl - 0) * 4 / (width ** 2)   ### 原本27-6，但覺得-0也可以，覺得原本-6應該是因為 拍照的誤差造成的 最小移動量要往上跑6，所以不要應該也是可以
        b = B * ((x - (width / 2))**2) + 0      ### 原本  +6，但覺得+0也可以，覺得原本+6應該是因為 拍照的誤差造成的 最小移動量要往上跑6，所以不要應該也是可以
        z = m * y + b
        return z

    def build_page_move_x(row, col, lr_shift=5):
        width  = col
        height = row
        x, y = get_xy_map(row, col)

        ### 公式推倒過程看筆記
        m = (-lr_shift - lr_shift) / width
        z = m * x + lr_shift
        return z

    move_y = build_page_move_y(row, col, top_curl=top_curl, down_curl=down_curl)
    move_x = build_page_move_x(row, col, lr_shift=lr_shift)
    move_map = np.dstack((move_x, move_y))
    return move_map


def distort_more_like_page(dst_dir, start_index, row, col, write_npy=True):
    move_maps = []
    start_time = time.time()
    # Check_dir_exist_and_build(data_access_path + dst_dir + "/"+"distorted_mesh_visuals")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "move_maps")
    print(data_access_path + dst_dir + "/" + "move_maps")
    # Check_dir_exist_and_build(data_access_path + dst_dir + "/"+"distorte_infos")

    ### 固定的參數
    down_curl   = 19
    lr_shift    = 0
    ratio_row   = 1.00
    ratio_col_d = 1.00
    index = start_index

    ### 變化的參數
    for go_top_curl in range(11, 30 + 1):
        for go_ratio_col_t in range(76, (85 + 1)):
            dis_start_time = time.time()
            go_ratio_col_t = go_ratio_col_t * 0.01

            page_move  = build_page_move_map       (row, col, top_curl=go_top_curl      , down_curl=down_curl, lr_shift=lr_shift)        ### 建立 page_move
            persp_move = build_perspective_move_map(row, col, ratio_col_t=go_ratio_col_t, ratio_row=ratio_row, ratio_col_d=ratio_col_d)  ### 建立 perspective_move
            combine_move = page_move + persp_move   ### 兩種 move 加起來
            combine_move = combine_move.astype(np.float32)
            if(write_npy) : np.save(data_access_path + dst_dir + "/" + "move_maps/%06i" % index, combine_move)  ### 把move_map存起來，記得要轉成float32！
            print("%06i process 1 mesh cost time:" % index, "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time))
            index += 1
            move_maps.append(combine_move)
    return np.array(move_maps.astype(np.float32))

def distort_just_page(dst_dir, start_index, row, col, repeat=5, write_npy=True):
    move_maps = []
    start_time = time.time()
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "move_maps")
    index = start_index
    for _ in range(repeat):
        for go_page_curl in range(1, 1 + 55):  ### go_page_curl 最小要1才行喔！
            dis_start_time = time.time()
            page_move = build_page_move_map(row, col, top_curl=go_page_curl, down_curl=go_page_curl, lr_shift=0)  ### 建立 page_move
            page_move = page_move.astype(np.float32)
            if(write_npy) : np.save(data_access_path + dst_dir + "/" + "move_maps/%06i"%(index), page_move)       ### 把move_map存起來，記得要轉成float32！
            print("%06i process 1 mesh cost time:" % index, "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time))
            index += 1
            move_maps.append(page_move)
    return np.array(move_maps, dtype=np.float32)

def distort_just_perspect(dst_dir, start_index, row, col, write_npy=True):
    move_maps = []
    start_time = time.time()
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "move_maps")

    step_amount = 9
    go_ratio_col_t  = np.linspace(0.7, 1, num=step_amount)
    go_ratio_row    = np.linspace(  1, 1, num=step_amount)
    for go_step in range(step_amount):
        dis_start_time = time.time()
        pers_move = build_perspective_move_map(row, col, ratio_col_t=go_ratio_col_t[go_step], ratio_row=go_ratio_row[go_step], ratio_col_d=1)  ### 建立 perspective_move
        pers_move = pers_move.astype(np.float32)
        if(write_npy) : np.save(data_access_path + dst_dir + "/" + "move_maps/%06i" % (start_index + go_step), pers_move)   ### 把move_map存起來，記得要轉成float32！
        print("%06i process 1 mesh cost time:" % (start_index + go_step), "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time))
        move_maps.append(pers_move)
    return np.array(move_maps, dtype=np.float32)


if(__name__ == "__main__"):
    row = 384  #360
    col = 256  #270
    # distort_more_like_page("step2_build_flow_h=384,w=256_page_more_like", start_index=2000, row=row, col=col)
    # distort_just_page("step2_build_flow_h=384,w=256_page", start_index=900, row=384, col=256, repeat=5) ### repeat是為了要讓 同種style 有repeat種 頁面內容
    distort_just_perspect("step2_build_flow_h=384,w=256_prep", start_index=0, row=384, col=256)



    ### 給spyder看變數內容
    # page_move  = build_page_move_map       (row, col, top_curl=10, down_curl=19, lr_shift=0) ### 建立 page_move
    # page_move_x = page_move[...,0]
    # page_move_y = page_move[...,1]
    # persp_move = build_perspective_move_map(row, col, ratio_col_t=0.75, ratio_row=1.00, ratio_col_d=1.00) ### 建立 perspective_move
    # persp_move_x = persp_move[...,0]
    # persp_move_y = persp_move[...,1]
    # combine_move = page_move+ persp_move   ### 兩種 move 加起來
    # combine_move_x = combine_move[...,0]
    # combine_move_y = combine_move[...,1]

    # Show_move_map_apply(combine_move)      ### 看一下 move_apply 起來長啥樣子

    ### 用真的影像show apply 並 秀出來看看
    # import cv2
    # ord_img = cv2.imread("ord_img.jpg")
    # page_img, _ = apply_move(ord_img, page_move)     ### 純頁面
    # pers_img, _ = apply_move(ord_img, persp_move)    ### 純透射
    # comb_img, _ = apply_move(ord_img, combine_move)  ### 頁面+透射
    # cv2.imshow("page_img.bmp", page_img)
    # cv2.imshow("pers_img.bmp", pers_img)
    # cv2.imshow("comb_img.bmp", comb_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
