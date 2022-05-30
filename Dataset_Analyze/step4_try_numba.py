from step0_access_path import Data_Access_Dir
from kong_util.util import get_dir_moves, get_max_db_move_xy_from_numpy, get_dir_certain_moves, get_dir_certain_imgs, method2, get_max_db_move_xy_from_certain_move
import numpy as np
from numba import cuda
import math
import argparse
from kong_util.util import get_xy_map

import cv2







@cuda.jit()
def kong(dis_img, move_map, rec_img):
    # rec_img = dis_img[move_map[...,1],move_map[...,0],:]
    H = move_map.shape[0] - 1
    W = move_map.shape[1] - 1

    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    rec_img = 0  ### 一直錯，先用最簡單的運算來看就好，儘管已經最簡單，比起numpy整張圖運算還是很慢！
    # for xr in range(start_x, W, stride_x):
    #     for yr in range(start_y, H, stride_y):
    #         y = move_map[yr,xr]
    #         x = move_map[yr,xr]
    #         rec_img[yr,xr] = 0
    #         i,j = iterSearchShader(move_map, xr, yr, maxIter, precision)




import time
def apply_move_to_rec2(dis_img, move_map, max_db_move_x, max_db_move_y):

    dis_row, dis_col = dis_img.shape[:2]  ### 原始的 5xx, 5xx，不是5125, 512喔！
    row, col = move_map.shape[:2]         ### 256, 256

    rec_img = np.zeros(shape=(row, col, 3), dtype=np.uint8)  ### 建立存恢復影像的畫布，256,256
    go_x, go_y = get_xy_map(row, col)
    move_map2 = move_map.copy()
    move_map2[..., 0] += go_x + max_db_move_x
    move_map2[..., 1] += go_y + max_db_move_y
    move_map2 = move_map2.astype(np.int32)


    ### 比較最核心的這一步，不用numba 花多少時間：
    ###   不用numba：10000 花8秒
    start_time = time.time()
    for i in range(10000):
        rec_img = dis_img[move_map2[..., 1], move_map2[..., 0], :]
    print("cost time:", time.time() - start_time)


    ###   用numba：10000 花79秒
    start_time = time.time()
    W = col
    H = row
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(W / threadsperblock[0])
    blockspergrid_y = math.ceil(H / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    for i in range(10000):
        kong[blockspergrid, threadsperblock](dis_img.astype(np.int32), move_map2, rec_img.astype(np.int32))
    print("cost time:", time.time() - start_time)




    ### 如果 move_map 本身的值已經爆掉了，那rec_img的該點就填0～ 這樣做沒問題！注意我現在是用 move_map 回去 dis_img找顏色填回來，
    # 所以 rec_img的每個pixel 只有一個相應的 move_map pixel，
    # 代表rec_img的每個pixel只會有一個來源，不會同時有 兩個來源，
    # 所以不會有一個合法一個不合法，然後不合法蓋過原本合法值的問題！
    # rec_img[ move_map2[...,0]<0 ] = 0
    # rec_img[ move_map2[...,0]>dis_col ] = 0
    # rec_img[ move_map2[...,1]<0 ] = 0
    # rec_img[ move_map2[...,1]>dis_row ] = 0


    return rec_img


if(__name__ == "__main__"):
    # Data_Access_Dir = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    ### 拿到 dis_img
    dis_imgs = get_dir_certain_imgs(Data_Access_Dir + "step3_apply_flow_result", "3a1-I1-patch")
    dis_img = dis_imgs[0]
    ### 拿到 move_map
    moves = get_dir_certain_moves(Data_Access_Dir + "step3_apply_flow_result", "2-q")
    move_map = moves[0]
    ### 拿到 當初建 dis_img_db時 用的 move_map max/min 的移動量
    max_db_move_x, max_db_move_y = get_max_db_move_xy_from_certain_move(Data_Access_Dir + "step3_apply_flow_result", "2-q")

    rec_img = apply_move_to_rec2(dis_img, move_map, max_db_move_x, max_db_move_y)
