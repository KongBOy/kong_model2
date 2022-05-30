import sys
sys.path.append("kong_util")
from step0_access_path import Data_Access_Dir
from kong_util.util import get_dir_moves, get_max_db_move_xy_from_numpy, get_dir_certain_moves, get_dir_certain_imgs, method2, get_max_db_move_xy_from_certain_move
import numpy as np
import cv2
from kong_util.util import get_xy_map
import time

def apply_move_to_rec(dis_img, move_map, max_db_move_x, max_db_move_y):
    dis_row, dis_col = dis_img.shape[:2]  ### 原始的 5xx, 5xx，不是5125, 512喔！
    row, col = move_map.shape[:2]         ### 256, 256

    rec_img = np.zeros(shape=(row, col, 3))  ### 建立存恢復影像的畫布，256,256

    ### 拿 move_map 來恢復 dis_img
    for go_row in range(row):
        for go_col in range(col):
            x = int(go_col + move_map[go_row, go_col, 0] + max_db_move_x)  ### 設定 rec_img的go_col 去dis_img 的哪裡抓，去你 原始影像 apply move_map 後的位置抓
            y = int(go_row + move_map[go_row, go_col, 1] + max_db_move_y)  ### 設定 rec_img的go_row 去dis_img 的哪裡抓，去你 原始影像 apply move_map 後的位置抓
            if(y >= 0 and y < dis_row and x >= 0 and x < dis_col):         ### 要注意可能 設定x,y 回去dis_img時，x,y 可能會因為train的不好 或 本身test扭曲就過大??(會嗎，有空想想) 超出dis_img，超出去就不抓值囉~~
                rec_img[go_row, go_col] = dis_img[y, x]
    return rec_img


def apply_move_to_rec2(dis_img, move_map, max_db_move_x, max_db_move_y):
    # start_time = time.time()
    dis_row, dis_col = dis_img.shape[:2]  ### 原始的 5xx, 5xx，不是512, 512喔！
    row, col = move_map.shape[:2]         ### 256, 256

    rec_img = np.zeros(shape=(row, col, 3), dtype=np.uint8)  ### 建立存恢復影像的畫布，256,256
    go_x, go_y = get_xy_map(row, col)
    proc_move_map = move_map.copy()
    proc_move_map[..., 0] += go_x + max_db_move_x
    proc_move_map[..., 1] += go_y + max_db_move_y
    proc_move_map = proc_move_map.astype(np.int32)


    over_bound_msk = np.zeros(shape=(row, col), dtype=np.uint8)     ### 用一個mask紀錄哪些pixel超出邊界
    over_bound_msk[proc_move_map[..., 0] >= dis_col     ] = 1     ### 只需=1即可，覺得不需要+=1，因為不需要 區分 x超過、y超過 還是 xy都超過，只要超過就算超過
    over_bound_msk[proc_move_map[..., 0] <  dis_col * -1] = 1
    over_bound_msk[proc_move_map[..., 1] >= dis_row     ] = 1
    over_bound_msk[proc_move_map[..., 1] <  dis_col * -1] = 1

    ### 修整move_map
    proc_move_map[proc_move_map[..., 0] >= dis_col      , 0] = dis_col - 1  ### 太大超出邊界，設定成邊界
    proc_move_map[proc_move_map[..., 1] >= dis_row      , 1] = dis_row - 1  ### 太大超出邊界，設定成邊界
    proc_move_map[proc_move_map[..., 0] <  dis_col * -1 , 0] = 0            ### 太小超出篇界，設定成0
    proc_move_map[proc_move_map[..., 1] <  dis_col * -1 , 1] = 0            ### 太小超出篇界，設定成0
    # np.save("proc_move_map_clip",proc_move_map)
    # cv2.imwrite("dis_img.bmp",dis_img)


    rec_img = dis_img[proc_move_map[..., 1], proc_move_map[..., 0], :]   ### 用修整後的move_map回復影像


    ### 如果 move_map 本身的值已經爆掉了，那rec_img的該點就填0～ 這樣做沒問題！注意我現在是用 move_map 回去 dis_img找顏色填回來，
    # 所以 rec_img的每個pixel 只有一個相應的 move_map pixel，
    # 代表rec_img的每個pixel只會有一個來源，不會同時有 兩個來源，
    # 所以不會有一個合法一個不合法，然後不合法蓋過原本合法值的問題！

    rec_img[over_bound_msk == 1] = 0  ### predict出的move_map超出邊界的pixel設0
    # rec_img[ proc_move_map[...,0]<0 ] = 0
    # rec_img[ proc_move_map[...,0]>dis_col ] = 0
    # rec_img[ proc_move_map[...,1]<0 ] = 0
    # rec_img[ proc_move_map[...,1]>dis_row ] = 0

    # print("cost time:",time.time()-start_time)
    return rec_img



def fun(dis_img, move_map):
    dis_row, dis_col = dis_img.shape[:2]  ### 原始的 5xx, 5xx，不是512, 512喔！
    row, col = move_map.shape[:2]         ### 256, 256

    over_bound_msk = np.zeros(shape=(row, col), dtype=np.uint8)  ### 用一個mask紀錄哪些pixel超出邊界
    over_bound_msk[move_map[..., 0] >= dis_col     ] = 1     ### 只需=1即可，覺得不需要+=1，因為不需要 區分 x超過、y超過 還是 xy都超過，只要超過就算超過
    over_bound_msk[move_map[..., 0] <  dis_col * -1] = 1
    over_bound_msk[move_map[..., 1] >= dis_row     ] = 1
    over_bound_msk[move_map[..., 1] <  dis_col * -1] = 1

    ### 修整move_map
    move_map[move_map[..., 0] >= dis_col     , 0] = dis_col - 1  ### 太大超出邊界，設定成邊界
    move_map[move_map[..., 1] >= dis_row     , 1] = dis_row - 1  ### 太大超出邊界，設定成邊界
    move_map[move_map[..., 0] <  dis_col * -1, 0] = 0            ### 太小超出篇界，設定成0
    move_map[move_map[..., 1] <  dis_col * -1, 1] = 0            ### 太小超出篇界，設定成0


    rec_img = dis_img[move_map[..., 1], move_map[..., 0], :]  ### 用修整後的move_map回復影像

    rec_img[over_bound_msk == 1] = 0  ### predict出的move_map超出邊界的pixel設0
    # rec_img[ move_map[...,0]<0 ]        = 0
    # rec_img[ move_map[...,0]>=dis_col ] = 0
    # rec_img[ move_map[...,1]<0 ]        = 0
    # rec_img[ move_map[...,1]>=dis_row ] = 0
    return rec_img[np.newaxis, ...]


import tensorflow as tf
@tf.function
def apply_move_to_rec_tf(dis_img, move_map, max_db_move_x, max_db_move_y):
    max_db_move_x = tf.convert_to_tensor(max_db_move_x, dtype=tf.float32)
    max_db_move_y = tf.convert_to_tensor(max_db_move_y, dtype=tf.float32)
    # move_map   = tf.convert_to_tensor(move_map,dtype=tf.float32)

    row, col = move_map.shape[:2]  ### 256, 256
    go_x, go_y = get_xy_map(row, col)


    ### 注意，無法用 move_map[...,0]=... 的寫法，好像是 不能 單格指定值給 tensor裡的元素， 只能 整個tensor一起給值，
    # 所以才分成下三行 造出 整體要操作的tensor 再一起操作喔！
    x_move = move_map[..., 0] + go_x + max_db_move_x
    y_move = move_map[..., 1] + go_y + max_db_move_y
    xy_move = tf.stack((x_move, y_move), axis=2)
    move_map += xy_move  ### 整包加進去～
    move_map = tf.cast(move_map, tf.int32)


    ### tensor沒有辦法 像numpy 一樣靈活 用 array[ [1,3,5,...] ] 這種用法，所以要用 tf.numpy_function，把tensor轉成 numpy處理囉！
    rec_img = tf.numpy_function(fun, [dis_img, move_map], tf.float32)

    # print(type(dis_img),dis_img.dtype)
    # print(type(move_map),move_map.dtype)
    # print(type(max_db_move_x),max_db_move_x.dtype)
    # print(type(max_db_move_y),max_db_move_y.dtype)
    # print(type(dis_img),dis_img.dtype)
    # print(type(move_map),move_map.dtype)
    # print(type(row))
    # print(type(col))
    # print(type(go_x),go_x.dtype)
    # print(type(go_y),go_y.dtype)
    # print(type(x_move),y_move.dtype)
    # print(type(y_move),y_move.dtype)
    return rec_img
    # return go_x  ### debug用，隨便傳個東西回去



if(__name__ == "__main__"):
    # Data_Access_Dir = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    ### 拿到 dis_img
    dis_imgs = get_dir_certain_imgs(Data_Access_Dir + "step3_apply_flow_h=384,w=256_complex+page", "3a1-I1-patch")
    dis_img = dis_imgs[0]
    ### 拿到 move_map
    moves = get_dir_certain_moves(Data_Access_Dir + "step3_apply_flow_h=384,w=256_complex+page", "2-q")
    move_map = moves[0]
    proc_move_map = move_map.copy()
    ### 拿到 當初建 dis_img_db時 用的 move_map max/min 的移動量
    max_db_move_x, max_db_move_y = get_max_db_move_xy_from_certain_move(Data_Access_Dir + "step3_apply_flow_h=384,w=256_complex+page", "2-q")

    ### 拿 dis_img 配 move_map 來做 rec囉！
    ### 有用tensorflow做rec_img
    print(dis_img.shape)
    print(proc_move_map.shape)
    print(max_db_move_x)
    print(max_db_move_y)
    rec_img = apply_move_to_rec_tf(dis_img, proc_move_map, max_db_move_x, max_db_move_y)
    cv2.imshow(Data_Access_Dir + "rec_img", rec_img.numpy().astype(np.uint8))

    ### 不用tensorflow做rec_img
    # rec_img = apply_move_to_rec2(dis_img, proc_move_map, max_db_move_x, max_db_move_y)
    # cv2.imshow(Data_Access_Dir+"rec_img", rec_img.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()



    # start_time = time.time()
    # for i in range(100):
    #     rec_img = apply_move_to_rec(dis_img, proc_move_map, max_db_move_x, max_db_move_y)
    # print("apply_move_to_rec1 cost_time:",time.time()-start_time)

    # start_time = time.time()
    # for i in range(1000):
    #     rec_img = apply_move_to_rec2(dis_img, proc_move_map, max_db_move_x, max_db_move_y)
    # print("apply_move_to_rec2 cost_time:",time.time()-start_time)

    # start_time = time.time()
    # for i in range(1000):
    #     rec_img = apply_move_to_rec_tf(dis_img, proc_move_map, max_db_move_x, max_db_move_y)
    # print("apply_move_to_rec_tf cost_time:",time.time()-start_time)




    ### 原來要用 inverse flow 的版本還是留一下好了
    # row, col = img.shape[:2]

    # rec_img = np.zeros_like(img)

    # for go_row in range(row):
    #     for go_col in range(col):
    #         x = int(go_col + move_map[go_row, go_col, 0] + move_x_min)
    #         y = int(go_row + move_map[go_row, go_col, 1] + move_y_min)

    #         rec_img[go_row, go_col,:] = dis_img[y, x,:]
    # cv2.imshow("rec_img", rec_img)
    # cv2.imwrite(Data_Access_Dir+"rec_img.png", rec_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
