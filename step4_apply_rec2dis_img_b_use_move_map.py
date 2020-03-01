from step0_access_path import access_path
from util import get_dir_move, get_max_move_xy_from_numpy, get_dir_certain_move, get_dir_certain_img, method2, get_max_move_xy_from_certain_move
import numpy as np 
import cv2

def apply_move_to_rec(dis_img, move_map, max_move_x, max_move_y):
    dis_row, dis_col = dis_img.shape[:2] ### 原始的 5xx, 5xx，不是5125, 512喔！
    row, col = move_map.shape[:2] ### 256, 256

    rec_img = np.zeros(shape=(row,col,3)) ### 建立存恢復影像的畫布，256,256

    ### 拿 move_map 來恢復 dis_img
    for go_row in range(row):
        for go_col in range(col):
            x = int(go_col + move_map[go_row, go_col, 0] + max_move_x)  ### 設定 rec_img的go_col 去dis_img 的哪裡抓，去你 原始影像 apply move_map 後的位置抓
            y = int(go_row + move_map[go_row, go_col, 1] + max_move_y)  ### 設定 rec_img的go_row 去dis_img 的哪裡抓，去你 原始影像 apply move_map 後的位置抓
            if(y>=0 and y<dis_row and x>=0 and x<dis_col): ### 要注意可能 設定x,y 回去dis_img時，x,y 可能會因為train的不好 或 本身test扭曲就過大??(會嗎，有空想想) 超出dis_img，超出去就不抓值囉~~
                rec_img[go_row, go_col] = dis_img[y, x]
    return rec_img




if(__name__=="__main__"):
    # access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    ### 拿到 dis_img
    dis_imgs = get_dir_certain_img(access_path+"step3_apply_flow_result","3a1-I1-patch")
    dis_img = dis_imgs[0]
    ### 拿到 move_map
    moves = get_dir_certain_move(access_path+"step3_apply_flow_result","2-q")
    move_map = moves[0]
    ### 拿到 當初建 dis_img_db時 用的 move_map max/min 的移動量
    max_move_x, max_move_y = get_max_move_xy_from_certain_move(access_path+"step3_apply_flow_result","2-q")

    ### 拿 dis_img 配 move_map 來做 rec囉！
    rec_img = apply_move_to_rec(dis_img, move_map, max_move_x, max_move_y)

    # cv2.imshow(access_path+"rec_img", rec_img.astype(np.uint8))
    # cv2.imwrite(access_path+"rec_img.png", rec_img.astype(np.uint8))
    # cv2.waitKey()
    # cv2.destroyAllWindows()



    ### 原來要用 inverse flow 的版本還是留一下好了
    # row, col = img.shape[:2]

    # rec_img = np.zeros_like(img)

    # for go_row in range(row):
    #     for go_col in range(col):
    #         x = int(go_col + move_map[go_row, go_col, 0] + move_x_min) 
    #         y = int(go_row + move_map[go_row, go_col, 1] + move_y_min)  

    #         rec_img[go_row, go_col,:] = dis_img[y, x,:]
    # cv2.imshow("rec_img", rec_img)
    # cv2.imwrite(access_path+"rec_img.png", rec_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
