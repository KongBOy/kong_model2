from step0_access_path import data_access_path
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from util import get_dir_move, get_reference_map, method1, method2

if(__name__ == "__main__"):
    # data_access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    ### rgb bgr 之類的細節還沒確認，先跳過快來不及趕meeting的東西了
    ord_dir = "step2_flow_build/move_map"
    move_map_list = get_dir_move(ord_dir)


    ### 以下是存 reference_map
    map1, map2, x, y = get_reference_map(ord_dir, color_shift=1)
    cv2.imwrite(data_access_path + "step2_flow_build/map1.jpg", map1.astype(np.uint8))  ### cv2用方法一存完會變黑色的，因為方法一的顏色是plt自動填的！
    cv2.imwrite(data_access_path + "step2_flow_build/map2.png", map2)                   ### cv2存沒問題！

    plt.imshow(map1)  ### plt存沒問題
    plt.figure()
    plt.imshow(map2)  ### plt存沒問題
    plt.show()


    ### 以下實際把 move拿出來 看最右上角 的移動 來檢查 圖的顏色 有沒有對應 相應的移動量
#    move_map = move_map_list[0].copy()
#    move_map_x=move_map[...,0]
#    move_map_y=move_map[...,1]
#
#    color_shift=1
#    fx=  -43
#    fy= -125
#    ang = np.arctan2(fy, fx) + np.pi    ### 得到運動的角度
#    val = np.sqrt(fx*fx+fy*fy)          ### 得到運動的位移長度
#    ch0 = ang*(180/np.pi/2)                 ### B channel為 角度訊息的顏色
#    ch1 = 255                               ### G channel為 255飽和度
#    ch2 = np.minimum(val*color_shift, 255)  ### R channel為 位移 和 255中較小值来表示亮度，因為值最大為255，val的除4拿掉就ok了！

    ### 注意！如果你的move是用cv2存的，那再看reference_map就要看用cv2存的ref_map喔，這樣bgr才會對，反之亦然：你的move用plt存的，那看reference map也就要看用plt存的ref_map，這樣rgb才會對！
    ### 注意！如果你的move是用cv2存的，那再看reference_map就要看用cv2存的ref_map喔，這樣bgr才會對，反之亦然：你的move用plt存的，那看reference map也就要看用plt存的ref_map，這樣rgb才會對！
    ### 注意！如果你的move是用cv2存的，那再看reference_map就要看用cv2存的ref_map喔，這樣bgr才會對，反之亦然：你的move用plt存的，那看reference map也就要看用plt存的ref_map，這樣rgb才會對！
    ### 很重要說三次ˊ口ˋ
    ### 以下是存 move_map
#    test_move = move_map_list[2]
#    visual_map = method1(test_move[...,0],test_move[...,1],max_value=238.45)
#    plt.imshow(visual_map)
#
#    plt.figure()
#    bgr = method2(test_move[...,0], test_move[...,1], color_shift=1)
#    plt.imshow(bgr)
#    plt.show()

    # cv2.imshow("hsv_map_max=176.6609.bmp",bgr)
    # cv2.waitKey(0)

    ### 方法3：不大行
    # plt.scatter(x=x.ravel(), y=y.ravel(),c = np.arange(visual_row**2), cmap="hsv")
    