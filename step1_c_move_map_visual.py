import os
import numpy as np 
import matplotlib.pyplot as plt
import cv2

    
def get_db_all_move(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ".npy" in file_name]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list)
    return move_map_list

def find_db_max_move(ord_dir):
    move_map_list = get_db_all_move(ord_dir)
    max_move = np.absolute(move_map_list).max()
    print("max_move:",max_move)
    return max_move



### 方法1：感覺可以！但缺點是沒辦法用cv2，而一定要搭配matplot的imshow來自動填色
def method1(x, y, max_value=-10000): ### 這個 max_value的值 意義上來說要是整個db內位移最大值喔！這樣子出來的圖的顏色強度才會準確
    h, w = x.shape[:2]
    z = np.ones(shape=(h, w))
    visual_map = np.dstack( (x,y) )                  ### step1.
    if(max_value==-10000):                           ### step2.確定max_value值，沒有指定 max_value的話，就用資料自己本身的
        max_value = visual_map.max()
    visual_map = ((visual_map/max_value)+1)/2        ### step3.先把值弄到 0~1
    visual_map = np.dstack( (visual_map, z))         ### step4.再concat channel3，來給imshow自動決定顏色
#    plt.imshow(visual_map)
    return visual_map

### 方法2：用hsv，感覺可以！
def method2(x, y, color_shift=5):       ### 最大位移量不可以超過 255，要不然顏色強度會不準，不過實際用了map來顯示發現通常值都不大，所以還加個color_shift喔~
    h, w = x.shape[:2]                  ### 影像寬高
    fx, fy = x, y                       ### u是x方向怎麼移動，v是y方向怎麼移動
    ang = np.arctan2(fy, fx) + np.pi    ### 得到運動的角度
    val = np.sqrt(fx*fx+fy*fy)          ### 得到運動的位移長度
    hsv = np.zeros((h, w, 3), np.uint8) ### 初始化一個canvas
    hsv[...,0] = ang*(180/np.pi/2)      ### B channel為 角度訊息的顏色
    hsv[...,1] = 255                    ### G channel為 255飽和度
    hsv[...,2] = np.minimum(val*color_shift, 255)   ### R channel為 位移 和 255中較小值来表示亮度，因為值最大為255，val的除4拿掉就ok了！
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) ### 把得到的HSV模型轉換為BGR顯示
    if(True):
        white_back = np.ones((h, w, 3),np.uint8)*255
        white_back[...,0] -= hsv[...,2]
        white_back[...,1] -= hsv[...,2]
        white_back[...,2] -= hsv[...,2]
    #        cv2.imshow("white_back",white_back)
        bgr += white_back
    return bgr


def get_reference_map(ord_dir,color_shift=5): ### 根據你的db內 最大最小值 產生 參考流的map
    max_move = find_db_max_move(ord_dir)
    visual_row = 512
    visual_col = visual_row
    x = np.linspace(-max_move,max_move,visual_col)
    x = np.tile(x, (visual_row,1))
    y = x.T

    map1 = method1(x, y, max_value=max_move)
    map2 = method2(x, y, color_shift=color_shift)
    return map1, map2, x, y
    
    
if(__name__=="__main__"):
    ### rgb bgr 之類的細節還沒確認，先跳過快來不及趕meeting的東西了
    ord_dir = "step1_result/move_map"
    move_map_list = get_db_all_move(ord_dir)


    ### 以下是存 reference_map
    map1, map2, x, y = get_reference_map(ord_dir,color_shift=1)
    cv2.imwrite("map1.jpg",map1.astype(np.uint8))  ### cv2用方法一存完會變黑色的，因為方法一的顏色是plt自動填的！
    cv2.imwrite("map2.png",map2)                   ### cv2存沒問題！
    
    plt.imshow(map1) ### plt存沒問題
    plt.figure()
    plt.imshow(map2) ### plt存沒問題
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
#    bgr = method2(test_move[...,0], test_move[...,1], color_shift=5)
#    plt.imshow(bgr)
#    plt.show()
    
    #cv2.imshow("hsv_map_max=176.6609.bmp",bgr)
    #cv2.waitKey(0)

    ### 方法3：不大行
    #plt.scatter(x=x.ravel(), y=y.ravel(),c = np.arange(visual_row**2), cmap="hsv")