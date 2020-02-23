import numpy as np  
import cv2 
import os

def get_xy_map(row, col):
    x = np.arange(col)
    x = np.tile(x,(row,1))
    
#    y = np.arange(row-1, -1, -1) ### 就是這裡要改一下拉！不要抄網路的，網路的是用scatter的方式來看(左下角(0,0)，x往右增加，y往上增加)
    y = np.arange(row) ### 改成這樣子 就是用image的方式來處理囉！(左上角(0,0)，x往右增加，y往上增加)
    y = np.tile(y,(col,1)).T
    return x, y

def get_dir_certain_file_name(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (certain_word in file_name)]
    return file_names

def get_dir_certain_img(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (".bmp" in file_name) and (certain_word in file_name) ]
    img_list = []
    for file_name in file_names:
        img_list.append( cv2.imread(ord_dir + "/" + file_name) )
    img_list = np.array(img_list)
    return img_list

def get_dir_certain_move(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (".npy" in file_name) and (certain_word in file_name)]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list)
    return move_map_list

def get_dir_img(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (".bmp" in file_name) or (".jpg" in file_name) ]
    img_list = []
    for file_name in file_names:
        img_list.append( cv2.imread(ord_dir + "/" + file_name) )
    img_list = np.array(img_list)
    return img_list


def get_dir_move(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ".npy" in file_name]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list)
    return move_map_list

#######################################################
### 用來給視覺化參考的顏色map
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

def find_db_max_move(ord_dir):
    move_map_list = get_dir_move(ord_dir)
    max_move = np.absolute(move_map_list).max()
    print("max_move:",max_move)
    return max_move

####################################################### 
### 視覺化方法1：感覺可以！但缺點是沒辦法用cv2，而一定要搭配matplot的imshow來自動填色
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

### 視覺化方法2：用hsv，感覺可以！
def method2(x, y, color_shift=1):       ### 最大位移量不可以超過 255，要不然顏色強度會不準，不過實際用了map來顯示發現通常值都不大，所以還加個color_shift喔~
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

#######################################################
