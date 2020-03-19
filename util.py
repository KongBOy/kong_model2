from step0_access_path import access_path
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

def check_img_file_name(file_name):
    file_name = file_name.lower()
    if(".bmp" in file_name or ".jpg" in file_name or ".jpeg" in file_name or ".png" in file_name ):return True
    else: return False


def get_dir_certain_file_name(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (certain_word in file_name)]
    return file_names

def get_dir_certain_img(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if check_img_file_name(file_name) and (certain_word in file_name) ]
    img_list = []
    for file_name in file_names:
        img_list.append( cv2.imread(ord_dir + "/" + file_name) )
    img_list = np.array(img_list, dtype=np.float32)
    return img_list

def get_dir_certain_move(ord_dir, certain_word):
    file_names = [file_name for file_name in os.listdir(ord_dir) if (".npy" in file_name) and (certain_word in file_name)]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list, dtype=np.float32)
    return move_map_list

def get_dir_img(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if check_img_file_name(file_name) ]
    img_list = []
    for file_name in file_names:
        img_list.append( cv2.imread(ord_dir + "/" + file_name) )
    img_list = np.array(img_list, dtype=np.uint8)
    return img_list


def get_dir_move(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if ".npy" in file_name]
    move_map_list = []
    for file_name in file_names:
        move_map_list.append( np.load(ord_dir + "/" + file_name) )
    move_map_list = np.array(move_map_list, dtype=np.float32)
    return move_map_list

def get_db_amount(ord_dir):
    file_names = [file_name for file_name in os.listdir(ord_dir) if check_img_file_name(file_name) or (".npy" in file_name) ]
    return len(file_names)

##########################################################
def apply_move_map_boundary_mask(move_maps):
    boundary_width = 20 
    _, row, col = move_maps.shape[:3]
    move_maps[:, boundary_width:row-boundary_width,boundary_width:col-boundary_width,:] = 0
    return move_maps

def get_max_db_move_xy_from_numpy(move_maps): ### 注意這裡的 max/min 是找位移最大，不管正負號！ 跟 normalize 用的max/min 不一樣喔！ 
    move_maps = abs(move_maps)
    print("move_maps.shape",move_maps.shape)
    # move_maps = apply_move_map_boundary_mask(move_maps) ### 目前的dataset還是沒有只看邊邊，有空再用它來產生db，雖然實驗過有沒有用差不多(因為1019位移邊邊很大)
    max_move_x = move_maps[:,:,:,0].max()
    max_move_y = move_maps[:,:,:,1].max()
    return max_move_x, max_move_y

def get_max_db_move_xy_from_dir(ord_dir):
    move_maps = get_dir_move(ord_dir)
    return get_max_db_move_xy_from_numpy(move_maps)

def get_max_db_move_xy_from_certain_move(ord_dir, certain_word):
    move_maps = get_dir_certain_move(ord_dir, certain_word)
    return get_max_db_move_xy_from_numpy(move_maps)


def get_max_db_move_xy(db_dir="datasets", db_name="1_unet_page_h=384,w=256"):
    move_map_train_path = db_dir + "/" + db_name + "/" + "train/move_maps" 
    move_map_test_path  = db_dir + "/" + db_name + "/" + "test/move_maps" 
    train_move_maps = get_dir_move(move_map_train_path) # (1800, 384, 256, 2)
    test_move_maps  = get_dir_move(move_map_test_path)  # (200, 384, 256, 2)
    db_move_maps = np.concatenate((train_move_maps, test_move_maps), axis=0) # (2000, 384, 256, 2)

    max_move_x = db_move_maps[:,:,:,0].max()
    max_move_y = db_move_maps[:,:,:,1].max()
    return max_move_x, max_move_y

#######################################################
### 複刻 step6_data_pipline.py 寫的 get_train_test_move_map_db 
def get_maxmin_train_move_from_path(move_map_train_path):
    train_move_maps = get_dir_move(move_map_train_path)
    max_train_move = train_move_maps.max() ###  236.52951204508076
    min_train_move = train_move_maps.min() ### -227.09562801056995
    return max_train_move, min_train_move

def get_maxmin_train_move(db_dir="datasets", db_name="1_unet_page_h=384,w=256"):
    move_map_train_path = db_dir + "/" + db_name + "/" + "train/move_maps" 
    train_move_maps = get_dir_move(move_map_train_path)
    max_train_move = train_move_maps.max() ###  236.52951204508076
    min_train_move = train_move_maps.min() ### -227.09562801056995
    return max_train_move, min_train_move

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
def predict_unet_move_maps_back(predict_move_maps):
    train_move_maps = get_dir_move(access_path+"datasets/pad2000-512to256/train/move_maps")
    max_train_move = train_move_maps.max()
    min_train_move = train_move_maps.min()
    predict_back_list = []
    for predict_move_map in predict_move_maps:
        predict_back = (predict_move_map[0]+1)/2 * (max_train_move-min_train_move) + min_train_move ### 把 -1~1 轉回原始的值域
        predict_back_list.append(predict_back)
    return np.array(predict_back_list, dtype=np.float32)



#######################################################

import matplotlib.pyplot as plt
def use_plt_show_move(move, color_shift=1):
    move_bgr = method2(move[:,:,0], move[:,:,1], color_shift=color_shift)
    move_rgb = move_bgr[:,:,::-1]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(move_rgb) ### 這裡不會秀出來喔！只是把圖畫進ax裡面而已
    return fig, ax



def time_util(cost_time):
    hour = cost_time//3600 
    minute = cost_time%3600//60 
    second = cost_time%3600%60
    return "%02i:%02i:%02i"%(hour, minute, second)

#######################################################

if(__name__=="__main__"):
    # in_imgs = get_dir_img(access_path+"datasets/wei_book/in_imgs")
    # gt_imgs = get_dir_img(access_path+"datasets/wei_book/gt_imgs")
    
    # db = zip(in_imgs, gt_imgs)
    # for imgs in db:
    #     print(type(imgs))

    get_max_db_move_xy(db_dir=access_path+"datasets", db_name="1_unet_page_h=384,w=256")