from step0_access_path import access_path
import tensorflow as tf
import os
import numpy as np
import cv2
from util import get_dir_img, get_dir_move, use_plt_show_move, get_db_amount

import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

### 把img_db 包成class 是因為 tf.data.Dataset().map(f)的這個f，沒有辦法丟參數壓！所以只好包成class，把要傳的參數當 data_member囉！ 另一方面是好管理、好閱讀～
class img_db():
    def __init__(self,img_path, img_type, resize_shape, batch_size):
        self.img_path = img_path 
        self.img_type = img_type
        self.resize_shape = resize_shape 
        self.batch_size = batch_size 

        self.img_db = None ### 原始讀進來的影像db
        self.pre_db = None ### 進 generator 之前 做resize 和 值變-1~1的處理 後的db
        # self.img = None

        self.get_db_from_file_name(img_path, batch_size)
        
    ####################################################################################################
    ### 以下是 bmp file_name -> tensor  成功！
    ### 這種寫法是 img 沒有用 self.img 來寫，比較 能夠顯現 img傳遞過程的概念，先用這個好了，想看上一種風格去git 6b63a99 調喔
    def step1_load_one_bmp_img(self, file_name):
        img = tf.io.read_file(file_name)
        img = tf.image.decode_bmp(img)
        img  = tf.cast(img, tf.float32)
        return img

    def step1_load_one_jpg_img(self, file_name):
        img = tf.io.read_file(file_name)
        img = tf.image.decode_jpeg(img)
        img  = tf.cast(img, tf.float32)
        return img

    def step2_resize(self, img):
        img = tf.image.resize(img ,self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        return img


    def step3_normalize(self, img): ### 因為用tanh，所以把值弄到 [-1, 1]
        img = (img / 127.5) - 1
        return img


    def load_bmp(self, file_name): 
        img = self.step1_load_one_bmp_img(file_name)  ### 根據檔名，把圖片讀近來
        img  = tf.cast(img, tf.uint8)  ### 不會拿來訓練，是拿來顯示的，所以轉乘uint8
        return img
    def load_bmp_resize_normalize(self, file_name):
        img = self.step1_load_one_bmp_img(file_name)  ### 根據檔名，把圖片讀近來
        img = self.step2_resize(img)
        img = self.step3_normalize(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 


    def load_jpg(self, file_name): 
        img = self.step1_load_one_jpg_img(file_name)  ### 根據檔名，把圖片讀近來
        img = tf.cast(img, tf.uint8)  ### 不會拿來訓練，是拿來顯示的，所以轉乘uint8
        return img
    def load_jpg_resize_normalize(self, file_name):
        img = self.step1_load_one_jpg_img(file_name)  ### 根據檔名，把圖片讀近來
        img = self.step2_resize(img)
        img = self.step3_normalize(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 

    def get_db_from_file_name(self, img_path, batch_size):
        self.img_db = tf.data.Dataset.list_files(img_path + "/" + "*." + self.img_type, shuffle=False)
        self.pre_db = tf.data.Dataset.list_files(img_path + "/" + "*." + self.img_type, shuffle=False)
        if  (self.img_type=="bmp"):
            self.img_db = self.img_db.map(self.load_bmp)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
            self.pre_db = self.pre_db.map(self.load_bmp_resize_normalize)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        elif(self.img_type=="jpg"):
            self.img_db = self.img_db.map(self.load_jpg)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
            self.pre_db = self.pre_db.map(self.load_jpg_resize_normalize)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        self.img_db = self.img_db.batch(batch_size)
        self.pre_db = self.pre_db.batch(batch_size)
        # self.img_db = self.img_db.prefetch(tf.data.experimental.AUTOTUNE)

########################################################################################################################################
### 以下是 numpy 直接整包load進記憶體，因為 file_name -> tensor失敗，
### 因為tf2不支援decode numpy檔案(只支援從記憶體load進去)，tf2 可以decode的檔案格式：text, imgs, csv, TFRecord，有空再試TFRecord

### get 和 resize 和 norm 拆開funtcion寫~~~ 因為直接get完直接norm，在外面就得不到原始 max min 了
# def get_move_map_db_and_resize(ord_dir, resize_shape=(256,256)):
#     import time
#     start_time = time.time()
#     move_maps = get_dir_move(ord_dir)
#     print("before resize shape", move_maps.shape)
#     move_map_resize_list = []
#     for move in move_maps[:]:
#         move_resize = cv2.resize( move, resize_shape, interpolation = cv2.INTER_NEAREST)
#         move_map_resize_list.append(move_resize)
#     move_map_resize_list = np.array(move_map_resize_list)
#     print("after resize shape", move_map_resize_list.shape)
#     print("get_move_map_db_and_resize cost time", time.time()-start_time)
#     return move_map_resize_list

def use_maxmin_train_move_to_normto_norm(move_maps): ### 給 train用
    max_train_move = move_maps.max() ###  236.52951204508076
    min_train_move = move_maps.min() ### -227.09562801056995
    move_maps = ((move_maps-min_train_move)/(max_train_move-min_train_move))*2-1
    return move_maps, max_train_move, min_train_move ### max_train_move, min_train_move 需要回傳回去喔，因為要給 test 用

def use_train_move_value_to_norm(move_maps, max_train_move, min_train_move): ### 給test來用，要用和train一樣的 max_train_move, min_train_move
    move_maps = ((move_maps-min_train_move)/(max_train_move-min_train_move))*2-1
    return move_maps


def get_train_test_move_map_db(db_dir, db_name, batch_size):
    move_map_train_path = db_dir + "/" + db_name + "/" + "train/move_maps" 
    train_move_map_db_ord = get_dir_move(move_map_train_path) #get_move_map_db_and_resize(move_map_train_path, resize_shape=resize_shape)
    train_move_map_db_norm, \
        max_train_move, min_train_move = use_maxmin_train_move_to_normto_norm(train_move_map_db_ord) ### 這裡會得到 max/min_train_move
    train_move_map_db_norm = tf.data.Dataset.from_tensor_slices(train_move_map_db_norm)
    train_move_map_db_norm = train_move_map_db_norm.batch(batch_size)
    # train_move_map_db = train_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)

    move_map_test_path = db_dir + "/" + db_name + "/" + "test/move_maps" 
    test_move_map_db_ord = get_dir_move(move_map_test_path) #get_move_map_db_and_resize(move_map_test_path, resize_shape=resize_shape)
    test_move_map_db_norm = use_train_move_value_to_norm(test_move_map_db_ord, max_train_move, min_train_move) ### 這裡要用 max/min_train_move 來對 test_move_map_db 做 norm
    test_move_map_db_norm = tf.data.Dataset.from_tensor_slices(test_move_map_db_norm)
    test_move_map_db_norm = test_move_map_db_norm.batch(batch_size)
    # test_move_map_db = test_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)

    return train_move_map_db_norm, max_train_move, min_train_move, test_move_map_db_norm, train_move_map_db_ord, test_move_map_db_ord

def get_1_pure_unet_db(db_dir, db_name, img_type="bmp", batch_size=1, img_resize=(512,512)):#, move_resize=(256,256)): 
    ### 建db的順序：input, input, output(gt), output(gt)，跟 get_rect2_dataset不一樣喔別混亂了！
    ### 拿到 dis_imgs_db 的 train dataset，從 檔名 → tensor
    data_dict = {}
    train_in_img_db_path = db_dir + "/" + db_name + "/" + "train/dis_imgs" 
    train_in_db = img_db(train_in_img_db_path, img_type, img_resize, 1)
    data_dict["train_in_db"]     = train_in_db.img_db
    data_dict["train_in_db_pre"] = train_in_db.pre_db
    
    ### 拿到 dis_imgs_db 的 test dataset，從 檔名 → tensor
    test_in_img_db_path = db_dir + "/" + db_name + "/" + "test/dis_imgs" 
    test_in_db = img_db(test_in_img_db_path, img_type, img_resize, 1)
    data_dict["test_in_db"]      = test_in_db.img_db
    data_dict["test_in_db_pre"]  = test_in_db.pre_db

    ### 因為 test_move_map 需要 train_move_map 的 max/min_move 來把值弄到 -1~1，所以包成一個function來拿比較不會漏掉！
    ### 拿到 move_map 的 train dataset，從 直切先全部讀出來成npy → tensor
    ### 拿到 move_map 的 test dataset，從 直切先全部讀出來成npy → tensor
    train_move_map_db_norm, \
    max_train_move, min_train_move, \
    test_move_map_db_norm, \
    train_move_map_db_ord, test_move_map_db_ord = get_train_test_move_map_db(db_dir, db_name, 1)

    data_dict["train_gt_db"    ] = train_move_map_db_ord
    data_dict["train_gt_db_pre"] = train_move_map_db_norm
    data_dict["test_gt_db"    ]  = test_move_map_db_ord
    data_dict["test_gt_db_pre"]  = test_move_map_db_norm

    data_dict["max_train_move"]  = max_train_move
    data_dict["min_train_move"]  = min_train_move

    data_dict["train_amount"]    = get_db_amount(train_in_img_db_path)
    data_dict["test_amount" ]    = get_db_amount(test_in_img_db_path )

    data_dict["in_type"] = "img"
    data_dict["gt_type"] = "move_map"


    ##########################################################################################################################################
    ### 勿刪！用來測試寫得對不對！
    # import matplotlib.pyplot as plt 
    # from util import method2

    # take_num = 5
    
    # for i, (img, img_pre, move) in enumerate(zip(data_dict["train_in_db"].take(take_num), data_dict["train_in_db_pre"].take(take_num), data_dict["train_gt_db_pre"].take(take_num))):     ### 想看test 的部分用這行 且 註解掉上行
    #     print("i",i)
    #     fig, ax = plt.subplots(1,3)
    #     ax_i = 0
    #     img = tf.cast(img[0], tf.uint8)
    #     ax[ax_i].imshow(img)
    #     print(img.numpy().dtype)
    #     ax_i += 1

    #     img_pre_back = (img_pre[0]+1.)*127.5
    #     img_pre_back = tf.cast(img_pre_back, tf.int32)
    #     ax[ax_i].imshow(img_pre_back)
    #     ax_i += 1

    #     # move_back = (move[0]+1)/2 * (max_train_move-min_train_move) + min_train_move  ### 想看train的部分用這行 且 註解掉下行
    #     move_back = (move[0]+1)/2 * (max_train_move-min_train_move) + min_train_move       ### 想看test 的部分用這行 且 註解掉上行
    #     move_bgr = method2(move_back[...,0], move_back[...,1],1)
    #     ax[ax_i].imshow(move_bgr)
    #     plt.show()
    #     plt.close()
    ##########################################################################################################################################
    return data_dict 

def get_in_img_and_gt_img_db(db_dir, db_name, in_dir_name, gt_dir_name,  img_type="bmp", batch_size=1, img_resize=(512,512)):
    ### 建db的順序：input, output(gt), input , output(gt)，跟 get_1_pure_unet_db不一樣喔別混亂了！
    train_in_img_db_path = db_dir + "/" + db_name + "/" + "train"+"/"+in_dir_name
    train_gt_img_db_path = db_dir + "/" + db_name + "/" + "train"+"/"+gt_dir_name
    test_in_img_db_path  = db_dir + "/" + db_name + "/" + "test"+"/" +in_dir_name
    test_gt_img_db_path  = db_dir + "/" + db_name + "/" + "test"+"/" +gt_dir_name
    
    data_dict = {}
    train_in_db = img_db(train_in_img_db_path, img_type, img_resize, 1)
    train_gt_db = img_db(train_gt_img_db_path, img_type, img_resize, 1)
    test_in_db  = img_db(test_in_img_db_path , img_type, img_resize, 1)
    test_gt_db  = img_db(test_gt_img_db_path , img_type, img_resize, 1)

    data_dict["train_in_db"]     = train_in_db.img_db
    data_dict["train_in_db_pre"] = train_in_db.pre_db
    data_dict["train_gt_db"]     = train_gt_db.img_db
    data_dict["train_gt_db_pre"] = train_gt_db.pre_db
    data_dict["test_in_db"]     = test_in_db.img_db
    data_dict["test_in_db_pre"] = test_in_db.pre_db
    data_dict["test_gt_db"]     = test_gt_db.img_db
    data_dict["test_gt_db_pre"] = test_gt_db.pre_db

    data_dict["train_amount"]    = get_db_amount(train_in_img_db_path)
    data_dict["test_amount" ]    = get_db_amount(test_in_img_db_path )
    data_dict["in_type"] = "img"
    data_dict["gt_type"] = "img"
    return data_dict 
############################################################
def get_2_pure_rect2_dataset(db_dir, db_name, img_type="bmp", batch_size=1, img_resize=(512,512)): 
    in_dir_name="dis_img_db"
    gt_dir_name="gt_ord_pad_img_db"
    return get_in_img_and_gt_img_db(db_dir=db_dir, db_name=db_name, in_dir_name=in_dir_name, gt_dir_name=gt_dir_name, img_type=img_type, batch_size=batch_size, img_resize=img_resize)

############################################################
def get_3_unet_rect2_dataset(db_dir, db_name, img_type="bmp", batch_size=1, img_resize=(512,512)): 
    in_dir_name="unet_rec_img_db"
    gt_dir_name="gt_ord_img_db"
    return get_in_img_and_gt_img_db(db_dir=db_dir, db_name=db_name, in_dir_name=in_dir_name, gt_dir_name=gt_dir_name, img_type=img_type, batch_size=batch_size, img_resize=img_resize)

############################################################
### 這應該算通用型抓test_db，如果沒有gt， test_gt_dir 就丟None就行囉！
def get_test_indicate_db(test_in_dir, test_gt_dir=None, gt_type="img", img_type="bmp", batch_size=1, img_resize=(512,512)):#, move_resize=(256,256)):
    ### 我目前的寫法是：
    # img_type：同時指定 train/test 的 in/gt 的(就是所有的意思啦) 附檔名為"bmp"或".jpg"
    # gt_type ：指定train/test 的 gt部分，是用 "img" 或 "move_map"
    data_dict = {}
    test_in_db = img_db(test_in_dir, img_type, img_resize, 1)
    data_dict["test_in_db"    ] = test_in_db.img_db  
    data_dict["test_in_db_pre"] = test_in_db.pre_db  
    data_dict["in_type"] = "img"

    if(test_gt_dir is None)      : data_dict["test_gt_db_pre"] = None
    else:
        if  (gt_type=="img")     : 
            test_gt_db = img_db(test_gt_dir, img_type, img_resize, 1)
            data_dict["test_gt_db"]     = test_gt_db.img_db  
            data_dict["test_gt_db_pre"] = test_gt_db.pre_db  
            data_dict["gt_type"] = "img"
        elif(gt_type=="move_map"): 
            test_move_map_db_ord, test_move_map_db_norm = get_test_move_map_db(test_gt_dir)
            data_dict["test_gt_db"] = test_move_map_db_ord
            data_dict["test_gt_db_pre"] = test_move_map_db_norm
            data_dict["gt_type"] = "move_map"
    return data_dict


def get_test_move_map_db(move_map_test_path, batch_size=1):
    from util import get_maxmin_train_move_from_path
    test_move_map_db_ord               = get_dir_move(move_map_test_path) #get_move_map_db_and_resize(move_map_test_path, resize_shape=resize_shape)
    max_train_move, min_train_move = get_maxmin_train_move_from_path(move_map_test_path)
    test_move_map_db_norm               = use_train_move_value_to_norm(test_move_map_db_ord, max_train_move, min_train_move) ### 這裡要用 max/min_train_move 來對 test_move_map_db 做 norm
    
    test_move_map_db_norm               = tf.data.Dataset.from_tensor_slices(test_move_map_db_norm)
    test_move_map_db_norm               = test_move_map_db_norm.batch(batch_size)
    # test_move_map_db = test_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)
    return test_move_map_db_ord, test_move_map_db_norm


# def get_test_indecate_dataset(db_dir="datasets", db_name="wei_book", img_type="jpg", batch_size=1, img_resize=(512,512)):     
#     test_in_img_db_path     = db_dir + "/" + db_name + "/" + "in_imgs"  
#     test_gt_img_db_path  = db_dir + "/" + db_name + "/" + "gt_imgs" 
    
#     test_img_db     = img_db(test_in_img_db_path,    img_type, img_resize, 1).img_db
#     test_gt_img_db  = img_db(test_gt_img_db_path, img_type, img_resize, 1).img_db
    
#     data_dict = {}
#     data_dict["test_in_db_pre"]=test_img_db
#     data_dict["test_gt_db_pre"]=test_gt_img_db
#     return test_img_db, test_gt_img_db


# def get_test_indecate_dataset_unet(db_dir="datasets", db_name="wei_book", img_type="jpg",  batch_size=1, img_resize=(512,512)):     
#     test_in_img_db_path     = db_dir + "/" + db_name + "/" + "in_imgs"  
#     test_img_db     = img_db(test_in_img_db_path, img_type, img_resize, 1).img_db

#     move_map_train_path = access_path+"datasets" + "/" + "pad2000-512to256" + "/" + "train/move_maps" 
#     train_move_map_db = get_move_map_db_and_resize(move_map_train_path, resize_shape=(512,512))
#     train_move_map_db, max_train_move, min_train_move = use_maxmin_train_move_to_normto_norm(train_move_map_db) ### 這裡會得到 max/min_train_move
    
#     data_dict = {}
#     data_dict["test_in_db_pre"]         = test_img_db
#     data_dict["max_train_move"] = max_train_move
#     data_dict["min_train_move"] = min_train_move
#     return test_img_db ,max_train_move ,min_train_move


if(__name__ == "__main__"):
    # access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    import time
    start_time = time.time()

    db_dir  = access_path+"datasets"
    db_name = "1_pure_unet_page_h=384,w=256"
    _ = get_1_pure_unet_db (db_dir=db_dir, db_name=db_name)

    # db_dir  = access_path+"datasets"
    # db_name = "2_pure_rect2_h=384,w=256"
    # _ = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name)

    print(time.time()- start_time)
    print("finish")
    