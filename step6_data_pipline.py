from step0_access_path import access_path
import tensorflow as tf
import os
import numpy as np
import cv2
from util import get_dir_img, get_dir_move, use_plt_show_move

import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

### 把img_db 包成class 是因為 tf.data.Dataset().map(f)的這個f，沒有辦法丟參數壓！所以只好包成class，把要傳的參數當 data_member囉！ 另一方面是好管理、好閱讀～
class img_db():
    def __init__(self,img_path, img_type, resize_shape, batch_size):
        self.img_path = img_path 
        self.img_type = img_type
        self.resize_shape = resize_shape 
        self.batch_size = batch_size 

        self.img_db = None
        # self.img = None

        self.get_img_db_from_file_name(img_path, batch_size)
        
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

    def step2_resize(self, img):### h=472, w=360
        img = tf.image.resize(img ,self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        return img


    def step3_normalize(self, img): ### 因為用tanh，所以把值弄到 [-1, 1]
        img = (img / 127.5) - 1
        return img


    def preprocess_bmp_img(self, file_name):
        img = self.step1_load_one_bmp_img(file_name)  ### 根據檔名，把圖片讀近來且把圖切開來
        img = self.step2_resize(img)
        img = self.step3_normalize(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 
    
    def preprocess_jpg_img(self, file_name):
        img = self.step1_load_one_jpg_img(file_name)  ### 根據檔名，把圖片讀近來且把圖切開來
        img = self.step2_resize(img)
        img = self.step3_normalize(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 

    def get_img_db_from_file_name(self, img_path, batch_size):
        self.img_db = tf.data.Dataset.list_files(img_path + "/" + "*." + self.img_type, shuffle=False)
        if  (self.img_type=="bmp"):self.img_db = self.img_db.map(self.preprocess_bmp_img)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        elif(self.img_type=="jpg"):self.img_db = self.img_db.map(self.preprocess_jpg_img)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        self.img_db = self.img_db.batch(batch_size)
        # self.img_db = self.img_db.prefetch(tf.data.experimental.AUTOTUNE)

####################################################################
### 以下是 numpy 直接整包load進記憶體，因為 file_name -> tensor失敗，
### 因為tf2不支援decode numpy檔案(只支援從記憶體load進去)，tf2 可以decode的檔案格式：text, imgs, csv, TFRecord，有空再試TFRecord

### get 和 resize 和 norm 拆開funtcion寫~~~ 因為直接get完直接norm，在外面就得不到原始 max min 了
def get_move_map_db_and_resize(ord_dir, resize_shape=(256,256)):
    import time
    start_time = time.time()
    move_map_list = get_dir_move(ord_dir)
    move_map_resize_list = []
    for move in move_map_list[:]:
        move_resize = cv2.resize( move, resize_shape, interpolation = cv2.INTER_NEAREST)
        move_map_resize_list.append(move_resize)
    move_map_resize_list = np.array(move_map_resize_list)
    print("get_move_map_db_and_resize cost time", time.time()-start_time)
    return move_map_resize_list

def use_maxmin_train_move_to_normto_norm(move_map_list): ### 給 train用
    max_train_move = move_map_list.max() ###  236.52951204508076
    min_train_move = move_map_list.min() ### -227.09562801056995
    move_map_list = ((move_map_list-min_train_move)/(max_train_move-min_train_move))*2-1
    return move_map_list, max_train_move, min_train_move ### max_train_move, min_train_move 需要回傳回去喔，因為要給 test 用

def use_train_move_value_to_norm(move_map_list, max_train_move, min_train_move): ### 給test來用，要用和train一樣的 max_train_move, min_train_move
    move_map_list = ((move_map_list-min_train_move)/(max_train_move-min_train_move))*2-1
    return move_map_list


def get_train_test_move_map_db(db_dir, db_name, resize_shape, batch_size):
    move_map_train_path = db_dir + "/" + db_name + "/" + "train/move_maps" 
    train_move_map_db = get_move_map_db_and_resize(move_map_train_path, resize_shape=resize_shape)
    train_move_map_db, max_train_move, min_train_move = use_maxmin_train_move_to_normto_norm(train_move_map_db) ### 這裡會得到 max/min_train_move
    train_move_map_db = tf.data.Dataset.from_tensor_slices(train_move_map_db)
    train_move_map_db = train_move_map_db.batch(batch_size)
    # train_move_map_db = train_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)

    move_map_test_path = db_dir + "/" + db_name + "/" + "test/move_maps" 
    test_move_map_db = get_move_map_db_and_resize(move_map_test_path, resize_shape=resize_shape)
    test_move_map_db = use_train_move_value_to_norm(test_move_map_db, max_train_move, min_train_move) ### 這裡要用 max/min_train_move 來對 test_move_map_db 做 norm
    test_move_map_db = tf.data.Dataset.from_tensor_slices(test_move_map_db)
    test_move_map_db = test_move_map_db.batch(batch_size)
    # test_move_map_db = test_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)

    return train_move_map_db, max_train_move, min_train_move, test_move_map_db
########################################################################################################

######################################################################################################################################
def get_unet_dataset(db_dir="datasets", db_name="stack_unet-256-100",img_type="bmp", batch_size=1, img_resize=(512,512), move_resize=(256,256)): 
    ### 建db的順序：input, input, output(gt), output(gt)，跟 get_rect2_dataset不一樣喔別混亂了！
    ### 拿到 dis_imgs_db 的 train dataset，從 檔名 → tensor
    train_dis_img_db_path = db_dir + "/" + db_name + "/" + "train/dis_imgs" 
    train_dis_img_db = img_db(train_dis_img_db_path, img_type, img_resize, 1).img_db
    ### 拿到 dis_imgs_db 的 test dataset，從 檔名 → tensor
    test_dis_img_db_path = db_dir + "/" + db_name + "/" + "test/dis_imgs" 
    test_dis_img_db = img_db(test_dis_img_db_path, img_type, img_resize, 1).img_db  
    

    ### 因為 test_move_map 需要 train_move_map 的 max/min_move 來把值弄到 -1~1，所以包成一個function來拿比較不會漏掉！
    ### 拿到 move_map 的 train dataset，從 直切先全部讀出來成npy → tensor
    ### 拿到 move_map 的 test dataset，從 直切先全部讀出來成npy → tensor
    train_move_map_db, \
    max_train_move, min_train_move, \
    test_move_map_db = get_train_test_move_map_db(db_dir, db_name, move_resize, 1)


    ##########################################################################################################################################
    ### 勿刪！用來測試寫得對不對！
    # import matplotlib.pyplot as plt 
    # from util import method2

    # take_num = 3
    # # for i, (img, move) in enumerate(zip(dis_imgs_db_train_in_db.take(take_num), train_move_map.take(take_num))): ### 想看train的部分用這行 且 註解掉下行
    # for i, (img, move) in enumerate(zip(test_dis_img_db.take(take_num), test_move_map_db.take(take_num))):     ### 想看test 的部分用這行 且 註解掉上行
    #     print("i",i)
    #     fig, ax = plt.subplots(1,2)
    #     img_back = (img[0]+1.)*127.5
    #     img_back = tf.cast(img_back, tf.int32)
    #     ax[0].imshow(img_back)


    #     # move_back = (move[0]+1)/2 * (max_train_move-min_train_move) + min_train_move  ### 想看train的部分用這行 且 註解掉下行
    #     move_back = (move[0]+1)/2 * (max_train_move-min_train_move) + min_train_move       ### 想看test 的部分用這行 且 註解掉上行
    #     move_bgr = method2(move_back[...,0], move_back[...,1],1)
    #     ax[1].imshow(move_bgr)
    #     plt.show()
    ##########################################################################################################################################
    data_dict = {}
    data_dict["train_in_db"]  = train_dis_img_db
    data_dict["train_gt_db"]  = train_move_map_db
    data_dict["test_db"]      = test_dis_img_db
    data_dict["test_gt_db"]   = test_move_map_db
    data_dict["max_train_move"] = max_train_move
    data_dict["min_train_move"] = min_train_move
    
    return data_dict #train_dis_img_db, train_move_map_db, test_dis_img_db, test_move_map_db, max_train_move, min_train_move


def get_rect2_dataset(db_dir="datasets", db_name="rect2_2000", batch_size=1, img_resize=(512,512)): 
    ### 建db的順序：input, output(gt), input , output(gt)，跟 get_unet_dataset不一樣喔別混亂了！
    if  ("pure_rect2" in db_name):
        train_img_db_path    = db_dir + "/" + db_name + "/" + "train/dis_img_db"  
        train_gt_img_db_path = db_dir + "/" + db_name + "/" + "train/gt_ord_pad_img_db" 
        test_img_db_path     = db_dir + "/" + db_name + "/" + "test/dis_img_db"  
        test_gt_img_db_path  = db_dir + "/" + db_name + "/" + "test/gt_ord_pad_img_db" 

    elif("unet_rect2" in db_name):
        train_img_db_path    = db_dir + "/" + db_name + "/" + "train/unet_rec_img_db"  
        train_gt_img_db_path = db_dir + "/" + db_name + "/" + "train/gt_ord_img" 
        test_img_db_path     = db_dir + "/" + db_name + "/" + "test/unet_rec_img_db"  
        test_gt_img_db_path  = db_dir + "/" + db_name + "/" + "test/gt_ord_img" 

    # elif(db_name == "rect2_add_dis_imgs_db"): ### 做錯的
    #     train_img_db_path    = db_dir + "/" + db_name + "/" + "train/dis_and_unet_rec_img_db"  
    #     train_gt_img_db_path = db_dir + "/" + db_name + "/" + "train/gt_dis_and_unet_rec_img_db" 
    #     test_img_db_path     = db_dir + "/" + db_name + "/" + "test/dis_and_unet_rec_img_db"  
    #     test_gt_img_db_path  = db_dir + "/" + db_name + "/" + "test/gt_dis_and_unet_rec_img_db" 

    train_img_db    = img_db(train_img_db_path,   "bmp", img_resize, 1).img_db
    train_gt_img_db = img_db(train_gt_img_db_path,"bmp", img_resize, 1).img_db
    test_img_db     = img_db(test_img_db_path,    "bmp", img_resize, 1).img_db
    test_gt_img_db  = img_db(test_gt_img_db_path, "bmp", img_resize, 1).img_db

    data_dict = {}
    data_dict["train_in_db"]  = train_img_db
    data_dict["train_gt_db"]  = train_gt_img_db
    data_dict["test_db"]      = test_img_db
    data_dict["test_gt_db"]   = test_gt_img_db
    return data_dict # train_img_db, train_gt_img_db, test_img_db, test_gt_img_db


def get_test_kong_dataset(db_dir="datasets", db_name="wei_book", img_type="jpg", batch_size=1, img_resize=(512,512)):     
    test_img_db_path     = db_dir + "/" + db_name + "/" + "in_imgs"  
    test_gt_img_db_path  = db_dir + "/" + db_name + "/" + "gt_imgs" 
    
    test_img_db     = img_db(test_img_db_path,    img_type, img_resize, 1).img_db
    test_gt_img_db  = img_db(test_gt_img_db_path, img_type, img_resize, 1).img_db
    
    data_dict = {}
    data_dict["test_db"]=test_img_db
    data_dict["test_gt_db"]=test_gt_img_db
    return test_img_db, test_gt_img_db


def get_test_kong_dataset_unet(db_dir="datasets", db_name="wei_book", img_type="jpg",  batch_size=1, img_resize=(512,512)):     
    test_img_db_path     = db_dir + "/" + db_name + "/" + "in_imgs"  
    test_img_db     = img_db(test_img_db_path, img_type, img_resize, 1).img_db

    move_map_train_path = access_path+"datasets" + "/" + "pad2000-512to256" + "/" + "train/move_maps" 
    train_move_map_db = get_move_map_db_and_resize(move_map_train_path, resize_shape=(512,512))
    train_move_map_db, max_train_move, min_train_move = use_maxmin_train_move_to_normto_norm(train_move_map_db) ### 這裡會得到 max/min_train_move
    
    data_dict = {}
    data_dict["test_db"]         = test_img_db
    data_dict["max_train_move"] = max_train_move
    data_dict["min_train_move"] = min_train_move
    return test_img_db ,max_train_move ,min_train_move


if(__name__ == "__main__"):
    # access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    import time
    start_time = time.time()

    db_dir  = access_path+"datasets"
    db_name = "pad300-512to256"
    _ = get_unet_dataset (db_dir=db_dir, db_name=db_name)

    db_dir  = access_path+"datasets"
    db_name = "rect2_add_dis_imgs"
    _ = get_rect2_dataset(db_dir=db_dir, db_name=db_name)

    print(time.time()- start_time)
    print("finish")
    