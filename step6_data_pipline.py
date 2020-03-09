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
    def __init__(self,img_path, resize_shape, batch_size):
        self.img_path = img_path 
        self.resize_shape = resize_shape 
        self.batch_size = batch_size 

        self.img_db = None
        # self.img = None
        self.get_img_db_from_file_name(img_path, resize_shape, batch_size)
        
    ####################################################################################################
    ### 以下是 bmp file_name -> tensor  成功！
    ### 這版是 有用 self.img， 各個step的參數就可以不用多一個img，直接存取self.img即可，但這樣好像就少了點 img傳遞過程的概念 ，所以還是先用下一版好了～這就先留著吧commit下一版在刪掉
    # def step1_load_one_img(self, file_name):
    #     self.img = tf.io.read_file(file_name)
    #     self.img = tf.image.decode_bmp(self.img)
    #     self.img  = tf.cast(self.img, tf.float32)
    #     return self.img

    # def step2_resize(self):### h=472, w=360
    #     self.img = tf.image.resize(self.img ,self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
    #     return self.img


    # def step3_normalize(self): ### 因為用tanh，所以把值弄到 [-1, 1]
    #     self.img = (self.img / 127.5) - 1
    #     return self.img


    # def preprocess_img(self, file_name):
    #     self.img = self.step1_load_one_img(file_name)  ### 根據檔名，把圖片讀近來且把圖切開來
    #     self.img = self.step2_resize()
    #     self.img = self.step3_normalize()           ### 因為用tanh，所以把值弄到 [-1, 1]
    #     return self.img 

    ####################################################################################################
    ### 這種寫法是 img 沒有用 self.img 來寫，比較 能夠顯現 img傳遞過程的概念，先用這個好了
    def step1_load_one_img(self, file_name):
        img = tf.io.read_file(file_name)
        img = tf.image.decode_bmp(img)
        img  = tf.cast(img, tf.float32)
        return img

    def step2_resize(self, img):### h=472, w=360
        img = tf.image.resize(img ,self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        return img


    def step3_normalize(self, img): ### 因為用tanh，所以把值弄到 [-1, 1]
        img = (img / 127.5) - 1
        return img


    def preprocess_img(self, file_name):
        img = self.step1_load_one_img(file_name)  ### 根據檔名，把圖片讀近來且把圖切開來
        img = self.step2_resize(img)
        img = self.step3_normalize(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 

    def get_img_db_from_file_name(self, img_path, resize_shape, batch_size):
        self.img_db = tf.data.Dataset.list_files(img_path + "/" + "*.bmp", shuffle=False)
        self.img_db = self.img_db.map(self.preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        self.img_db = self.img_db.batch(batch_size)
        self.img_db = self.img_db.prefetch(tf.data.experimental.AUTOTUNE)

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

    # max_train_move = move_map_list.max() ###  236.52951204508076
    # min_train_move = move_map_list.min() ### -227.09562801056995
    # move_map_list = ((move_map_list-min_train_move)/(max_train_move-min_train_move))*2-1
    # return move_map_list, max_train_move, min_train_move

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
    train_move_maps = get_move_map_db_and_resize(move_map_train_path, resize_shape=resize_shape)
    train_move_maps, max_train_move, min_train_move = use_maxmin_train_move_to_normto_norm(train_move_maps) ### 這裡會得到 max/min_train_move
    train_move_map = tf.data.Dataset.from_tensor_slices(train_move_maps)
    train_move_map = train_move_map.batch(batch_size)

    move_map_test_path = db_dir + "/" + db_name + "/" + "test/move_maps" 
    test_move_maps = get_move_map_db_and_resize(move_map_test_path, resize_shape=resize_shape)
    test_move_maps = use_train_move_value_to_norm(test_move_maps, max_train_move, min_train_move) ### 這裡要用 max/min_train_move 來對 test_move_maps 做 norm
    test_move_maps = tf.data.Dataset.from_tensor_slices(test_move_maps)
    test_move_maps = test_move_maps.batch(batch_size)

    return train_move_map, max_train_move, min_train_move, test_move_maps
########################################################################################################



######################################################################################################################################
### 之前 dis_imgs 用 file_name -> tensor 失敗時，整包 dis_imgs load進記憶體的方法，現在有點不適用因為 太大了！只裝16gb的記憶體就塞滿了！
# def dis_imgs_resize_and_nrom(dis_imgs,resize_shape=(256,256)): ### 把dis_imgs resize 和 值弄到-1~1
#     proc_list = []
#     for dis_img in dis_imgs:
#         proc = cv2.resize(dis_img, resize_shape, interpolation=cv2.INTER_CUBIC)
#         proc = proc[:,:,::-1]   ### bgr -> rgb
#         proc_list.append(proc)
#     proc_list = np.array(proc_list)
#     proc_list = (proc_list / 127.5)-1
#     return proc_list.astype(np.float32)

# def get_img_db_from_file_name(ord_dir, resize_shape=(256,256)):
#     import time
#     start_time = time.time()
#     dis_imgs = get_dir_img(ord_dir)
#     dis_imgs = dis_imgs_resize_and_nrom(dis_imgs, resize_shape)
#     print("get_img_db_from_file_name cost time", time.time()-start_time)
#     return dis_imgs #distorted_list
######################################################################################################################################


def get_unet_dataset(db_dir="datasets", db_name="stack_unet-256-100", batch_size=1, img_resize=(512,512), move_resize=(256,256)): 
    ### 建db的順序：input, input, output(gt), output(gt)，跟 get_rect2_dataset不一樣喔別混亂了！
    ### 拿到 dis_imgs 的 train dataset，從 檔名 → tensor
    dis_imgs_train_load_path = db_dir + "/" + db_name + "/" + "train/dis_imgs" 
    train_dis_imgs = img_db(dis_imgs_train_load_path, img_resize, 1).img_db
    ### 拿到 dis_imgs 的 test dataset，從 檔名 → tensor
    dis_imgs_test_load_path = db_dir + "/" + db_name + "/" + "test/dis_imgs" 
    test_dis_imgs = img_db(dis_imgs_test_load_path, img_resize, 1).img_db  
    

    ### 因為 test_move_map 需要 train_move_map 的 max/min_move 來把值弄到 -1~1，所以包成一個function來拿比較不會漏掉！
    ### 拿到 move_map 的 train dataset，從 直切先全部讀出來成npy → tensor
    ### 拿到 move_map 的 test dataset，從 直切先全部讀出來成npy → tensor
    train_move_map, max_train_move, min_train_move, \
    test_move_maps = get_train_test_move_map_db(db_dir, db_name, move_resize, 1)


    ##########################################################################################################################################
    ### 勿刪！用來測試寫得對不對！
    # import matplotlib.pyplot as plt 
    # from util import method2

    # take_num = 3
    # # for i, (img, move) in enumerate(zip(dis_imgs_train_db.take(take_num), train_move_map.take(take_num))): ### 想看train的部分用這行 且 註解掉下行
    # for i, (img, move) in enumerate(zip(test_dis_imgs.take(take_num), test_move_maps.take(take_num))):     ### 想看test 的部分用這行 且 註解掉上行
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
    
    return train_dis_imgs, train_move_map, test_dis_imgs, test_move_maps, max_train_move, min_train_move


def get_rect2_dataset(db_dir="datasets", db_name="rect2_add_dis_imgs", batch_size=1, img_resize=(512,512)): 
    ### 建db的順序：input, output(gt), input , output(gt)，跟 get_unet_dataset不一樣喔別混亂了！
    train_dis_and_unet_rec_imgs_path = db_dir + "/" + db_name + "/" + "train/dis_and_unet_rec_imgs"  
    train_dis_and_unet_rec_imgs      = img_db(train_dis_and_unet_rec_imgs_path, img_resize, 1).img_db

    train_gt_dis_and_unet_imgs_path  = db_dir + "/" + db_name + "/" + "train/gt_dis_and_unet_rec_imgs" 
    train_gt_dis_and_unet_rec_imgs   = img_db(train_gt_dis_and_unet_imgs_path, img_resize, 1).img_db

    test_dis_and_unet_rec_imgs_path  = db_dir + "/" + db_name + "/" + "test/dis_and_unet_rec_imgs"  
    test_dis_and_unet_rec_imgs       = img_db(test_dis_and_unet_rec_imgs_path, img_resize, 1).img_db

    test_gt_dis_and_unet_imgs_path   = db_dir + "/" + db_name + "/" + "test/gt_dis_and_unet_rec_imgs" 
    test_gt_dis_and_unet_rec_imgs    = img_db(test_gt_dis_and_unet_imgs_path, img_resize, 1).img_db


if(__name__ == "__main__"):
    # access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    import time
    start_time = time.time()

    db_dir  = access_path+"datasets"
    db_name = "pad2000-512to256_index"

    _ = get_unet_dataset(db_dir=db_dir, db_name=db_name)
    # _ = get_unet_dataset_from_file_name(db_dir=db_dir, db_name=db_name)


    print(time.time()- start_time)
    print("finish")
    