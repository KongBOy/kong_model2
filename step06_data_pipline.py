import sys 
sys.path.append("kong_util")
from util import get_dir_img, get_dir_move, use_plt_show_move, get_db_amount

from step0_access_path import access_path
from step08_model_obj import MODEL_NAME, KModel_builder
from step10_db_obj import DB_C, DB_N, DB_GM, Dataset_builder
import tensorflow as tf
import os
import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

### 把img_db 包成class 是因為 tf.data.Dataset().map(f)的這個f，沒有辦法丟參數壓！所以只好包成class，把要傳的參數當 data_member囉！ 另一方面是好管理、好閱讀～
# class img_db():
class tf_Datapipline:
    def __init__(self,img_path, img_type, resize_shape, batch_size): ### img_type 是 bmp/jpg喔！
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

class tf_Data:
    def __init__(self):
        self.db_obj           = None
        self.batch_size       = None
        self.train_shuffle    = None

        self.img_resize       = None

        self.train_in_db      = None
        self.train_in_db_pre  = None
        self.train_gt_db      = None
        self.train_gt_db_pre  = None
        self.train_db_combine = None 
        self.train_amount     = None

        self.test_in_db       = None
        self.test_in_db_pre   = None
        self.test_gt_db       = None
        self.test_gt_db_pre   = None
        self.test_amount      = None

        self.see_in_db        = None
        self.see_in_db_pre    = None
        self.see_gt_db        = None
        self.see_gt_db_pre    = None
        self.see_amount       = None
        self.see_type         = None

        ### 最主要是再 step7 unet generate image 時用到，但我覺得可以改寫！所以先註解掉了！
        # self.in_type          = None
        # self.gt_type          = None

        self.max_train_move   = None
        self.min_train_move   = None


class tf_Data_init_builder:
    def __init__(self, tf_data=None):
        if(tf_data is None):self.tf_data = tf_Data()
        else:               self.tf_data = tf_data

    def set_basic(self, db_obj, batch_size=1, train_shuffle=True):
        self.tf_data.db_obj        = db_obj
        self.tf_data.batch_size    = batch_size
        self.tf_data.train_shuffle = train_shuffle
        return self

    def set_img_resize(self, model_name):
        if  (model_name == MODEL_NAME.unet): 
            self.tf_data.img_resize = (math.floor(self.tf_data.db_obj.h/128)*128 *2, math.floor(self.tf_data.db_obj.w/128)*128 *2) ### 128的倍數，且要是gt_img的兩倍大喔！
            
        elif(model_name == MODEL_NAME.rect or 
             model_name == MODEL_NAME.mrf_rect or
             model_name == MODEL_NAME.just_G ):
            self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h/4)*4, math.ceil(self.tf_data.db_obj.w/4)*4) ### dis_img(in_img的大小)的大小且要是4的倍數
        return self
    
    def build(self):
        return self.tf_data

class tf_Data_in_dis_gt_move_map_builder(tf_Data_init_builder):
    @staticmethod
    def use_maxmin_train_move_to_norm_to_norm( move_maps): ### 給 train用
        max_train_move = move_maps.max() ###  236.52951204508076
        min_train_move = move_maps.min() ### -227.09562801056995
        move_maps = ((move_maps-min_train_move)/(max_train_move-min_train_move))*2-1
        return move_maps, max_train_move, min_train_move ### max_train_move, min_train_move 需要回傳回去喔，因為要給 test 用

    @staticmethod
    def use_train_move_value_to_norm( move_maps, max_train_move, min_train_move): ### 給test來用，要用和train一樣的 max_train_move, min_train_move
        move_maps = ((move_maps-min_train_move)/(max_train_move-min_train_move))*2-1
        return move_maps

    def _get_train_test_move_map_db(self, batch_size):
        move_map_train_path = self.tf_data.db_obj.train_gt_dir
        train_move_map_db_ord = get_dir_move(move_map_train_path) #get_move_map_db_and_resize(move_map_train_path, resize_shape=resize_shape)
        train_move_map_db_norm, \
            max_train_move, min_train_move = self.use_maxmin_train_move_to_norm_to_norm(train_move_map_db_ord) ### 這裡會得到 max/min_train_move
        train_move_map_db_norm = tf.data.Dataset.from_tensor_slices(train_move_map_db_norm)
        train_move_map_db_norm = train_move_map_db_norm.batch(batch_size)
        # train_move_map_db = train_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)

        move_map_test_path = self.tf_data.db_obj.test_gt_dir 
        test_move_map_db_ord = get_dir_move(move_map_test_path) #get_move_map_db_and_resize(move_map_test_path, resize_shape=resize_shape)
        test_move_map_db_norm = self.use_train_move_value_to_norm(test_move_map_db_ord, max_train_move, min_train_move) ### 這裡要用 max/min_train_move 來對 test_move_map_db 做 norm
        test_move_map_db_norm = tf.data.Dataset.from_tensor_slices(test_move_map_db_norm)
        test_move_map_db_norm = test_move_map_db_norm.batch(batch_size)
        # test_move_map_db = test_move_map_db.prefetch(tf.data.experimental.AUTOTUNE)

        train_move_map_db_ord = tf.data.Dataset.from_tensor_slices(train_move_map_db_ord)
        train_move_map_db_ord = train_move_map_db_ord.batch(batch_size)
        test_move_map_db_ord = tf.data.Dataset.from_tensor_slices(test_move_map_db_ord)
        test_move_map_db_ord = test_move_map_db_ord.batch(batch_size)
        return train_move_map_db_norm, max_train_move, min_train_move, test_move_map_db_norm, train_move_map_db_ord, test_move_map_db_ord

    def build_by_in_dis_gt_move_map(self):
        ### 建db的順序：input, input, output(gt), output(gt)，跟 get_rect2_dataset不一樣喔別混亂了！
        ### 拿到 dis_imgs_db 的 train dataset，從 檔名 → tensor
        train_in_db = tf_Datapipline(self.tf_data.db_obj.train_in_dir, self.tf_data.db_obj.in_type, self.tf_data.img_resize, self.tf_data.batch_size)
        self.tf_data.train_in_db     = train_in_db.img_db
        self.tf_data.train_in_db_pre = train_in_db.pre_db
        
        ### 拿到 dis_imgs_db 的 test dataset，從 檔名 → tensor
        test_in_db = tf_Datapipline(self.tf_data.db_obj.test_in_dir, self.tf_data.db_obj.in_type, self.tf_data.img_resize, self.tf_data.batch_size)
        self.tf_data.test_in_db      = test_in_db.img_db
        self.tf_data.test_in_db_pre  = test_in_db.pre_db

        ### 因為 test_move_map 需要 train_move_map 的 max/min_move 來把值弄到 -1~1，所以包成一個function來拿比較不會漏掉！
        ### 拿到 move_map 的 train dataset，從 直切先全部讀出來成npy → tensor
        ### 拿到 move_map 的 test dataset，從 直切先全部讀出來成npy → tensor
        train_move_map_db_norm, \
        max_train_move, min_train_move, \
        test_move_map_db_norm, \
        train_move_map_db_ord, test_move_map_db_ord = self._get_train_test_move_map_db(self.tf_data.db_obj, 1)

        self.tf_data.train_gt_db     = train_move_map_db_ord
        self.tf_data.train_gt_db_pre = train_move_map_db_norm
        self.tf_data.test_gt_db      = test_move_map_db_ord
        self.tf_data.test_gt_db_pre  = test_move_map_db_norm

        self.tf_data.max_train_move  = max_train_move
        self.tf_data.min_train_move  = min_train_move

        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir )

        ### 我覺得step7 那邊可以改寫，所以in_type就改記 bmp/jpg了，這邊就先註解掉囉！
        # self.tf_data.in_type = "img"
        # self.tf_data.gt_type = "move_map"

        ##########################################################################################################################################
        self.tf_data.train_db_combine = tf.data.Dataset.zip( (self.tf_data.train_in_db, self.tf_data.train_in_db_pre,
                                                              self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre) )
        if(self.tf_data.train_shuffle):
            self.tf_data.train_db_combine=self.tf_data.train_db_combine.shuffle( int(self.tf_data.train_amount/2) ) ### shuffle 的 buffer_size 太大會爆記憶體，嘗試了一下大概 /1.8 左右ok這樣子~ 但 /2 應該比較保險！
        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt 
        # from util import method2

        # take_num = 5
        
        # for i, (img, img_pre, move, move_pre) in enumerate(self.tf_data.db_combine.take(take_num)):     ### 想看test 的部分用這行 且 註解掉上行
        #     print("i",i)
        #     fig, ax = plt.subplots(1,4)
        #     fig.set_size_inches(15,5)
        #     ax_i = 0
        #     img = tf.cast(img[0], tf.uint8)
        #     ax[ax_i].imshow(img)

        #     print(img.numpy().dtype)
        #     ax_i += 1

        #     img_pre_back = (img_pre[0]+1.)*127.5
        #     img_pre_back = tf.cast(img_pre_back, tf.int32)
        #     ax[ax_i].imshow(img_pre_back)
        #     ax_i += 1

        #     move_bgr = method2(move[0,...,0], move[0,...,1])
        #     ax[ax_i].imshow(move_bgr)
        #     ax_i += 1

        #     # move_back = (move[0]+1)/2 * (max_train_move-min_train_move) + min_train_move  ### 想看train的部分用這行 且 註解掉下行
        #     move_back = (move_pre[0]+1)/2 * (max_train_move-min_train_move) + min_train_move    ### 想看test 的部分用這行 且 註解掉上行
        #     move_back_bgr = method2(move_back[...,0], move_back[...,1],1)
        #     ax[ax_i].imshow(move_back_bgr)
        #     plt.show()
        #     plt.close()
        ##########################################################################################################################################
        return self


class tf_Data_in_dis_gt_img_builder(tf_Data_in_dis_gt_move_map_builder):
    def build_by_in_img_and_gt_img_db(self):
        ### 建db的順序：input, output(gt), input , output(gt)，跟 in_dis_gt_move_map不一樣喔別混亂了！
        train_in_db = tf_Datapipline(self.tf_data.db_obj.train_in_dir, self.tf_data.db_obj.in_type, self.tf_data.img_resize, 1)
        train_gt_db = tf_Datapipline(self.tf_data.db_obj.train_gt_dir, self.tf_data.db_obj.gt_type, self.tf_data.img_resize, 1)
        test_in_db  = tf_Datapipline(self.tf_data.db_obj.test_in_dir , self.tf_data.db_obj.in_type, self.tf_data.img_resize, 1)
        test_gt_db  = tf_Datapipline(self.tf_data.db_obj.test_gt_dir , self.tf_data.db_obj.gt_type, self.tf_data.img_resize, 1)

        self.tf_data.train_in_db     = train_in_db.img_db
        self.tf_data.train_in_db_pre = train_in_db.pre_db
        self.tf_data.train_gt_db     = train_gt_db.img_db
        self.tf_data.train_gt_db_pre = train_gt_db.pre_db
        self.tf_data.test_in_db      = test_in_db.img_db
        self.tf_data.test_in_db_pre  = test_in_db.pre_db
        self.tf_data.test_gt_db      = test_gt_db.img_db
        self.tf_data.test_gt_db_pre  = test_gt_db.pre_db

        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir )

        self.tf_data.train_db_combine = tf.data.Dataset.zip( (self.tf_data.train_in_db, self.tf_data.train_in_db_pre, 
                                                            self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre) )
                                                
        if(self.tf_data.train_shuffle): 
            self.tf_data.train_db_combine = self.tf_data.train_db_combine.shuffle( int(self.tf_data.train_amount/2) ) ### shuffle 的 buffer_size 太大會爆記憶體，嘗試了一下大概 /1.8 左右ok這樣子~ 但 /2 應該比較保險！
        #########################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt 
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     train_in     = train_in[0]     ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0] ### 值 0. ~ 1.
        #     train_gt     = train_gt[0]     ### 值 0  ~ 255
        #     train_gt_pre = train_gt_pre[0] ### 值 0. ~ 1.

        #     fig, ax = plt.subplots(1,4)
        #     fig.set_size_inches(15,5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()
        #########################################################
        
        if(self.tf_data.db_obj.have_see):
            see_in_db  = tf_Datapipline(self.tf_data.db_obj.see_in_dir , self.tf_data.db_obj.see_type, self.tf_data.img_resize, self.tf_data.batch_size)
            see_gt_db  = tf_Datapipline(self.tf_data.db_obj.see_gt_dir , self.tf_data.db_obj.see_type, self.tf_data.img_resize, self.tf_data.batch_size)
            self.tf_data.see_in_db     = see_in_db.img_db
            self.tf_data.see_in_db_pre = see_in_db.pre_db
            self.tf_data.see_gt_db     = see_gt_db.img_db
            self.tf_data.see_gt_db_pre = see_gt_db.pre_db
            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir )
        return self
    ############################################################

class tf_Data_builder(tf_Data_in_dis_gt_img_builder): 
    def build_by_db_get_method(self):
        if  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_move_map):
            self.build_by_in_img_and_gt_img_db()
        elif(self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord or
             self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord_pad or
             self.tf_data.db_obj.get_method == DB_GM.in_rec_gt_ord):        
             self.build_by_in_img_and_gt_img_db()
        return self


if(__name__ == "__main__"):
    # access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

    import time
    start_time = time.time()
    db_obj = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_type(in_type="bmp", gt_type="bmp", see_type="bmp").set_detail(have_train=True, have_see=True).build()
    model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
    print(db_obj)
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size-1, train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()
    # print(tf_data.img_resize)

    print(time.time()- start_time)
    print("finish")
    