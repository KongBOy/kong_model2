import sys 
sys.path.append("kong_util")
from util import get_dir_img, get_dir_move, use_plt_show_move, get_db_amount

from step0_access_path import access_path
from step08_b_model_obj import MODEL_NAME, KModel_builder
from step06_a_datas_obj import DB_C, DB_N, DB_GM, Dataset_builder
import tensorflow as tf
import os
import numpy as np
import cv2
import math

import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

####################################################################################################
class mapping_util():
    def _resize(self, img):
        img = tf.image.resize(img ,self.resize_shape, method=tf.image.ResizeMethod.AREA )
        return img

    def _norm_to_tanh(self, img): ### 因為用tanh，所以把值弄到 [-1, 1]
        img = (img / 127.5) - 1
        return img

    def _norm_to_tanh_by_max_min_val(self, mov, max_val, min_val): ### 因為用tanh，所以把值弄到 [-1, 1]
        mov = ((mov-min_val)/(max_val-min_val))*2-1
        return mov

####################################################################################################
class img_mapping_util(mapping_util):
    ### 以下是 bmp file_name -> tensor  成功！
    ### 這種寫法是 img 沒有用 self.img 來寫，比較 能夠顯現 img傳遞過程的概念，先用這個好了，想看上一種風格去git 6b63a99 調喔
    def _step1_load_one_bmp_img(self, file_name):
        img = tf.io.read_file(file_name)
        img = tf.image.decode_bmp(img)
        img  = tf.cast(img, tf.float32)
        return img

    def _step0_load_bmp_ord_map(self, file_name): 
        img = self._step1_load_one_bmp_img(file_name)  ### 根據檔名，把圖片讀進來
        img  = tf.cast(img, tf.uint8)  ### 不會拿來訓練，是拿來顯示的，所以轉乘uint8
        return img

    def _step0_load_bmp_pre_map_resize_normalize(self, file_name):
        img = self._step1_load_one_bmp_img(file_name)  ### 根據檔名，把圖片讀進來
        img = self._resize(img)
        img = self._norm_to_tanh(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 

    ####################################################################################################
    def _step1_load_one_jpg_img(self, file_name):
        img = tf.io.read_file(file_name)
        img = tf.image.decode_jpeg(img)
        img  = tf.cast(img, tf.float32)
        return img

    def _step0_load_jpg_map_ord(self, file_name): 
        img = self._step1_load_one_jpg_img(file_name)  ### 根據檔名，把圖片讀進來
        img = tf.cast(img, tf.uint8)  ### 不會拿來訓練，是拿來顯示的，所以轉乘uint8
        return img
    def _step0_load_jpg_pre_map_resize_normalize(self, file_name):
        img = self._step1_load_one_jpg_img(file_name)  ### 根據檔名，把圖片讀進來
        img = self._resize(img)
        img = self._norm_to_tanh(img)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return img 


class mov_mapping_util(mapping_util):
    def _step1_load_one_move_map(self, file_name):
        mov = tf.io.read_file(file_name)
        mov = tf.io.decode_raw(mov , tf.float32)
        mov  = tf.cast(mov, tf.float32)
        return mov

    def _step0_load_mov_ord_map(self, file_name): 
        mov = self._step1_load_one_move_map(file_name)  ### 根據檔名，把圖片讀進來
        mov = tf.reshape( mov, [1, self.h, self.w, 2])
        mov = tf.cast(mov, tf.float32)  ### 不會拿來訓練，是拿來顯示的，所以轉乘uint8
        return mov
        
    def _step0_load_mov_pre_map_normalize(self, file_name):
        mov = self._step1_load_one_move_map(file_name)  ### 根據檔名，把圖片讀進來
        mov = tf.reshape( mov, [1, self.h, self.w, 2])
        mov = self._norm_to_tanh_by_max_min_val(mov, self.max_train_move, self.min_train_move)           ### 因為用tanh，所以把值弄到 [-1, 1]
        return mov 

####################################################################################################
class tf_Datapipline_builder():
    def __init__(self, tf_pipline=None):
        if(tf_pipline is None): self.tf_pipline = tf_Datapipline()
        else:                   self.tf_pipline = tf_pipline

    ### 建立empty tf_pipline
    def build(self):
        return self.tf_pipline

    ### 建立 img 的 pipline
    def build_img_pipline(self, ord_dir, img_type, resize_shape, batch_size):
        self.tf_pipline.ord_dir      = ord_dir
        self.tf_pipline.img_type     = img_type
        self.tf_pipline.resize_shape = resize_shape
        self.tf_pipline.batch_size   = batch_size
        return self.tf_pipline

    ### 建立 move_map 的 pipline
    def build_mov_pipline(self, ord_dir, h, w, max_train_move, min_train_move):
        self.tf_pipline.ord_dir = ord_dir
        self.tf_pipline.h = h
        self.tf_pipline.w = w
        self.tf_pipline.max_train_move = max_train_move
        self.tf_pipline.min_train_move = min_train_move
        return self.tf_pipline


####################################################################################################
### 把img_db 包成class 是因為 tf.data.Dataset().map(f)的這個f，沒有辦法丟參數壓！所以只好包成class，把要傳的參數當 data_member囉！ 另一方面是好管理、好閱讀～
class tf_Datapipline(img_mapping_util, mov_mapping_util):
    def __init__(self): ### img_type 是 bmp/jpg喔！
        self.ord_dir = None

        ### img類型要存的
        self.img_type = None
        self.resize_shape = None 
        self.batch_size = None 

        ### move_map類型要存的
        self.max_train_move = None
        self.min_train_move = None

        ### 以下兩個是最重要需要求得的
        self.ord_db = None ### 原始讀進來的影像db
        self.pre_db = None ### 進 generator 之前 做resize 和 值變-1~1的處理 後的db
    ####################################################################################################
    ### 這裡是練習用factory的寫法，如果之後覺得難看要改用builder的寫法也可以喔！
    @staticmethod
    def new_img_pipline(ord_dir, img_type, resize_shape, batch_size):
        img_pipline = tf_Datapipline_builder().build_img_pipline(ord_dir, img_type=img_type, resize_shape=resize_shape, batch_size=batch_size)
        img_pipline.get_img_db_from_file_name()
        return img_pipline 

    @staticmethod
    def new_mov_pipline(ord_dir, h, w, max_train_move, min_train_move):
        mov_pipline = tf_Datapipline_builder().build_mov_pipline(ord_dir, h=h, w=w, max_train_move=max_train_move, min_train_move=min_train_move)
        mov_pipline.get_mov_db_from_file_name()
        return mov_pipline
    ####################################################################################################
    def get_img_db_from_file_name(self):
        self.ord_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.img_type, shuffle=False)
        self.pre_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.img_type, shuffle=False)
        if  (self.img_type=="bmp"):
            self.ord_db = self.ord_db.map(self._step0_load_bmp_ord_map)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
            self.pre_db = self.pre_db.map(self._step0_load_bmp_pre_map_resize_normalize)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        elif(self.img_type=="jpg"):
            self.ord_db = self.ord_db.map(self._step0_load_jpg_map_ord)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
            self.pre_db = self.pre_db.map(self._step0_load_jpg_pre_map_resize_normalize)#, num_parallel_calls=tf.data.experimental.AUTOTUNE) ### 如果 gpu 記憶體不構，把num_parallew_calls註解掉即可！
        self.ord_db = self.ord_db.batch(self.batch_size)
        self.pre_db = self.pre_db.batch(self.batch_size)
        # self.ord_db = self.ord_db.prefetch(tf.data.experimental.AUTOTUNE)

    def get_mov_db_from_file_name(self):
        self.ord_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*.knpy" , shuffle=False)
        self.pre_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*.knpy" , shuffle=False)
        self.ord_db = self.ord_db.map(self._step0_load_mov_ord_map)
        self.pre_db = self.pre_db.map(self._step0_load_mov_pre_map_normalize )

########################################################################################################################################
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
        if  ("unet" in model_name.value ): 
            self.tf_data.img_resize = (math.floor(self.tf_data.db_obj.h/128)*128 *2, math.floor(self.tf_data.db_obj.w/128)*128 *2) ### 128的倍數，且要是gt_img的兩倍大喔！
        elif("rect" in model_name.value or "justG" in model_name.value ):
            self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h/4)*4, math.ceil(self.tf_data.db_obj.w/4)*4) ### dis_img(in_img的大小)的大小且要是4的倍數
        return self
    
    def build(self):
        return self.tf_data

class tf_Data_in_dis_gt_move_map_builder(tf_Data_init_builder):
    def build_by_in_dis_gt_move_map(self):
        ### 建db的順序：input, input, output(gt), output(gt)，跟 get_rect2_dataset不一樣喔別混亂了！
        ### 拿到 dis_imgs_db 的 train dataset，從 檔名 → tensor
        train_in_db = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.train_in_dir, self.tf_data.db_obj.in_type, self.tf_data.img_resize, self.tf_data.batch_size)
        self.tf_data.train_in_db     = train_in_db.ord_db
        self.tf_data.train_in_db_pre = train_in_db.pre_db
        
        ### 拿到 dis_imgs_db 的 test dataset，從 檔名 → tensor
        test_in_db = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.test_in_dir, self.tf_data.db_obj.in_type, self.tf_data.img_resize, self.tf_data.batch_size)
        self.tf_data.test_in_db      = test_in_db.ord_db
        self.tf_data.test_in_db_pre  = test_in_db.pre_db


        ### 在拿move_map db 之前，要先去抓 max/min train_move，我是設計放 train_gt_dir 下的.npy，如果怕混淆 要改放.txt之類的都可以喔！
        ### 決定還是放在上一層好了，因為下面會用 get_db_amount 是算檔案數量的，雖然是去in_dir抓影像跟gt_dir沒關係，但還是怕有意外(以後忘記之類的)～放外面最安全囉！
        ### 且放外面容易看到可以提醒自己有這東西的存在覺得ˊ口ˋ
        if(os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy") and 
           os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")):
            self.tf_data.max_train_move = np.load(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy")
            self.tf_data.min_train_move = np.load(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")
        else: ### 如果.npy不存在，就去重新找一次 max/min train_move，找完也順便存一份給之後用囉！
            print("因為現在已經存成.knpy，沒辦法抓 max/min train_move 囉！麻煩先去以前的dataset撈出來啦！")
            ### 偷懶可以把 .npy 放同個資料夾，把註解拿掉就可以順便求囉！只是因為這有return，所以還是要重新執行一次才會完整跑完喔～
            # move_maps = get_dir_move(self.tf_data.db_obj.train_gt_dir)
            # self.tf_data.max_train_move = move_maps.max()
            # self.tf_data.min_train_move = move_maps.min()
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../max_train_move", self.tf_data.max_train_move)
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../min_train_move", self.tf_data.min_train_move)
            # print("self.tf_data.max_train_move",self.tf_data.max_train_move)
            # print("self.tf_data.min_train_move",self.tf_data.min_train_move)
            return 
        
        train_gt_db = tf_Datapipline.new_mov_pipline(self.tf_data.db_obj.train_gt_dir, self.tf_data.db_obj.h, self.tf_data.db_obj.w, self.tf_data.max_train_move, self.tf_data.min_train_move)
        test_gt_db  = tf_Datapipline.new_mov_pipline(self.tf_data.db_obj.test_gt_dir,  self.tf_data.db_obj.h, self.tf_data.db_obj.w, self.tf_data.max_train_move, self.tf_data.min_train_move)
        self.tf_data.train_gt_db     = train_gt_db.ord_db
        self.tf_data.train_gt_db_pre = train_gt_db.pre_db
        self.tf_data.test_gt_db      = test_gt_db.ord_db
        self.tf_data.test_gt_db_pre  = test_gt_db.pre_db

        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir )
        ##########################################################################################################################################
        self.tf_data.train_db_combine = tf.data.Dataset.zip( (self.tf_data.train_in_db, self.tf_data.train_in_db_pre,
                                                              self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre) )
        if(self.tf_data.train_shuffle):
            self.tf_data.train_db_combine=self.tf_data.train_db_combine.shuffle( int(self.tf_data.train_amount/2) ) ### shuffle 的 buffer_size 太大會爆記憶體，嘗試了一下大概 /1.8 左右ok這樣子~ 但 /2 應該比較保險！
        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            see_in_db  = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.see_in_dir , self.tf_data.db_obj.in_type, self.tf_data.img_resize, self.tf_data.batch_size)
            self.tf_data.see_in_db     = see_in_db.ord_db
            self.tf_data.see_in_db_pre = see_in_db.pre_db
            see_gt_db  = tf_Datapipline.new_mov_pipline(self.tf_data.db_obj.see_gt_dir , self.tf_data.db_obj.h, self.tf_data.db_obj.w, self.tf_data.max_train_move, self.tf_data.min_train_move)
            self.tf_data.see_gt_db     = see_gt_db.ord_db
            self.tf_data.see_gt_db_pre = see_gt_db.pre_db
            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir )

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt 
        # from util import method2

        # take_num = 5
        # print(self.tf_data.max_train_move)
        # print(self.tf_data.min_train_move)
        # for i, (img, img_pre, move, move_pre) in enumerate(self.tf_data.train_db_combine.take(take_num)):     ### 想看test 的部分用這行 且 註解掉上行
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
        #     move_back = (move_pre[0]+1)/2 * (self.tf_data.max_train_move-self.tf_data.min_train_move) + self.tf_data.min_train_move    ### 想看test 的部分用這行 且 註解掉上行
        #     move_back_bgr = method2(move_back[...,0], move_back[...,1],1)
        #     ax[ax_i].imshow(move_back_bgr)
        #     plt.show()
        #     plt.close()
        #########################################################################################################################################
        return self


class tf_Data_in_dis_gt_img_builder(tf_Data_in_dis_gt_move_map_builder):
    def build_by_in_img_and_gt_img_db(self):
        ### 建db的順序：input, output(gt), input , output(gt)，跟 in_dis_gt_move_map不一樣喔別混亂了！
        train_in_db = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.train_in_dir, self.tf_data.db_obj.in_type, self.tf_data.img_resize, 1)
        train_gt_db = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.train_gt_dir, self.tf_data.db_obj.gt_type, self.tf_data.img_resize, 1)
        test_in_db  = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.test_in_dir , self.tf_data.db_obj.in_type, self.tf_data.img_resize, 1)
        test_gt_db  = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.test_gt_dir , self.tf_data.db_obj.gt_type, self.tf_data.img_resize, 1)

        self.tf_data.train_in_db     = train_in_db.ord_db
        self.tf_data.train_in_db_pre = train_in_db.pre_db
        self.tf_data.train_gt_db     = train_gt_db.ord_db
        self.tf_data.train_gt_db_pre = train_gt_db.pre_db
        self.tf_data.test_in_db      = test_in_db.ord_db
        self.tf_data.test_in_db_pre  = test_in_db.pre_db
        self.tf_data.test_gt_db      = test_gt_db.ord_db
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
            see_in_db  = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.see_in_dir , self.tf_data.db_obj.see_type, self.tf_data.img_resize, self.tf_data.batch_size)
            see_gt_db  = tf_Datapipline.new_img_pipline(self.tf_data.db_obj.see_gt_dir , self.tf_data.db_obj.see_type, self.tf_data.img_resize, self.tf_data.batch_size)
            self.tf_data.see_in_db     = see_in_db.ord_db
            self.tf_data.see_in_db_pre = see_in_db.pre_db
            self.tf_data.see_gt_db     = see_gt_db.ord_db
            self.tf_data.see_gt_db_pre = see_gt_db.pre_db
            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir )
        return self
    ############################################################

class tf_Data_builder(tf_Data_in_dis_gt_img_builder): 
    def build_by_db_get_method(self):
        if  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_move_map):
            self.build_by_in_dis_gt_move_map()
        elif(self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord or
             self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord_pad or
             self.tf_data.db_obj.get_method == DB_GM.in_rec_gt_ord):        
             self.build_by_in_img_and_gt_img_db()
        return self


if(__name__ == "__main__"):
    import time
    start_time = time.time()

    # db_obj = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_type(in_type="bmp", gt_type="bmp", see_type="bmp").set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size-1, train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()
    
    db_obj = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page, DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_type(in_type="bmp", gt_type="...", see_type="...").set_detail(have_train=True, have_see=True).build()
    model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_unet()
    tf_data = tf_Data_builder().set_basic(db_obj,1 , train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()
    

    print(time.time()- start_time)
    print("finish")
    