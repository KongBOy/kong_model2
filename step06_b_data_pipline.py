"""
利用 step06_a_datas_obj的資訊 來建立 tf_data
"""
import sys
sys.path.append("kong_util")
# from util import get_dir_moves
from util import get_db_amount
import time

from step06_a_datas_obj import DB_C, DB_N, DB_GM, VALUE_RANGE, Dataset_builder, Range
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

import numpy as np
import math

import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')  ### 這步非常非常重要！用了才可以加速！

# import pdb
debug_dict = {}
####################################################################################################
class norm_mapping_util():
    def _norm_img_to_tanh(self, img):  ### 因為用tanh，所以把值弄到 [-1, 1]
        img = (img / 127.5) - 1
        return img

    def _norm_img_to_01(self, img):
        img = img / 255.0
        return img

    def _norm_to_01_by_max_min_val(self, data):  ### 因為用tanh，所以把值弄到 [-1, 1]
        data = (data - self.db_range.min) / (self.db_range.max - self.db_range.min)
        # print("self.db_range.max", self.db_range.max)
        # print("self.db_range.min", self.db_range.min)
        return data

    def _norm_to_tanh_by_max_min_val(self, data):  ### 因為用tanh，所以把值弄到 [-1, 1]
        data = self._norm_to_01_by_max_min_val(data) * 2 - 1
        return data

    def check_use_range_and_db_range_then_normalize_data(self, data):
        if  (self.use_range == self.db_range): pass  ### 此狀況比如 uv值方面blender都幫我們弄好了：值在 0~1 之間，所以不用normalize～
        elif(self.use_range == VALUE_RANGE.zero_to_one   .value): data = self._norm_to_01_by_max_min_val   (data)
        elif(self.use_range == VALUE_RANGE.neg_one_to_one.value): data = self._norm_to_tanh_by_max_min_val (data)  ### resize 和 值弄到 -1~1，假設(99%正確拉懶得考慮其他可能ˊ口ˋ)blender 的 flow一定0~1，所以不等於發生時 一定是要弄成 -1~1
        elif(self.use_range == VALUE_RANGE.img_range     .value): print("img 的 in/gt range 設錯囉！ 不能夠直接用 0~255 的range 來train 模型喔~~")   ### 防呆一下
        elif(self.use_range is None): print("tf_data 忘記設定 in/use_gt_range 了！，你可能會看到 Dataset.zip() 的錯誤喔 ~ ")              ### 防呆一下
        return data


####################################################################################################
####################################################################################################
class img_mapping_util(norm_mapping_util):
    def _resize(self, img):
        img = tf.image.resize(img, self.img_resize, method=tf.image.ResizeMethod.AREA)
        return img

    def step0a_load_byte_img(self, file_name):
        byte_img = tf.io.read_file(file_name)
        return byte_img

    '''
    寫得這麼難看，是因為 dataset.map() 不能傳參數呀~~ 網路上查到用lambda 傳參數 可跑過，但是無法autograph！
    所以就乖乖寫得這麼難看囉！
    '''
    def step0b_decode_bmp(self, byte_img): return tf.image.decode_bmp(byte_img)
    def step0b_decode_jpg(self, byte_img): return tf.image.decode_jpeg(byte_img)
    def step0b_decode_png(self, byte_img): return tf.image.decode_png(byte_img)
    def step0b_decode_gif(self, byte_img): return tf.image.decode_gif(byte_img)[0]  ### gif 好像比較特殊，我猜是 THWC, T是時間


    def step1_uint8_resize(self, img):
        img = self._resize(img)
        img  = tf.cast(img, tf.uint8)
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉


    def step1_float32_resize_and_to_01(self, img):
        img  = tf.cast(img, tf.float32)
        img = self._resize(img)
        img = self._norm_img_to_01(img)
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉


    def step1_float32_resize_and_to_tanh(self, img):
        img  = tf.cast(img, tf.float32)
        img = self._resize(img)
        img = self._norm_img_to_tanh(img)
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉


####################################################################################################
####################################################################################################
class mov_mapping_util(norm_mapping_util):
    def _step0_load_knpy(self, file_name):
        '''
        uv(flow):
            ch1 : mask
            ch2 : y
            ch3 : x
        wc:
            ch1：待查
            ch2：待查
            ch3：待查
        '''
        flow = tf.io.read_file(file_name)
        flow = tf.io.decode_raw(flow , tf.float32)
        flow = tf.cast(flow, tf.float32)
        return flow

    def step1_flow_load(self, file_name):
        raw_data = self._step0_load_knpy(file_name)
        F = tf.reshape(raw_data, [self.img_resize[0], self.img_resize[1], 3])  ### ch1:mask, ch2:y, ch3:x
        return F

    def step2_M_clip_and_C_normalize(self, F):
        M     = F[..., 0:1]
        C     = F[..., 1:3]
        M_pre = self._mask_binary_clip(M)
        C_pre = self.check_use_range_and_db_range_then_normalize_data(C)
        F_pre = tf.concat([M_pre, C_pre], axis=-1)
        return F_pre

    def step2_M_clip_and_C_normalize_try_mul_M(self, wc):
        M     = F[..., 0:1]
        C     = F[..., 1:3]
        M_pre = self._mask_binary_clip(M)
        C_pre = self.check_use_range_and_db_range_then_normalize_data(C)
        C_pre = C_pre * M
        F_pre = tf.concat([M_pre, C_pre], axis=-1)
        return F_pre

    # def step2_flow_split_to_M_and_C(self, F):
    #     M = F[..., 0:1]
    #     C = F[..., 1:3]
    #     return M, C

    # def step2_flow_split_to_M_and_C_then_M_clip_and_C_normalize(self, F):
    #     M = F[..., 0:1]
    #     M = self._mask_binary_clip(M)

    #     C = F[..., 1:3]
    #     C = self.check_use_range_and_db_range_then_normalize_data(C)
    #     return M, C

    def step2_flow_extract_just_mask(self, F):
        M = F[..., 0:1]
        return M


    def step1_wc_load(self, file_name):
        raw_data = self._step0_load_knpy(file_name)
        wc = tf.reshape(raw_data, [self.img_resize[0], self.img_resize[1], 4])  ### ch1:z, ch2:y, ch3:x, ch4:後處理成mask了
        return wc   #[..., :3]


    def step2_M_clip_and_W_normalize(self, wc):
        M = wc[..., 3:4]
        M = self._mask_binary_clip(M)
        wc   = wc[...,  :3]
        wc   = self.check_use_range_and_db_range_then_normalize_data(wc)
        W = tf.concat([wc, M], axis=-1)
        return W

    def step2_M_clip_and_W_normalize_try_mul_M(self, wc):
        M = wc[..., 3:4]
        M = self._mask_binary_clip(M)
        wc = wc[...,  :3]
        wc = self.check_use_range_and_db_range_then_normalize_data(wc)
        wc = wc * M
        W = tf.concat([wc, M], axis=-1)
        return W

class mask_mapping_util(norm_mapping_util):
    def _3ch_get_1ch(self, img): return img[..., 0]

    def _mask_binary_clip(self, mask):
        threshold = 0.8
        mask = tf.where(mask > threshold, 1, 0)
        mask = tf.cast(mask, tf.float32)
        return mask

####################################################################################################
####################################################################################################
### 下面的 tf_Datapipline_factory_builder/Factory 都是為了要建 tf_Datapipline_factory 這個物件喔！
### 把img_db 包成class 是因為 tf.data.Dataset().map(f)的這個f，沒有辦法丟參數壓！所以只好包成class，把要傳的參數當 data_member囉！ 另一方面是好管理、好閱讀～
class tf_Datapipline_factory(img_mapping_util, mov_mapping_util, mask_mapping_util):
    def __init__(self):  ### file_format 是 bmp/jpg喔！
        self.ord_dir = None
        '''
        db_range: DB 本身的 range(由 db_obj 決定)
        use_range: 進網路前想用的 range(由 exp_obj 決定)
        '''

        ### img類型要存的
        self.file_format = None
        self.img_resize = None

        self.db_range  = None
        self.use_range = None

        ### 以下兩個是最重要需要求得的
        self.ord_db = None  ### 原始讀進來的影像db
        self.pre_db = None  ### 進 generator 之前 做resize 和 值變-1~1的處理 後的db

    ####################################################################################################
    def _build_file_name_db(self):
        # print("debug _build_file_name_db self.ord_dir:", self.ord_dir)
        return tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.file_format, shuffle=False)

    def step1_file_paths_to_names(self, file_path):
        return (tf.strings.split(file_path, "\\")[-1])

    def build_name_db(self):
        file_names = self._build_file_name_db()
        self.ord_db = file_names.map(self.step1_file_paths_to_names)
        self.pre_db = self.ord_db  ### 好像用不到不過為求統一性，還是指定一下好了
        return self.ord_db, self.pre_db

    ####################################################################################################
    def build_img_db(self):
        ### 測map速度用， 這兩行純讀檔 不map， 要用的時候拿掉這兩行註解， 把兩行外有座map動作的地方都註解調， 結論是map不怎麼花時間， 是shuffle 的 buffer_size 設太大 花時間！
        # self.ord_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.file_format, shuffle=False)
        # self.pre_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.file_format, shuffle=False)

        file_names = self._build_file_name_db()
        byte_imgs = file_names.map(self.step0a_load_byte_img)

        if  (self.file_format == "bmp"): decoded_imgs = byte_imgs.map(self.step0b_decode_bmp)
        elif(self.file_format == "jpg"): decoded_imgs = byte_imgs.map(self.step0b_decode_jpg)
        elif(self.file_format == "png"): decoded_imgs = byte_imgs.map(self.step0b_decode_png)
        elif(self.file_format == "gif"): decoded_imgs = byte_imgs.map(self.step0b_decode_gif)

        self.ord_db = decoded_imgs.map(self.step1_uint8_resize)
        ### 測試 use_range 有沒有設成功
        # print("self.use_range:", self.use_range)
        # print("VALUE_RANGE.neg_one_to_one.value:", VALUE_RANGE.neg_one_to_one.value, self.use_range == VALUE_RANGE.neg_one_to_one.value)
        # print("VALUE_RANGE.zero_to_one.value:", VALUE_RANGE.zero_to_one.value, self.use_range == VALUE_RANGE.zero_to_one.value)
        if  (self.use_range == Range(-1,   1)): self.pre_db = decoded_imgs.map(self.step1_float32_resize_and_to_tanh)
        elif(self.use_range == Range( 0,   1)): self.pre_db = decoded_imgs.map(self.step1_float32_resize_and_to_01)
        elif(self.use_range == Range( 0, 255)): self.pre_db = decoded_imgs.map(self.step1_uint8_resize)
        elif(self.use_range is None): print("tf_data 忘記設定 db_in/use_gt_range 或 rec_hope_range 了！，你可能會看到 Dataset.zip() 的錯誤喔 ~ ")
        return self.ord_db, self.pre_db


    def build_mov_db(self):
        self.ord_db = self._build_file_name_db()
        self.ord_db = self.ord_db.map(self.step1_flow_load)
        self.pre_db = self._build_file_name_db()
        self.pre_db = self.pre_db.map(self.step1_flow_load)
        self.pre_db = self.pre_db.map(self.check_use_range_and_db_range_then_normalize_data)
        return self.ord_db, self.pre_db

    ###  mask+coord(先y再x) 3ch合併 的形式 (最原本的)
    def build_flow_db(self):
        ### 測map速度用， 這兩行純讀檔 不map， 要用的時候拿掉這兩行註解， 把兩行外有座map動作的地方都註解調， 結論是map不怎麼花時間， 是shuffle 的 buffer_size 設太大 花時間！
        # self.ord_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*.knpy", shuffle=False)
        # self.pre_db = tf.data.Dataset.list_files(self.ord_dir + "/" + "*.knpy", shuffle=False)
        self.ord_db = self._build_file_name_db()
        self.ord_db = self.ord_db.map(self.step1_flow_load)

        self.pre_db = self._build_file_name_db()
        self.pre_db = self.pre_db.map(self.step1_flow_load)
        self.pre_db = self.pre_db.map(self.check_use_range_and_db_range_then_normalize_data)
        return self.ord_db, self.pre_db

    def build_wc_db(self):
        self.ord_db = self._build_file_name_db()
        self.ord_db = self.ord_db.map(self.step1_wc_load)

        self.pre_db = self._build_file_name_db()
        self.pre_db = self.pre_db.map(self.step1_wc_load)
        self.pre_db = self.pre_db.map(self.step2_M_clip_and_W_normalize)
        return self.ord_db, self.pre_db

    ###  mask1ch, coord(先y再x) 2ch 的形式
    def build_mask_coord_db(self):
        self.ord_db = self._build_file_name_db()
        self.ord_db = self.ord_db.map(self.step1_flow_load)

        self.pre_db = self._build_file_name_db()
        self.pre_db = self.pre_db.map(self.step1_flow_load)
        self.pre_db = self.pre_db.map(self.step2_M_clip_and_C_normalize)
        return self.ord_db, self.pre_db

    def build_wc_db_try_mul_M(self):
        self.ord_db = self._build_file_name_db()
        self.ord_db = self.ord_db.map(self.step1_wc_load)

        self.pre_db = self._build_file_name_db()
        self.pre_db = self.pre_db.map(self.step1_wc_load)
        self.pre_db = self.pre_db.map(self.step2_M_clip_and_W_normalize_try_mul_M)
        return self.ord_db, self.pre_db

    def build_mask_coord_db_try_mul_M(self):
        self.ord_db = self._build_file_name_db()
        self.ord_db = self.ord_db.map(self.step1_flow_load)

        self.pre_db = self._build_file_name_db()
        self.pre_db = self.pre_db.map(self.step1_flow_load)
        self.pre_db = self.pre_db.map(self.step2_M_clip_and_C_normalize_try_mul_M)
        return self.ord_db, self.pre_db

    ###  mask1ch， 目前好像沒用到， 這是在用 車子db 測試的mask時候 用的， 測試完後好像就沒再用到了
    def build_mask_db(self):
        file_names = self._build_file_name_db()
        if(self.file_format != "knpy"):
            byte_imgs = file_names.map(self.step0a_load_byte_img)
            ### 處理 format
            if  (self.file_format == "jpg"): decoded_imgs = byte_imgs.map(self.step0b_decode_jpg)
            elif(self.file_format == "gif"): decoded_imgs = byte_imgs.map(self.step0b_decode_gif)

            ### 處理 range
            ### db_range先寫 0~255 的case， 有遇到 db_range 0~1的case的話再去加寫
            if(self.db_range == VALUE_RANGE.img_range.value):
                self.ord_db = decoded_imgs.map(self.step1_uint8_resize)
                if  (self.use_range == VALUE_RANGE.neg_one_to_one.value): self.pre_db = decoded_imgs.map(self.step1_float32_resize_and_to_tanh)
                elif(self.use_range == VALUE_RANGE.zero_to_one.value):    self.pre_db = decoded_imgs.map(self.step1_float32_resize_and_to_01)
                elif(self.use_range == VALUE_RANGE.img_range.value):      self.pre_db = decoded_imgs.map(self.step1_uint8_resize)
                elif(self.use_range is None): print("tf_data 忘記設定 in/use_gt_range 或 rec_hope_range 了！，你可能會看到 Dataset.zip() 的錯誤喔 ~ ")
        elif(self.file_format == "knpy"):
            self.build_flow_db()
            self.ord_db = self.ord_db.map(self.step2_flow_extract_just_mask)
            self.pre_db = self.pre_db.map(self.step2_flow_extract_just_mask)


        ### 處理channel數
        if(self.pre_db.element_spec.shape[2] == 3): self.pre_db = self.pre_db.map(self._3ch_get_1ch)
        return self.ord_db, self.pre_db


####################################################################################################
####################################################################################################
class tf_Datapipline_factory_builder():
    def __init__(self, tf_pipline_factory=None):
        if(tf_pipline_factory is None): self.tf_pipline_factory = tf_Datapipline_factory()
        else:                   self.tf_pipline_factory = tf_pipline_factory

    ### 建立empty tf_pipline_factory
    def build(self):
        return self.tf_pipline_factory

    def set_factory(self, ord_dir, file_format, img_resize, db_range, use_range):
        print("DB ord_dir:", ord_dir)
        self.tf_pipline_factory.ord_dir      = ord_dir
        self.tf_pipline_factory.file_format  = file_format
        self.tf_pipline_factory.img_resize   = img_resize
        self.tf_pipline_factory.db_range     = db_range
        self.tf_pipline_factory.use_range    = use_range
        return self


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
class tf_Data:   ### 以上 以下 都是為了設定這個物件
    def __init__(self):
        self.db_obj           = None
        self.batch_size       = None
        self.train_shuffle    = None

        self.img_resize       = None

        self.use_in_range = None
        self.use_gt_range = None
        self.use_rec_hope_range = None

        self.train_name_db    = None
        self.train_in_db      = None
        self.train_in_db_pre  = None
        self.train_gt_db      = None
        self.train_gt_db_pre  = None
        self.train_db_combine = None
        self.train_amount     = None

        self.test_name_db     = None
        self.test_in_db       = None
        self.test_in_db_pre   = None
        self.test_gt_db       = None
        self.test_gt_db_pre   = None
        self.test_db_combine = None
        self.test_amount      = None

        self.see_name_db     = None
        self.see_in_db        = None
        self.see_in_db_pre    = None
        self.see_gt_db        = None
        self.see_gt_db_pre    = None
        self.see_amount       = None

        self.rec_hope_train_db     = None
        self.rec_hope_train_db_pre = None
        self.rec_hope_test_db      = None
        self.rec_hope_test_db_pre  = None
        self.rec_hope_see_db       = None
        self.rec_hope_see_db_pre   = None

        ### 最主要是再 step7 unet generate image 時用到，但我覺得可以改寫！所以先註解掉了！
        # self.in_format          = None
        # self.gt_format          = None

########################################################################################################################################
class tf_Data_init_builder:
    def __init__(self, tf_data=None):
        if(tf_data is None): self.tf_data = tf_Data()
        else:                self.tf_data = tf_data

    def set_basic(self, db_obj, batch_size=1, train_shuffle=True):
        self.tf_data.db_obj        = db_obj
        self.tf_data.batch_size    = batch_size
        self.tf_data.train_shuffle = train_shuffle
        return self

    def set_data_use_range(self, use_in_range, use_gt_range, use_rec_hope_range=Range(0, 255)):
    # def set_data_use_range(self, use_in_range=Range(0, 1), use_gt_range=Range(0, 1), use_rec_hope_range=Range(0, 255), in_min=None, in_max=None, gt_min=None, gt_max=None, rec_hope_min=None, rec_hope_max=None):
        self.tf_data.use_in_range = use_in_range
        self.tf_data.use_gt_range = use_gt_range
        self.tf_data.use_rec_hope_range = use_rec_hope_range
        # self.tf_data.in_max       = in_max
        # self.tf_data.in_min       = in_min
        # self.tf_data.gt_min       = gt_min
        # self.tf_data.gt_max       = gt_max
        # self.tf_data.rec_hope_min = rec_hope_min
        # self.tf_data.rec_hope_max = rec_hope_max
        return self

    def set_img_resize(self, model_name):
        # print("doing tf_data resize according model_name")
        # print("self.tf_data.db_obj.h = ", self.tf_data.db_obj.h)
        # print("self.tf_data.db_obj.w = ", self.tf_data.db_obj.w)
        # print("math.ceil(self.tf_data.db_obj.h / 128) * 128 = ", math.ceil(self.tf_data.db_obj.h / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
        # print("math.ceil(self.tf_data.db_obj.w / 128) * 128 = ", math.ceil(self.tf_data.db_obj.w / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
        if  ("unet" in model_name.value):
            self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h / 128) * 128 , math.ceil(self.tf_data.db_obj.w / 128) * 128)  ### 128的倍數，且要是gt_img的兩倍大喔！
        elif("rect" in model_name.value or "justG" in model_name.value):
            self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h / 4) * 4, math.ceil(self.tf_data.db_obj.w / 4) * 4)  ### dis_img(in_img的大小)的大小且要是4的倍數
        return self

    def build_by_db_get_method(self):
        self.train_in_factory = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.train_in_dir, file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.train_gt_factory = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.train_gt_dir, file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()
        self.test_in_factory  = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.test_in_dir,  file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.test_gt_factory  = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.test_gt_dir,  file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()
        self.see_in_factory   = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.see_in_dir,   file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.see_gt_factory   = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.see_gt_dir,   file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()

        if("未指定" not in self.tf_data.db_obj.train_in2_dir): self.train_in2_factory = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.train_in2_dir, file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.train_gt2_dir): self.train_gt2_factory = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.train_gt2_dir, file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()
        if("未指定" not in self.tf_data.db_obj.test_in2_dir ): self.test_in2_factory  = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.test_in2_dir,  file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.test_gt2_dir ): self.test_gt2_factory  = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.test_gt2_dir,  file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()
        if("未指定" not in self.tf_data.db_obj.see_in2_dir  ): self.see_in2_factory   = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.see_in2_dir,   file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.see_gt2_dir  ): self.see_gt2_factory   = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.see_gt2_dir,   file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()

        self.rec_hope_train_factory = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_train_dir, file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_test_factory  = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_test_dir,  file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_see_factory   = tf_Datapipline_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_see_dir,   file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()


        if    (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_move_map):
            self.build_by_in_dis_gt_move_map()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord or
               self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord_pad or
               self.tf_data.db_obj.get_method == DB_GM.in_rec_gt_ord):
            self.build_by_in_img_and_gt_img_db()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_flow):          self.build_by_in_dis_gt_flow_or_wc(get_what="flow")
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc):            self.build_by_in_dis_gt_flow_or_wc(get_what="wc")
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc_try_mul_M):  self.build_by_in_dis_gt_wc_try_mul_M()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_img_gt_mask):          self.build_by_in_img_gt_mask()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_mask_coord):    self.build_by_in_dis_gt_mask_coord()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_wc_gt_flow):           self.build_by_in_wc_gt_flow()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_wc_gt_flow_try_mul_M): self.build_by_in_wc_gt_flow_try_mul_M()

        return self



    def _build_train_test_in_img_db(self):
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_in_db  , self.tf_data.train_in_db_pre = self.train_in_factory.build_img_db()
        self.tf_data.train_name_db, _                            = self.train_in_factory.build_name_db()

        self.tf_data.test_in_db  , self.tf_data.test_in_db_pre   = self.test_in_factory.build_img_db()
        self.tf_data.test_name_db, _                             = self.test_in_factory.build_name_db()

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

    def _train_in_gt_and_test_in_gt_combine_then_train_shuffle(self):
        ### 先 zip 再 batch == 先 batch 再 zip (已經實驗過了，詳細內容看 try_lots 的 try10_資料pipline囉)
        ### train_in,gt 打包
        self.tf_data.train_db_combine = tf.data.Dataset.zip((self.tf_data.train_in_db, self.tf_data.train_in_db_pre,
                                                             self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre,
                                                             self.tf_data.train_name_db))
        ### test_in,gt 打包
        self.tf_data.test_db_combine  = tf.data.Dataset.zip((self.tf_data.test_in_db, self.tf_data.test_in_db_pre,
                                                             self.tf_data.test_gt_db, self.tf_data.test_gt_db_pre,
                                                             self.tf_data.test_name_db))
        ### 先shuffle(只有 train_db 需要) 再 batch
        ### train shuffle
        # if(self.tf_data.train_shuffle): self.tf_data.train_db_combine = self.tf_data.train_db_combine.shuffle(int(self.tf_data.train_amount / 2))  ### shuffle 的 buffer_size 太大會爆記憶體，嘗試了一下大概 /1.8 左右ok這樣子~ 但 /2 應該比較保險！
        if(self.tf_data.train_shuffle): self.tf_data.train_db_combine = self.tf_data.train_db_combine.shuffle(200)  ### 在 kong_model2_lots_try/try10_資料pipline 有很棒的模擬， 忘記可以去參考看看喔！
        ### train 取 batch 和 prefetch(因為有做訓練，訓練中就可以先取下個data了)
        self.tf_data.train_db_combine = self.tf_data.train_db_combine.batch(self.tf_data.batch_size)       ### shuffle完 打包成一包包 batch
        self.tf_data.train_db_combine = self.tf_data.train_db_combine.prefetch(-1)                         ### -1 代表 AUTOTUNE：https://www.tensorflow.org/api_docs/python/tf/data#AUTOTUNE，我自己的觀察是 他會視系統狀況自動調速度， 所以比較不會 cuda load failed 這樣子 ~~bb
        # self.tf_data.train_db_combine = self.tf_data.train_db_combine.prefetch(self.tf_data.train_amount)  ### 反正就一個很大的數字， 可以穩定的用最高速跑， 但如果有再做別的事情可能會 cuda load failed 這樣子~ 如果設一個很大的數字， 觀察起來是會盡可能全速跑， 如果再做其他事情(看YT) 可能會　cuda error 喔～
        ### test 取 batch
        self.tf_data.test_db_combine  = self.tf_data.test_db_combine.batch(self.tf_data.batch_size)   ### shuffle完 打包成一包包 batch


class tf_Data_in_dis_gt_move_map_builder(tf_Data_init_builder):
    def build_by_in_dis_gt_move_map(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()


        ### 在拿move_map db 之前，要先去抓 max/min train_move，我是設計放 train_gt_dir 下的.npy，如果怕混淆 要改放.txt之類的都可以喔！
        ### 決定還是放在上一層好了，因為下面會用 get_db_amount 是算檔案數量的，雖然是去in_dir抓影像跟gt_dir沒關係，但還是怕有意外(以後忘記之類的)～放外面最安全囉！
        ### 且放外面容易看到可以提醒自己有這東西的存在覺得ˊ口ˋ
        if(os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy") and
           os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")):
            self.tf_data.gt_max = np.load(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy")
            self.tf_data.gt_min = np.load(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")
        else:  ### 如果.npy不存在，就去重新找一次 max/min train_move，找完也順便存一份給之後用囉！
            print("因為現在已經存成.knpy，沒辦法抓 max/min train_move 囉！麻煩先去以前的dataset撈出來啦！")
            ### 偷懶可以把 .npy 放同個資料夾，把註解拿掉就可以順便求囉！只是因為這有return，所以還是要重新執行一次才會完整跑完喔～
            # move_maps = get_dir_moves(self.tf_data.db_obj.train_gt_dir)
            # self.tf_data.gt_max = move_maps.max()
            # self.tf_data.gt_min = move_maps.min()
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../max_train_move", self.tf_data.max_train_move)
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../min_train_move", self.tf_data.min_train_move)
            # print("self.tf_data.max_train_move",self.tf_data.max_train_move)
            # print("self.tf_data.min_train_move",self.tf_data.min_train_move)
            return

        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_mov_db()
        self.tf_data.test_gt_db , self.tf_data.test_gt_db_pre  = self.test_gt_factory.build_mov_db()


        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db  , self.tf_data.see_in_db_pre = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db, _                          = self.see_in_factory.build_name_db()

            self.tf_data.see_gt_db, self.tf_data.see_gt_db_pre = self.see_gt_factory.build_mov_db()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method2

        # take_num = 5
        # print(self.tf_data.train_max)
        # print(self.tf_data.train_min)
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
        #     # move_back = (move[0]+1)/2 * (train_max-train_min) + train_min  ### 想看train的部分用這行 且 註解掉下行
        #     move_back = (move_pre[0]+1)/2 * (self.tf_data.train_max-self.tf_data.train_min) + self.tf_data.train_min    ### 想看test 的部分用這行 且 註解掉上行
        #     move_back_bgr = method2(move_back[...,0], move_back[...,1],1)
        #     ax[ax_i].imshow(move_back_bgr)
        #     plt.show()
        #     plt.close()
        #########################################################################################################################################
        return self


class tf_Data_in_dis_gt_img_builder(tf_Data_in_dis_gt_move_map_builder):
    def build_by_in_img_and_gt_img_db(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_img_db()
        self.tf_data.test_gt_db,  self.tf_data.test_gt_db_pre  = self.test_gt_factory .build_img_db()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        #########################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     train_in     = train_in[0]      ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0]  ### 值 0. ~ 1.
        #     train_gt     = train_gt[0]      ### 值 0  ~ 255
        #     train_gt_pre = train_gt_pre[0]  ### 值 0. ~ 1.

        #     fig, ax = plt.subplots(1, 4)
        #     fig.set_size_inches(15, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()
        #########################################################

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_name_db , _                          = self.see_in_factory.build_name_db()
            self.tf_data.see_in_db   , self.tf_data.see_in_db_pre = self.see_in_factory.build_img_db()
            self.tf_data.see_gt_db   , self.tf_data.see_gt_db_pre = self.see_gt_factory.build_img_db()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)
        return self
    ############################################################


class tf_Data_in_dis_gt_flow_or_wc_builder(tf_Data_in_dis_gt_img_builder):
    def build_by_in_dis_gt_flow_or_wc(self, get_what="flow"):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        if  (get_what == "flow"): self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_flow_db()
        elif(get_what == "wc"  ): self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_wc_db()

        if  (get_what == "flow"): self.tf_data.test_gt_db, self.tf_data.test_gt_db_pre = self.test_gt_factory.build_flow_db()
        elif(get_what == "wc"):   self.tf_data.test_gt_db, self.tf_data.test_gt_db_pre = self.test_gt_factory.build_wc_db()


        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   , self.tf_data.see_in_db_pre = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db , _                          = self.see_in_factory.build_name_db()

            if  (get_what == "flow"): self.tf_data.see_gt_db, self.tf_data.see_gt_db_pre = self.see_gt_factory.build_flow_db()
            elif(get_what == "wc"):   self.tf_data.see_gt_db, self.tf_data.see_gt_db_pre = self.see_gt_factory.build_wc_db()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)


        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db, self.tf_data.rec_hope_train_db_pre = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db,  self.tf_data.rec_hope_test_db_pre  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db,   self.tf_data.rec_hope_see_db_pre   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1

        # # for i in enumerate(self.tf_data.train_gt_db): pass
        # # print("train_gt_finish")

        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(5)):
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1 train_in"    ] = train_in[0].numpy()
        #     debug_dict[f"{i}--2-2 train_in_pre"] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 train_Mgt"    ] = train_gt[0][0].numpy()
        #     debug_dict[f"{i}--2-4 train_Mgt_pre"] = train_gt_pre[0][0].numpy()
        #     debug_dict[f"{i}--2-5 train_Wgt"    ] = train_gt[1][0].numpy()
        #     debug_dict[f"{i}--2-6 train_Wgt_pre"] = train_gt_pre[1][0].numpy()

        #     if(get_what == "flow"):
        #         train_gt_visual     = method1(train_gt[0, ..., 2]    , train_gt[0, ..., 1])
        #         train_gt_pre_visual = method1(train_gt_pre[0, ..., 2], train_gt_pre[0, ..., 1])

        #         fig, ax = plt.subplots(1, 4)
        #         fig.set_size_inches(15, 5)
        #         ax[0].imshow(train_in[0])
        #         ax[1].imshow(train_in_pre[0])
        #         ax[2].imshow(train_gt_visual)
        #         ax[3].imshow(train_gt_pre_visual)
        #         ax[2].imshow(train_gt[0])
        #         ax[3].imshow(train_gt_pre[0])

        #     elif(get_what == "wc"):
        #         fig, ax = plt.subplots(2, 5)
        #         fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #         ### ord vs pre
        #         ax[0, 0].imshow(train_in    [0])
        #         ax[0, 1].imshow(train_in_pre[0])

        #         ### W_ord vs W_pre
        #         ax[0, 2].imshow(train_gt    [0, ..., :3])
        #         ax[0, 3].imshow(train_gt_pre[0, ..., :3])

        #         ### Wx, Wy, Wz 看一下長什麼樣子
        #         ax[1, 0].imshow(train_gt_pre[0, ..., 0])
        #         ax[1, 1].imshow(train_gt_pre[0, ..., 1])
        #         ax[1, 2].imshow(train_gt_pre[0, ..., 2])

        #         ### M_ord vs M_pre
        #         ax[1, 3].imshow(train_gt    [0, ..., 3])
        #         ax[1, 4].imshow(train_gt_pre[0, ..., 3])
        #     fig.tight_layout()
        #     plt.show()
        #########################################################################################################################################
        return self

    def build_by_in_dis_gt_wc_try_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()
        # self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_wc_db()  ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        # self.tf_data.test_gt_db,   self.tf_data.test_gt_db_pre = self.test_gt_factory.build_wc_db()   ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_wc_db_try_mul_M()
        self.tf_data.test_gt_db, self.tf_data.test_gt_db_pre = self.test_gt_factory.build_wc_db_try_mul_M()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   , self.tf_data.see_in_db_pre = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db , _                          = self.see_in_factory.build_name_db()
            self.tf_data.see_gt_db, self.tf_data.see_gt_db_pre = self.see_gt_factory.build_wc_db_try_mul_M()
            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # from util import method1
            # for i in enumerate(self.tf_data.train_gt_db): pass
            # print("train_gt_finish")

            # for i, (see_in, see_in_pre, see_gt, see_gt_pre, name) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.batch(1), self.tf_data.see_in_db_pre.batch(1),
            #                                                                                                 self.tf_data.see_gt_db.batch(1), self.tf_data.see_gt_db_pre.batch(1),
            #                                                                                                 self.tf_data.see_name_db))):
            #     debug_dict[f"{i}--1-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--1-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--1-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--1-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--2-1 see_in"    ]  = see_in    [0].numpy()
            #     debug_dict[f"{i}--2-2 see_in_pre"]  = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--2-3 see_Mgt"    ] = see_gt    [0].numpy()
            #     debug_dict[f"{i}--2-4 see_Mgt_pre"] = see_gt_pre[0].numpy()
            #     debug_dict[f"{i}--2-5 see_Wgt"    ] = see_gt    [0].numpy()
            #     debug_dict[f"{i}--2-6 see_Wgt_pre"] = see_gt_pre[0].numpy()

            #     fig, ax = plt.subplots(2, 5)
            #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
            #     ### ord vs pre
            #     ax[0, 0].imshow(see_in    [0])
            #     ax[0, 1].imshow(see_in_pre[0])

            #     ### W_ord vs W_pre
            #     ax[0, 2].imshow(see_gt    [0, ..., :3])
            #     ax[0, 3].imshow(see_gt_pre[0, ..., :3])

            #     ### Wx, Wy, Wz 看一下長什麼樣子
            #     ax[1, 0].imshow(see_gt_pre[0, ..., 0])
            #     ax[1, 1].imshow(see_gt_pre[0, ..., 1])
            #     ax[1, 2].imshow(see_gt_pre[0, ..., 2])

            #     ### M_ord vs M_pre
            #     Mgt     = see_gt    [0, ..., 3:4]
            #     Mgt_pre = see_gt_pre[0, ..., 3:4]
            #     ax[1, 3].imshow(see_gt    [0, ..., 3:4])
            #     ax[1, 4].imshow(see_gt_pre[0, ..., 3:4])

            #     in_pre_w_Mgt_pre = see_in_pre[0] * Mgt_pre
            #     ax[0, 4].imshow(in_pre_w_Mgt_pre)

            #     fig.tight_layout()
            #     plt.show()
            ##########################################################################################################################################

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db, self.tf_data.rec_hope_train_db_pre = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db,  self.tf_data.rec_hope_test_db_pre  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db,   self.tf_data.rec_hope_see_db_pre   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # import os
        # for i in enumerate(self.tf_data.train_gt_db): pass
        # print("train_gt_finish")

        # # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(5)):
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine):
        #     # if(i < 620): continue
        #     # print(i)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1 train_in"    ]  = train_in    [0].numpy()
        #     debug_dict[f"{i}--2-2 train_in_pre"]  = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 train_Mgt"    ] = train_gt    [0].numpy()
        #     debug_dict[f"{i}--2-4 train_Mgt_pre"] = train_gt_pre[0].numpy()
        #     debug_dict[f"{i}--2-5 train_Wgt"    ] = train_gt    [0].numpy()
        #     debug_dict[f"{i}--2-6 train_Wgt_pre"] = train_gt_pre[0].numpy()

        #     fig, ax = plt.subplots(2, 5)
        #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #     ### ord vs pre
        #     in_ord = train_in    [0]
        #     in_pre = train_in_pre[0]
        #     ax[0, 0].imshow(in_ord)
        #     ax[0, 1].imshow(in_pre)

        #     ### W_ord vs W_pre
        #     W_ord = train_gt    [0, ..., :3]
        #     W_pre = train_gt_pre[0, ..., :3]
        #     ax[0, 2].imshow(W_ord)
        #     ax[0, 3].imshow(W_pre)

        #     ### Wx, Wy, Wz 看一下長什麼樣子
        #     Wx_pre = train_gt_pre[0, ..., 0]
        #     Wy_pre = train_gt_pre[0, ..., 1]
        #     Wz_pre = train_gt_pre[0, ..., 2]
        #     ax[1, 0].imshow(Wx_pre)
        #     ax[1, 1].imshow(Wy_pre)
        #     ax[1, 2].imshow(Wz_pre)

        #     ### M_ord vs M_pre
        #     Mgt     = train_gt    [0, ..., 3:4]
        #     Mgt_pre = train_gt_pre[0, ..., 3:4]
        #     ax[1, 3].imshow(Mgt)
        #     ax[1, 4].imshow(Mgt_pre)

        #     ### W_pre * M
        #     W_pre_w_Mgt_pre = W_pre * Mgt_pre
        #     ax[0, 4].imshow(W_pre_w_Mgt_pre)

        #     fig.tight_layout()
        #     if(os.path.isdir(self.tf_data.db_obj.check_train_gt_dir) is False): os.makedirs(self.tf_data.db_obj.check_train_gt_dir)
        #     plt.show()
        #     # plt.savefig(f"{self.tf_data.db_obj.check_train_gt_dir}/" + "%05i" % (i + 1) )
        #     # plt.close()
        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # import os
        # # for i in enumerate(self.tf_data.train_gt_db): pass
        # # print("train_gt_finish")

        # # for i, (test_in, test_in_pre, test_gt, test_gt_pre, name) in enumerate(self.tf_data.test_db_combine.take(5)):
        # for i, (test_in, test_in_pre, test_gt, test_gt_pre, name) in enumerate(self.tf_data.test_db_combine):
        #     debug_dict[f"{i}--1-1 test_in"    ] = test_in
        #     debug_dict[f"{i}--1-2 test_in_pre"] = test_in_pre
        #     debug_dict[f"{i}--1-3 test_gt"    ] = test_gt
        #     debug_dict[f"{i}--1-4 test_gt_pre"] = test_gt_pre

        #     debug_dict[f"{i}--2-1 test_in"    ]  = test_in    [0].numpy()
        #     debug_dict[f"{i}--2-2 test_in_pre"]  = test_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 test_Mgt"    ] = test_gt    [0].numpy()
        #     debug_dict[f"{i}--2-4 test_Mgt_pre"] = test_gt_pre[0].numpy()
        #     debug_dict[f"{i}--2-5 test_Wgt"    ] = test_gt    [0].numpy()
        #     debug_dict[f"{i}--2-6 test_Wgt_pre"] = test_gt_pre[0].numpy()

        #     fig, ax = plt.subplots(2, 5)
        #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #     ### ord vs pre
        #     ax[0, 0].imshow(test_in    [0])
        #     ax[0, 1].imshow(test_in_pre[0])

        #     ### W_ord vs W_pre
        #     ax[0, 2].imshow(test_gt    [0, ..., :3])
        #     ax[0, 3].imshow(test_gt_pre[0, ..., :3])

        #     ### Wx, Wy, Wz 看一下長什麼樣子
        #     ax[1, 0].imshow(test_gt_pre[0, ..., 0])
        #     ax[1, 1].imshow(test_gt_pre[0, ..., 1])
        #     ax[1, 2].imshow(test_gt_pre[0, ..., 2])

        #     ### M_ord vs M_pre
        #     ax[1, 3].imshow(test_gt    [0, ..., 3])
        #     ax[1, 4].imshow(test_gt_pre[0, ..., 3])
        #     fig.tight_layout()
        #     if os.path.isdir(self.tf_data.db_obj.check_test_gt_dir) is False: os.makedirs(self.tf_data.db_obj.check_test_gt_dir)
        #     # plt.show()
        #     plt.savefig(f"{self.tf_data.db_obj.check_test_gt_dir}/" + "%05i" % (i + 1) )
        #     plt.close()
        #########################################################################################################################################
        return self

class tf_Data_in_dis_gt_mask_coord_builder(tf_Data_in_dis_gt_flow_or_wc_builder):
    def build_by_in_dis_gt_mask_coord(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_mask_coord_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.test_gt_db, self.tf_data.test_gt_db_pre  = self.test_gt_factory.build_mask_coord_db()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0].numpy()
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0]
        #     train_in_pre      = train_in_pre[0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(1, 6)
        #     fig.set_size_inches(30, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt_mask)
        #     ax[3].imshow(train_gt_pre_mask)
        #     ax[4].imshow(train_gt_move_visual)
        #     ax[5].imshow(train_gt_pre_move_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db  , self.tf_data.see_in_db_pre = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db, _                          = self.see_in_factory.build_name_db()

            self.tf_data.see_gt_db , self.tf_data.see_gt_db_pre  = self.see_gt_factory.build_mask_coord_db()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)
            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.batch(1), self.tf_data.see_in_db_pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.batch(1), self.tf_data.see_gt_db_pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db, self.tf_data.rec_hope_train_db_pre = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db,  self.tf_data.rec_hope_test_db_pre  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db,   self.tf_data.rec_hope_see_db_pre   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

class tf_Data_in_wc_gt_flow_builder(tf_Data_in_dis_gt_mask_coord_builder):
    def build_by_in_wc_gt_flow(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_name_db, _                             = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db  , self.tf_data.train_in_db_pre  = self.train_in_factory .build_wc_db()

        self.tf_data.test_in_db  , self.tf_data.test_in_db_pre   = self.test_in_factory.build_wc_db()
        self.tf_data.test_name_db, _                             = self.test_in_factory.build_name_db()

        ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
        self.tf_data.train_in2_db , _                             = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db  , _                             = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db = tf.data.Dataset.zip((self.tf_data.train_in_db, self.tf_data.train_in2_db))
        self.tf_data.test_in_db  = tf.data.Dataset.zip((self.tf_data.test_in_db, self.tf_data.test_in2_db))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_mask_coord_db()
        self.tf_data.test_gt_db,  self.tf_data.test_gt_db_pre  = self.test_gt_factory .build_mask_coord_db()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     ''' 注意這裡的train_in 有多 dis_img 喔！
        #            train_in[0] 是 wc,      shape=(N, H, W, C)
        #            train_in[1] 是 dis_img, shape=(N, H, W, C)
        #     '''
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in[0]  ### [0]第一個是 取 wc, [1] 是取 dis_img
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0][0]  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     train_in_pre      = train_in_pre[0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(1, 6)
        #     fig.set_size_inches(30, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt_mask)
        #     ax[3].imshow(train_gt_pre_mask)
        #     ax[4].imshow(train_gt_move_visual)
        #     ax[5].imshow(train_gt_pre_move_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db  , self.tf_data.see_in_db_pre = self.see_in_factory.build_wc_db()
            self.tf_data.see_name_db, _                          = self.see_in_factory.build_name_db()

            ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
            self.tf_data.see_in2_db  , _                             = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db = tf.data.Dataset.zip((self.tf_data.see_in_db, self.tf_data.see_in2_db))

            self.tf_data.see_gt_db , self.tf_data.see_gt_db_pre  = self.see_gt_factory.build_mask_coord_db()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.batch(1), self.tf_data.see_in_db_pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.batch(1), self.tf_data.see_gt_db_pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db, self.tf_data.rec_hope_train_db_pre = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db,  self.tf_data.rec_hope_test_db_pre  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db,   self.tf_data.rec_hope_see_db_pre   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

    def build_by_in_wc_gt_flow_try_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_name_db, _                             = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db  , self.tf_data.train_in_db_pre  = self.train_in_factory .build_wc_db_try_mul_M()

        self.tf_data.test_in_db  , self.tf_data.test_in_db_pre   = self.test_in_factory.build_wc_db_try_mul_M()
        self.tf_data.test_name_db, _                             = self.test_in_factory.build_name_db()

        ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
        self.tf_data.train_in2_db , _                             = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db  , _                             = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db = tf.data.Dataset.zip((self.tf_data.train_in_db, self.tf_data.train_in2_db))
        self.tf_data.test_in_db  = tf.data.Dataset.zip((self.tf_data.test_in_db, self.tf_data.test_in2_db))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_mask_coord_db()
        self.tf_data.test_gt_db,  self.tf_data.test_gt_db_pre  = self.test_gt_factory .build_mask_coord_db()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     ''' 注意這裡的train_in 有多 dis_img 喔！
        #            train_in[0] 是 wc,      shape=(N, H, W, C)
        #            train_in[1] 是 dis_img, shape=(N, H, W, C)
        #     '''
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in[0]  ### [0]第一個是 取 wc, [1] 是取 dis_img
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0][0]  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     train_in_pre      = train_in_pre[0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(1, 6)
        #     fig.set_size_inches(30, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt_mask)
        #     ax[3].imshow(train_gt_pre_mask)
        #     ax[4].imshow(train_gt_move_visual)
        #     ax[5].imshow(train_gt_pre_move_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db  , self.tf_data.see_in_db_pre = self.see_in_factory.build_wc_db_try_mul_M()
            self.tf_data.see_name_db, _                          = self.see_in_factory.build_name_db()

            ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
            self.tf_data.see_in2_db  , _                             = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db = tf.data.Dataset.zip((self.tf_data.see_in_db, self.tf_data.see_in2_db))

            self.tf_data.see_gt_db , self.tf_data.see_gt_db_pre  = self.see_gt_factory.build_mask_coord_db()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.batch(1), self.tf_data.see_in_db_pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.batch(1), self.tf_data.see_gt_db_pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db, self.tf_data.rec_hope_train_db_pre = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db,  self.tf_data.rec_hope_test_db_pre  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db,   self.tf_data.rec_hope_see_db_pre   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

class tf_Data_in_img_gt_mask_builder(tf_Data_in_wc_gt_flow_builder):
    def build_by_in_img_gt_mask(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()


        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db, self.tf_data.train_gt_db_pre = self.train_gt_factory.build_mask_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.test_gt_db, self.tf_data.test_gt_db_pre = self.test_gt_factory.build_mask_db()

        print("self.tf_data.train_in_db", self.tf_data.train_in_db)
        print("self.tf_data.train_in_db_pre", self.tf_data.train_in_db_pre)
        print("self.tf_data.train_gt_db", self.tf_data.train_gt_db)
        print("self.tf_data.train_gt_db_pre", self.tf_data.train_gt_db_pre)

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     # print(train_in.numpy().shape)       ### (10, 768, 768, 3)
        #     train_in     = train_in[0]          ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0]      ### 值 0. ~ 1.
        #     print("train_in", train_in.numpy().dtype)       ### uint8
        #     print("train_in", train_in.numpy().shape)       ### (h, w, 3)
        #     print("train_in", train_in.numpy().min())       ### 0
        #     print("train_in", train_in.numpy().max())       ### 255
        #     print("train_in_pre", train_in_pre.numpy().dtype)   ### float32
        #     print("train_in_pre", train_in_pre.numpy().shape)   ### (h, w, 3)
        #     print("train_in_pre", train_in_pre.numpy().min())   ### 0.0
        #     print("train_in_pre", train_in_pre.numpy().max())   ### 1.0

        #     # print(train_gt.numpy().shape)       ### (10, 768, 768, 3)
        #     train_gt     = train_gt[0]          ### 值 0. ~ 1.
        #     train_gt_pre = train_gt_pre[0]      ### 值 0. ~ 1.
        #     print("train_gt", train_gt.numpy().dtype)       ### float32
        #     print("train_gt", train_gt.numpy().shape)       ### (h, w, 3)
        #     print("train_gt", train_gt.numpy().min())       ### 0.0
        #     print("train_gt", train_gt.numpy().max())       ### 1.0
        #     print("train_gt_pre", train_gt_pre.numpy().dtype)   ### float32
        #     print("train_gt_pre", train_gt_pre.numpy().min())   ### 0.0
        #     print("train_gt_pre", train_gt_pre.numpy().max())   ### 1.0


        #     fig, ax = plt.subplots(1, 4)
        #     fig.set_size_inches(15, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()

        '''
        還沒弄see
        '''
        return self

class tf_Data_builder(tf_Data_in_img_gt_mask_builder):
    def build(self):
        print(f"TF_data_builder build finish")
        return self.tf_data

### 因為我們還需要根據 使用的model(其實就是看model_name) 來決定如何resize，所以就不在這邊 先建構好 許多tf_data物件囉！


if(__name__ == "__main__"):
    from step09_d_KModel_builder_combine_step789 import MODEL_NAME, KModel_builder
    from step06_a_datas_obj import *

    start_time = time.time()

    # db_obj = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="bmp", db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=batch_size-1, train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    # db_obj = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page, DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="...", db_gt_range=Range(...)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_unet()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    ''' mask_flow 3ch合併 的形式'''
    # db_obj = Dataset_builder().set_basic(DB_C.type8_blender                      , DB_N.blender_os_hw768      , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", db_in_range=Range(0, 255), gt_format="knpy", db_gt_range=Range(0, 1), rec_hope_format="jpg", db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True).build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(model_obj.model_name).set_data_use_range(use_in_range=Range(-1, 1), use_gt_range=Range(-1, 1), use_rec_hope_range=Range(0, 255)).build_by_db_get_method().build()

    '''in_img, gt_mask'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_try_segmentation.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(model_obj.model_name).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(model_obj.model_name).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(model_obj.model_name).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(model_obj.model_name).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    db_obj = type8_blender_wc_try_mul_M.build()
    print(db_obj)
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(model_obj.model_name).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    print(time.time() - start_time)
    print("finish")
