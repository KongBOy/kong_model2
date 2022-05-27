"""
利用 step06_a_datas_obj的資訊 來建立 tf_data
"""
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from step06_a_datas_obj import DB_C, DB_N, DB_GM, Dataset_builder, Range
from step06_b_data_pipline import tf_Data, tf_Data_element_factory_builder

from kong_util.util import get_db_amount

import tensorflow as tf

tf.keras.backend.set_floatx('float32')  ### 這步非常非常重要！用了才可以加速！

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

    def set_img_resize(self, img_resize):
        self.tf_data.img_resize = img_resize
        # '''
        # tuple / list 或 <enum 'MODEL_NAME'>
        # '''
        # # print(type(img_resize))

        # ### tuple / list 的 case
        # if(type(img_resize) == type( () ) or type(img_resize) == type( [] )):
        #     self.tf_data.img_resize = img_resize
        # ### <enum 'MODEL_NAME'> 的 case
        # else:
        #     # print("doing tf_data resize according model_name")
        #     # print("self.tf_data.db_obj.h = ", self.tf_data.db_obj.h)
        #     # print("self.tf_data.db_obj.w = ", self.tf_data.db_obj.w)
        #     # print("math.ceil(self.tf_data.db_obj.h / 128) * 128 = ", math.ceil(self.tf_data.db_obj.h / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
        #     # print("math.ceil(self.tf_data.db_obj.w / 128) * 128 = ", math.ceil(self.tf_data.db_obj.w / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
        #     if  ("unet" in img_resize.value):
        #         self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h / 128) * 128 , math.ceil(self.tf_data.db_obj.w / 128) * 128)  ### 128的倍數，且要是gt_img的兩倍大喔！
        #     elif("rect" in img_resize.value or "justG" in img_resize.value):
        #         self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h / 4) * 4, math.ceil(self.tf_data.db_obj.w / 4) * 4)  ### dis_img(in_img的大小)的大小且要是4的倍數
        return self

    def _build_by_db_get_method_init(self):
        self.train_in_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_in_dir, file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range, db_ch_ranges=self.tf_data.db_obj.in_ch_ranges).build()
        self.train_gt_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_gt_dir, file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range, db_ch_ranges=self.tf_data.db_obj.gt_ch_ranges).build()
        self.test_in_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_in_dir,  file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range, db_ch_ranges=self.tf_data.db_obj.in_ch_ranges).build()
        self.test_gt_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_gt_dir,  file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range, db_ch_ranges=self.tf_data.db_obj.gt_ch_ranges).build()
        self.see_in_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_in_dir,   file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range, db_ch_ranges=self.tf_data.db_obj.in_ch_ranges).build()
        self.see_gt_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_gt_dir,   file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range, db_ch_ranges=self.tf_data.db_obj.gt_ch_ranges).build()

        if("未指定" not in self.tf_data.db_obj.train_in2_dir): self.train_in2_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_in2_dir, file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range, db_ch_ranges=self.tf_data.db_obj.in2_ch_ranges).build()
        if("未指定" not in self.tf_data.db_obj.train_gt2_dir): self.train_gt2_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_gt2_dir, file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range, db_ch_ranges=self.tf_data.db_obj.gt2_ch_ranges).build()
        if("未指定" not in self.tf_data.db_obj.test_in2_dir ): self.test_in2_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_in2_dir,  file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range, db_ch_ranges=self.tf_data.db_obj.in2_ch_ranges).build()
        if("未指定" not in self.tf_data.db_obj.test_gt2_dir ): self.test_gt2_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_gt2_dir,  file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range, db_ch_ranges=self.tf_data.db_obj.gt2_ch_ranges).build()
        if("未指定" not in self.tf_data.db_obj.see_in2_dir  ): self.see_in2_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_in2_dir,   file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range, db_ch_ranges=self.tf_data.db_obj.in2_ch_ranges).build()
        if("未指定" not in self.tf_data.db_obj.see_gt2_dir  ): self.see_gt2_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_gt2_dir,   file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range, db_ch_ranges=self.tf_data.db_obj.gt2_ch_ranges).build()

        self.rec_hope_train_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_train_dir, file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_test_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_test_dir,  file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_see_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_see_dir,   file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()

    def _build_train_test_in_img_db(self):
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_in_db   = self.train_in_factory.build_img_db()
        self.tf_data.train_name_db = self.train_in_factory.build_name_db()

        self.tf_data.test_in_db    = self.test_in_factory.build_img_db()
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

    def _train_in_gt_and_test_in_gt_combine_then_train_shuffle(self):
        ### 先 zip 再 batch == 先 batch 再 zip (已經實驗過了，詳細內容看 try_lots 的 try10_資料pipline囉)
        ### train_in,gt 打包
        self.tf_data.train_db_combine = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in_db.pre,
                                                             self.tf_data.train_gt_db.ord, self.tf_data.train_gt_db.pre,
                                                             self.tf_data.train_name_db.ord))
        ### test_in,gt 打包
        self.tf_data.test_db_combine  = tf.data.Dataset.zip((self.tf_data.test_in_db.ord, self.tf_data.test_in_db.pre,
                                                             self.tf_data.test_gt_db.ord, self.tf_data.test_gt_db.pre,
                                                             self.tf_data.test_name_db.ord))
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
