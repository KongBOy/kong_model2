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

    def build_by_db_get_method(self):
        self.train_in_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_in_dir, file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.train_gt_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_gt_dir, file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()
        self.test_in_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_in_dir,  file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.test_gt_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_gt_dir,  file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()
        self.see_in_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_in_dir,   file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.see_gt_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_gt_dir,   file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()

        if("未指定" not in self.tf_data.db_obj.train_in2_dir): self.train_in2_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_in2_dir, file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.train_gt2_dir): self.train_gt2_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_gt2_dir, file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()
        if("未指定" not in self.tf_data.db_obj.test_in2_dir ): self.test_in2_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_in2_dir,  file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.test_gt2_dir ): self.test_gt2_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_gt2_dir,  file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()
        if("未指定" not in self.tf_data.db_obj.see_in2_dir  ): self.see_in2_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_in2_dir,   file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.see_gt2_dir  ): self.see_gt2_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_gt2_dir,   file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()

        self.rec_hope_train_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_train_dir, file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_test_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_test_dir,  file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_see_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_see_dir,   file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()


        if    (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_move_map):
            self.build_by_in_dis_gt_move_map()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord or
               self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord_pad or
               self.tf_data.db_obj.get_method == DB_GM.in_rec_gt_ord):
            self.build_by_in_img_and_gt_img_db()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_mask_coord):    self.build_by_in_dis_gt_mask_coord()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_flow):          self.build_by_in_dis_gt_flow_or_wc(get_what="flow")
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc):            self.build_by_in_dis_gt_flow_or_wc(get_what="wc")
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc_try_mul_M):  self.build_by_in_dis_gt_wc_try_mul_M()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_img_gt_mask):          self.build_by_in_img_gt_mask()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_wc_gt_flow):           self.build_by_in_wc_gt_flow()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_wc_gt_flow_try_mul_M): self.build_by_in_wc_gt_flow_try_mul_M()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc_flow_try_mul_M): self.build_by_in_dis_gt_wc_flow_try_mul_M()

        return self



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


from step06_c5_in_I_gt_W_F import tf_Data_in_dis_gt_wc_flow_builder

class tf_Data_builder(tf_Data_in_dis_gt_wc_flow_builder):
    def build(self):
        print(f"TF_data_builder build finish")
        return self.tf_data

########################################################################################################################################
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
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(-1, 1), use_gt_range=Range(-1, 1), use_rec_hope_range=Range(0, 255)).build_by_db_get_method().build()

    '''in_img, gt_mask'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_try_segmentation.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d_in_W_and_I_gt_F.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    db_obj = type8_blender_kong_doc3d_in_I_gt_MC.build()
    print(db_obj)
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 448, 448) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    # print("here")
    # img1 = tf_data.train_db_combine.take(10)


    ### 雖然來源相同， 不過兩個iter是獨立運作的
    # db1 = tf_data.train_db_combine
    # db2 = tf_data.train_db_combine

    # iter1 = iter(db1)
    # iter2 = iter(db2)

    # data1 = next(iter1)
    # data2 = next(iter2)
    # plt.figure()
    # plt.imshow(data1[0][1][0][..., :3])
    # plt.figure()
    # plt.imshow(data2[0][1][0][..., :3])
    # plt.show()
    # print("data1 == data2", data1)


    # print(len(img1))
    # print("here~ ", img1)
    # img2 = tf_data.train_db_combine.take(10)
    # print("here~~", img2)
    # ### 有shuffle蒔 從這裡知道 take 不花時間， 有點像定義動作的概念

    # ### 有shuffle蒔 從下面知道， 真正用for去取資料的時候才花時間～ 用for 才是真正去取資料喔！ take()不是， take()只是定義動作而已～
    # plt.figure()
    # for data in img1:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here1")

    # plt.figure()
    # for data in img2:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here2")

    # plt.figure()
    # for data in img2:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here3")
    # it1 = iter(img1)
    # it2 = iter(img2)

    # data = next(it2)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # data = next(it2)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # data = next(it1)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # plt.figure()
    # for data in img1:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here1")

    # plt.figure()
    # it1 = iter(img1)
    # data = next(it1)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])
    # print("here1")

    # i = 0
    # it1 = iter(img1.repeat(2))
    # while(True):
    #     data = next(it1)
    #     i = i + 1
    #     print(i)

    # plt.show()
    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_dis_wc_flow_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    print(time.time() - start_time)
    print("finish")
