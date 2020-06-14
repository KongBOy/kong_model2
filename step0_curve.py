from step0_access_path import access_path
import sys 
sys.path.append("kong_util")

### 步驟一：中間畫線
import os 
import cv2
import numpy as np
from build_dataset_combine import Page_num, Crop_use_center, Resize_hw, Select_lt_rt_ld_rd_train_test_see



        
produce_curve_dir = access_path + "/datasets/cut_os_book/produce_curve/" 

step0_dir = produce_curve_dir + "0_os_photo/1_photoimpact_black_bg"
step1_dir = produce_curve_dir + "1_page_num_ok"
step2_dir = produce_curve_dir + "2_crop_use_center"
step3_dir = produce_curve_dir + "3_resize_w=332,h=500"
step3big_dir = produce_curve_dir + "3_resize_w=396,h=600"
final_dir = produce_curve_dir + "final_os_book_1532data"
final_big_dir = produce_curve_dir + "final_os_book_1532data_big"
final_400_dir = produce_curve_dir + "final_os_book_400data"
final_800_dir = produce_curve_dir + "final_os_book_800data"
###################################################################################################################
Page_num( step0_dir , step1_dir)

Crop_use_center( step1_dir , step2_dir, center_xy_file = produce_curve_dir + "center.txt",
        crop_window_size_w=1168, crop_window_size_h=1800,
         lt_s_y =  50, ### left_top_shift_y
         lt_s_x =  20, ### left_top_shift_x
         lt_a_h = -90, ### left_top_add_h
         rt_s_y =  20, ### right_top_shift_y
         rt_s_x = -50, ### right_top_shift_x
         rt_a_h = -110, ### right_top_add_h
         ld_s_y =  10, ### left_down_shift_y
         ld_s_x =  80, ### left_down_shift_y
         ld_a_h =   0, ### left_down_add_height
         rd_s_y = -10, ### right_down_shift_y
         rd_s_x =  60, ### right_down_shift_x
         rd_a_h =  20  ### right_down_add_height):
         )    

### 原始版本(1532data)
# Resize_hw(step2_dir, step3_dir ,width=332, height=500, method="cv2")
# ### 這要和 straight的對應到喔！建議直接複製過去～然後這裡是"dis_imgs"喔！
# Select_lt_rt_ld_rd_train_test_see( step3_dir, final_dir, result_dir_name="dis_imgs", 
#                            train_4page_index_list=range(387), test_4page_index_list=[101,102,103,104], see_train_4page_index_list=[1,2,4,6])



### big版本(1532data)     
Resize_hw(step2_dir, step3big_dir ,width=396, height=600, method="cv2")   ### 註解拿掉就代表用big版
### 這要和 straight的對應到喔！建議直接複製過去～然後這裡是"dis_imgs"喔！
Select_lt_rt_ld_rd_train_test_see( step3big_dir, final_big_dir, result_dir_name="dis_imgs", 
                           train_4page_index_list=range(387), test_4page_index_list=[101,102,103,104], see_train_4page_index_list=[1,2,4,6])

### 800data版本
# Select_lt_rt_ld_rd_train_test_see( step3_dir, final_400_dir, result_dir_name="dis_imgs", 
#                            train_4page_index_list=range(204), test_4page_index_list=[101,102,103,104], see_train_4page_index_list=[1,2,4,6])

### 400data版本
### 下面用的是共103個4page_train，剛好最後3個train跟test重複到會被刪掉就剛好100個4page_train囉！
# Select_lt_rt_ld_rd_train_test_see( step3_dir, final_800_dir, result_dir_name="dis_imgs", 
#                            train_4page_index_list=range(1,104), test_4page_index_list=[101,102,103,104], see_train_4page_index_list=[1,2,4,6])

