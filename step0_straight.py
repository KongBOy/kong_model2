from step0_access_path import data_access_path
import sys
sys.path.append("kong_util")

### 步驟一：中間畫線
from build_dataset_combine import Page_num, \
                                  Find_ltrd_and_crop, \
                                  Save_as_gray, \
                                  Center_draw_photo_frame, \
                                  Photo_frame_padding, \
                                  Pick_manually, \
                                  Crop_use_center, \
                                  Resize_hw, \
                                  Select_lt_rt_ld_rd_train_test_see



thick_v = 280
thick_h = 100

width  = 2481; height = 3590  #3508

produce_curve_dir = data_access_path + "/datasets/cut_os_book/produce_straight/" 

step00_dir = produce_curve_dir + "00_c_jpg"
step01_dir = produce_curve_dir + "01_page_num_ok"
step02_dir = produce_curve_dir + "02_crop_ok"
step03_dir = produce_curve_dir + "03_gray_ok"
step04_dir = produce_curve_dir + "04_photo_frame"
step05_dir = produce_curve_dir + "05_center_std_line"
step06_dir = produce_curve_dir + "06_Photo_frame_padding"
step07_dir = produce_curve_dir + "07_Pick_manually"
step08_dir = produce_curve_dir + "08_Pick_page_num_ok-this_for_pdf"
step09_dir = produce_curve_dir + "09_Crop_use_center"
step10_dir = produce_curve_dir + "10_resize_w=332,h=500"
step10focus_dir = produce_curve_dir + "10focus_Crop_focus"
step11focus_dir = produce_curve_dir + "11focus_resize_w=332,h=500"
step10big_dir = produce_curve_dir + "10big_resize_w=396,h=600"
final_dir  = produce_curve_dir + "final_os_book_1532data"
final_focus_dir = produce_curve_dir + "final_os_book_1532data_focus"
final_big_dir   = produce_curve_dir + "final_os_book_1532data_big"
final_800_dir   = produce_curve_dir + "final_os_book_800data"
final_400_dir   = produce_curve_dir + "final_os_book_400data"
###################################################################################################################


# Page_num                (step00_dir, step01_dir)
# Find_ltrd_and_crop      (step01_dir, step02_dir, padding=0, crop_according_lr_page=True, odd_x_shift=150, even_x_shift=-30)  ### 不要padding了，因為印表機會自動pad！在pad會太多！印表機左右pad:100px左右，上下pad:50px左右
# Save_as_gray            (step02_dir, step03_dir)
# Center_draw_photo_frame (step03_dir, step04_dir, thick_v=thick_v, thick_h=thick_h, color=(255,255,255))
# Center_draw_photo_frame (step04_dir, step05_dir, thick_v=2,       thick_h=2,       color=(240,240,240))

# Photo_frame_padding(step05_dir,step06_dir,left_pad=100, top_pad=50, right_pad=100, down_pad=50) ### 這padding 是可以自己決定 上、下、左、右 pad多少，Find_ltrd_and_crop裡沒寫這麼general喔！
# pick_manually = [5, 6, 7, 9, 10, 12, 14, 17, 20, 21,
#                  23, 24, 25, 26, 27, 29, 30, 31, 32, 33,
#                  34, 36, 37, 38, 39, 40, 41, 42, 44, 45,
#                  49, 50, 53, 59, 62, 63, 64, 65, 66, 67,
#                  68, 71, 75, 76, 78, 82, 83, 87, 88, 89,
#                  90, 91, 92, 93, 94, 108, 110, 113, 114, 118, 119,
#                  120, 121, 122, 123, 124, 125, 130, 132, 137, 140,
#                  141, 142, 152, 154, 157, 158, 159, 160, 161, 163,
#                  166, 167, 169, 170, 171, 173, 178, 179, 180, 191,
#                  192, 194, 200, 203, 205, 207, 209, 210, 211, 212,
#                  213, 214, 219, 221, 229, 231, 233, 244, 245, 246,
#                  249, 250, 251, 252, 253, 254, 257, 258, 259, 262,
#                  263, 264, 266, 268, 271, 272, 280, 283, 285, 286,
#                  288, 289, 293, 295, 298, 304, 312, 313, 314, 315,
#                  316, 326, 327, 328, 329, 331, 334, 340, 342, 343,
#                  344, 345, 346, 347, 348, 358, 361, 362, 363, 364,
#                  367, 368, 370, 371, 372, 373, 376, 377, 378, 379,
#                  384, 385, 386, 387, 388, 389, 391, 392, 400, 402,
#                  404, 406, 408, 409, 410, 411, 412, 413, 414, 415,
#                  417, 418, 419, 420, 421, 425, 426, 429, 430, 432,
#                  433, 435, 439, 441, 442, 444, 445, 448, 449, 450,
#                  451, 452, 453, 454, 455, 459, 460, 461, 468, 469,
#                  471, 472, 473, 475, 476, 478, 479, 481, 484, 485,
#                  486, 487, 488, 489, 490, 491, 494, 497, 499, 501,
#                  503, 506, 507, 508, 512, 513, 516, 517, 520, 522,
#                  524, 525, 530, 531, 532, 533, 534, 540, 541, 542,
#                  544, 545, 546, 547, 548, 549, 550, 554, 556, 557,
#                  558, 559, 562, 563, 565, 566, 568, 569, 570, 571,
#                  573, 575, 576, 577, 579, 580, 581, 582, 583, 584,
#                  585, 586, 587, 588, 589, 590, 596, 597, 598, 599,
#                  600, 602, 603, 604, 605, 606, 607, 609, 617, 618,
#                  619, 620, 622, 623, 624, 625, 627, 628, 630, 631,
#                  632, 633, 634, 635, 636, 637, 638, 639, 642, 643,
#                  644, 645, 646, 647, 648, 649, 651, 652, 653, 656,
#                  657, 658, 659, 662, 664, 665, 666, 667, 668, 670,
#                  671, 672, 673, 674, 675, 676, 677, 678, 680, 681,
#                  682, 683, 684, 685, 686, 687, 688, 689, 690, 691,
#                  693, 694, 695, 696, 700, 702, 704, 706, 708, 710,
#                  712, 714, 716, 718, 720, 722]


# Pick_manually      (step06_dir, step07_dir, pick_page_indexes = pick_manually)
# Page_num           (step07_dir, step08_dir)
# Crop_use_center    (step08_dir,step09_dir)


### 原始版本(1532data)
# Resize_hw          (step09_dir,step10_dir,width=332, height=500)
## 這要和 curve的對應到喔！建議從curve直接複製過來～然後要記得改 "gt_ord_imgs"
# Select_lt_rt_ld_rd_train_test_see(step10_dir, final_dir,  result_dir_name="gt_ord_imgs",
#                            train_4page_index_list=range(387),test_4page_index_list=[101,102,103,104] ,see_train_4page_index_list=[1,2,4,6])


### big版本(1532data)
Resize_hw          (step09_dir, step10big_dir, width=396, height=600)   ### 註解拿掉就代表用big版
## 這要和 curve的對應到喔！建議從curve直接複製過來～然後要記得改 "gt_ord_imgs"
Select_lt_rt_ld_rd_train_test_see(step10big_dir, final_big_dir,  result_dir_name="gt_ord_imgs",
                           train_4page_index_list=range(387), test_4page_index_list=[101, 102, 103, 104], see_train_4page_index_list=[1, 2, 4, 6])


### gt_focus版本(1532data)
# Find_ltrd_and_crop (step09_dir , step10focus_dir, padding=10)  ### 不要padding了，因為印表機會自動pad！在pad會太多！印表機左右pad:100px左右，上下pad:50px左右
# Resize_hw          (step10focus_dir, step11focus_dir, width=332, height=500)
# Select_lt_rt_ld_rd_train_test_see(step10focus_dir, final_focus_dir,  result_dir_name="gt_ord_imgs",
#                            train_4page_index_list=range(387), test_4page_index_list=[101,102,103,104] ,see_train_4page_index_list=[1,2,4,6])

### 800data版本
# Select_lt_rt_ld_rd_train_test_see(step10_dir, final_800_dir,  result_dir_name="gt_ord_imgs",
#                            train_4page_index_list=range(204), test_4page_index_list=[101,102,103,104] ,see_train_4page_index_list=[1,2,4,6])

### 400data版本
### 下面用的是共103個4page_train，剛好最後3個train跟test重複到會被刪掉就剛好100個4page_train囉！
# Select_lt_rt_ld_rd_train_test_see(step10_dir, final_400_dir,  result_dir_name="gt_ord_imgs",
#                            train_4page_index_list=range(1,104), test_4page_index_list=[101,102,103,104] ,see_train_4page_index_list=[1,2,4,6])
