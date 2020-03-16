from step0_access_path import access_path
import os
import shutil
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name
# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

def build_datasets(build_dir_name, in_dir_name, gt_dir_name, in_src_dir, in_src_word, gt_src_dir, gt_src_word):
    dir_name = build_dir_name #"padding2000"

    train_dir    = access_path+"datasets"+"/"+dir_name+"/"+"train"
    test_dir     = access_path+"datasets"+"/"+dir_name+"/"+"test"

    train_in_dir = access_path+"datasets"+"/"+dir_name+"/"+"train/"+in_dir_name
    train_gt_dir = access_path+"datasets"+"/"+dir_name+"/"+"train/"+gt_dir_name
    test_in_dir  = access_path+"datasets"+"/"+dir_name+"/"+"test/" +in_dir_name
    test_gt_dir  = access_path+"datasets"+"/"+dir_name+"/"+"test/" +gt_dir_name

    Check_dir_exist_and_build_new_dir( train_dir )
    Check_dir_exist_and_build_new_dir( train_in_dir )
    Check_dir_exist_and_build_new_dir( train_gt_dir )
    Check_dir_exist_and_build_new_dir( test_dir )
    Check_dir_exist_and_build_new_dir( test_in_dir )
    Check_dir_exist_and_build_new_dir( test_gt_dir )

    
    # in_dir  = access_path+"step3_apply_flow_result"
    # in_word = "3a1-I1-patch.bmp"
    # gt_dir  = access_path+"step2_flow_build/move_map"
    # gt_word = ".npy"
    in_list    = get_dir_certain_file_name(  in_src_dir, certain_word=in_src_word)
    gt_list    = get_dir_certain_file_name(  gt_src_dir, certain_word=gt_src_word )

    data_amount = len(in_list)
    train_amount = int(data_amount*0.9)
    # test_amount = data_amount - train_amount

    ### train部分
    for i in range(train_amount):
        src_in_path = in_src_dir       + "/" + in_list[i]
        dst_in_path = train_in_dir + "/" + in_list[i]
        shutil.copy(src=src_in_path, dst=dst_in_path)

        src_gt_path = gt_src_dir       + "/" + gt_list[i]
        dst_gt_path = train_gt_dir + "/" + gt_list[i]
        shutil.copy(src=src_gt_path, dst=dst_gt_path)

    ### test部分
    for i in range(train_amount,data_amount):
        src_in_path = in_src_dir      + "/" + in_list[i]
        dst_in_path = test_in_dir + "/" + in_list[i]
        shutil.copy(src=src_in_path, dst=dst_in_path)

        src_gt_path = gt_src_dir      + "/" + gt_list[i]
        dst_gt_path = test_gt_dir + "/" + gt_list[i]
        shutil.copy(src=src_gt_path, dst=dst_gt_path)


### 1.建立給 unet 用的 pad2000-512to256，但先不要執行喔！因為目前用的是手動複製的，然後忘記當初怎麼複製的ˊ口ˋ 咪挺完看能不能就先 生成新的DB然後再改囉！
# build_datasets(build_dir_name="1_pad2000-512to256",
#                in_dir_name   = "dis_imgs",
#                gt_dir_name   = "move_maps",
#                in_src_dir    = access_path+"step3_apply_flow_result",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_dir    = access_path+"step2_flow_build/move_map",
#                gt_src_word   = ".npy" )


# build_datasets(build_dir_name="1_page_h=384,w=256",
#                in_dir_name   = "dis_imgs",
#                gt_dir_name   = "move_maps",
#                in_src_dir    = access_path+"step3_apply_flow_result",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_dir    = access_path+"step2_flow_build/move_maps",
#                gt_src_word   = ".npy" )


### 2.建立給 rect2 用的 pure_rect2
# build_datasets(build_dir_name= "2_pure_rect2_h=256,w=256",
#                in_dir_name   = "dis_img_db",
#                gt_dir_name   = "gt_ord_pad_img_db",
#                in_src_dir    = access_path+"step3_apply_flow_result",
#                gt_src_dir    = access_path+"step12_gt_ord_pad",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_word   = "img.bmp" )

build_datasets(build_dir_name= "2_pure_rect2_h=384,w=256",
               in_dir_name   = "dis_img_db",
               gt_dir_name   = "gt_ord_pad_img_db",
               in_src_dir    = access_path+"step3_apply_flow_result",
               gt_src_dir    = access_path+"step3_apply_flow_result",
               in_src_word   = "3a1-I1-patch.bmp",
               gt_src_word   = "4-gt_ord_pad.bmp" )


### 3.建立給 unet+rect2 用的 rect2_2000
# build_datasets(build_dir_name= "3_unet_rect2_h=256,w=256",
#                in_dir_name   = "unet_rec_img_db",
#                gt_dir_name   = "gt_unet_rec_img_db",
#                in_src_dir    = access_path+"step11_unet_rec_img",
#                gt_src_dir    = access_path+"step12_gt_ord",
#                in_src_word   = "unet_rec_img.bmp",
#                gt_src_word   = "img.bmp" )


