from step0_access_path import access_path
import os
import shutil
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name, get_maxmin_train_move_from_path, get_max_db_move_xy
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

    
    # in_dir  = access_path+"step3_apply_flow"
    # in_word = "3a1-I1-patch.bmp"
    # gt_dir  = access_path+"step2_build_flow/move_map"
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


#####################################################################################################################################################
### 1.建立給 unet 用的 pad2000-512to256，但先不要執行喔！因為目前用的是手動複製的，然後忘記當初怎麼複製的ˊ口ˋ 咪挺完看能不能就先 生成新的DB然後再改囉！
# build_datasets(build_dir_name="h=256,w=256_complex_1_pure_unet",
#                in_dir_name   = "dis_imgs",
#                gt_dir_name   = "move_maps",
#                in_src_dir    = access_path+"step3_apply_flow_complex",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_dir    = access_path+"step2_build_flow_complex/move_maps",
#                gt_src_word   = ".npy" )


# build_datasets(build_dir_name="h=384,w=256_complex_1_pure_unet",
#                in_dir_name   = "dis_imgs",
#                gt_dir_name   = "move_maps",
#                in_src_dir    = access_path+"step3_apply_flow_complex_h=384,w=256",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_dir    = access_path+"step2_build_flow_complex_h=384,w=256/move_maps",
#                gt_src_word   = ".npy" )


### 記得 complex+page做完需要手動去 挑 complex：0~1799 和 page：2000~3499 當train， complex：1800~1999 和 page：3500~3559
# build_datasets(build_dir_name="h=384,w=256_complex+page_1_pure_unet",
#                in_dir_name   = "dis_imgs",
#                gt_dir_name   = "move_maps",
#                in_src_dir    = access_path+"step3_apply_flow_complex+page_h=384,w=256",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_dir    = access_path+"step2_build_flow_complex+page_h=384,w=256/move_maps",
#                gt_src_word   = ".npy" )


# build_datasets(build_dir_name="h=384,w=256_old_page_1_pure_unet",
#                in_dir_name   = "dis_imgs",
#                gt_dir_name   = "move_maps",
#                in_src_dir    = access_path+"step3_apply_flow_page",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_dir    = access_path+"step2_build_flow_page/move_maps",
#                gt_src_word   = ".npy" )



#####################################################################################################################################################
## 2.建立給 rect2 用的 pure_rect2
# build_datasets(build_dir_name= "h=256,w=256_complex_2_pure_rect2",
#                in_dir_name   = "dis_img_db",
#                gt_dir_name   = "gt_ord_pad_img_db",
#                in_src_dir    = access_path+"step3_apply_flow_complex",
#                gt_src_dir    = access_path+"step3_apply_flow_complex",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_word   = "4-gt_ord_pad.bmp" )


# build_datasets(build_dir_name= "h=384,w=256_complex_2_pure_rect2",
#                in_dir_name   = "dis_img_db",
#                gt_dir_name   = "gt_ord_pad_img_db",
#                in_src_dir    = access_path+"step3_apply_flow_complex_h=384,w=256",
#                gt_src_dir    = access_path+"step3_apply_flow_complex_h=384,w=256",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_word   = "4-gt_ord_pad.bmp" )


# build_datasets(build_dir_name= "h=256,w=256_complex+page_2_pure_rect2",
#                in_dir_name   = "dis_img_db",
#                gt_dir_name   = "gt_ord_pad_img_db",
#                in_src_dir    = access_path+"step3_apply_flow_complex+page_h=384,w=256",
#                gt_src_dir    = access_path+"step3_apply_flow_complex+page_h=384,w=256",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_word   = "4-gt_ord_pad.bmp" )


# build_datasets(build_dir_name= "h=384,w=256_old_page_2_pure_rect2",
#                in_dir_name   = "dis_img_db",
#                gt_dir_name   = "gt_ord_pad_img_db",
#                in_src_dir    = access_path+"step3_apply_flow",
#                gt_src_dir    = access_path+"step3_apply_flow",
#                in_src_word   = "3a1-I1-patch.bmp",
#                gt_src_word   = "4-gt_ord_pad.bmp" )

#####################################################################################################################################################
### 3.建立給 unet+rect2 用的 rect2_2000
# build_datasets(build_dir_name= "h=256,w=256_complex_3_unet_rect2",
#                in_dir_name   = "unet_rec_img_db",
#                gt_dir_name   = "gt_ord_img_db",
#                in_src_dir    = access_path+"result/20200328-170738_1_pure_unet_complex_h=256,w=256_model2_UNet_512to256_finish/test_indicate_1_pure_unet_complex_h=256,w=256",
#                gt_src_dir    = access_path+"step3_apply_flow_complex",
#                in_src_word   = "g_rec_img.bmp",
#                gt_src_word   = "1-I.bmp" )



# build_datasets(build_dir_name= "h=384,w=256_complex_3_unet_rect2",
#                in_dir_name   = "unet_rec_img_db",
#                gt_dir_name   = "gt_ord_img_db",
#                in_src_dir    = access_path+"result/h=384,w=256_20200329-215628_1_pure_unet_complex_h=384,w=256_model2_UNet_512to256/test_indicate_1_pure_unet_complex+page_h=384,w=256",
#                gt_src_dir    = access_path+"step3_apply_flow_h=384,w=256_complex",
#                in_src_word   = "g_rec_img.bmp",
#                gt_src_word   = "1-I.bmp" )


build_datasets(build_dir_name= "h=384,w=256_complex+page_3_unet_rect2",
               in_dir_name   = "unet_rec_img_db",
               gt_dir_name   = "gt_ord_img_db",
               in_src_dir    = access_path+"result/h=384,w=256_complex+page_1_pure_unet_20200329-232144_model2_UNet_512to256_finish/test_indicate_h=384,w=256_complex+page_1_pure_unet",
               gt_src_dir    = access_path+"step3_apply_flow_h=384,w=256_complex+page",
               in_src_word   = "g_rec_img.bmp",
               gt_src_word   = "1-I.bmp" )


# build_datasets(build_dir_name= "h=384,w=256_old_page_3_unet_rect2",
#                in_dir_name   = "unet_rec_img_db",
#                gt_dir_name   = "gt_ord_img_db",
#                in_src_dir    = access_path+"result/20200316-114012_1_page_h=384,w=256_model2_UNet_512to256_127.28_finish/test_1_unet_page_h=384,w=256",
#                gt_src_dir    = access_path+"step3_apply_flow_page",
#                in_src_word   = "unet_rec_img.bmp",
#                gt_src_word   = "1-I.bmp" )




# db_name = "1_pure_unet2000-512to256_index"
# max_train_move, min_train_move = get_maxmin_train_move_from_path(access_path+"datasets"+"/"+db_name+"/"+"train"+"/"+"move_maps")
# np.save(access_path+"datasets"+"/"+db_name+"/"+"max_train_move",max_train_move)
# np.save(access_path+"datasets"+"/"+db_name+"/"+"min_train_move",min_train_move)

# db_name = "1_pure_unet_page_h=384,w=256"
# max_db_move_x, max_db_move_y = get_max_db_move_xy(db_dir=access_path+"datasets", db_name=db_name)
# np.save(access_path+"datasets"+"/"+db_name+"/"+"max_db_move_x",max_db_move_x)
# np.save(access_path+"datasets"+"/"+db_name+"/"+"max_db_move_y",max_db_move_y)