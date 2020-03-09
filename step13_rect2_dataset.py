from step0_access_path import access_path
import os
import shutil
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name
# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

dir_name = "rect2_2000"

train_dir     = access_path+"datasets"+"/"+dir_name+"/"+"train"
train_dis_dir = access_path+"datasets"+"/"+dir_name+"/"+"train/unet_rec_img_db"
train_rec_dir = access_path+"datasets"+"/"+dir_name+"/"+"train/gt_unet_rec_img_db"
test_dir      = access_path+"datasets"+"/"+dir_name+"/"+"test"
test_dis_dir  = access_path+"datasets"+"/"+dir_name+"/"+"test/unet_rec_img_db"
test_rec_dir  = access_path+"datasets"+"/"+dir_name+"/"+"test/gt_unet_rec_img_db"

Check_dir_exist_and_build_new_dir( train_dir )
Check_dir_exist_and_build_new_dir( train_dis_dir )
Check_dir_exist_and_build_new_dir( train_rec_dir )
Check_dir_exist_and_build_new_dir( test_dir )
Check_dir_exist_and_build_new_dir( test_dis_dir )
Check_dir_exist_and_build_new_dir( test_rec_dir )

# os.listdir("step3_apply_flow_result")
img_src_dir = access_path+"step11_unet_rec_img"
imgs  = get_dir_certain_file_name(  img_src_dir, certain_word="unet_rec_img.bmp")
gt_img_src_dir = access_path+"step12_ord_pad_gt"
gt_imgs = get_dir_certain_file_name(  gt_img_src_dir, certain_word="img.bmp" )

data_amount = len(imgs)
train_amount = int(data_amount*0.9)
# test_amount = data_amount - train_amount

### train部分
for i in range(train_amount):
    src_img_path = img_src_dir   + "/" + imgs[i]
    dst_img_path = train_dis_dir + "/" + imgs[i]
    shutil.copy(src=src_img_path, dst=dst_img_path)

    src_mov_path = gt_img_src_dir   + "/" + gt_imgs[i]
    dst_mov_path = train_rec_dir + "/" + gt_imgs[i]
    shutil.copy(src=src_mov_path, dst=dst_mov_path)

### test部分
for i in range(train_amount,data_amount):
    src_img_path = img_src_dir      + "/" + imgs[i]
    dst_img_path = test_dis_dir + "/" + imgs[i]
    shutil.copy(src=src_img_path, dst=dst_img_path)

    src_mov_path = gt_img_src_dir      + "/" + gt_imgs[i]
    dst_mov_path = test_rec_dir + "/" + gt_imgs[i]
    shutil.copy(src=src_mov_path, dst=dst_mov_path)
