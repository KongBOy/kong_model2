from step0_access_path import access_path
import os
import shutil
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name
# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

dir_name = "stack_unet-padding2000"

train_dir     = access_path+"datasets"+"/"+dir_name+"/"+"train"
train_dis_dir = access_path+"datasets"+"/"+dir_name+"/"+"train/dis_imgs"
train_rec_dir = access_path+"datasets"+"/"+dir_name+"/"+"train/move_map"
test_dir      = access_path+"datasets"+"/"+dir_name+"/"+"test"
test_dis_dir  = access_path+"datasets"+"/"+dir_name+"/"+"test/dis_imgs"
test_rec_dir  = access_path+"datasets"+"/"+dir_name+"/"+"test/move_map"

Check_dir_exist_and_build_new_dir( train_dir )
Check_dir_exist_and_build_new_dir( train_dis_dir )
Check_dir_exist_and_build_new_dir( train_rec_dir )
Check_dir_exist_and_build_new_dir( test_dir )
Check_dir_exist_and_build_new_dir( test_dis_dir )
Check_dir_exist_and_build_new_dir( test_rec_dir )

# os.listdir("step3_apply_flow_result")
img_src_dir = access_path+"step3_apply_flow_result"
img_list  = get_dir_certain_file_name(  img_src_dir, certain_word="3a1-I1-patch.bmp")
mov_src_dir = access_path+"step2_flow_build/move_map"
move_list = get_dir_certain_file_name(  mov_src_dir, certain_word=".npy" )

data_amount = len(img_list)
train_amount = int(data_amount*0.9)
# test_amount = data_amount - train_amount

### train部分
for i in range(train_amount):
    src_img_path = img_src_dir   + "/" + img_list[i]
    dst_img_path = train_dis_dir + "/" + img_list[i]
    shutil.copy(src=src_img_path, dst=dst_img_path)

    src_mov_path = mov_src_dir   + "/" + move_list[i]
    dst_mov_path = train_rec_dir + "/" + move_list[i]
    shutil.copy(src=src_mov_path, dst=dst_mov_path)

### test部分
for i in range(train_amount,data_amount):
    src_img_path = img_src_dir      + "/" + img_list[i]
    dst_img_path = test_dis_dir + "/" + img_list[i]
    shutil.copy(src=src_img_path, dst=dst_img_path)

    src_mov_path = mov_src_dir      + "/" + move_list[i]
    dst_mov_path = test_rec_dir + "/" + move_list[i]
    shutil.copy(src=src_mov_path, dst=dst_mov_path)
