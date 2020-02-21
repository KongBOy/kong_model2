import os
import shutil
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import get_dir_certain_file_name

dir_name = "stack_unet-easy300"

train_dir     = "datasets"+"/"+dir_name+"/"+"train"
train_dis_dir = "datasets"+"/"+dir_name+"/"+"train/distorted_img"
train_rec_dir = "datasets"+"/"+dir_name+"/"+"train/rec_move_map"
test_dir      = "datasets"+"/"+dir_name+"/"+"test"
test_dis_dir  = "datasets"+"/"+dir_name+"/"+"test/distorted_img"
test_rec_dir  = "datasets"+"/"+dir_name+"/"+"test/rec_move_map"

Check_dir_exist_and_build_new_dir( train_dir )
Check_dir_exist_and_build_new_dir( train_dis_dir )
Check_dir_exist_and_build_new_dir( train_rec_dir )
Check_dir_exist_and_build_new_dir( test_dir )
Check_dir_exist_and_build_new_dir( test_dis_dir )
Check_dir_exist_and_build_new_dir( test_rec_dir )

# os.listdir("step3_apply_flow_result")
src_dir = "step3_apply_flow_result"
img_list  = get_dir_certain_file_name(  src_dir, certain_word="3a1-I1-patch.bmp")
move_list = get_dir_certain_file_name(  src_dir, certain_word="3b-rec_mov_map.npy" )

data_amount = len(img_list)
train_amount = int(data_amount*0.9)
# test_amount = data_amount - train_amount

### train部分
for i in range(train_amount):
    src_img_path = src_dir       + "/" + img_list[i]
    dst_img_path = train_dis_dir + "/" + img_list[i]
    shutil.copy(src=src_img_path, dst=dst_img_path)

    src_mov_path = src_dir       + "/" + move_list[i]
    dst_mov_path = train_rec_dir + "/" + move_list[i]
    shutil.copy(src=src_mov_path, dst=dst_mov_path)

### test部分
for i in range(train_amount,data_amount):
    src_img_path = src_dir      + "/" + img_list[i]
    dst_img_path = test_dis_dir + "/" + img_list[i]
    shutil.copy(src=src_img_path, dst=dst_img_path)

    src_mov_path = src_dir      + "/" + move_list[i]
    dst_mov_path = test_rec_dir + "/" + move_list[i]
    shutil.copy(src=src_mov_path, dst=dst_mov_path)
