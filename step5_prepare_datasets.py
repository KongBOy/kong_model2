import os
from build_dataset_combine import Check_dir_exist_and_build_new_dir


dir_name = "stack_unet-easy2000"

Check_dir_exist_and_build_new_dir("datasets"+"/"+dir_name+"/"+"train")
Check_dir_exist_and_build_new_dir("datasets"+"/"+dir_name+"/"+"train/distorted_img")
Check_dir_exist_and_build_new_dir("datasets"+"/"+dir_name+"/"+"train/rec_move_map")
Check_dir_exist_and_build_new_dir("datasets"+"/"+dir_name+"/"+"test")
Check_dir_exist_and_build_new_dir("datasets"+"/"+dir_name+"/"+"test/distorted_img")
Check_dir_exist_and_build_new_dir("datasets"+"/"+dir_name+"/"+"test/rec_move_map")

os.listdir("step3_apply_flow_result")
