from step0_access_path import access_path
from step9_load_and_train_and_test import step2_build_model_and_optimizer, step3_build_checkpoint, step4_get_result_dir_default_logs_ckpt_dir_name
from util import get_dir_certain_img
import tensorflow as tf 


db_name = "pad2000-512to256" ### 這資料集 要搭配 512to256 的架構喔！
model_name="model2_UNet_512to256"

#############################################################################################################
generator, generator_optimizer, discriminator, discriminator_optimizer, generate_images, train_step = step2_build_model_and_optimizer(model_name=model_name)
ckpt  = step3_build_checkpoint (model_name=model_name, generator=generator, generator_optimizer=generator_optimizer, discriminator=discriminator, discriminator_optimizer=discriminator_optimizer)

result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"
_, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(result_dir)
manager     = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份
ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
start_epoch = ckpt.epoch_log.numpy()
print("start_epoch",start_epoch)
print("load model ok~~~~~~~~~~~")

#############################################################################################################
import cv2
import numpy as np 
from util import predict_move_maps_back, get_max_move_xy_from_certain_move
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec
from step6_data_pipline import distorted_resize_and_norm

result_dir = access_path+"step11_unet_rec_img"
Check_dir_exist_and_build_new_dir(result_dir)

resize_shape = (512,512)

dis_imgs = get_dir_certain_img(access_path+"step3_apply_flow_result","3a1-I1-patch.bmp")
dis_imgs_resize_norm = distorted_resize_and_norm(dis_imgs, resize_shape)
move_maps = []
for i, dis_img in enumerate(dis_imgs_resize_norm):
    print("doing %06i"%i)
    move_map = generator(np.expand_dims(dis_img, axis=0), training=True)
    move_maps.append(move_map.numpy())
move_maps = np.array(move_maps)
move_maps = predict_move_maps_back(move_maps)


max_move_x, max_move_y = get_max_move_xy_from_certain_move(access_path+"step3_apply_flow_result","2-q") ### 注意這裡要去 step3才對！因為當初建db時是用整個db的最大移動量(step3裡的即整個db的資料)，如果去dataset/train的話只有train的資料喔

import matplotlib.pyplot as plt
for i, dis_img in enumerate(dis_imgs):
    g_rec_img = apply_move_to_rec(dis_img, move_maps[i], max_move_x, max_move_y)
    cv2.imwrite(result_dir+"/%06i_unet_rec_img.bmp"%i, g_rec_img)
