import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from util import method2
from step3_apply_mov2ord_img import apply_move
from step6_data_pipline import get_all_distorted_and_resize_and_norm, get_db_all_move_map_and_resize, use_db_to_norm, use_number_to_norm
from step7_kong_model_use_GAN import Generator, Discriminator
########################################################################################################################

db_dir="datasets"
db_name="stack_unet-easy2000"
# model_name = "use_GAN"
model_name = "just_G"



### 讀 distorted_test_db
ord_dir = db_dir + "/" + db_name + "/" + "test/distorted_img" 
distorted_test_db = get_all_distorted_and_resize_and_norm(ord_dir)
# distorted_img = distorted_test_db[0]

### 讀 rec_move_map_train_db，用抓 max/min 值給test_ref用
rec_move_map_train_path = db_dir + "/" + db_name + "/" + "train/rec_move_map" 
rec_move_map_train_list = get_db_all_move_map_and_resize(rec_move_map_train_path)
rec_move_map_train_list, max_value_train, min_value_train = use_db_to_norm(rec_move_map_train_list)

### 讀 rec_move_map_test_db，用來當test_ref來看train的對不對(GT的概念)
rec_move_map_test_path = db_dir + "/" + db_name + "/" + "test/rec_move_map" 
rec_move_map_test_list = get_db_all_move_map_and_resize(rec_move_map_test_path)
rec_move_map_test_list = use_number_to_norm(rec_move_map_test_list, max_value_train, min_value_train)
########################################################################################################################
### 讀 model 的 checkpoint
generator = Generator(out_channel=2)
discriminator = Discriminator()
generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir    = './training_checkpoints'+"_"+db_name+"_"+model_name
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# manager = tf.train.CheckpointManager(checkpoint, './training_checkpoints/ckpt', max_to_keep=3)
# checkpoint.restore(manager.latest_checkpoint)

########################################################################################################################
### 把test 丟入 train好的model
result_dir = "step9_model_result"+"_"+db_name+"_"+model_name
Check_dir_exist_and_build_new_dir(result_dir)

for i, distorted_img in enumerate(distorted_test_db):
    print("doing %06i test image"%i)
    ### 把 test 的 distorted_img 丟進去 generator，生成rec_move_map
    result = generator(distorted_img.reshape(1,256,256,3), training=True)
    result = result.numpy()
    result = result[0] ### BHWC，所以取第零張~~
    result_back = (result+1)/2 * (max_value_train-min_value_train) + min_value_train ### G生成的rec_move_map值是-1~1， 用 train_move_map_db 的 max_value, min_value 恢復

    ### rec_move_map apply進去
    distorted_img = (distorted_img+1)*127.5
    distorted_img = distorted_img[:,:,::-1]
    rec_img , rec_debug_move = apply_move(distorted_img, result_back)
    cv2.imwrite( result_dir + "/" + "%06i_b_4-G-rec_img.jpg"%i, rec_img.astype(np.uint8))
    ####################################################################################
    ### rec_move_map 視覺化 並存起來
    result_back_bgr = method2(result_back[...,0], result_back[...,1], 1)
    cv2.imwrite(result_dir+"/%06i_a_2-G_rec_move_map_visual.jpg"%i,result_back_bgr) ### 視覺化的結果存起來
    np.save(result_dir+"/%06i_c_1-G_rec_move_map"%i, result_back) ### 存起來

    ### rec_debug_move 視覺化 並 存起來
    rec_debug_move_bgr = method2(rec_debug_move[...,0], rec_debug_move[...,1], 1)
    cv2.imwrite(result_dir + "/%06i_b_2-G_rec_debug_move_map.bmp"%i, rec_debug_move_bgr) ### 視覺化的結果存起來
    np.save(result_dir + "/%06i_c_3-G_rec_debug_move_map"%i, rec_debug_move) ### 存起來
    ################################################################################################################################
    ### 讀取 test 的 GT rec_move_map
    result_ref = rec_move_map_test_list[i]
    result_ref_back = (result_ref+1)/2 * (max_value_train-min_value_train) + min_value_train

    ### GT rec_move_map apply進去
    gt_rec_img, gt_rec_debug_move = apply_move(distorted_img, result_ref_back)
    cv2.imwrite( result_dir + "/" + "%06i_b_5-rec_img.jpg"%i,gt_rec_img.astype(np.uint8))
    ####################################################################################
    ### GT rec_move_map 視覺化 並存起來
    result_ref_bgr = method2(result_ref_back[...,0], result_ref_back[...,1],1)
    cv2.imwrite(result_dir+"/%06i_a_3-GT-rec_move_map.jpg"%i,result_ref_bgr) ### 視覺化的結果存起來
    np.save(result_dir + "/%06i_c_2_GT_rec_move_map"%i ,result_ref_back )

    ### GT_rec_debug_move 視覺化 並 存起來
    gt_rec_debug_move_bgr = method2(gt_rec_debug_move[...,0], gt_rec_debug_move[...,1], 1)
    cv2.imwrite(result_dir + "/%06i_b_3-GT_rec_debug_move_map.bmp"%i, gt_rec_debug_move_bgr) ### 視覺化的結果存起來
    np.save(result_dir + "/%06i_c_4-GT_rec_debug_move_map"%i, gt_rec_debug_move) ### 存起來

    cv2.imwrite(result_dir+"/%06i_a_1-ord_distorted_img.jpg"%i,distorted_img.astype(np.uint8)) 
    cv2.imwrite(result_dir+"/%06i_b_1-ord_distorted_img.jpg"%i,distorted_img.astype(np.uint8))

    plt.close()
print("finish~~~~~~~~~~~~~~~~~")
