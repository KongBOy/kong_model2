import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from step0_unet_util import method2
from step1_data_pipline_Stack_UNet import get_all_distorted, get_db_all_move_map, use_db_to_norm, use_number_to_norm
from step2_kong_model import Generator, Discriminator

########################################################################################################################
### 讀 distorted_test_db
db_dir="datasets"
db_name="stack_unet-256-4000"
ord_dir = db_dir + "/" + db_name + "/" + "test/distorted" 
distorted_test_db = get_all_distorted(ord_dir)
# distorted_img = distorted_test_db[0]

### 讀 rec_move_map_train_db，用抓 max/min 值給test_ref用
rec_move_map_train_path = db_dir + "/" + db_name + "/" + "train/rec_move_map" 
rec_move_map_train_list = get_db_all_move_map(rec_move_map_train_path)
rec_move_map_train_list, max_value_train, min_value_train = use_db_to_norm(rec_move_map_train_list)

### 讀 rec_move_map_test_db，用來當test_ref來看train的對不對
rec_move_map_test_path = db_dir + "/" + db_name + "/" + "test/rec_move_map" 
rec_move_map_test_list = get_db_all_move_map(rec_move_map_test_path)
rec_move_map_test_list = use_number_to_norm(rec_move_map_test_list, max_value_train, min_value_train)
########################################################################################################################
### 讀 model 的 checkpoint
generator = Generator(out_channel=2)
discriminator = Discriminator()
generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir    = './training_checkpoints-4000'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# manager = tf.train.CheckpointManager(checkpoint, './training_checkpoints/ckpt', max_to_keep=3)
# checkpoint.restore(manager.latest_checkpoint)

########################################################################################################################
### 把test 丟入 train好的model
Check_dir_exist_and_build_new_dir("step5_result")

for i, distorted_img in enumerate(distorted_test_db):
    result = generator(distorted_img.reshape(1,256,256,3))
    result = result.numpy()
    result = result[0]

    result_back = (result+1)/2 * (max_value_train-min_value_train) + min_value_train
    result_back_bgr = method2(result_back[...,0], result_back[...,1],2)
    plt.imshow(result_back_bgr)


    # cv2.imshow("ord_distorted",((distorted_img+1)*127.5)[:,:,::-1].astype(np.uint8))
    # cv2.waitKey(0)
    cv2.imwrite("step5_result/%02i_ord_distorted_img.jpg"%i,((distorted_img+1)*127.5)[:,:,::-1].astype(np.uint8))
    np.save("step5_result/%02i_rec_move_map"%i, result_back)


    plt.figure()
    result_ref = rec_move_map_test_list[i]
    result_ref_back = (result_ref+1)/2 * (max_value_train-min_value_train) + min_value_train
    result_ref_bgr = method2(result_ref_back[...,0], result_ref_back[...,1],2)
    plt.imshow(result_ref_bgr)
    plt.show()
    # cv2.imshow("back_bgr",back_bgr)
    # cv2.waitKey(0)
print("finish~~~~~~~~~~~~~~~~~")
# unet = tf.keras.models.load_model("training_checkpoints/ckpt-7")