from step0_access_path import access_path
# from step9_load_and_train_and_test import step2_build_model_and_optimizer, step3_build_checkpoint, step4_get_result_dir_default_logs_ckpt_dir_name
from step9_load_and_train_and_test import step2_3_build_model_opti_ckpt, step4_get_result_dir_default_logs_ckpt_dir_name
from util import get_dir_certain_img
import tensorflow as tf 


model_name="model2_UNet_512to256"
model_result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"
pad_result = True ### pad 和 沒pad 的差別只有：1.存結果的地方不同、2.unet_rec_img 有沒有pad成 H,W 的大小

#############################################################################################################
### 讀出model的部分
# generator, generator_optimizer, discriminator, discriminator_optimizer, generate_images, train_step = step2_build_model_and_optimizer(model_name=model_name)
# ckpt  = step3_build_checkpoint (model_name=model_name, generator=generator, generator_optimizer=generator_optimizer, discriminator=discriminator, discriminator_optimizer=discriminator_optimizer)
model_dict, generate_images, train_step, ckpt = step2_3_build_model_opti_ckpt(model_name=model_name)

_, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(model_result_dir) ### 不需要log_dir所以用"_"來接
manager     = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份
ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
start_epoch = ckpt.epoch_log.numpy()
print("start_epoch",start_epoch)
print("load model ok~~~~~~~~~~~")

#############################################################################################################
import cv2
import numpy as np 
from util import predict_unet_move_maps_back, get_max_move_xy_from_certain_move
from build_dataset_combine import Check_dir_exist_and_build_new_dir
from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec2



def dis_imgs_resize_and_nrom(dis_imgs, resize_shape):
    proc_list = []
    for dis_img in dis_imgs:
        proc = cv2.resize(dis_img, resize_shape, interpolation=cv2.INTER_NEAREST) 
        proc_list.append(proc)
    dis_imgs = np.array(proc_list)
    dis_imgs = dis_img/127.5 -1
    return dis_imgs


step11_result_dir = access_path+"step11_unet_rec_img"
if(pad_result):
    step11_result_dir = access_path+"step11_unet_rec_img_pad"
Check_dir_exist_and_build_new_dir(step11_result_dir)

resize_shape = (512,512)


### 用 dis_img 得到 unet_move_map
dis_imgs = get_dir_certain_img(access_path+"step3_apply_flow_result","3a1-I1-patch.bmp")  ### 讀取dis_imgs 等等輸入unet
dis_imgs_resize_norm = dis_imgs_resize_and_nrom(dis_imgs, resize_shape) ### unet前處理，為了要符合unet的格式：-1~1 和 resize_shape 
unet_move_maps = []
for i, dis_img in enumerate(dis_imgs_resize_norm):
    print("doing %06i"%i)
    unet_move_map = model_dict["generator"](np.expand_dims(dis_img, axis=0), training=True) ### dis_img 丟進去generator 來 predict unet_move_map
    unet_move_maps.append(unet_move_map.numpy()) ### unet_move_map 存起來
unet_move_maps = np.array(unet_move_maps) 
unet_move_maps = predict_unet_move_maps_back(unet_move_maps)  ### 把 unet_move_map"s" 的值 從-1~1 還原

### 用得到 unet_move_map 來還原 dis_img
max_move_x, max_move_y = get_max_move_xy_from_certain_move(access_path+"step3_apply_flow_result","2-q") ### 注意這裡要去 step3才對！因為當初建db時是用整個db的最大移動量(step3裡的即整個db的資料)，如果去dataset/train的話只有train的資料喔
import matplotlib.pyplot as plt
for i, dis_img in enumerate(dis_imgs):
    unet_rec_img = apply_move_to_rec2(dis_img, unet_move_maps[i], max_move_x, max_move_y) ### 用unet_move_map來還原dis_img
    if(pad_result): 
        # unet_rec_img = np.pad( unet_rec_img, ( (max_move_y,max_move_x), (max_move_y,max_move_x), (0,0) )) ### 不精確的寫法，結果x就少1個pixel了！
        ### 以下完全復刻 產生dis_img時的padding 方式囉！他媽的有時間一定要回去改簡單一點
        row, col = unet_move_maps[i].shape[:2]
        
        ### 先建出一個大大的 unet_rec_img_pad 畫布
        unet_rec_img_pad_h = int( np.around(max_move_y + row + max_move_y) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
        unet_rec_img_pad_w = int( np.around(max_move_x + col + max_move_x) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
        unet_rec_img_pad  = np.zeros(shape=(unet_rec_img_pad_h,unet_rec_img_pad_w,3), dtype=np.uint8)

        ### 在把 unet_rec_img 畫進去 unet_rec_img_pad
        int_max_move_x = int(max_move_x)
        int_max_move_y = int(max_move_y)
        unet_rec_img_pad[int_max_move_y:int_max_move_y+row, int_max_move_x:int_max_move_x+col, :] = unet_rec_img
        unet_rec_img = unet_rec_img_pad
    cv2.imwrite(step11_result_dir+"/%06i_unet_rec_img.bmp"%i, unet_rec_img)
