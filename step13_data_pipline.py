from step0_access_path import access_path
from step9_load_and_train_and_test import step4_get_result_dir_default_logs_ckpt_dir_name
from util import get_dir_img, time_util
from build_dataset_combine import Check_dir_exist_and_build
import time
import tensorflow as tf 

phase = "train"
start_epoch = 0
epochs = 160
epoch_down_step = 100
epoch_save_freq = 20

def norm_imgs(imgs):
    return (imgs/127.5)-1


### train 和 test 參數
restore_model = False ### 如果 restore_model 設True，下面 restore_result_dir 才會有用處喔！
restore_result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"

data_maount = 1800

#####################################################################################
batch_size = 1
unet_imgs  = get_dir_img(access_path+"step11_unet_rec_img")
gt_imgs = get_dir_img(access_path+"step12_ord_pad_gt")

unet_imgs  = norm_imgs(unet_imgs)
gt_imgs = norm_imgs(gt_imgs)

unet_imgs_train_db = tf.data.Dataset.from_tensor_slices(unet_imgs[:1800])
unet_imgs_train_db = unet_imgs_train_db.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
gt_imgs_train_db = tf.data.Dataset.from_tensor_slices(gt_imgs[:1800])
gt_imgs_train_db = gt_imgs_train_db.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

unet_imgs_test_db = tf.data.Dataset.from_tensor_slices(unet_imgs[1800:])
unet_imgs_test_db = unet_imgs_test_db.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
gt_imgs_test_db = tf.data.Dataset.from_tensor_slices(gt_imgs[1800:])
gt_imgs_test_db = gt_imgs_test_db.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# from step10_kong_model5_Rect2 import Generator, Discriminator, generate_images, train_step
from step10_kong_model5_Rect2 import Rect2, generate_images, train_step
#####################################################################################
rect2 = Rect2()
generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
discriminator_optimizer   = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

#####################################################################################
epoch_log = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
ckpt = tf.train.Checkpoint(epoch_log=epoch_log, rect2=rect2, 
                                                generator_optimizer=generator_optimizer,
                                                discriminator_optimizer=discriminator_optimizer)
#####################################################################################
### 決定 結果要寫哪邊 或 從哪邊讀資料
import datetime 
### 預設用 rect2_2020....-...... 當開頭
result_dir = "rect2_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logs_dir, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(result_dir)
##################################################
### 如果要restore_model，再改用restore的位置並重新取logs、ckpt資料夾
if(restore_model==True):
    result_dir = restore_result_dir
    logs_dir, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(result_dir)
##################################################
Check_dir_exist_and_build(result_dir)
Check_dir_exist_and_build(logs_dir)
Check_dir_exist_and_build(ckpt_dir)

#####################################################################################
summary_writer = tf.summary.create_file_writer( logs_dir ) ### 建tensorboard，這會自動建資料夾喔！
manager        = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份


### 決定 要不要讀取上次的結果
if(restore_model==True): 
    ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
    start_epoch = ckpt.epoch_log.numpy()
    print("load model ok~~~~~~~~~~~")

#####################################################################################
def save_rect2_train_code(result_dir):
    import shutil
    dst_dir = result_dir+"/"+"train_code"
    Check_dir_exist_and_build(dst_dir)
    shutil.copy("step10_kong_model5_Rect2.py",dst_dir + "/" + "step10_kong_model5_Rect2.py")
    shutil.copy("step11_unet_rec_img.py"     ,dst_dir + "/" + "step11_unet_rec_img.py")
    shutil.copy("step12_ord_pad_gt.py"       ,dst_dir + "/" + "step12_ord_pad_gt.py")
    shutil.copy("step13_data_pipline.py"     ,dst_dir + "/" + "step13_data_pipline.py")
    shutil.copy("util.py"                    ,dst_dir + "/" + "util.py")

if(phase.lower() == "train"):
    save_rect2_train_code(result_dir)

######################################################################################################################
## training 的部分 ####################################################################################################
total_start = time.time()
for epoch in range(start_epoch, epochs):
    print("Epoch: ", epoch)
    e_start = time.time()

    lr = 0.0002 if epoch < epoch_down_step else 0.0002*(epochs-epoch)/(epochs-epoch_down_step)
    generator_optimizer.lr     = lr
    discriminator_optimizer.lr = lr
    ##     用來看目前訓練的狀況 
    for test_input, test_label in zip(unet_imgs_test_db.take(1), gt_imgs_test_db.take(1)): 
        generate_images( rect2.generator, test_input, test_label, epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
    ###     訓練
    for n, (input_image, target) in enumerate( zip(unet_imgs_train_db, gt_imgs_train_db) ):
        print('.', end='')
        if (n+1) % 100 == 0:print()
        train_step(rect2, input_image, target, generator_optimizer, discriminator_optimizer, summary_writer, epoch)

    ###     儲存模型 (checkpoint) the model every 20 epochs
    if (epoch + 1) % epoch_save_freq == 0:
        ckpt.epoch_log.assign(epoch+1) ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
        manager.save()

    epoch_cost_time = time.time()-e_start
    total_cost_time = time.time()-total_start
    print('epoch %i cost time:%.2f'%(epoch, epoch_cost_time)  )
    print("batch cost time:%.2f"   %(epoch_cost_time/data_maount) )
    print("total cost time:%s"     %(time_util(total_cost_time))  )
    print("")