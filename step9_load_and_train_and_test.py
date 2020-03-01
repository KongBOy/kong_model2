from step0_access_path import access_path
import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from util import method2, get_db_amount, time_util
from step6_data_pipline import get_dataset
import time


from build_dataset_combine import Check_dir_exist_and_build
import os

# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！


def step1_data_pipline(db_dir="datasets", db_name="easy300", batch_size=1):
    ### step1.讀取 data_pipline
    img_resize  = None
    move_resize = None
    if  (model_name == "model1_UNet"):          
        img_resize =(256,256)
        move_resize=(256,256)
    elif(model_name == "model2_UNet_512to256"): 
        img_resize =(512,512)
        move_resize=(256,256)
    elif(model_name == "model3_UNet_stack"):    
        img_resize =(256,256)
        move_resize=(256,256)
    elif(model_name == "model4_UNet_and_D"):    
        img_resize =(256,256)
        move_resize=(256,256)

    train_db, train_label_db, \
    test_db , test_label_db , \
    max_value_train, min_value_train = get_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, move_resize=move_resize)

    return train_db, train_label_db, \
           test_db , test_label_db , \
           max_value_train, min_value_train

def step2_build_model_and_optimizer(model_name="model1_UNet"):
    ### step2.建立 model 和 optimizer
    start_time = time.time()
    generator               = None
    generator_optimizer     = None
    discriminator           = None
    discriminator_optimizer = None
    if  (model_name == "model1_UNet"):
        from step7_kong_model1_UNet import Generator, generate_images, train_step
        generator     = Generator(out_channel=2)
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    elif(model_name == "model2_UNet_512to256"):
        from step7_kong_model2_UNet_512to256 import Generator512to256, generate_images, train_step
        generator     = Generator512to256(out_channel=2)
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    elif(model_name == "model3_UNet_stack"):
        from step7_kong_model3_UNet_stack import Generator_stack, generate_images, train_step
        generator     = Generator_stack() ### 建立模型
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    elif(model_name == "model4_UNet_and_D"):
        from step7_kong_model4_UNet_and_D import Generator, Discriminator, generate_images, train_step
        generator     = Generator(out_channel=2)
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator = Discriminator()
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) ### 覺得還是不要統一寫出來，這樣比較好看每個模型需要什麼東西
    print("build model cost time:", time.time()-start_time)
    return generator, generator_optimizer, discriminator, discriminator_optimizer, generate_images, train_step

def step3_build_checkpoint(model_name, generator, generator_optimizer, discriminator, discriminator_optimizer):
    ### step3.建立 save/load model 的checkpoint
    ckpt = None
    epoch_log = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
    if  (model_name == "model1_UNet"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator)
    elif(model_name == "model2_UNet_512to256"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator)
    elif(model_name == "model3_UNet_stack"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator)
    elif(model_name == "model4_UNet_and_D"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator,
                                                        discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)
    return ckpt


def step4_get_default_dir_name(db_name, model_name):
    import datetime
    result_dir = access_path+"result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name+"_"+model_name  ### result資料夾，裡面放checkpoint和tensorboard資料夾
    log_dir  = result_dir + "/" + "logs"
    ckpt_dir = result_dir + '/' + 'ckpt_dir'+"_"+db_name+"_"+model_name 
    return result_dir, log_dir, ckpt_dir




##############################################################################################################################
### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數
BATCH_SIZE = 1

db_dir  = access_path+"datasets"

# db_name = "easy300"
# db_name = "easy2000"
db_name = "pad2000-512to256" ### 這資料集 要搭配 512to256 的架構喔！

# model_name="model1_UNet"
model_name="model2_UNet_512to256"
# model_name="model3_UNet_stack"
# model_name="model4_UNet_and_D"

### train 參數
epochs = 160
epoch_down_step = 100 ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
epoch_save_freq = 20
start_epoch = 0

### train 和 test 參數
restore_model = True ### 如果 restore_model 設True，下面 restore_result_dir 和 restore_ckpt_dir 才會有用處喔！
# restore_model = True ### 如果 restore_model 設True，下面 restore_result_dir 和 restore_ckpt_dir 才會有用處喔！
# restore_result_dir = "result/20200226-194945_pad2000-512to256_model2_UNet_512to256"
restore_result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"
restore_log_dir    = restore_result_dir + "/"  + "logs"
restore_ckpt_dir   = restore_result_dir + "/"  + "ckpt_dir" + "_" + db_name + "_" + model_name

### 目前只有在算 b_cost_time會用到
data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "distorted_img" )

####################################################################################################
####################################################################################################
train_db, train_label_db, test_db, test_label_db, \
max_value_train, min_value_train         = step1_data_pipline(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE)

generator, generator_optimizer,\
discriminator, discriminator_optimizer,\
generate_images, train_step              = step2_build_model_and_optimizer(model_name=model_name)

ckpt  = step3_build_checkpoint (model_name=model_name, 
    generator=generator, generator_optimizer=generator_optimizer, 
    discriminator=discriminator, discriminator_optimizer=discriminator_optimizer)

### 決定 結果要寫哪邊 或 從哪邊讀資料
result_dir, log_dir, ckpt_dir = step4_get_default_dir_name(db_name=db_name, model_name=model_name)
if(restore_model==True):
    result_dir = restore_result_dir
    log_dir    = restore_log_dir 
    ckpt_dir   = restore_ckpt_dir
summary_writer = tf.summary.create_file_writer( log_dir ) ### 建tensorboard，這會自動建資料夾喔！
manager        = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份


### 決定 要不要讀取上次的結果
if(restore_model==True): 
    ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
    start_epoch = ckpt.epoch_log.numpy()
    print("load model ok~~~~~~~~~~~")
####################################################################################################
####################################################################################################

#######################################################################################################################
### training 的部分 ####################################################################################################
# total_start = time.time()
# for epoch in range(start_epoch, epochs):
#     print("Epoch: ", epoch)
#     e_start = time.time()

#     lr = 0.0002 if epoch < epoch_down_step else 0.0002*(epochs-epoch)/(epochs-epoch_down_step)
#     generator_optimizer.lr = lr
#     ###     用來看目前訓練的狀況 
#     for test_input, test_label in zip(test_db.take(1), test_label_db.take(1)): 
#         if  (model_name == "model1_UNet"):
#             generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#         elif(model_name == "model2_UNet_512to256"):
#             generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#         elif(model_name == "model3_UNet_stack"):
#             generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#         elif(model_name == "model4_UNet_and_D"):
#             generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#     ###     訓練
#     for n, (input_image, target) in enumerate( zip(train_db, train_label_db) ):
#         print('.', end='')
#         if (n+1) % 100 == 0:
#             print()
#         if  (model_name == "model1_UNet"):
#             train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
#         elif(model_name == "model2_UNet_512to256"):
#             train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
#         elif(model_name == "model3_UNet_stack"):
#             train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
#         elif(model_name == "model4_UNet_and_D"):
#             train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
#     print()

#     ###     儲存模型 (checkpoint) the model every 20 epochs
#     if (epoch + 1) % epoch_save_freq == 0:
#         ckpt.epoch_log.assign(epoch+1) ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
#         manager.save()

#     epoch_cost_time = time.time()-e_start
#     total_cost_time = time.time()-total_start
#     print('epoch %i cost time:%.2f'%(epoch + 1, epoch_cost_time)  )
#     print("batch cost time:%.2f"   %(epoch_cost_time/data_maount) )
#     print("total cost time:%s"     %(time_util(total_cost_time))  )
#     print("")
#######################################################################################################################
#######################################################################################################################



#######################################################################################################################
#######################################################################################################################
### testing 的部分 ####################################################################################################
from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec
import matplotlib.pyplot as plt
from util import get_dir_img, get_dir_move, get_max_move_xy_from_certain_move
from build_dataset_combine import Check_dir_exist_and_build
import numpy as np 
test_dir = access_path+result_dir + "/" + "testing"
Check_dir_exist_and_build(test_dir)
print("current_epoch_log", ckpt.epoch_log)
for i, (test_input, test_label) in enumerate(zip(test_db.take(200), test_label_db.take(200))): 
    print("i=",i)
    # if(i<65):
    #     continue

    col_img_num = 5
    fig, ax = plt.subplots(1,col_img_num)
    fig.set_size_inches(col_img_num*5,col_img_num) ### 2200~2300可以放4張圖，配500的高度，所以一張圖大概550~575寬，500高，但為了好計算還是用 500寬配500高好了！

    ### 圖. dis_img
    dis_imgs = get_dir_img(access_path+"datasets/pad2000-512to256/test/distorted_img") ### 這是沒有resize過的
    dis_img = dis_imgs[i]
    ### test_input是有resize過的！我們不是recover這個喔！
    # dis_img  = test_input[0].numpy() 
    # dis_img = (dis_img+1)*127.5
    # dis_img = dis_img.astype(np.uint8)
    ax[0].imshow(dis_img)
    ax[0].set_title("distorted_img")


    ### 圖. G predict的 move_map
    prediction = generator(test_input, training=True)   ### 用generator 去 predict扭曲流，注意這邊值是 -1~1
    prediction_back = (prediction[0]+1)/2 * (max_value_train-min_value_train) + min_value_train ### 把 -1~1 轉回原始的值域
    # prediction_back_bgr = method2(prediction_back[...,0], prediction_back[...,1],1)             ### predict出的扭曲流 視覺化
    # plt.imshow(prediction_back_bgr)
    # plt.show()
    g_move_map = prediction_back.numpy() ### 把 tensor轉numpy，在下面處理速度才會快
    g_move_bgr =  method2(g_move_map[...,0], g_move_map[...,1],1)
    ax[1].imshow(g_move_bgr)
    ax[1].set_title("predict_dis_flow")


    ###  拿g/gt 的move_map 來恢復dis_img
    ###   前置動作：拿到 當初建 dis_img_db時 用的 move_map max/min 的移動量
    max_move_x, max_move_y = get_max_move_xy_from_certain_move(access_path+"step3_apply_flow_result","2-q") ### 注意這裡要去 step3才對！因為當初建db時是用整個db的最大移動量(step3裡的即整個db的資料)，如果去dataset/train的話只有train的資料喔
    # print("max_move_x, max_move_y", max_move_x, max_move_y)

    ### 拿 dis_img 配 g_move_map 來做 rec囉！
    g_rec_img = apply_move_to_rec(dis_img, g_move_map, max_move_x, max_move_y)
    ax[3].imshow(g_rec_img.astype(np.uint8))
    ax[3].set_title("predict_rec")


    ### 拿gt流
    gt_moves = get_dir_move(access_path+"datasets/pad2000-512to256/test/rec_move_map")
    gt_move_map = gt_moves[i]
    gt_move_map_bgr = method2(gt_move_map[:,:,0],gt_move_map[:,:,1])
    ax[2].imshow(gt_move_map_bgr.astype(np.uint8))
    ax[2].set_title("gt_rec")
    ### 拿 dis_img 配 gt_move_map 來做 rec囉！
    gt_rec_img = apply_move_to_rec(dis_img, gt_move_map, max_move_x, max_move_y)
    ax[4].imshow(gt_rec_img.astype(np.uint8))
    ax[4].set_title("gt_rec")

    plt.savefig(access_path+test_dir + "/" + "index%02i-result.png"%i)
    # plt.show()
    plt.close()
#######################################################################################################################
#######################################################################################################################

### 下面是分流進去 step7裡面，目前失敗，因為進去step7後好像用不到這邊load的G，有空再調
# from step7_kong_model2_UNet_512to256 import testing

# from util import get_dir_move, get_max_move_xy_from_numpy
# move_list = get_dir_move("step2_flow_build/move_map")
# max_move_x, max_move_y = get_max_move_xy_from_numpy(move_list)
# row = max_move_y+max_move_y+256
# col = max_move_x+max_move_x+256
# row_rate = 512/row
# col_rate = 512/col
# max_move_x_resize = max_move_x * col_rate
# max_move_y_resize = max_move_y * row_rate

# print("row, col", row, col)
# print("max_move_x, max_move_y", max_move_x, max_move_y)
# print("max_move_x_resize, max_move_y_resize", max_move_x_resize, max_move_y_resize)

# result_dir = "testing"
# for index,(test_input, test_label)  in enumerate(zip(test_db.take(1), test_label_db.take(1))): 
#     if  (model_name == "model1_UNet"):
#         generate_images( generator, test_input, test_label, max_value_train, min_value_train, index, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#     elif(model_name == "model2_UNet_512to256"):
#         testing( generator, test_input, test_label, max_value_train, min_value_train, max_move_x_resize, max_move_y_resize, index, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#     elif(model_name == "model3_UNet_stack"):
#         generate_images( generator, test_input, test_label, max_value_train, min_value_train, index, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
#     elif(model_name == "model4_UNet_and_D"):
#         generate_images( generator, test_input, test_label, max_value_train, min_value_train, index, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，