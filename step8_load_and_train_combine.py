import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from util import method2, get_db_amount
from step6_data_pipline import get_dataset
import time

tf.random.set_seed(123)
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

def time_util(cost_time):
    hour = cost_time//3600 
    minute = cost_time%3600//60 
    second = cost_time%3600%60
    return "%02i:%02i:%02i"%(hour, minute, second)
#######################################################################################################################################
# db_name = "stack_unet-easy300"
# db_name = "stack_unet-easy300"
# db_name = "stack_unet-padding2000" ### 這資料集 要搭配 512to256 的架構喔！
# model_name="model_1_G"
# model_name="model_2_G_512to256"
# model_name="model_3_G_stack"
# model_name="model_4_G_and_D"

if(__name__=="__main__"):
    from build_dataset_combine import Check_dir_exist_and_build
    import os
    ##############################################################################################################################
    ### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數
    BATCH_SIZE = 1
    
    db_dir  = "datasets"

    # db_name = "easy300"
    # db_name = "easy2000"
    db_name = "pad2000-512to256" ### 這資料集 要搭配 512to256 的架構喔！

    # model_name="model_1_G"
    model_name="model_2_G_512to256"
    # model_name="model_3_G_stack"
    # model_name="model_4_G_and_D"

    epochs = 160
    epoch_down_step = 100 ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
    epoch_save_freq = 20
    start_epoch = 0

    restore_train = False ### 如果 restore_train 設True，下面 restore_result_dir 和 restore_ckpt_dir 才會有用處喔！
    # restore_ckpt_dir = "." 
    restore_result_dir = "result/20200226-194945_pad2000-512to256_model_2_G_512to256"
    restore_ckpt_dir   = restore_result_dir + "/"  + "ckpt_dir" + "_" + db_name + "_" + model_name

    ### 目前只有在算 b_cost_time會用到
    data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "distorted_img" )
    ##############################################################################################################################
    ### step1.讀取 data_pipline
    img_resize  = None
    move_resize = None
    if  (model_name == "model_1_G"):          
        img_resize =(256,256)
        move_resize=(256,256)
    elif(model_name == "model_2_G_512to256"): 
        img_resize =(512,512)
        move_resize=(256,256)
    elif(model_name == "model_3_G_stack"):    
        img_resize =(256,256)
        move_resize=(256,256)
    elif(model_name == "model_4_G_and_D"):    
        img_resize =(256,256)
        move_resize=(256,256)

    start_time = time.time()
    train_db, train_label_db, \
    test_db , test_label_db , \
    max_value_train, min_value_train = get_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, move_resize=move_resize)
    ##############################################################################################################################
    ### step2.建立 model 和 optimizer
    start_time = time.time()
    generator               = None
    generator_optimizer     = None
    discriminator           = None
    discriminator_optimizer = None
    if  (model_name == "model_1_G"):
        from step7_kong_model_1_G import Generator, generate_images, train_step
        generator     = Generator(out_channel=2)
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    elif(model_name == "model_2_G_512to256"):
        from step7_kong_model_2_G_512to256 import Generator512to256, generate_images, train_step
        generator     = Generator512to256(out_channel=2)
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    elif(model_name == "model_3_G_stack"):
        from step7_kong_model_3_G_stack import Generator_stack, generate_images, train_step
        generator     = Generator_stack() ### 建立模型
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    elif(model_name == "model_4_G_and_D"):
        from step7_kong_model_4_G_and_D import Generator, Discriminator, generate_images, train_step
        generator     = Generator(out_channel=2)
        generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator = Discriminator()
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) ### 覺得還是不要統一寫出來，這樣比較好看每個模型需要什麼東西
    print("build model cost time:", time.time()-start_time)
    ##############################################################################################################################
    ### step3.建立 tensorboard 和 save/load model 的checkpoint
    ###     建立 checkpoint
    ckpt = None
    epoch_log = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
    if  (model_name == "model_1_G"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator)
    elif(model_name == "model_2_G_512to256"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator)
    elif(model_name == "model_3_G_stack"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator)
    elif(model_name == "model_4_G_and_D"):
        ckpt = tf.train.Checkpoint(epoch_log=epoch_log, generator_optimizer=generator_optimizer, generator=generator,
                                                        discriminator_optimizer=discriminator_optimizer, discriminator=discriminator)


    import datetime
    if(restore_train == False):
        ###     建立 放結果的資料夾名稱，大概長這樣： result/20200225-195407_stack_unet-pad2000_G_stack
        result_dir = "result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name+"_"+model_name  ### result資料夾，裡面放checkpoint和tensorboard資料夾
        ckpt_dir    = result_dir + '/' + 'ckpt_dir'+"_"+db_name+"_"+model_name ### checkpoint資料夾
        log_dir="logs/"  ### tensorboard 資料夾

        summary_writer = tf.summary.create_file_writer( result_dir + "/" + log_dir ) ### 建tensorboard，這會自動建資料夾喔！
        manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份

    elif(restore_train == True): ### 如果是要繼續上次的結果繼續訓練
        result_dir = restore_result_dir  ### result資料夾，裡面放checkpoint和tensorboard資料夾
        ckpt_dir   = restore_ckpt_dir    ### checkpoint資料夾
        log_dir    ="logs/"              ### tensorboard 資料夾
        
        summary_writer = tf.summary.create_file_writer( result_dir + "/" + log_dir ) ### 建tensorboard，
        ### 這裡沿用上次的資料夾但是會建立新的 envents.out.tfevents，在用tensorboard的時候遇到多個檔案放同個資料夾，他會把結果同時show在同個board上，所以可能會有些小片段會重複，除非save_freq設1，但花時間不切實際~~
        manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2分
        ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
        start_epoch = ckpt.epoch_log.numpy()        ### 讀出上次訓練到哪個epoch
    ##############################################################################################################################
    ### step4.開始訓練囉！
    total_start = time.time()
    for epoch in range(start_epoch, epochs):
        print("Epoch: ", epoch)
        e_start = time.time()

        lr = 0.0002 if epoch < epoch_down_step else 0.0002*(epochs-epoch)/(epochs-epoch_down_step)
        generator_optimizer.lr = lr
        ###     用來看目前訓練的狀況 
        for test_input, test_label in zip(test_db.take(1), test_label_db.take(1)): 
            if  (model_name == "model_1_G"):
                generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif(model_name == "model_2_G_512to256"):
                generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif(model_name == "model_3_G_stack"):
                generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            elif(model_name == "model_4_G_and_D"):
                generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
        ###     訓練
        for n, (input_image, target) in enumerate( zip(train_db, train_label_db) ):
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            if  (model_name == "model_1_G"):
                train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
            elif(model_name == "model_2_G_512to256"):
                train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
            elif(model_name == "model_3_G_stack"):
                train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
            elif(model_name == "model_4_G_and_D"):
                train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
        print()

        ###     儲存模型 (checkpoint) the model every 20 epochs
        if (epoch + 1) % epoch_save_freq == 0:
            epoch_log.assign(epoch)
            manager.save()

        epoch_cost_time = time.time()-e_start
        total_cost_time = time.time()-total_start
        print('epoch %i cost time:%.2f'%(epoch + 1, epoch_cost_time)  )
        print("batch cost time:%.2f"   %(epoch_cost_time/data_maount) )
        print("total cost time:%s"     %(time_util(total_cost_time))  )
        print("")