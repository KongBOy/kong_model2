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

def step2_3_build_model_opti_ckpt(model_name): ### 我覺得這兩步是需要包起來做的，所以才多這個function，但又覺得有點多餘，先留著好了~
    generator, generator_optimizer,\
    discriminator, discriminator_optimizer,\
    generate_images, train_step              = step2_build_model_and_optimizer(model_name=model_name)

    ckpt  = step3_build_checkpoint (model_name=model_name, 
        generator=generator, generator_optimizer=generator_optimizer, 
        discriminator=discriminator, discriminator_optimizer=discriminator_optimizer)
    return  generator, generator_optimizer,\
            discriminator, discriminator_optimizer,\
            generate_images, train_step, ckpt


def step4_get_result_dir_default_logs_ckpt_dir_name(result_dir):
    logs_dir  = result_dir + "/" + "logs"
    #ckpt_dir = result_dir + '/' + 'ckpt_dir'+"_"+db_name+"_"+model_name ### 試試看不加感受起來會怎麼樣
    ckpt_dir = result_dir + '/' + 'ckpt_dir'
    return  logs_dir, ckpt_dir

def step4_get_datetime_default_result_logs_ckpt_dir_name(db_name, model_name):
    import datetime
    result_dir = access_path+"result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name+"_"+model_name  ### result資料夾，裡面放checkpoint和tensorboard資料夾
    logs_dir  = result_dir + "/" + "logs"
    #ckpt_dir = result_dir + '/' + 'ckpt_dir'+"_"+db_name+"_"+model_name ### 試試看不加感受起來會怎麼樣
    ckpt_dir = result_dir + '/' + 'ckpt_dir'
    return result_dir, logs_dir, ckpt_dir




if(__name__=="__main__"):
    ##############################################################################################################################
    ### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數
    BATCH_SIZE = 1

    phase = "test"
    db_dir  = access_path+"datasets"
    
    db_name = "pad2000-512to256" ### 這資料集 要搭配 512to256 的架構喔！

    # model_name="model1_UNet"
    model_name="model2_UNet_512to256"
    # model_name="model3_UNet_stack"
    # model_name="model4_UNet_and_D"

    ### train, train_reload 參數
    epochs = 160
    epoch_down_step = 100 ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
    epoch_save_freq = 20  ### 訓練 epoch_save_freq 個 epoch 存一次模型
    start_epoch = 0

    ### train_reload 和 test 參數
    restore_model = True ### 如果 restore_model 設True，下面 restore_result_dir 才會有用處喔！
    # restore_result_dir = "result/20200226-194945_pad2000-512to256_model2_UNet_512to256"
    restore_result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"

    ### 目前只有在算 b_cost_time會用到
    data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "distorted_img" )

    ### 參數設定結束
    ################################################################################################################################################
    
    ################################################################################################################################################
    ### 第一階段：datapipline、模型、訓練結果存哪邊
    ###    step1_data_pipline、step2_3_build_model_opti_ckpt 是 train, train_reload, test 都要做的事情
    train_db, train_label_db, test_db, test_label_db, max_value_train, min_value_train = step1_data_pipline(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE)
    generator, generator_optimizer,discriminator, discriminator_optimizer,generate_images, train_step, ckpt = step2_3_build_model_opti_ckpt(model_name=model_name)
        
    ###    step4 決定result, logs, ckpt 存哪裡，train的話跟據現在時間新增，train_reload和test 根據 restore_result_dir
    if  (phase=="train"):
        result_dir, logs_dir, ckpt_dir = step4_get_datetime_default_result_logs_ckpt_dir_name(db_name=db_name, model_name=model_name)
    elif(phase=="train_reload" or phase=="test"):
        result_dir = restore_result_dir
        logs_dir, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(result_dir)

    ###    step5 建立tensorboard，只有train 和 train_reload需要
    if  (phase=="train" or phase=="train_reload"):
        summary_writer = tf.summary.create_file_writer( logs_dir ) ### 建tensorboard，這會自動建資料夾喔！

    ###    step6 建立checkpoint manager，三者都需要
    manager = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份

    ###    step7 看需不需要reload model，只有train_reload 和 test需要
    if (phase=="train_reload" or phase=="test"):
        ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
        start_epoch = ckpt.epoch_log.numpy()
        print("load model ok~~~~~~~~~~~ current epoch log", start_epoch)

    ################################################################################################################################################
    ### 第二階段：train 和 test
    ###     training 的部分 ###################################################################################################
    ###     以下的概念就是，每個模型都有自己的 generate_images 和 train_step，根據model_name 去各別import 各自的 function過來用喔！
    if(phase=="train"or phase=="train_reload"):            
        total_start = time.time()
        for epoch in range(start_epoch, epochs):
            print("Epoch: ", epoch)
            e_start = time.time()

            lr = 0.0002 if epoch < epoch_down_step else 0.0002*(epochs-epoch)/(epochs-epoch_down_step)
            generator_optimizer.lr = lr
            ###     用來看目前訓練的狀況 
            for test_input, test_label in zip(test_db.take(1), test_label_db.take(1)): 
                if  (model_name == "model1_UNet"):
                    generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model2_UNet_512to256"):
                    generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model3_UNet_stack"):
                    generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model4_UNet_and_D"):
                    generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
            ###     訓練
            for n, (input_image, target) in enumerate( zip(train_db, train_label_db) ):
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                if  (model_name == "model1_UNet"):
                    train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
                elif(model_name == "model2_UNet_512to256"):
                    train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
                elif(model_name == "model3_UNet_stack"):
                    train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch)
                elif(model_name == "model4_UNet_and_D"):
                    train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
            print()

            ###     儲存模型 (checkpoint) the model every 20 epochs
            if (epoch + 1) % epoch_save_freq == 0:
                ckpt.epoch_log.assign(epoch+1) ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                manager.save()

            epoch_cost_time = time.time()-e_start
            total_cost_time = time.time()-total_start
            print('epoch %i cost time:%.2f'%(epoch + 1, epoch_cost_time)  )
            print("batch cost time:%.2f"   %(epoch_cost_time/data_maount) )
            print("total cost time:%s"     %(time_util(total_cost_time))  )
            print("")
    ######################################################################################################################
    ###     testing 的部分 ####################################################################################################
    elif(phase=="test"):
        if  (model_name == "model1_UNet"):pass ### 還沒做

        elif(model_name == "model2_UNet_512to256"):
            from step7_kong_model2_UNet_512to256 import test
            test(result_dir, test_db, max_value_train, min_value_train, generator)

        elif(model_name == "model3_UNet_stack"):pass ### 還沒做
        elif(model_name == "model4_UNet_and_D"):pass ### 還沒做


 