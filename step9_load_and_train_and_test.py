from step0_access_path import access_path
import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from util import method2, get_db_amount, time_util
from step6_data_pipline import get_unet_dataset, get_rect2_dataset, get_test_kong_dataset, get_test_kong_dataset_unet
import time


from build_dataset_combine import Check_dir_exist_and_build
import os

# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

### 第零階段：決定result, logs, ckpt 存哪裡 並 把source code存起來
import shutil
def step0_save_rect1_train_code(result_dir):
    code_dir = result_dir+"/"+"train_code"
    Check_dir_exist_and_build(code_dir)
    shutil.copy("step5_build_dataset.py"           ,code_dir + "/" + "step5_build_dataset.py")
    shutil.copy("step6_data_pipline.py"            ,code_dir + "/" + "step6_data_pipline.py")
    if  (model_name == "model1_UNet"):          shutil.copy("step7_kong_model1_UNet.py"          ,code_dir + "/" + "step7_kong_model1_UNet.py")
    elif(model_name == "model2_UNet_512to256"): shutil.copy("step7_kong_model2_UNet_512to256.py" ,code_dir + "/" + "step7_kong_model2_UNet_512to256.py")
    elif(model_name == "model3_UNet_stack"):    shutil.copy("step7_kong_model3_UNet_stack.py"    ,code_dir + "/" + "step7_kong_model3_UNet_stack.py")
    elif(model_name == "model4_UNet_and_D"):    shutil.copy("step7_kong_model4_UNet_and_D.py"    ,code_dir + "/" + "step7_kong_model4_UNet_and_D.py")
    
    shutil.copy("step9_load_and_train_and_test.py" ,code_dir + "/" + "step9_load_and_train_and_test.py")
    shutil.copy("util.py"                          ,code_dir + "/" + "util.py")

def step0_save_rect2_train_code(result_dir):
    code_dir = result_dir+"/"+"train_code"
    Check_dir_exist_and_build(code_dir)
    shutil.copy("step5_build_dataset.py"           ,code_dir + "/" + "step5_build_dataset.py")
    shutil.copy("step6_data_pipline.py"            ,code_dir + "/" + "step6_data_pipline.py")
    shutil.copy("step8_kong_model5_Rect2.py"       ,code_dir + "/" + "step8_kong_model5_Rect2.py")
    shutil.copy("step9_load_and_train_and_test.py" ,code_dir + "/" + "step9_load_and_train_and_test.py")
    shutil.copy("step11_unet_rec_img.py"           ,code_dir + "/" + "step11_unet_rec_img.py")
    shutil.copy("step12_gt_ord_or_ord_pad.py"             ,code_dir + "/" + "step12_gt_ord_or_ord_pad.py")
    shutil.copy("util.py"                          ,code_dir + "/" + "util.py")




def step1_data_pipline(phase, db_dir, db_name, model_name, batch_size=1):
    ### step1.讀取 data_pipline
    img_resize  = None
    move_resize = None
    data_dict = {}
    if(phase=="train" or phase=="train_reload" or phase=="test"):
        if(model_name in ["model1_UNet", "model2_UNet_512to256", "model3_UNet_stack", "model4_UNet_and_D"]):
            if  (model_name == "model1_UNet"):          img_resize =(256,256);move_resize=(256,256)
            elif(model_name == "model2_UNet_512to256"): 
                if  (db_name=="1_pad2000-512to256"):    img_resize =(256*2, 256*2);move_resize=(256, 256)
                elif(db_name=="1_page_h=384,w=256"):    img_resize =(384*2, 256*2);move_resize=(256, 384) ### 注意img_resize用tf的resize，h放前面喔！
            elif(model_name == "model3_UNet_stack"):    img_resize =(256, 256);move_resize=(256, 256)
            elif(model_name == "model4_UNet_and_D"):    img_resize =(256, 256);move_resize=(256, 256)

            data_dict = get_unet_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, move_resize=move_resize)

        elif(model_name == "model_rect2"):
            if  (db_name=="2_pure_rect2_h=256,w=256" ): img_resize = (512,512) 
            elif(db_name=="2_pure_rect2_h=384,w=256" ): img_resize = (512,384) 

            elif(db_name=="3_unet_rect2_h=256,w=256" ): img_resize = (256,256)
            elif(db_name=="3_unet_rect2_h=384,w=256" ): img_resize = (384,256)
            elif(db_name=="rect2_add_dis_imgs"): img_resize = (512,512) ### 做錯

    
            data_dict = get_rect2_dataset(db_dir=db_dir, db_name=db_name, img_resize=img_resize, batch_size=1)
    elif(phase=="test_kong"):
        ### 還沒寫if做區隔，先用註解區分這樣
        # img_resize = (512,512)
        # data_dict = get_test_kong_dataset(db_dir=db_dir, db_name=db_name, img_type="jpg" batch_size=1, img_resize=img_resize)


        img_resize =(512,512)
        move_resize=(256,256)
        data_dict = get_test_kong_dataset_unet(db_dir=db_dir, db_name=db_name, img_type="jpg", batch_size=BATCH_SIZE, img_resize=img_resize)

    return data_dict
    
def step2_build_model_and_optimizer(model_name="model1_UNet"):
    ### step2.建立 model 和 optimizer
    start_time = time.time()
    model_dict = {}
    if  (model_name == "model1_UNet"):
        from step7_kong_model1_UNet import Generator, generate_images, train_step
        model_dict["generator"] = Generator(out_channel=2)
        model_dict["generator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    elif(model_name == "model2_UNet_512to256"):
        from step7_kong_model2_UNet_512to256 import Generator512to256, generate_images, train_step
        model_dict["generator"] = Generator512to256(out_channel=2)
        model_dict["generator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    elif(model_name == "model3_UNet_stack"):
        from step7_kong_model3_UNet_stack import Generator_stack, generate_images, train_step
        model_dict["generator"] = Generator(out_channel=2)
        model_dict["generator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    elif(model_name == "model4_UNet_and_D"):
        from step7_kong_model4_UNet_and_D import Generator, Discriminator, generate_images, train_step
        model_dict["generator"] = Generator(out_channel=2)
        model_dict["generator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        model_dict["discriminator"] = Discriminator()
        model_dict["discriminator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    elif(model_name == "model_rect2"):
        from step8_kong_model5_Rect2 import Rect2, generate_images, train_step
        model_dict["rect2"] = Rect2()
        model_dict["generator_optimizer"]     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        model_dict["discriminator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    print("build model cost time:", time.time()-start_time)
    return model_dict, generate_images, train_step 

def step3_build_checkpoint(model_name, model_dict):
    ### step3.建立 save/load model 的checkpoint
    model_dict["epoch_log"] = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
    ckpt = tf.train.Checkpoint(**model_dict)
    return ckpt


def step2_3_build_model_opti_ckpt(model_name): ### 我覺得這兩步是需要包起來做的，所以才多這個function，但又覺得有點多餘，先留著好了~
    model_dict, \
    generate_images, \
    train_step = step2_build_model_and_optimizer(model_name=model_name)
    ckpt       = step3_build_checkpoint (model_name, model_dict)
    return  model_dict, generate_images, train_step, ckpt


def step4_get_result_dir_default_logs_ckpt_dir_name(result_dir):
    logs_dir = result_dir + "/" + "logs"
    ckpt_dir = result_dir + '/' + 'ckpt_dir'
    return  logs_dir, ckpt_dir

def step4_get_datetime_default_result_logs_ckpt_dir_name(db_name, model_name):
    import datetime
    result_dir = access_path+"result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name+"_"+model_name  ### result資料夾，裡面放checkpoint和tensorboard資料夾
    logs_dir  = result_dir + "/" + "logs"
    ckpt_dir = result_dir + '/' + 'ckpt_dir'
    return result_dir, logs_dir, ckpt_dir




if(__name__=="__main__"):
    ##############################################################################################################################
    ### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數
    BATCH_SIZE = 1

    phase = "train"
    # phase = "train_reload"
    # phase = "test"
    # phase = "test_kong"

    db_dir  = access_path+"datasets"
    
    # db_name = "1_pad2000-512to256" ### 這pad2000資料集 要搭配 512to256 的架構喔！
    db_name = "1_page_h=384,w=256"
    # db_name = "2_pure_rect2_h=256,w=256" 
    # db_name = "3_unet_rect2_h=256,w=256" 
    # db_name = "wei_book" 
    # db_name = "wei_book_w=576,h=575" 
    # db_name = "rect2_add_dis_imgs" ### 錯的

    # model_name="model1_UNet"
    model_name="model2_UNet_512to256"
    # model_name="model3_UNet_stack"
    # model_name="model4_UNet_and_D"
    # model_name="model_rect2"

    ### train, train_reload 參數
    epochs = 160
    epoch_down_step = 100 ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
    epoch_save_freq = 2  ### 訓練 epoch_save_freq 個 epoch 存一次模型
    start_epoch = 0

    ### train_reload 和 test 參數
    # restore_result_dir = "result/20200226-194945_pad2000-512to256_model2_UNet_512to256"
    # restore_result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"
    
    restore_result_dir = access_path+"result/20200227-071341_pad2000-512to256_model2_UNet_512to256"  ### unet
    # restore_result_dir = access_path+"result/20200309-214802_rect2_2000_model_rect2"               ### unet+rect2
    

    ### 目前只有在算 b_cost_time會用到
    if(phase == "train" or phase == "train_reload"):
        if  (model_name in ["model1_UNet", "model2_UNet_512to256", "model3_UNet_stack", "model4_UNet_and_D"]):
            data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "dis_imgs" )
        elif(model_name == "model_rect2"):
            if  ("2_pure_rect2" in db_name): data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "dis_img_db" )
            elif("3_unet_rect2" in db_name): data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "unet_rec_img_db" )
            # elif(db_name=="rect2_add_dis_imgs"):data_maount = get_db_amount(db_dir + "/" + db_name + "/" + "train" + "/" + "dis_and_unet_rec_img_db" )

    ### 參數設定結束
    ################################################################################################################################################
    ### 第零階段：決定result, logs, ckpt 存哪裡 並 把source code存起來

    ###    決定result, logs, ckpt 存哪裡
    if  (phase=="train"): ### train的話跟據 "現在時間" 
        result_dir, logs_dir, ckpt_dir = step4_get_datetime_default_result_logs_ckpt_dir_name(db_name=db_name, model_name=model_name)
    elif(phase=="train_reload" or phase=="test" or phase=="test_kong"): ### train_reload和test 根據 "restore_result_dir"
        result_dir = restore_result_dir
        logs_dir, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(result_dir)

    ###    把source code存起來
    if  (phase=="train" or phase=="train_reload"): ### 如果是訓練或重新訓練的話，把source_code存一份起來，reload的話就蓋過去
        if  (model_name in ["model1_UNet", "model2_UNet_512to256", "model3_UNet_stack", "model4_UNet_and_D"]):
            step0_save_rect1_train_code(result_dir)
        elif(model_name=="model_rect2"):
            step0_save_rect2_train_code(result_dir)

    ################################################################################################################################################
    ### 第一階段：datapipline、模型、訓練結果存哪邊
    ###    step1_data_pipline、step2_3_build_model_opti_ckpt 是 train, train_reload, test 都要做的事情
    data_dict  = step1_data_pipline(phase=phase, db_dir=db_dir, db_name=db_name, model_name=model_name, batch_size=BATCH_SIZE)
    
    model_dict, generate_images, train_step, ckpt = step2_3_build_model_opti_ckpt(model_name=model_name)

    ###    step4 建立tensorboard，只有train 和 train_reload需要
    if  (phase=="train" or phase=="train_reload"):
        summary_writer = tf.summary.create_file_writer( logs_dir ) ### 建tensorboard，這會自動建資料夾喔！

    ###    step5 建立checkpoint manager，三者都需要
    manager = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份

    ###    step6 看需不需要reload model，只有train_reload 和 test需要
    if (phase=="train_reload" or phase=="test" or phase=="test_kong"):
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

            model_dict["generator_optimizer"].lr = lr
            ###     用來看目前訓練的狀況 
            for test_input, test_gt in zip(data_dict["test_db"].take(1), data_dict["test_gt_db"].take(1)): 
                if  (model_name == "model1_UNet"):
                    generate_images( model_dict["generator"], test_input, test_gt, data_dict["max_train_move"], data_dict["min_train_move"],  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model2_UNet_512to256"):
                    generate_images( model_dict["generator"], test_input, test_gt, data_dict["max_train_move"], data_dict["min_train_move"],  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model3_UNet_stack"):
                    generate_images( model_dict["generator"], test_input, test_gt, data_dict["max_train_move"], data_dict["min_train_move"],  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model4_UNet_and_D"):
                    generate_images( model_dict["generator"], test_input, test_gt, data_dict["max_train_move"], data_dict["min_train_move"],  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model_rect2"):
                    generate_images( model_dict["rect2"].generator, test_input, test_gt, epoch, result_dir) 
            ###     訓練
            for n, (input_image, target) in enumerate( zip(data_dict["train_in_db"], data_dict["train_gt_db"]) ):
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                if  (model_name == "model1_UNet"):
                    train_step(model_dict["generator"], model_dict["generator_optimizer"], summary_writer, input_image, target, epoch)
                elif(model_name == "model2_UNet_512to256"):
                    train_step(model_dict["generator"], model_dict["generator_optimizer"], summary_writer, input_image, target, epoch)
                elif(model_name == "model3_UNet_stack"):
                    train_step(model_dict["generator"], model_dict["generator_optimizer"], summary_writer, input_image, target, epoch)
                elif(model_name == "model4_UNet_and_D"):
                    train_step(model_dict["generator"], model_dict["discriminator"], model_dict["generator_optimizer"], model_dict["discriminator_optimizer"], summary_writer, input_image, target, epoch)
                elif(model_name == "model_rect2"):
                    train_step(model_dict["rect2"], input_image, target, model_dict["generator_optimizer"], model_dict["discriminator_optimizer"], summary_writer, epoch)
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
    elif(phase=="test" or phase=="test_kong"):
        ### 決定 test_dir_name
        test_dir_name = ""
        if  (phase=="test"):
            test_dir_name  = "test"
            test_db_amount = 200
        elif(phase=="test_kong"): 
            test_dir_name  = "test_kong"
            test_db_amount = 83

        if  (model_name == "model1_UNet"):pass ### 還沒做

        elif(model_name == "model2_UNet_512to256"):
            from step7_kong_model2_UNet_512to256 import test
            test(phase=phase, result_dir=result_dir, test_dir_name=test_dir_name, test_db=data_dict["test_db"],test_db_amount=test_db_amount, 
                max_train_move=data_dict["max_train_move"], min_train_move=data_dict["min_train_move"],  generator=model_dict["generator"])

        elif(model_name == "model3_UNet_stack"):pass ### 還沒做
        elif(model_name == "model4_UNet_and_D"):pass ### 還沒做

        elif(model_name == "model_rect2"):
            from step8_kong_model5_Rect2 import test
            test(result_dir=result_dir, test_dir_name=test_dir_name, test_db=data_dict["test_db"], test_gt_db=data_dict["test_gt_db"],test_db_amount=test_db_amount, rect2=model_dict["rect2"])