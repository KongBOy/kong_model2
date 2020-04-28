import sys
sys.path.append("kong_util")
from step0_access_path import access_path
import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from util import method2, get_db_amount, time_util

import time


from build_dataset_combine import Check_dir_exist_and_build
import os

from enum import Enum 
import numpy as np 

class MODEL_NAME(Enum):
    Unet      = 1
    Rect2     = 2
    Mrf_rect2 = 3

# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

### 第零階段：決定result, logs, ckpt 存哪裡 並 把source code存起來
import shutil
def step0_save_rect1_train_code(result_dir):
    code_dir = result_dir+"/"+"train_code"
    Check_dir_exist_and_build(code_dir)
    shutil.copy("step5_build_dataset.py" ,code_dir + "/" + "step5_build_dataset.py")
    shutil.copy("step6_data_pipline.py"  ,code_dir + "/" + "step6_data_pipline.py")
    if  (model_name == "model1_UNet"):          shutil.copy("step7_kong_model1_UNet.py"          ,code_dir + "/" + "step7_kong_model1_UNet.py")
    elif(model_name == "model2_UNet_512to256"): shutil.copy("step7_kong_model2_UNet_512to256.py" ,code_dir + "/" + "step7_kong_model2_UNet_512to256.py")
    elif(model_name == "model3_UNet_stack"):    shutil.copy("step7_kong_model3_UNet_stack.py"    ,code_dir + "/" + "step7_kong_model3_UNet_stack.py")
    elif(model_name == "model4_UNet_and_D"):    shutil.copy("step7_kong_model4_UNet_and_D.py"    ,code_dir + "/" + "step7_kong_model4_UNet_and_D.py")
    
    shutil.copy("step9_load_and_train_and_test.py" ,code_dir + "/" + "step9_load_and_train_and_test.py")

def step0_save_rect2_train_code(result_dir):
    code_dir = result_dir+"/"+"train_code"
    Check_dir_exist_and_build(code_dir)
    shutil.copy("step5_build_dataset.py"           ,code_dir + "/" + "step5_build_dataset.py")
    shutil.copy("step6_data_pipline.py"            ,code_dir + "/" + "step6_data_pipline.py")
    shutil.copy("step8_kong_model5_Rect2.py"       ,code_dir + "/" + "step8_kong_model5_Rect2.py")
    shutil.copy("step9_load_and_train_and_test.py" ,code_dir + "/" + "step9_load_and_train_and_test.py")
    # shutil.copy("util.py"                          ,code_dir + "/" + "util.py")


def step1_build_model_and_optimizer(model_name="model1_UNet"):
    ### step2.建立 model 和 optimizer
    start_time = time.time()
    model_dict = {}
    
    if  (model_name == "model2_UNet_512to256"):
        from step7_kong_model2_UNet_512to256 import Generator512to256, generate_images, train_step
        model_dict["generator"] = Generator512to256(out_channel=2)
        model_dict["generator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        model_dict["max_train_move"] = tf.Variable(1) ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
        model_dict["min_train_move"] = tf.Variable(1) ### 在test時 把move_map值弄到-1~1需要，所以需要存起來
        model_dict["max_db_move_x"]  = tf.Variable(1) ### 在test時 rec_img需要，所以需要存起來
        model_dict["max_db_move_y"]  = tf.Variable(1) ### 在test時 rec_img需要，所以需要存起來
    
    elif(model_name == "model5_rect2"):
        from step8_kong_model5_Rect2 import Rect2, generate_images, train_step
        model_dict["rect2"] = Rect2()
        model_dict["generator"] = model_dict["rect2"].generator
        model_dict["generator_optimizer"]     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        model_dict["discriminator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    elif(model_name == "model6_mrf_rect2"):
        from step8_kong_model5_Rect2 import Rect2, generate_images, train_step
        model_dict["mrf_rect2"] = Rect2(use_mrfb=True)
        model_dict["generator"] = model_dict["mrf_rect2"].generator
        model_dict["generator_optimizer"]     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        model_dict["discriminator_optimizer"] = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    print("build model cost time:", time.time()-start_time)
    return model_dict, generate_images, train_step 

def step2_build_checkpoint(model_name, model_dict):
    ### step3.建立 save/load model 的checkpoint
    model_dict["epoch_log"] = tf.Variable(1) ### 用來記錄 在呼叫.save()時 是訓練到幾個epoch
    ckpt = tf.train.Checkpoint(**model_dict)
    return ckpt


def step1_2_build_model_opti_ckpt(model_name): ### 我覺得這兩步是需要包起來做的，所以才多這個function，但又覺得有點多餘，先留著好了~
    model_dict, \
    generate_images, \
    train_step = step1_build_model_and_optimizer(model_name=model_name)
    ckpt       = step2_build_checkpoint (model_name, model_dict)
    return  model_dict, generate_images, train_step, ckpt

def step3_build_tensorboard(model_name, logs_dir):

    def see_loss(result_dir, baord_dict, epochs):
        loss_dir = result_dir + "/" + "logs"
        for loss_name in baord_dict.keys():
            plt.figure(figsize=(20,6))                              ### 建立畫布
            plt.ylim(0,1.3)
            plt.ylabel(loss_name)
            y_loss_array = np.load(loss_dir + "/" + loss_name + ".npy")
            
            plt.xlim(0,epochs)
            plt.xlabel("epoch_num")
            x_epoch = np.arange(len(y_loss_array))

            plt.plot(x_epoch, y_loss_array)
            plt.savefig(loss_dir + "/" + loss_name + ".png")
            plt.close()
            # print("plot %s loss ok~"%loss_name )
        print("plot loss ok~" )


    summary_writer = tf.summary.create_file_writer( logs_dir ) ### 建tensorboard，這會自動建資料夾喔！
    board_dict = {}
    if  (model_name == MODEL_NAME.Unet):
        board_dict["gen_l1_loss"]    = tf.keras.metrics.Mean('gen_l1_loss', dtype=tf.float32)
    elif(model_name == MODEL_NAME.Rect2):
        board_dict["1_loss_rec"]     = tf.keras.metrics.Mean('1_loss_rec'    , dtype=tf.float32)
        board_dict["2_loss_g2d"]     = tf.keras.metrics.Mean('2_loss_g2d'    , dtype=tf.float32)
        board_dict["3_g_total_loss"] = tf.keras.metrics.Mean('3_g_total_loss', dtype=tf.float32)
        board_dict["4_loss_d_fake"]  = tf.keras.metrics.Mean('4_loss_d_fake' , dtype=tf.float32)
        board_dict["5_loss_d_real"]  = tf.keras.metrics.Mean('5_loss_d_real' , dtype=tf.float32)
        board_dict["6_d_total_loss"] = tf.keras.metrics.Mean('6_d_total_loss', dtype=tf.float32)
    return summary_writer, board_dict, see_loss


def step4_get_result_dir_default_logs_ckpt_dir_name(result_dir):
    logs_dir = result_dir + "/" + "logs"
    ckpt_dir = result_dir + '/' + 'ckpt_dir'
    return  logs_dir, ckpt_dir

def step4_get_datetime_default_result_logs_ckpt_dir_name(db_name, model_name):
    import datetime
    # result_dir = access_path+"result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name+"_"+model_name  ### result資料夾，裡面放checkpoint和tensorboard資料夾
    
    ### 大概長這樣 h=256,w=256_complex_2_pure_rect2_20200328-215330_model5_rect2 
    result_dir = access_path+"result" + "/" + "%s_%s_%s"%(db_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), model_name)  ### result資料夾，裡面放checkpoint和tensorboard資料夾
    logs_dir  = result_dir + "/" + "logs"
    ckpt_dir = result_dir + '/' + 'ckpt_dir'
    return result_dir, logs_dir, ckpt_dir



def step6_data_pipline(phase, db_dir="", db_name="", model_name="", test_in_dir=None, test_gt_dir=None, gt_type="img", img_type="bmp", batch_size=1, have_see=False):
    from step6_data_pipline import get_1_pure_unet_db  , \
                               get_2_pure_rect2_dataset, \
                               get_2_pure_rect2_v2_dataset, \
                               get_3_unet_rect2_dataset, \
                               get_test_indicate_db 

    ### 第一部分：根據 model_name 和 db_name 做相對應的 resize
    ### 注意img_resize用tf的resize，h放前面喔！
    img_resize  = None
    if  (model_name == "model2_UNet_512to256"): 
        if  (db_name== "h=384,w=256_old_page_1_pure_unet"                  ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "h=256,w=256_complex_1_pure_unet"                   ): img_resize =(256*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "h=384,w=256_complex_1_pure_unet"                   ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "h=384,w=256_complex+page_1_pure_unet"              ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "h=384,w=256_complex+page_more_like_1_pure_unet"    ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "h=384,w=256_smooth-curl+fold_and_page_1_pure_unet" ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_type1_h=256,w=256_complex"                ): img_resize =(256*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_type2_h=384,w=256_complex"                ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_type3_h=384,w=256_complex+page"           ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_type4_h=384,w=256_complex+page_more_like" ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_pad_type1_h=256,w=256_complex"                ): img_resize =(256*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_pad_type2_h=384,w=256_complex"                ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_pad_type3_h=384,w=256_complex+page"           ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "wei_book_pad_type4_h=384,w=256_complex+page_more_like" ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "no-bg_gt_color"                                        ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "no-bg_gt_gray_3ch"                                     ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "type5c-real_have_see-no_bg-gt-gray3ch"                 ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        elif(db_name== "type5c-real_have_see-no_bg-gt-color"                   ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！
        
        elif(db_name== "apple_shoot_1_pure_unet"                               ): img_resize =(384*2, 256*2) ### 比dis_img(in_img的大小) 大一點且接近的 128的倍數，且要是gt_img的兩倍大喔！

    elif(model_name == "model5_rect2" or 
         model_name == "model6_mrf_rect2"):
        if  (db_name== "h=384,w=256_old_page_2_pure_rect2"                 ): img_resize = (494+2,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "h=256,w=256_complex_2_pure_rect2"                  ): img_resize = (365+3,336) ### dis_img(in_img的大小)的大小且要是4的倍數 ###(512, 512)
        elif(db_name== "h=384,w=256_complex_2_pure_rect2"                  ): img_resize = (492+0,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "h=384,w=256_complex+page_2_pure_rect2"             ): img_resize = (492+0,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "h=384,w=256_complex+page_more_like_2_pure_rect2"   ): img_resize = (492+0,384) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "h=384,w=256_smooth-curl+fold_and_page_2_pure_rect2"): img_resize = (492+0,384) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_type1_h=256,w=256_complex"                ): img_resize = (356+3,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_type2_h=384,w=256_complex"                ): img_resize = (492+0,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_type3_h=384,w=256_complex+page"           ): img_resize = (492+0,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_type4_h=384,w=256_complex+page_more_like" ): img_resize = (492+0,384) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_pad_type1_h=256,w=256_complex"                ): img_resize = (356+3,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_pad_type2_h=384,w=256_complex"                ): img_resize = (492+0,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_pad_type3_h=384,w=256_complex+page"           ): img_resize = (492+0,336) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_pad_type4_h=384,w=256_complex+page_more_like" ): img_resize = (492+0,384) ### dis_img(in_img的大小)的大小且要是4的倍數
        
        elif(db_name== "wei_book_1_type4_complex+page_more_like"               ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_2_tf1_db"                                     ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "wei_book_3_tf1_db+type4_complex+page_more_like"        ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "no-bg_gt_color"                                        ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "no-bg_gt_gray_3ch"                                     ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "type5c-real_have_see-no_bg-gt-gray3ch"                 ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        elif(db_name== "type5c-real_have_see-no_bg-gt-color"                   ): img_resize = (472+0,304) ### dis_img(in_img的大小)的大小且要是4的倍數
        
        

        elif(db_name== "h=384,w=256_old_page_3_unet_rect2"                 ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "h=256,w=256_complex_3_unet_rect2"                  ): img_resize = (256,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "h=384,w=256_complex_3_unet_rect2"                  ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "h=384,w=256_complex+page_3_unet_rect2"             ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "h=384,w=256_complex+page_more_like_3_unet_rect2"   ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "h=384,w=256_smooth-curl+fold_and_page_3_unet_rect2"): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_type1_h=256,w=256_complex"                ): img_resize = (256,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_type2_h=384,w=256_complex"                ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_type3_h=384,w=256_complex+page"           ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_type4_h=384,w=256_complex+page_more_like" ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_pad_type1_h=256,w=256_complex"                ): img_resize = (256,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_pad_type2_h=384,w=256_complex"                ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_pad_type3_h=384,w=256_complex+page"           ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "wei_book_pad_type4_h=384,w=256_complex+page_more_like" ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "type5c-real_have_see-no_bg-gt-gray3ch"                 ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
        elif(db_name== "type5c-real_have_see-no_bg-gt-color"                   ): img_resize = (384,256) ### ord_img(in_img的大小)的大小
    

    ### 第二部分：根據 db_name 去相應的 dir結構抓出所有data
    # ( 然後我覺得不要管什麼 test就只抓test、train就全抓，這些什麼model抓什麼資料這種邏輯判斷應該要寫再外面，這裡專心抓資料就好！要不然會不好擴增！
    #   我現在已經改成從file_name讀所以不占記憶體不用弄這麼麻煩，也不好擴增，所以現在就改成抓該db的全部囉！)
    data_dict = {}

    if(phase=="train" or phase=="train_reload" or phase=="test"):
        if  (db_name == "h=384,w=256_old_page_1_pure_unet"      ): data_dict = get_1_pure_unet_db      (db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
        elif(db_name == "h=384,w=256_old_page_2_pure_rect2"     ): data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "h=384,w=256_old_page_3_unet_rect2"     ): data_dict = get_3_unet_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )

        elif(db_name == "h=256,w=256_complex_1_pure_unet"       ): data_dict = get_1_pure_unet_db      (db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
        elif(db_name == "h=256,w=256_complex_2_pure_rect2"      ): data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "h=256,w=256_complex_3_unet_rect2"      ): data_dict = get_3_unet_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        
        elif(db_name == "h=384,w=256_complex_1_pure_unet"       ): data_dict = get_1_pure_unet_db      (db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
        elif(db_name == "h=384,w=256_complex_2_pure_rect2"      ): data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "h=384,w=256_complex_3_unet_rect2"      ): data_dict = get_3_unet_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        
        elif(db_name == "h=384,w=256_complex+page_1_pure_unet"  ): data_dict = get_1_pure_unet_db      (db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
        elif(db_name == "h=384,w=256_complex+page_2_pure_rect2" ): data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "h=384,w=256_complex+page_3_unet_rect2" ): data_dict = get_3_unet_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        
        elif(db_name == "h=384,w=256_complex+page_more_like_1_pure_unet"  ): data_dict = get_1_pure_unet_db      (db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
        elif(db_name == "h=384,w=256_complex+page_more_like_2_pure_rect2" ): data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "h=384,w=256_complex+page_more_like_3_unet_rect2" ): data_dict = get_3_unet_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        
        elif(db_name == "h=384,w=256_smooth-curl+fold_and_page_1_pure_unet"  ): data_dict = get_1_pure_unet_db      (db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
        elif(db_name == "h=384,w=256_smooth-curl+fold_and_page_2_pure_rect2" ): data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "h=384,w=256_smooth-curl+fold_and_page_3_unet_rect2" ): data_dict = get_3_unet_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        
        elif(db_name == "wei_book_1_type4_complex+page_more_like"           ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "wei_book_2_tf1_db"                                 ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, have_see=have_see )
        elif(db_name == "wei_book_3_tf1_db+type4_complex+page_more_like"    ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
        elif(db_name == "no-bg_gt_color"                                    ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, have_see=have_see )
        elif(db_name == "no-bg_gt_gray_3ch"                                 ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, have_see=have_see )
        elif(db_name == "type5c-real_have_see-no_bg-gt-gray3ch"             ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, have_see=have_see )
        elif(db_name == "type5c-real_have_see-no_bg-gt-color"               ): data_dict = get_2_pure_rect2_v2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize, have_see=have_see )

        elif(db_name == "wei_book_h=384,w=256"                  ): data_dict = get_test_indicate_db  (test_in_dir=test_in_dir, test_gt_dir=test_gt_dir, gt_type="img", img_type="jpg", img_resize=img_resize)
    elif(phase=="test_indicate"):
        indicate_args = {}
        indicate_args["img_resize"]  = img_resize
        indicate_args["test_in_dir"] = test_in_dir
        indicate_args["test_gt_dir"] = test_gt_dir

        ### 根據 model_name 決定 gt_type
        if  (model_name == "model2_UNet_512to256"): indicate_args["gt_type"] = "move_map"
        elif(model_name == "model5_rect2" or 
             model_name == "model6_mrf_rect2"    ): indicate_args["gt_type"] = "img"


        ### 根據 db_name 決定 gt_img 是用 jpg 或 bmp 或是 其他種類喔，覺得有點多餘，先統一用bmp拉！
        # if  (db_name== "h=256,w=256_complex_1_pure_unet"       or
        #      db_name== "h=256,w=256_complex_2_pure_rect2"      or
        #      db_name== "h=256,w=256_complex_3_unet_rect2"      or

        #      db_name== "h=384,w=256_complex_1_pure_unet"       or
        #      db_name== "h=384,w=256_complex_2_pure_rect2"      or
        #      db_name== "h=384,w=256_complex_3_unet_rect2"      or

        #      db_name== "h=384,w=256_complex+page_1_pure_unet"  or
        #      db_name== "h=384,w=256_complex+page_2_pure_rect2" or
        #      db_name== "h=384,w=256_complex+page_3_unet_rect2" or

        #      db_name== "h=384,w=256_complex+page_more_like_1_pure_unet"  or
        #      db_name== "h=384,w=256_complex+page_more_like_2_pure_rect2" or
        #      db_name== "h=384,w=256_complex+page_more_like_3_unet_rect2" ):  indicate_args["img_type"] = "bmp"
        # elif(db_name== "wei_book_type1_h=256,w=256_complex"       or
        #      db_name== "wei_book_type2_h=384,w=256_complex"       or
        #      db_name== "wei_book_type3_h=384,w=256_complex+page"  or
        #      db_name== "wei_book_type4_h=384,w=256_complex+page_more_like"): indicate_args["img_type"] = "jpg"
        indicate_args["img_type"] = "bmp"

        data_dict = get_test_indicate_db(**indicate_args)
        # if  (model_name == "model2_UNet_512to256"): data_dict = get_test_indicate_db(test_in_dir=test_in_dir, test_gt_dir=test_gt_dir, gt_type="move_map", img_type="bmp", img_resize=img_resize)
        # elif(model_name == "model5_rect2" or 
        #      model_name == "model6_mrf_rect2"    ): data_dict = get_test_indicate_db(test_in_dir=test_in_dir, test_gt_dir=test_gt_dir, gt_type="img"     , img_type="bmp", img_resize=img_resize)
    return data_dict




if(__name__=="__main__"):
    ##############################################################################################################################
    ### step0.設定 要用的資料庫 和 要使用的模型 和 一些訓練參數

    ### train, train_reload 參數
    BATCH_SIZE = 1
    epochs = 1300  ### 看opencv合成的video覺得1300左右就沒變了
    epoch_down_step = 100 ### 在第 epoch_down_step 個 epoch 後開始下降learning rate
    epoch_save_freq = 1   ### 訓練 epoch_save_freq 個 epoch 存一次模型
    start_epoch = 0

    have_see = True
    # phase = "train"

    # restore_model_name = ""

    # test_in_dir = ""
    # test_gt_dir = ""
    # phase = "train_reload" ### 要記得去決定 restore_model_name 喔！
    # phase = "test"         ### test是用固定 train/test 資料夾架構的讀法 ### 要記得去決定 restore_model_name 喔！
    ####################################################################################################################

    ### model_name/db_name 決定如何resize
    # model_name="model2_UNet_512to256"
    model_name="model5_rect2"
    model_name_enum = MODEL_NAME.Rect2
    # model_name="model6_mrf_rect2"

    ### 設定 restore_model_name 來讀取網路weight，在phase==train_reload、test、test_indicate 時需要
    ### h=256,w=256_complex
    # restore_model_name = "h=256,w=256_complex_1_pure_unet_20200328-170738_model2_UNet_512to256_finish" ### 1.pure_unet
    # restore_model_name = "h=256,w=256_complex_2_pure_rect2_20200329-001847_model5_rect2_finish"        ### 2.pure_rect2
    # restore_model_name = "h=256,w=256_complex_3_unet_rect2_20200328-215330_model5_rect2_finish"        ### 3.unet_rect2
    # restore_model_name = "h=256,w=256_complex_4_3_unet_rect2_20200329-135446_model6_mrf_rect2_finish"  ### 4.unet_mrf-rect2
    # restore_model_name = "h=256,w=256_complex_5_2_pure_rect2_20200330-111543_model6_mrf_rect2_finish"  ### 5.pure_mrf-rect2

    ### h=384,w=256_complex
    # restore_model_name = "h=384,w=256_complex_1_pure_unet_20200329-215628_model2_UNet_512to256_finish" ### 1.pure_unet
    # restore_model_name = "h=384,w=256_complex_2_pure_rect2_20200329-213756_model5_rect2_finish"        ### 2.pure_rect2
    # restore_model_name = "h=384,w=256_complex_3_unet_rect2_20200330-095524_model5_rect2_finish"        ### 3.unet_rect2
    # restore_model_name = "h=384,w=256_complex_4_3_unet_rect2_20200331-090143_model6_mrf_rect2_finish"  ### 4.unet_mrf-rect2
    # restore_model_name = "h=384,w=256_complex_5_3_unet_rect2_20200402-120915_model6_mrf_rect2_finish"  ### 5.pure_mrf-rect2

    ### h=384,w=256_complex+page
    # restore_model_name = "h=384,w=256_complex+page_1_pure_unet_20200329-232144_model2_UNet_512to256_finish" ### 1.pure_unet
    # restore_model_name = "h=384,w=256_complex+page_2_pure_rect2_20200331-084031_model5_rect2 finish"        ### 2.pure_rect2
    # restore_model_name = "h=384,w=256_complex+page_3_unet_rect2_20200330-231953_model5_rect2_finish"        ### 3.unet_rect2
    # restore_model_name = "h=384,w=256_complex+page_4_3_unet_rect2_20200401-030348_model6_mrf_rect2_finish"  ### 4.unet_mrf-rect2
    # restore_model_name = "h=384,w=256_complex+page_5_2_pure_rect2_20200401-085744_model6_mrf_rect2_finish"  ### 5.pure_mrf-rect2

    ### h=384,w=256_complex+page_more_like
    # restore_model_name = "h=384,w=256_complex+page_more_like_1_pure_unet_20200406-214854_model2_UNet_512to256_finish" ### 1.pure_unet
    # restore_model_name = "h=384,w=256_complex+page_more_like_2_pure_rect2_20200406-215120_model5_rect2_finish"        ### 2.pure_rect2
    # restore_model_name = "h=384,w=256_complex+page_more_like_3_unet_rect2_20200407-093115_model5_rect2_finish"        ### 3.unet_rect2
    # restore_model_name = "h=384,w=256_complex+page_more_like_4_3_unet_rect2_20200407-110250_model6_mrf_rect2_finish"    ### 4.unet_mrf_rect2
    # restore_model_name =  ### 5.pure_mrf_rect2


    ### 合成影像/真實影像/兩者混和 loss太大
    # restore_model_name = "pure_rect2_wrong-loss_no-shuffle/" + "wei_book_1_type4_complex+page_more_like_20200413-230418_model5_rect2_127.40-epoch-392"  ### 合成影像 的     rect2 
    # restore_model_name = "pure_rect2_wrong-loss_no-shuffle/" + "wei_book_1_type4_complex+page_more_like_20200413-230835_model6_mrf_rect2_128.242-epoch=203"  ### 合成影像 的 mrf-rect2 
    # restore_model_name = "pure_rect2_wrong-loss_no-shuffle/" + "wei_book_2_tf1_db_20200408-225902_model5_rect2"                               ### 真實影像 的     rect2 
    # restore_model_name = "pure_rect2_wrong-loss_no-shuffle/" + "wei_book_2_tf1_db_20200410-025655_model6_mrf_rect2"                           ### 真實影像 的 mrf-rect2 
    # restore_model_name = "pure_rect2_wrong-loss_no-shuffle/" + "wei_book_3_tf1_db+type4_complex+page_more_like_20200413-220059_model5_rect2"  ### 兩者混和 的     rect2 
    # restore_model_name = "pure_rect2_wrong-loss_no-shuffle/" + "wei_book_3_tf1_db+type4_complex+page_more_like_20200413-220341_model6_mrf_rect2_128.243-epoch=183"  ### 兩者混和 的 mrf-rect2 

    ### 合成影像/真實影像/兩者混和 loss正確
    # restore_model_name = "pure_rect2_right-loss/" + "wei_book_1_type4_complex+page_more_like_20200416-104901_model5_rect2_127.40-epoch=935"  ### 合成影像 的     rect2 
    # restore_model_name = "pure_rect2_right-loss/" + "wei_book_1_type4_complex+page_more_like_20200416-105102_model6_mrf_rect2_128.243-epoch=548"  ### 合成影像 的 mrf-rect2 
    # restore_model_name = "pure_rect2_right-loss/" + "wei_book_2_tf1_db_20200415-160605_model5_rect2"                               ### 真實影像 的     rect2 
    # restore_model_name = "pure_rect2_right-loss/" + "wei_book_2_tf1_db_20200416-005448_model6_mrf_rect2_right-loss_no-shuffle"                           ### 真實影像 的 mrf-rect2 
    # restore_model_name =    "wei_book_3_tf1_db+type4_complex+page_more_like_20200422-011813_model5_rect2"  ### 兩者混和 的     rect2 
    # restore_model_name = "pure_rect2_right-loss/" + "還沒train"  ### 兩者混和 的 mrf-rect2 
    
    
    
    # restore_model_name = "no-bg_gt_color_20200427-004020_model5_rect2" 
    restore_model_name = "type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2" 


    ####################################################################################################################
    ### 看要讀取 哪個特定的in/gt資料集，在phase== train、train_load、test 時需要

    # db_dir  = access_path+"datasets/type1_h=256,w=256_complex"
    # db_name = "h=256,w=256_complex_1_pure_unet"
    # db_name = "h=256,w=256_complex_2_pure_rect2" 
    # db_name = "h=256,w=256_complex_3_unet_rect2" 

    # db_dir  = access_path+"datasets/type2_h=384,w=256_complex"
    # db_name = "h=384,w=256_complex_1_pure_unet"
    # db_name = "h=384,w=256_complex_2_pure_rect2" 
    # db_name = "h=384,w=256_complex_3_unet_rect2" 
    
    # db_dir  = access_path+"datasets/type3_h=384,w=256_complex+page"
    # db_name = "h=384,w=256_complex+page_1_pure_unet"
    # db_name = "h=384,w=256_complex+page_2_pure_rect2" 
    # db_name = "h=384,w=256_complex+page_3_unet_rect2" 
    
    # db_dir  = access_path+"datasets/type4_h=384,w=256_complex+page_more_like"
    # db_name = "h=384,w=256_complex+page_more_like_1_pure_unet"
    # db_name = "h=384,w=256_complex+page_more_like_2_pure_rect2" 
    # db_name = "h=384,w=256_complex+page_more_like_3_unet_rect2" 


    # db_dir  = access_path+"datasets/type5_wei_book_try_real_or_sync"
    # db_name = "wei_book_1_type4_complex+page_more_like"  ### 合成影像
    # db_name = "wei_book_2_tf1_db"  ### 真實影像
    # db_name = "wei_book_3_tf1_db+type4_complex+page_more_like"   ### 真實影像+合成影像

    ### type5b 先留著好了，之後用不到可刪除
    # db_dir  = access_path+"datasets"
    # db_name = "no-bg_gt_color"  ### 去背，gt彩色
    # db_name = "no-bg_gt_gray_3ch"  ### 去背，gt灰階

    ### type5c
    db_dir  = access_path+"datasets/type5c-real_have_see-no_bg-gt-color&gray3ch"
    db_name = "type5c-real_have_see-no_bg-gt-gray3ch"  ### 去背，gt灰階 有see的概念
    # db_name = "type5c-real_have_see-no_bg-gt-color"  ### 去背，gt彩色 有see的概念
    


    # db_dir  = access_path+"datasets/type6_h=384,w=256_smooth-curl+fold_and_page"
    # db_name = "h=384,w=256_smooth-curl+fold_and_page_1_pure_unet"  
    # db_name = "h=384,w=256_smooth-curl+fold_and_page_2_pure_rect2"  
    # db_name = "h=384,w=256_smooth-curl+fold_and_page_3_unet_rect2"   

    ####################################################################################################################
    ### 讀取自訂的 in/gt 資料集，在phase== test_indicate 決定
    # phase = "test_indicate" ###用自己決定的db來做test

    # db_name = "apple_shoot_1_pure_unet"
    # test_in_dir = access_path+"datasets"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = None



    ### wei_book系列 ###
    # db_name = "wei_book_type1_h=256,w=256_complex"
    # db_name = "wei_book_pad_type1_h=256,w=256_complex"
    ### 1.pure_unet
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = None
    ### 2.pure_rect2 和 5.pure_mrf-rect2
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"gt_imgs"
    ### 3.unet+rect2 和 4.unet_mrf-rect2
    # test_in_dir = access_path + "result"  + "/" + "h=256,w=256_complex_1_pure_unet_20200328-170738_model2_UNet_512to256_finish" + "/" + "test_indicate" + "_" + db_name
    # test_gt_dir = access_path + "datasets/test_indicate"+ "/" + db_name + "/" + "gt_imgs"


    # db_name = "wei_book_type2_h=384,w=256_complex"
    # db_name = "wei_book_pad_type2_h=384,w=256_complex"
    ### 1.pure_unet
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = None
    ### 2.pure_rect2 和 5.pure_mrf-rect2
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"gt_imgs"
    ### 3.unet+rect2 和 4.unet_mrf-rect2
    # test_in_dir = access_path + "result"  + "/" + "h=384,w=256_complex_1_pure_unet_20200329-215628_model2_UNet_512to256_finish" + "/" + "test_indicate" + "_" + db_name
    # test_gt_dir = access_path + "datasets/test_indicate"+ "/" + db_name + "/" + "gt_imgs"

    
    # db_name = "wei_book_type3_h=384,w=256_complex+page" 
    # db_name = "wei_book_pad_type3_h=384,w=256_complex+page" 
    ### 1.pure_unet
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = None
    ### 2.pure_rect2 和 5.pure_mrf-rect2
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"gt_imgs"
    ### 3.unet+rect2 和 4.unet_mrf-rect2
    # test_in_dir = access_path + "result"  + "/" + "h=384,w=256_complex+page_1_pure_unet_20200329-232144_model2_UNet_512to256_finish" + "/" + "test_indicate" + "_" + db_name
    # test_gt_dir = access_path + "datasets/test_indicate"+ "/" + db_name + "/" + "gt_imgs"



    # db_name = "wei_book_type4_h=384,w=256_complex+page_more_like"
    # db_name = "wei_book_pad_type4_h=384,w=256_complex+page_more_like"
    ### 1.pure_unet
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = None
    ### 2.pure_rect2 和 5.pure_mrf-rect2
    # test_in_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"in_imgs"
    # test_gt_dir = access_path+"datasets/test_indicate"+"/"+db_name+"/"+"gt_imgs"
    ### 3.unet+rect2 和 4.unet_mrf-rect2
    # test_in_dir = access_path + "result"  + "/" + "h=384,w=256_complex+page_more_like_1_pure_unet_20200406-214854_model2_UNet_512to256_finish" + "/" + "test_indicate" + "_" + db_name
    # test_gt_dir = access_path + "datasets/test_indicate"+ "/" + db_name + "/" + "gt_imgs"
    ####################################################################################################################
    

    ### 訓練時 生成 3_unet_rect2系列
    # db_name = "h=384,w=256_smooth-curl+fold_and_page_1_pure_unet"
    # test_in_dir = access_path+"datasets/type6_h=384,w=256_smooth-curl+fold_and_page/h=384,w=256_smooth-curl+fold_and_page_pure_unet/train+test/dis_imgs"
    # test_gt_dir = access_path+"datasets/type6_h=384,w=256_smooth-curl+fold_and_page/h=384,w=256_smooth-curl+fold_and_page_pure_unet/train+test/move_maps"
    
    # db_name = "h=384,w=256_complex+page_more_like_1_pure_unet"
    # test_in_dir = access_path+"datasets/type4_h=384,w=256_complex+page_more_like/h=384,w=256_complex+page_more_like_1_pure_unet/train+test/dis_imgs"
    # test_gt_dir = access_path+"datasets/type4_h=384,w=256_complex+page_more_like/h=384,w=256_complex+page_more_like_1_pure_unet/train+test/move_maps"
    
    # db_name = "h=384,w=256_complex+page_1_pure_unet"
    # test_in_dir = access_path+"datasets/type3_h=384,w=256_complex+page/h=384,w=256_complex+page_1_pure_unet/train+test/dis_imgs"
    # test_gt_dir = access_path+"datasets/type3_h=384,w=256_complex+page/h=384,w=256_complex+page_1_pure_unet/train+test/move_maps"
    
    # db_name = "h=256,w=256_complex_1_pure_unet"
    # test_in_dir = access_path+"datasets/h=256,w=256,complex/h=256,w=256_complex_1_pure_unet/train+test/dis_imgs"
    # test_gt_dir = access_path+"datasets/h=256,w=256,complex/h=256,w=256_complex_1_pure_unet/train+test/move_maps"

    # db_name = "h=384,w=256_complex_1_pure_unet"
    # test_in_dir = access_path+"datasets/2_h=384,w=256_complex/h=384,w=256_complex_1_pure_unet/train+test/dis_imgs"
    # test_gt_dir = access_path+"datasets/2_h=384,w=256_complex/h=384,w=256_complex_1_pure_unet/train+test/move_maps"

    
    ### 參數設定結束
    ################################################################################################################################################
    ### 第零階段：決定result, logs, ckpt 存哪裡 並 把source code存起來
    ###  決定result, logs, ckpt 存哪裡 
    ###     train的話跟據 "現在時間"
    if  (phase=="train"):  
        result_dir, logs_dir, ckpt_dir = step4_get_datetime_default_result_logs_ckpt_dir_name(db_name=db_name, model_name=model_name)
    ###     train_reload和test 根據 "restore_result_dir"
    elif(phase=="train_reload" or phase=="test" or phase=="test_indicate"): 
        result_dir = access_path+ "result" + "/" + restore_model_name
        logs_dir, ckpt_dir = step4_get_result_dir_default_logs_ckpt_dir_name(result_dir)

    ###    把source code存起來
    if  (phase=="train" or phase=="train_reload"): ### 如果是訓練或重新訓練的話，把source_code存一份起來，reload的話就蓋過去
        if  (model_name == "model2_UNet_512to256"): step0_save_rect1_train_code(result_dir)
        elif(model_name=="model5_rect2" or 
             model_name=="model6_mrf_rect2")          : step0_save_rect2_train_code(result_dir)

    ################################################################################################################################################
    ### 第一階段：1.,2.模型、3.tensorboard、4.checkpoint、5.看需不需要reload model
    model_dict, generate_images, train_step, ckpt = step1_2_build_model_opti_ckpt(model_name=model_name)

    ###    step3 建立tensorboard，只有train 和 train_reload需要
    if  (phase=="train" or phase=="train_reload"):
        # summary_writer = tf.summary.create_file_writer( logs_dir ) ### 建tensorboard，這會自動建資料夾喔！
        summary_writer, board_dict, see_loss = step3_build_tensorboard( model_name_enum, logs_dir)


    ###    step4 建立checkpoint manager，三者都需要
    manager = tf.train.CheckpointManager (checkpoint=ckpt, directory=ckpt_dir, max_to_keep=2) ### checkpoint管理器，設定最多存2份

    ###    step5 看需不需要reload model，只有train_reload 和 test需要
    if (phase=="train_reload" or phase=="test" or phase=="test_indicate"):
        ckpt.restore(manager.latest_checkpoint)     ### 從restore_ckpt_dir 抓存的model出來
        start_epoch = ckpt.epoch_log.numpy()
        print("load model ok~~~~~~~~~~~ current epoch log", start_epoch)
    
    ################################################################################################################################################
    ### 第二階段：抓資料
    if  (phase in ["train", "train_reload", "test"]):
        data_dict  = step6_data_pipline(phase=phase, db_dir=db_dir, db_name=db_name, model_name=model_name, batch_size=BATCH_SIZE, have_see=have_see)
    elif(phase == "test_indicate"):
        data_dict  = step6_data_pipline(phase=phase,                db_name=db_name, model_name=model_name, test_in_dir=test_in_dir, test_gt_dir=test_gt_dir, batch_size=BATCH_SIZE)
    ################################################################################################################################################
    ### 第三階段：train 和 test
    ###  training 的部分 ###################################################################################################
    ###     以下的概念就是，每個模型都有自己的 generate_images 和 train_step，根據model_name 去各別import 各自的 function過來用喔！
    if(phase=="train"or phase=="train_reload"):            
        total_start = time.time()
        if(model_name == "model2_UNet_512to256"): ### 因為 unet 有move_map的部分，所以要多做以下操作 把 move_map相關會用到的東西存起來
            from util import get_max_db_move_xy
            ckpt.max_train_move.assign(data_dict["max_train_move"])  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            ckpt.min_train_move.assign(data_dict["min_train_move"])  ### 在test時 把move_map值弄到-1~1需要，所以要存起來
            max_db_move_x, max_db_move_y = get_max_db_move_xy(db_dir=db_dir, db_name=db_name) ### g生成的結果 做 apply_rec_move用
            ckpt.max_db_move_x.assign(max_db_move_x)  ### 在test時 rec_img需要，所以要存起來
            ckpt.max_db_move_y.assign(max_db_move_y)  ### 在test時 rec_img需要，所以要存起來
            manager.save()
            print("save ok ~~~~~~~~~~~~~~~~~")

        for epoch in range(start_epoch, epochs):
            print("Epoch: ", epoch)
            e_start = time.time()

            lr = 0.0002 if epoch < epoch_down_step else 0.0002*(epochs-epoch)/(epochs-epoch_down_step)
            model_dict["generator_optimizer"].lr = lr


            ###############################################################################################################################
            ###     用來看目前訓練的狀況 
            sample_start_time = time.time()
            see_in = "test_in_db_pre"
            see_gt = "test_gt_db_pre"
            see_amount = 1
            if(epoch==0):print("Initializing Model~~~") ### sample的時候就會initial model喔！
            if(have_see):
                see_in = "see_in_db_pre"
                see_gt = "see_gt_db_pre"
                see_amount = data_dict["see_amount"]
            for see_index, (test_in_pre, test_gt_pre) in enumerate(zip(data_dict[see_in].take(see_amount), data_dict[see_gt].take(see_amount))): 
                if  (model_name == "model2_UNet_512to256" ): generate_images( model_dict["generator"]          ,see_index, test_in_pre, test_gt_pre, data_dict["max_train_move"], data_dict["min_train_move"],  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
                elif(model_name == "model5_rect2" ):         generate_images( model_dict["rect2"]    .generator,see_index, test_in_pre, test_gt_pre, epoch, result_dir) 
                elif(model_name == "model6_mrf_rect2" ):     generate_images( model_dict["mrf_rect2"].generator,see_index, test_in_pre, test_gt_pre, epoch, result_dir) 
            print("sample all see time:", time.time()-sample_start_time)
            ###############################################################################################################################
            ###     訓練
            for n, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate( data_dict["train_db_combine"] ):
                print('.', end='')
                if (n+1) % 100 == 0: print()
                if  (model_name == "model2_UNet_512to256"):train_step(model_dict["generator"], model_dict["generator_optimizer"], train_in_pre, train_gt_pre, board_dict)
                elif(model_name == "model5_rect2")        :train_step(model_dict["rect2"]    , train_in_pre, train_gt_pre, model_dict["generator_optimizer"], model_dict["discriminator_optimizer"], board_dict)
                elif(model_name == "model6_mrf_rect2")    :train_step(model_dict["mrf_rect2"], train_in_pre, train_gt_pre, model_dict["generator_optimizer"], model_dict["discriminator_optimizer"], board_dict)

            ###############################################################
            ###     整個epoch 的 loss 算平均，存進tensorboard
            with summary_writer.as_default():
                for loss_name, loss_containor in board_dict.items():
                    tf.summary.scalar(loss_name, loss_containor.result(), step=epoch)
                    loss_value = loss_containor.result().numpy()
                    if(epoch == 0): ### 第一次 直接把值存成np.array
                        np.save(logs_dir + "/" + loss_name, np.array(loss_value.reshape(1)))

                    else: ### 第二次後，先把np.array先讀出來append值後 再存進去
                        loss_array = np.load(logs_dir + "/" + loss_name + ".npy")
                        loss_array = np.append(loss_array, loss_value)
                        np.save(logs_dir + "/" + loss_name, np.array(loss_array))
                        # print(loss_array)
            see_loss(result_dir, board_dict, epochs)
            ###############################################################
            ###    reset tensorboard 的 loss紀錄容器
            for loss_containor in board_dict.values():
                loss_containor.reset_states()
            ###############################################################################################################################
            ###     儲存模型 (checkpoint) the model every 20 epochs
            if (epoch + 1) % epoch_save_freq == 0:
                ckpt.epoch_log.assign(epoch+1) ### 要存+1才對喔！因為 這個時間點代表的是 本次epoch已做完要進下一個epoch了！
                manager.save()
                print("save ok ~~~~~~~~~~~~~~~~~")

            epoch_cost_time = time.time()-e_start
            total_cost_time = time.time()-total_start
            print('epoch %i cost time:%.2f'      %(epoch , epoch_cost_time     ))
            print("batch cost time:%.2f average" %(epoch_cost_time/data_dict["train_amount"] ))
            print("total cost time:%s"           %(time_util(total_cost_time)  ))
            print("esti total time:%s"           %(time_util(epoch_cost_time*epochs)))
            print("esti least time:%s"           %(time_util(epoch_cost_time*(epochs-(epoch+1)))))
            print("")
            with open(result_dir + "/" +"cost_time.txt","a") as f:
                f.write( phase )                                                                         ;f.write("\n") 
                f.write('epoch %i cost time:%.2f'      %(epoch , epoch_cost_time     ))                  ;f.write("\n")
                f.write("batch cost time:%.2f average" %(epoch_cost_time/data_dict["train_amount"] ))    ;f.write("\n")
                f.write("total cost time:%s"           %(time_util(total_cost_time)  ))                  ;f.write("\n")
                f.write("esti total time:%s"           %(time_util(epoch_cost_time*epochs)))             ;f.write("\n")
                f.write("esti least time:%s"           %(time_util(epoch_cost_time*(epochs-(epoch+1))))) ;f.write("\n")
                f.write("\n")
                

    
    ######################################################################################################################
    ###  testing 的部分 ####################################################################################################
    elif(phase=="test" or phase=="test_indicate"):
        import numpy as np 
        import cv2
        from util import get_dir_img
        
        ### 決定 test_dir_name
        if(phase=="test"):          test_dir_name = result_dir + "/" + "test_"          + db_name 
        if(phase=="test_indicate"): test_dir_name = result_dir + "/" + "test_indicate_" + db_name 
        Check_dir_exist_and_build(test_dir_name)


        ### 如果是用unet，會需要一些額外處理move_map的東西，在這邊從model裡load出來喔！
        if  (model_name=="model2_UNet_512to256"):
            max_train_move = ckpt.max_train_move.numpy() ### g生成的結果 值-1~1 還原用
            min_train_move = ckpt.min_train_move.numpy() ### g生成的結果 值-1~1 還原用
            max_db_move_x  = ckpt.max_db_move_x.numpy()  ### g生成的結果 做 apply_rec_move用
            max_db_move_y  = ckpt.max_db_move_y.numpy()  ### g生成的結果 做 apply_rec_move用

            Check_dir_exist_and_build(test_dir_name+"/"+"move_map_visual")  ### g生成的結果視覺化存的地方，一定要這個喔，因為它存的是bmp，我在test_indicate回來讀的時候會讀到他，所以要多個資料夾存裡面，才不會被讀到這樣子 ~
        
        ##################################################################################################################################
        ### test部分
        for i, (test_input, test_input_pre) in enumerate(zip(data_dict["test_in_db"], data_dict["test_in_db_pre"])):
            ### 把 preprocess過的test_input丟進去generator生成prediction
            prediction = model_dict["generator"](test_input_pre, training=True)  ### prediction 的shape是 BWHC 喔！
            prediction = prediction[0].numpy()
            print("testing %06i"%i, "predict success~~")
            ### unet predict出來的是move_map
            if  (model_name=="model2_UNet_512to256"):
                ###  G 生成 的 move_map 轉回對的值域 並 存起來
                prediction_back = (prediction+1)/2 * (max_train_move-min_train_move) + min_train_move ### 把 -1~1 轉回原始的值域
                g_move_map = prediction_back  ### 換個好懂的名字
                np.save(test_dir_name+"/%06i_g_move_map"%i,g_move_map) ### 存起來

                ### 生成的move_map視覺化 並 存起來
                g_move_map_bgr =  method2(g_move_map[...,0], g_move_map[...,1],1) ### method2回傳的是 uint8的array喔！
                cv2.imwrite(test_dir_name+"/move_map_visual/%06i_g_move_map_visual.bmp"%i, g_move_map_bgr) ### 存起來，記得存到 move_map_visual 資料夾裡喔！

                ### 用生成的move_map 來 還原 test_input 並 存起來
                dis_img   = test_input.numpy()[0] ### 這是沒有resize過的！recover是要用這個來做喔！不是用test_input_pre resize過的dis_img來做！[0]是因為建tf.dataset時有用batch        
                from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec2
                g_rec_img = apply_move_to_rec2(dis_img, g_move_map, max_db_move_x, max_db_move_y) ### 生成的move_map 來 還原 test_input
                g_rec_img = g_rec_img.astype(np.uint8)
                g_rec_img = g_rec_img[...,::-1]
                cv2.imwrite(test_dir_name+"/%06i_g_rec_img.bmp"%i, g_rec_img) ### 存起來

            ### rect2 predict出來的是img
            elif("rect2" in model_name):
                ###  G 生成 的 img 轉回對的值域 並 存起來
                prediction_back = (prediction+1)*127.5  ### 把 -1~1 轉回原始的值域
                prediction_back =  prediction_back.astype(np.uint8)
                rec_img = prediction_back[:,:,::-1]   ### 換個好懂的名字
                cv2.imwrite(test_dir_name+"/%06i.bmp"%i, rec_img)  ### 存起來

        ##########################################################################################
        ### test_visual
        if  (model_name == "model2_UNet_512to256"):
            from step7_kong_model2_UNet_512to256 import test_visual
            test_visual( test_dir_name=test_dir_name, model_dict=model_dict, data_dict=data_dict, start_index=0)
        elif(model_name == "model5_rect2" or model_name == "model6_mrf_rect2"):
            from step8_kong_model5_Rect2 import test_visual
            test_visual( test_dir_name=test_dir_name, data_dict=data_dict, start_index=0)