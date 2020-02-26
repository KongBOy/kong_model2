import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from util import method2
from step6_data_pipline import get_dataset
from step7_kong_model_stack import Generator_stack

import time

tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！


def generator_loss(y1, y2, target):
    target = tf.cast(target,tf.float32)

    y1_l1_loss = tf.reduce_mean(tf.abs(target - y1))
    y2_l1_loss = tf.reduce_mean(tf.abs(target - y2))

    total_gen_loss = y1_l1_loss + y2_l1_loss

    return y1_l1_loss, y2_l1_loss, total_gen_loss

#######################################################################################################################################
@tf.function()
def train_step(generator,generator_optimizer, summary_writer, input_image, target, epoch):
    with tf.GradientTape() as gen_tape:
        y1, y2 = generator(input_image, training=True)
        y1_l1_loss, y2_l1_loss, total_gen_loss  = generator_loss( y1, y2, target)

    generator_gradients     = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', total_gen_loss, step=epoch)
        tf.summary.scalar('y1_l1_loss', y1_l1_loss, step=epoch)
        tf.summary.scalar('y2_l1_loss', y2_l1_loss, step=epoch)




def generate_images( model, test_input, test_label, max_value_train, min_value_train,  epoch=0, result_dir="."):
    sample_start_time = time.time()
    y1, y2 = model(test_input, training=True)

    plt.figure(figsize=(20,6))
    display_list = [test_input[0], test_label[0], y1[0], y2[0]]
    title = ['Input Image', 'Ground Truth', 'y1 Image', 'y2 Image']

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        if(i==0):
            plt.imshow(display_list[i] * 0.5 + 0.5)
        else:
            back = (display_list[i]+1)/2 * (max_value_train-min_value_train) + min_value_train
            back_bgr = method2(back[...,0], back[...,1],1)
            plt.imshow(back_bgr)
        plt.axis('off')
    # plt.show()
    plt.savefig(result_dir + "/" + "epoch_%02i-result.png"%epoch)
    plt.close()
    print("sample image cost time:", time.time()-sample_start_time)


#######################################################################################################################################
if(__name__=="__main__"):
    from build_dataset_combine import Check_dir_exist_and_build
    import os

    DATA_AMOUNT = 400
    BATCH_SIZE = 1
    
    db_dir  = "datasets"
    db_name = "stack_unet-padding2000"

    model_name = "G_stack"

    restore_train = False

    start_time = time.time()
    train_db, train_label_db, \
    test_db , test_label_db , \
    max_value_train, min_value_train = get_dataset(db_dir=db_dir, db_name=db_name,batch_size=BATCH_SIZE)
    ##############################################################################################################################
    start_time = time.time()
    generator     = Generator_stack() ### 建立模型
    generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5) ### 建立 optimizer

    ### 建立 load/save 的checkpoint
    checkpoint_dir    = './training_checkpoints'+"_"+db_name+"_"+model_name
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint        = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                            generator=generator)
    if(restore_train): ### 如果是要繼續上次的結果繼續訓練
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)) 
    print("build model cost time:", time.time()-start_time)
    ##############################################################################################################################
    epochs = 160
    epoch_step = 100 ### 從 epoch_step 後開始下降learning rate
    import datetime

    ### 大概長這樣： result/20200225-195407_stack_unet-padding2000_G_stack
    result_dir = "result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name+"_"+model_name
    log_dir="logs/"
    summary_writer = tf.summary.create_file_writer( result_dir + "/" + log_dir )

    # restart_epoch = 20
    restart_epoch = 0
    for epoch in range(restart_epoch,epochs):
        print("Epoch: ", epoch)
        start = time.time()

        lr = 0.0002 if epoch < epoch_step else 0.0002*(epochs-epoch)/(epochs-epoch_step)
        generator_optimizer.lr = lr
        for test_input, test_label in zip(test_db.take(1), test_label_db.take(1)): ### 用來看目前訓練的狀況
            generate_images( generator, test_input, test_label, max_value_train, min_value_train,  epoch, result_dir) ### 這的視覺化用的max/min應該要丟 train的才合理，因為訓練時是用train的max/min，
        
        # Train
        for n, (input_image, target) in enumerate( zip(train_db, train_label_db) ):
            print('.', end='')
            if (n+1) % 100 == 0:
                print()
            train_step(generator, generator_optimizer, summary_writer, input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))