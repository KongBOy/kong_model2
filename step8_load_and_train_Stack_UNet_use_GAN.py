import os 
import tensorflow as tf
import matplotlib.pyplot as plt 
from util import method2
from step6_data_pipline_Stack_UNet import get_dataset
from step7_kong_model import Generator, Discriminator

import time

tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！


def generator_loss(disc_generated_output, gen_output, target):
    target = tf.cast(target,tf.float32)
    LAMBDA = 100

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


#######################################################################################################################################
@tf.function()
def train_step(generator, discriminator,generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output      = discriminator([input_image, target]    , training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, \
        gen_gan_loss  , \
        gen_l1_loss  = generator_loss(disc_generated_output, gen_output, target)
        disc_loss    = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients     = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)




def generate_images( model, test_input, test_label, max_value_train, min_value_train,  epoch=0, result_dir="."):
    sample_start_time = time.time()
    prediction = model(test_input, training=True)

    plt.figure(figsize=(20,6))
    display_list = [test_input[0], test_label[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
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
    from step6_data_pipline_Stack_UNet import get_dataset
    from build_dataset_combine import Check_dir_exist_and_build
    import os

    DATA_AMOUNT = 400
    BATCH_SIZE = 1
    
    db_dir  = "datasets"
    db_name = "stack_unet-easy2000"
    model_name="use_GAN"

    start_time = time.time()
    train_db, train_label_db, \
    test_db , test_label_db , \
    max_value_train, min_value_train = get_dataset(db_dir=db_dir, db_name=db_name,batch_size=BATCH_SIZE)
    ##############################################################################################################################
    start_time = time.time()
    generator     = Generator(out_channel=2)
    discriminator = Discriminator()

    generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir    = './training_checkpoints'+"_"+db_name
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint        = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                            discriminator_optimizer=discriminator_optimizer,
                                            generator=generator,
                                            discriminator=discriminator)
    print("build model cost time:", time.time()-start_time)
    ##############################################################################################################################
    epochs = 160
    epoch_step = 100
    import datetime

    result_dir = "result" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"_"+db_name
    # summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir="logs/"
    summary_writer = tf.summary.create_file_writer( result_dir + "/" + log_dir )


    for epoch in range(epochs):
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
            train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer, input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))