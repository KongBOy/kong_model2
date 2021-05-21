import sys
sys.path.append("kong_util")
import tensorflow as tf
import pdb


@tf.function
def train_step_pure_G(model_obj, in_data, gt_data, loss_info_obj=None):
    # print("gt_data.min()", gt_data.numpy().min())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data.max()", gt_data.numpy().max())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data[..., 0].min()", gt_data.numpy()[..., 0].min())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data[..., 0].max()", gt_data.numpy()[..., 0].max())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data[..., 1].min()", gt_data.numpy()[..., 1].min())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data[..., 1].max()", gt_data.numpy()[..., 1].max())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data[..., 2].min()", gt_data.numpy()[..., 2].min())  ### 用這show的時候要先把 @tf.function註解掉
    # print("gt_data[..., 2].max()", gt_data.numpy()[..., 2].max())  ### 用這show的時候要先把 @tf.function註解掉
    # print("((gt_data.numpy() + 1) / 2).min()", ((gt_data.numpy() + 1) / 2).min())  ### 用這show的時候要先把 @tf.function註解掉
    # print("((gt_data.numpy() + 1) / 2).max()", ((gt_data.numpy() + 1) / 2).max())  ### 用這show的時候要先把 @tf.function註解掉
    with tf.GradientTape() as gen_tape:
        model_output = model_obj.generator(in_data)
        # print("model_output.min()", model_output.numpy().min())  ### 用這show的時候要先把 @tf.function註解掉
        # print("model_output.max()", model_output.numpy().max())  ### 用這show的時候要先把 @tf.function註解掉
        gen_loss  = loss_info_obj.loss_funs_dict["G"](model_output, gt_data)

    generator_gradients               = gen_tape .gradient(gen_loss, model_obj.generator.trainable_variables)
    # for gradient in generator_gradients:
    #     print("gradient", gradient)
    model_obj .optimizer_G .apply_gradients(zip(generator_gradients, model_obj.generator.trainable_variables))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    for loss_name in loss_info_obj.loss_containors.keys():
        loss_info_obj.loss_containors[loss_name](gen_loss)


@tf.function
# def train_step(rect2, in_data, gt_data, optimizer_G, optimizer_D, loss_info_obj ):
def train_step_GAN(model_obj, in_data, gt_data, loss_info_obj=None):
    with tf.GradientTape(persistent=True) as tape:
        g_g_data, fake_score, real_score = model_obj.rect(in_data, gt_data)
        loss_rec = loss_info_obj.loss_funs_dict["G"]     (g_g_data, gt_data, lamb=tf.constant(3., tf.float32))  ### 40 調回 3
        loss_g2d = loss_info_obj.loss_funs_dict["G_to_D"](fake_score, tf.ones_like(fake_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
        g_total_loss = loss_rec + loss_g2d

        loss_d_fake = loss_info_obj.loss_funs_dict["D_Fake"](fake_score, tf.zeros_like(fake_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
        loss_d_real = loss_info_obj.loss_funs_dict["D_Real"](real_score, tf.ones_like (real_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
        d_total_loss = (loss_d_real + loss_d_fake) / 2

    grad_D = tape.gradient(d_total_loss, model_obj.rect.discriminator.trainable_weights)
    grad_G = tape.gradient(g_total_loss, model_obj.rect.generator.    trainable_weights)
    model_obj.optimizer_D.apply_gradients(zip(grad_D, model_obj.rect.discriminator.trainable_weights))
    model_obj.optimizer_G.apply_gradients(zip(grad_G, model_obj.rect.generator.    trainable_weights))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    loss_info_obj.loss_containors["1_loss_rec"](loss_rec)
    loss_info_obj.loss_containors["2_loss_g2d"](loss_g2d)
    loss_info_obj.loss_containors["3_g_total_loss"](g_total_loss)
    loss_info_obj.loss_containors["4_loss_d_fake"](loss_d_fake)
    loss_info_obj.loss_containors["5_loss_d_real"](loss_d_real)
    loss_info_obj.loss_containors["6_d_total_loss"](d_total_loss)


@tf.function
def train_step_GAN2(model_obj, in_data, gt_data, loss_fun=None, loss_info_obj=None):
    for _ in range(1):
        with tf.GradientTape(persistent=True) as tape:
            g_g_data, fake_score, real_score = model_obj.rect(in_data, gt_data)
            loss_d_fake = loss_info_obj.loss_funs_dict["D_Fake"](fake_score, tf.zeros_like(fake_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
            loss_d_real = loss_info_obj.loss_funs_dict["D_Real"](real_score, tf.ones_like (real_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
            d_total_loss = (loss_d_real + loss_d_fake) / 2
        grad_D = tape.gradient(d_total_loss, model_obj.rect.discriminator.trainable_weights)
        model_obj.optimizer_D.apply_gradients(zip(grad_D, model_obj.rect.discriminator.trainable_weights))

        loss_info_obj.loss_containors["4_loss_d_fake"](loss_d_fake)
        loss_info_obj.loss_containors["5_loss_d_real"](loss_d_real)
        loss_info_obj.loss_containors["6_d_total_loss"](d_total_loss)


    for _ in range(5):
        with tf.GradientTape(persistent=True) as g_tape:
            g_g_data, fake_score, real_score = model_obj.rect(in_data, gt_data)
            loss_rec = loss_info_obj.loss_funs_dict["G"](g_g_data, gt_data, lamb=tf.constant(3., tf.float32))  ### 40 調回 3
            loss_g2d = loss_info_obj.loss_funs_dict["G_to_D"](fake_score, tf.ones_like(fake_score, dtype=tf.float32), lamb=tf.constant(0.1, tf.float32))
            g_total_loss = loss_rec + loss_g2d
        grad_G = g_tape.gradient(g_total_loss, model_obj.rect.generator.    trainable_weights)
        model_obj.optimizer_G.apply_gradients(zip(grad_G, model_obj.rect.generator.    trainable_weights))
        ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
        loss_info_obj.loss_containors["1_loss_rec"](loss_rec)
        loss_info_obj.loss_containors["2_loss_g2d"](loss_g2d)
        loss_info_obj.loss_containors["3_g_total_loss"](g_total_loss)
