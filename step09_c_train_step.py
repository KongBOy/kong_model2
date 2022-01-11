import sys
sys.path.append("kong_util")
import tensorflow as tf
import pdb
from step09_a_loss import *
'''
會想把 train_step 獨立一個.py 寫 function， 還不包成 class 的原因是：
    因為 有些架構 用 的 train_step 是一樣的， 所以 先只寫成 function， 給各個架構掛上去
'''

def one_loss_info_obj_total_loss(loss_info_objs, model_output, gt_data):
    losses = []
    total_loss = 0
    for loss_name, loss_fun in loss_info_objs.loss_funs_dict.items():
        # print("loss_name:", loss_name)
        if("tv" in loss_name): losses.append(loss_fun(model_output))
        else:                  losses.append(loss_fun(gt_data, model_output))
        total_loss += losses[-1]
    return total_loss, losses
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################

def _train_step_Multi_output(model_obj, in_data, gt_datas, loss_info_objs=None):
    with tf.GradientTape() as gen_tape:
        model_outputs = model_obj.generator(in_data)
        # print("in_data.numpy().shape", in_data.numpy().shape)
        # print("model_output.min()", model_output.numpy().min())  ### 用這show的時候要先把 @tf.function註解掉
        # print("model_output.max()", model_output.numpy().max())  ### 用這show的時候要先把 @tf.function註解掉
        multi_losses = []
        multi_total_loss = 0
        for go_m, model_output in enumerate(model_outputs):
            total_loss, losses = one_loss_info_obj_total_loss(loss_info_objs[go_m], model_output, gt_datas[go_m])
            multi_losses.append(losses)
            multi_total_loss += total_loss

        # gen_loss = loss_info_objs.loss_funs_dict["mask_BCE"]      (gt_data, model_output)
        # sob_loss = loss_info_objs.loss_funs_dict["mask_Sobel_MAE"](gt_data, model_output)
        # total_loss = gen_loss + sob_loss

    total_gradients = gen_tape .gradient(multi_total_loss, model_obj.generator.trainable_variables)
    # for gradient in generator_gradients:
    #     print("gradient", gradient)
    model_obj .optimizer_G .apply_gradients(zip(total_gradients, model_obj.generator.trainable_variables))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    for go_l, loss_info_objs in enumerate(loss_info_objs):
        for go_containor, loss_containor in enumerate(loss_info_objs.loss_containors.values()):
            loss_containor( multi_losses[go_l][go_containor] )
    # loss_info_objs.loss_containors["mask_bce_loss"]      (gen_loss)
    # loss_info_objs.loss_containors["mask_sobel_MAE_loss"](sob_loss)
####################################################
@tf.function
def train_step_Multi_output_I_to_M_w_I_to_C(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    gt_mask  = gt_data[..., 0:1]
    gt_coord = gt_data[..., 1:3]
    gt_datas = [gt_mask, gt_coord]

    _train_step_Multi_output(model_obj, in_data=in_data, gt_datas=gt_datas, loss_info_objs=loss_info_objs)


@tf.function
def train_step_Multi_output_I_w_M_to_Cx_Cy(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    gt_mask  = gt_data[..., 0:1]
    I_with_M = in_data * gt_mask

    gt_cx = gt_data[..., 2:3]
    gt_cy = gt_data[..., 1:2]
    gt_datas = [gt_cx, gt_cy]  ### 沒辦法當初設定成這樣子train， 就只能繼續保持這樣子了，要不然以前train好的東西 不能繼續用下去 QQ
    # print("gt_cx.numpy().shape", gt_cx.numpy().shape)
    # print("gt_cy.numpy().shape", gt_cy.numpy().shape)

    ## debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(I_with_M[0])
    # ax[2].imshow(gt_cx[0])
    # ax[3].imshow(gt_cy[0])
    # fig.tight_layout()
    # plt.show()

    _train_step_Multi_output(model_obj, in_data=I_with_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs)

@tf.function
def train_step_Multi_output_W_w_M_to_Cx_Cy(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    in_Mask  = in_data[..., 3:4]
    in_W     = in_data[..., 0:3]
    W_w_M = in_W * in_Mask

    gt_cx = gt_data[..., 2:3]
    gt_cy = gt_data[..., 1:2]
    gt_datas = [gt_cx, gt_cy]  ### 沒辦法當初設定成這樣子train， 就只能繼續保持這樣子了，要不然以前train好的東西 不能繼續用下去 QQ
    # print("gt_cx.numpy().shape", gt_cx.numpy().shape)
    # print("gt_cy.numpy().shape", gt_cy.numpy().shape)

    ## debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(W_w_M[0])
    # ax[2].imshow(gt_cx[0])
    # ax[3].imshow(gt_cy[0])
    # fig.tight_layout()
    # plt.show()

    _train_step_Multi_output(model_obj, in_data=W_w_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs)

@tf.function
def train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    gt_mask  = gt_data[..., 3:4]
    I_with_M = in_data * gt_mask

    Wxgt = gt_data[..., 2:3]
    Wygt = gt_data[..., 1:2]
    Wzgt = gt_data[..., 0:1]
    gt_datas = [Wzgt, Wygt, Wxgt]

    ### debug 時 記得把 @tf.function 拿掉
    # print("in_data.shape", in_data.shape)
    # print("gt_data.shape", gt_data.shape)
    # print("gt_mask.shape", gt_mask.shape)
    # print("I_with_M.shape", I_with_M.shape)
    # print("Wxgt.shape", Wxgt.shape)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    # ax[0, 0].imshow(in_data[0])
    # ax[0, 1].imshow(gt_mask[0])
    # ax[0, 2].imshow(I_with_M[0])
    # ax[1, 0].imshow(Wxgt[0])
    # ax[1, 1].imshow(Wygt[0])
    # ax[1, 2].imshow(Wzgt[0])
    # fig.tight_layout()
    # plt.show()

    _train_step_Multi_output(model_obj, in_data=I_with_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs)


###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
### 因為外層function 已經有 @tf.function， 裡面這層自動會被 decorate 到喔！ 所以這裡不用 @tf.function
def _train_step_Single_output(model_obj, in_data, gt_data, loss_info_objs):
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
        # print("in_data.numpy().shape", in_data.numpy().shape)
        # print("model_output.min()", model_output.numpy().min())  ### 用這show的時候要先把 @tf.function註解掉
        # print("model_output.max()", model_output.numpy().max())  ### 用這show的時候要先把 @tf.function註解掉
        total_loss, losses = one_loss_info_obj_total_loss(loss_info_objs[0], model_output, gt_data)
        # gen_loss = loss_info_objs.loss_funs_dict["mask_BCE"]      (gt_data, model_output)
        # sob_loss = loss_info_objs.loss_funs_dict["mask_Sobel_MAE"](gt_data, model_output)
        # total_loss = gen_loss + sob_loss

    total_gradients = gen_tape .gradient(total_loss, model_obj.generator.trainable_variables)
    # for gradient in generator_gradients:
    #     print("gradient", gradient)
    model_obj .optimizer_G .apply_gradients(zip(total_gradients, model_obj.generator.trainable_variables))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    for go_containor, loss_containor in enumerate(loss_info_objs[0].loss_containors.values()):
        loss_containor( losses[go_containor] )
    # loss_info_objs.loss_containors["mask_bce_loss"]      (gen_loss)
    # loss_info_objs.loss_containors["mask_sobel_MAE_loss"](sob_loss)

####################################################
@tf.function
def train_step_Single_output_I_w_Mgt_to_Cx(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    ### in_img.shape (1, h, w, 3)
    gt_mask = gt_data[..., 0:1]   ### (1, h, w, 1)
    gt_cx   = gt_data[..., 2:3]   ### (1, h, w, 1)， 注意 藥用 slice 取 才能保持 shape 喔！
    I_with_M = in_data * gt_mask
    # print("gt_cx.numpy().shape", gt_cx.numpy().shape)

    ### debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(I_with_M[0])
    # ax[2].imshow(gt_cx[0])
    # fig.tight_layout()
    # plt.show()

    _train_step_Single_output(model_obj=model_obj, in_data=I_with_M, gt_data=gt_cx, loss_info_objs=loss_info_objs)

@tf.function
def train_step_Single_output_I_w_Mgt_to_C(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    ### in_img.shape (1, h, w, 3)
    gt_mask  = gt_data[..., 0:1]  ### (1, h, w, 1)
    gt_coord = gt_data[..., 1:3]  ### (1, h, w, 2)
    I_with_M = in_data * gt_mask

    ### debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(I_with_M[0])
    # fig.tight_layout()
    # plt.show()

    _train_step_Single_output(model_obj=model_obj, in_data=I_with_M, gt_data=gt_coord, loss_info_objs=loss_info_objs)


####################################################
@tf.function
def train_step_Single_output_Mgt_to_C(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    Mgt_to_C 是 Mask(gt)_to_Coord 的縮寫
    '''
    gt_mask  = gt_data[..., 0:1]
    gt_coord = gt_data[..., 1:3]

    _train_step_Single_output(model_obj=model_obj, in_data=gt_mask, gt_data=gt_coord, loss_info_objs=loss_info_objs)

####################################################
@tf.function
def train_step_Single_output_I_to_C(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_to_C 是 Image_to_Coord 的縮寫
    '''
    gt_coord = gt_data[..., 1:3]

    _train_step_Single_output(model_obj=model_obj, in_data=in_data, gt_data=gt_coord, loss_info_objs=loss_info_objs)

@tf.function
def train_step_Single_output_I_to_W(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_to_W 是 Image_to_wc 的縮寫
    '''
    gt_wc = gt_data[..., :3]
    ### debug 時 記得把 @tf.function 拿掉
    # print("gt_data.shape", gt_data.shape)
    # print("in_data.shape", in_data.shape)
    # print("gt_wc.shape", gt_wc.shape)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(gt_wc[0, ..., 0:1])
    # ax[2].imshow(gt_wc[0, ..., 1:2])
    # ax[3].imshow(gt_wc[0, ..., 2:3])
    # fig.tight_layout()
    # plt.show()

    _train_step_Single_output(model_obj=model_obj, in_data=in_data, gt_data=gt_wc, loss_info_objs=loss_info_objs)

####################################################
@tf.function
def train_step_Single_output_I_to_M(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_to_C 是 Image_to_Coord 的縮寫
    '''
    gt_mask = gt_data[..., 0:1]

    _train_step_Single_output(model_obj=model_obj, in_data=in_data, gt_data=gt_mask, loss_info_objs=loss_info_objs)

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
@tf.function
def train_step_Single_output_I_w_Mgt_to_F(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    相當於無背景的訓練
    '''
    gt_mask = gt_data[..., 0:1]   ### (1, h, w, 1)
    I_with_M = in_data * gt_mask

    ### debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(I_with_M[0])
    # fig.tight_layout()
    # plt.show()
    _train_step_Single_output(model_obj, I_with_M, gt_data, loss_info_objs)

@tf.function
def train_step_Single_output_I_to_F_or_R(model_obj, in_data, gt_data, loss_info_objs=None):
    _train_step_Single_output(model_obj, in_data, gt_data, loss_info_objs)


@tf.function
def train_step_first(model_obj, in_dis_img, gt_coord_map, board_obj):
    with tf.GradientTape() as gen_tape:
        gen_output = model_obj.generator(in_dis_img, training=True)
        gen_l1_loss  = mae_kong(gen_output, gt_coord_map)

    generator_gradients     = gen_tape.gradient(gen_l1_loss, model_obj.generator.trainable_variables)
    model_obj.generator_optimizer.apply_gradients(zip(generator_gradients, model_obj.generator.trainable_variables))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    board_obj.losses["gen_l1_loss"](gen_l1_loss)

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################


@tf.function
# def train_step(rect2, in_data, gt_data, optimizer_G, optimizer_D, loss_info_objs ):
def train_step_GAN(model_obj, in_data, gt_data, loss_info_objs=None):
    with tf.GradientTape(persistent=True) as tape:
        g_g_data, fake_score, real_score = model_obj.rect(in_data, gt_data)
        loss_rec = loss_info_objs[0].loss_funs_dict["G"]     (g_g_data, gt_data, lamb=tf.constant(3., tf.float32))  ### 40 調回 3
        loss_g2d = loss_info_objs[0].loss_funs_dict["G_to_D"](fake_score, tf.ones_like(fake_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
        g_total_loss = loss_rec + loss_g2d

        loss_d_fake = loss_info_objs[0].loss_funs_dict["D_Fake"](fake_score, tf.zeros_like(fake_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
        loss_d_real = loss_info_objs[0].loss_funs_dict["D_Real"](real_score, tf.ones_like (real_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
        d_total_loss = (loss_d_real + loss_d_fake) / 2

    grad_D = tape.gradient(d_total_loss, model_obj.rect.discriminator.trainable_weights)
    grad_G = tape.gradient(g_total_loss, model_obj.rect.generator.    trainable_weights)
    model_obj.optimizer_D.apply_gradients(zip(grad_D, model_obj.rect.discriminator.trainable_weights))
    model_obj.optimizer_G.apply_gradients(zip(grad_G, model_obj.rect.generator.    trainable_weights))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    loss_info_objs[0].loss_containors["1_loss_rec"](loss_rec)
    loss_info_objs[0].loss_containors["2_loss_g2d"](loss_g2d)
    loss_info_objs[0].loss_containors["3_g_total_loss"](g_total_loss)
    loss_info_objs[0].loss_containors["4_loss_d_fake"](loss_d_fake)
    loss_info_objs[0].loss_containors["5_loss_d_real"](loss_d_real)
    loss_info_objs[0].loss_containors["6_d_total_loss"](d_total_loss)


@tf.function
def train_step_GAN2(model_obj, in_data, gt_data, loss_fun=None, loss_info_objs=None):
    for _ in range(1):
        with tf.GradientTape(persistent=True) as tape:
            g_g_data, fake_score, real_score = model_obj.rect(in_data, gt_data)
            loss_d_fake = loss_info_objs[0].loss_funs_dict["D_Fake"](fake_score, tf.zeros_like(fake_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
            loss_d_real = loss_info_objs[0].loss_funs_dict["D_Real"](real_score, tf.ones_like (real_score, dtype=tf.float32), lamb=tf.constant(1., tf.float32))
            d_total_loss = (loss_d_real + loss_d_fake) / 2
        grad_D = tape.gradient(d_total_loss, model_obj.rect.discriminator.trainable_weights)
        model_obj.optimizer_D.apply_gradients(zip(grad_D, model_obj.rect.discriminator.trainable_weights))

        loss_info_objs[0].loss_containors["4_loss_d_fake"](loss_d_fake)
        loss_info_objs[0].loss_containors["5_loss_d_real"](loss_d_real)
        loss_info_objs[0].loss_containors["6_d_total_loss"](d_total_loss)


    for _ in range(5):
        with tf.GradientTape(persistent=True) as g_tape:
            g_g_data, fake_score, real_score = model_obj.rect(in_data, gt_data)
            loss_rec = loss_info_objs[0].loss_funs_dict["G"](g_g_data, gt_data, lamb=tf.constant(3., tf.float32))  ### 40 調回 3
            loss_g2d = loss_info_objs[0].loss_funs_dict["G_to_D"](fake_score, tf.ones_like(fake_score, dtype=tf.float32), lamb=tf.constant(0.1, tf.float32))
            g_total_loss = loss_rec + loss_g2d
        grad_G = g_tape.gradient(g_total_loss, model_obj.rect.generator.    trainable_weights)
        model_obj.optimizer_G.apply_gradients(zip(grad_G, model_obj.rect.generator.    trainable_weights))
        ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
        loss_info_objs[0].loss_containors["1_loss_rec"](loss_rec)
        loss_info_objs[0].loss_containors["2_loss_g2d"](loss_g2d)
        loss_info_objs[0].loss_containors["3_g_total_loss"](g_total_loss)
