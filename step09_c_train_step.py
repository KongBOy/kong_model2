from distutils.log import debug
import sys
sys.path.append("kong_util")
import tensorflow as tf
import pdb
from step08_b_use_G_generate_0_util import Tight_crop

from step10_a1_loss import *
'''
會想把 train_step 獨立一個.py 寫 function， 還不包成 class 的原因是：
    因為 有些架構 用 的 train_step 是一樣的， 所以 先只寫成 function， 給各個架構掛上去
'''
debug_i = 1

class Ttrain_step_w_GAN:
    def __init__(self, op_type, BCE_use_mask=False, BCE_Mask_type="Area", D_train_amount=1, G_train_amount=1, just_train_D=False, D_train_many_diff=True, G_train_many_diff=True, DG_train_many_diff=True):
        self.op_type = op_type
        self.BCE_use_mask = BCE_use_mask
        self.BCE_Mask_type = BCE_Mask_type

        self.D_train_amount = D_train_amount
        self.G_train_amount = G_train_amount

        self.just_train_D = just_train_D
        self.D_train_many_diff = D_train_many_diff
        self.G_train_many_diff = G_train_many_diff
        self.DG_train_many_diff = DG_train_many_diff

        self.init_graph_finished = 0
        '''
        我只有一個 __call__ 可以使用，
        所以就把 train_D_G 都寫在一個method裡，
        用attribute來控制行為囉～
        '''

    # def train_step_Cxy_GAN(model_obj, in_data, gt_data, loss_info_objs=None):
    def multi_output_w_GAN_train(self, model_obj, in_data, gt_datas, loss_info_objs=None, Mask=None, D_training=True, G_training=False):
        BCE_mask = None
        if(self.BCE_use_mask): BCE_mask = Mask

        ### 訓練 Discriminato
        if(D_training):
            with tf.GradientTape() as D_tape:
                ### 生成 fake_data， 並丟入 D 取得 fake_score
                model_outputs_raw = model_obj.generator(in_data)           ### 舉例：model_outputs_raw == [Cx_pre_raw, Cy_pre_raw], in_data == W_w_M
                model_output_raw  = tf.concat(model_outputs_raw, axis=-1)  ### 舉例：model_output_raw  ==  C_pre_raw
                model_output_w_M  = model_output_raw * Mask                ### 舉例：model_output_w_M  ==  C_pre_w_M
                fake_score = model_obj.discriminator(model_output_w_M)
                ### 取用 real_data， 並丟入 D 取得 real_score
                gt_data    = tf.concat(gt_datas, axis=-1)                  ### 舉例：gt_data == Cgt_pre, gt_datas == [Cygt_pre, Cxgt_pre]
                real_score = model_obj.discriminator(gt_data)

                ### 訓練D： fake 越低分越好， real 越高分越好
                fake_score_gt0 = tf.zeros_like(fake_score, dtype=tf.float32)
                real_score_gt1 = tf.ones_like (real_score, dtype=tf.float32)

                BCE_posi = len(model_outputs_raw)  ### 舉例： 第一個放 Cx 的 loss_info_obj, 第二個放 Cy  的 loss_info_obj, 第三個 才放 GAN 的 loss
                BCE_D_fake = loss_info_objs[BCE_posi].loss_funs_dict["BCE_D_fake"](fake_score_gt0, fake_score, BCE_mask, Mask_type=self.BCE_Mask_type)
                BCE_D_real = loss_info_objs[BCE_posi].loss_funs_dict["BCE_D_real"](real_score_gt1, real_score, BCE_mask, Mask_type=self.BCE_Mask_type)
                D_total_loss = (BCE_D_fake + BCE_D_real) / 2
                D_total_loss *= self.init_graph_finished  ### 如果是在 init_graph的話， init_graph_finished == 0， 這樣就不會更新到 model 了喔！ 當 init_graph完成後， 在exp層級會幫這裡的 init_graph_finished 設 1， 之後的訓練時 loss值不會歸零 就會正常更新了～
                print("calculate D finish")

            grad_D = D_tape.gradient(D_total_loss, model_obj.discriminator.trainable_weights)
            model_obj.optimizer_D.apply_gradients(zip(grad_D, model_obj.discriminator.trainable_weights))

            loss_info_objs[2].loss_containors["BCE_D_fake"](BCE_D_fake)
            loss_info_objs[2].loss_containors["BCE_D_real"](BCE_D_real)

            ### 如果是在 init_graph的話， init_graph_finished == 0， 此時算出來的 loss值是被歸零的， 不用儲存， 但為了建立graph還是要有存的動作， 存完reset就好囉！ 當 init_graph完成後， 在exp層級會幫這裡的 init_graph_finished 設 1， 之後的訓練時 loss值不會歸零 就不用reset拉～
            if(self.init_graph_finished == 0):
                loss_info_objs[2].loss_containors["BCE_D_fake"].reset_states()
                loss_info_objs[2].loss_containors["BCE_D_real"].reset_states()
                print("reset D metric finish (init_graph dosen't need to save so reset it)")
            print("update D finish")


        ### 更新完D 後 訓練 Generator， 所以應該要重新丟一次資料進去G
        if(G_training and self.just_train_D is False):
            with tf.GradientTape() as G_tape:
                model_outputs_raw = model_obj.generator(in_data)  ### 舉例：model_outputs_raw == [Cx_pre_raw, Cy_pre_raw] (同上)
                multi_losses = []
                multi_total_loss = 0
                for go_m, model_output in enumerate(model_outputs_raw):  ### 舉例：model_outputs_raw == [Cx_pre_raw, Cy_pre_raw]， 第一次跑 Cx， 第二次跑 Cy
                    total_loss, losses = one_loss_info_obj_total_loss(loss_info_objs[go_m], model_output, gt_datas[go_m], Mask=Mask)
                    multi_losses.append(losses)
                    multi_total_loss += total_loss

                model_output_raw  = tf.concat(model_outputs_raw, axis=-1)  ### 舉例：model_output_raw  ==  C_pre_raw (同上)
                model_output_w_M  = model_output_raw * Mask                ### 舉例：model_output_w_M  ==  C_pre_w_M (同上)
                fake_score = model_obj.discriminator(model_output_w_M)
                fake_score_gt1 = tf.ones_like(fake_score, dtype=tf.float32)   ### 訓練G時， 希望騙過D， 所以希望越高分越好

                BCE_posi = len(model_outputs_raw)  ### 舉例： 第一個放 Cx 的 loss_info_obj, 第二個放 Cy  的 loss_info_obj, 第三個 才放 GAN 的 loss (同上)
                BCE_G_to_D = loss_info_objs[BCE_posi].loss_funs_dict["BCE_G_to_D"](fake_score_gt1, fake_score, BCE_mask, Mask_type=self.BCE_Mask_type)
                G_total_loss = BCE_G_to_D + multi_total_loss
                G_total_loss *= self.init_graph_finished  ### 如果是在 init_graph的話， init_graph_finished == 0， 這樣就不會更新到 model 了喔！ 當 init_graph完成後， 在exp層級會幫這裡的 init_graph_finished 設 1， 之後的訓練時 loss值不會歸零 就會正常更新了～
                # G_total_loss = multi_total_loss  ### debug 用， 看看 不加 GAN loss 效果如何
                print("calculate G finish")

            grad_G = G_tape.gradient(G_total_loss, model_obj.generator.trainable_weights)
            model_obj.optimizer_G.apply_gradients(zip(grad_G, model_obj.generator.trainable_weights))

            ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
            for go_m, _ in enumerate(model_outputs_raw):
                loss_info_obj = loss_info_objs[go_m]
                for go_containor, loss_containor in enumerate(loss_info_obj.loss_containors.values()):
                    loss_containor( multi_losses[go_m][go_containor] )
            loss_info_objs[2].loss_containors["BCE_G_to_D"](BCE_G_to_D)

            ### 如果是在 init_graph的話， init_graph_finished == 0， 此時算出來的 loss值是被歸零的， 不用儲存， 但為了建立graph還是要有存的動作， 存完reset就好囉！ 當 init_graph完成後， 在exp層級會幫這裡的 init_graph_finished 設 1， 之後的訓練時 loss值不會歸零 就不用reset拉～
            if(self.init_graph_finished == 0):
                for go_m, _ in enumerate(model_outputs_raw):
                    loss_info_obj = loss_info_objs[go_m]
                    for go_containor, loss_containor in enumerate(loss_info_obj.loss_containors.values()):
                        if(self.init_graph_finished == 0): loss_containor.reset_states()  ### 如果是在 init_graph的話， init_graph_finished == 0， 此時算出來的 loss值是被歸零的， 不用儲存， 但為了建立graph還是要有存的動作， 存完reset就好囉！ 當 init_graph完成後， 在exp層級會幫這裡的 init_graph_finished 設 1， 之後的訓練時 loss值不會歸零 就不用reset拉～
                loss_info_objs[2].loss_containors["BCE_G_to_D"].reset_states()
                print("reset G metric finish (init_graph dosen't need to save so reset it)")
            print("update G finish")


    @tf.function
    def __call__(self, model_obj, in_data, gt_data, loss_info_objs=None, D_training=True, G_training=False):
        if(self.op_type == "W_w_Mgt_to_Cx_Cy_focus"):
            in_Mask  = in_data[..., 3:4]
            in_W     = in_data[..., 0:3]
            W_w_M = in_W * in_Mask

            Cxgt_pre = gt_data[..., 2:3]
            Cygt_pre = gt_data[..., 1:2]
            self.multi_output_w_GAN_train(model_obj,
                                         in_data=W_w_M,
                                         gt_datas=[Cxgt_pre, Cygt_pre],  ### 沒辦法當初設定成這樣子train， 就只能繼續保持這樣子了，要不然以前train好的東西 不能繼續用下去 QQ
                                         loss_info_objs=loss_info_objs,  ### 第一個放 Cx 的 loss_info_obj, 第二個放 Cy  的 loss_info_obj, 第三個 才放 GAN 的 loss
                                         Mask=in_Mask,
                                         D_training=D_training,
                                         G_training=G_training)
            # import matplotlib.pyplot as plt
            # import tensorflow as tf
            # bce_mask = in_Mask
            # # bce_mask = tf.image.resize(bce_mask, size=(3, 3))
            # # bce_mask = bce_mask * tf.constant( [[[ 0 ], [ 0 ], [ 0 ]],
            # #                                     [[ 0 ], [ 1 ], [ 0 ]],
            # #                                     [[ 0 ], [ 0 ], [ 0 ]]], dtype=tf.float32)
            # kernel = tf.ones((3, 3, 1))
            # bce_mask = tf.image.resize(bce_mask, size=(7, 7))
            # bce_mask = tf.nn.erosion2d(bce_mask, filters=kernel, strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", dilations=(1, 1, 1, 1)) + 1
            # plt.figure()
            # plt.imshow(bce_mask[0], vmin=0, vmax=1)
            # global debug_i
            # plt.savefig("debug_data/try_Mask_erosion/%03i" % debug_i)
            # debug_i += 1
            # plt.close()
            # # plt.show()

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
def one_loss_info_obj_total_loss(loss_info_objs, model_output, gt_data, Mask=None):
    losses = []
    total_loss = 0
    for loss_name, loss_fun in loss_info_objs.loss_funs_dict.items():
        # print("loss_name:", loss_name)
        if  ("tv"  in loss_name): losses.append(loss_fun(model_output))
        elif("bce" in loss_name): losses.append(loss_fun(gt_data, model_output))
        else:                     losses.append(loss_fun(gt_data, model_output, Mask))
        total_loss += losses[-1]
    return total_loss, losses
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################


def _train_step_Multi_output(model_obj, in_data, gt_datas, loss_info_objs=None, Mask=None):
    with tf.GradientTape() as gen_tape:
        model_outputs = model_obj.generator(in_data)
        # print("in_data.numpy().shape", in_data.numpy().shape)
        # print("model_output.min()", model_output.numpy().min())  ### 用這show的時候要先把 @tf.function註解掉
        # print("model_output.max()", model_output.numpy().max())  ### 用這show的時候要先把 @tf.function註解掉
        multi_losses = []
        multi_total_loss = 0
        for go_m, model_output in enumerate(model_outputs):
            total_loss, losses = one_loss_info_obj_total_loss(loss_info_objs[go_m], model_output, gt_datas[go_m], Mask=Mask)
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
    for go_m, _ in enumerate(model_outputs):
        loss_info_obj = loss_info_objs[go_m]
        for go_containor, loss_containor in enumerate(loss_info_obj.loss_containors.values()):
            loss_containor( multi_losses[go_m][go_containor] )
    # loss_info_objs.loss_containors["mask_bce_loss"]      (gen_loss)
    # loss_info_objs.loss_containors["mask_sobel_MAE_loss"](sob_loss)
####################################################
@tf.function
def train_step_Multi_output_I_w_M_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus(model_obj, in_data, gt_data, loss_info_objs=None):
    I_pre   = in_data
    Wgt     = gt_data[0]
    Fgt     = gt_data[1]
    Mgt_pre = gt_data[0][..., 3:4]  ### 配合抓DB的方式用 in_dis_gt_wc_flow 的話：第一個 [0]是W， [1]是F， [0][..., 3:4] 或 [1][..., 0:1] 都可以取道Mask

    I_pre_w_M_pre = I_pre * Mgt_pre

    Wzgt = Wgt[..., 0:1]
    Wygt = Wgt[..., 1:2]
    Wxgt = Wgt[..., 2:3]
    Cxgt = Fgt[..., 2:3]
    Cygt = Fgt[..., 1:2]
    gt_datas = [Wzgt, Wygt, Wxgt, Cxgt, Cygt]

    ## debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(40, 5))
    # ax[0].imshow(I_pre[0])
    # ax[1].imshow(Mgt_pre[0])
    # ax[2].imshow(I_pre_w_M_pre[0])
    # ax[3].imshow(Wzgt[0])
    # ax[4].imshow(Wygt[0])
    # ax[5].imshow(Wxgt[0])
    # ax[6].imshow(Cxgt[0])
    # ax[7].imshow(Cygt[0])
    # fig.tight_layout()
    # plt.show()

    _train_step_Multi_output(model_obj, in_data=I_pre_w_M_pre, gt_datas=gt_datas, loss_info_objs=loss_info_objs, Mask=Mgt_pre)

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
def train_step_Multi_output_I_w_M_to_Cx_Cy_focus(model_obj, in_data, gt_data, loss_info_objs=None):
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

    _train_step_Multi_output(model_obj, in_data=I_with_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs, Mask=gt_mask)

####################################################
####################################################
class Train_step_W_w_M_to_Cx_Cy():
    def __init__(self, separate_out=False, focus=False, tight_crop=None):
        self.separate_out   = separate_out
        self.focus      = focus
        self.tight_crop = tight_crop

    @tf.function
    def __call__(self, model_obj, in_data, gt_data, loss_info_objs=None):
        '''
        I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
        '''
        Mgt_pre_for_crop = in_data[0][..., 3:4]
        W_con_M = in_data[0]
        if(self.tight_crop is not None):
            self.tight_crop.reset_jit()
            W_con_M, _ = self.tight_crop(W_con_M, Mgt_pre_for_crop)
            gt_data, _ = self.tight_crop(gt_data, Mgt_pre_for_crop)

        in_Mask  = W_con_M[..., 3:4]
        in_W     = W_con_M[..., 0:3]
        W_w_M = in_W * in_Mask

        gt_c  = gt_data[..., 1:3]
        gt_cx = gt_data[..., 2:3]
        gt_cy = gt_data[..., 1:2]
        gt_datas = [gt_cx, gt_cy]  ### 沒辦法當初設定成這樣子train， 就只能繼續保持這樣子了，要不然以前train好的東西 不能繼續用下去 QQ
        # print("gt_cx.numpy().shape", gt_cx.numpy().shape)
        # print("gt_cy.numpy().shape", gt_cy.numpy().shape)

        ### debug 時 記得把 @tf.function 拿掉
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        # ax[0].imshow(in_data[1][0])
        # ax[1].imshow(W_w_M[0])
        # ax[2].imshow(gt_cx[0])
        # ax[3].imshow(gt_cy[0])
        # fig.tight_layout()
        # plt.show()

        if(self.separate_out is False):
            if(self.focus is False): _train_step_Single_output(model_obj, in_data=W_w_M, gt_data =gt_c    , loss_info_objs=loss_info_objs)
            else:                    _train_step_Single_output(model_obj, in_data=W_w_M, gt_data =gt_c    , loss_info_objs=loss_info_objs, Mask=in_Mask)
        else:
            if(self.focus is False): _train_step_Multi_output (model_obj, in_data=W_w_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs)
            else:                    _train_step_Multi_output (model_obj, in_data=W_w_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs, Mask=in_Mask)


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
def train_step_Multi_output_W_w_M_to_Cx_Cy_focus(model_obj, in_data, gt_data, loss_info_objs=None):
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

    _train_step_Multi_output(model_obj, in_data=W_w_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs, Mask=in_Mask)

####################################################
####################################################
class Train_step_I_w_M_to_W():
    def __init__(self, separate_out=False, focus=False, tight_crop=None):
        self.separate_out  = separate_out
        self.focus        = focus
        self.tight_crop   = tight_crop

    @tf.function
    def __call__(self, model_obj, in_data, gt_data, loss_info_objs=None):
        '''
        I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
        '''
        Mgt_pre_for_crop  = gt_data[..., 3:4]
        if(self.tight_crop is not None):
            self.tight_crop.reset_jit()
            in_data, _ = self.tight_crop(in_data, Mgt_pre_for_crop)
            gt_data, _ = self.tight_crop(gt_data, Mgt_pre_for_crop)

        gt_mask  = gt_data[..., 3:4]
        I_with_M = in_data * gt_mask

        Wgt  = gt_data[..., 0:3]
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
        # ax[0, 0].imshow(in_data[0]   , vmin=0, vmax=1)
        # ax[0, 1].imshow(gt_mask[0]   , vmin=0, vmax=1)
        # ax[0, 2].imshow(I_with_M[0]  , vmin=0, vmax=1)
        # ax[1, 0].imshow(Wxgt[0]      , vmin=0, vmax=1)
        # ax[1, 1].imshow(Wygt[0]      , vmin=0, vmax=1)
        # ax[1, 2].imshow(Wzgt[0]      , vmin=0, vmax=1)
        # fig.tight_layout()
        # plt.show()

        if(self.separate_out is False):
            if(self.focus is False): _train_step_Single_output(model_obj, in_data=I_with_M, gt_data=Wgt     , loss_info_objs=loss_info_objs)
            else:                    _train_step_Single_output(model_obj, in_data=I_with_M, gt_data=Wgt     , loss_info_objs=loss_info_objs, Mask=gt_mask)
        else:
            if(self.focus is False): _train_step_Multi_output(model_obj, in_data=I_with_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs)
            else:                    _train_step_Multi_output(model_obj, in_data=I_with_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs, Mask=gt_mask)

        
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

@tf.function
def train_step_Multi_output_I_w_Mgt_to_Wx_Wy_Wz_focus(model_obj, in_data, gt_data, loss_info_objs=None):
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

    _train_step_Multi_output(model_obj, in_data=I_with_M, gt_datas=gt_datas, loss_info_objs=loss_info_objs, Mask=gt_mask)


###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
### 因為外層function 已經有 @tf.function， 裡面這層自動會被 decorate 到喔！ 所以這裡不用 @tf.function
def _train_step_Single_output(model_obj, in_data, gt_data, loss_info_objs, Mask=None):
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
        total_loss, losses = one_loss_info_obj_total_loss(loss_info_objs[0], model_output, gt_data, Mask=Mask)
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
def train_step_Single_output_I_w_Mgt_to_Cx_focus(model_obj, in_data, gt_data, loss_info_objs=None):
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
    _train_step_Single_output(model_obj=model_obj, in_data=I_with_M, gt_data=gt_cx, loss_info_objs=loss_info_objs, Mask=gt_mask)

@tf.function
def train_step_Single_output_I_w_Mgt_to_Cy(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    ### in_img.shape (1, h, w, 3)
    gt_mask = gt_data[..., 0:1]   ### (1, h, w, 1)
    gt_cy   = gt_data[..., 1:2]   ### (1, h, w, 1)， 注意 藥用 slice 取 才能保持 shape 喔！
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
    _train_step_Single_output(model_obj=model_obj, in_data=I_with_M, gt_data=gt_cy, loss_info_objs=loss_info_objs)

@tf.function
def train_step_Single_output_I_w_Mgt_to_Cy_focus(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_with_Mgt_to_C 是 Image_with_Mask(gt)_to_Coord 的縮寫
    '''
    ### in_img.shape (1, h, w, 3)
    gt_mask = gt_data[..., 0:1]   ### (1, h, w, 1)
    gt_cy   = gt_data[..., 1:2]   ### (1, h, w, 1)， 注意 藥用 slice 取 才能保持 shape 喔！
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
    _train_step_Single_output(model_obj=model_obj, in_data=I_with_M, gt_data=gt_cy, loss_info_objs=loss_info_objs, Mask=gt_mask)

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
########################################################################################################
########################################################################################################
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
####################################################
class Train_step_I_to_M():
    def __init__(self, tight_crop=None):
        self.tight_crop = tight_crop

    @tf.function
    def __call__(self, model_obj, in_data, gt_data, loss_info_objs=None):
        gt_mask = gt_data[..., 0:1]
        if(self.tight_crop is not None):
            self.tight_crop.reset_jit()
            in_data, _ = self.tight_crop(in_data, gt_mask)
            gt_mask, _ = self.tight_crop(gt_mask, gt_mask)

        ### debug 時 記得把 @tf.function 拿掉
        # global debug_i
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # ax[0].imshow(in_data[0])
        # ax[1].imshow(gt_mask[0])
        # fig.tight_layout()
        # # plt.show()
        # plt.savefig("debug_data/try_tight_crop/%03i" % debug_i)
        # plt.close()
        # debug_i += 1

        _train_step_Single_output(model_obj=model_obj, in_data=in_data, gt_data=gt_mask, loss_info_objs=loss_info_objs)

@tf.function
def train_step_Single_output_I_to_M(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_to_C 是 Image_to_Coord 的縮寫
    '''
    gt_mask = gt_data[..., 0:1]

    _train_step_Single_output(model_obj=model_obj, in_data=in_data, gt_data=gt_mask, loss_info_objs=loss_info_objs)


@tf.function
def train_step_Single_output_I_to_M_tight_crop(model_obj, in_data, gt_data, loss_info_objs=None):
    '''
    I_to_C 是 Image_to_Coord 的縮寫
    '''
    gt_mask = gt_data[..., 0:1]
    tight_crop = Tight_crop(pad_size=20, resize=(256, 256))
    in_data = tight_crop(in_data, gt_mask)
    gt_mask = tight_crop(gt_mask, gt_mask)

    ### debug 時 記得把 @tf.function 拿掉
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # ax[0].imshow(in_data[0])
    # ax[1].imshow(gt_mask[0])
    # # fig.tight_layout()
    # plt.show()
    # plt.close()

    _train_step_Single_output(model_obj=model_obj, in_data=in_data, gt_data=gt_mask, loss_info_objs=loss_info_objs)

###################################################################################################################################################
###################################################################################################################################################
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
