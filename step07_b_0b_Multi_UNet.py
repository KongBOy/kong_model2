
import tensorflow as tf
import matplotlib.pyplot as plt

class Multi_Generator(tf.keras.models.Model):
    def __init__(self, op_type, gens_dict, **kwargs):
        super(Multi_Generator, self).__init__(**kwargs)
        self.gens_dict = gens_dict
        self.op_type = op_type

    def call(self, input_tensor, Mask=None, training=None):
        if  (self.op_type == "I_to_M_and_C"):  ### 最左邊的 I 是只 Model內本身的行為， 不會管 Model外 怎麼包喔， 意思就是 I 在 Model 外可以包成 I_w_M 也行， 反正 Model內都是唯一張img這樣子
            I = input_tensor
            M = self.gens_dict["I_to_M"](input_tensor)
            C = self.gens_dict["I_to_C"](input_tensor)
            return M, C

        elif(self.op_type == "I_to_M_w_I_to_C"):  ### 最左邊的 I 是只 Model內本身的行為， 不會管 Model外 怎麼包喔， 意思就是 I 在 Model 外可以包成 I_w_M 也行， 反正 Model內都是唯一張img這樣子
            I = input_tensor
            M = self.gens_dict["I_to_M"](input_tensor)
            M_w_I = M * I
            C = self.gens_dict["M_w_I_to_C"](M_w_I)
            return M, C
        elif(self.op_type == "I_to_M_w_I_to_W_to_C"):  ### 最左邊的 I 是只 Model內本身的行為， 不會管 Model外 怎麼包喔， 意思就是 I 在 Model 外可以包成 I_w_M 也行， 反正 Model內都是唯一張img這樣子
            I = input_tensor
            M = self.gens_dict["I_to_M"](input_tensor)
            M_w_I = M * I
            W = self.gens_dict["M_w_I_to_W"](M_w_I)
            C = self.gens_dict["W_to_C"](W)
            return M, W, C

        elif(self.op_type == "I_or_W_to_Cx_Cy"):  ### 最左邊的 I 是只 Model內本身的行為， 不會管 Model外 怎麼包喔， 意思就是 I 在 Model 外可以包成 I_w_M 也行， 反正 Model內都是唯一張img這樣子
            I = input_tensor
            Cx = self.gens_dict["I_to_Cx"](I)
            Cy = self.gens_dict["I_to_Cy"](I)
            return Cx, Cy   ### 這個順序要跟 step8b_useG, step9c_train_step 對應到喔！
        elif(self.op_type == "I_to_Wx_Wy_Wz"):  ### 最左邊的 I 是只 Model內本身的行為， 不會管 Model外 怎麼包喔， 意思就是 I 在 Model 外可以包成 I_w_M 也行， 反正 Model內都是唯一張img這樣子
            I = input_tensor
            Wx = self.gens_dict["I_to_Wx"](I)
            Wy = self.gens_dict["I_to_Wy"](I)
            Wz = self.gens_dict["I_to_Wz"](I)
            return Wz, Wy, Wx   ### 這個順序要跟 step8b_useG, step9c_train_step 對應到喔！
        elif(self.op_type == "I_to_Wx_Wy_Wz_focus_to_Cx_Cy_focus"):  ### 最左邊的 I 是只 Model內本身的行為， 不會管 Model外 怎麼包喔， 意思就是 I 在 Model 外可以包成 I_w_M 也行， 反正 Model內都是唯一張img這樣子
            '''
            注意 不能在這邊把 Wxyz concat 起來喔， 因為要分開算 loss！
            '''
            # I_pre = input_tensor
            # Wz_pre_raw, Wy_pre_raw, Wx_pre_raw = self.gens_dict["I_to_Wx_Wy_Wz"](I_pre)
            # W_pre_raw = tf.concat([Wz_pre_raw, Wy_pre_raw, Wx_pre_raw], axis=-1)
            # W_pre_w_M = W_pre_raw * Mask

            # Cx_pre_raw, Cy_pre_raw = self.gens_dict["W_to_Cx_Cy"](W_pre_w_M)
            # C_pre_raw = tf.concat([Cy_pre_raw, Cx_pre_raw], axis=-1)
            # C_pre_w_M = C_pre_raw * Mask
            # return W_pre_raw, W_pre_w_M, C_pre_raw, C_pre_w_M

            I_pre = input_tensor
            Wz_pre_raw, Wy_pre_raw, Wx_pre_raw = self.gens_dict["I_to_Wx_Wy_Wz"](I_pre)
            W_pre_raw = tf.concat([Wz_pre_raw, Wy_pre_raw, Wx_pre_raw], axis=-1)
            b, h, w, c = W_pre_raw.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
            W_pre_w_M = W_pre_raw * Mask[:, :h, :w, :]

            Cx_pre_raw, Cy_pre_raw = self.gens_dict["W_to_Cx_Cy"](W_pre_w_M)

            return Wz_pre_raw, Wy_pre_raw, Wx_pre_raw, Cx_pre_raw, Cy_pre_raw

def see(model_obj, train_in_pre):
    M_pre, C_pre = model_obj.generator(train_in_pre)

    M_visual = (M_pre[0].numpy() * 255.).astype(np.uint8)

    from step08_b_use_G_generate_0_util import F_01_or_C_01_method1_visual_op, Value_Range_Postprocess_to_01
    C = Value_Range_Postprocess_to_01(C_pre[0])
    C_visual = F_01_or_C_01_method1_visual_op(C)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(M_visual)
    ax[1].imshow(C_visual)
    plt.show()


if(__name__ == "__main__"):
    import numpy as np
    import time
    from kong_util.tf_model_util import Show_model_layer_names, Show_model_weights
    # data = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)
    # start_time = time.time()  # 看資料跑一次花多少時間
    # # test_g = Generator(hid_ch=64, depth_level=7, use_bias=False)
    # test_g = Generator(hid_ch= 128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14)
    # test_g(data)
    # print("cost time", time.time() - start_time)
    # test_g.summary()
    # print(test_g(data))



    ############################################################################################################################
    ### 嘗試 真的 load tf_data 進來 train 看看
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import *
    from step06_d_tf_Data_builder import tf_Data_builder
    from step10_a2_loss_info_obj import Loss_info_builder
    from step09_c_train_step import *


    from step09_f1_multi_unet2_obj_I_to_M_w_I_to_C import *

    # model_obj = try_multi_unet
    model_obj = I_to_M_L4_ch032_and_M_w_I_to_C_L5_ch032
    model_obj = model_obj.build()  ### 可替換成 上面 想測試的 model
    print(model_obj)

    ### 2. db_obj 和 tf_data
    db_obj  = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1, train_shuffle=False).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_img_resize(( 512, 512) ).build_by_db_get_method().build()

    # ### 3. loss_info_obj
    G_mae_loss_infos = [Loss_info_builder().set_loss_type("mae").build(),
                        Loss_info_builder().set_loss_type("mae").build()]
    ### 4. 跑起來試試看
    for n, (train_in, train_in_pre, train_gt, train_gt_pre, _) in enumerate(tqdm(tf_data.train_db_combine.take(50))):
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=G_mae_loss_infos)
        if(n ==  0):
            model_obj.generator.summary()
            Show_model_weights(model_obj.generator)

            see(model_obj, train_in_pre)

        if(n == 2):
            print(model_obj.generator.gens_dict)
            ckpt_I_to_M     = tf.train.Checkpoint(generator=model_obj.generator.gens_dict["I_to_M"])
            ckpt_M_w_I_to_C = tf.train.Checkpoint(generator=model_obj.generator.gens_dict["M_w_I_to_C"])

            ckpt_path_I_to_M     = "F:/kong_model2/data_dir/result/6_mask_unet/5_2_bce_block1_45678l/type8_blender-2_4l_ch032-flow_unet2-block1_ch032_sig_bce_s001_4l_ep060_copy-20211204_203747/ckpt"
            ckpt_path_I_w_M_to_C = "F:/kong_model2/data_dir/result/7_flow_unet/5_2_mae_block1_45678l_I_with_Mgt_to_C/type8_blender_os_book-2_L5_ch032-flow_unet2-block1_L5_ch032_mae_s001-20211125_170346/ckpt"

            ckpt_read_manager_I_to_M     = tf.train.CheckpointManager(ckpt_I_to_M, ckpt_path_I_to_M, max_to_keep=1)
            ckpt_read_manager_M_w_I_to_C = tf.train.CheckpointManager(ckpt_M_w_I_to_C, ckpt_path_I_w_M_to_C, max_to_keep=1)

            ckpt_I_to_M.    restore(ckpt_read_manager_I_to_M.latest_checkpoint)
            ckpt_M_w_I_to_C.restore(ckpt_read_manager_M_w_I_to_C.latest_checkpoint)
            print("ckpt_read_manager_I_to_M.latest_checkpoint:", ckpt_read_manager_I_to_M.latest_checkpoint)

            see(model_obj, train_in_pre)


        if(n == 10):
            model_obj.generator.save_weights("debug_data/try_save/weights")
            iter10 = model_obj.generator.layers[0].weights[1]
            print("iter10:", iter10)
        if(n == 20):
            iter20 = model_obj.generator.layers[0].weights[1]
            print("iter20:", iter20)
            model_obj.generator.load_weights("debug_data/try_save/weights")
            iter20_load10 = model_obj.generator.layers[0].weights[1]
            print("iter20_load10:", iter20_load10)
