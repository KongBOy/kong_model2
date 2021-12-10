
import tensorflow as tf

class Multi_Generator(tf.keras.models.Model):
    def __init__(self, op_type, gens_dict, **kwargs):
        super(Multi_Generator, self).__init__(**kwargs)
        self.gens_dict = gens_dict
        self.op_type = op_type
        print("here~~")

    def call(self, input_tensor, training=None):
        if  (self.op_type == "I_to_M_and_C"):
            I = input_tensor
            M = self.gens_dict["I_to_M"](input_tensor)
            C = self.gens_dict["I_to_C"](input_tensor)
            return M, C

        elif(self.op_type == "I_to_M_w_I_to_C"):
            I = input_tensor
            M = self.gens_dict["I_to_M"](input_tensor)
            M_w_I = M * I
            C = self.gens_dict["M_w_I_to_C"](M_w_I)
            return M, C
        elif(self.op_type == "I_to_M_w_I_to_W_to_C"):
            I = input_tensor
            M = self.gens_dict["I_to_M"](input_tensor)
            M_w_I = M * I
            W = self.gens_dict["M_w_I_to_W"](M_w_I)
            C = self.gens_dict["W_to_C"](W)
            return M, W, C


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
    from step06_b_data_pipline import tf_Data_builder
    from step09_b_loss_info_obj import Loss_info_builder
    from step09_c_train_step import *


    from step09_f1_multi_unet2_obj_I_to_M_w_I_to_C import *

    model_obj = try_multi_unet
    model_obj = model_obj.build()  ### 可替換成 上面 想測試的 model
    print(model_obj)

    ### 2. db_obj 和 tf_data
    db_obj  = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1, train_shuffle=False).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    # ### 3. loss_info_obj
    G_mae_loss_infos = [Loss_info_builder().set_loss_type("mae").build(),
                        Loss_info_builder().set_loss_type("mae").build()]
    ### 4. 跑起來試試看
    for n, (train_in, train_in_pre, train_gt, train_gt_pre, _) in enumerate(tqdm(tf_data.train_db_combine)):
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=G_mae_loss_infos)
