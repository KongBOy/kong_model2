from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount
import tensorflow as tf
##########################################################################################################################################
debug_dict = {}
def debug_tf_data(tf_data, use_train_test_see="train"):
    ### 勿刪！用來測試寫得對不對！
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import os

    debug_what_db = None
    if  (use_train_test_see == "train"): debug_what_db = tf_data.train_db_combine
    elif(use_train_test_see == "test" ): debug_what_db = tf_data.test_db_combine
    elif(use_train_test_see == "see"  ): debug_what_db = tf.data.Dataset.zip((tf_data.see_in_db.ord.batch(1), tf_data.see_in_db.pre.batch(1),
                                                                              tf_data.see_gt_db.ord.batch(1), tf_data.see_gt_db.pre.batch(1),
                                                                              tf_data.see_name_db.ord))

    for i, (in_ord, in_pre, gt_ord, gt_pre, name) in enumerate(debug_what_db):  ### .take(5)):
        # if(i < 620): continue
        # print(i)
        debug_dict[f"{i}--1-1 in_ord"] = in_ord
        debug_dict[f"{i}--1-2 in_pre"] = in_pre
        debug_dict[f"{i}--1-3 gt_ord"] = gt_ord
        debug_dict[f"{i}--1-4 gt_pre"] = gt_pre

        debug_dict[f"{i}--2-1 in_ord[0]"  ] = in_ord[0][0].numpy()
        debug_dict[f"{i}--2-2 in_pre[0]"  ] = in_pre[0][0].numpy()
        debug_dict[f"{i}--2-3 in_Mgt"     ] = in_ord[0][0, ..., 3:4].numpy()
        debug_dict[f"{i}--2-4 in_Mgt_pre" ] = in_pre[0][0, ..., 3:4].numpy()
        debug_dict[f"{i}--2-5 in_W"       ] = in_ord[0][0, ..., 0:3].numpy()
        debug_dict[f"{i}--2-6 in_W_pre"   ] = in_pre[0][0, ..., 0:3].numpy()
        debug_dict[f"{i}--2-3 gt_ord[0]"  ] = gt_ord[0].numpy()
        debug_dict[f"{i}--2-4 gt_pre[0]"  ] = gt_pre[0].numpy()
        debug_dict[f"{i}--2-3 gt_Mgt"     ] = gt_ord[0, ..., 3:4].numpy()
        debug_dict[f"{i}--2-4 gt_Mgt_pre" ] = gt_pre[0, ..., 3:4].numpy()
        debug_dict[f"{i}--2-5 gt_W"       ] = gt_ord[0, ..., 0:3].numpy()
        debug_dict[f"{i}--2-6 gt_W_pre"   ] = gt_pre[0, ..., 0:3].numpy()

        canvas_base_size = 3
        nrows = 4
        ncols = 7
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(canvas_base_size * ncols, canvas_base_size * nrows)

        ### IN Wx, Wy, Wz, M 看一下ord長什麼樣子
        W_in_ord  = in_ord[0][0, ..., 0:3]
        Wx_in_ord = in_ord[0][0, ..., 0]
        Wy_in_ord = in_ord[0][0, ..., 1]
        Wz_in_ord = in_ord[0][0, ..., 2]
        M_in      = in_ord[0][0, ..., 3:4]
        I_in_ord  = in_ord[1][0]
        ax[0, 0].imshow(W_in_ord , vmin=tf_data.db_obj.db_in_range.min, vmax=tf_data.db_obj.db_in_range.max)
        ax[0, 1].imshow(Wx_in_ord, vmin=tf_data.db_obj.db_in_range.min, vmax=tf_data.db_obj.db_in_range.max)
        ax[0, 2].imshow(Wy_in_ord, vmin=tf_data.db_obj.db_in_range.min, vmax=tf_data.db_obj.db_in_range.max)
        ax[0, 3].imshow(Wz_in_ord, vmin=tf_data.db_obj.db_in_range.min, vmax=tf_data.db_obj.db_in_range.max)
        ax[0, 4].imshow(M_in     , vmin=tf_data.use_in_range.min, vmax=tf_data.use_in_range.max)
        ax[0, 5].imshow(I_in_ord)
        ### IN Wx, Wy, Wz, M 看一下pre長什麼樣子
        W_in_pre  = in_pre[0][0, ..., 0:3]
        Wx_in_pre = in_pre[0][0, ..., 0]
        Wy_in_pre = in_pre[0][0, ..., 1]
        Wz_in_pre = in_pre[0][0, ..., 2]
        M_in_pre  = in_pre[0][0, ..., 3:4]
        I_in_pre  = in_pre[1][0]
        ax[1, 0].imshow(W_in_pre , vmin=tf_data.use_in_range.min, vmax=tf_data.use_in_range.max)
        ax[1, 1].imshow(Wx_in_pre, vmin=tf_data.use_in_range.min, vmax=tf_data.use_in_range.max)
        ax[1, 2].imshow(Wy_in_pre, vmin=tf_data.use_in_range.min, vmax=tf_data.use_in_range.max)
        ax[1, 3].imshow(Wz_in_pre, vmin=tf_data.use_in_range.min, vmax=tf_data.use_in_range.max)
        ax[1, 4].imshow(M_in_pre  , vmin=tf_data.use_in_range.min, vmax=tf_data.use_in_range.max)
        ax[0, 6].imshow(I_in_pre)

        ### GT Wx, Wy, Wz, M 看一下ord長什麼樣子
        W_gt_ord  = gt_ord[0, ..., 0:3]
        Wx_gt_ord = gt_ord[0, ..., 0]
        Wy_gt_ord = gt_ord[0, ..., 1]
        Wz_gt_ord = gt_ord[0, ..., 2]
        Mgt       = gt_ord[0, ..., 3:4]
        ax[2, 0].imshow(W_gt_ord , vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ax[2, 1].imshow(Wx_gt_ord, vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ax[2, 2].imshow(Wy_gt_ord, vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ax[2, 3].imshow(Wz_gt_ord, vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ax[2, 4].imshow(Mgt      , vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ### GT Wx, Wy, Wz, M 看一下pre長什麼樣子
        W_gt_pre  = gt_pre[0, ..., 0:3]
        Wx_gt_pre = gt_pre[0, ..., 0]
        Wy_gt_pre = gt_pre[0, ..., 1]
        Wz_gt_pre = gt_pre[0, ..., 2]
        M_gt_pre   = gt_pre[0, ..., 3:4]
        ax[3, 0].imshow(W_gt_pre , vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[3, 1].imshow(Wx_gt_pre, vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[3, 2].imshow(Wy_gt_pre, vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[3, 3].imshow(Wz_gt_pre, vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[3, 4].imshow(M_gt_pre  , vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)


        ### W_pre * M
        W_in_pre_w_M_in_pre = W_in_pre * M_in_pre
        W_gt_pre_w_M_gt_pre = W_gt_pre * M_gt_pre
        ax[1, 5].imshow(W_in_pre_w_M_in_pre  , vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[3, 5].imshow(W_gt_pre_w_M_gt_pre  , vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)

        fig.tight_layout()

        ### 寫進 db 裡面的 check_dir
        check_dst_dir = None
        if  (use_train_test_see == "train"): check_dst_dir = f"{tf_data.db_obj.check_train_gt_dir}/{tf_data.db_obj.get_method.value}"
        elif(use_train_test_see == "test" ): check_dst_dir = f"{tf_data.db_obj.check_test_gt_dir}/{tf_data.db_obj.get_method.value}"
        elif(use_train_test_see == "see"  ): check_dst_dir = f"{tf_data.db_obj.check_see_gt_dir}/{tf_data.db_obj.get_method.value}"

        if(os.path.isdir(check_dst_dir) is False): os.makedirs(check_dst_dir)
        plt.savefig(f"{check_dst_dir}/" + "%05i" % (i + 1) ) 
        plt.show()
        plt.close()

##########################################################################################################################################
class tf_Data_in_W_gt_W_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_W_gt_W_builder, self).__init__(tf_data)

    def build_by_in_W_gt_W_hole_norm_then_mul_M_right(self):
        ##########################################################################################################################################
        ##### IN 部分
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_W_db_by_MW_hole_norm_then_mul_M_right()
        self.tf_data.test_name_db  = self.test_in_factory  .build_name_db()
        self.tf_data.test_in_db    = self.test_in_factory  .build_W_db_by_MW_hole_norm_then_mul_M_right()
        ''' 這裡的 train_in2_db 是 dis_img， 只是為了 visualize 而已， 不會丟進去model裡面 '''
        self.tf_data.train_in2_db     = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db      = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db.ord  = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in2_db.ord))
        self.tf_data.test_in_db .ord  = tf.data.Dataset.zip((self.tf_data.test_in_db.ord , self.tf_data.test_in2_db .ord))
        self.tf_data.train_in_db.pre  = tf.data.Dataset.zip((self.tf_data.train_in_db.pre, self.tf_data.train_in2_db.pre))
        self.tf_data.test_in_db .pre  = tf.data.Dataset.zip((self.tf_data.test_in_db.pre , self.tf_data.test_in2_db .pre))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ##### GT 部分
        self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_W_db_by_MW_hole_norm_then_mul_M_right()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_W_db_by_MW_hole_norm_then_mul_M_right()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in2_db = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db.ord  = tf.data.Dataset.zip((self.tf_data.see_in_db.ord, self.tf_data.see_in2_db.ord))
            self.tf_data.see_in_db.pre  = tf.data.Dataset.zip((self.tf_data.see_in_db.pre, self.tf_data.see_in2_db.pre))

            self.tf_data.see_gt_db   = self.see_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # debug_tf_data(self.tf_data, use_train_test_see="train")
        # debug_tf_data(self.tf_data, use_train_test_see="test")
        # debug_tf_data(self.tf_data, use_train_test_see="see")

        return self

    def build_by_in_W_gt_W_ch_norm_then_mul_M_right(self):
        ##########################################################################################################################################
        ##### IN 部分
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_W_db_by_MW_ch_norm_then_mul_M_right()
        self.tf_data.test_name_db  = self.test_in_factory  .build_name_db()
        self.tf_data.test_in_db    = self.test_in_factory  .build_W_db_by_MW_ch_norm_then_mul_M_right()
        ''' 這裡的 train_in2_db 是 dis_img， 只是為了 visualize 而已， 不會丟進去model裡面 '''
        self.tf_data.train_in2_db     = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db      = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db.ord  = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in2_db.ord))
        self.tf_data.test_in_db .ord  = tf.data.Dataset.zip((self.tf_data.test_in_db.ord , self.tf_data.test_in2_db .ord))
        self.tf_data.train_in_db.pre  = tf.data.Dataset.zip((self.tf_data.train_in_db.pre, self.tf_data.train_in2_db.pre))
        self.tf_data.test_in_db .pre  = tf.data.Dataset.zip((self.tf_data.test_in_db.pre , self.tf_data.test_in2_db .pre))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ##### GT 部分
        self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_ch_norm_then_mul_M_right()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_W_db_by_MW_ch_norm_then_mul_M_right()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_W_db_by_MW_ch_norm_then_mul_M_right()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in2_db = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db.ord  = tf.data.Dataset.zip((self.tf_data.see_in_db.ord, self.tf_data.see_in2_db.ord))
            self.tf_data.see_in_db.pre  = tf.data.Dataset.zip((self.tf_data.see_in_db.pre, self.tf_data.see_in2_db.pre))

            self.tf_data.see_gt_db   = self.see_gt_factory.build_W_db_by_MW_ch_norm_then_mul_M_right()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # debug_tf_data(self.tf_data, use_train_test_see="train")
        # debug_tf_data(self.tf_data, use_train_test_see="test")
        # debug_tf_data(self.tf_data, use_train_test_see="see")



if(__name__ == "__main__"):
    from step09_d_KModel_builder_combine_step789 import MODEL_NAME, KModel_builder
    from step06_a_datas_obj import *
    from step06_cFinal_tf_Data_builder import tf_Data_builder
    import time

    start_time = time.time()

    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    ''' kong_doc3d'''
    # db_obj = type8_blender_kong_doc3d_in_W_gt_W_ch_norm_v2.build()  ### 有 mul_M_right, ch_norm
    db_obj = type8_blender_in_W_gt_W_ch_norm_cylinder.build()  ### 有 mul_M_right, ch_norm
    print(db_obj)
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()
