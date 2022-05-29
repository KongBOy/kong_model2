from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount
import tensorflow as tf

debug_dict = {}

class tf_Data_in_dis_gt_wc_flow_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_dis_gt_wc_flow_builder, self).__init__(tf_data)

    def build_by_in_I_gt_W_and_F_hole_norm_then_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        ### train_in
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_img_db()
        ### test_in
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()
        self.tf_data.test_in_db    = self.test_in_factory.build_img_db()


        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount  = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount   = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### train_gt
        self.tf_data.train_gt_db  = self.train_gt_factory .build_W_db_by_MW_hole_norm_then_mul_M_right()
        self.tf_data.train_gt2_db = self.train_gt2_factory.build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK()
        self.tf_data.train_gt_db.ord = tf.data.Dataset.zip((self.tf_data.train_gt_db.ord, self.tf_data.train_gt2_db.ord))
        self.tf_data.train_gt_db.pre = tf.data.Dataset.zip((self.tf_data.train_gt_db.pre, self.tf_data.train_gt2_db.pre))

        ### test_gt
        self.tf_data.test_gt_db  = self.test_gt_factory .build_W_db_by_MW_hole_norm_then_mul_M_right()
        self.tf_data.test_gt2_db = self.test_gt2_factory.build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK()
        self.tf_data.test_gt_db.ord = tf.data.Dataset.zip((self.tf_data.test_gt_db.ord, self.tf_data.test_gt2_db.ord    ))
        self.tf_data.test_gt_db.pre = tf.data.Dataset.zip((self.tf_data.test_gt_db.pre, self.tf_data.test_gt2_db.pre))
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     ''' 注意這裡的train_in 有多 dis_img 喔！
        #            train_in[0] 是 wc,      shape=(N, H, W, C)
        #            train_in[1] 是 dis_img, shape=(N, H, W, C)
        #     '''
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"      ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"  ] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt_W"    ] = train_gt[0]      ### [0]是 取 wc, [1] 是取 flow
        #     debug_dict[f"{i}--1-4 train_gt_W_pre"] = train_gt_pre[0]  ### [0]是 取 wc, [1] 是取 flow
        #     debug_dict[f"{i}--1-3 train_gt_F"    ] = train_gt[1]      ### [0]是 取 wc, [1] 是取 flow
        #     debug_dict[f"{i}--1-4 train_gt_F_pre"] = train_gt_pre[1]  ### [0]是 取 wc, [1] 是取 flow

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in    [0]
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_Mgt"]     = train_gt    [0][0, ..., 3:4].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-3b train_Wgt"]     = train_gt    [0][0, ..., 0:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-4a train_Mgt_pre"] = train_gt_pre[0][0, ..., 3:4].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-4b train_Wgt_pre"] = train_gt_pre[0][0, ..., 0:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-5a train_Mgt"]     = train_gt    [1][0, ..., 0:1].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-5b train_Cgt"]     = train_gt    [1][0, ..., 1:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-6a train_Mgt_pre"] = train_gt_pre[1][0, ..., 0:1].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-6b train_Cgt_pre"] = train_gt_pre[1][0, ..., 1:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0]
        #     train_in_pre      = train_in_pre[0]

        #     train_Mgt_at_W      = train_gt    [0][0, ..., 3:4].numpy()
        #     train_Mgt_pre_at_W  = train_gt_pre[0][0, ..., 3:4].numpy()
        #     train_Wgt           = train_gt    [0][0, ..., 0:3].numpy()
        #     train_Wgt_pre       = train_gt_pre[0][0, ..., 0:3].numpy()

        #     train_Mgt      = train_gt    [1][0, ..., 0:1].numpy()
        #     train_Mgt_pre  = train_gt_pre[1][0, ..., 0:1].numpy()
        #     train_Cgt      = train_gt    [1][0, ..., 1:3].numpy()
        #     train_Cgt_pre  = train_gt_pre[1][0, ..., 1:3].numpy()

        #     train_Fgt_visual     = method1(train_Cgt[..., 1]    , train_Cgt[..., 0])
        #     train_Fgt_pre_visual = method1(train_Cgt_pre[..., 1], train_Cgt_pre[..., 0])

        #     fig, ax = plt.subplots(3, 4)
        #     fig.set_size_inches(20, 15)
        #     ax[0, 0].imshow(train_in)
        #     ax[0, 1].imshow(train_in_pre)

        #     ax[1, 0].imshow(train_Mgt_at_W)
        #     ax[1, 1].imshow(train_Mgt_pre_at_W)
        #     ax[1, 2].imshow(train_Wgt)
        #     ax[1, 3].imshow(train_Wgt_pre)

        #     ax[2, 0].imshow(train_Mgt)
        #     ax[2, 1].imshow(train_Mgt_pre)
        #     ax[2, 2].imshow(train_Fgt_visual)
        #     ax[2, 3].imshow(train_Fgt_pre_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            ### see_in
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()

            ### see_gt
            self.tf_data.see_gt_db  = self.see_gt_factory .build_W_db_by_MW_hole_norm_then_mul_M_right()
            self.tf_data.see_gt2_db = self.see_gt2_factory.build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK()
            self.tf_data.see_gt_db.ord = tf.data.Dataset.zip((self.tf_data.see_gt_db.ord, self.tf_data.see_gt2_db.ord))
            self.tf_data.see_gt_db.pre = tf.data.Dataset.zip((self.tf_data.see_gt_db.pre, self.tf_data.see_gt2_db.pre))

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.batch(1), self.tf_data.see_in_db_pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.batch(1), self.tf_data.see_gt_db_pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_Wgt"    ] = see_gt    [0]
            #     debug_dict[f"{i}--3-4 see_Wgt_pre"] = see_gt_pre[0]
            #     debug_dict[f"{i}--3-5 see_Fgt"    ] = see_gt    [1]
            #     debug_dict[f"{i}--3-6 see_Fgt_pre"] = see_gt_pre[1]

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_Mgt"]     = see_gt    [0][0, ..., 3:4].numpy()
            #     debug_dict[f"{i}--4-3b see_Wgt"]     = see_gt    [0][0, ..., 0:3].numpy()
            #     debug_dict[f"{i}--4-4a see_Mgt_pre"] = see_gt_pre[0][0, ..., 3:4].numpy()
            #     debug_dict[f"{i}--4-4b see_Wgt_pre"] = see_gt_pre[0][0, ..., 0:3].numpy()
            #     debug_dict[f"{i}--4-5a see_Mgt"]     = see_gt    [1][0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-5b see_Cgt"]     = see_gt    [1][0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-6a see_Mgt_pre"] = see_gt_pre[1][0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-6b see_Cgt_pre"] = see_gt_pre[1][0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

    def build_by_in_I_gt_W_and_F_ch_norm_then_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        ### train_in
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_img_db()
        ### test_in
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()
        self.tf_data.test_in_db    = self.test_in_factory.build_img_db()


        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount  = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount   = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### train_gt
        self.tf_data.train_gt_db  = self.train_gt_factory .build_W_db_by_MW_ch_norm_then_mul_M_right()
        self.tf_data.train_gt2_db = self.train_gt2_factory.build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK()
        self.tf_data.train_gt_db.ord = tf.data.Dataset.zip((self.tf_data.train_gt_db.ord, self.tf_data.train_gt2_db.ord))
        self.tf_data.train_gt_db.pre = tf.data.Dataset.zip((self.tf_data.train_gt_db.pre, self.tf_data.train_gt2_db.pre))

        ### test_gt
        self.tf_data.test_gt_db  = self.test_gt_factory .build_W_db_by_MW_ch_norm_then_mul_M_right()
        self.tf_data.test_gt2_db = self.test_gt2_factory.build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK()
        self.tf_data.test_gt_db.ord = tf.data.Dataset.zip((self.tf_data.test_gt_db.ord, self.tf_data.test_gt2_db.ord    ))
        self.tf_data.test_gt_db.pre = tf.data.Dataset.zip((self.tf_data.test_gt_db.pre, self.tf_data.test_gt2_db.pre))
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            ### see_in
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()

            ### see_gt
            self.tf_data.see_gt_db  = self.see_gt_factory .build_W_db_by_MW_ch_norm_then_mul_M_right()
            self.tf_data.see_gt2_db = self.see_gt2_factory.build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK()
            self.tf_data.see_gt_db.ord = tf.data.Dataset.zip((self.tf_data.see_gt_db.ord, self.tf_data.see_gt2_db.ord))
            self.tf_data.see_gt_db.pre = tf.data.Dataset.zip((self.tf_data.see_gt_db.pre, self.tf_data.see_gt2_db.pre))

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)
        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

if(__name__ == "__main__"):
    from step09_d_KModel_builder_combine_step789 import MODEL_NAME, KModel_builder
    from step06_a_datas_obj import *
    import time

    start_time = time.time()
    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()
