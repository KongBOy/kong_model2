from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount

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

        debug_dict[f"{i}--2-1 in_ord[0]"  ] = in_ord[0].numpy()
        debug_dict[f"{i}--2-2 in_pre[0]"  ] = in_pre[0].numpy()
        debug_dict[f"{i}--2-3 gt_ord[0]"  ] = gt_ord[0].numpy()
        debug_dict[f"{i}--2-4 gt_pre[0]"  ] = gt_pre[0].numpy()
        debug_dict[f"{i}--2-3 gt_Mgt"     ] = gt_ord[0, ..., 3:4].numpy()
        debug_dict[f"{i}--2-4 gt_Mgt_pre" ] = gt_pre[0, ..., 3:4].numpy()
        debug_dict[f"{i}--2-5 gt_W"       ] = gt_ord[0, ..., 0:3].numpy()
        debug_dict[f"{i}--2-6 gt_W_pre"   ] = gt_pre[0, ..., 0:3].numpy()

        canvas_base_size = 3
        nrows = 3
        ncols = 5
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(canvas_base_size * ncols, canvas_base_size * nrows)
        ### ord vs pre
        in_ord = in_ord[0]
        in_pre = in_pre[0]
        ax[0, 0].imshow(in_ord, vmin=0, vmax=255)
        ax[0, 1].imshow(in_pre, vmin=0, vmax=1)

        ### W_ord vs W_pre
        W_ord = gt_ord[0, ..., :3]
        W_pre = gt_pre[0, ..., :3]
        ax[0, 2].imshow(W_ord, vmin=0, vmax=1)
        ax[0, 3].imshow(W_pre, vmin=0, vmax=1)

        ### Wx, Wy, Wz 看一下ord長什麼樣子
        Wx_ord = gt_ord[0, ..., 0]
        Wy_ord = gt_ord[0, ..., 1]
        Wz_ord = gt_ord[0, ..., 2]
        ax[1, 0].imshow(Wx_ord, vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ax[1, 1].imshow(Wy_ord, vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ax[1, 2].imshow(Wz_ord, vmin=tf_data.db_obj.db_gt_range.min, vmax=tf_data.db_obj.db_gt_range.max)
        ### Wx, Wy, Wz 看一下pre長什麼樣子
        Wx_pre = gt_pre[0, ..., 0]
        Wy_pre = gt_pre[0, ..., 1]
        Wz_pre = gt_pre[0, ..., 2]
        ax[2, 0].imshow(Wx_pre, vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[2, 1].imshow(Wy_pre, vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)
        ax[2, 2].imshow(Wz_pre, vmin=tf_data.use_gt_range.min, vmax=tf_data.use_gt_range.max)

        ### M_ord vs M_pre
        Mgt     = gt_ord[0, ..., 3:4]
        Mgt_pre = gt_pre[0, ..., 3:4]
        ax[2, 3].imshow(Mgt     , vmin=0, vmax=1)
        ax[2, 4].imshow(Mgt_pre , vmin=0, vmax=1)

        ### W_pre * M
        W_pre_w_Mgt_pre = W_pre * Mgt_pre
        ax[0, 4].imshow(W_pre_w_Mgt_pre)

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
class tf_Data_in_dis_gt_flow_or_wc_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_dis_gt_flow_or_wc_builder, self).__init__(tf_data)

    def build_by_in_I_gt_F_or_W_hole_norm_then_no_mul_M_wrong(self, get_what="flow"):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        if  (get_what == "flow"): self.tf_data.train_gt_db = self.train_gt_factory.build_flow_db()
        elif(get_what == "wc"  ): self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_hole_norm_no_mul_M_wrong()

        if  (get_what == "flow"): self.tf_data.test_gt_db = self.test_gt_factory.build_flow_db()
        elif(get_what == "wc"):   self.tf_data.test_gt_db = self.test_gt_factory.build_W_db_by_MW_hole_norm_no_mul_M_wrong()


        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        # print('self.tf_data.train_in_db',     self.tf_data.train_in_db.ord)
        # print('self.tf_data.train_in_db_pre', self.tf_data.train_in_db.pre)
        # print('self.tf_data.train_gt_db',     self.tf_data.train_gt_db.ord)
        # print('self.tf_data.train_gt_db_pre', self.tf_data.train_gt_db.pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            if  (get_what == "flow"): self.tf_data.see_gt_db = self.see_gt_factory.build_flow_db()
            elif(get_what == "wc"):   self.tf_data.see_gt_db = self.see_gt_factory.build_W_db_by_MW_hole_norm_no_mul_M_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)


        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

        if(self.tf_data.db_obj.have_DewarpNet_result):
            self.tf_data.DewarpNet_result_test = self.DewarpNet_result_test_factory.build_img_db()
            self.tf_data.DewarpNet_result_see  = self.DewarpNet_result_see_factory .build_img_db()

            self.tf_data.DewarpNet_result_test_amount = get_db_amount(self.tf_data.db_obj.DewarpNet_result_test_dir)
            self.tf_data.DewarpNet_result_see_amount  = get_db_amount(self.tf_data.db_obj.DewarpNet_result_see_dir)
            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # debug_tf_data(self.tf_data, use_train_test_see="train")
        # debug_tf_data(self.tf_data, use_train_test_see="test")
        # debug_tf_data(self.tf_data, use_train_test_see="see")

        return self

    def build_by_in_I_gt_W_hole_norm_then_mul_M_right(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()
        # self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_hole_norm_no_mul_M_wrong()  ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        # self.tf_data.test_gt_db  = self.test_gt_factory.build_W_db_by_MW_hole_norm_no_mul_M_wrong()   ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_W_db_by_MW_hole_norm_then_mul_M_right()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        # print('self.tf_data.train_in_db',     self.tf_data.train_in_db.ord)
        # print('self.tf_data.train_in_db_pre', self.tf_data.train_in_db.pre)
        # print('self.tf_data.train_gt_db',     self.tf_data.train_gt_db.ord)
        # print('self.tf_data.train_gt_db_pre', self.tf_data.train_gt_db.pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

        if(self.tf_data.db_obj.have_DewarpNet_result):
            self.tf_data.DewarpNet_result_test = self.DewarpNet_result_test_factory.build_img_db()
            self.tf_data.DewarpNet_result_see  = self.DewarpNet_result_see_factory .build_img_db()

            self.tf_data.DewarpNet_result_test_amount = get_db_amount(self.tf_data.db_obj.DewarpNet_result_test_dir)
            self.tf_data.DewarpNet_result_see_amount  = get_db_amount(self.tf_data.db_obj.DewarpNet_result_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            #####################################################################################################################################

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # debug_tf_data(self.tf_data, use_train_test_see="train")
        # debug_tf_data(self.tf_data, use_train_test_see="test")
        # debug_tf_data(self.tf_data, use_train_test_see="see")

        return self

    def build_by_in_I_gt_W_ch_norm_then_mul_M_right(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()
        self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_ch_norm_then_mul_M_right()
        self.tf_data.test_gt_db  = self.test_gt_factory.build_W_db_by_MW_ch_norm_then_mul_M_right()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_W_db_by_MW_ch_norm_then_mul_M_right()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

        if(self.tf_data.db_obj.have_DewarpNet_result):
            self.tf_data.DewarpNet_result_test = self.DewarpNet_result_test_factory.build_img_db()
            self.tf_data.DewarpNet_result_see  = self.DewarpNet_result_see_factory .build_img_db()

            self.tf_data.DewarpNet_result_test_amount = get_db_amount(self.tf_data.db_obj.DewarpNet_result_test_dir)
            self.tf_data.DewarpNet_result_see_amount  = get_db_amount(self.tf_data.db_obj.DewarpNet_result_see_dir)
            # print("self.tf_data.DewarpNet_result_test_amount", self.tf_data.DewarpNet_result_test_amount)
            # print("self.tf_data.DewarpNet_result_see_amount", self.tf_data.DewarpNet_result_see_amount)
        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # debug_tf_data(self.tf_data, use_train_test_see="train")
        # debug_tf_data(self.tf_data, use_train_test_see="test")
        # debug_tf_data(self.tf_data, use_train_test_see="see")

    ### Kong_Doc3D V1 才用這個， 升級到 V2就不用了喔， 因為在DB方面就已經把x軸反轉了
    def build_by_in_I_gt_W_ch_norm_then_mul_M_right_only_for_doc3d_x_value_reverse(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()
        self.tf_data.train_gt_db = self.train_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right_only_for_doc3d_x_value_reverse()
        self.tf_data.test_gt_db  = self.test_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right_only_for_doc3d_x_value_reverse()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_W_db_by_MW_hole_norm_then_mul_M_right_only_for_doc3d_x_value_reverse()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

        if(self.tf_data.db_obj.have_DewarpNet_result):
            self.tf_data.DewarpNet_result_test = self.DewarpNet_result_test_factory.build_img_db()
            self.tf_data.DewarpNet_result_see  = self.DewarpNet_result_see_factory .build_img_db()

            self.tf_data.DewarpNet_result_test_amount = get_db_amount(self.tf_data.db_obj.DewarpNet_result_test_dir)
            self.tf_data.DewarpNet_result_see_amount  = get_db_amount(self.tf_data.db_obj.DewarpNet_result_see_dir)
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
    ''' old '''
    # db_obj = Dataset_builder().set_basic(DB_C.type8_blender                      , DB_N.blender_os_hw768      , DB_GM.build_by_in_I_gt_F_or_W_hole_norm_then_no_mul_M_wrong, h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", db_in_range=Range(0, 255), gt_format="knpy", db_gt_range=Range(0, 1), rec_hope_format="jpg", db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True).build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(-1, 1), use_gt_range=Range(-1, 1), use_rec_hope_range=Range(0, 255)).build_by_db_get_method().build()

    ''' 柱狀 '''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc.build()  ### 沒有 mul_M_wrpmg
    # db_obj = type8_blender_wc_try_mul_M.build()  ### 有 mul_M_right
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' kong_doc3d'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d_in_I_gt_W.build()  ### 有 mul_M_right, hole_norm
    db_obj = type8_blender_kong_doc3d_in_I_gt_W_ch_norm.build()  ### 有 mul_M_right, ch_norm
    ### Kong_Doc3D V1 才用這個， 升級到 V2就不用了喔， 因為在DB方面就已經把x軸反轉了
    # db_obj = type8_blender_kong_doc3d_in_I_gt_W_ch_norm_only_for_doc3d_x_value_reverse.build()  ### 有 mul_M_right, ch_norm
    print(db_obj)
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()
