from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount

class tf_Data_in_dis_gt_flow_or_wc_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_dis_gt_flow_or_wc_builder, self).__init__(tf_data)

    def build_by_in_I_gt_F_or_W_hole_norm_then_no_mul_M_wrong(self, get_what="flow"):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        if  (get_what == "flow"): self.tf_data.train_gt_db = self.train_gt_factory.build_flow_db()
        elif(get_what == "wc"  ): self.tf_data.train_gt_db = self.train_gt_factory.build_M_W_db_wrong()

        if  (get_what == "flow"): self.tf_data.test_gt_db = self.test_gt_factory.build_flow_db()
        elif(get_what == "wc"):   self.tf_data.test_gt_db = self.test_gt_factory.build_M_W_db_wrong()


        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            if  (get_what == "flow"): self.tf_data.see_gt_db = self.see_gt_factory.build_flow_db()
            elif(get_what == "wc"):   self.tf_data.see_gt_db = self.see_gt_factory.build_M_W_db_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)


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
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1

        # # for i in enumerate(self.tf_data.train_gt_db): pass
        # # print("train_gt_finish")

        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(5)):
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1 train_in"    ] = train_in[0].numpy()
        #     debug_dict[f"{i}--2-2 train_in_pre"] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 train_Mgt"    ] = train_gt[0][0].numpy()
        #     debug_dict[f"{i}--2-4 train_Mgt_pre"] = train_gt_pre[0][0].numpy()
        #     debug_dict[f"{i}--2-5 train_Wgt"    ] = train_gt[1][0].numpy()
        #     debug_dict[f"{i}--2-6 train_Wgt_pre"] = train_gt_pre[1][0].numpy()

        #     if(get_what == "flow"):
        #         train_gt_visual     = method1(train_gt[0, ..., 2]    , train_gt[0, ..., 1])
        #         train_gt_pre_visual = method1(train_gt_pre[0, ..., 2], train_gt_pre[0, ..., 1])

        #         fig, ax = plt.subplots(1, 4)
        #         fig.set_size_inches(15, 5)
        #         ax[0].imshow(train_in[0])
        #         ax[1].imshow(train_in_pre[0])
        #         ax[2].imshow(train_gt_visual)
        #         ax[3].imshow(train_gt_pre_visual)
        #         ax[2].imshow(train_gt[0])
        #         ax[3].imshow(train_gt_pre[0])

        #     elif(get_what == "wc"):
        #         fig, ax = plt.subplots(2, 5)
        #         fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #         ### ord vs pre
        #         ax[0, 0].imshow(train_in    [0])
        #         ax[0, 1].imshow(train_in_pre[0])

        #         ### W_ord vs W_pre
        #         ax[0, 2].imshow(train_gt    [0, ..., :3])
        #         ax[0, 3].imshow(train_gt_pre[0, ..., :3])

        #         ### Wx, Wy, Wz 看一下長什麼樣子
        #         ax[1, 0].imshow(train_gt_pre[0, ..., 0])
        #         ax[1, 1].imshow(train_gt_pre[0, ..., 1])
        #         ax[1, 2].imshow(train_gt_pre[0, ..., 2])

        #         ### M_ord vs M_pre
        #         ax[1, 3].imshow(train_gt    [0, ..., 3])
        #         ax[1, 4].imshow(train_gt_pre[0, ..., 3])
        #     fig.tight_layout()
        #     plt.show()
        #########################################################################################################################################
        return self

    def build_by_in_I_gt_W_hole_norm_then_mul_M_right(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()
        # self.tf_data.train_gt_db = self.train_gt_factory.build_M_W_db_wrong()  ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        # self.tf_data.test_gt_db  = self.test_gt_factory.build_M_W_db_wrong()   ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_W_db_right()
        self.tf_data.test_gt_db  = self.test_gt_factory.build_M_W_db_right()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        # print('self.tf_data.train_in_db',self.tf_data.train_in_db.ord)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db.pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db.ord)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db.pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_M_W_db_right()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # from util import method1
            # for i in enumerate(self.tf_data.train_gt_db): pass
            # print("train_gt_finish")

            # for i, (see_in, see_in_pre, see_gt, see_gt_pre, name) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                         self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1),
            #                                                                                         self.tf_data.see_name_db.ord))):
            #     debug_dict[f"{i}--1-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--1-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--1-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--1-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--2-1 see_in"    ]  = see_in    [0].numpy()
            #     debug_dict[f"{i}--2-2 see_in_pre"]  = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--2-3 see_Mgt"    ] = see_gt    [0].numpy()
            #     debug_dict[f"{i}--2-4 see_Mgt_pre"] = see_gt_pre[0].numpy()
            #     debug_dict[f"{i}--2-5 see_Wgt"    ] = see_gt    [0].numpy()
            #     debug_dict[f"{i}--2-6 see_Wgt_pre"] = see_gt_pre[0].numpy()

            #     fig, ax = plt.subplots(2, 5)
            #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
            #     ### ord vs pre
            #     ax[0, 0].imshow(see_in    [0])
            #     ax[0, 1].imshow(see_in_pre[0])

            #     ### W_ord vs W_pre
            #     ax[0, 2].imshow(see_gt    [0, ..., :3])
            #     ax[0, 3].imshow(see_gt_pre[0, ..., :3])

            #     ### Wx, Wy, Wz 看一下長什麼樣子
            #     ax[1, 0].imshow(see_gt_pre[0, ..., 0])
            #     ax[1, 1].imshow(see_gt_pre[0, ..., 1])
            #     ax[1, 2].imshow(see_gt_pre[0, ..., 2])

            #     ### M_ord vs M_pre
            #     Mgt     = see_gt    [0, ..., 3:4]
            #     Mgt_pre = see_gt_pre[0, ..., 3:4]
            #     ax[1, 3].imshow(see_gt    [0, ..., 3:4])
            #     ax[1, 4].imshow(see_gt_pre[0, ..., 3:4])

            #     in_pre_w_Mgt_pre = see_in_pre[0] * Mgt_pre
            #     ax[0, 4].imshow(in_pre_w_Mgt_pre)

            #     fig.tight_layout()
            #     plt.show()
            ##########################################################################################################################################

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


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # import os
        # for i in enumerate(self.tf_data.train_gt_db.ord): pass
        # print("train_gt_finish")

        # # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(5)):
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine):
        #     # if(i < 620): continue
        #     # print(i)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1 train_in"    ]  = train_in    [0].numpy()
        #     debug_dict[f"{i}--2-2 train_in_pre"]  = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 train_Mgt"    ] = train_gt    [0].numpy()
        #     debug_dict[f"{i}--2-4 train_Mgt_pre"] = train_gt_pre[0].numpy()
        #     debug_dict[f"{i}--2-5 train_Wgt"    ] = train_gt    [0].numpy()
        #     debug_dict[f"{i}--2-6 train_Wgt_pre"] = train_gt_pre[0].numpy()

        #     fig, ax = plt.subplots(2, 5)
        #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #     ### ord vs pre
        #     in_ord = train_in    [0]
        #     in_pre = train_in_pre[0]
        #     ax[0, 0].imshow(in_ord)
        #     ax[0, 1].imshow(in_pre)

        #     ### W_ord vs W_pre
        #     W_ord = train_gt    [0, ..., :3]
        #     W_pre = train_gt_pre[0, ..., :3]
        #     ax[0, 2].imshow(W_ord)
        #     ax[0, 3].imshow(W_pre)

        #     ### Wx, Wy, Wz 看一下長什麼樣子
        #     Wx_pre = train_gt_pre[0, ..., 0]
        #     Wy_pre = train_gt_pre[0, ..., 1]
        #     Wz_pre = train_gt_pre[0, ..., 2]
        #     ax[1, 0].imshow(Wx_pre)
        #     ax[1, 1].imshow(Wy_pre)
        #     ax[1, 2].imshow(Wz_pre)

        #     ### M_ord vs M_pre
        #     Mgt     = train_gt    [0, ..., 3:4]
        #     Mgt_pre = train_gt_pre[0, ..., 3:4]
        #     ax[1, 3].imshow(Mgt)
        #     ax[1, 4].imshow(Mgt_pre)

        #     ### W_pre * M
        #     W_pre_w_Mgt_pre = W_pre * Mgt_pre
        #     ax[0, 4].imshow(W_pre_w_Mgt_pre)

        #     fig.tight_layout()
        #     if(os.path.isdir(self.tf_data.db_obj.check_train_gt_dir) is False): os.makedirs(self.tf_data.db_obj.check_train_gt_dir)
        #     plt.show()
        #     # plt.savefig(f"{self.tf_data.db_obj.check_train_gt_dir}/" + "%05i" % (i + 1) )
        #     # plt.close()
        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # import os
        # # for i in enumerate(self.tf_data.train_gt_db.ord): pass
        # # print("train_gt_finish")

        # # for i, (test_in, test_in_pre, test_gt, test_gt_pre, name) in enumerate(self.tf_data.test_db_combine.take(5)):
        # for i, (test_in, test_in_pre, test_gt, test_gt_pre, name) in enumerate(self.tf_data.test_db_combine):
        #     debug_dict[f"{i}--1-1 test_in"    ] = test_in
        #     debug_dict[f"{i}--1-2 test_in_pre"] = test_in_pre
        #     debug_dict[f"{i}--1-3 test_gt"    ] = test_gt
        #     debug_dict[f"{i}--1-4 test_gt_pre"] = test_gt_pre

        #     debug_dict[f"{i}--2-1 test_in"    ]  = test_in    [0].numpy()
        #     debug_dict[f"{i}--2-2 test_in_pre"]  = test_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 test_Mgt"    ] = test_gt    [0].numpy()
        #     debug_dict[f"{i}--2-4 test_Mgt_pre"] = test_gt_pre[0].numpy()
        #     debug_dict[f"{i}--2-5 test_Wgt"    ] = test_gt    [0].numpy()
        #     debug_dict[f"{i}--2-6 test_Wgt_pre"] = test_gt_pre[0].numpy()

        #     fig, ax = plt.subplots(2, 5)
        #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #     ### ord vs pre
        #     ax[0, 0].imshow(test_in    [0])
        #     ax[0, 1].imshow(test_in_pre[0])

        #     ### W_ord vs W_pre
        #     ax[0, 2].imshow(test_gt    [0, ..., :3])
        #     ax[0, 3].imshow(test_gt_pre[0, ..., :3])

        #     ### Wx, Wy, Wz 看一下長什麼樣子
        #     ax[1, 0].imshow(test_gt_pre[0, ..., 0])
        #     ax[1, 1].imshow(test_gt_pre[0, ..., 1])
        #     ax[1, 2].imshow(test_gt_pre[0, ..., 2])

        #     ### M_ord vs M_pre
        #     ax[1, 3].imshow(test_gt    [0, ..., 3])
        #     ax[1, 4].imshow(test_gt_pre[0, ..., 3])
        #     fig.tight_layout()
        #     if os.path.isdir(self.tf_data.db_obj.check_test_gt_dir) is False: os.makedirs(self.tf_data.db_obj.check_test_gt_dir)
        #     # plt.show()
        #     plt.savefig(f"{self.tf_data.db_obj.check_test_gt_dir}/" + "%05i" % (i + 1) )
        #     plt.close()
        #########################################################################################################################################
        return self
