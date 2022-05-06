from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount

import numpy as np
import os

debug_dict = {}

class tf_Data_in_dis_gt_move_map_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_dis_gt_move_map_builder, self).__init__(tf_data)

    def build_by_in_dis_gt_move_map(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()


        ### 在拿move_map db 之前，要先去抓 max/min train_move，我是設計放 train_gt_dir 下的.npy，如果怕混淆 要改放.txt之類的都可以喔！
        ### 決定還是放在上一層好了，因為下面會用 get_db_amount 是算檔案數量的，雖然是去in_dir抓影像跟gt_dir沒關係，但還是怕有意外(以後忘記之類的)～放外面最安全囉！
        ### 且放外面容易看到可以提醒自己有這東西的存在覺得ˊ口ˋ
        if(os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy") and
           os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")):
            self.tf_data.gt_max = np.load(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy")
            self.tf_data.gt_min = np.load(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")
        else:  ### 如果.npy不存在，就去重新找一次 max/min train_move，找完也順便存一份給之後用囉！
            print("因為現在已經存成.knpy，沒辦法抓 max/min train_move 囉！麻煩先去以前的dataset撈出來啦！")
            ### 偷懶可以把 .npy 放同個資料夾，把註解拿掉就可以順便求囉！只是因為這有return，所以還是要重新執行一次才會完整跑完喔～
            # move_maps = get_dir_moves(self.tf_data.db_obj.train_gt_dir)
            # self.tf_data.gt_max = move_maps.max()
            # self.tf_data.gt_min = move_maps.min()
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../max_train_move", self.tf_data.max_train_move)
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../min_train_move", self.tf_data.min_train_move)
            # print("self.tf_data.max_train_move",self.tf_data.max_train_move)
            # print("self.tf_data.min_train_move",self.tf_data.min_train_move)
            return

        self.tf_data.train_gt_db = self.train_gt_factory.build_mov_db()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_mov_db()


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

            self.tf_data.see_gt_db   = self.see_gt_factory.build_mov_db()

            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method2

        # take_num = 5
        # print(self.tf_data.train_max)
        # print(self.tf_data.train_min)
        # for i, (img, img_pre, move, move_pre) in enumerate(self.tf_data.train_db_combine.take(take_num)):     ### 想看test 的部分用這行 且 註解掉上行
        #     print("i",i)
        #     fig, ax = plt.subplots(1,4)
        #     fig.set_size_inches(15,5)
        #     ax_i = 0
        #     img = tf.cast(img[0], tf.uint8)
        #     ax[ax_i].imshow(img)
        #     print(img.numpy().dtype)

        #     ax_i += 1
        #     img_pre_back = (img_pre[0]+1.)*127.5
        #     img_pre_back = tf.cast(img_pre_back, tf.int32)
        #     ax[ax_i].imshow(img_pre_back)

        #     ax_i += 1
        #     move_bgr = method2(move[0,...,0], move[0,...,1])
        #     ax[ax_i].imshow(move_bgr)

        #     ax_i += 1
        #     # move_back = (move[0]+1)/2 * (train_max-train_min) + train_min  ### 想看train的部分用這行 且 註解掉下行
        #     move_back = (move_pre[0]+1)/2 * (self.tf_data.train_max-self.tf_data.train_min) + self.tf_data.train_min    ### 想看test 的部分用這行 且 註解掉上行
        #     move_back_bgr = method2(move_back[...,0], move_back[...,1],1)
        #     ax[ax_i].imshow(move_back_bgr)
        #     plt.show()
        #     plt.close()
        #########################################################################################################################################
        return self


class tf_Data_in_dis_gt_img_builder(tf_Data_in_dis_gt_move_map_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_dis_gt_img_builder, self).__init__(tf_data)

    def build_by_in_I_and_gt_I_db(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        self.tf_data.train_gt_db = self.train_gt_factory.build_img_db()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_img_db()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        #########################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     train_in     = train_in[0]      ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0]  ### 值 0. ~ 1.
        #     train_gt     = train_gt[0]      ### 值 0  ~ 255
        #     train_gt_pre = train_gt_pre[0]  ### 值 0. ~ 1.

        #     fig, ax = plt.subplots(1, 4)
        #     fig.set_size_inches(15, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()
        #########################################################

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_img_db()

            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)
        return self
    ############################################################


class tf_Data_in_img_gt_mask_builder(tf_Data_in_dis_gt_img_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_img_gt_mask_builder, self).__init__(tf_data)

    def build_by_in_img_gt_mask(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()


        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_mask_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.test_gt_db  = self.test_gt_factory.build_mask_db()

        print("self.tf_data.train_in_db.ord", self.tf_data.train_in_db.ord)
        print("self.tf_data.train_in_db.pre", self.tf_data.train_in_db.pre)
        print("self.tf_data.train_gt_db.ord", self.tf_data.train_gt_db.ord)
        print("self.tf_data.train_gt_db.pre", self.tf_data.train_gt_db.pre)

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     # print(train_in.numpy().shape)       ### (10, 768, 768, 3)
        #     train_in     = train_in[0]          ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0]      ### 值 0. ~ 1.
        #     print("train_in", train_in.numpy().dtype)       ### uint8
        #     print("train_in", train_in.numpy().shape)       ### (h, w, 3)
        #     print("train_in", train_in.numpy().min())       ### 0
        #     print("train_in", train_in.numpy().max())       ### 255
        #     print("train_in_pre", train_in_pre.numpy().dtype)   ### float32
        #     print("train_in_pre", train_in_pre.numpy().shape)   ### (h, w, 3)
        #     print("train_in_pre", train_in_pre.numpy().min())   ### 0.0
        #     print("train_in_pre", train_in_pre.numpy().max())   ### 1.0

        #     # print(train_gt.numpy().shape)       ### (10, 768, 768, 3)
        #     train_gt     = train_gt[0]          ### 值 0. ~ 1.
        #     train_gt_pre = train_gt_pre[0]      ### 值 0. ~ 1.
        #     print("train_gt", train_gt.numpy().dtype)       ### float32
        #     print("train_gt", train_gt.numpy().shape)       ### (h, w, 3)
        #     print("train_gt", train_gt.numpy().min())       ### 0.0
        #     print("train_gt", train_gt.numpy().max())       ### 1.0
        #     print("train_gt_pre", train_gt_pre.numpy().dtype)   ### float32
        #     print("train_gt_pre", train_gt_pre.numpy().min())   ### 0.0
        #     print("train_gt_pre", train_gt_pre.numpy().max())   ### 1.0


        #     fig, ax = plt.subplots(1, 4)
        #     fig.set_size_inches(15, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()

        '''
        還沒弄see
        '''
        return self

if(__name__ == "__main__"):
    from step09_d_KModel_builder_combine_step789 import MODEL_NAME, KModel_builder
    from step06_a_datas_obj import *
    import time

    start_time = time.time()

    # db_obj = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="bmp", db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=batch_size-1, train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    # db_obj = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page, DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="...", db_gt_range=Range(...)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_unet()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    '''in_img, gt_mask'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_try_segmentation.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()
