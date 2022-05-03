from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount
import tensorflow as tf

class tf_Data_in_wc_gt_flow_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_wc_gt_flow_builder, self).__init__(tf_data)

    def build_by_in_W_and_I_gt_F_MC_norm_then_no_mul_M_wrong(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_M_W_db_wrong()

        self.tf_data.test_in_db    = self.test_in_factory.build_M_W_db_wrong()
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()

        ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
        self.tf_data.train_in2_db     = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db      = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db.ord  = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in2_db.ord))
        self.tf_data.test_in_db .ord  = tf.data.Dataset.zip((self.tf_data.test_in_db.ord , self.tf_data.test_in2_db .ord))
        self.tf_data.train_in_db.pre  = tf.data.Dataset.zip((self.tf_data.train_in_db.pre, self.tf_data.train_in2_db.pre))
        self.tf_data.test_in_db .pre  = tf.data.Dataset.zip((self.tf_data.test_in_db.pre , self.tf_data.test_in2_db .pre))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_C_db_wrong()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_M_C_db_wrong()

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
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in[0]  ### [0]第一個是 取 wc, [1] 是取 dis_img
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0][0]  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     train_in_pre      = train_in_pre[0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(1, 6)
        #     fig.set_size_inches(30, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt_mask)
        #     ax[3].imshow(train_gt_pre_mask)
        #     ax[4].imshow(train_gt_move_visual)
        #     ax[5].imshow(train_gt_pre_move_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_M_W_db_wrong()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
            self.tf_data.see_in2_db = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db.ord  = tf.data.Dataset.zip((self.tf_data.see_in_db.ord, self.tf_data.see_in2_db.ord))
            self.tf_data.see_in_db.pre  = tf.data.Dataset.zip((self.tf_data.see_in_db.pre, self.tf_data.see_in2_db.pre))

            self.tf_data.see_gt_db  = self.see_gt_factory.build_M_C_db_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

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

    def build_by_in_W_and_I_gt_F_WC_norm_then_mul_M_right(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_M_W_db_right()

        self.tf_data.test_in_db   = self.test_in_factory.build_M_W_db_right()
        self.tf_data.test_name_db = self.test_in_factory.build_name_db()

        ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
        self.tf_data.train_in2_db = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db  = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db.ord = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in2_db.ord))
        self.tf_data.test_in_db .ord = tf.data.Dataset.zip((self.tf_data.test_in_db .ord, self.tf_data.test_in2_db .ord))
        self.tf_data.train_in_db.pre = tf.data.Dataset.zip((self.tf_data.train_in_db.pre, self.tf_data.train_in2_db.pre))
        self.tf_data.test_in_db .pre = tf.data.Dataset.zip((self.tf_data.test_in_db .pre, self.tf_data.test_in2_db .pre))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount  = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_C_db_wrong()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_M_C_db_wrong()

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
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in[0]  ### [0]第一個是 取 wc, [1] 是取 dis_img
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre[0]
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0][0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0][0]  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     train_in_pre      = train_in_pre[0][0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     #  # print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(3, 5)
        #     fig.set_size_inches(25, 15)
        #     ax[0, 0].imshow(train_in)
        #     ax[0, 1].imshow(train_in_pre)
        #     ax[0, 2].imshow(train_in_pre[..., 0:1])
        #     ax[0, 3].imshow(train_in_pre[..., 1:2])
        #     ax[0, 4].imshow(train_in_pre[..., 2:3])

        #     ax[1, 0].imshow(train_gt_mask)
        #     ax[1, 1].imshow(train_gt_pre_mask)
        #     ax[1, 2].imshow(train_gt_move_visual)

        #     ax[2, 0].imshow(train_gt_pre_move[..., 0:1])
        #     ax[2, 1].imshow(train_gt_pre_move[..., 1:2])
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_M_W_db_right()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
            self.tf_data.see_in2_db = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db.ord  = tf.data.Dataset.zip((self.tf_data.see_in_db.ord, self.tf_data.see_in2_db.ord))
            self.tf_data.see_in_db.pre  = tf.data.Dataset.zip((self.tf_data.see_in_db.pre, self.tf_data.see_in2_db.pre))

            self.tf_data.see_gt_db = self.see_gt_factory.build_M_C_db_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！ 這要用sypder開才看的到喔
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

            #     # ####### Become doc3d
            #     # ##### 弄成 doc3D 的 448*448， 用任何一個能抓到 flow/wc 的 get_method 都可以寫， 只是上次是用這個寫 update_wc， 就乾脆沿用繼續用這個 get_method 來寫囉～
            #     # from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
            #     # import cv2
            #     # wc      = see_in[0][0, ..., 0:3].numpy()
            #     # dis_img = see_in[1][0].numpy()
            #     # uv      = see_gt[0].numpy()
            #     # mask    = see_gt[0][..., 0:1].numpy()
            #     # ### 視覺化一下
            #     # fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(5 * 6, 5))
            #     # ax[0].imshow(wc[..., 0])
            #     # ax[1].imshow(wc[..., 1])
            #     # ax[2].imshow(wc[..., 2])
            #     # ax[3].imshow(uv[..., 0])
            #     # ax[4].imshow(uv[..., 1])
            #     # ax[5].imshow(uv[..., 2])
            #     # plt.tight_layout()
            #     # # plt.show()

            #     # dis_img_doc3d = dis_img[..., ::-1]  ### tf2 讀出來是 rgb， cv2存圖是bgr， 所以要 rgb2bgr

            #     # wc_ch_min = np.array([0.0        , -0.13532962, -0.080751580]).reshape(1, 1, 3)
            #     # wc_ch_max = np.array([0.039187048,  0.1357405 ,  0.07755918 ]).reshape(1, 1, 3)
            #     # wc_01 = (wc - wc_ch_min) / (wc_ch_max - wc_ch_min)

            #     # wc_ch_min_doc3d = np.array([-0.67187124, -1.2280148, -1.2410645]).reshape(1, 1, 3)
            #     # wc_ch_max_doc3d = np.array([ 0.63452387,  1.2387834,  1.2485291]).reshape(1, 1, 3)
            #     # wc_doc3d = wc_01 * (wc_ch_max_doc3d - wc_ch_min_doc3d) + wc_ch_min_doc3d

            #     # uv_doc3d   = cv2.resize(uv,       (448, 448)).astype(np.float32)
            #     # wc_doc3d   = cv2.resize(wc_doc3d, (448, 448)).astype(np.float32)
            #     # mask_doc3d = cv2.resize(mask,     (448, 448)).reshape(448, 448, 1)
            #     # W_w_M_doc3d = np.concatenate((wc_doc3d, mask_doc3d), axis=-1).astype(np.float32)


            #     # become_doc3d = "F:/kong_model2/debug_data/DB_become_doc3d/see"
            #     # become_doc3d_dis_img_dir    = f"{become_doc3d}/0_dis_img"
            #     # become_doc3d_uv_npy_dir     = f"{become_doc3d}/1_uv-1_npy"
            #     # become_doc3d_uv_knpy_dir    = f"{become_doc3d}/1_uv-3_knpy"
            #     # become_doc3d_wc_npy_dir     = f"{become_doc3d}/2_wc-1_npy"
            #     # become_doc3d_W_w_M_npy_dir  = f"{become_doc3d}/2_wc-4_W_w_M_npy"
            #     # become_doc3d_W_w_M_knpy_dir = f"{become_doc3d}/2_wc-5_W_w_M_knpy"
            #     # Check_dir_exist_and_build(become_doc3d_dis_img_dir    )
            #     # Check_dir_exist_and_build(become_doc3d_uv_npy_dir    )
            #     # Check_dir_exist_and_build(become_doc3d_uv_knpy_dir   )
            #     # Check_dir_exist_and_build(become_doc3d_wc_npy_dir    )
            #     # Check_dir_exist_and_build(become_doc3d_W_w_M_npy_dir )
            #     # Check_dir_exist_and_build(become_doc3d_W_w_M_knpy_dir)
            #     # see_name = "see_%03i" % (i + 1)
            #     # doc3d_dis_img_path    = become_doc3d_dis_img_dir    + "/" + see_name + ".png"
            #     # doc3d_uv_npy_path     = become_doc3d_uv_npy_dir     + "/" + see_name + ".npy"
            #     # doc3d_uv_knpy_path    = become_doc3d_uv_knpy_dir    + "/" + see_name + ".knpy"
            #     # doc3d_wc_npy_path     = become_doc3d_wc_npy_dir     + "/" + see_name + ".npy"
            #     # doc3d_W_w_M_npy_path  = become_doc3d_W_w_M_npy_dir  + "/" + see_name + ".npy"
            #     # doc3d_W_w_M_knpy_path = become_doc3d_W_w_M_knpy_dir + "/" + see_name + ".knpy"

            #     # cv2.imwrite(doc3d_dis_img_path, dis_img_doc3d)

            #     # np.save(doc3d_uv_npy_path, uv_doc3d)
            #     # Save_npy_path_as_knpy(src_path=doc3d_uv_npy_path, dst_path=doc3d_uv_knpy_path)

            #     # np.save(doc3d_wc_npy_path, wc_doc3d)

            #     # np.save(doc3d_W_w_M_npy_path, W_w_M_doc3d)
            #     # Save_npy_path_as_knpy(src_path=doc3d_W_w_M_npy_path, dst_path=doc3d_W_w_M_knpy_path)

            # #     # ##### flow 的 mask 是更新後還不錯的mask， 把他抓出來 更新 W 的 mask
            # #     # from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
            # #     # update_M_dir = "F:/kong_model2/debug_data/DB_update_wc_mask"
            # #     # npy_dir  = update_M_dir + "/npy"
            # #     # knpy_dir = update_M_dir + "/knpy"
            # #     # Check_dir_exist_and_build(update_M_dir)
            # #     # Check_dir_exist_and_build(npy_dir)
            # #     # Check_dir_exist_and_build(knpy_dir)
            # #     # W_w_M  = see_in[0][0].numpy()
            # #     # M_good = see_gt[0, ..., 0:1].numpy()
            # #     # W = W_w_M[..., 0:3]
            # #     # W_w_M_good = np.concatenate([W, M_good], axis=-1)
            # #     # npy_path  = f"{npy_dir}/{i + 1}_W_w_M_good.npy"
            # #     # knpy_path = f"{knpy_dir}/{i + 1}_W_w_M_good.knpy"
            # #     # np.save(npy_path, W_w_M_good)
            # #     # Save_npy_path_as_knpy(src_path=npy_path, dst_path=knpy_path)
            # #     # ### 視覺化一下
            # #     # # fig, ax = plt.subplots(nrows=1, ncols=2)
            # #     # # ax[0].imshow(W_w_M [..., 3])
            # #     # # ax[1].imshow(M_good[..., 0])
            # #     # # plt.show()
            # #     # print("see finish")

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
            # import cv2
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.ord):
            #     ####### Become doc3d
            #     rec_hope = rec_hope_see.numpy()[..., ::-1]  ### tf2 讀出來是 rgb， cv2存圖是bgr， 所以要 rgb2bgr

            #     become_doc3d = "F:/kong_model2/debug_data/DB_become_doc3d/see"
            #     become_doc3d_rec_hope_dir     = f"{become_doc3d}/0_rec_hope"
            #     Check_dir_exist_and_build(become_doc3d_rec_hope_dir    )
            #     see_name = "see_%03i" % (i + 1)
            #     doc3d_rec_hope_path     = become_doc3d_rec_hope_dir     + "/" + see_name + ".png"
            #     cv2.imwrite(doc3d_rec_hope_path, rec_hope)

            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope)
            #     # plt.show()
            #     # plt.close()
            ##########################################################################################################################################
        return self
