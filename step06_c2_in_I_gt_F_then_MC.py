from step06_c0_tf_Data_initial_builder import tf_Data_init_builder
from kong_util.util import get_db_amount

class tf_Data_in_dis_gt_mask_coord_builder(tf_Data_init_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_in_dis_gt_mask_coord_builder, self).__init__(tf_data)

    def build_by_in_I_gt_F_MC_norm_then_no_mul_M_wrong(self):
        '''
        in_ord: dis_img, shape = (1, ord_h, ord_w, 3), value:0 ~255
        in_pre: dis_img, shape = (1,  db_h,  db_w, 3), value:0 ~  1
        gt_ord: flow,    shape = (1,  db_h , db_w, 3), value:0 ~  1, ch0: mask, ch1:y, ch2:x
        gt_pre: flow,    shape = (1,  db_h , db_w, 3), value:0 ~  1, ch0: mask, ch1:y, ch2:x
        '''
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_C_db_wrong()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.test_gt_db = self.test_gt_factory.build_M_C_db_wrong()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine):
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0].numpy()
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()
        #     # ####################
        #     # ### 確認一下 tight_crop 的效果如何， mask 有沒有被 reflect pad 到
        #     # from step08_b_use_G_generate_0_util import Tight_crop
        #     # from kong_util.build_dataset_combine import Check_dir_exist_and_build
        #     # print(name)
        #     # Mgt_pre     = train_gt_pre[0, ..., 0:1].numpy()
        #     # dis_img_pre = train_in_pre[0].numpy()
        #     # flow_pre    = train_gt_pre[0].numpy()
        #     # tight_crop = Tight_crop(pad_size= 60, resize=(256, 256))
        #     # crop_dis_img_pre , boundary = tight_crop(dis_img_pre, Mgt_pre)
        #     # crop_flow_pre    , boundary = tight_crop(flow_pre   , Mgt_pre)

        #     # plt_img = 3
        #     # fig, ax = plt.subplots(nrows=1, ncols=plt_img, figsize=(5 * plt_img, 5))
        #     # ax[0].imshow(dis_img_pre)
        #     # ax[1].imshow(crop_dis_img_pre)
        #     # ax[2].imshow(crop_flow_pre[..., 0:1])

        #     # import cv2
        #     # debug_dir = r"C:\Users\TKU\Desktop\kong_model2\debug_data\doc3d_tight_crop_check"
        #     # Check_dir_exist_and_build(debug_dir)
        #     # cv2.imwrite(f"{debug_dir}/%06i_dis_img.png"      % i, (dis_img_pre * 255.).astype(np.uint8))
        #     # cv2.imwrite(f"{debug_dir}/%06i_mask.png"         % i, (flow_pre[..., 0:1] * 255.).astype(np.uint8))
        #     # cv2.imwrite(f"{debug_dir}/%06i_dis_img_crop.png" % i, (crop_dis_img_pre.numpy() * 255.).astype(np.uint8))
        #     # cv2.imwrite(f"{debug_dir}/%06i_mask_crop.png"    % i, (crop_flow_pre[..., 0:1].numpy() * 255.).astype(np.uint8))
        #     # plt.show()
        #     # print("finish")
        #     # # tight_crop.reset_jit()  ### 測試看看沒設定 jit_scale 會不會跳出錯誤訊息


        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0]
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
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            self.tf_data.see_gt_db   = self.see_gt_factory.build_M_C_db_wrong()

            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)
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

            #     from step08_b_use_G_generate_0_util import Tight_crop
            #     gt_mask_pre = see_gt_pre[..., 0:1]
            #     tight_crop = Tight_crop(pad_size=20, resize=(256, 256))
            #     crop_see_in    , _ = tight_crop(see_in    , gt_mask_pre)
            #     crop_see_in_pre, _ = tight_crop(see_in_pre, gt_mask_pre)
            #     crop_see_gt    , _ = tight_crop(see_gt    , gt_mask_pre)
            #     crop_see_gt_pre, _ = tight_crop(see_gt_pre, gt_mask_pre)
            #     tight_crop.reset_jit()  ### 測試看看沒設定 jit_scale 會不會跳出錯誤訊息

            #     debug_dict[f"{i}--5-1 crop_see_in"    ] = crop_see_in
            #     debug_dict[f"{i}--5-2 crop_see_in_pre"] = crop_see_in_pre
            #     debug_dict[f"{i}--5-3 crop_see_gt"    ] = crop_see_gt
            #     debug_dict[f"{i}--5-4 crop_see_gt_pre"] = crop_see_gt_pre

            #     debug_dict[f"{i}--6-1  crop_see_in"     ] = crop_see_in[0].numpy()
            #     debug_dict[f"{i}--6-2  crop_see_in_pre" ] = crop_see_in_pre[0].numpy()
            #     debug_dict[f"{i}--6-3a crop_see_gt_mask"] = crop_see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--6-3b crop_see_gt_move"] = crop_see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--6-4a crop_see_gt_pre_mask"] = crop_see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--6-4b crop_see_gt_pre_move"] = crop_see_gt_pre[0, ..., 1:3].numpy()

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
