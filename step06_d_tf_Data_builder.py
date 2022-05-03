from step06_a_datas_obj          import DB_GM
from step06_c1_old               import tf_Data_in_img_gt_mask_builder
from step06_c2_in_I_gt_F_then_MC import tf_Data_in_dis_gt_mask_coord_builder
from step06_c3_in_I_gt_F_or_W    import tf_Data_in_dis_gt_flow_or_wc_builder
from step06_c4_in_W_gt_F         import tf_Data_in_wc_gt_flow_builder
from step06_c5_in_I_gt_W_F       import tf_Data_in_dis_gt_wc_flow_builder

import time

class tf_Data_builder(tf_Data_in_img_gt_mask_builder, tf_Data_in_dis_gt_mask_coord_builder, tf_Data_in_dis_gt_flow_or_wc_builder, tf_Data_in_wc_gt_flow_builder, tf_Data_in_dis_gt_wc_flow_builder):
    def __init__(self, tf_data=None):
        super(tf_Data_builder, self).__init__(tf_data)

    def build_by_db_get_method(self):
        self._build_by_db_get_method_init()
        if    (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_move_map):
            self.build_by_in_dis_gt_move_map()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord or
               self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord_pad or
               self.tf_data.db_obj.get_method == DB_GM.build_by_in_I_and_gt_I_db):                             self.build_by_in_I_and_gt_I_db()
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_img_gt_mask):                               self.build_by_in_img_gt_mask()

        ### I_to_F
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_I_gt_F_hole_norm_then_no_mul_M_wrong):      self.build_by_in_I_gt_F_or_W_hole_norm_then_no_mul_M_wrong(get_what="flow")

        ### I_to_M
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_I_gt_F_MC_norm_then_no_mul_M_wrong):        self.build_by_in_I_gt_F_MC_norm_then_no_mul_M_wrong()
        ### I_to_W
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_I_gt_W_hole_norm_then_no_mul_M_wrong):      self.build_by_in_I_gt_F_or_W_hole_norm_then_no_mul_M_wrong(get_what="wc")
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_I_gt_W_hole_norm_then_mul_M_right):         self.build_by_in_I_gt_W_hole_norm_then_mul_M_right()
        ### W_to_C
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_W_and_I_gt_F_MC_norm_then_no_mul_M_wrong):  self.build_by_in_W_and_I_gt_F_MC_norm_then_no_mul_M_wrong()  ### wrong
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_W_and_I_gt_F_WC_norm_then_mul_M_right):     self.build_by_in_W_and_I_gt_F_WC_norm_then_mul_M_right()     ### right
        ### I_w_M_to_W_to_C
        elif  (self.tf_data.db_obj.get_method == DB_GM.build_by_in_I_gt_W_and_F_try_mul_M):                    self.build_by_in_I_gt_W_and_F_try_mul_M()


        return self


    def build(self):
        print(f"TF_data_builder build finish")
        return self.tf_data

########################################################################################################################################
### 因為我們還需要根據 使用的model(其實就是看model_name) 來決定如何resize，所以就不在這邊 先建構好 許多tf_data物件囉！

if(__name__ == "__main__"):
    from step09_d_KModel_builder_combine_step789 import MODEL_NAME, KModel_builder
    from step06_a_datas_obj import *

    start_time = time.time()

    # db_obj = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="bmp", db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=batch_size-1, train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    # db_obj = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page, DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="...", db_gt_range=Range(...)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_unet()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    ''' mask_flow 3ch合併 的形式'''
    # db_obj = Dataset_builder().set_basic(DB_C.type8_blender                      , DB_N.blender_os_hw768      , DB_GM.build_by_in_I_gt_F_or_W_hole_norm_then_no_mul_M_wrong, h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", db_in_range=Range(0, 255), gt_format="knpy", db_gt_range=Range(0, 1), rec_hope_format="jpg", db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True).build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(-1, 1), use_gt_range=Range(-1, 1), use_rec_hope_range=Range(0, 255)).build_by_db_get_method().build()

    '''in_img, gt_mask'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_try_segmentation.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d_in_W_and_I_gt_F.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    db_obj = type8_blender_kong_doc3d_in_I_gt_MC.build()
    print(db_obj)
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 448, 448) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    # print("here")
    # img1 = tf_data.train_db_combine.take(10)


    ### 雖然來源相同， 不過兩個iter是獨立運作的
    # db1 = tf_data.train_db_combine
    # db2 = tf_data.train_db_combine

    # iter1 = iter(db1)
    # iter2 = iter(db2)

    # data1 = next(iter1)
    # data2 = next(iter2)
    # plt.figure()
    # plt.imshow(data1[0][1][0][..., :3])
    # plt.figure()
    # plt.imshow(data2[0][1][0][..., :3])
    # plt.show()
    # print("data1 == data2", data1)


    # print(len(img1))
    # print("here~ ", img1)
    # img2 = tf_data.train_db_combine.take(10)
    # print("here~~", img2)
    # ### 有shuffle蒔 從這裡知道 take 不花時間， 有點像定義動作的概念

    # ### 有shuffle蒔 從下面知道， 真正用for去取資料的時候才花時間～ 用for 才是真正去取資料喔！ take()不是， take()只是定義動作而已～
    # plt.figure()
    # for data in img1:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here1")

    # plt.figure()
    # for data in img2:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here2")

    # plt.figure()
    # for data in img2:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here3")
    # it1 = iter(img1)
    # it2 = iter(img2)

    # data = next(it2)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # data = next(it2)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # data = next(it1)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # plt.figure()
    # for data in img1:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here1")

    # plt.figure()
    # it1 = iter(img1)
    # data = next(it1)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])
    # print("here1")

    # i = 0
    # it1 = iter(img1.repeat(2))
    # while(True):
    #     data = next(it1)
    #     i = i + 1
    #     print(i)

    # plt.show()
    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_dis_wc_flow_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    print(time.time() - start_time)
    print("finish")
