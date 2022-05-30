import sys
sys.path.append("kong_util")
from kong_util.build_dataset_combine import build_datasets

#####################################################################################################################################################
# dst_db_dir = "D:/Users/user/Desktop/db/"  ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！
### 1.建立給 unet 用的 pad2000-512to256，但先不要執行喔！因為目前用的是手動複製的，然後忘記當初怎麼複製的ˊ口ˋ 咪挺完看能不能就先 生成新的DB然後再改囉！
# build_datasets(src_in_dir   = dst_db_dir + "step3_apply_flow_h=384,w=256_old_page",            ### 好像已經刪掉了
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_dir   = dst_db_dir+"step2_build_flow_h=384,w=256_old_page/move_maps",  ### 好像已經刪掉了
#                src_gt_word  = ".npy",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_old_page_1_pure_unet",
#                db_in_name   = "dis_imgs",
#                db_gt_name   = "move_maps")


# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=256,w=256_complex",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_dir   = dst_db_dir+"step2_build_flow_h=256,w=256_complex/move_maps",
#                src_gt_word  = ".npy",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=256,w=256_complex_1_pure_unet",
#                db_in_name   = "dis_imgs",
#                db_gt_name   = "move_maps")


# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_dir   = dst_db_dir+"step2_build_flow_h=384,w=256_complex/move_maps",
#                src_gt_word  = ".npy",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_complex_1_pure_unet",
#                db_in_name   = "dis_imgs",
#                db_gt_name   = "move_maps")


### 記得 complex+page做完需要手動去 挑 complex：0~1799 和 page：2000~3499 當train， complex：1800~1999 和 page：3500~3559
# build_datasets(src_in_dir    = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page",
#                src_in_word   = "3a1-I1-patch.bmp",
#                src_gt_dir    = dst_db_dir+"step2_build_flow_h=384,w=256_complex+page/move_maps",
#                src_gt_word   = ".npy",
#                dst_db_dir    = dst_db_dir,
#                db_name       = "h=384,w=256_complex+page_1_pure_unet",
#                db_in_name    = "dis_imgs",
#                db_gt_name    = "move_maps")

### 記得 complex+page_more_like做完需要手動去 挑 complex：0~1799 和 page：2000~2179 當train， complex：1800~1999 和 page：2180~2199
# build_datasets(src_in_dir    = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page_more_like",
#                src_in_word   = "3a1-I1-patch.bmp",
#                src_gt_dir    = dst_db_dir+"step2_build_flow_h=384,w=256_complex+page_more_like/move_maps",
#                src_gt_word   = ".npy",
#                dst_db_dir    = dst_db_dir,
#                db_name       = "h=384,w=256_complex+page_more_like_1_pure_unet",
#                db_in_name    = "dis_imgs",
#                db_gt_name    = "move_maps")


# build_datasets(src_in_dir    = dst_db_dir + "step3_apply_flow_h=384,w=256_smooth-curl+fold_and_page",
#                src_gt_dir    = dst_db_dir + "step2_build_flow_h=384,w=256_smooth-curl+fold_and_page/move_maps",
#                src_in_word   = "3a1-I1-patch.bmp",
#                src_gt_word   = ".npy" ,
#                dst_db_dir    = dst_db_dir,
#                db_name       = "h=384,w=256_smooth-curl+fold_and_page_1_pure_unet",
#                db_in_name    = "dis_imgs",
#                db_gt_name    = "move_maps",
#                train_amount  = 1350)



#####################################################################################################################################################
## 2.建立給 rect2 用的 pure_rect2
# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_old_page",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_old_page",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_word  = "4-gt_ord_pad.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_old_page_2_pure_rect2",
#                db_in_name   = "dis_img_db",
#                db_gt_name   = "gt_ord_pad_img_db")

# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_word  = "4-gt_ord_pad.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=256,w=256_complex_2_pure_rect2",
#                db_in_name   = "dis_img_db",
#                db_gt_name   = "gt_ord_pad_img_db")


# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_word  = "4-gt_ord_pad.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_complex_2_pure_rect2",
#                db_in_name   = "dis_img_db",
#                db_gt_name   = "gt_ord_pad_img_db")

### 記得 complex+page做完需要手動去 挑 complex：0~1799 和 page：2000~3499 當train， complex：1800~1999 和 page：3500~3559
# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_word  = "4-gt_ord_pad.bmp",
#                db_name      = "h=384,w=256_complex+page_2_pure_rect2",
#                db_in_name   = "dis_img_db",
#                db_gt_name   = "gt_ord_pad_img_db")

### 記得 complex+page_more_like做完需要手動去 挑 complex：0~1799 和 page：2000~2179 當train， complex：1800~1999 和 page：2180~2199
# build_datasets(src_in_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page_more_like",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page_more_like",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_word  = "4-gt_ord_pad.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_complex+page_more_like_2_pure_rect2",
#                db_in_name   = "dis_img_db",
#                db_gt_name   = "gt_ord_pad_img_db")


# build_datasets(src_in_dir   = dst_db_dir + "step3_apply_flow_h=384,w=256_smooth-curl+fold_and_page",
#                src_gt_dir   = dst_db_dir + "step3_apply_flow_h=384,w=256_smooth-curl+fold_and_page",
#                src_in_word  = "3a1-I1-patch.bmp",
#                src_gt_word  = "4-gt_ord_pad.bmp" ,
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_smooth-curl+fold_and_page_2_pure_rect2",
#                db_in_name   = "dis_img_db",
#                db_gt_name   = "gt_ord_pad_img_db",
#                train_amount = 1350)


#####################################################################################################################################################
### 3.建立給 unet+rect2 用的 rect2_2000
# build_datasets(src_in_dir   = dst_db_dir+"result/20200316-114012_1_page_h=384,w=256_model2_UNet_512to256_127.28_finish/test_1_unet_page_h=384,w=256",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_page",
#                src_in_word  = "unet_rec_img.bmp",
#                src_gt_word  = "1-I.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_old_page_3_unet_rect2",
#                db_in_name   = "unet_rec_img_db",
#                db_gt_name   = "gt_ord_img_db")


# build_datasets(src_in_dir   = dst_db_dir+"result/20200328-170738_1_pure_unet_complex_h=256,w=256_model2_UNet_512to256_finish/test_indicate_1_pure_unet_complex_h=256,w=256",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_complex",
#                src_in_word  = "g_rec_img.bmp",
#                src_gt_word  = "1-I.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=256,w=256_complex_3_unet_rect2",
#                db_in_name   = "unet_rec_img_db",
#                db_gt_name   = "gt_ord_img_db")



# build_datasets(src_in_dir   = dst_db_dir+"result/h=384,w=256_20200329-215628_1_pure_unet_complex_h=384,w=256_model2_UNet_512to256/test_indicate_1_pure_unet_complex+page_h=384,w=256",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex",
#                src_in_word  = "g_rec_img.bmp",
#                src_gt_word  = "1-I.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_complex_3_unet_rect2",
#                db_in_name   = "unet_rec_img_db",
#                db_gt_name   = "gt_ord_img_db")


### 記得 complex+page做完需要手動去 挑 complex：0~1799 和 page：2000~3499 當train， complex：1800~1999 和 page：3500~3559
# build_datasets(src_in_dir   = dst_db_dir+"result/h=384,w=256_complex+page_1_pure_unet_20200329-232144_model2_UNet_512to256_finish/test_indicate_h=384,w=256_complex+page_1_pure_unet",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page",
#                src_in_word  = "g_rec_img.bmp",
#                src_gt_word  = "1-I.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_complex+page_3_unet_rect2",
#                db_in_name   = "unet_rec_img_db",
#                db_gt_name   = "gt_ord_img_db")


### 記得 complex+page_more_like做完需要手動去 挑 complex：0~1799 和 page：2000~2179 當train， complex：1800~1999 和 page：2180~2199
# build_datasets(src_in_dir   = dst_db_dir+"result/h=384,w=256_complex+page_more_like_1_pure_unet_20200406-214854_model2_UNet_512to256/test_indicate_h=384,w=256_complex+page_more_like_1_pure_unet",
#                src_gt_dir   = dst_db_dir+"step3_apply_flow_h=384,w=256_complex+page_more_like",
#                src_in_word  = "g_rec_img.bmp",
#                src_gt_word  = "1-I.bmp",
#                dst_db_dir   = dst_db_dir,
#                db_name      = "h=384,w=256_complex+page_more_like_3_unet_rect2",
#                db_in_name   = "unet_rec_img_db",
#                db_gt_name   = "gt_ord_img_db")


# from kong_util.util import  get_maxmin_train_move_from_path, get_max_db_move_xy
# db_name = "1_pure_unet2000-512to256_index"
# max_train_move, min_train_move = get_maxmin_train_move_from_path(dst_db_dir+"datasets"+"/"+db_name+"/"+"train"+"/"+"move_maps")
# np.save(dst_db_dir+"datasets"+"/"+db_name+"/"+"max_train_move",max_train_move)
# np.save(dst_db_dir+"datasets"+"/"+db_name+"/"+"min_train_move",min_train_move)

# db_name = "1_pure_unet_page_h=384,w=256"
# max_db_move_x, max_db_move_y = get_max_db_move_xy(db_dir=dst_db_dir+"datasets", db_name=db_name)
# np.save(dst_db_dir+"datasets"+"/"+db_name+"/"+"max_db_move_x",max_db_move_x)
# np.save(dst_db_dir+"datasets"+"/"+db_name+"/"+"max_db_move_y",max_db_move_y)
