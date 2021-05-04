"""
補充：無法直接 from step10_a import * 直接處理，
    因為裡面包含太多其他物件了！光要抽出自己想用的 exp物件就是一大工程覺得~
    還有我也不知道要怎麼 直接用 ，也是要一個個 名字 打出來 才能用，名字 都打出來了，不如就直接 包成exps 囉！
"""
from step10_a_load_and_train_and_test import *
### 不小心放到copy的東西也沒關係，只是相同的 result 會被執行兩次而已~~
rec_bm_exps_compress = [
    unet_IN_7l_skip_use_cnn1_NO_relu,
    # testest,

    # unet_IN_7l_2to2noC,       ### 3491
    # unet_IN_7l_2to2noC_ch32,  ### 2985
    # unet_IN_7l_2to3noC,       ### 2157

    # unet_IN_7l_2to4noC,
    # unet_IN_7l_2to5noC,
    # unet_IN_7l_2to6noC,   ### 2194
    # unet_IN_7l_2to7noC,
    # unet_IN_7l_2to8noC,   ### 3597

    # t1_in_01_mo_th_gt_01_mse,  ### 4964
    # t2_in_01_mo_01_gt_01_mse,  ### 3577
    # t3_in_01_mo_th_gt_th_mse,
    # t4_in_01_mo_01_gt_th_mse,
    # t5_in_th_mo_th_gt_01_mse,
    # t6_in_th_mo_01_gt_01_mse,
    # t7_in_th_mo_th_gt_th_mse,  ### 3632
    # t8_in_th_mo_01_gt_th_mse,

    # t1_in_01_mo_th_gt_01_mae,
    # t2_in_01_mo_01_gt_01_mae,
    # t3_in_01_mo_th_gt_th_mae,
    # t4_in_01_mo_01_gt_th_mae,
    # t5_in_th_mo_th_gt_01_mae,
    # t6_in_th_mo_01_gt_01_mae,
    # t7_in_th_mo_th_gt_th_mae,
    # t8_in_th_mo_01_gt_th_mae,

    # unet_8l,

    # unet_8l_skip_use_add,
    # unet_7l_skip_use_add,
    # unet_6l_skip_use_add,
    # unet_5l_skip_use_add,
    # unet_4l_skip_use_add,
    # unet_3l_skip_use_add,
    # unet_2l_skip_use_add,

    # rect_fk3_ch64_tfIN_resb_ok9_epoch500,
    # rect_7_level_fk7,
    # rect_2_level_fk3,
    # rect_3_level_fk3,
    # rect_4_level_fk3,
    # rect_5_level_fk3,
    # rect_6_level_fk3,
    # rect_7_level_fk3,

    # rect_2_level_fk3_ReLU,
    # rect_3_level_fk3_ReLU,
    # rect_4_level_fk3_ReLU,
    # rect_5_level_fk3_ReLU,
    # rect_6_level_fk3_ReLU,
    # rect_7_level_fk3_ReLU,


    # epoch700_bn_see_arg_T_no_down,

    # unet_2l,
    # unet_3l,
    # unet_4l,
    # unet_5l,
    # unet_6l,
    # unet_7l,

    # concat_A,

    # epoch050_bn_see_arg_T.result_obj,
    # epoch100_bn_see_arg_T.result_obj,
    # epoch200_bn_see_arg_T.result_obj,
    # epoch300_bn_see_arg_T.result_obj,
    # epoch700_bn_see_arg_T.result_obj,

    # ch128_bn_see_arg_T.result_obj,
    # ch032_bn_see_arg_T.result_obj,
    # ch016_bn_see_arg_T.result_obj,
    # ch008_bn_see_arg_T.result_obj,

    # epoch050_new_shuf_bn_see_arg_T.result_obj,
    # epoch100_new_shuf_bn_see_arg_T.result_obj,
    # epoch200_new_shuf_bn_see_arg_T.result_obj,
    # epoch300_new_shuf_bn_see_arg_T.result_obj,
    # epoch500_new_shuf_bn_see_arg_T.result_obj,
    # epoch500_new_shuf_bn_see_arg_F.result_obj,
    # epoch700_new_shuf_bn_see_arg_T.result_obj,

    # ch128_new_shuf_bn_see_arg_F.result_obj,
    # ch032_new_shuf_bn_see_arg_F.result_obj,
    # ch016_new_shuf_bn_see_arg_F.result_obj,
    # ch008_new_shuf_bn_see_arg_F.result_obj,

    # ch128_new_shuf_bn_see_arg_T.result_obj,
    # ch032_new_shuf_bn_see_arg_T.result_obj,
    # ch016_new_shuf_bn_see_arg_T.result_obj,
    # ch008_new_shuf_bn_see_arg_T.result_obj,

    # ch64_bn04_bn_see_arg_F.result_obj,
    # ch64_bn04_bn_see_arg_T.result_obj,
    # ch64_bn08_bn_see_arg_F.result_obj,
    # ch64_bn08_bn_see_arg_T.result_obj,

    # ch32_bn04_bn_see_arg_T.result_obj,
    # ch32_bn08_bn_see_arg_T.result_obj,
    # ch32_bn16_bn_see_arg_T.result_obj,

    # ch32_bn04_bn_see_arg_F.result_obj,
    # ch32_bn08_bn_see_arg_F.result_obj,
    # ch32_bn16_bn_see_arg_F.result_obj,

    # ch64_in_epoch500.result_obj,
    # ch64_in_epoch700.result_obj,
]

### matplot_visual   ### 覺得我也不會看所以就省了ˊ口ˋ
# for result in compress_results:
#     result.save_all_single_see_as_matplot_visual_multiprocess(add_loss=True)

if(__name__ == "__main__"):
    ### bm, rec，會順便把 npy 轉 npz 喔！
    for go_exp, rec_bm_exp in enumerate(rec_bm_exps_compress):
        if(go_exp == 0):  ### 這個 if 是給 中途不小心中斷，可以從某個指定的 see_num 開始做，要手動把中斷的放第一個result拉
            # rec_bm_exp.result_obj.save_all_single_see_as_matplot_bm_rec_visual_multiprocess(add_loss=False, bgr2rgb=True, print_msg=True)
            rec_bm_exp.result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=0, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=1, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=2, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=3, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=4, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=5, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_exp.result_obj.save_single_see_as_matplot_bm_rec_visual(see_num=6, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
        else:   ### 中斷的作完，剩下就是作全部囉！
            rec_bm_exp.result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)


        # rec_bm_exp.result_obj.save_all_single_see_as_matplot_bm_rec_visual_at_final_epoch(add_loss=False, bgr2rgb=True)
