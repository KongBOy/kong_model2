import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a_load_and_train_and_test.py"]
run = "build().run()"
compress_all = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_1   = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=1 , add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_2te = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=1, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_3te = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=2, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_4te = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=3, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_8te = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=4, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_9te = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=5, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"
compress_10te = "build().result_obj.save_all_single_see_as_matplot_bm_rec_visual(start_index=6, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True, print_msg=True)"

# sb.run(same_command + [f"test2.{compress_1}"])

# ### hid_ch=64, 來測試 epoch系列 ##############################
# sb.run(same_command + [f"epoch050_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch100_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch200_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch300_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch700_bn_see_arg_T.{run}"])

# sb.run(same_command + [f"ch128_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"ch032_bn_see_arg_T.{compress_3te}"])
# sb.run(same_command + [f"ch016_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"ch008_bn_see_arg_T.{compress_all}"])

# sb.run(same_command + [f"epoch050_new_shuf_bn_see_arg_T.{compress_all}"])  ### 802
# sb.run(same_command + [f"epoch100_new_shuf_bn_see_arg_T.{compress_all}"])  ### 1275
# sb.run(same_command + [f"epoch200_new_shuf_bn_see_arg_T.{compress_all}"])  ### 1309
# sb.run(same_command + [f"epoch300_new_shuf_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"epoch500_new_shuf_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"epoch500_new_shuf_bn_see_arg_F.{compress_all}"])
# sb.run(same_command + [f"epoch700_bn_see_arg_T_no_down.{compress_4te}"])  ### 看看 lr 都不下降的效果
# sb.run(same_command + [f"epoch700_new_shuf_bn_see_arg_T.{compress_9te}"])

# sb.run(same_command + [f"ch128_new_shuf_bn_see_arg_F.{compress_all}"])
# sb.run(same_command + [f"ch032_new_shuf_bn_see_arg_F.{compress_all}"])
# sb.run(same_command + [f"ch016_new_shuf_bn_see_arg_F.{compress_all}"])
# sb.run(same_command + [f"ch008_new_shuf_bn_see_arg_F.{compress_all}"])

# sb.run(same_command + [f"ch128_new_shuf_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"ch032_new_shuf_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"ch016_new_shuf_bn_see_arg_T.{compress_all}"])
# sb.run(same_command + [f"ch008_new_shuf_bn_see_arg_T.{compress_all}"])


# sb.run(same_command + [f"ch64_bn04_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"ch64_bn04_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"ch64_bn08_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"ch64_bn08_bn_see_arg_T.{run}"])


# sb.run(same_command + [f"ch32_bn04_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"ch32_bn08_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"ch32_bn16_bn_see_arg_T.{run}"])
# ### sb.run(same_command + [f"blender_os_book_flow_unet_ch32_bn32.{run}"])  ### 失敗
# ### sb.run(same_command + [f"blender_os_book_flow_unet_ch32_bn64.{run}"])  ### 失敗


# sb.run(same_command + [f"ch32_bn04_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"ch32_bn08_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"ch32_bn16_bn_see_arg_F.{run}"])


# sb.run(same_command + [f"ch64_in_epoch320.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch340.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch360.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch380.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch400.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch420.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch440.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch460.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch480.{compress_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch500.{run}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch700.{run}"])  ### 測試真的IN

# ######################
# sb.run(same_command + [f"ch64_in_concat_A.{run}"])  ### 看看Activation 完再concat的效果

# sb.run(same_command + [f"unet_2l.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_3l.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_4l.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_5l.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_6l.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_7l.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_8l.{compress_all}"])  ### 看看Activation 完再concat的效果

# sb.run(same_command + [f"unet_8l_skip_use_add.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_7l_skip_use_add.{compress_1}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_6l_skip_use_add.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_5l_skip_use_add.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_4l_skip_use_add.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_3l_skip_use_add.{compress_all}"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + [f"unet_2l_skip_use_add.{compress_all}"])  ### 看看Activation 完再concat的效果

#############################################################################################
### 8
### 之前 都跑錯的 mse 喔！將錯就錯， 下面重跑 正確的 mae 試試看 順便比較 mse/mae的效果
# sb.run(same_command + [f"t2_in_01_mo_01_gt_01_mse.{compress_all}"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + [f"t1_in_01_mo_th_gt_01_mse.{compress_all}"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + [f"t7_in_th_mo_th_gt_th_mse.{compress_all}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t3_in_01_mo_th_gt_th_mse.{compress_all}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t5_in_th_mo_th_gt_01_mse.{compress_all}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t6_in_th_mo_01_gt_01_mse.{compress_all}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t4_in_01_mo_01_gt_th_mse.{compress_all}"])  ### 圖確認過覺得怪怪的 , 不想重train
# sb.run(same_command + [f"t8_in_th_mo_01_gt_th_mse.{compress_all}"])  ### 已確認錯了, 隨然沒 train完，但前面的175 epoch 也幾乎一樣囉！, 不想重train


### 上面 應該是 沒改到loss所以才用mse，現在改用mae試試看
# sb.run(same_command + [f"t2_in_01_mo_01_gt_01_mae.{compress_all}"])  ### ok
# sb.run(same_command + [f"t1_in_01_mo_th_gt_01_mae.{compress_all}"])  ### 127.28 ok
# sb.run(same_command + [f"t3_in_01_mo_th_gt_th_mae.{compress_all}"])  ### 127.28 ok
# sb.run(same_command + [f"t5_in_th_mo_th_gt_01_mae.{compress_all}"])  ### 127.28 ok
# sb.run(same_command + [f"t6_in_th_mo_01_gt_01_mae.{compress_all}"])  ### 127.28 ok
# sb.run(same_command + [f"t7_in_th_mo_th_gt_th_mae.{compress_all}"])  ### 127.28 ok
# sb.run(same_command + [f"t4_in_01_mo_01_gt_th_mae.{compress_2te}"])  ### 127.35 ok
# sb.run(same_command + [f"t8_in_th_mo_01_gt_th_mae.{compress_all}"])  ### 127.35 沒train完
#############################################################################################
### 9 的東西
# sb.run(same_command + [f"unet_IN_7l_2to2noC     .{run}"])
# sb.run(same_command + [f"unet_IN_7l_2to2noC_ch32.{run}"])
# sb.run(same_command + [f"unet_IN_7l_2to3noC     .{run}"])  ### 3254
# sb.run(same_command + [f"unet_IN_7l_2to4noC     .{run}"])
# sb.run(same_command + [f"unet_IN_7l_2to5noC     .{run}"])
# sb.run(same_command + [f"unet_IN_7l_2to6noC     .{run}"])  ### 3073
# sb.run(same_command + [f"unet_IN_7l_2to7noC     .{run}"])  ### 2851
# sb.run(same_command + [f"unet_IN_7l_2to8noC     .{run}"])  ### 2920


#############################################################################################
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_NO_relu.{run}"])   ### 127.35
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_USErelu.{run}"])   ### 127.28
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn3_USErelu.{run}"])   ### 127.28

# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_USEsigmoid.{run}"])   ### 127.35
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn3_USEsigmoid.{run}"])   ### 127.28


#######################
# sb.run(same_command + [f"rect_fk3_ch64_tfIN_resb_ok9_epoch500.{run}"])
# sb.run(same_command + [f"rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.{run}"])

# sb.run(same_command + [f"rect_2_level_fk3.{run}"])
# sb.run(same_command + [f"rect_3_level_fk3.{run}"])
# sb.run(same_command + [f"rect_4_level_fk3.{run}"])
# sb.run(same_command + [f"rect_5_level_fk3.{run}"])
# sb.run(same_command + [f"rect_6_level_fk3.{run}"])
# sb.run(same_command + [f"rect_7_level_fk3.{run}"])

# sb.run(same_command + [f"rect_2_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_3_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_4_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_5_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_6_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_7_level_fk3_ReLU.{run}"])  ### 127.28跑
