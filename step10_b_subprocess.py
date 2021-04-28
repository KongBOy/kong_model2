import subprocess as sb
same_command = ["python", "step10_a_load_and_train_and_test.py"]

# ### hid_ch=64, 來測試 epoch系列 ##############################
# sb.run(same_command + ["epoch050_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch100_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch200_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch300_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch700_bn_see_arg_T.run()"])

# sb.run(same_command + ["ch128_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch032_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch016_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch008_bn_see_arg_T.run()"])

# sb.run(same_command + ["epoch050_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch100_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch200_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch300_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch500_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch500_new_shuf_bn_see_arg_F.run()"])
# sb.run(same_command + ["epoch700_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["epoch700_bn_see_arg_T_no_down.run()"])  ### 看看 lr 都不下降的效果

# sb.run(same_command + ["ch128_new_shuf_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch032_new_shuf_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch016_new_shuf_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch008_new_shuf_bn_see_arg_F.run()"])

# sb.run(same_command + ["ch128_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch032_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch016_new_shuf_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch008_new_shuf_bn_see_arg_T.run()"])


# sb.run(same_command + ["ch64_bn04_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch64_bn04_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch64_bn08_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch64_bn08_bn_see_arg_T.run()"])


# sb.run(same_command + ["ch32_bn04_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch32_bn08_bn_see_arg_T.run()"])
# sb.run(same_command + ["ch32_bn16_bn_see_arg_T.run()"])
# ### sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn32.run()"])  ### 失敗
# ### sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn64.run()"])  ### 失敗


# sb.run(same_command + ["ch32_bn04_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch32_bn08_bn_see_arg_F.run()"])
# sb.run(same_command + ["ch32_bn16_bn_see_arg_F.run()"])


# sb.run(same_command + ["ch64_in_epoch500.run()"])  ### 測試真的IN
# sb.run(same_command + ["ch64_in_epoch700.run()"])  ### 測試真的IN

# ######################
# sb.run(same_command + ["ch64_in_concat_A.run()"])  ### 看看Activation 完再concat的效果

# sb.run(same_command + ["unet_2l.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_3l.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_4l.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_5l.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_6l.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_7l.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_8l.run()"])  ### 看看Activation 完再concat的效果

# sb.run(same_command + ["unet_8l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_7l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_6l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_5l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_4l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_3l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["unet_2l_skip_use_add.run()"])  ### 看看Activation 完再concat的效果

#############################################################################################
### 8
### 之前 都跑錯的 mse 喔！將錯就錯， 下面重跑 正確的 mae 試試看 順便比較 mse/mae的效果
# sb.run(same_command + ["t2_in_01_mo_01_gt_01_mse.run()"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + ["t1_in_01_mo_th_gt_01_mse.run()"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + ["t7_in_th_mo_th_gt_th_mse.run()"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + ["t3_in_01_mo_th_gt_th_mse.run()"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + ["t5_in_th_mo_th_gt_01_mse.run()"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + ["t6_in_th_mo_01_gt_01_mse.run()"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + ["t4_in_01_mo_01_gt_th_mse.run()"])  ### 圖確認過覺得怪怪的 , 不想重train
# sb.run(same_command + ["t8_in_th_mo_01_gt_th_mse.run()"])  ### 已確認錯了, 隨然沒 train完，但前面的175 epoch 也幾乎一樣囉！, 不想重train


### 上面 應該是 沒改到loss所以才用mse，現在改用mae試試看
# sb.run(same_command + ["t2_in_01_mo_01_gt_01_mae.run()"])  ### ok
# sb.run(same_command + ["t1_in_01_mo_th_gt_01_mae.run()"])  ### 127.28 ok
# sb.run(same_command + ["t3_in_01_mo_th_gt_th_mae.run()"])  ### 127.28 ok
# sb.run(same_command + ["t5_in_th_mo_th_gt_01_mae.run()"])  ### 127.28 ok
# sb.run(same_command + ["t6_in_th_mo_01_gt_01_mae.run()"])  ### 127.28 ok
# sb.run(same_command + ["t7_in_th_mo_th_gt_th_mae.run()"])  ### 127.28 ok
# sb.run(same_command + ["t4_in_01_mo_01_gt_th_mae.run()"])  ### 127.35 ok
# sb.run(same_command + ["t8_in_th_mo_01_gt_th_mae.run()"])  ### 127.35 沒train完
#############################################################################################
### 9 的東西
# sb.run(same_command + ["unet_IN_7l_firstnoC.run()"])





#######################
# sb.run(same_command + ["rect_fk3_ch64_tfIN_resb_ok9_epoch500.run()"])
# sb.run(same_command + ["rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.run()"])

# sb.run(same_command + ["rect_2_level_fk3.run()"])
# sb.run(same_command + ["rect_3_level_fk3.run()"])
# sb.run(same_command + ["rect_4_level_fk3.run()"])
# sb.run(same_command + ["rect_5_level_fk3.run()"])
# sb.run(same_command + ["rect_6_level_fk3.run()"])
# sb.run(same_command + ["rect_7_level_fk3.run()"])

# sb.run(same_command + ["rect_2_level_fk3_ReLU.run()"])  ### 127.28跑
# sb.run(same_command + ["rect_3_level_fk3_ReLU.run()"])  ### 127.28跑
# sb.run(same_command + ["rect_4_level_fk3_ReLU.run()"])  ### 127.28跑
# sb.run(same_command + ["rect_5_level_fk3_ReLU.run()"])  ### 127.28跑
# sb.run(same_command + ["rect_6_level_fk3_ReLU.run()"])  ### 127.28跑
# sb.run(same_command + ["rect_7_level_fk3_ReLU.run()"])  ### 127.28跑
