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

#######################
# sb.run(same_command + ["epoch700_bn_see_arg_T_no_down.run()"])  ### 看看 lr 都不下降的效果
# sb.run(same_command + ["concat_A.run()"])  ### 看看Activation 完再concat的效果

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

# 01_01_01 已跑完 500epoch 有lr下降
# 01_th_th 已跑完 500epoch 有lr下降
# 以下先跑 200epoch 無lr下降
# sb.run(same_command + ["in_th_mo_th_gt_th.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["in_01_mo_th_gt_th.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["in_01_mo_01_gt_th.run()"])  ### 看看Activation 完再concat的效果
# sb.run(same_command + ["in_th_mo_th_gt_01.run()"])  ### 看看Activation 完再concat的效果
sb.run(same_command + ["in_th_mo_01_gt_01.run()"])  ### 看看Activation 完再concat的效果
sb.run(same_command + ["in_th_mo_01_gt_th.run()"])  ### 看看Activation 完再concat的效果



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
