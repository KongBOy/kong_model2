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
sb.run(same_command + ["rect_fk3_ch64_tfIN_resb_ok9_epoch500.run()"])
sb.run(same_command + ["rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.run()"])