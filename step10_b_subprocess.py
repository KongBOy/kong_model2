import subprocess as sb
same_command = ["python", "step10_a_load_and_train_and_test.py"]

### hid_ch=64, 來測試 epoch系列 ##############################
sb.run(same_command + ["blender_os_book_flow_unet_epoch050.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_epoch100.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_epoch200.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_epoch300.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_epoch500.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_epoch700.run()"])   ### 127.35  ### 正在弄 new_shuf

sb.run(same_command + ["blender_os_book_flow_unet_hid_ch_008.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_hid_ch_016.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_hid_ch_032.run()"])   ### 127.35  ### 正在弄 new_shuf
sb.run(same_command + ["blender_os_book_flow_unet_hid_ch_128.run()"])   ### 127.35  ### 正在弄 new_shuf


# sb.run(same_command + ["blender_os_book_flow_unet_bn04.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_bn08.run()"])   ### 127.28


# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn04.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn08.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn16.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn32.run()"])   ### 127.28  ### 失敗
# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn64.run()"])   ### 127.28  ### 失敗


# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn04_set_arg_ok.run()"])   ### 127.35  ### 正在弄 new_shuf
# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn08_set_arg_ok.run()"])   ### 127.35  ### 正在弄 new_shuf
# sb.run(same_command + ["blender_os_book_flow_unet_ch32_bn16_set_arg_ok.run()"])   ### 127.35  ### 正在弄 new_shuf


# sb.run(same_command + ["blender_os_book_flow_unet_IN_epoch500.run()"])   ### 127.28  ### 測試真的IN
# sb.run(same_command + ["blender_os_book_flow_unet_IN_epoch700.run()"])   ### 127.28  ### 測試真的IN
