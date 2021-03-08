import subprocess as sb
same_command = ["python", "step10_a_load_and_train_and_test.py"]


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


# sb.run(same_command + ["blender_os_book_flow_unet_bn04.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_bn08.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_bn16.run()"])   ### 127.28
# sb.run(same_command + ["blender_os_book_flow_unet_bn32.run()"])   ### 127.28  ### 失敗
# sb.run(same_command + ["blender_os_book_flow_unet_bn64.run()"])   ### 127.28  ### 失敗



########################################################### 08b2
# os_book_1532_justG_mrf7_k3.run()   ### 128.51
# os_book_1532_justG_mrf5_k3.run()   ### 128.51
# os_book_1532_justG_mrf3_k3.run()   ### 128.48
########################################################### 08b3
# os_book_1532_justG_mrf79_k3.run()  ### 127.246
# os_book_1532_justG_mrf57_k3.run()  ### 127.246
# os_book_1532_justG_mrf35_k3.run()  ### 127.35
########################################################### 08b4
# os_book_1532_justG_mrf_replace5.run()  ### 127.35
# os_book_1532_justG_mrf_replace3.run()  ### 127.48
########################################################### 08b5
# os_book_1532_justG_mrf_replace75.run()  ### 127.55_to127.28
# os_book_1532_justG_mrf_replace35.run()  ### 127.28

########################################################### 08c
# os_book_1532_justG_mrf135_k3.run()  ### 128.246
# os_book_1532_justG_mrf357_k3.run()  ### 127.51
# os_book_1532_justG_mrf3579_k3.run() ### 127.28

########################################################### 08d
# os_book_1532_rect_mrf135_Gk3_DnoC_k4.run() ### 128.246
# os_book_1532_rect_mrf357_Gk3_DnoC_k4.run() ### 127.51
# os_book_1532_rect_mrf3579_Gk3_DnoC_k4.run() ### 127.28
# os_book_1532_rect_mrf35_Gk3_DnoC_k4.run()   ### 127.48

########################################################### 09a Gk4的情況下，D try concat 和 k_size
# os_book_1532_rect_Gk4_D_concat_k3.run()    ### 127.51
# os_book_1532_rect_Gk4_D_no_concat_k4.run() ### 128.246
# os_book_1532_rect_Gk4_D_no_concat_k3.run() ### 127.28

########################################################### 09b Gk3的情況下，D try concat 和 k_size
# os_book_1532_rect_Gk3_D_concat_k4.run()    ###
# os_book_1532_rect_Gk3_D_concat_k3.run()    ###
# os_book_1532_rect_Gk3_D_no_concat_k4.run() ### 127.55
# os_book_1532_rect_Gk3_D_no_concat_k3.run() ### 127.48

########################################################### 10 GAN裡的 G訓練多次有沒有用
# os_book_1532_rect_Gk3_train3_Dk4_no_concat.run() ### 128.246
# os_book_1532_rect_Gk3_train5_Dk4_no_concat.run() ### no machine

########################################################### 11 resblock的add有沒有用
# os_book_1532_Gk3_no_res.run()               ### 127.51
# os_book_1532_Gk3_no_res_D_no_concat.run()   ### 127.28
# os_book_1532_Gk3_no_res_mrf357.run()        ### 128.246

########################################################### 12 resblock用多少個
# os_book_1532_Gk3_resb00.run()  ### 127.48 ###完成
# os_book_1532_Gk3_resb01.run()  ### 127.48
# os_book_1532_Gk3_resb03.run()  ### 127.35
# os_book_1532_Gk3_resb05.run()  ### no
# os_book_1532_Gk3_resb07.run()  ### no
# os_book_1532_Gk3_resb09.run()  ### finish
# os_book_1532_Gk3_resb11.run()  ### 127.55
# os_book_1532_Gk3_resb15.run()  ### 127.28
# os_book_1532_Gk3_resb20.run()  ### 128.244

########################################################### 13 加coord_conv試試看
# os_book_1532_justGk3_coord_conv.run()         ### 127.35
# os_book_1532_justGk3_mrf357_coord_conv.run()  ### 127.28
########################################################### 14
# blender_os_book_flow_unet.run()            ### 127.35  60.5  GB   最低loss:0.0000945
# blender_os_book_flow_unet_epoch050.run()   ### 127.35  05.38 GB   最低loss:0.00035705  total cost time:01:29:33
# blender_os_book_flow_unet_epoch100.run()   ### 127.35  09.72 GB   最低loss:0.00023004  total cost time:02:41:56
# blender_os_book_flow_unet_epoch200.run()   ### 127.35  18.30 GB   最低loss:0.00015143  total cost time:05:45:19
# blender_os_book_flow_unet_epoch300.run()   ### 127.35  27.00 GB   最低loss:0.00012906  total cost time:08:51:23
# blender_os_book_flow_unet_epoch700.run()   ### 127.35  27.00 GB   最低loss:0.00012906  total cost time:08:51:23

########################################################### 14
# blender_os_book_flow_unet_hid_ch_128.run()
# blender_os_book_flow_unet_hid_ch_032.run()
# blender_os_book_flow_unet_hid_ch_016.run()
# blender_os_book_flow_unet_hid_ch_008.run()

########################################################### 14
# blender_os_book_flow_unet_bn04.run()  ##bn04
# blender_os_book_flow_unet_bn08.run()  ##bn08
# blender_os_book_flow_unet_bn16.run()  ##bn16
# blender_os_book_flow_unet_bn32.run()  ##bn32
# blender_os_book_flow_unet_bn64.run()  ##bn64

########################################################### 14
# blender_os_book_flow_unet_epoch002.run()

# tf.keras.backend.clear_session()

# del blender_os_book_flow_unet_epoch002.model_obj.generator
# del blender_os_book_flow_unet_epoch002.model_obj
# del blender_os_book_flow_unet_epoch002

# from numba import cuda
# cuda.select_device(0)
# cuda.cudadrv.driver.Device.reset()
# cuda.cudadrv.driver.Context.reset()
# print(dir(cuda))
# print("close cuda finish")



# print("while_loop~")
# while(True): pass

# blender_os_book_flow_unet_epoch003.run()
# tf.keras.backend.clear_session()
########################################################### 14
# print(blender_os_book_flow_unet_epoch002.run)
# print(blender_os_book_flow_unet_epoch003.run)

# from multiprocessing import Process
# p1 = Process(target=blender_os_book_flow_unet_epoch002.run)
# p1.start()
# p1.join()
# p2 = Process(target=blender_os_book_flow_unet_epoch003.run)
# p2.start()
# p2.join()
pass
