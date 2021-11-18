from step09_c_train_step import *
from step09_d_KModel_builder import KModel_builder, MODEL_NAME

import time
start_time = time.time()
### 直接先建好 obj 給外面import囉！
unet                     = KModel_builder().set_model_name(MODEL_NAME.unet ).build_unet().set_train_step(train_step_first)
#######################################################################################################################
rect                     = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=False).set_train_step(train_step_GAN)   ### G 只train 1次
rect_firstk3             = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True ).set_train_step(train_step_GAN)   ### G 只train 1次
rect_Gk4_many_Dk4_concat = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=False).set_train_step(train_step_GAN2)  ### G 可train 多次， 目前G_train幾次要手動改喔！

rect_mrfall         = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use1=True, use3=True, use5=True, use7=True, use9=True).set_train_step(train_step_GAN)  ### G 只train 1次
rect_mrf7           = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use7=True) .set_train_step(train_step_GAN)  ### G 只train 1次
rect_mrf79          = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True).set_train_step(train_step_GAN)  ### G 只train 1次
rect_replace_mrf7   = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=True , use7=True).set_train_step(train_step_GAN)  ### G 只train 1次
rect_replace_mrf79  = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=False, mrf_replace=True , use7=True, use9=True).set_train_step(train_step_GAN)  ### G 只train 1次
#######################################################################################################################
justG               = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=False).set_train_step(train_step_pure_G)
justG_firstk3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True ).set_train_step(train_step_pure_G)
########################################################### 2
justG_mrf7          = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=False, use7=True).set_train_step(train_step_pure_G)
justG_mrf7_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use7=True).set_train_step(train_step_pure_G)
justG_mrf5_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use5=True).set_train_step(train_step_pure_G)
justG_mrf3_k3       = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True).set_train_step(train_step_pure_G)
########################################################### 3
justG_mrf79         = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=False, use7=True, use9=True).set_train_step(train_step_pure_G)
justG_mrf79_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use7=True, use9=True).set_train_step(train_step_pure_G)
justG_mrf57_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use5=True, use7=True).set_train_step(train_step_pure_G)
justG_mrf35_k3      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True).set_train_step(train_step_pure_G)
########################################################### 4
justG_mrf_replace7  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True).set_train_step(train_step_pure_G)
justG_mrf_replace5  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use5=True).set_train_step(train_step_pure_G)
justG_mrf_replace3  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use3=True).set_train_step(train_step_pure_G)
########################################################### 5
justG_mrf_replace79 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use9=True).set_train_step(train_step_pure_G)
justG_mrf_replace75 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use7=True, use5=True).set_train_step(train_step_pure_G)
justG_mrf_replace35 = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=False, mrf_replace=True, use3=True, use5=True).set_train_step(train_step_pure_G)
########################################################### 2c
justG_mrf135_k3     = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True).set_train_step(train_step_pure_G)
justG_mrf357_k3     = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True).set_train_step(train_step_pure_G)
justG_mrf3579_k3    = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True).set_train_step(train_step_pure_G)

rect_mrf35_Gk3_DnoC_k4     = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
rect_mrf135_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use1=True, use3=True, use5=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
rect_mrf357_Gk3_DnoC_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
rect_mrf3579_Gk3_DnoC_k4   = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2_mrf(first_k3=True , mrf_replace=False, use3=True, use5=True, use7=True, use9=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_pure_G)
########################################################### 9a
# rect_D_concat_k4    = "rect_D_concat_k4" ### 原始版本
rect_Gk4_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=True , D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk4_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk4_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(D_first_concat=False, D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
########################################################### 9b
rect_Gk3_D_concat_k4       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=True , D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk3_D_concat_k3       = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=True , D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk3_D_no_concat_k4    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次
rect_Gk3_D_no_concat_k3    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=3).set_train_step(train_step_GAN)  ### G 只train 1次
########################################################### 10
rect_Gk3_train3_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN2)  ### G 可train 多次， 目前G_train幾次要手動改喔！
rect_Gk3_train5_Dk4_no_concat    = KModel_builder().set_model_name(MODEL_NAME.rect ).use_rect2(first_k3=True, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN2)  ### G 可train 多次， 目前G_train幾次要手動改喔！
########################################################### 11
justG_fk3_no_res             = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=False).set_train_step(train_step_pure_G)  ### 127.51
rect_fk3_no_res_D_no_concat  = KModel_builder().set_model_name(MODEL_NAME.rect  ).use_rect2(first_k3=True, use_res_learning=False, D_first_concat=False, D_kernel_size=4).set_train_step(train_step_GAN)  ### G 只train 1次  ### 127.28
justG_fk3_no_res_mrf357      = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True, mrf_replace=False, use_res_learning=False, use3=True, use5=True, use7=True).set_train_step(train_step_pure_G)   ### 128.246
########################################################### 12
Gk3_resb00  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=0).set_train_step(train_step_pure_G)   ### 127.48
Gk3_resb01  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=1).set_train_step(train_step_pure_G)   ### 127.35
Gk3_resb03  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=3).set_train_step(train_step_pure_G)   ### 127.55
Gk3_resb05  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=5).set_train_step(train_step_pure_G)   ### 128.246
Gk3_resb07  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=7).set_train_step(train_step_pure_G)   ### 127.28
Gk3_resb09  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=9).set_train_step(train_step_pure_G)   ### 127.51
Gk3_resb11  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=11).set_train_step(train_step_pure_G)   ### 127.51
Gk3_resb15  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=15).set_train_step(train_step_pure_G)   ### 127.28
Gk3_resb20  = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG(first_k3=True, use_res_learning=True, resb_num=20).set_train_step(train_step_pure_G)   ### 127.51

########################################################### 13 加coord_conv試試看
justGk3_coord_conv        = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG    (first_k3=True, coord_conv=True).set_train_step(train_step_pure_G)
justGk3_mrf357_coord_conv = KModel_builder().set_model_name(MODEL_NAME.justG ).use_justG_mrf(first_k3=True, coord_conv=True, mrf_replace=False, use3=True, use5=True, use7=True).set_train_step(train_step_pure_G)


###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 14 快接近IN了
flow_unet       = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

### 在 git commit bc25e 之前都是 old ch 喔！ 最大都是 32*8=256, 16*8=128, 8*8=64 而已， 而128*8=1024 又有點太大， new_ch 就是根據層 做 2**layer，最大取512 囉！
### 如果想回復的話，要用 git 回復到 bc25e 或 之前的版本囉！
flow_unet_old_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_old_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_old_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_old_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 8 , out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 測試 subprocess
flow_unet_epoch2 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=4, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_epoch3 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=6, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_epoch4 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=8, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 真的IN
flow_unet_IN_ch64 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)


flow_unet_IN_new_ch128 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=128, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch032 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 32, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch016 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch= 16, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch008 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  8, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_new_ch004 = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=  4, true_IN=True).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 真的IN，跟DewarpNet一樣 CNN 不用 bias
flow_unet_IN_ch64_cnnNoBias = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, use_bias=False).use_flow_unet().set_train_step(train_step_pure_G)


########################################################### 14 看 concat Activation 有沒有差
flow_unet_ch64_in_concat_A = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, out_ch=3, concat_Activation=True).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 看 不同level 的效果
flow_unet_2_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_3_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_4_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_5_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_6_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_7_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_8_level  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
########################################################### 14 看 unet 的 concat 改成 + 會有什麼影響
flow_unet_8_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=8, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_7_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=7, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_6_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=6, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_5_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=5, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_4_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=4, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_3_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=3, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_2_level_skip_use_add  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, depth_level=2, true_IN=True, skip_use_add=True, out_ch=3).use_flow_unet().set_train_step(train_step_pure_G)

########################################################### 14 看 unet 的 output 改成sigmoid
flow_unet_IN_ch64_sigmoid  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, out_tanh=False, true_IN=True).use_flow_unet()
########################################################### 14 看 unet 的 第一層試試看 不 concat 效果如何
flow_unet_IN_L7_ch64_2to2noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch32_2to2noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=32, true_IN=True, no_concat_layer=2).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to3noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=3).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to4noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=4).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to5noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=5).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to6noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=6).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to7noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=7).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to8noC = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=8).use_flow_unet().set_train_step(train_step_pure_G)
########################################################### 14 看 unet 的 skip 中間接 cnn 的效果
flow_unet_IN_L7_ch64_skip_use_cnn1_NO_relu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=None).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_skip_use_cnn1_USErelu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=tf.nn.relu).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_skip_use_cnn1_USEsigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=1, skip_use_Acti=tf.nn.sigmoid).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_skip_use_cnn3_USErelu    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=3, skip_use_Acti=tf.nn.relu).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_skip_use_cnn3_USEsigmoid = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cnn=True, skip_cnn_k=3, skip_use_Acti=tf.nn.sigmoid).use_flow_unet().set_train_step(train_step_pure_G)


########################################################### 14 看 unet 的 skip 中間接 cSE, sSE, csSE 的效果
flow_unet_IN_L7_ch64_2to3noC_sk_cSE  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=3, skip_use_cSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to3noC_sk_sSE  = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2, skip_use_sSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_2to3noC_sk_scSE = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, no_concat_layer=2, skip_use_scSE=True).use_flow_unet().set_train_step(train_step_pure_G)

flow_unet_IN_L7_ch64_skip_use_cSE    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_cSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_skip_use_sSE    = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_sSE=True).use_flow_unet().set_train_step(train_step_pure_G)
flow_unet_IN_L7_ch64_skip_use_scSE   = KModel_builder().set_model_name(MODEL_NAME.flow_unet).set_unet(hid_ch=64, true_IN=True, skip_use_scSE=True).use_flow_unet().set_train_step(train_step_pure_G)

###############################################################################################################################################################################################
###############################################################################################################################################################################################
########################################################### 15 用 resblock 來試試看
flow_rect_fk3_ch64_tfIN_resb_ok9 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect(first_k3=True, hid_ch=64, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

flow_rect_7_level_fk7 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=7, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

flow_rect_2_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_3_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_4_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_5_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_6_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_7_level_fk3 = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)

flow_rect_2_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=2, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_3_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=3, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_4_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=4, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_5_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=5, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_6_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=6, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)
flow_rect_7_level_fk3_ReLU = KModel_builder().set_model_name(MODEL_NAME.flow_rect).use_flow_rect_7_level(first_k=3, hid_ch=64, depth_level=7, true_IN=True, use_ReLU=True, use_res_learning=True, resb_num=9, out_ch=3).set_train_step(train_step_pure_G)



if(__name__ == "__main__"):
    # print(flow_rect_2_level_fk3.build())
    # print(mask_unet_ch032_tanh_L7.build())
    print(flow_rect_7_level_fk3_ReLU.build())
    print("build_model cost time:", time.time() - start_time)
    pass
