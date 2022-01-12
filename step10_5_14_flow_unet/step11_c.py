from step10_a import *
"""
group寫法2：from step10_b1_exp_obj_load_and_train_and_test import * 直接包 exps
補充：無法直接 from step10_a import * 直接處理，
    因為裡面包含太多其他物件了！光要抽出自己想用的 exp物件就是一大工程覺得~
    還有我也不知道要怎麼 直接用 ，也是要一個個 名字 打出來 才能用，名字 都打出來了，不如就直接 包成exps 囉！
"""

### copy的示範
# ch_064_300 = copy.deepcopy(epoch300_bn_see_arg_T.build()); ch_064_300.ana_describe = "flow_unet-ch64_300"
# ch_064_700 = copy.deepcopy(epoch700_bn_see_arg_T.build()); ch_064_700.ana_describe = "flow_unet-ch64_700"

### 0 old vs new shuffle
###    想比較是因為bn==1的情況下 old 和 new shuffle 理論上是一樣的，實際上也是差萬分之幾而已，所以先把old收起來囉，想刪記得只刪see就好
###################################################################################################
### 0_1 epoch old_shuffle
epoch_old_shuf_exps  = [
    epoch050_bn_see_arg_T.build(),
    epoch100_bn_see_arg_T.build(),
    epoch200_bn_see_arg_T.build(),
    epoch300_bn_see_arg_T.build(),
    ### jump epoch500 真的是失誤， 因為無法重現 old shuffle，所以真的無法
    epoch700_bn_see_arg_T.build()]


# ### 0_2 epoch new_shuffle
epoch_new_shuf_exps  = [
    epoch050_new_shuf_bn_see_arg_T.build(),
    epoch100_new_shuf_bn_see_arg_T.build(),
    epoch200_new_shuf_bn_see_arg_T.build(),
    epoch300_new_shuf_bn_see_arg_T.build(),
    epoch500_new_shuf_bn_see_arg_T.build(),
    epoch700_new_shuf_bn_see_arg_T.build()]

### 0_3 epoch old vs new shuffle
epoch_old_new_shuf_exps  = [
    epoch050_bn_see_arg_T.build(),
    epoch050_new_shuf_bn_see_arg_T.build(),
    epoch100_bn_see_arg_T.build(),
    epoch100_new_shuf_bn_see_arg_T.build(),
    epoch200_bn_see_arg_T.build(),
    epoch200_new_shuf_bn_see_arg_T.build(),
    epoch300_bn_see_arg_T.build(),
    epoch300_new_shuf_bn_see_arg_T.build(),
    ### jump epoch500
    ### jump epoch500_new_shuf_bn_see_arg_T ，因為上面jumpg保持統一性，儘管有也jump
    epoch700_bn_see_arg_T.build(),
    epoch700_new_shuf_bn_see_arg_T.build()]

### 0_4 ch old_shuffle
ch_old_shuf_exps = [
    old_ch128_bn_see_arg_T.build(),
    ### jump ch064 真的是失誤， 因為無法重現 old shuffle，所以真的無法
    old_ch032_bn_see_arg_T.build(),
    old_ch016_bn_see_arg_T.build(),
    old_ch008_bn_see_arg_T.build()]

### 0_5 ch new_shuffle
ch064_new_shuf_bn_see_arg_T = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch064_new_shuf_bn_see_arg_T.result_obj.ana_describe = "ch064_new_shuf_bn_see_arg_T"
ch_new_shuf_exps = [
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ch064_new_shuf_bn_see_arg_T,
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build()]

### 0_6 ch  old vs new shuffle
ch_old_new_shuf_exps = [
    old_ch128_bn_see_arg_T.build(),
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ### jump ch064
    ### jump ch064，因為上面jumpg保持統一性，儘管有也jump
    old_ch032_bn_see_arg_T.build(),
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build()]
###################################################################################################
###################################################################################################
###################################################################################################
### 1 epoch
epoch_exps  = [
    epoch050_new_shuf_bn_see_arg_T.build(),
    epoch100_new_shuf_bn_see_arg_T.build(),
    epoch200_new_shuf_bn_see_arg_T.build(),
    epoch300_new_shuf_bn_see_arg_T.build(),
    epoch500_new_shuf_bn_see_arg_T.build(),
    epoch500_new_shuf_bn_see_arg_F.build(),  ### 順便比一個 bn 設錯會怎麼樣
    epoch700_new_shuf_bn_see_arg_T.build(),
    epoch700_bn_see_arg_T_no_down.build(), ]   ### 如果 lr 都不下降 會怎麼樣

epoch020_500_exps  = [
    unet_IN_L7_2to3noC_e020.build(),  # 0
    unet_IN_L7_2to3noC_e040.build(),
    unet_IN_L7_2to3noC_e060.build(),
    unet_IN_L7_2to3noC_e080.build(),
    unet_IN_L7_2to3noC_e100.build(),
    unet_IN_L7_2to3noC_e120.build(),
    unet_IN_L7_2to3noC_e140.build(),  ## 6
    unet_IN_L7_2to3noC_e160.build(),
    unet_IN_L7_2to3noC_e180.build(),
    ch64_in_epoch200.build(),
    ch64_in_epoch220.build(),
    ch64_in_epoch240.build(),
    ch64_in_epoch260.build(),  ### 12
    ch64_in_epoch280.build(),
    ch64_in_epoch300.build(),
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),
    ch64_in_epoch380.build(),  ### 18
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),  ### 24
    ]

epoch100_500_exps  = [
    unet_IN_L7_2to3noC_e100.build(),  # 0
    unet_IN_L7_2to3noC_e120.build(),
    unet_IN_L7_2to3noC_e140.build(),
    unet_IN_L7_2to3noC_e160.build(),
    unet_IN_L7_2to3noC_e180.build(),
    ch64_in_epoch200.build(),
    ch64_in_epoch220.build(),  ###  6
    ch64_in_epoch240.build(),
    ch64_in_epoch260.build(),
    ch64_in_epoch280.build(),
    ch64_in_epoch300.build(),
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),  ### 13
    ch64_in_epoch380.build(),
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),  ### 20
    ]

epoch200_500_exps  = [
    ch64_in_epoch200.build(),
    ch64_in_epoch220.build(),
    ch64_in_epoch240.build(),
    ch64_in_epoch260.build(),
    ch64_in_epoch280.build(),
    ch64_in_epoch300.build(),
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),
    ch64_in_epoch380.build(),
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),
    ]

epoch300_500_exps  = [
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),
    ch64_in_epoch380.build(),
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),
    ]

# # ### 1_2 ch，但 bn_see_arg_F，所以結果圖醜，收起來，留 old_ch032_new_shuf_bn_see_arg_F 是為了給 bn 比較用
# old_ch128_new_shuf_bn_see_arg_F
# ch064_new_shuf_bn_see_arg_F = copy.deepcopy(epoch500_new_shuf_bn_see_arg_F.build()); ch064_new_shuf_bn_see_arg_F.result_obj.ana_describe = "ch064_new_shuf_bn_see_arg_F"
# old_ch032_new_shuf_bn_see_arg_F
# old_ch016_new_shuf_bn_see_arg_F
# old_ch008_new_shuf_bn_see_arg_F
###################################################################################################
###################################################################################################
### 2 ch
ch_exps = [
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ch064_new_shuf_bn_see_arg_T,
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build(),]

### 這 epoch_no_down_vs_ch_exps 真的很重要，
### 理解到了 training loss 在 0.000x 上閜跑是沒有什麼差的，
### 因為 ch008 跟 epoch700_no_down 的 training loss 最後是一樣的！
### 但是 在 real 影像上 是截然不同的喔！ epoch700_no_down 比 ch008 好很多！
### 所以不能只看 training loss，一定要直接看 real影像 處理的效果
epoch_no_down_vs_ch_exps = [
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ch064_new_shuf_bn_see_arg_T,
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build(),
    epoch700_bn_see_arg_T_no_down.build(), ]
###################################################################################################
###################################################################################################
### 3 bn
###   3_1. ch64 只能 bn 1, 4, 8，覺得不夠明顯
ch64_bn01_bn_see_arg_T = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch64_bn01_bn_see_arg_T.result_obj.ana_describe = "ch64_bn01_bn_see_arg_T"
bn_ch64_exps_bn_see_arg_T = [
    ch64_bn01_bn_see_arg_T,
    ch64_bn04_bn_see_arg_T.build(),
    ch64_bn08_bn_see_arg_T.build(), ]

###   3_2. ch32 就能 bn 1, 4, 8, 16
ch32_bn01_bn_see_arg_T = copy.deepcopy(old_ch032_new_shuf_bn_see_arg_T.build()); ch32_bn01_bn_see_arg_T.result_obj.ana_describe = "ch32_bn01_bn_see_arg_T"
bn_ch32_exps_bn_see_arg_T = [
    ch32_bn01_bn_see_arg_T,
    old_ch32_bn04_bn_see_arg_T.build(),
    old_ch32_bn08_bn_see_arg_T.build(),
    old_ch32_bn16_bn_see_arg_T.build(), ]

###   3_3. ch32 bn1, 4, 8, 16 see_arg 設 True/False 來比較看看，ch64的 bn數比較少就跳過囉~~
ch64_bn01_bn_see_arg_F = copy.deepcopy(epoch500_new_shuf_bn_see_arg_F.build()); ch64_bn01_bn_see_arg_F.result_obj.ana_describe = "ch64_bn01_bn_see_arg_F"
bn_ch64_exps_bn_see_arg_F_and_T = [
    ch64_bn01_bn_see_arg_F,
    ch64_bn01_bn_see_arg_T,
    ch64_bn04_bn_see_arg_F.build(),
    ch64_bn04_bn_see_arg_T.build(),
    ch64_bn08_bn_see_arg_F.build(),
    ch64_bn08_bn_see_arg_T.build(), ]

###   3_4. ch32 bn1, 4, 8, 16 see_arg 設 True/False 來比較看看，ch64的 bn數比較少就跳過囉~~
ch32_bn01_bn_see_arg_F = copy.deepcopy(old_ch032_new_shuf_bn_see_arg_F.build()); ch32_bn01_bn_see_arg_F.result_obj.ana_describe = "ch32_bn01_bn_see_arg_F"
bn_ch32_exps_bn_see_arg_F_and_T = [
    ch32_bn01_bn_see_arg_F,
    ch32_bn01_bn_see_arg_T,
    old_ch32_bn04_bn_see_arg_F.build(),
    old_ch32_bn04_bn_see_arg_T.build(),
    old_ch32_bn08_bn_see_arg_F.build(),
    old_ch32_bn08_bn_see_arg_T.build(),
    old_ch32_bn16_bn_see_arg_F.build(),
    old_ch32_bn16_bn_see_arg_T.build(), ]

###################################################################################################
###################################################################################################
### 4 bn_in
###   4_1. in 的 batch_size一定只能等於1 所以拿 epoch500 來比較，也像看train 久一點的效果，所以就多train 一個 epoch700 的 並拿相應的 bn來比較
###   不管 epoch500, 700 都是 in 比 bn 好！
ch64_bn_epoch500 = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch64_bn_epoch500.result_obj.ana_describe = "ch64_bn_epoch500"
ch64_bn_epoch700 = copy.deepcopy(epoch700_new_shuf_bn_see_arg_T).build(); ch64_bn_epoch700.result_obj.ana_describe = "ch64_bn_epoch700"
bn_in_size1_exps = [
    ch64_bn_epoch500,
    ch64_in_epoch500.build(),
    ch64_bn_epoch700,
    ch64_in_epoch700.build(),
]


### 4_2. in vs bn batch_size > 1
###   batch_size 越大，效果越差
ch64_1_bn01 = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch64_1_bn01.result_obj.ana_describe = "ch64_1_bn01"
ch64_2_in01 = copy.deepcopy(ch64_in_epoch500.build());               ch64_2_in01.result_obj.ana_describe = "ch64_2_in01"
ch64_3_bn04 = copy.deepcopy(ch64_bn04_bn_see_arg_T.build());         ch64_3_bn04.result_obj.ana_describe = "ch64_3_bn04"
ch64_4_bn08 = copy.deepcopy(ch64_bn08_bn_see_arg_T.build());         ch64_4_bn08.result_obj.ana_describe = "ch64_4_bn08"
bn_in_sizen_exps = [
    ch64_1_bn01,
    ch64_2_in01,
    ch64_3_bn04,
    ch64_4_bn08,
]
###################################################################################################
###################################################################################################
### 5 unet concat Activation vs concat BN，都是epoch500 且 lr有下降， 先不管 concatA loss 相當於表現差的哪種結果 只放兩個exp比較
###    concat_A 的效果比較好，且從架構圖上已經看出來 確實 concat_A 較合理
ch64_in_concat_B = copy.deepcopy(ch64_in_epoch500.build()); ch64_in_concat_B.result_obj.ana_describe = "ch64_in_concat_B"
in_concat_AB = [
    ch64_in_concat_A.build(),
    ch64_in_concat_B,
]

###################################################################################################
###################################################################################################
### 6 unet level 2~7
###   想看看 差一層 差多少，先不管 8_layer 表現好 的 相當於 哪種結果
###   最後發現 層越少效果越差，但 第8層 效果跟 第7層 差不多， 沒辦法試 第9層 因為要512的倍數
unet_layers = [
    unet_L2.build(),
    unet_L3.build(),
    unet_L4.build(),
    unet_L5.build(),
    unet_L6.build(),
    unet_L7.build(),
    unet_L8.build(),
]


###################################################################################################
### 7a_1 unet 的concat 改成 add 的效果如何，效果超差
unet_skip_use_add = [
    unet_L8_skip_use_add.build(),
    unet_L7_skip_use_add.build(),
    unet_L6_skip_use_add.build(),
    unet_L5_skip_use_add.build(),
    unet_L4_skip_use_add.build(),
    unet_L3_skip_use_add.build(),
    unet_L2_skip_use_add.build(),
]

### 7a_2 unet 的concat vs add 的效果如何，concat好，add不好
unet_skip_use_concat_vs_add = [
    unet_L7.build(),
    unet_L7_skip_use_add.build(),
    unet_L6.build(),
    unet_L6_skip_use_add.build(),
    unet_L5.build(),
    unet_L5_skip_use_add.build(),
]


### 7b unet 的 第一層不concat 來跟 前面還不錯的結果比較
###    train loss 在 全接~前兩個skip省略 表現差不多，
###    see來看的話 train/test 都差不多，但在 real 前兩個skip 的結果 看起來都比 全接好！ 且 覺得 2to3noC 比 2to2noC 更好些！
###    2to4noC 在 real3 表現差，且 2to4noC 之後 邊緣 的部分就越做越差囉～
unet_IN_L7_all_C_ch64_in_epoch500 = copy.deepcopy(ch64_in_epoch500.build()); unet_IN_L7_all_C_ch64_in_epoch500.result_obj.ana_describe = "1a-unet_IN_L7_all_C_ch64_in_epoch500"  ### 當初的train_code沒寫好沒有存到 model用的 code
# unet_IN_L7_all_C_ch64_bn_epoch500 = copy.deepcopy(ch64_bn_epoch500.build()); unet_IN_L7_all_C_ch64_bn_epoch500.result_obj.ana_describe = "1b-unet_IN_L7_all_C_ch64_bn_epoch500"  ### 從4_1就知道bn沒有in好，所以就不用這個了
# unet_IN_L7_all_C_unet_L7          = copy.deepcopy(unet_L7.build()); unet_IN_L7_all_C_unet_L7.result_obj.ana_describe = "1c-unet_IN_L7_all_C_unet_L7"  ### 他的loss好像最低，但沒有train完
unet_IN_L7_2to2noC       .build().result_obj.ana_describe = "2a-unet_IN_L7_2to2noC"
# unet_IN_L7_2to2noC_ch32.build().result_obj.ana_describe = "2b-unet_IN_L7_2to2noC_ch32"  ### ch32 效果比較沒那麼好，先註解跳過不看
unet_IN_L7_2to3noC       .build().result_obj.ana_describe = "3-unet_IN_L7_2to3noC"
unet_IN_L7_2to4noC       .build().result_obj.ana_describe = "4-unet_IN_L7_2to4noC"
unet_IN_L7_2to5noC       .build().result_obj.ana_describe = "5-unet_IN_L7_2to5noC"
unet_IN_L7_2to6noC       .build().result_obj.ana_describe = "6-unet_IN_L7_2to6noC"
unet_IN_L7_2to7noC       .build().result_obj.ana_describe = "7-unet_IN_L7_2to7noC"
# unet_IN_L7_2to8noC               .result_obj.ana_describe = "8-unet_IN_L7_2to8noC"   ### 當初訓練就怪怪的，先跳過！

unet_skip_noC = [
    unet_IN_L7_all_C_ch64_in_epoch500,
    # unet_IN_L7_all_C_ch64_bn_epoch500,
    # unet_IN_L7_all_C_unet_L7,
    unet_IN_L7_2to2noC,
    # unet_IN_L7_2to2noC_ch32,
    unet_IN_L7_2to3noC,
    unet_IN_L7_2to4noC,
    unet_IN_L7_2to5noC,
    unet_IN_L7_2to6noC,
    unet_IN_L7_2to7noC,
    # unet_IN_L7_2to8noC,   ### 好像train壞掉怪怪的
]


### 7c.看看 UNet 的 skip 學 印度方法 看看skip connection 中間加 cnn 的效果
unet_IN_L7_skip_clean = copy.deepcopy(ch64_in_epoch500.build()); unet_IN_L7_skip_clean.result_obj.ana_describe = "1-unet_IN_L7_skip_clean"  ### 當初的train_code沒寫好沒有存到 model用的 code
unet_IN_L7_skip_use_cnn1_NO_relu     .build().result_obj.ana_describe = "2-unet_IN_L7_skip_use_cnn1_NO_relu"
unet_IN_L7_skip_use_cnn1_USErelu     .build().result_obj.ana_describe = "3a-unet_IN_L7_skip_use_cnn1_USErelu"
unet_IN_L7_skip_use_cnn1_USEsigmoid  .build().result_obj.ana_describe = "3b-unet_IN_L7_skip_use_cnn1_USEsigmoid"
unet_IN_L7_skip_use_cnn3_USErelu     .build().result_obj.ana_describe = "4a-unet_IN_L7_skip_use_cnn3_USErelu"
unet_IN_L7_skip_use_cnn3_USEsigmoid  .build().result_obj.ana_describe = "4b-unet_IN_L7_skip_use_cnn3_USEsigmoid"

unet_skip_use_cnn = [
    unet_IN_L7_skip_clean,
    unet_IN_L7_skip_use_cnn1_NO_relu,
    unet_IN_L7_skip_use_cnn1_USErelu,
    unet_IN_L7_skip_use_cnn1_USEsigmoid,
    unet_IN_L7_skip_use_cnn3_USErelu,
    unet_IN_L7_skip_use_cnn3_USEsigmoid,
]


### 7d.看看 UNet 的 skip 用 cSE/ sSE/ csSE 試試看
unet_L7_skip_SE = [
    ch64_in_epoch060          .build(),
    ch64_in_sk_cSE_e060_wrong .build(),
    ch64_in_sk_sSE_e060       .build(),
    ch64_in_sk_scSE_e060_wrong.build(),
]

ch64_2to3noC_sk_no_e060         = copy.deepcopy(unet_IN_L7_2to3noC_e060.build()); ch64_2to3noC_sk_no_e060         .result_obj.ana_describe = "1-ch64_2to3noC_sk_no_e060"  ### 當初的train_code沒寫好沒有存到 model用的 code
ch64_2to3noC_sk_cSE_e060_wrong  = ch64_2to3noC_sk_cSE_e060_wrong .build();        ch64_2to3noC_sk_cSE_e060_wrong  .result_obj.ana_describe = "2-ch64_2to3noC_sk_cSE_e060_wrong"
ch64_2to3noC_sk_sSE_e060        = ch64_2to3noC_sk_sSE_e060       .build();        ch64_2to3noC_sk_sSE_e060        .result_obj.ana_describe = "3-ch64_2to3noC_sk_sSE_e060"
ch64_2to3noC_sk_scSE_e060_wrong = ch64_2to3noC_sk_scSE_e060_wrong.build();        ch64_2to3noC_sk_scSE_e060_wrong .result_obj.ana_describe = "4-ch64_2to3noC_sk_scSE_e060_wrong"
unet_L7_2to3noC_skip_SE = [
    ch64_2to3noC_sk_no_e060,
    ch64_2to3noC_sk_cSE_e060_wrong,
    ch64_2to3noC_sk_sSE_e060,
    ch64_2to3noC_sk_scSE_e060_wrong,
]

unet_L7_skip_SE_ep = [
    ch64_in_sk_cSE_e060_wrong.build(),
    ch64_in_sk_cSE_e100_wrong.build(),
    ch64_in_sk_sSE_e060.build(),
    ch64_in_sk_sSE_e100.build(),
    ch64_in_sk_scSE_e060_wrong.build(),
    ch64_in_sk_scSE_e100_wrong.build(),

    ch64_2to3noC_sk_cSE_e060_wrong,
    ch64_2to3noC_sk_cSE_e100_wrong.build(),
    ch64_2to3noC_sk_sSE_e060,
    ch64_2to3noC_sk_sSE_e100.build(),
    ch64_2to3noC_sk_scSE_e060_wrong,
    ch64_2to3noC_sk_scSE_e100_wrong.build(),
]

###################################################################################################
###################################################################################################
### 8_1 unet 的 range(mae)
unet_range_mae = [
    t1_in_01_mo_th_gt_01_mae.build(),
    t2_in_01_mo_01_gt_01_mae.build(),
    t3_in_01_mo_th_gt_th_mae.build(),
    t4_in_01_mo_01_gt_th_mae.build(),
    t5_in_th_mo_th_gt_01_mae.build(),
    t6_in_th_mo_01_gt_01_mae.build(),
    t7_in_th_mo_th_gt_th_mae.build(),
    t8_in_th_mo_01_gt_th_mae.build(),
]

unet_range_mae_good = [
    t1_in_01_mo_th_gt_01_mae.build(),
    t2_in_01_mo_01_gt_01_mae.build(),
    t5_in_th_mo_th_gt_01_mae.build(),
    t6_in_th_mo_01_gt_01_mae.build(),
]

unet_range_mae_ok = [
    t3_in_01_mo_th_gt_th_mae.build(),
    t7_in_th_mo_th_gt_th_mae.build(),
]

unet_range_mse = [
    t1_in_01_mo_th_gt_01_mse.build(),
    t2_in_01_mo_01_gt_01_mse.build(),
    t3_in_01_mo_th_gt_th_mse.build(),
    t4_in_01_mo_01_gt_th_mse.build(),
    t5_in_th_mo_th_gt_01_mse.build(),
    t6_in_th_mo_01_gt_01_mse.build(),
    t7_in_th_mo_th_gt_th_mse.build(),
    t8_in_th_mo_01_gt_th_mse.build(),
]

########################################################################################################################
###########################################################################
### 另一個架構
rect_layers_wrong_lrelu = [
    rect_2_level_fk3.build(),
    rect_3_level_fk3.build(),
    rect_4_level_fk3.build(),
    rect_5_level_fk3.build(),
    rect_6_level_fk3.build(),
    rect_7_level_fk3.build(),
]

rect_layers_right_relu = [
    rect_2_level_fk3_ReLU.build(),
    rect_3_level_fk3_ReLU.build(),
    rect_4_level_fk3_ReLU.build(),
    rect_5_level_fk3_ReLU.build(),
    rect_6_level_fk3_ReLU.build(),
    rect_7_level_fk3_ReLU.build(),
]

rect_fk3_ch64_tfIN_resb_ok9_epoch500
rect_7_level_fk7

########################################################################################################################