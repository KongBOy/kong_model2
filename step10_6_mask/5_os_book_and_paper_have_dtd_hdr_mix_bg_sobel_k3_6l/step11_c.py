from step10_a import *
"""
group寫法2：from step10_a_load_and_train_and_test import * 直接包 exps
補充：無法直接 from step10_a import * 直接處理，
    因為裡面包含太多其他物件了！光要抽出自己想用的 exp物件就是一大工程覺得~
    還有我也不知道要怎麼 直接用 ，也是要一個個 名字 打出來 才能用，名字 都打出來了，不如就直接 包成exps 囉！
"""

### copy的示範
# ch_064_300 = copy.deepcopy(epoch300_bn_see_arg_T.build()); ch_064_300.ana_describe = "flow_unet-ch64_300"
# ch_064_700 = copy.deepcopy(epoch700_bn_see_arg_T.build()); ch_064_700.ana_describe = "flow_unet-ch64_700"

### 大概怎麼包的示範
# ch64_2to3noC_sk_no_e060         = copy.deepcopy(unet_IN_7l_2to3noC_e060.build()); ch64_2to3noC_sk_no_e060         .result_obj.ana_describe = "1-ch64_2to3noC_sk_no_e060"  ### 當初的train_code沒寫好沒有存到 model用的 code
# ch64_2to3noC_sk_cSE_e060_wrong  = ch64_2to3noC_sk_cSE_e060_wrong .build();        ch64_2to3noC_sk_cSE_e060_wrong  .result_obj.ana_describe = "2-ch64_2to3noC_sk_cSE_e060_wrong"
# ch64_2to3noC_sk_sSE_e060        = ch64_2to3noC_sk_sSE_e060       .build();        ch64_2to3noC_sk_sSE_e060        .result_obj.ana_describe = "3-ch64_2to3noC_sk_sSE_e060"
# ch64_2to3noC_sk_scSE_e060_wrong = ch64_2to3noC_sk_scSE_e060_wrong.build();        ch64_2to3noC_sk_scSE_e060_wrong .result_obj.ana_describe = "4-ch64_2to3noC_sk_scSE_e060_wrong"
# unet_7l_2to3noC_skip_SE = [
#     ch64_2to3noC_sk_no_e060,
#     ch64_2to3noC_sk_cSE_e060_wrong,
#     ch64_2to3noC_sk_sSE_e060,
#     ch64_2to3noC_sk_scSE_e060_wrong,
# ]