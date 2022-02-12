from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import time
start_time = time.time()
###############################################################################################################################################################################################
###############################################################################################################################################################################################
disc_L1_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L1_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=1, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L2_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L2_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=2, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L3_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L3_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=3, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L4_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L4_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=4, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L5_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L5_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=5, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L6_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L6_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=6, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L7_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L7_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=7, out_acti="sigmoid", padding="valid", kernel_size=3)

disc_L8_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L8_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=8, out_acti="sigmoid", padding="valid", kernel_size=3)

'''
discriminator L9 直接完全跑步起來
disc_L9_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
disc_L9_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=9, out_acti="sigmoid", padding="valid", kernel_size=3)
'''
###############################################################################################################################################################################################
###############################################################################################################################################################################################

try_discriminator  = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(D_first_concat=False)

if(__name__ == "__main__"):
    # try_use_model = disc_L1_ch016_sig.build().discriminator  ### (1, 255, 255, 1)  ch128:        4,737    , ch064:      2,369    , ch032:      1,185, ch016:       593, ch008:      297, ch004:      149, ch002:     75, ch001:     38
    # try_use_model = disc_L2_ch016_sig.build().discriminator  ### (1, 127, 127, 1)  ch128:      301,569    , ch064:     77,057    , ch032:     20,097, ch016:     5,441, ch008:    1,569, ch004:      497, ch002:    177, ch001:     71
    # try_use_model = disc_L3_ch016_sig.build().discriminator  ### (1,  63,  63, 1)  ch128:    1,485,057    , ch064:    373,889    , ch032:     94,785, ch016:    24,353, ch008:    6,417, ch004:    1,769, ch002:    525, ch001:    171
    # try_use_model = disc_L4_ch016_sig.build().discriminator  ### (1,  31,  31, 1)  ch128:    6,211,329    , ch064:  1,557,377    , ch032:    391,617, ch016:    99,041, ch008:   25,329, ch004:    6,617, ch002:  1,797, ch001:    521
    # try_use_model = disc_L5_ch016_sig.build().discriminator  ### (1,  15,  15, 1)  ch128:   25,101,057    , ch064:  6,283,649    , ch032:  1,575,105, ch016:   395,873, ch008:  100,017, ch004:   25,529, ch002:  6,645, ch001:  1,793
    # try_use_model = disc_L6_ch016_sig.build().discriminator  ### (1,   7,   7, 1)  ch128:  100,629,249    , ch064: 25,173,377    , ch032:  6,301,377, ch016: 1,579,361, ch008:  396,849, ch004:  100,217, ch002: 25,557, ch001:  6,641
    # try_use_model = disc_L7_ch016_sig.build().discriminator  ### (1,   3,   3, 1)  ch128:  402,680,577(超), ch064:100,701,569    , ch032: 25,191,105, ch016: 6,305,633, ch008:1,580,337, ch004:  397,049, ch002:100,245, ch001: 25,553
    try_use_model = disc_L8_ch016_sig.build().discriminator  ### (1,   1,   1, 1)  ch128:1,610,763,009(超), ch064:402,752,897(超), ch032:100,719,297, ch016:25,195,361, ch008:6,306,609, ch004:1,580,537, ch002:397,077, ch001:100,241
    print("build_model cost time:", time.time() - start_time)

    import numpy as np
    data = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)
    disc_out = try_use_model(data)
    try_use_model.summary()
    print("disc_out:", disc_out.numpy().shape)
