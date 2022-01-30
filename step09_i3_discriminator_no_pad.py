from step09_d_KModel_builder_combine_step789 import KModel_builder, MODEL_NAME

import time
start_time = time.time()
###############################################################################################################################################################################################
###############################################################################################################################################################################################
disc_L1_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=1, out_acti="sigmoid", padding="valid")
disc_L1_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=1, out_acti="sigmoid", padding="valid")

disc_L2_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=2, out_acti="sigmoid", padding="valid")
disc_L2_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=2, out_acti="sigmoid", padding="valid")

disc_L3_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=3, out_acti="sigmoid", padding="valid")
disc_L3_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=3, out_acti="sigmoid", padding="valid")

disc_L4_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=4, out_acti="sigmoid", padding="valid")
disc_L4_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=4, out_acti="sigmoid", padding="valid")

disc_L5_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=5, out_acti="sigmoid", padding="valid")
disc_L5_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=5, out_acti="sigmoid", padding="valid")

disc_L6_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=6, out_acti="sigmoid", padding="valid")
disc_L6_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=6, out_acti="sigmoid", padding="valid")

disc_L7_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=7, out_acti="sigmoid", padding="valid")
disc_L7_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=7, out_acti="sigmoid", padding="valid")

'''
discriminator L8 output shape 為 1, 0, 0, 1
disc_L8_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=8, out_acti="sigmoid", padding="valid")
disc_L8_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=8, out_acti="sigmoid", padding="valid")

discriminator L9 直接完全跑步起來
disc_L9_ch128_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=128, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch064_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 64, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch032_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 32, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch016_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch= 16, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch008_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  8, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch004_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  4, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch002_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  2, depth_level=9, out_acti="sigmoid", padding="valid")
disc_L9_ch001_sig = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(hid_ch=  1, depth_level=9, out_acti="sigmoid", padding="valid")
'''
###############################################################################################################################################################################################
###############################################################################################################################################################################################

try_discriminator  = KModel_builder().set_model_name(MODEL_NAME.discriminator).set_discriminator(D_first_concat=False)

if(__name__ == "__main__"):
    # try_use_model = disc_L5_ch064_sig.build().discriminator
    try_use_model = disc_L7_ch001_sig.build().discriminator
    print("build_model cost time:", time.time() - start_time)

    import numpy as np
    data = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)
    disc_out = try_use_model(data)
    try_use_model.summary()
    print("disc_out:", disc_out.numpy().shape)
