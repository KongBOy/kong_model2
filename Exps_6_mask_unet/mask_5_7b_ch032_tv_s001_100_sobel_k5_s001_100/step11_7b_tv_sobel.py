import step10_a as tv_s001_100_sobel_k5_s001_100

import numpy as np
####################################################################################################################################
####################################################################################################################################
### 5b_ch032-tv_s001_100_sobel_k5_s001_100
mask_ch032_tv_s001_100_sobel_k5_s001_100 = np.array([
          [tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s001.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s020.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s040.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s060.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s080.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s001_sobel_k5_s100.build().result_obj, ],
          [tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s001.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s020.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s040.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s060.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s080.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s020_sobel_k5_s100.build().result_obj, ],
          [tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s001.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s020.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s040.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s060.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s080.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s040_sobel_k5_s100.build().result_obj, ],
          [tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s001.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s020.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s040.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s060.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s080.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s060_sobel_k5_s100.build().result_obj, ],
          [tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s001.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s020.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s040.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s060.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s080.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s080_sobel_k5_s100.build().result_obj, ],
          [tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s001.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s020.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s040.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s060.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s080.build().result_obj,
           tv_s001_100_sobel_k5_s001_100.mask_h_bg_ch032_sig_L6_ep060_tv_s100_sobel_k5_s100.build().result_obj, ],
])
