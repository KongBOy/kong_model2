import sobel_k5_s001_6l.step10_a_mask_5_3_sobel_k5_s001_6l as sobel_k5_s001
import sobel_k5_s020_6l.step10_a_mask_5_3_sobel_k5_s020_6l as sobel_k5_s020
import sobel_k5_s040_6l.step10_a_mask_5_3_sobel_k5_s040_6l as sobel_k5_s040
import sobel_k5_s060_6l.step10_a_mask_5_3_sobel_k5_s060_6l as sobel_k5_s060
import sobel_k5_s080_6l.step10_a_mask_5_3_sobel_k5_s080_6l as sobel_k5_s080
import sobel_k5_s100_6l.step10_a_mask_5_3_sobel_k5_s100_6l as sobel_k5_s100

from  sobel_k5_s001_6l.step11 import *
from  sobel_k5_s020_6l.step11 import *
from  sobel_k5_s040_6l.step11 import *
from  sobel_k5_s060_6l.step11 import *
from  sobel_k5_s080_6l.step11 import *
from  sobel_k5_s100_6l.step11 import *

import sobel_k5_s120_260_6l_ch032.step10_a_mask_5_3_sobel_k5_s120_260_6l_ch032 as sobel_k5_s120_260_ch032
####################################################################################################################################
####################################################################################################################################

### 3_ch032-sobel_k5_s1~260
mask_ch032_sobel_k5_s1_260 = [
                            sobel_k5_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                            sobel_k5_s020.mask_h_bg_ch032_sig_L6_ep060.build(),
                            sobel_k5_s040.mask_h_bg_ch032_sig_L6_ep060.build(),
                            sobel_k5_s060.mask_h_bg_ch032_sig_L6_ep060.build(),
                            sobel_k5_s080.mask_h_bg_ch032_sig_L6_ep060.build(),
                            sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s120_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s140_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s160_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s180_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s200_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s220_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s240_L6_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s260_L6_ep060.build(),
                         ]
mask_ch032_sobel_k5_s1_260 = [ exp.result_obj for exp in mask_ch032_sobel_k5_s1_260]
############################################
