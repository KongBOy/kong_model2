import mask_5_5_bce_sobel_k3_6l     .step10_a as bce_sobel_k3
import mask_5_5_bce_sobel_k5_6l     .step10_a as bce_sobel_k5_s001
import mask_5_5_bce_sobel_k7_6l     .step10_a as bce_sobel_k7

import mask_5_5_bce_sobel_k5_s20_6l .step10_a as bce_sobel_k5_s020
import mask_5_5_bce_sobel_k5_s40_6l .step10_a as bce_sobel_k5_s040
import mask_5_5_bce_sobel_k5_s60_6l .step10_a as bce_sobel_k5_s060
import mask_5_5_bce_sobel_k5_s80_6l .step10_a as bce_sobel_k5_s080
import mask_5_5_bce_sobel_k5_s100_6l.step10_a as bce_sobel_k5_s100
import mask_5_5_bce_sobel_k5_s120_6l.step10_a as bce_sobel_k5_s120
import mask_5_5_bce_sobel_k5_s140_6l.step10_a as bce_sobel_k5_s140
import mask_5_5_bce_sobel_k5_s160_6l.step10_a as bce_sobel_k5_s160


####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k3-1_ch
mask_L6_bce_sobel_k3_ch = [bce_sobel_k3.mask_h_bg_ch128_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch064_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch032_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch016_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch008_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch004_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch002_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k3_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k3_ch]
############################################
### 5-6l_bce_sobel_k3-2_ep
mask_L6_bce_sobel_k3_ep = [bce_sobel_k3.mask_h_bg_ch128_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch064_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch032_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch016_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch008_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch004_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch002_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch001_sig_L6_ep060.build(),
                           bce_sobel_k3.mask_h_bg_ch128_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch064_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch032_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch016_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch008_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch004_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch002_sig_L6_ep200.build(),
                           bce_sobel_k3.mask_h_bg_ch001_sig_L6_ep200.build()]
mask_L6_bce_sobel_k3_ep = [ exp.result_obj for exp in mask_L6_bce_sobel_k3_ep]
############################################
### 5-6l_bce_sobel_k3-4_no-concat_and_add
mask_L6_bce_sobel_k3_noC_and_add = [bce_sobel_k3.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                    bce_sobel_k3.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                    bce_sobel_k3.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                    bce_sobel_k3.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                    bce_sobel_k3.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                    bce_sobel_k3.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k3_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k3_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5-1_ch
mask_L6_bce_sobel_k5_s001_ch = [bce_sobel_k5_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s001_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s001_ch]
############################################
### 5-6l_bce_sobel_k5-2_ep
mask_L6_bce_sobel_k5_ep = [bce_sobel_k5_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch001_sig_L6_ep060.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch128_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch064_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch032_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch016_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch008_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch004_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch002_sig_L6_ep200.build(),
                           bce_sobel_k5_s001.mask_h_bg_ch001_sig_L6_ep200.build()]
mask_L6_bce_sobel_k5_ep = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_ep]
############################################
### 5-6l_bce_sobel_k5-4_no-concat_and_add
mask_L6_bce_sobel_k5_noC_and_add = [bce_sobel_k5_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                    bce_sobel_k5_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                    bce_sobel_k5_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                    bce_sobel_k5_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                    bce_sobel_k5_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                    bce_sobel_k5_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k5_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k7-1_ch
mask_L6_bce_sobel_k7_ch = [bce_sobel_k7.mask_h_bg_ch128_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch064_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch032_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch016_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch008_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch004_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch002_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k7_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k7_ch]
############################################
### 5-6l_bce_sobel_k7-2_ep
mask_L6_bce_sobel_k7_ep = [bce_sobel_k7.mask_h_bg_ch128_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch064_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch032_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch016_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch008_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch004_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch002_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch001_sig_L6_ep060.build(),
                           bce_sobel_k7.mask_h_bg_ch128_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch064_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch032_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch016_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch008_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch004_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch002_sig_L6_ep200.build(),
                           bce_sobel_k7.mask_h_bg_ch001_sig_L6_ep200.build()]
mask_L6_bce_sobel_k7_ep = [ exp.result_obj for exp in mask_L6_bce_sobel_k7_ep]
############################################
### 5-6l_bce_sobel_k7-4_no-concat_and_add
mask_L6_bce_sobel_k7_noC_and_add = [bce_sobel_k7.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                    bce_sobel_k7.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                    bce_sobel_k7.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                    bce_sobel_k7.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                    bce_sobel_k7.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                    bce_sobel_k7.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k7_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k7_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s20-1_ch
mask_L6_bce_sobel_k5_s020_ch = [bce_sobel_k5_s020.mask_h_bg_ch128_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch064_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch032_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch016_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch008_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch004_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch002_sig_L6_ep060.build(),
                               bce_sobel_k5_s020.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s020_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s020_ch]
############################################
### 5-6l_bce_sobel_k5_s20_no-concat_and_add
mask_L6_bce_sobel_k5_s20_noC_and_add = [bce_sobel_k5_s020.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                        bce_sobel_k5_s020.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                        bce_sobel_k5_s020.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                        bce_sobel_k5_s020.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                        bce_sobel_k5_s020.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                        bce_sobel_k5_s020.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k5_s20_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s20_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s40-1_ch
mask_L6_bce_sobel_k5_s040_ch = [bce_sobel_k5_s040.mask_h_bg_ch128_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch064_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch032_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch016_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch008_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch004_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch002_sig_L6_ep060.build(),
                               bce_sobel_k5_s040.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s040_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s040_ch]
############################################
### 5-6l_bce_sobel_k5_s40_no-concat_and_add
mask_L6_bce_sobel_k5_s40_noC_and_add = [bce_sobel_k5_s040.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                        bce_sobel_k5_s040.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                        bce_sobel_k5_s040.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                        bce_sobel_k5_s040.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                        bce_sobel_k5_s040.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                        bce_sobel_k5_s040.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k5_s40_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s40_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s60-1_ch
mask_L6_bce_sobel_k5_s060_ch = [bce_sobel_k5_s060.mask_h_bg_ch128_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch064_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch032_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch016_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch008_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch004_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch002_sig_L6_ep060.build(),
                               bce_sobel_k5_s060.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s060_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s060_ch]
############################################
### 5-6l_bce_sobel_k5_s60_no-concat_and_add
mask_L6_bce_sobel_k5_s60_noC_and_add = [bce_sobel_k5_s060.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                        bce_sobel_k5_s060.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                        bce_sobel_k5_s060.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                        bce_sobel_k5_s060.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                        bce_sobel_k5_s060.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                        bce_sobel_k5_s060.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k5_s60_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s60_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s80-1_ch
mask_L6_bce_sobel_k5_s080_ch = [bce_sobel_k5_s080.mask_h_bg_ch128_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch064_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch032_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch016_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch008_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch004_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch002_sig_L6_ep060.build(),
                               bce_sobel_k5_s080.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s080_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s080_ch]
############################################
### 5-6l_bce_sobel_k5_s80_no-concat_and_add
mask_L6_bce_sobel_k5_s80_noC_and_add = [bce_sobel_k5_s080.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                        bce_sobel_k5_s080.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                        bce_sobel_k5_s080.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                        bce_sobel_k5_s080.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                        bce_sobel_k5_s080.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                        bce_sobel_k5_s080.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k5_s80_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s80_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s100-1_ch
mask_L6_bce_sobel_k5_s100_ch = [bce_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
                                bce_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s100_ch]
############################################
### 5-6l_bce_sobel_k5_s100_no-concat_and_add
mask_L6_bce_sobel_k5_s100_noC_and_add = [bce_sobel_k5_s100.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                         bce_sobel_k5_s100.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                         bce_sobel_k5_s100.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                         bce_sobel_k5_s100.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                         bce_sobel_k5_s100.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                         bce_sobel_k5_s100.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s100_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s120-1_ch
mask_L6_bce_sobel_k5_s120_ch = [bce_sobel_k5_s120.mask_h_bg_ch128_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch064_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch032_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch016_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch008_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch004_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch002_sig_L6_ep060.build(),
                                bce_sobel_k5_s120.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s120_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s120_ch]
############################################
### 5-6l_bce_sobel_k5_s120_no-concat_and_add
mask_L6_bce_sobel_k5_s120_noC_and_add = [bce_sobel_k5_s120.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                         bce_sobel_k5_s120.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                         bce_sobel_k5_s120.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                         bce_sobel_k5_s120.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                         bce_sobel_k5_s120.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                         bce_sobel_k5_s120.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s140-1_ch
mask_L6_bce_sobel_k5_s140_ch = [bce_sobel_k5_s140.mask_h_bg_ch128_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch064_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch032_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch016_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch008_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch004_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch002_sig_L6_ep060.build(),
                                bce_sobel_k5_s140.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s140_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s140_ch]
############################################
### 5-6l_bce_sobel_k5_s140_no-concat_and_add
mask_L6_bce_sobel_k5_s140_noC_and_add = [bce_sobel_k5_s140.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                         bce_sobel_k5_s140.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                         bce_sobel_k5_s140.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                         bce_sobel_k5_s140.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                         bce_sobel_k5_s140.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                         bce_sobel_k5_s140.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s160-1_ch
mask_L6_bce_sobel_k5_s160_ch = [bce_sobel_k5_s160.mask_h_bg_ch128_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch064_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch032_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch016_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch008_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch004_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch002_sig_L6_ep060.build(),
                                bce_sobel_k5_s160.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_sobel_k5_s160_ch = [ exp.result_obj for exp in mask_L6_bce_sobel_k5_s160_ch]
############################################
### 5-6l_bce_sobel_k5_s160_no-concat_and_add
mask_L6_bce_sobel_k5_s160_noC_and_add = [bce_sobel_k5_s160.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                         bce_sobel_k5_s160.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                         bce_sobel_k5_s160.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                         bce_sobel_k5_s160.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                         bce_sobel_k5_s160.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                         bce_sobel_k5_s160.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
