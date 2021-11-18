
import mask_5_8_tv_s01_bce_sobel_k5_s001.step10_a as tv_s01_bce_s001_sobel_k5_s001
import mask_5_8_tv_s01_bce_sobel_k5_s100.step10_a as tv_s01_bce_s001_sobel_k5_s100
import mask_5_8_tv_s04_bce_sobel_k5_s001.step10_a as tv_s04_bce_s001_sobel_k5_s001
import mask_5_8_tv_s04_bce_sobel_k5_s100.step10_a as tv_s04_bce_s001_sobel_k5_s100
import mask_5_8_tv_s08_bce_sobel_k5_s100.step10_a as tv_s08_bce_s001_sobel_k5_s100
import mask_5_8_tv_s12_bce_sobel_k5_s100.step10_a as tv_s12_bce_s001_sobel_k5_s100
####################################################################################################################################
####################################################################################################################################
### 8-6l_tv_s01_bce_sobel_k5_s001-1_ch
mask_L6_tv_s01_bce_sobel_k5_s001_ch = [tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s01_bce_sobel_k5_s001_ch = [ exp.result_obj for exp in mask_L6_tv_s01_bce_sobel_k5_s001_ch]
############################################
### 8-6l_tv_s01_bce_sobel_k5_s001-concat_and_add
mask_L6_tv_s01_bce_sobel_k5_s001_noC_and_add = [tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s01_bce_sobel_k5_s001_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s01_bce_sobel_k5_s001_noC_and_add]
####################################################################################################################################
### 8-6l_tv_s01_bce_sobel_k5_s100-1_ch
mask_L6_tv_s01_bce_sobel_k5_s100_ch = [tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
                                       tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s01_bce_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_L6_tv_s01_bce_sobel_k5_s100_ch]
############################################
### 8-6l_tv_s01_bce_sobel_k5_s100-concat_and_add
mask_L6_tv_s01_bce_sobel_k5_s100_noC_and_add = [tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                                tv_s01_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s01_bce_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s01_bce_sobel_k5_s100_noC_and_add]
####################################################################################################################################
### 8-6l_tv_s04_bce_sobel_k5_s001-1_ch
mask_L6_tv_s04_bce_sobel_k5_s001_ch = [tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s04_bce_sobel_k5_s001_ch = [ exp.result_obj for exp in mask_L6_tv_s04_bce_sobel_k5_s001_ch]
############################################
### 8-6l_tv_s04_bce_sobel_k5_s001-concat_and_add
mask_L6_tv_s04_bce_sobel_k5_s001_noC_and_add = [tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s04_bce_sobel_k5_s001_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s04_bce_sobel_k5_s001_noC_and_add]
####################################################################################################################################
### 8-6l_tv_s04_bce_sobel_k5_s100-1_ch
mask_L6_tv_s04_bce_sobel_k5_s100_ch = [tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
                                       tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s04_bce_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_L6_tv_s04_bce_sobel_k5_s100_ch]
############################################
### 8-6l_tv_s04_bce_sobel_k5_s100-concat_and_add
mask_L6_tv_s04_bce_sobel_k5_s100_noC_and_add = [tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                                tv_s04_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s04_bce_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s04_bce_sobel_k5_s100_noC_and_add]
####################################################################################################################################
### 8-6l_tv_s08_bce_sobel_k5_s100-1_ch
mask_L6_tv_s08_bce_sobel_k5_s100_ch = [tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
                                       tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s08_bce_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_L6_tv_s08_bce_sobel_k5_s100_ch]
############################################
### 8-6l_tv_s08_bce_sobel_k5_s100-concat_and_add
mask_L6_tv_s08_bce_sobel_k5_s100_noC_and_add = [tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                                tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                                tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                                tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                                tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                                tv_s08_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s08_bce_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s08_bce_sobel_k5_s100_noC_and_add]
####################################################################################################################################
### 8-6l_tv_s12_bce_sobel_k5_s100-1_ch
mask_L6_tv_s12_bce_sobel_k5_s100_ch = [tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch128_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch064_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch016_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch008_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch004_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch002_sig_L6_ep060.build(),
                                       tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s12_bce_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_L6_tv_s12_bce_sobel_k5_s100_ch]
############################################
### 8-6l_tv_s12_bce_sobel_k5_s100-concat_and_add
mask_L6_tv_s12_bce_sobel_k5_s100_noC_and_add = [tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                                tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                                tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                                tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                                tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                                tv_s12_bce_s001_sobel_k5_s100.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s12_bce_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s12_bce_sobel_k5_s100_noC_and_add]
