import mask_5_6_tv_bce         .step10_a as tv_s01_bce_s001
import mask_5_6_tv_s04_bce     .step10_a as tv_s04_bce_s001
import mask_5_6_tv_s08_bce     .step10_a as tv_s08_bce_s001
import mask_5_6_tv_s12_bce     .step10_a as tv_s12_bce_s001
import mask_5_6_tv_s16_bce     .step10_a as tv_s16_bce_s001
import mask_5_6_tv_s20_bce     .step10_a as tv_s20_bce_s001
import mask_5_6_tv_s40_bce     .step10_a as tv_s40_bce_s001
import mask_5_6_tv_s60_bce     .step10_a as tv_s60_bce_s001
import mask_5_6_tv_s80_bce     .step10_a as tv_s80_bce_s001
import mask_5_6_tv_s20_30_bce_s001_080. step10_a as tv_s20_30_bce_s001_080

####################################################################################################################################
####################################################################################################################################
### 6-6l_tv_bce-1_ch
mask_L6_tv_bce_ch = [tv_s01_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                     tv_s01_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_bce_ch = [ exp.result_obj for exp in mask_L6_tv_bce_ch]
############################################
### 6-6l_tv_bce-concat_and_add
mask_L6_tv_bce_noC_and_add = [tv_s01_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                              tv_s01_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                              tv_s01_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                              tv_s01_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                              tv_s01_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                              tv_s01_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s04_bce-1_ch
mask_L6_tv_s04_bce_ch = [tv_s04_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s04_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s04_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s04_bce_ch]
############################################
### 6-6l_tv_s04_bce-concat_and_add
mask_L6_tv_s04_bce_noC_and_add = [tv_s04_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s04_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s04_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s04_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s04_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s04_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s04_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s04_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s08_bce-1_ch
mask_L6_tv_s08_bce_ch = [tv_s08_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s08_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s08_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s08_bce_ch]
############################################
### 6-6l_tv_s08_bce-concat_and_add
mask_L6_tv_s08_bce_noC_and_add = [tv_s08_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s08_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s08_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s08_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s08_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s08_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s08_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s08_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s12_bce-1_ch
mask_L6_tv_s12_bce_ch = [tv_s12_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s12_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s12_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s12_bce_ch]
############################################
### 6-6l_tv_s12_bce-concat_and_add
mask_L6_tv_s12_bce_noC_and_add = [tv_s12_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s12_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s12_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s12_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s12_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s12_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s12_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s12_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s16_bce-1_ch
mask_L6_tv_s16_bce_ch = [tv_s16_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s16_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s16_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s16_bce_ch]
############################################
### 6-6l_tv_s16_bce-concat_and_add
mask_L6_tv_s16_bce_noC_and_add = [tv_s16_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s16_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s16_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s16_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s16_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s16_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s16_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s16_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s20_bce-1_ch
mask_L6_tv_s20_bce_ch = [tv_s20_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s20_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s20_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s20_bce_ch]
############################################
### 6-6l_tv_s20_bce-concat_and_add
mask_L6_tv_s20_bce_noC_and_add = [tv_s20_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s20_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s20_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s20_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s20_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s20_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s20_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s20_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s40_bce-1_ch
mask_L6_tv_s40_bce_ch = [tv_s40_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s40_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s40_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s40_bce_ch]
############################################
### 6-6l_tv_s40_bce-concat_and_add
mask_L6_tv_s40_bce_noC_and_add = [tv_s40_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s40_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s40_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s40_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s40_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s40_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s40_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s40_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s60_bce-1_ch
mask_L6_tv_s60_bce_ch = [tv_s60_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s60_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s60_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s60_bce_ch]
############################################
### 6-6l_tv_s60_bce-concat_and_add
mask_L6_tv_s60_bce_noC_and_add = [tv_s60_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s60_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s60_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s60_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s60_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s60_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s60_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s60_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s80_bce-1_ch
mask_L6_tv_s80_bce_ch = [tv_s80_bce_s001.mask_h_bg_ch128_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch064_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch032_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch016_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch008_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch004_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch002_sig_L6_ep060.build(),
                         tv_s80_bce_s001.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_s80_bce_ch = [ exp.result_obj for exp in mask_L6_tv_s80_bce_ch]
############################################
### 6-6l_tv_s80_bce-concat_and_add
mask_L6_tv_s80_bce_noC_and_add = [tv_s80_bce_s001.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                                  tv_s80_bce_s001.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                                  tv_s80_bce_s001.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                                  tv_s80_bce_s001.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                                  tv_s80_bce_s001.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                                  tv_s80_bce_s001.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_s80_bce_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_s80_bce_noC_and_add]

############################################
mask_tv_s20_30_bce_s001_080s = [
    [tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s20_bce_s001.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s20_bce_s020.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s20_bce_s040.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s20_bce_s060.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s20_bce_s080.build().result_obj, ],
    [tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s30_bce_s001.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s30_bce_s020.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s30_bce_s040.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s30_bce_s060.build().result_obj,
     tv_s20_30_bce_s001_080.mask_h_bg_ch032_sig_L6_ep060_tv_s30_bce_s080.build().result_obj, ],
]
