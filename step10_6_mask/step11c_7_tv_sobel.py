import mask_5_7_tv_s01_sobel_k5_s001       .step10_a as tv_s01_sobel_k5_s001
import mask_5_7_tv_s01_sobel_k5_s080       .step10_a as tv_s01_sobel_k5_s080
import mask_5_7_tv_s01_sobel_k5_s100       .step10_a as tv_s01_sobel_k5_s100
import mask_5_7_tv_s01_sobel_k5_s120       .step10_a as tv_s01_sobel_k5_s120
import mask_5_7_tv_s01_sobel_k5_s140       .step10_a as tv_s01_sobel_k5_s140
import mask_5_7_tv_s04_sobel_k5_s080       .step10_a as tv_s04_sobel_k5_s080
import mask_5_7_tv_s04_sobel_k5_s100       .step10_a as tv_s04_sobel_k5_s100
import mask_5_7_tv_s04_sobel_k5_s120       .step10_a as tv_s04_sobel_k5_s120
import mask_5_7_tv_s04_sobel_k5_s140       .step10_a as tv_s04_sobel_k5_s140
import mask_5_7_tv_s20_30_sobel_k5_s001_140.step10_a as tv_s20_30_sobel_k5_s001_140
####################################################################################################################################
####################################################################################################################################
### 7-6l_tv_s01_sobel_k5_s001-1_ch
mask_6l_tv_s01_sobel_k5_s001_ch = [tv_s01_sobel_k5_s001.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s001.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s01_sobel_k5_s001_ch = [ exp.result_obj for exp in mask_6l_tv_s01_sobel_k5_s001_ch]
############################################
### 7-6l_tv_s01_sobel_k5_s001-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s01_sobel_k5_s001.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s001.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s001.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s001.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s001.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s001.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s01_sobel_k5_s080-1_ch
mask_6l_tv_s01_sobel_k5_s080_ch = [tv_s01_sobel_k5_s080.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s080.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s01_sobel_k5_s080_ch = [ exp.result_obj for exp in mask_6l_tv_s01_sobel_k5_s080_ch]
############################################
### 7-6l_tv_s01_sobel_k5_s080-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s01_sobel_k5_s080.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s080.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s080.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s080.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s080.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s080.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s01_sobel_k5_s100-1_ch
mask_6l_tv_s01_sobel_k5_s100_ch = [tv_s01_sobel_k5_s100.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s100.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s01_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_6l_tv_s01_sobel_k5_s100_ch]
############################################
### 7-6l_tv_s01_sobel_k5_s100-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s01_sobel_k5_s100.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s100.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s100.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s100.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s100.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s100.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s01_sobel_k5_s120-1_ch
mask_6l_tv_s01_sobel_k5_s120_ch = [tv_s01_sobel_k5_s120.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s120.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s01_sobel_k5_s120_ch = [ exp.result_obj for exp in mask_6l_tv_s01_sobel_k5_s120_ch]
############################################
### 7-6l_tv_s01_sobel_k5_s120-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s01_sobel_k5_s120.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s120.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s120.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s120.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s120.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s120.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s01_sobel_k5_s140-1_ch
mask_6l_tv_s01_sobel_k5_s140_ch = [tv_s01_sobel_k5_s140.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s01_sobel_k5_s140.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s01_sobel_k5_s140_ch = [ exp.result_obj for exp in mask_6l_tv_s01_sobel_k5_s140_ch]
############################################
### 7-6l_tv_s01_sobel_k5_s140-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s01_sobel_k5_s140.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s140.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s140.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s140.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s140.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s01_sobel_k5_s140.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s04_sobel_k5_s080-1_ch
mask_6l_tv_s04_sobel_k5_s080_ch = [tv_s04_sobel_k5_s080.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s080.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s04_sobel_k5_s080_ch = [ exp.result_obj for exp in mask_6l_tv_s04_sobel_k5_s080_ch]
############################################
### 7-6l_tv_s04_sobel_k5_s080-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s04_sobel_k5_s080.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s080.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s080.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s080.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s080.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s080.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s04_sobel_k5_s100-1_ch
mask_6l_tv_s04_sobel_k5_s100_ch = [tv_s04_sobel_k5_s100.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s100.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s04_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_6l_tv_s04_sobel_k5_s100_ch]
############################################
### 7-6l_tv_s04_sobel_k5_s100-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s04_sobel_k5_s100.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s100.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s100.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s100.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s100.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s100.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s04_sobel_k5_s120-1_ch
mask_6l_tv_s04_sobel_k5_s120_ch = [tv_s04_sobel_k5_s120.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s120.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s04_sobel_k5_s120_ch = [ exp.result_obj for exp in mask_6l_tv_s04_sobel_k5_s120_ch]
############################################
### 7-6l_tv_s04_sobel_k5_s120-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s04_sobel_k5_s120.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s120.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s120.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s120.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s120.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s120.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]
####################################################################################################################################
### 7-6l_tv_s04_sobel_k5_s140-1_ch
mask_6l_tv_s04_sobel_k5_s140_ch = [tv_s04_sobel_k5_s140.mask_h_bg_ch128_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch064_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch032_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch016_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch008_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch004_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch002_sig_6l_ep060.build(),
                                   tv_s04_sobel_k5_s140.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s04_sobel_k5_s140_ch = [ exp.result_obj for exp in mask_6l_tv_s04_sobel_k5_s140_ch]
############################################
### 7-6l_tv_s04_sobel_k5_s140-concat_and_add
mask_6l_tv_sobel_noC_and_add = [tv_s04_sobel_k5_s140.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s140.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s140.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s140.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s140.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                tv_s04_sobel_k5_s140.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_sobel_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_sobel_noC_and_add]

############################################

tv_s20_30_sobel_k5_s001_140s = [
    [tv_s01_sobel_k5_s001.mask_h_bg_ch032_sig_6l_ep060.build().result_obj,
     tv_s01_sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep060.build().result_obj,
     tv_s01_sobel_k5_s120.mask_h_bg_ch032_sig_6l_ep060.build().result_obj, ],
    [tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s20_sobel_k5_s001.build().result_obj,
     tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s20_sobel_k5_s080.build().result_obj,
     # tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s20_sobel_k5_s100.build().result_obj,
     tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s20_sobel_k5_s120.build().result_obj,
     # tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s20_sobel_k5_s140.build().result_obj,
                                                                                    ],
    [tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s30_sobel_k5_s001.build().result_obj,
     # tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s30_sobel_k5_s080.build().result_obj,
     tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s30_sobel_k5_s100.build().result_obj,
     # tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s30_sobel_k5_s120.build().result_obj,
     tv_s20_30_sobel_k5_s001_140.mask_h_bg_ch032_sig_6l_ep060_tv_s30_sobel_k5_s140.build().result_obj, ],
]
