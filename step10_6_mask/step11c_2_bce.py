import mask_5_2_6l         .step10_a as bce_s01_6l
import mask_5_2_bce_s10_6l .step10_a as bce_s10_6l
import mask_5_2_bce_s20_6l .step10_a as bce_s20_6l
import mask_5_2_bce_s40_6l .step10_a as bce_s40_6l
import mask_5_2_bce_s60_6l .step10_a as bce_s60_6l
import mask_5_2_bce_s80_6l .step10_a as bce_s80_6l

####################################################################################################################################
### 2-6l_bce_s01-1_ch
mask_6l_bce_s01_ch = [bce_s01_6l.mask_h_bg_ch128_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch064_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch032_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch016_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch008_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch004_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch002_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch001_sig_bce_6l_ep060.build()]
mask_6l_bce_s01_ch = [ exp.result_obj for exp in mask_6l_bce_s01_ch]
############################################
### 2-6l_bce_s01-2_ep
mask_6l_bce_s01_ep = [bce_s01_6l.mask_h_bg_ch128_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch064_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch032_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch016_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch008_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch004_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch002_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch001_sig_bce_6l_ep060.build(),
                      bce_s01_6l.mask_h_bg_ch128_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch064_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch032_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch016_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch008_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch004_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch002_sig_bce_6l_ep200.build(),
                      bce_s01_6l.mask_h_bg_ch001_sig_bce_6l_ep200.build()]
mask_6l_bce_s01_ep = [ exp.result_obj for exp in mask_6l_bce_s01_ep]
############################################
### 2-6l_bce_s01-4_no-concat_and_add
mask_6l_bce_s01_noC_and_add = [bce_s01_6l.mask_h_bg_ch032_6l_2to2noC_sig_bce_ep060.build(),
                               bce_s01_6l.mask_h_bg_ch032_6l_2to3noC_sig_bce_ep060.build(),
                               bce_s01_6l.mask_h_bg_ch032_6l_2to4noC_sig_bce_ep060.build(),
                               bce_s01_6l.mask_h_bg_ch032_6l_2to5noC_sig_bce_ep060.build(),
                               bce_s01_6l.mask_h_bg_ch032_6l_2to6noC_sig_bce_ep060.build(),
                               bce_s01_6l.mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060.build()]
mask_6l_bce_s01_noC_and_add = [ exp.result_obj for exp in mask_6l_bce_s01_noC_and_add]

####################################################################################################################################
### 2-6l_bce_s10-1_ch
mask_6l_bce_s10_ch = [bce_s10_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch016_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch008_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch004_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch002_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_bce_s10_ch = [ exp.result_obj for exp in mask_6l_bce_s10_ch]
############################################
### 2-6l_bce_s10-2_ep
mask_6l_bce_s10_ep = [bce_s10_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch016_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch008_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch004_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch002_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch001_sig_6l_ep060.build(),
                      bce_s10_6l.mask_h_bg_ch128_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch064_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch032_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch016_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch008_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch004_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch002_sig_6l_ep200.build(),
                      bce_s10_6l.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_6l_bce_s10_ep = [ exp.result_obj for exp in mask_6l_bce_s10_ep]
############################################
### 2-6l_bce_s10-4_no-concat_and_add
mask_6l_bce_s10_noC_and_add = [bce_s10_6l.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                               bce_s10_6l.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                               bce_s10_6l.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                               bce_s10_6l.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                               bce_s10_6l.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                               bce_s10_6l.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_bce_s10_noC_and_add = [ exp.result_obj for exp in mask_6l_bce_s10_noC_and_add]
####################################################################################################################################
### 2-6l_bce_s20-1_ch
mask_6l_bce_s20_ch = [bce_s20_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch016_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch008_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch004_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch002_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_bce_s20_ch = [ exp.result_obj for exp in mask_6l_bce_s20_ch]
############################################
### 2-6l_bce_s20-2_ep
mask_6l_bce_s20_ep = [bce_s20_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch016_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch008_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch004_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch002_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch001_sig_6l_ep060.build(),
                      bce_s20_6l.mask_h_bg_ch128_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch064_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch032_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch016_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch008_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch004_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch002_sig_6l_ep200.build(),
                      bce_s20_6l.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_6l_bce_s20_ep = [ exp.result_obj for exp in mask_6l_bce_s20_ep]
############################################
### 2-6l_bce_s20-4_no-concat_and_add
mask_6l_bce_s20_noC_and_add = [bce_s20_6l.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                               bce_s20_6l.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                               bce_s20_6l.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                               bce_s20_6l.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                               bce_s20_6l.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                               bce_s20_6l.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_bce_s20_noC_and_add = [ exp.result_obj for exp in mask_6l_bce_s20_noC_and_add]
####################################################################################################################################
### 2-6l_bce_s40-1_ch
mask_6l_bce_s40_ch = [bce_s40_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s40_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s40_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s40_6l.mask_h_bg_ch016_sig_6l_ep060.build()]
mask_6l_bce_s40_ch = [ exp.result_obj for exp in mask_6l_bce_s40_ch]
############################################
### 2-6l_bce_s40-2_ep
mask_6l_bce_s40_ep = [bce_s40_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s40_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s40_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s40_6l.mask_h_bg_ch016_sig_6l_ep060.build(),

                      bce_s40_6l.mask_h_bg_ch128_sig_6l_ep200.build(),
                      bce_s40_6l.mask_h_bg_ch064_sig_6l_ep200.build(),
                      bce_s40_6l.mask_h_bg_ch032_sig_6l_ep200.build(),
                      bce_s40_6l.mask_h_bg_ch016_sig_6l_ep200.build()]
mask_6l_bce_s40_ep = [ exp.result_obj for exp in mask_6l_bce_s40_ep]
############################################
### 2-6l_bce_s40-4_no-concat_and_add
mask_6l_bce_s40_noC_and_add = [bce_s40_6l.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                               bce_s40_6l.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                               bce_s40_6l.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                               bce_s40_6l.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                               bce_s40_6l.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                               bce_s40_6l.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_bce_s40_noC_and_add = [ exp.result_obj for exp in mask_6l_bce_s40_noC_and_add]
####################################################################################################################################
### 2-6l_bce_s60-1_ch
mask_6l_bce_s60_ch = [bce_s60_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s60_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s60_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s60_6l.mask_h_bg_ch016_sig_6l_ep060.build()]
mask_6l_bce_s60_ch = [ exp.result_obj for exp in mask_6l_bce_s60_ch]
############################################
### 2-6l_bce_s60-2_ep
mask_6l_bce_s60_ep = [bce_s60_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s60_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s60_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s60_6l.mask_h_bg_ch016_sig_6l_ep060.build(),

                      bce_s60_6l.mask_h_bg_ch128_sig_6l_ep200.build(),
                      bce_s60_6l.mask_h_bg_ch064_sig_6l_ep200.build(),
                      bce_s60_6l.mask_h_bg_ch032_sig_6l_ep200.build(),
                      bce_s60_6l.mask_h_bg_ch016_sig_6l_ep200.build()]
mask_6l_bce_s60_ep = [ exp.result_obj for exp in mask_6l_bce_s60_ep]
############################################
### 2-6l_bce_s60-4_no-concat_and_add
mask_6l_bce_s60_noC_and_add = [bce_s60_6l.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                               bce_s60_6l.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                               bce_s60_6l.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                               bce_s60_6l.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                               bce_s60_6l.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                               bce_s60_6l.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_bce_s60_noC_and_add = [ exp.result_obj for exp in mask_6l_bce_s60_noC_and_add]
####################################################################################################################################
### 2-6l_bce_s80-1_ch
mask_6l_bce_s80_ch = [bce_s80_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s80_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s80_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s80_6l.mask_h_bg_ch016_sig_6l_ep060.build()]
mask_6l_bce_s80_ch = [ exp.result_obj for exp in mask_6l_bce_s80_ch]
############################################
### 2-6l_bce_s80-2_ep
mask_6l_bce_s80_ep = [bce_s80_6l.mask_h_bg_ch128_sig_6l_ep060.build(),
                      bce_s80_6l.mask_h_bg_ch064_sig_6l_ep060.build(),
                      bce_s80_6l.mask_h_bg_ch032_sig_6l_ep060.build(),
                      bce_s80_6l.mask_h_bg_ch016_sig_6l_ep060.build(),

                      bce_s80_6l.mask_h_bg_ch128_sig_6l_ep200.build(),
                      bce_s80_6l.mask_h_bg_ch064_sig_6l_ep200.build(),
                      bce_s80_6l.mask_h_bg_ch032_sig_6l_ep200.build(),
                      bce_s80_6l.mask_h_bg_ch016_sig_6l_ep200.build()]
mask_6l_bce_s80_ep = [ exp.result_obj for exp in mask_6l_bce_s80_ep]
############################################
### 2-6l_bce_s80-4_no-concat_and_add
mask_6l_bce_s80_noC_and_add = [bce_s80_6l.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                               bce_s80_6l.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                               bce_s80_6l.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                               bce_s80_6l.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                               bce_s80_6l.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                               bce_s80_6l.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_bce_s80_noC_and_add = [ exp.result_obj for exp in mask_6l_bce_s80_noC_and_add]
