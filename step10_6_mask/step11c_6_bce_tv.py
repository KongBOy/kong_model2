import mask_5_6_tv_bce         .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce
import mask_5_6_tv_s04_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce
import mask_5_6_tv_s08_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce
import mask_5_6_tv_s12_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce
import mask_5_6_tv_s16_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce
import mask_5_6_tv_s20_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce
import mask_5_6_tv_s40_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce
import mask_5_6_tv_s60_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce
import mask_5_6_tv_s80_bce     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce

####################################################################################################################################
####################################################################################################################################
### 6-6l_tv_bce-1_ch
mask_6l_tv_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_bce_ch = [ exp.result_obj for exp in mask_6l_tv_bce_ch]
############################################
### 6-6l_tv_bce-concat_and_add
mask_6l_tv_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                              os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                              os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                              os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                              os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                              os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s04_bce-1_ch
mask_6l_tv_s04_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s04_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s04_bce_ch]
############################################
### 6-6l_tv_s04_bce-concat_and_add
mask_6l_tv_s04_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s04_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s04_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s04_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s08_bce-1_ch
mask_6l_tv_s08_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s08_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s08_bce_ch]
############################################
### 6-6l_tv_s08_bce-concat_and_add
mask_6l_tv_s08_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s08_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s08_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s08_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s12_bce-1_ch
mask_6l_tv_s12_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s12_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s12_bce_ch]
############################################
### 6-6l_tv_s12_bce-concat_and_add
mask_6l_tv_s12_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s12_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s12_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s12_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s16_bce-1_ch
mask_6l_tv_s16_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s16_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s16_bce_ch]
############################################
### 6-6l_tv_s16_bce-concat_and_add
mask_6l_tv_s16_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s16_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s16_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s16_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s20_bce-1_ch
mask_6l_tv_s20_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s20_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s20_bce_ch]
############################################
### 6-6l_tv_s20_bce-concat_and_add
mask_6l_tv_s20_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s20_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s20_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s20_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s40_bce-1_ch
mask_6l_tv_s40_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s40_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s40_bce_ch]
############################################
### 6-6l_tv_s40_bce-concat_and_add
mask_6l_tv_s40_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s40_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s40_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s40_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s60_bce-1_ch
mask_6l_tv_s60_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s60_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s60_bce_ch]
############################################
### 6-6l_tv_s60_bce-concat_and_add
mask_6l_tv_s60_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s60_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s60_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s60_bce_noC_and_add]
####################################################################################################################################
### 6-6l_tv_s80_bce-1_ch
mask_6l_tv_s80_bce_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch128_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch064_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch016_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch008_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch004_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch002_sig_6l_ep060.build(),
                         os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_6l_tv_s80_bce_ch = [ exp.result_obj for exp in mask_6l_tv_s80_bce_ch]
############################################
### 6-6l_tv_s80_bce-concat_and_add
mask_6l_tv_s80_bce_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  os_book_and_paper_have_dtd_hdr_mix_bg_6l_tv_s80_bce.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_6l_tv_s80_bce_noC_and_add = [ exp.result_obj for exp in mask_6l_tv_s80_bce_noC_and_add]