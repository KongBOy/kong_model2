import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k3_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k7_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7

import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s20_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s40_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s60_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s80_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s100_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s120_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s140_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_5_bce_sobel_k5_s160_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160


####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k3-1_ch
mask_6l_sobel_k3_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep060.build()]
mask_6l_sobel_k3_ch = [ exp.result_obj for exp in mask_6l_sobel_k3_ch]
############################################
### 5-6l_bce_sobel_k3-2_ep
mask_6l_sobel_k3_ep = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep200.build()]
mask_6l_sobel_k3_ep = [ exp.result_obj for exp in mask_6l_sobel_k3_ep]
############################################
### 5-6l_bce_sobel_k3-4_no-concat_and_add
mask_6l_sobel_k3_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k3_ep060.build()]
mask_6l_sobel_k3_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k3_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5-1_ch
mask_6l_sobel_k5_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch128_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch064_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch016_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch008_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch004_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch002_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch001_sig_bce_sobel_k5_6l_ep060.build()]
mask_6l_sobel_k5_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_ch]
############################################
### 5-6l_bce_sobel_k5-2_ep
mask_6l_sobel_k5_ep = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch128_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch064_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch016_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch008_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch004_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch002_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch001_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch128_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch064_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch016_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch008_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch004_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch002_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch001_sig_bce_sobel_k5_6l_ep200.build()]
mask_6l_sobel_k5_ep = [ exp.result_obj for exp in mask_6l_sobel_k5_ep]
############################################
### 5-6l_bce_sobel_k5-4_no-concat_and_add
mask_6l_sobel_k5_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s001.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_ep060.build()]
mask_6l_sobel_k5_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k5_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k7-1_ch
mask_6l_sobel_k7_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch128_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch064_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch016_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch008_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch004_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch002_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch001_sig_bce_sobel_k7_6l_ep060.build()]
mask_6l_sobel_k7_ch = [ exp.result_obj for exp in mask_6l_sobel_k7_ch]
############################################
### 5-6l_bce_sobel_k7-2_ep
mask_6l_sobel_k7_ep = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch128_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch064_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch016_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch008_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch004_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch002_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch001_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch128_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch064_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch016_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch008_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch004_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch002_sig_bce_sobel_k7_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch001_sig_bce_sobel_k7_6l_ep200.build()]
mask_6l_sobel_k7_ep = [ exp.result_obj for exp in mask_6l_sobel_k7_ep]
############################################
### 5-6l_bce_sobel_k7-4_no-concat_and_add
mask_6l_sobel_k7_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k7_ep060.build()]
mask_6l_sobel_k7_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k7_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s20-1_ch
mask_6l_sobel_k5_s20_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch128_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch064_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch016_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch008_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch004_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch002_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch001_sig_bce_sobel_k5_s20_6l_ep060.build()]
mask_6l_sobel_k5_s20_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s20_ch]
############################################
### 5-6l_bce_sobel_k5_s20_no-concat_and_add
mask_6l_sobel_k5_s20_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s20_ep060.build()]
mask_6l_sobel_k5_s20_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k5_s20_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s40-1_ch
mask_6l_sobel_k5_s40_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch128_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch064_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch016_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch008_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch004_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch002_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch001_sig_bce_sobel_k5_s40_6l_ep060.build()]
mask_6l_sobel_k5_s40_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s40_ch]
############################################
### 5-6l_bce_sobel_k5_s40_no-concat_and_add
mask_6l_sobel_k5_s40_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s40_ep060.build()]
mask_6l_sobel_k5_s40_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k5_s40_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s60-1_ch
mask_6l_sobel_k5_s60_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch128_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch064_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch016_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch008_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch004_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch002_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch001_sig_bce_sobel_k5_s60_6l_ep060.build()]
mask_6l_sobel_k5_s60_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s60_ch]
############################################
### 5-6l_bce_sobel_k5_s60_no-concat_and_add
mask_6l_sobel_k5_s60_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s60_ep060.build()]
mask_6l_sobel_k5_s60_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k5_s60_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s80-1_ch
mask_6l_sobel_k5_s80_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch128_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch064_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch016_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch008_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch004_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch002_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch001_sig_bce_sobel_k5_s80_6l_ep060.build()]
mask_6l_sobel_k5_s80_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s80_ch]
############################################
### 5-6l_bce_sobel_k5_s80_no-concat_and_add
mask_6l_sobel_k5_s80_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s80_ep060.build()]
mask_6l_sobel_k5_s80_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k5_s80_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s100-1_ch
mask_6l_sobel_k5_s100_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch128_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch064_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch016_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch008_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch004_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch002_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch001_sig_bce_sobel_k5_s100_6l_ep060.build()]
mask_6l_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s100_ch]
############################################
### 5-6l_bce_sobel_k5_s100_no-concat_and_add
mask_6l_sobel_k5_s100_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s100_ep060.build()]
mask_6l_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_6l_sobel_k5_s100_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s120-1_ch
mask_6l_sobel_k5_s120_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch128_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch064_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch016_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch008_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch004_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch002_sig_bce_sobel_k5_s120_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch001_sig_bce_sobel_k5_s120_6l_ep060.build()]
mask_6l_sobel_k5_s120_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s120_ch]
############################################
### 5-6l_bce_sobel_k5_s120_no-concat_and_add
mask_6l_sobel_k5_s120_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s120_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s120_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s120_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s120_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s120_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s120.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s120_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s140-1_ch
mask_6l_sobel_k5_s140_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch128_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch064_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch016_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch008_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch004_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch002_sig_bce_sobel_k5_s140_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch001_sig_bce_sobel_k5_s140_6l_ep060.build()]
mask_6l_sobel_k5_s140_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s140_ch]
############################################
### 5-6l_bce_sobel_k5_s140_no-concat_and_add
mask_6l_sobel_k5_s140_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s140_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s140_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s140_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s140_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s140_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s140.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s140_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s160-1_ch
mask_6l_sobel_k5_s160_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch128_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch064_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch016_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch008_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch004_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch002_sig_bce_sobel_k5_s160_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch001_sig_bce_sobel_k5_s160_6l_ep060.build()]
mask_6l_sobel_k5_s160_ch = [ exp.result_obj for exp in mask_6l_sobel_k5_s160_ch]
############################################
### 5-6l_bce_sobel_k5_s160_no-concat_and_add
mask_6l_sobel_k5_s160_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s160_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s160_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s160_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s160_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s160_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s160.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s160_ep060.build()]

####################################################################################################################################
####################################################################################################################################
### 5-6l_bce_sobel_k5_s180-1_ch  還沒train
# mask_6l_sobel_k5_s180_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch128_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch064_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch016_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch008_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch004_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch002_sig_bce_sobel_k5_s180_6l_ep060.build(),
#                            os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch001_sig_bce_sobel_k5_s180_6l_ep060.build()]
# ############################################
# ### 5-6l_bce_sobel_k5_s180_no-concat_and_add
# mask_6l_sobel_k5_s180_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s180_ep060.build(),
#                                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s180_ep060.build(),
#                                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s180_ep060.build(),
#                                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s180_ep060.build(),
#                                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s180_ep060.build(),
#                                     os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s180.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s180_ep060.build()]

