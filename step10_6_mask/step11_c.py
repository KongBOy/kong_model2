import mask_1_os_book_no_bg                                         .step10_a as os_book_no_bg
import mask_2_os_book_have_bg                                       .step10_a as os_book_have_bg
import mask_3_os_book_and_paper_have_bg                             .step10_a as os_book_and_paper_have_bg
import mask_4_os_book_and_paper_have_dtd_bg                         .step10_a as os_book_and_paper_have_dtd_bg
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg                 .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_6l              .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k3_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k5_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k5_s20_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k5_s40_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k5_s60_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k5_s80_6l .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k5_s100_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_sobel_k7_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7


### 直接看 dtd_hdr_mix 的狀況
############################################
### 1-7l-1_ch
mask_7l_ch = [os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch128_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch064_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch016_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch008_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch004_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch002_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch001_sig_bce_ep060.build()]

############################################
### 1-7l-2_layer
mask_layer = [os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_2l_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_3l_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_4l_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_5l_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_6l_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_7l_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_8l_sig_bce_ep060.build()]
### 1->2 確定 用 6l
####################################################################################################################################
### 2-6l-1_ch
mask_6l_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch128_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch064_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch016_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch008_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch004_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch002_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch001_sig_bce_6l_ep060.build()]

############################################
### 2-6l-2_ep
mask_6l_ep = [os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch128_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch064_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch016_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch008_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch004_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch002_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch001_sig_bce_6l_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch128_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch064_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch016_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch008_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch004_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch002_sig_bce_6l_ep200.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch001_sig_bce_6l_ep200.build()]

############################################
### 2-6l-4_no-concat_and_add
mask_6l_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_6l_2to2noC_sig_bce_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_6l_2to3noC_sig_bce_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_6l_2to4noC_sig_bce_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_6l_2to5noC_sig_bce_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_6l_2to6noC_sig_bce_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l.mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060.build()]

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
### 3-6l_sobel_k3-1_ch
mask_6l_sobel_k3_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch128_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch064_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch016_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch008_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch004_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch002_sig_bce_sobel_k3_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch001_sig_bce_sobel_k3_6l_ep060.build()]

############################################
### 3-6l_sobel_k3-2_ep
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

############################################
### 3-6l_sobel_k3-4_no-concat_and_add
mask_6l_sobel_k3_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k3_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k3.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k3_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 3-6l_sobel_k5-1_ch
mask_6l_sobel_k5_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch128_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch064_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch016_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch008_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch004_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch002_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch001_sig_bce_sobel_k5_6l_ep060.build()]

############################################
### 3-6l_sobel_k5-2_ep
mask_6l_sobel_k5_ep = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch128_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch064_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch016_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch008_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch004_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch002_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch001_sig_bce_sobel_k5_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch128_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch064_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch016_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch008_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch004_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch002_sig_bce_sobel_k5_6l_ep200.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch001_sig_bce_sobel_k5_6l_ep200.build()]

############################################
### 3-6l_sobel_k5-4_no-concat_and_add
mask_6l_sobel_k5_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 3-6l_sobel_k7-1_ch
mask_6l_sobel_k7_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch128_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch064_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch016_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch008_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch004_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch002_sig_bce_sobel_k7_6l_ep060.build(),
                       os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch001_sig_bce_sobel_k7_6l_ep060.build()]

############################################
### 3-6l_sobel_k7-2_ep
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

############################################
### 3-6l_sobel_k7-4_no-concat_and_add
mask_6l_sobel_k7_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k7_ep060.build(),
                                os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k7.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k7_ep060.build()]

####################################################################################################################################
####################################################################################################################################
### 5-6l_sobel_k5_s20-1_ch
mask_6l_sobel_k5_s20_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch128_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch064_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch016_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch008_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch004_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch002_sig_bce_sobel_k5_s20_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch001_sig_bce_sobel_k5_s20_6l_ep060.build()]
############################################
### 5-6l_sobel_k5_s20_no-concat_and_add
mask_6l_sobel_k5_s20_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s20_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s20.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s20_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_sobel_k5_s40-1_ch
mask_6l_sobel_k5_s40_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch128_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch064_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch016_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch008_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch004_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch002_sig_bce_sobel_k5_s40_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch001_sig_bce_sobel_k5_s40_6l_ep060.build()]
############################################
### 5-6l_sobel_k5_s40_no-concat_and_add
mask_6l_sobel_k5_s40_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s40_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s40.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s40_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_sobel_k5_s60-1_ch
mask_6l_sobel_k5_s60_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch128_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch064_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch016_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch008_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch004_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch002_sig_bce_sobel_k5_s60_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch001_sig_bce_sobel_k5_s60_6l_ep060.build()]
############################################
### 5-6l_sobel_k5_s60_no-concat_and_add
mask_6l_sobel_k5_s60_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s60_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s60.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s60_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_sobel_k5_s80-1_ch
mask_6l_sobel_k5_s80_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch128_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch064_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch016_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch008_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch004_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch002_sig_bce_sobel_k5_s80_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch001_sig_bce_sobel_k5_s80_6l_ep060.build()]
############################################
### 5-6l_sobel_k5_s80_no-concat_and_add
mask_6l_sobel_k5_s80_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s80_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s80.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s80_ep060.build()]
####################################################################################################################################
####################################################################################################################################
### 5-6l_sobel_k5_s100-1_ch
mask_6l_sobel_k5_s100_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch128_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch064_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch016_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch008_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch004_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch002_sig_bce_sobel_k5_s100_6l_ep060.build(),
                           os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch001_sig_bce_sobel_k5_s100_6l_ep060.build()]
############################################
### 5-6l_sobel_k5_s100_no-concat_and_add
mask_6l_sobel_k5_s100_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to2noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to3noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to4noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to5noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_2to6noC_sig_bce_sobel_k5_s100_ep060.build(),
                                    os_book_and_paper_have_dtd_hdr_mix_bg_6l_sobel_k5_s100.mask_h_bg_ch032_6l_skipAdd_sig_bce_sobel_k5_s100_ep060.build()]

"""
group寫法2：from step10_a_load_and_train_and_test import * 直接包 exps
補充：無法直接 from step10_a import * 直接處理，
    因為裡面包含太多其他物件了！光要抽出自己想用的 exp物件就是一大工程覺得~
    還有我也不知道要怎麼 直接用 ，也是要一個個 名字 打出來 才能用，名字 都打出來了，不如就直接 包成exps 囉！
"""

### copy的示範
# ch_064_300 = copy.deepcopy(epoch300_bn_see_arg_T.build()); ch_064_300.ana_describe = "flow_unet-ch64_300"
# ch_064_700 = copy.deepcopy(epoch700_bn_see_arg_T.build()); ch_064_700.ana_describe = "flow_unet-ch64_700"

### 大概怎麼包的示範
# ch64_2to3noC_sk_no_e060         = copy.deepcopy(unet_IN_7l_2to3noC_e060.build()); ch64_2to3noC_sk_no_e060         .result_obj.ana_describe = "1-ch64_2to3noC_sk_no_e060"  ### 當初的train_code沒寫好沒有存到 model用的 code
# ch64_2to3noC_sk_cSE_e060_wrong  = ch64_2to3noC_sk_cSE_e060_wrong .build();        ch64_2to3noC_sk_cSE_e060_wrong  .result_obj.ana_describe = "2-ch64_2to3noC_sk_cSE_e060_wrong"
# ch64_2to3noC_sk_sSE_e060        = ch64_2to3noC_sk_sSE_e060       .build();        ch64_2to3noC_sk_sSE_e060        .result_obj.ana_describe = "3-ch64_2to3noC_sk_sSE_e060"
# ch64_2to3noC_sk_scSE_e060_wrong = ch64_2to3noC_sk_scSE_e060_wrong.build();        ch64_2to3noC_sk_scSE_e060_wrong .result_obj.ana_describe = "4-ch64_2to3noC_sk_scSE_e060_wrong"
# unet_7l_2to3noC_skip_SE = [
#     ch64_2to3noC_sk_no_e060,
#     ch64_2to3noC_sk_cSE_e060_wrong,
#     ch64_2to3noC_sk_sSE_e060,
#     ch64_2to3noC_sk_scSE_e060_wrong,
# ]