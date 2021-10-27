import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_6l     .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s020_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s040_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s060_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s080_6l.step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080
####################################################################################################################################
####################################################################################################################################
### 3-6l_just_sobel_k5_s001-1_ch
mask_6l_just_sobel_k5_s001_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch128_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch064_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch016_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch008_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch004_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch002_sig_sobel_k5_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch001_sig_sobel_k5_6l_ep060.build()]
mask_6l_just_sobel_k5_s001_ch = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s001_ch]
############################################
### 3-6l_just_sobel_k5_s001_no-concat_and_add
mask_6l_just_sobel_k5_s001_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_6l_2to2noC_sig_sobel_k5_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_6l_2to3noC_sig_sobel_k5_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_6l_2to4noC_sig_sobel_k5_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_6l_2to5noC_sig_sobel_k5_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_6l_2to6noC_sig_sobel_k5_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s001.mask_h_bg_ch032_6l_skipAdd_sig_sobel_k5_ep060.build()]
mask_6l_just_sobel_k5_s001_noC_and_add = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s001_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-6l_just_sobel_k5_s020-1_ch
mask_6l_just_sobel_k5_s020_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch128_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch064_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch016_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch008_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch004_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch002_sig_sobel_k5_s020_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch001_sig_sobel_k5_s020_6l_ep060.build()]
mask_6l_just_sobel_k5_s020_ch = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s020_ch]
############################################
### 3-6l_just_sobel_k5_s020_no-concat_and_add
mask_6l_just_sobel_k5_s020_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_6l_2to2noC_sig_sobel_k5_s020_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_6l_2to3noC_sig_sobel_k5_s020_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_6l_2to4noC_sig_sobel_k5_s020_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_6l_2to5noC_sig_sobel_k5_s020_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_6l_2to6noC_sig_sobel_k5_s020_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s020.mask_h_bg_ch032_6l_skipAdd_sig_sobel_k5_s020_ep060.build()]
mask_6l_just_sobel_k5_s020_noC_and_add = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s020_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-6l_just_sobel_k5_s040-1_ch
mask_6l_just_sobel_k5_s040_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch128_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch064_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch016_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch008_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch004_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch002_sig_sobel_k5_s040_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch001_sig_sobel_k5_s040_6l_ep060.build()]
mask_6l_just_sobel_k5_s040_ch = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s040_ch]
############################################
### 3-6l_just_sobel_k5_s040_no-concat_and_add
mask_6l_just_sobel_k5_s040_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_6l_2to2noC_sig_sobel_k5_s040_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_6l_2to3noC_sig_sobel_k5_s040_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_6l_2to4noC_sig_sobel_k5_s040_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_6l_2to5noC_sig_sobel_k5_s040_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_6l_2to6noC_sig_sobel_k5_s040_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s040.mask_h_bg_ch032_6l_skipAdd_sig_sobel_k5_s040_ep060.build()]
mask_6l_just_sobel_k5_s040_noC_and_add = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s040_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-6l_just_sobel_k5_s060-1_ch
mask_6l_just_sobel_k5_s060_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch128_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch064_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch016_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch008_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch004_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch002_sig_sobel_k5_s060_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch001_sig_sobel_k5_s060_6l_ep060.build()]
mask_6l_just_sobel_k5_s060_ch = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s060_ch]
############################################
### 3-6l_just_sobel_k5_s060_no-concat_and_add
mask_6l_just_sobel_k5_s060_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_6l_2to2noC_sig_sobel_k5_s060_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_6l_2to3noC_sig_sobel_k5_s060_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_6l_2to4noC_sig_sobel_k5_s060_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_6l_2to5noC_sig_sobel_k5_s060_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_6l_2to6noC_sig_sobel_k5_s060_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s060.mask_h_bg_ch032_6l_skipAdd_sig_sobel_k5_s060_ep060.build()]
mask_6l_just_sobel_k5_s060_noC_and_add = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s060_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-6l_just_sobel_k5_s080-1_ch
mask_6l_just_sobel_k5_s080_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch128_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch064_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch016_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch008_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch004_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch002_sig_sobel_k5_s080_6l_ep060.build(),
                                 os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch001_sig_sobel_k5_s080_6l_ep060.build()]
mask_6l_just_sobel_k5_s080_ch = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s080_ch]
############################################
### 3-6l_just_sobel_k5_s080_no-concat_and_add
mask_6l_just_sobel_k5_s080_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_6l_2to2noC_sig_sobel_k5_s080_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_6l_2to3noC_sig_sobel_k5_s080_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_6l_2to4noC_sig_sobel_k5_s080_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_6l_2to5noC_sig_sobel_k5_s080_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_6l_2to6noC_sig_sobel_k5_s080_ep060.build(),
                                          os_book_and_paper_have_dtd_hdr_mix_bg_6l_just_sobel_k5_s080.mask_h_bg_ch032_6l_skipAdd_sig_sobel_k5_s080_ep060.build()]
mask_6l_just_sobel_k5_s080_noC_and_add = [ exp.result_obj for exp in mask_6l_just_sobel_k5_s080_noC_and_add]