import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_6l     .step10_a as sobel_k5_s001
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s020_6l.step10_a as sobel_k5_s020
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s040_6l.step10_a as sobel_k5_s040
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s060_6l.step10_a as sobel_k5_s060
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s080_6l.step10_a as sobel_k5_s080
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s100_6l.step10_a as sobel_k5_s100
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_3_just_sobel_k5_s120_260_6l_ch032.step10_a as sobel_k5_s120_260_ch032
####################################################################################################################################
####################################################################################################################################
### 3-sobel_k5_s001-1_ch
mask_sobel_k5_s001_ch = [sobel_k5_s001.mask_h_bg_ch128_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch064_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch032_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch016_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch008_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch004_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch002_sig_6l_ep060.build(),
                                 sobel_k5_s001.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_sobel_k5_s001_ch = [ exp.result_obj for exp in mask_sobel_k5_s001_ch]
############################################
### 3-sobel_k5_s001-2_ep
mask_sobel_k5_s001_ep = [sobel_k5_s001.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch001_sig_6l_ep060.build(),
                         sobel_k5_s001.mask_h_bg_ch128_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch064_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch032_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch016_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch008_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch004_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch002_sig_6l_ep200.build(),
                         sobel_k5_s001.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_sobel_k5_s001_ep = [ exp.result_obj for exp in mask_sobel_k5_s001_ep]
############################################
### 3-sobel_k5_s001_no-concat_and_add
mask_sobel_k5_s001_noC_and_add = [sobel_k5_s001.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  sobel_k5_s001.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  sobel_k5_s001.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  sobel_k5_s001.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  sobel_k5_s001.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  sobel_k5_s001.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_sobel_k5_s001_noC_and_add = [ exp.result_obj for exp in mask_sobel_k5_s001_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-sobel_k5_s020-1_ch
mask_sobel_k5_s020_ch = [sobel_k5_s020.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_sobel_k5_s020_ch = [ exp.result_obj for exp in mask_sobel_k5_s020_ch]
############################################
### 3-sobel_k5_s020-2_ep
mask_sobel_k5_s020_ep = [sobel_k5_s020.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch001_sig_6l_ep060.build(),
                         sobel_k5_s020.mask_h_bg_ch128_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch064_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch032_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch016_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch008_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch004_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch002_sig_6l_ep200.build(),
                         sobel_k5_s020.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_sobel_k5_s020_ep = [ exp.result_obj for exp in mask_sobel_k5_s020_ep]
############################################
### 3-sobel_k5_s020_no-concat_and_add
mask_sobel_k5_s020_noC_and_add = [sobel_k5_s020.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  sobel_k5_s020.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  sobel_k5_s020.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  sobel_k5_s020.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  sobel_k5_s020.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  sobel_k5_s020.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_sobel_k5_s020_noC_and_add = [ exp.result_obj for exp in mask_sobel_k5_s020_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-sobel_k5_s040-1_ch
mask_sobel_k5_s040_ch = [sobel_k5_s040.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_sobel_k5_s040_ch = [ exp.result_obj for exp in mask_sobel_k5_s040_ch]
############################################
### 3-sobel_k5_s040-2_ep
mask_sobel_k5_s040_ep = [sobel_k5_s040.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch001_sig_6l_ep060.build(),
                         sobel_k5_s040.mask_h_bg_ch128_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch064_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch032_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch016_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch008_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch004_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch002_sig_6l_ep200.build(),
                         sobel_k5_s040.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_sobel_k5_s040_ep = [ exp.result_obj for exp in mask_sobel_k5_s040_ep]
############################################
### 3-sobel_k5_s040_no-concat_and_add
mask_sobel_k5_s040_noC_and_add = [sobel_k5_s040.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  sobel_k5_s040.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  sobel_k5_s040.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  sobel_k5_s040.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  sobel_k5_s040.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  sobel_k5_s040.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_sobel_k5_s040_noC_and_add = [ exp.result_obj for exp in mask_sobel_k5_s040_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-sobel_k5_s060-1_ch
mask_sobel_k5_s060_ch = [sobel_k5_s060.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_sobel_k5_s060_ch = [ exp.result_obj for exp in mask_sobel_k5_s060_ch]
############################################
### 3-sobel_k5_s060-2_ep
mask_sobel_k5_s060_ep = [sobel_k5_s060.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch001_sig_6l_ep060.build(),
                         sobel_k5_s060.mask_h_bg_ch128_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch064_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch032_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch016_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch008_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch004_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch002_sig_6l_ep200.build(),
                         sobel_k5_s060.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_sobel_k5_s060_ep = [ exp.result_obj for exp in mask_sobel_k5_s060_ep]
############################################
### 3-sobel_k5_s060_no-concat_and_add
mask_sobel_k5_s060_noC_and_add = [sobel_k5_s060.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  sobel_k5_s060.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  sobel_k5_s060.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  sobel_k5_s060.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  sobel_k5_s060.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  sobel_k5_s060.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_sobel_k5_s060_noC_and_add = [ exp.result_obj for exp in mask_sobel_k5_s060_noC_and_add]
####################################################################################################################################
####################################################################################################################################
### 3-sobel_k5_s080-1_ch
mask_sobel_k5_s080_ch = [sobel_k5_s080.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_sobel_k5_s080_ch = [ exp.result_obj for exp in mask_sobel_k5_s080_ch]
############################################
### 3-sobel_k5_s080-2_ep
mask_sobel_k5_s080_ep = [sobel_k5_s080.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch001_sig_6l_ep060.build(),
                         sobel_k5_s080.mask_h_bg_ch128_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch064_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch016_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch008_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch004_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch002_sig_6l_ep200.build(),
                         sobel_k5_s080.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_sobel_k5_s080_ep = [ exp.result_obj for exp in mask_sobel_k5_s080_ep]
############################################
### 3-sobel_k5_s080_no-concat_and_add
mask_sobel_k5_s080_noC_and_add = [sobel_k5_s080.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  sobel_k5_s080.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  sobel_k5_s080.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  sobel_k5_s080.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  sobel_k5_s080.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  sobel_k5_s080.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_sobel_k5_s080_noC_and_add = [ exp.result_obj for exp in mask_sobel_k5_s080_noC_and_add]
####################################################################################################################################
### 3-sobel_k5_s100-1_ch
mask_sobel_k5_s100_ch = [sobel_k5_s100.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch001_sig_6l_ep060.build()]
mask_sobel_k5_s100_ch = [ exp.result_obj for exp in mask_sobel_k5_s100_ch]
############################################
### 3-sobel_k5_s100-2_ep
mask_sobel_k5_s100_ep = [sobel_k5_s100.mask_h_bg_ch128_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch064_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch032_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch016_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch008_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch004_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch002_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch001_sig_6l_ep060.build(),
                         sobel_k5_s100.mask_h_bg_ch128_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch064_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch032_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch016_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch008_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch004_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch002_sig_6l_ep200.build(),
                         sobel_k5_s100.mask_h_bg_ch001_sig_6l_ep200.build()]
mask_sobel_k5_s100_ep = [ exp.result_obj for exp in mask_sobel_k5_s100_ep]
############################################
### 3-sobel_k5_s100_no-concat_and_add
mask_sobel_k5_s100_noC_and_add = [sobel_k5_s100.mask_h_bg_ch032_6l_2to2noC_sig_ep060.build(),
                                  sobel_k5_s100.mask_h_bg_ch032_6l_2to3noC_sig_ep060.build(),
                                  sobel_k5_s100.mask_h_bg_ch032_6l_2to4noC_sig_ep060.build(),
                                  sobel_k5_s100.mask_h_bg_ch032_6l_2to5noC_sig_ep060.build(),
                                  sobel_k5_s100.mask_h_bg_ch032_6l_2to6noC_sig_ep060.build(),
                                  sobel_k5_s100.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build()]
mask_sobel_k5_s100_noC_and_add = [ exp.result_obj for exp in mask_sobel_k5_s100_noC_and_add]
####################################################################################################################################
### 3_ch032-sobel_k5_s1~260
mask_ch032_sobel_k5_s1_260 = [
                            sobel_k5_s001.mask_h_bg_ch032_sig_6l_ep060.build(),
                            sobel_k5_s020.mask_h_bg_ch032_sig_6l_ep060.build(),
                            sobel_k5_s040.mask_h_bg_ch032_sig_6l_ep060.build(),
                            sobel_k5_s060.mask_h_bg_ch032_sig_6l_ep060.build(),
                            sobel_k5_s080.mask_h_bg_ch032_sig_6l_ep060.build(),
                            sobel_k5_s100.mask_h_bg_ch032_sig_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s120_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s140_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s160_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s180_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s200_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s220_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s240_6l_ep060.build(),
                            sobel_k5_s120_260_ch032.mask_h_bg_ch032_sig_sobel_k5_s260_6l_ep060.build(),
                         ]
mask_ch032_sobel_k5_s1_260 = [ exp.result_obj for exp in mask_ch032_sobel_k5_s1_260]
############################################
