import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_2_6l         .step10_a as bce_s01_6l
import mask_5_os_book_and_paper_have_dtd_hdr_mix_bg_2_bce_s10_6l .step10_a as bce_s01_6l

####################################################################################################################################
### 2-6l-1_ch
mask_6l_bce_ch = [bce_s01_6l.mask_h_bg_ch128_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch064_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch032_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch016_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch008_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch004_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch002_sig_bce_6l_ep060.build(),
                  bce_s01_6l.mask_h_bg_ch001_sig_bce_6l_ep060.build()]
mask_6l_bce_ch = [ exp.result_obj for exp in mask_6l_bce_ch]
############################################
### 2-6l-2_ep
mask_6l_bce_ep = [bce_s01_6l.mask_h_bg_ch128_sig_bce_6l_ep060.build(),
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
mask_6l_bce_ep = [ exp.result_obj for exp in mask_6l_bce_ep]
############################################
### 2-6l-4_no-concat_and_add
mask_6l_noC_and_add = [bce_s01_6l.mask_h_bg_ch032_6l_2to2noC_sig_bce_ep060.build(),
                       bce_s01_6l.mask_h_bg_ch032_6l_2to3noC_sig_bce_ep060.build(),
                       bce_s01_6l.mask_h_bg_ch032_6l_2to4noC_sig_bce_ep060.build(),
                       bce_s01_6l.mask_h_bg_ch032_6l_2to5noC_sig_bce_ep060.build(),
                       bce_s01_6l.mask_h_bg_ch032_6l_2to6noC_sig_bce_ep060.build(),
                       bce_s01_6l.mask_h_bg_ch032_6l_skipAdd_sig_bce_ep060.build()]
mask_6l_noC_and_add = [ exp.result_obj for exp in mask_6l_noC_and_add]
