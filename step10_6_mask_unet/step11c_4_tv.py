import mask_5_4_tv             .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv
####################################################################################################################################
####################################################################################################################################
### 4-6l_tv-1_ch
mask_L6_tv_ch = [os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch128_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch064_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch016_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch008_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch004_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch002_sig_L6_ep060.build(),
                 os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_tv_ch = [ exp.result_obj for exp in mask_L6_tv_ch]
############################################
### 4-6l_tv-concat_and_add
mask_L6_tv_noC_and_add = [os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
                          os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
                          os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
                          os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
                          os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
                          os_book_and_paper_have_dtd_hdr_mix_bg_L6_tv.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_tv_noC_and_add = [ exp.result_obj for exp in mask_L6_tv_noC_and_add]