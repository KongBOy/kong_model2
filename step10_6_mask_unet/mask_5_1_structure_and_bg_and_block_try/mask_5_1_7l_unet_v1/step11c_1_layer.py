# import mask_1_os_book_no_bg                                         .step10_a as os_book_no_bg
# import mask_2_os_book_have_bg                                       .step10_a as os_book_have_bg
# import mask_3_os_book_and_paper_have_bg                             .step10_a as os_book_and_paper_have_bg
# import mask_4_os_book_and_paper_have_dtd_bg                         .step10_a as os_book_and_paper_have_dtd_bg

### 直接看 dtd_hdr_mix 的狀況
import mask_5_1_7l                 .step10_a as os_book_and_paper_have_dtd_hdr_mix_bg

############################################
### 1-7l-1_ch
mask_L7_ch = [os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch128_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch064_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch016_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch008_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch004_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch002_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch001_sig_bce_ep060.build()]
mask_L7_ch = [ exp.result_obj for exp in mask_L7_ch]
############################################
### 1-7l-2_layer
mask_layer = [os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L2_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L3_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L4_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L5_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L6_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L7_sig_bce_ep060.build(),
              os_book_and_paper_have_dtd_hdr_mix_bg.mask_h_bg_ch032_L8_sig_bce_ep060.build()]
mask_layer = [ exp.result_obj for exp in mask_layer]
