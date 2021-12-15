import step10_a_mask_5_2_bce_s020_6l as L6_bce_s20

####################################################################################################################################
### 2-6l_s01-1_ch
mask_L6_bce_s20_ch = [
    L6_bce_s20.mask_h_bg_ch128_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch064_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch016_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch008_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch004_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch002_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch001_sig_L6_ep060.build()]
mask_L6_bce_s20_ch = [ exp.result_obj for exp in mask_L6_bce_s20_ch]
############################################
### 2-6l_s01-2_ep
mask_L6_bce_s20_ep = [
    L6_bce_s20.mask_h_bg_ch128_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch064_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch016_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch008_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch004_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch002_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch001_sig_L6_ep060.build(),
    L6_bce_s20.mask_h_bg_ch128_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch064_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch032_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch016_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch008_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch004_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch002_sig_L6_ep200.build(),
    L6_bce_s20.mask_h_bg_ch001_sig_L6_ep200.build()]
mask_L6_bce_s20_ep = [ exp.result_obj for exp in mask_L6_bce_s20_ep]
############################################
### 2-6l_s01-4_no-concat_and_add
mask_L6_bce_s20_noC_and_add =  [
    L6_bce_s20.mask_h_bg_ch032_L6_2to2noC_sig_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_L6_2to3noC_sig_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_L6_2to4noC_sig_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_L6_2to5noC_sig_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_L6_2to6noC_sig_ep060.build(),
    L6_bce_s20.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build()]
mask_L6_bce_s20_noC_and_add = [ exp.result_obj for exp in mask_L6_bce_s20_noC_and_add]
