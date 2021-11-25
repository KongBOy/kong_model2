import mask_5_1_7l_unet2_block1.step10_a as mask_L7_block1

############################################
### 1-7l-1_ch
mask_L7_block1_ch = [
    mask_L7_block1.mask_h_bg_ch128_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch064_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch016_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch008_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch004_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch002_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch001_sig_ep060.build()]
mask_L7_block1_ch = [ exp.result_obj for exp in mask_L7_block1_ch]
############################################
### 1-7l-2_layer
mask_L7_block1_layer = [
    mask_L7_block1.mask_h_bg_ch032_L2_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L3_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L4_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L5_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L6_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L8_sig_ep060.build()]
mask_L7_block1_layer = [ exp.result_obj for exp in mask_L7_block1_layer]
############################################
### 1-7l-3_noC
mask_L7_block1_layer_noC = [
    mask_L7_block1.mask_h_bg_ch032_L7_2to2noC_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_2to3noC_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_2to4noC_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_2to5noC_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_2to6noC_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_2to7noC_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_2to8noC_sig_ep060.build()]
mask_L7_block1_layer_noC = [ exp.result_obj for exp in mask_L7_block1_layer_noC]
############################################
### 1-7l-4_skip_add
mask_L7_block1_skip_add = [
    mask_L7_block1.mask_h_bg_ch032_L2_skipAdd_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L3_skipAdd_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L4_skipAdd_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L5_skipAdd_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L6_skipAdd_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L7_skipAdd_sig_ep060.build(),
    mask_L7_block1.mask_h_bg_ch032_L8_skipAdd_sig_ep060.build()]
mask_L7_block1_skip_add = [ exp.result_obj for exp in mask_L7_block1_skip_add]
