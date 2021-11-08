import mask_5_1_7l_unet2_block1.step10_a as mask_7l_block1

############################################
### 1-7l-1_ch
mask_7l_block1_ch = [
    mask_7l_block1.mask_h_bg_ch128_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch064_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch016_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch008_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch004_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch002_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch001_sig_ep060.build()]
mask_7l_block1_ch = [ exp.result_obj for exp in mask_7l_block1_ch]
############################################
### 1-7l-2_layer
mask_7l_block1_layer = [
    mask_7l_block1.mask_h_bg_ch032_2l_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_3l_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_4l_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_5l_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_6l_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_8l_sig_ep060.build()]
mask_7l_block1_layer = [ exp.result_obj for exp in mask_7l_block1_layer]
############################################
### 1-7l-3_noC
mask_7l_block1_layer_noC = [
    mask_7l_block1.mask_h_bg_ch032_7l_2to2noC_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_2to3noC_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_2to4noC_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_2to5noC_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_2to6noC_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_2to7noC_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_2to8noC_sig_ep060.build()]
mask_7l_block1_layer_noC = [ exp.result_obj for exp in mask_7l_block1_layer_noC]
############################################
### 1-7l-4_skip_add
mask_7l_block1_skip_add = [
    mask_7l_block1.mask_h_bg_ch032_2l_skipAdd_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_3l_skipAdd_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_4l_skipAdd_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_5l_skipAdd_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_6l_skipAdd_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_7l_skipAdd_sig_ep060.build(),
    mask_7l_block1.mask_h_bg_ch032_8l_skipAdd_sig_ep060.build()]
mask_7l_block1_skip_add = [ exp.result_obj for exp in mask_7l_block1_skip_add]
