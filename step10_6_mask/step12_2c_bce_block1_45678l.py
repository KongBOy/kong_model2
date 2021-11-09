'''
目前只有 step12 一定需要切換資料夾到 該komg_model所在的資料夾 才能執行喔！
'''
if(__name__ == "__main__"):
    ##########################################################################################################################################################################################################################################################################################
    ### 把 current_dir 轉回到 kong_model 裡面
    import os
    import sys
    curr_path = os.getcwd()
    curr_layer = len(curr_path.split("\\")) - 1                ### 看 目前執行python的位置在哪一層， -1 是 因為 為了配合下面.index() 從0開始算
    kong_layer = curr_path.split("\\").index("kong_model2")    ### 看kong_model2 在哪一層
    back_to_kong_layer_amount = curr_layer - kong_layer        ### 看 目前執行python的位置在哪一層 到 kong_model2 差幾層
    for _ in range(back_to_kong_layer_amount): os.chdir("..")  ### 看差幾層 往前跳 幾次dir
    sys.path.append(".")                                         ### 把 kong_model2 加進 sys.path
    ##########################################################################################################################################################################################################################################################################################
    from step12_result_analyzer import Col_results_analyzer, Row_col_results_analyzer, Bm_Rec_exps_analyze
    from step11c_2c_bce_block1_45678l import  *
    ##########################################################################################################################################################################################################################################################################################
    mask_ana_dir = "mask"
    """
    以下留下一些example這樣子
    core_amount == 7 是因為 目前 see_amount == 7 ，想 一個core 一個see
    task_amount == 7 是因為 目前 see_amount == 7

    single_see_multiprocess == True 代表 see內 還要 切 multiprocess，
    single_see_core_amount == 2 代表切2分

    所以總共會有 7*2 = 14 份 process 要同時處理，
    但建議不要用，已經測過，爆記憶體了
    """
    #################################################################################################################################################################################################################
    ana_name = "2c_1-l4_ch064_block1_bce_s001_100"
    l4_ch064_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_4l_ch064[:3],
                                                                   bce_block1_4l_ch064[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ana_name = "2c_1-l4_ch032_block1_bce_s001_100"
    l4_ch032_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_4l_ch032[:3],
                                                                   bce_block1_4l_ch032[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ana_name = "2c_1-l4_ch016_block1_bce_s001_100"
    l4_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_4l_ch016[:3],
                                                                   bce_block1_4l_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ana_name = "2c_1-l4_ch008_block1_bce_s001_100"
    l4_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_4l_ch008[:3],
                                                                   bce_block1_4l_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ################################################################################
    ana_name = "2c_2-l4_ch64,32,16,8_block1_bce_s001_100"
    l4_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_4l_ch064,
                                                                   bce_block1_4l_ch032,
                                                                   bce_block1_4l_ch016,
                                                                   bce_block1_4l_ch008], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2c_1-l5_ch032_block1_bce_s001_100"
    l5_ch032_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_5l_ch032[:3],
                                                                   bce_block1_5l_ch032[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2c_1-l6_ch016_block1_bce_s001_100"
    l6_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_6l_ch016[:3],
                                                                   bce_block1_6l_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2c_1-l7_ch008_block1_bce_s001_100"
    l7_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_7l_ch008[:3],
                                                                   bce_block1_7l_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2c_1-l8_ch004_block1_bce_s001_100"
    l8_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_8l_ch004[:3],
                                                                   bce_block1_8l_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "2c_2-l45678_ch64,32,16,8,4_block1_bce_s001_100"
    l45678_ch64_32_16_8_4_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_4l_ch064,
                                                                   bce_block1_5l_ch032,
                                                                   bce_block1_6l_ch016,
                                                                   bce_block1_7l_ch008,
                                                                   bce_block1_8l_ch004], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)