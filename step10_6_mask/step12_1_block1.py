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
    from step11c_1_block import  *
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
    ### 直接看 dtd_hdr_mix 的狀況
    #################################################################################################################################################################################################################
    ana_name = "1-7l_block1-1_ch-1_row"
    mask_L7_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_ch, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "1-7l_block1-1_ch-2_row"
    mask_layer_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_ch[:4],
                                                                   mask_L7_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ########################################################################################
    ana_name = "1-7l_block1-2_layer-1_row"
    mask_L7_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_layer, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "1-7l_block1-2_layer-2_row"
    mask_layer_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_layer[:4],
                                                                   mask_L7_block1_layer[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ########################################################################################
    ana_name = "1-7l_block1-3_noC-1_row"
    mask_L7_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_layer_noC, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "1-7l_block1-3_noC-2_row"
    mask_layer_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_layer_noC[:4],
                                                                   mask_L7_block1_layer_noC[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ########################################################################################
    ana_name = "1-7l_block1-4_skip_add-1_row"
    mask_L7_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_skip_add, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "1-7l_block1-4_skip_add-2_row"
    mask_layer_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_skip_add[:4],
                                                                   mask_L7_block1_skip_add[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)

