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
    from step12_result_analyzer import Col_results_analyzer, Row_col_results_analyzer
    from step11c_3_sobel import  *
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
    #################################################################################################################################################################################################################
    ana_name = "3_1-sobel_k5_s001-1_ch"
    mask_sobel_k5_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s001_ch[:4],
                                                                   mask_sobel_k5_s001_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s001-2_ep"
    mask_sobel_k5_s001_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s001_ep[:8],
                                                                   mask_sobel_k5_s001_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s001-4_no_concat_and_add"
    mask_sobel_k5_s001_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s001_noC_and_add[:3] + [mask_sobel_k5_s001_ch[2]],
                                                                   mask_sobel_k5_s001_noC_and_add[3:] + [mask_sobel_k5_s001_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_1-sobel_k5_s020-1_ch"
    mask_sobel_k5_s020_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s020_ch[:4],
                                                                   mask_sobel_k5_s020_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s020-2_ep"
    mask_sobel_k5_s020_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s020_ep[:8],
                                                                   mask_sobel_k5_s020_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s020-4_no_concat_and_add"
    mask_sobel_k5_s020_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s020_noC_and_add[:3] + [mask_sobel_k5_s020_ch[2]],
                                                                   mask_sobel_k5_s020_noC_and_add[3:] + [mask_sobel_k5_s020_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_1-sobel_k5_s040-1_ch"
    mask_sobel_k5_s040_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s040_ch[:4],
                                                                   mask_sobel_k5_s040_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s040-2_ep"
    mask_sobel_k5_s040_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s040_ep[:8],
                                                                   mask_sobel_k5_s040_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s040-4_no_concat_and_add"
    mask_sobel_k5_s040_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s040_noC_and_add[:3] + [mask_sobel_k5_s040_ch[2]],
                                                                   mask_sobel_k5_s040_noC_and_add[3:] + [mask_sobel_k5_s040_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_1-sobel_k5_s060-1_ch"
    mask_sobel_k5_s060_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s060_ch[:4],
                                                                   mask_sobel_k5_s060_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s060-2_ep"
    mask_sobel_k5_s060_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s060_ep[:8],
                                                                   mask_sobel_k5_s060_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s060-4_no_concat_and_add"
    mask_sobel_k5_s060_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s060_noC_and_add[:3] + [mask_sobel_k5_s060_ch[2]],
                                                                   mask_sobel_k5_s060_noC_and_add[3:] + [mask_sobel_k5_s060_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_1-sobel_k5_s080-1_ch"
    mask_sobel_k5_s080_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s080_ch[:4],
                                                                   mask_sobel_k5_s080_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s080-2_ep"
    mask_sobel_k5_s080_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s080_ep[:8],
                                                                   mask_sobel_k5_s080_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s080-4_no_concat_and_add"
    mask_sobel_k5_s080_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s080_noC_and_add[:3] + [mask_sobel_k5_s080_ch[2]],
                                                                   mask_sobel_k5_s080_noC_and_add[3:] + [mask_sobel_k5_s080_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_1-sobel_k5_s100-1_ch"
    mask_sobel_k5_s100_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s100_ch[:4],
                                                                   mask_sobel_k5_s100_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s100-2_ep"
    mask_sobel_k5_s100_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s100_ep[:8],
                                                                   mask_sobel_k5_s100_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_1-sobel_k5_s100-4_no_concat_and_add"
    mask_sobel_k5_s100_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_sobel_k5_s100_noC_and_add[:3] + [mask_sobel_k5_s100_ch[2]],
                                                                   mask_sobel_k5_s100_noC_and_add[3:] + [mask_sobel_k5_s100_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_2-sobel_k5_s1,20,40,60,80,100-1_ch"
    mask_L6_bce_s1_10_20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[ mask_sobel_k5_s001_ch[:3] + mask_sobel_k5_s060_ch[:3],
                                                                  mask_sobel_k5_s020_ch[:3] + mask_sobel_k5_s080_ch[:3],
                                                                  mask_sobel_k5_s040_ch[:3] + mask_sobel_k5_s100_ch[:3],
                                                                  ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_2-sobel_k5_s1,20,40,60,80,100-4_no_concat_and_add"
    mask_L6_bce_s1_10_20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[ [mask_sobel_k5_s001_ch[2]] + mask_sobel_k5_s001_noC_and_add + [mask_sobel_k5_s001_ch[2]],
                                                                  [mask_sobel_k5_s020_ch[2]] + mask_sobel_k5_s020_noC_and_add + [mask_sobel_k5_s020_ch[2]],
                                                                  [mask_sobel_k5_s040_ch[2]] + mask_sobel_k5_s040_noC_and_add + [mask_sobel_k5_s040_ch[2]],
                                                                  [mask_sobel_k5_s060_ch[2]] + mask_sobel_k5_s060_noC_and_add + [mask_sobel_k5_s060_ch[2]],
                                                                  [mask_sobel_k5_s080_ch[2]] + mask_sobel_k5_s080_noC_and_add + [mask_sobel_k5_s080_ch[2]],
                                                                  [mask_sobel_k5_s100_ch[2]] + mask_sobel_k5_s100_noC_and_add + [mask_sobel_k5_s100_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3_3_ch032-sobel_k5_s1~260"
    mask_L6_bce_s1_10_20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[ mask_ch032_sobel_k5_s1_260[  : 5],
                                                                  mask_ch032_sobel_k5_s1_260[ 5:10],
                                                                  mask_ch032_sobel_k5_s1_260[10:  ],
                                                                  ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
