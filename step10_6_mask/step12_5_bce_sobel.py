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
    from step11c_2_bce import  *
    from step11c_3_sobel import *
    from step11c_5_bce_sobel import  *
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
    #################################################################################################################################################################################################################
    ana_name = "5_1-6l_bce_sobel_k3-1_ch"
    mask_L6_bce_sobel_k3_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k3_ch[:4],
                                                                   mask_L6_bce_sobel_k3_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k3-2_ep"
    mask_L6_bce_sobel_k3_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k3_ep[:8],
                                                                   mask_L6_bce_sobel_k3_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k3-4_no_concat_and_add"
    mask_L6_bce_sobel_k3_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k3_noC_and_add[:3],
                                                                   mask_L6_bce_sobel_k3_noC_and_add[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_1-6l_bce_sobel_k5-1_ch"
    mask_L6_bce_sobel_k5_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k5_s001_ch[:4],
                                                                   mask_L6_bce_sobel_k5_s001_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k5-2_ep"
    mask_L6_bce_sobel_k5_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k5_ep[:8],
                                                                   mask_L6_bce_sobel_k5_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k5-4_no_concat_and_add"
    mask_L6_bce_sobel_k5_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k5_noC_and_add[:3],
                                                                   mask_L6_bce_sobel_k5_noC_and_add[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_1-6l_bce_sobel_k7-1_ch"
    mask_L6_bce_sobel_k7_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k7_ch[:4],
                                                                   mask_L6_bce_sobel_k7_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k7-2_ep"
    mask_L6_bce_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k7_ep[:8],
                                                                   mask_L6_bce_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k7-4_no_concat_and_add"
    mask_L6_bce_sobel_k7_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_sobel_k7_noC_and_add[:3],
                                                                   mask_L6_bce_sobel_k7_noC_and_add[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_2-6l_bce_sobel_k0357-1_ch"
    mask_L6_bce_sobel_k7_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_L6_bce_s01_ch,
                                                                    mask_L6_bce_sobel_k3_ch,
                                                                    mask_L6_bce_sobel_k5_s001_ch,
                                                                    mask_L6_bce_sobel_k7_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_2-6l_bce_sobel_k0357-2_ep"
    mask_L6_bce_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_s01_ep[:8],
                                                                   mask_L6_bce_s01_ep[8:],
                                                                   mask_L6_bce_sobel_k3_ep[:8],
                                                                   mask_L6_bce_sobel_k3_ep[8:],
                                                                   mask_L6_bce_sobel_k5_ep[:8],
                                                                   mask_L6_bce_sobel_k5_ep[8:],
                                                                   mask_L6_bce_sobel_k7_ep[:8],
                                                                   mask_L6_bce_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_2-6l_bce_sobel_k0357-4_no_concat_and_add"
    mask_L6_bce_sobel_k7_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_s01_noC_and_add,
                                                                   mask_L6_bce_sobel_k3_noC_and_add,
                                                                   mask_L6_bce_sobel_k5_noC_and_add,
                                                                   mask_L6_bce_sobel_k7_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_3-6l_bce_sobel_k5_s20~100-1_ch"
    mask_L6_bce_sobel_k5_s20_s100_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_L6_bce_s01_ch,
                                                                    mask_L6_bce_sobel_k5_s020_ch,
                                                                    mask_L6_bce_sobel_k5_s040_ch,
                                                                    mask_L6_bce_sobel_k5_s060_ch,
                                                                    mask_L6_bce_sobel_k5_s080_ch,
                                                                    mask_L6_bce_sobel_k5_s100_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_3-6l_bce_sobel_k5_s20~100-4_no_concat_and_add"
    mask_L6_bce_sobel_k5_s20_s100_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_s01_noC_and_add,
                                                                   mask_L6_bce_sobel_k5_s20_noC_and_add,
                                                                   mask_L6_bce_sobel_k5_s40_noC_and_add,
                                                                   mask_L6_bce_sobel_k5_s60_noC_and_add,
                                                                   mask_L6_bce_sobel_k5_s80_noC_and_add,
                                                                   mask_L6_bce_sobel_k5_s100_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_3-6l_bce_sobel_k5_s40~140-1_ch"
    mask_L6_bce_sobel_k5_s40_s140_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_L6_bce_sobel_k5_s040_ch,
                                                                    mask_L6_bce_sobel_k5_s060_ch,
                                                                    mask_L6_bce_sobel_k5_s080_ch,
                                                                    mask_L6_bce_sobel_k5_s100_ch,
                                                                    mask_L6_bce_sobel_k5_s120_ch,
                                                                    mask_L6_bce_sobel_k5_s140_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    # ana_name = "5_3-6l_bce_sobel_k5_s20~100-4_no_concat_and_add"  ### 直覺就不行，有空再來研究看看 先趕meeting
    # mask_L6_bce_sobel_k5_s20_s100_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
    #                                               ana_what="mask",
    #                                               row_col_results=[mask_L6_bce_s01_noC_and_add,
    #                                                                mask_L6_bce_sobel_k5_s20_noC_and_add,
    #                                                                mask_L6_bce_sobel_k5_s40_noC_and_add,
    #                                                                mask_L6_bce_sobel_k5_s60_noC_and_add,
    #                                                                mask_L6_bce_sobel_k5_s80_noC_and_add,
    #                                                                mask_L6_bce_sobel_k5_s100_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
    #                                         .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_3-6l_bce_sobel_k5_s60~160-1_ch"
    mask_L6_bce_sobel_k5_s60_s160_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_L6_bce_sobel_k5_s060_ch,
                                                                    mask_L6_bce_sobel_k5_s080_ch,
                                                                    mask_L6_bce_sobel_k5_s100_ch,
                                                                    mask_L6_bce_sobel_k5_s120_ch,
                                                                    mask_L6_bce_sobel_k5_s140_ch,
                                                                    mask_L6_bce_sobel_k5_s160_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "5_3-6l_bce_sobel_k5_s60~160-1_ch"
    mask_L6_bce_sobel_k5_s1_s160_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_L6_bce_sobel_k5_s060_ch,
                                                                    mask_L6_bce_sobel_k5_s080_ch,
                                                                    mask_L6_bce_sobel_k5_s100_ch,
                                                                    mask_L6_bce_sobel_k5_s120_ch,
                                                                    mask_L6_bce_sobel_k5_s140_ch,
                                                                    mask_L6_bce_sobel_k5_s160_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)

    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_4-6l_bce_sobel_k5_s001~160-1_ch"
    mask_L6_bce_sobel_k5_s60_s160_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_L6_bce_s01_ch          [:3] + mask_L6_bce_sobel_k5_s060_ch[:3],
                                                                    mask_sobel_k5_s001_ch       [:3] + mask_L6_bce_sobel_k5_s120_ch[:3],
                                                                    mask_L6_bce_sobel_k5_s001_ch[:3] + mask_L6_bce_sobel_k5_s160_ch[:3],
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
