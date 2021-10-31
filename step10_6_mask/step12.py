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
    from step11c import  *
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
    ana_name = "1-7l-1_ch"
    mask_7l_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}", ana_what="mask", col_results=mask_7l_ch, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_col_results_single_see(see_num=0, single_see_multiprocess=False)
    ############################################
    ana_name = "1-7l-2_layer"
    mask_layer_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_layer[:4],
                                                                   mask_layer[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "2_1-6l_bce_s01-1_ch"
    mask_6l_bce_s01_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s01_ch[:4],
                                                                   mask_6l_bce_s01_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s01-2_ep"
    mask_6l_bce_s01_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s01_ep[:8],
                                                                   mask_6l_bce_s01_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s01-4_no_concat_and_add"
    mask_6l_bce_s01_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s01_noC_and_add[:3] + [mask_6l_bce_s01_ch[2]],
                                                                   mask_6l_bce_s01_noC_and_add[3:] + [mask_6l_bce_s01_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2_1-6l_bce_s10-1_ch"
    mask_6l_bce_s10_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s10_ch[:4],
                                                                   mask_6l_bce_s10_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s10-2_ep"
    mask_6l_bce_s10_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s10_ep[:8],
                                                                   mask_6l_bce_s10_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s10-4_no_concat_and_add"
    mask_6l_bce_s10_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s10_noC_and_add[:3] + [mask_6l_bce_s10_ch[2]],
                                                                   mask_6l_bce_s10_noC_and_add[3:] + [mask_6l_bce_s10_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2_1-6l_bce_s20-1_ch"
    mask_6l_bce_s20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s20_ch[:4],
                                                                   mask_6l_bce_s20_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s20-2_ep"
    mask_6l_bce_s20_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s20_ep[:8],
                                                                   mask_6l_bce_s20_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s20-4_no_concat_and_add"
    mask_6l_bce_s20_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s20_noC_and_add[:3] + [mask_6l_bce_s20_ch[2]],
                                                                   mask_6l_bce_s20_noC_and_add[3:] + [mask_6l_bce_s20_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2_1-6l_bce_s40-1_ch"
    mask_6l_bce_s40_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  col_results=mask_6l_bce_s40_ch, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s40-2_ep"
    mask_6l_bce_s40_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s40_ep[:4],
                                                                   mask_6l_bce_s40_ep[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s40-4_no_concat_and_add"
    mask_6l_bce_s40_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s40_noC_and_add[:3] + [mask_6l_bce_s40_ch[2]],
                                                                   mask_6l_bce_s40_noC_and_add[3:] + [mask_6l_bce_s40_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2_1-6l_bce_s60-1_ch"
    mask_6l_bce_s60_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  col_results=mask_6l_bce_s60_ch, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s60-2_ep"
    mask_6l_bce_s60_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s60_ep[:4],
                                                                   mask_6l_bce_s60_ep[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s60-4_no_concat_and_add"
    mask_6l_bce_s60_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s60_noC_and_add[:3] + [mask_6l_bce_s60_ch[2]],
                                                                   mask_6l_bce_s60_noC_and_add[3:] + [mask_6l_bce_s60_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "2_1-6l_bce_s80-1_ch"
    mask_6l_bce_s80_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  col_results=mask_6l_bce_s80_ch, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s80-2_ep"
    mask_6l_bce_s80_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s80_ep[:4],
                                                                   mask_6l_bce_s80_ep[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_1-6l_bce_s80-4_no_concat_and_add"
    mask_6l_bce_s80_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s80_noC_and_add[:3] + [mask_6l_bce_s80_ch[2]],
                                                                   mask_6l_bce_s80_noC_and_add[3:] + [mask_6l_bce_s80_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "2_2-6l_bce_s1,10,20-1_ch"
    mask_6l_bce_s1_10_20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[ mask_6l_bce_s01_ch,
                                                                  mask_6l_bce_s10_ch,
                                                                  mask_6l_bce_s20_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_2-6l_bce_s1,10,20-2_ep"
    mask_6l_bce_s1_10_20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[ mask_6l_bce_s01_ep[:8 - 4],
                                                                  mask_6l_bce_s01_ep[8 + 4:],
                                                                  mask_6l_bce_s10_ep[:8 - 4],
                                                                  mask_6l_bce_s10_ep[8 + 4:],
                                                                  mask_6l_bce_s20_ep[:8 - 4],
                                                                  mask_6l_bce_s20_ep[8 + 4:] ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "2_2-6l_bce_s1,10,20-4_no_concat_and_add"
    mask_6l_bce_s1_10_20_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[ mask_6l_bce_s01_noC_and_add + [mask_6l_bce_s01_ch[2]],
                                                                  mask_6l_bce_s10_noC_and_add + [mask_6l_bce_s10_ch[2]],
                                                                  mask_6l_bce_s20_noC_and_add + [mask_6l_bce_s20_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "3-6l_just_sobel_k5_s001~080-1_ch"
    mask_sobel_k5_s001_s080_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                    mask_sobel_k5_s001_ch,
                                                                    mask_sobel_k5_s020_ch,
                                                                    mask_sobel_k5_s040_ch,
                                                                    mask_sobel_k5_s060_ch,
                                                                    mask_sobel_k5_s080_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_1-6l_bce_sobel_k3-1_ch"
    mask_6l_sobel_k3_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k3_ch[:4],
                                                                   mask_6l_sobel_k3_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k3-2_ep"
    mask_6l_sobel_k3_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k3_ep[:8],
                                                                   mask_6l_sobel_k3_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k3-4_no_concat_and_add"
    mask_6l_sobel_k3_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k3_noC_and_add[:3],
                                                                   mask_6l_sobel_k3_noC_and_add[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_1-6l_bce_sobel_k5-1_ch"
    mask_6l_sobel_k5_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k5_ch[:4],
                                                                   mask_6l_sobel_k5_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k5-2_ep"
    mask_6l_sobel_k5_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k5_ep[:8],
                                                                   mask_6l_sobel_k5_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k5-4_no_concat_and_add"
    mask_6l_sobel_k5_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k5_noC_and_add[:3],
                                                                   mask_6l_sobel_k5_noC_and_add[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_1-6l_bce_sobel_k7-1_ch"
    mask_6l_sobel_k7_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k7_ch[:4],
                                                                   mask_6l_sobel_k7_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k7-2_ep"
    mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k7_ep[:8],
                                                                   mask_6l_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_1-6l_bce_sobel_k7-4_no_concat_and_add"
    mask_6l_sobel_k7_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k7_noC_and_add[:3],
                                                                   mask_6l_sobel_k7_noC_and_add[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_2-6l_bce__sobel_k0357-1_ch"
    mask_6l_sobel_k7_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_6l_bce_s01_ch,
                                                                    mask_6l_sobel_k3_ch,
                                                                    mask_6l_sobel_k5_ch,
                                                                    mask_6l_sobel_k7_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_2-6l_bce__sobel_k0357-2_ep"
    mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s01_ep[:8],
                                                                   mask_6l_bce_s01_ep[8:],
                                                                   mask_6l_sobel_k3_ep[:8],
                                                                   mask_6l_sobel_k3_ep[8:],
                                                                   mask_6l_sobel_k5_ep[:8],
                                                                   mask_6l_sobel_k5_ep[8:],
                                                                   mask_6l_sobel_k7_ep[:8],
                                                                   mask_6l_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_2-6l_bce__sobel_k0357-4_no_concat_and_add"
    mask_6l_sobel_k7_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s01_noC_and_add,
                                                                   mask_6l_sobel_k3_noC_and_add,
                                                                   mask_6l_sobel_k5_noC_and_add,
                                                                   mask_6l_sobel_k7_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_3-6l_bce__sobel_k5_s20~100-1_ch"
    mask_6l_sobel_k5_s20_s100_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_6l_bce_s01_ch,
                                                                    mask_6l_sobel_k5_s20_ch,
                                                                    mask_6l_sobel_k5_s40_ch,
                                                                    mask_6l_sobel_k5_s60_ch,
                                                                    mask_6l_sobel_k5_s80_ch,
                                                                    mask_6l_sobel_k5_s100_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    # ana_name = "5_3-6l_bce_sobel_k5_s20~100-2_ep"  ### 生太久且效果好像都差不多，所以先註解掉
    # mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
    #                                               ana_what="mask",
    #                                               row_col_results=[mask_6l_bce_s01_ep[:8],
    #                                                                mask_6l_bce_s01_ep[8:],
    #                                                                mask_6l_sobel_k3_ep[:8],
    #                                                                mask_6l_sobel_k3_ep[8:],
    #                                                                mask_6l_sobel_k5_ep[:8],
    #                                                                mask_6l_sobel_k5_ep[8:],
    #                                                                mask_6l_sobel_k7_ep[:8],
    #                                                                mask_6l_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
    #                               .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ana_name = "5_3-6l_bce__sobel_k5_s20~100-4_no_concat_and_add"
    mask_6l_sobel_k5_s20_s100_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_bce_s01_noC_and_add,
                                                                   mask_6l_sobel_k5_s20_noC_and_add,
                                                                   mask_6l_sobel_k5_s40_noC_and_add,
                                                                   mask_6l_sobel_k5_s60_noC_and_add,
                                                                   mask_6l_sobel_k5_s80_noC_and_add,
                                                                   mask_6l_sobel_k5_s100_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_3-6l_bce__sobel_k5_s40~140-1_ch"
    mask_6l_sobel_k5_s40_s140_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_6l_sobel_k5_s40_ch,
                                                                    mask_6l_sobel_k5_s60_ch,
                                                                    mask_6l_sobel_k5_s80_ch,
                                                                    mask_6l_sobel_k5_s100_ch,
                                                                    mask_6l_sobel_k5_s120_ch,
                                                                    mask_6l_sobel_k5_s140_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    # ana_name = "5_3-6l_bce__sobel_k5_s20~100-2_ep"  ### 生太久且效果好像都差不多，所以先註解掉
    # mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
    #                                               ana_what="mask",
    #                                               row_col_results=[mask_6l_bce_s01_ep[:8],
    #                                                                mask_6l_bce_s01_ep[8:],
    #                                                                mask_6l_sobel_k3_ep[:8],
    #                                                                mask_6l_sobel_k3_ep[8:],
    #                                                                mask_6l_sobel_k5_ep[:8],
    #                                                                mask_6l_sobel_k5_ep[8:],
    #                                                                mask_6l_sobel_k7_ep[:8],
    #                                                                mask_6l_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
    #                                         .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    # ana_name = "5_3-6l_bce_sobel_k5_s20~100-4_no_concat_and_add"  ### 直覺就不行，有空再來研究看看 先趕meeting
    # mask_6l_sobel_k5_s20_s100_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
    #                                               ana_what="mask",
    #                                               row_col_results=[mask_6l_bce_s01_noC_and_add,
    #                                                                mask_6l_sobel_k5_s20_noC_and_add,
    #                                                                mask_6l_sobel_k5_s40_noC_and_add,
    #                                                                mask_6l_sobel_k5_s60_noC_and_add,
    #                                                                mask_6l_sobel_k5_s80_noC_and_add,
    #                                                                mask_6l_sobel_k5_s100_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
    #                                         .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "5_3-6l_bce__sobel_k5_s60~160-1_ch"
    mask_6l_sobel_k5_s60_s160_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_6l_sobel_k5_s60_ch,
                                                                    mask_6l_sobel_k5_s80_ch,
                                                                    mask_6l_sobel_k5_s100_ch,
                                                                    mask_6l_sobel_k5_s120_ch,
                                                                    mask_6l_sobel_k5_s140_ch,
                                                                    mask_6l_sobel_k5_s160_ch], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    ############################################
    # ana_name = "5_3-6l_bce_sobel_k5_s20~100-2_ep"  ### 生太久且效果好像都差不多，所以先註解掉
    # mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
    #                                               ana_what="mask",
    #                                               row_col_results=[mask_6l_bce_s01_ep[:8],
    #                                                                mask_6l_bce_s01_ep[8:],
    #                                                                mask_6l_sobel_k3_ep[:8],
    #                                                                mask_6l_sobel_k3_ep[8:],
    #                                                                mask_6l_sobel_k5_ep[:8],
    #                                                                mask_6l_sobel_k5_ep[8:],
    #                                                                mask_6l_sobel_k7_ep[:8],
    #                                                                mask_6l_sobel_k7_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
    #                                         .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    # ############################################
    # ana_name = "5_3-6l_bce_sobel_k5_s20~100-4_no_concat_and_add"  ### 直覺就不行，有空再來研究看看 先趕meeting
    # mask_6l_sobel_k5_s20_s100_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
    #                                               ana_what="mask",
    #                                               row_col_results=[mask_6l_bce_s01_noC_and_add,
    #                                                                mask_6l_sobel_k5_s20_noC_and_add,
    #                                                                mask_6l_sobel_k5_s40_noC_and_add,
    #                                                                mask_6l_sobel_k5_s60_noC_and_add,
    #                                                                mask_6l_sobel_k5_s80_noC_and_add,
    #                                                                mask_6l_sobel_k5_s100_noC_and_add], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
    #                                         .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "6-6l_tv_s01~20_bce-1_ch"
    mask_6l_tv_s01_20_bce_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[ 
                                                                    mask_6l_tv_bce_ch,
                                                                    mask_6l_tv_s04_bce_ch,
                                                                    mask_6l_tv_s08_bce_ch,
                                                                    mask_6l_tv_s12_bce_ch,
                                                                    mask_6l_tv_s16_bce_ch,
                                                                    mask_6l_tv_s20_bce_ch
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                                # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    ana_name = "6-6l_tv_s01~80_bce-1_ch"
    mask_6l_tv_s01_80_bce_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                    mask_6l_tv_bce_ch,
                                                                    mask_6l_tv_s20_bce_ch,
                                                                    mask_6l_tv_s40_bce_ch,
                                                                    mask_6l_tv_s60_bce_ch,
                                                                    mask_6l_tv_s80_bce_ch
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                                # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "7-6l_tv_s01_sobel_k5_s01,80~140-1_ch"
    mask_6l_tv_s01_sobel_k5_s01_80_140_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                    mask_6l_tv_s01_sobel_k5_s001_ch,
                                                                    mask_6l_tv_s01_sobel_k5_s080_ch,
                                                                    mask_6l_tv_s01_sobel_k5_s100_ch,
                                                                    mask_6l_tv_s01_sobel_k5_s120_ch,
                                                                    mask_6l_tv_s01_sobel_k5_s140_ch
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                                # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "8-tv_s01,04_bce_sobel_k5_s001,100_ch"
    mask_6l_tv_mix_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                    mask_6l_bce_s01_ch,               ### just_bce
                                                                    mask_sobel_k5_s001_ch,    ### just_sobel_k5_s001
                                                                    mask_6l_tv_s01_bce_sobel_k5_s001_ch,  ### just_tv
                                                                    mask_6l_tv_s01_bce_sobel_k5_s100_ch,  ### tv + bce
                                                                    mask_6l_tv_s04_bce_sobel_k5_s001_ch,  ### tv + sobel_k5_s001
                                                                    mask_6l_tv_s04_bce_sobel_k5_s100_ch,  ### sobel_k5 + bce
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                                # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "9-6l_mix-1_ch"
    mask_6l_mix_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                ana_what="mask",
                                                row_col_results=[
                                                                mask_6l_bce_s01_ch,               ### just_bce
                                                                mask_sobel_k5_s001_ch,    ### just_sobel_k5_s001
                                                                mask_6l_tv_ch,                    ### just_tv
                                                                mask_6l_tv_bce_ch,                ### tv + bce
                                                                mask_6l_tv_s01_sobel_k5_s001_ch,  ### tv + sobel_k5_s001
                                                                mask_6l_sobel_k5_ch,      ### sobel_k5 + bce
                                                                ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "9-6l_tv_vs_tv_bce_vs_tv_sobel_k5_s001-1_ch"
    mask_6l_tv_vs_tv_bce_vs_tv_sobel_k5_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                    mask_6l_tv_ch,
                                                                    mask_6l_tv_bce_ch,
                                                                    mask_6l_tv_s01_sobel_k5_s001_ch
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                                # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
