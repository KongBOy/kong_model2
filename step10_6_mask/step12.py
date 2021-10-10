'''
目前只有 step12 一定需要切換資料夾到 該komg_model所在的資料夾 才能執行喔！
'''
if(__name__ == "__main__"):
    #############################################################################################################################################################################################################
    ### 把 current_dir 轉回到 kong_model 裡面
    import os
    import sys
    curr_path = os.getcwd()
    curr_layer = len(curr_path.split("\\")) - 1                ### 看 目前執行python的位置在哪一層， -1 是 因為 為了配合下面.index() 從0開始算
    kong_layer = curr_path.split("\\").index("kong_model2")    ### 看kong_model2 在哪一層
    back_to_kong_layer_amount = curr_layer - kong_layer        ### 看 目前執行python的位置在哪一層 到 kong_model2 差幾層
    for _ in range(back_to_kong_layer_amount): os.chdir("..")  ### 看差幾層 往前跳 幾次dir
    sys.path.append(".")                                         ### 把 kong_model2 加進 sys.path
    #############################################################################################################################################################################################################
    from step12_result_analyzer import Col_results_analyzer, Row_col_results_analyzer, Bm_Rec_exps_analyze
    from step11_c import  *
    #############################################################################################################################################################################################################
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
    ####################################################################################################################################
    ### 1-7l-1_ch
    mask_7l_ch_results = [ exp.result_obj for exp in mask_7l_ch]
    mask_7l_ch_analyze = Col_results_analyzer(ana_describe=f"{mask_ana_dir}/1-7l-1_ch", ana_what="mask", col_results=mask_7l_ch_results, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_7l_ch_analyze.analyze_col_results_single_see(see_num=0, single_see_multiprocess=False)
    ############################################
    ### 1-7l-2_layer
    mask_layer_results = [ exp.result_obj for exp in mask_layer]
    mask_layer_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/1-7l-2_layer",
                                                  ana_what="mask",
                                                  row_col_results=[mask_layer_results[:4],
                                                                   mask_layer_results[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_layer_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ####################################################################################################################################
    ####################################################################################################################################
    ### 2-6l-1_ch
    mask_6l_ch_results = [ exp.result_obj for exp in mask_6l_ch]
    mask_6l_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/2-6l-1_ch",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_ch_results[:4],
                                                                   mask_6l_ch_results[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_ch_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 2-6l-2_ep
    mask_6l_ep_results = [ exp.result_obj for exp in mask_6l_ep]
    mask_6l_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/2-6l-2_ep",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_ep_results[:8],
                                                                   mask_6l_ep_results[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_ep_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 2-6l-4_no-concat_and_add
    mask_6l_noC_and_add_results = [ exp.result_obj for exp in mask_6l_noC_and_add]
    mask_6l_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/2-6l-4_no-concat_and_add",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_noC_and_add_results[:3],
                                                                   mask_6l_noC_and_add_results[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_noC_and_add_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ####################################################################################################################################
    ####################################################################################################################################
    ### 3-6l_sobel_k3-1_ch
    mask_6l_sobel_k3_ch_results = [ exp.result_obj for exp in mask_6l_sobel_k3_ch]
    mask_6l_sobel_k3_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k3-1_ch",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k3_ch_results[:4],
                                                                   mask_6l_sobel_k3_ch_results[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k3_ch_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 3-6l_sobel_k3-2_ep
    mask_6l_sobel_k3_ep_results = [ exp.result_obj for exp in mask_6l_sobel_k3_ep]
    mask_6l_sobel_k3_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k3-2_ep",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k3_ep_results[:8],
                                                                   mask_6l_sobel_k3_ep_results[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k3_ep_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 3-6l_sobel_k3-4_no-concat_and_add
    mask_6l_sobel_k3_noC_and_add_results = [ exp.result_obj for exp in mask_6l_sobel_k3_noC_and_add]
    mask_6l_sobel_k3_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k3-4_no-concat_and_add",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k3_noC_and_add_results[:3],
                                                                   mask_6l_sobel_k3_noC_and_add_results[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k3_noC_and_add_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ####################################################################################################################################
    ### 3-6l_sobel_k5-1_ch
    mask_6l_sobel_k5_ch_results = [ exp.result_obj for exp in mask_6l_sobel_k5_ch]
    mask_6l_sobel_k5_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k5-1_ch",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k5_ch_results[:4],
                                                                   mask_6l_sobel_k5_ch_results[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k5_ch_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 3-6l_sobel_k5-2_ep
    mask_6l_sobel_k5_ep_results = [ exp.result_obj for exp in mask_6l_sobel_k5_ep]
    mask_6l_sobel_k5_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k5-2_ep",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k5_ep_results[:8],
                                                                   mask_6l_sobel_k5_ep_results[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k5_ep_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 3-6l_sobel_k5-4_no-concat_and_add
    mask_6l_sobel_k5_noC_and_add_results = [ exp.result_obj for exp in mask_6l_sobel_k5_noC_and_add]
    mask_6l_sobel_k5_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k5-4_no-concat_and_add",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k5_noC_and_add_results[:3],
                                                                   mask_6l_sobel_k5_noC_and_add_results[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k5_noC_and_add_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ####################################################################################################################################
    ### 3-6l_sobel_k7-1_ch
    mask_6l_sobel_k7_ch_results = [ exp.result_obj for exp in mask_6l_sobel_k7_ch]
    mask_6l_sobel_k7_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k7-1_ch",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k7_ch_results[:4],
                                                                   mask_6l_sobel_k7_ch_results[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k7_ch_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 3-6l_sobel_k7-2_ep
    mask_6l_sobel_k7_ep_results = [ exp.result_obj for exp in mask_6l_sobel_k7_ep]
    mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k7-2_ep",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k7_ep_results[:8],
                                                                   mask_6l_sobel_k7_ep_results[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k7_ep_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 3-6l_sobel_k7-4_no-concat_and_add
    mask_6l_sobel_k7_noC_and_add_results = [ exp.result_obj for exp in mask_6l_sobel_k7_noC_and_add]
    mask_6l_sobel_k7_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/3-6l_sobel_k7-4_no-concat_and_add",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_sobel_k7_noC_and_add_results[:3],
                                                                   mask_6l_sobel_k7_noC_and_add_results[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k7_noC_and_add_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ####################################################################################################################################
    ### 4-6l_sobel_k0357-1_ch
    mask_6l_sobel_k7_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/4-6l_sobel_k0357-1_ch",
                                                  ana_what="mask",
                                                  row_col_results=[ mask_6l_ch_results,
                                                                    mask_6l_sobel_k3_ch_results,
                                                                    mask_6l_sobel_k5_ch_results,
                                                                    mask_6l_sobel_k7_ch_results], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k7_ch_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ### 4-6l_sobel_k0357-2_ep
    mask_6l_sobel_k7_ep_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/4-6l_sobel_k0357-2_ep",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_ep_results[:8],
                                                                   mask_6l_ep_results[8:],
                                                                   mask_6l_sobel_k3_ep_results[:8],
                                                                   mask_6l_sobel_k3_ep_results[8:],
                                                                   mask_6l_sobel_k5_ep_results[:8],
                                                                   mask_6l_sobel_k5_ep_results[8:],
                                                                   mask_6l_sobel_k7_ep_results[:8],
                                                                   mask_6l_sobel_k7_ep_results[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    mask_6l_sobel_k7_ep_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ############################################
    ###  4-6l_sobel_k0357-4_no-concat_and_add
    mask_6l_sobel_k7_noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/4-6l_sobel_k0357-4_no-concat_and_add",
                                                  ana_what="mask",
                                                  row_col_results=[mask_6l_noC_and_add_results,
                                                                   mask_6l_sobel_k3_noC_and_add_results,
                                                                   mask_6l_sobel_k5_noC_and_add_results,
                                                                   mask_6l_sobel_k7_noC_and_add_results], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # mask_6l_sobel_k7_noC_and_add_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
