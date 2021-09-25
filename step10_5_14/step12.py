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
    ##################################################################################################################################################################
    epoch300_500_results = [ exp.result_obj for exp in epoch300_500_exps]
    epoch300_500_analyze = Col_results_analyzer(ana_describe="epoch300_500_exps2", col_results=epoch300_500_results, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # epoch300_500_analyze.analyze_col_results_single_see(see_num=0, single_see_multiprocess=False)
    # epoch300_500_analyze.analyze_col_results_single_see(see_num=0, single_see_multiprocess=True)
    # epoch300_500_analyze.analyze_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=14)
    # epoch300_500_analyze.analyze_col_results_all_single_see_multiprocess(core_amount=2, task_amount=7, single_see_multiprocess=True, single_see_core_amount=8)
    """
    core_amount == 7 是因為 目前 see_amount == 7 ，想 一個core 一個see
    task_amount == 7 是因為 目前 see_amount == 7

    single_see_multiprocess == True 代表 see內 還要 切 multiprocess，
    single_see_core_amount == 2 代表切2分

    所以總共會有 7*2 = 14 份 process 要同時處理，
    但建議不要用，已經測過，爆記憶體了
    """
    # epoch300_500_analyze.analyze_col_results_multi_see(see_nums=[0, 1], save_name="see_1_and_2", multiprocess=True, core_amount=8)

    ############################################
    epoch300_500_analyze_2row_all = Row_col_results_analyzer(ana_describe="2row_ep300_500_a020",
                                                            row_col_results=[epoch300_500_results[:5],
                                                                            epoch300_500_results[5:10]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch300_500_analyze_2row_all.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    epoch300_500_analyze_2row_ep300_500_a060 = Row_col_results_analyzer(ana_describe="2row_ep300_500_a060",
                                                            row_col_results=[[epoch300_500_results[0], epoch300_500_results[3]],
                                                                            [epoch300_500_results[6], epoch300_500_results[9]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch300_500_analyze_2row_ep300_500_a060.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    epoch200_500_results = [exp.result_obj for exp in epoch200_500_exps]
    epoch200_500_analyze_2row_ep200_500_a100 = Row_col_results_analyzer(ana_describe="2row_ep200_500_a100",
                                                            row_col_results=[[epoch200_500_results [0], epoch200_500_results [5]],
                                                                            [epoch200_500_results[10], epoch200_500_results[15]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch200_500_analyze_2row_ep200_500_a100.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    ##################################################################################################################################################################
    ##################################################################################################################################################################
    epoch100_500_results = [exp.result_obj for exp in epoch100_500_exps]
    epoch100_500_analyze_2row_ep100_500_a120 = Row_col_results_analyzer(ana_describe="2row_ep100_500_a120",
                                                            row_col_results=[[epoch100_500_results [0], epoch100_500_results [6]],
                                                                            [epoch100_500_results[13], epoch100_500_results[20]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch100_500_analyze_2row_ep100_500_a120.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    epoch100_500_analyze_2row_ep100_500_a020 = Row_col_results_analyzer(ana_describe="2row_ep100_500_a020",
                                                            row_col_results=[[epoch100_500_results[0], epoch100_500_results[1]],
                                                                            [epoch100_500_results[2], epoch100_500_results[4]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch100_500_analyze_2row_ep100_500_a020.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)
    ##################################################################################################################################################################
    ##################################################################################################################################################################
    epoch020_500_results = [exp.result_obj for exp in epoch020_500_exps]
    epoch020_500_analyze_2row_ep020_500_a120 = Row_col_results_analyzer(ana_describe="2row_ep020_500_a120",
                                                            row_col_results=[[epoch020_500_results [0], epoch020_500_results [6]],
                                                                            [epoch020_500_results[12], epoch020_500_results[18]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch020_500_analyze_2row_ep020_500_a120.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    epoch020_500_analyze_2row_ep020_080_a020 = Row_col_results_analyzer(ana_describe="2row_ep020_080_a020",
                                                            row_col_results=[[epoch020_500_results[0], epoch020_500_results[1]],
                                                                            [epoch020_500_results[2], epoch020_500_results[3]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch020_500_analyze_2row_ep020_080_a020.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    epoch020_500_analyze_2row_ep040_100_a020 = Row_col_results_analyzer(ana_describe="2row_ep040_100_a020",
                                                            row_col_results=[[epoch020_500_results[1], epoch020_500_results[2]],
                                                                            [epoch020_500_results[3], epoch020_500_results[4]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    epoch020_500_analyze_2row_ep040_100_a020.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=16)

    ##################################################################################################################################################################
    ##################################################################################################################################################################
    unet_7l_skip_SE_results = [exp.result_obj for exp in unet_7l_skip_SE]
    unet_7l_skip_SE_analyze = Row_col_results_analyzer(ana_describe="unet_7l_skip_SE",
                                                            row_col_results=[[unet_7l_skip_SE_results[0], unet_7l_skip_SE_results[1]],
                                                                            [unet_7l_skip_SE_results[2], unet_7l_skip_SE_results[3]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    unet_7l_skip_SE_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=8)
    Bm_Rec_exps_analyze("unet_7l_skip_SE", unet_7l_skip_SE).all_single_see_final_rec_analyze(reset_dir=True).analyze_tensorboard(reset_dir=True)


    unet_7l_2to3noC_skip_SE_results = [exp.result_obj for exp in unet_7l_2to3noC_skip_SE]
    unet_7l_2to3noC_skip_SE_analyze = Row_col_results_analyzer(ana_describe="unet_7l_2to3noC_skip_SE",
                                                            row_col_results=[[unet_7l_2to3noC_skip_SE_results[0], unet_7l_2to3noC_skip_SE_results[1]],
                                                                            [unet_7l_2to3noC_skip_SE_results[2], unet_7l_2to3noC_skip_SE_results[3]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)
    unet_7l_2to3noC_skip_SE_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=8)
    Bm_Rec_exps_analyze("unet_7l_2to3noC_skip_SE", unet_7l_2to3noC_skip_SE).all_single_see_final_rec_analyze(reset_dir=True).analyze_tensorboard(reset_dir=True)

    ##################################################################################################################################################################
    ##################################################################################################################################################################
    unet_7l_skip_SE_ep_results = [exp.result_obj for exp in unet_7l_skip_SE_ep]
    unet_7l_skip_SE_ep_analyze = Row_col_results_analyzer(ana_describe="unet_7l_skip_SE_ep",
                                                            row_col_results=[[unet_7l_skip_SE_ep_results[0], unet_7l_skip_SE_ep_results[2], unet_7l_skip_SE_ep_results[4]],
                                                                            [unet_7l_skip_SE_ep_results[1], unet_7l_skip_SE_ep_results[3], unet_7l_skip_SE_ep_results[5]]], show_in_imgs=False, show_gt_imgs=False, bgr2rgb=True, add_loss=False)

    unet_7l_skip_SE_ep_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=8, print_msg=True)
    Bm_Rec_exps_analyze("unet_7l_skip_SE_ep", unet_7l_skip_SE_ep).all_single_see_final_rec_analyze(reset_dir=True).analyze_tensorboard(reset_dir=True)

