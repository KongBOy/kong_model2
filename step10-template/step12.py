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
    '''
    以下留下一些example這樣子
    '''
    # epoch300_500_results = [ exp.result_obj for exp in epoch300_500_exps]
    # epoch300_500_analyze = Col_results_analyzer(ana_describe="epoch300_500_exps2", col_results=epoch300_500_results, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
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
    # epoch300_500_analyze_2row_all = Row_col_results_analyzer(ana_describe="2row_ep300_500_a020",
    #                                                         row_col_results=[epoch300_500_results[:5],
    #                                                                         epoch300_500_results[5:10]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # epoch300_500_analyze_2row_all.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ##################################################################################################################################################################
    # Bm_Rec_exps_analyze("unet_L7_skip_SE", unet_L7_skip_SE).all_single_see_final_rec_analyze(reset_dir=True).analyze_tensorboard(reset_dir=True)


    # unet_L7_2to3noC_skip_SE_results = [exp.result_obj for exp in unet_L7_2to3noC_skip_SE]
    # unet_L7_2to3noC_skip_SE_analyze = Row_col_results_analyzer(ana_describe="unet_L7_2to3noC_skip_SE",
    #                                                         row_col_results=[[unet_L7_2to3noC_skip_SE_results[0], unet_L7_2to3noC_skip_SE_results[1]],
    #                                                                         [unet_L7_2to3noC_skip_SE_results[2], unet_L7_2to3noC_skip_SE_results[3]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)
    # unet_L7_2to3noC_skip_SE_analyze.analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=8)
    # Bm_Rec_exps_analyze("unet_L7_2to3noC_skip_SE", unet_L7_2to3noC_skip_SE).all_single_see_final_rec_analyze(reset_dir=True).analyze_tensorboard(reset_dir=True)

    # ##################################################################################################################################################################


