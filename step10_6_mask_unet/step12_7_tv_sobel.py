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
    from step11c_7_tv_sobel import  *
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
    #################################################################################################################################################################################################################
    ana_name = "7-6l_tv_s01_sobel_k5_s01,80~140-1_ch"
    mask_L6_tv_s01_sobel_k5_s01_80_140_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                    mask_L6_tv_s01_sobel_k5_s001_ch,
                                                                    mask_L6_tv_s01_sobel_k5_s080_ch,
                                                                    mask_L6_tv_s01_sobel_k5_s100_ch,
                                                                    mask_L6_tv_s01_sobel_k5_s120_ch,
                                                                    mask_L6_tv_s01_sobel_k5_s140_ch
                                                                    ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                                # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "7b_ch032_1-tv_s01,20,30_sobel_k5_s001~140"
    mask_ch032_tv_s01_20_30_sobel_k5_s001_140_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=tv_s20_30_sobel_k5_s001_140s,
                                                  show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                    .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)