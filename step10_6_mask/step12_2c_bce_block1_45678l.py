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
    ### 覺得L2好像不用細看
    ################################################################################
    ana_name = "2c_block1_L2_2-ch128,64,32,16,8,4_bce_s001_100"
    L2_ch64_32_16_8_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L2_ch128,
                                                                   bce_block1_L2_ch064,
                                                                   bce_block1_L2_ch032,
                                                                   bce_block1_L2_ch016,
                                                                   bce_block1_L2_ch008,
                                                                   bce_block1_L2_ch004], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ### 覺得L3好像不用細看
    ################################################################################
    ana_name = "2c_block1_L3_2-ch128,64,32,16,8,4_bce_s001_100"
    L3_ch64_32_16_8_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L3_ch128,
                                                                   bce_block1_L3_ch064,
                                                                   bce_block1_L3_ch032,
                                                                   bce_block1_L3_ch016,
                                                                   bce_block1_L3_ch008,
                                                                   bce_block1_L3_ch004], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    #################################################################################################################################################################################################################
    ana_name = "2c_block1_L4_1-ch064_bce_s001_100"
    L4_ch064_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch064[:3],
                                                                   bce_block1_L4_ch064[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_1-ch032_bce_s001_100"
    L4_ch032_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch032[:3],
                                                                   bce_block1_L4_ch032[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_1-ch016_bce_s001_100"
    L4_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch016[:3],
                                                                   bce_block1_L4_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_1-ch008_bce_s001_100"
    L4_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch008[:3],
                                                                   bce_block1_L4_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_1-ch004_bce_s001_100"
    L4_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch004[:3],
                                                                   bce_block1_L4_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_1-ch002_bce_s001_100"
    L4_ch002_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch002[:3],
                                                                   bce_block1_L4_ch002[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_1-ch001_bce_s001_100"
    L4_ch001_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch001[:3],
                                                                   bce_block1_L4_ch001[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "2c_block1_L4_2-ch64,32,16,8_bce_s001_100"
    L4_ch64_32_16_8_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch064,
                                                                   bce_block1_L4_ch032,
                                                                   bce_block1_L4_ch016,
                                                                   bce_block1_L4_ch008], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_2-ch8,4,2,1_bce_s001_100"
    L4_ch8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch008,
                                                                   bce_block1_L4_ch004,
                                                                   bce_block1_L4_ch002,
                                                                   bce_block1_L4_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_2-ch64,32,16,8,4,2,1_bce_s001_100"
    L4_ch64_32_16_8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch064,
                                                                   bce_block1_L4_ch032,
                                                                   bce_block1_L4_ch016,
                                                                   bce_block1_L4_ch008,
                                                                   bce_block1_L4_ch004,
                                                                   bce_block1_L4_ch002,
                                                                   bce_block1_L4_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "2c_block1_L4_3_E_relu_no_Bias-ch32_bce_s001_100"
    L4_ch32_E_relu_no_Bias_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch032,
                                                                   bce_block1_L4_ch032_E_relu,
                                                                   bce_block1_L4_ch032_no_Bias], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_3_E_relu_no_Bias-ch16_bce_s001_100"
    L4_ch16_E_relu_no_Bias_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch016,
                                                                   bce_block1_L4_ch016_E_relu,
                                                                   bce_block1_L4_ch016_no_Bias], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L4_3_E_relu_no_Bias-ch08_bce_s001_100"
    L4_ch08_E_relu_no_Bias_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch008,
                                                                   bce_block1_L4_ch008_E_relu,
                                                                   bce_block1_L4_ch008_no_Bias], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "2c_block1_L5_1-ch032_bce_s001_100"
    L5_ch032_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch032[:3],
                                                                   bce_block1_L5_ch032[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L5_1-ch016_bce_s001_100"
    L5_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch016[:3],
                                                                   bce_block1_L5_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L5_1-ch008_bce_s001_100"
    L5_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch008[:3],
                                                                   bce_block1_L5_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L5_1-ch004_bce_s001_100"
    L5_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch004[:3],
                                                                   bce_block1_L5_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L5_1-ch002_bce_s001_100"
    L5_ch002_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch002[:3],
                                                                   bce_block1_L5_ch002[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L5_1-ch001_bce_s001_100"
    L5_ch001_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch001[:3],
                                                                   bce_block1_L5_ch001[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "2c_block1_L5_2-ch32,16,8,4,2,1_bce_s001_100"
    L5_ch32_16_8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L5_ch032,
                                                                   bce_block1_L5_ch016,
                                                                   bce_block1_L5_ch008,
                                                                   bce_block1_L5_ch004,
                                                                   bce_block1_L5_ch002,
                                                                   bce_block1_L5_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "2c_block1_L6_1-ch016_bce_s001_100"
    L6_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L6_ch016[:3],
                                                                   bce_block1_L6_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "2c_block1_L6_2-ch16,8,4,2,1_bce_s001_100"
    L6_ch16_8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L6_ch016,
                                                                   bce_block1_L6_ch008,
                                                                   bce_block1_L6_ch004,
                                                                   bce_block1_L6_ch002,
                                                                   bce_block1_L6_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "2c_block1_L7_1-ch008_bce_s001_100"
    L7_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L7_ch008[:3],
                                                                   bce_block1_L7_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "2c_block1_L8_1-ch004_bce_s001_100"
    L8_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L8_ch004[:3],
                                                                   bce_block1_L8_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    #################################################################################################################################################################################################################
    ana_name = "2c_block1_L45678_2-ch64,32,16,8,4_bce_s001_100"
    L45678_ch64_32_16_8_4_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_block1_L4_ch064,
                                                                   bce_block1_L5_ch032,
                                                                   bce_block1_L6_ch016,
                                                                   bce_block1_L7_ch008,
                                                                   bce_block1_L8_ch004], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "2c_block1_L2345_3_E_relu-bce_s001_100"
    L2345_block1_E_relu_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345_E_relu, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L2345_3_E_lrelu_vs_E_relu-bce_s001_100"
    L2345_block1_E_relu_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345__E_lrelu_vs_E_relu, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L34_3_E_lrelu_vs_E_relu-bce_s001_100"
    L34_block1_E_relu_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345__E_lrelu_vs_E_relu[2:6], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    ###############################
    ana_name = "2c_block1_L2345_4_no_Bias-bce_s001_100"
    L2345_block1_no_Bias_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345_no_Bias, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L2345_4_have_Bias_vs_no_Bias-bce_s001_100"
    L2345_block1_have_Bias_vs_no_Bias_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345_have_Bias_vs_no_Bias, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "2c_block1_L34_4_have_Bias_vs_no_Bias-bce_s001_100"
    L34_block1_have_Bias_vs_no_Bias_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345_have_Bias_vs_no_Bias[2:6], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ###############################
    ana_name = "2c_block1_L2345_5_coord_conv-bce_s001_100"
    L2345_block1_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L2345_coord_conv, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "2c_block1_L2345_5_ord_vs_coord_conv-bce_s001_100"
    L2345_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L23456_ord_vs_coord_conv, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "2c_block1_L34_5_ord_vs_coord_conv-bce_s001_100"
    L34_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L23456_ord_vs_coord_conv[2:6], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "2c_block1_L45_5_ord_vs_coord_conv-bce_s001_100"
    L45_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L23456_ord_vs_coord_conv[4:8], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "2c_block1_L456_5_ord_vs_coord_conv-bce_s001_100"
    L456_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=bce_block1_L23456_ord_vs_coord_conv[4:10], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
