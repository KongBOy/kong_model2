'''
目前只有 step12 一定需要切換資料夾到 該komg_model所在的資料夾 才能執行喔！
'''
if(__name__ == "__main__"):
    #############################################################################################################################################################################################################
    ### 把 kong_model2 加入 sys.path
    import os
    code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
    code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
    kong_layer = code_exe_path_element.index("kong_model2") + 1  ### 找出 kong_model2 在第幾層
    kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])    ### 定位出 kong_model2 的 dir
    import sys                                                   ### 把 kong_model2 加入 sys.path
    sys.path.append(kong_model2_dir)
    # print(__file__.split("\\")[-1])
    # print("    code_exe_path:", code_exe_path)
    # print("    code_exe_path_element:", code_exe_path_element)
    # print("    kong_layer:", kong_layer)
    # print("    kong_model2_dir:", kong_model2_dir)
    #############################################################################################################################################################################################################
    from step12_result_analyzer import Row_col_results_analyzer
    from step11b_ch_loss import  *
    ##########################################################################################################################################################################################################################################################################################
    ana_dir = code_exe_path_element[-3][7:] + "/" + code_exe_path.split("\\")[-2][5:]  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
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
    ana_name = "L2_block1_2-ch128,64,32,16,8,4_bce_s001_100"
    L2_ch64_32_16_8_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L2_block1_ch128,
                                                                   bce_L2_block1_ch064,
                                                                   bce_L2_block1_ch032,
                                                                   bce_L2_block1_ch016,
                                                                   bce_L2_block1_ch008,
                                                                   bce_L2_block1_ch004], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ### 覺得L3好像不用細看
    ################################################################################
    ana_name = "L3_block1_2-ch128,64,32,16,8,4_bce_s001_100"
    L3_ch64_32_16_8_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L3_block1_ch128,
                                                                   bce_L3_block1_ch064,
                                                                   bce_L3_block1_ch032,
                                                                   bce_L3_block1_ch016,
                                                                   bce_L3_block1_ch008,
                                                                   bce_L3_block1_ch004], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    #################################################################################################################################################################################################################
    ana_name = "L4_block1_1-ch064_bce_s001_100"
    L4_ch064_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch064[:3],
                                                                   bce_L4_block1_ch064[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_1-ch032_bce_s001_100"
    L4_ch032_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch032[:3],
                                                                   bce_L4_block1_ch032[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_1-ch016_bce_s001_100"
    L4_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch016[:3],
                                                                   bce_L4_block1_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_1-ch008_bce_s001_100"
    L4_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch008[:3],
                                                                   bce_L4_block1_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_1-ch004_bce_s001_100"
    L4_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch004[:3],
                                                                   bce_L4_block1_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_1-ch002_bce_s001_100"
    L4_ch002_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch002[:3],
                                                                   bce_L4_block1_ch002[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_1-ch001_bce_s001_100"
    L4_ch001_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch001[:3],
                                                                   bce_L4_block1_ch001[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "L4_block1_2-ch64,32,16,8_bce_s001_100"
    L4_ch64_32_16_8_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch064,
                                                                   bce_L4_block1_ch032,
                                                                   bce_L4_block1_ch016,
                                                                   bce_L4_block1_ch008], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_2-ch8,4,2,1_bce_s001_100"
    L4_ch8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch008,
                                                                   bce_L4_block1_ch004,
                                                                   bce_L4_block1_ch002,
                                                                   bce_L4_block1_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_2-ch64,32,16,8,4,2,1_bce_s001_100"
    L4_ch64_32_16_8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch064,
                                                                   bce_L4_block1_ch032,
                                                                   bce_L4_block1_ch016,
                                                                   bce_L4_block1_ch008,
                                                                   bce_L4_block1_ch004,
                                                                   bce_L4_block1_ch002,
                                                                   bce_L4_block1_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "L4_block1_3_E_relu_no_Bias-ch32_bce_s001_100"
    L4_ch32_E_relu_no_Bias_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch032,
                                                                   bce_L4_block1_ch032_E_relu,
                                                                   bce_L4_block1_ch032_no_Bias], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_3_E_relu_no_Bias-ch16_bce_s001_100"
    L4_ch16_E_relu_no_Bias_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch016,
                                                                   bce_L4_block1_ch016_E_relu,
                                                                   bce_L4_block1_ch016_no_Bias], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L4_block1_3_E_relu_no_Bias-ch08_bce_s001_100"
    L4_ch08_E_relu_no_Bias_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L4_block1_ch008,
                                                                   bce_L4_block1_ch008_E_relu,
                                                                   bce_L4_block1_ch008_no_Bias], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "L5_block1_1-ch032_bce_s001_100"
    L5_ch032_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch032[:3],
                                                                   bce_L5_block1_ch032[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L5_block1_1-ch016_bce_s001_100"
    L5_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch016[:3],
                                                                   bce_L5_block1_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L5_block1_1-ch008_bce_s001_100"
    L5_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch008[:3],
                                                                   bce_L5_block1_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L5_block1_1-ch004_bce_s001_100"
    L5_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch004[:3],
                                                                   bce_L5_block1_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L5_block1_1-ch002_bce_s001_100"
    L5_ch002_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch002[:3],
                                                                   bce_L5_block1_ch002[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L5_block1_1-ch001_bce_s001_100"
    L5_ch001_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch001[:3],
                                                                   bce_L5_block1_ch001[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "L5_block1_2-ch32,16,8,4,2,1_bce_s001_100"
    L5_ch32_16_8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L5_block1_ch032,
                                                                   bce_L5_block1_ch016,
                                                                   bce_L5_block1_ch008,
                                                                   bce_L5_block1_ch004,
                                                                   bce_L5_block1_ch002,
                                                                   bce_L5_block1_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "L6_block1_1-ch016_bce_s001_100"
    L6_ch016_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L6_block1_ch016[:3],
                                                                   bce_L6_block1_ch016[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ################################################################################
    ana_name = "L6_block1_2-ch16,8,4,2,1_bce_s001_100"
    L6_ch16_8_4_2_1_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L6_block1_ch016,
                                                                   bce_L6_block1_ch008,
                                                                   bce_L6_block1_ch004,
                                                                   bce_L6_block1_ch002,
                                                                   bce_L6_block1_ch001], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "L7_block1_1-ch008_bce_s001_100"
    L7_ch008_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L7_block1_ch008[:3],
                                                                   bce_L7_block1_ch008[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    #################################################################################################################################################################################################################
    ana_name = "L8_block1_1-ch004_bce_s001_100"
    L8_ch004_block1_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[bce_L8_block1_ch004[:3],
                                                                   bce_L8_block1_ch004[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
