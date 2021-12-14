'''
目前只有 step12 一定需要切換資料夾到 該komg_model所在的資料夾 才能執行喔！
'''
if(__name__ == "__main__"):
    #############################################################################################################################################################################################################
    ### 把 kong_model2 加入 sys.path
    import os
    code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
    code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
    kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
    kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])    ### 定位出 kong_model2 的 dir
    import sys                                                   ### 把 kong_model2 加入 sys.path
    sys.path.append(kong_model2_dir)
    # print(__file__.split("\\")[-1])
    # print("    code_exe_path:", code_exe_path)
    # print("    code_exe_path_element:", code_exe_path_element)
    # print("    kong_layer:", kong_layer)
    # print("    kong_model2_dir:", kong_model2_dir)
    #############################################################################################################################################################################################################
    from step12_result_analyzer import Row_col_results_analyzer
    from step11_normal_vs_special import  *
    #############################################################################################################################################################################################################
    kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer
    if  (kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][7:]  ### [7:] 是為了去掉 step1x_
    elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
    elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
    ##########################################################################################################################################################################################################################################################################################
    ana_dir = template_dir
    ##########################################################################################################################################################################################################################################################################################
    """
    以下留下一些example這樣子
    core_amount == 7 是因為 目前 see_amount == 7 ，想 一個core 一個see
    task_amount == 7 是因為 目前 see_amount == 7

    single_see_multiprocess == True 代表 see內 還要 切 multiprocess，
    single_see_core_amount == 2 代表切2分

    所以總共會有 7*2 = 14 份 process 要同時處理，
    但建議不要用，已經測過，爆記憶體了
    """

    ################################################################################
    ana_name = "L2345_block1_3_E_relu-bce_s001_100"
    L2345_block1_E_relu_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345_block1_bce_E_lrelu_vs_E_relu[::2], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L2345_block1_3_E_lrelu_vs_E_relu-bce_s001_100"
    L2345_block1_E_relu_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345_block1_bce_E_lrelu_vs_E_relu, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L34_block1_3_E_lrelu_vs_E_relu-bce_s001_100"
    L34_block1_E_relu_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345_block1_bce_E_lrelu_vs_E_relu[2:6], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    ###############################
    ana_name = "L2345_block1_4_no_Bias-bce_s001_100"
    L2345_block1_no_Bias_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345_block1_bce_have_Bias_vs_no_Bias[::2], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L2345_block1_4_have_Bias_vs_no_Bias-bce_s001_100"
    L2345_block1_have_Bias_vs_no_Bias_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345_block1_bce_have_Bias_vs_no_Bias, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ana_name = "L34_block1_4_have_Bias_vs_no_Bias-bce_s001_100"
    L34_block1_have_Bias_vs_no_Bias_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345_block1_bce_have_Bias_vs_no_Bias[2:6], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ###############################
    ana_name = "L2345678_block1_5_coord_conv-bce_s001_100"
    L2345_block1_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L2345678_block1_bce_coord_conv, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()

    ana_name = "L23456_block1_5_coord_conv-bce_s001_100"
    L2345_block1_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L23456_block1_bce_coord_conv, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "L2345_block1_5_ord_vs_coord_conv-bce_s001_100"
    L2345_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L23456_block1_bce_ord_vs_coord_conv, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "L34_block1_5_ord_vs_coord_conv-bce_s001_100"
    L34_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L23456_block1_bce_ord_vs_coord_conv[2:6], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "L45_block1_5_ord_vs_coord_conv-bce_s001_100"
    L45_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L23456_block1_bce_ord_vs_coord_conv[4:8], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
    ana_name = "L456_block1_5_ord_vs_coord_conv-bce_s001_100"
    L456_block1_ord_vs_coord_conv_bce_s001_100_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=L23456_block1_bce_ord_vs_coord_conv[4:10], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                          #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                          #   .Gather_all_see_final_img()
