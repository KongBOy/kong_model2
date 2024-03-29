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
    from step12_result_analyzer import Col_exps_analyzer, Row_col_exps_analyzer
    from step11 import  *
    ##########################################################################################################################################################################################################################################################################################    
    kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
    if  (kong_to_py_layer == 2): ana_dir = code_exe_path_element[kong_layer + 1][7:]  ### [7:] 是為了去掉 step1x_
    elif(kong_to_py_layer == 3): ana_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
    elif(kong_to_py_layer >  3): ana_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
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
    ### 直接看 dtd_hdr_mix 的狀況
    #################################################################################################################################################################################################################
    ana_name = "1_ch-1_row"
    mask_L7_ch_analyze = Col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_ch, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "1_ch-2_row"
    mask_layer_analyze = Row_col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_ch[:4],
                                                                   mask_L7_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ########################################################################################
    ana_name = "2_layer-1_row"
    mask_L7_ch_analyze = Col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_layer, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "2_layer-2_row"
    mask_layer_analyze = Row_col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_layer[:4],
                                                                   mask_L7_block1_layer[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ########################################################################################
    ana_name = "3_noC-1_row"
    mask_L7_ch_analyze = Col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_layer_noC, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "3_noC-2_row"
    mask_layer_analyze = Row_col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_layer_noC[:4],
                                                                   mask_L7_block1_layer_noC[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ########################################################################################
    ana_name = "4_skip_add-1_row"
    mask_L7_ch_analyze = Col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                              ana_what="mask",
                                              col_results=mask_L7_block1_skip_add, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_col_results_all_single_see(single_see_multiprocess=True)
    ############################################
    ana_name = "4_skip_add-2_row"
    mask_layer_analyze = Row_col_exps_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L7_block1_skip_add[:4],
                                                                   mask_L7_block1_skip_add[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)

