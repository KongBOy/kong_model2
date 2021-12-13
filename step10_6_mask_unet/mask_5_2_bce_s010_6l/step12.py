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
    from step11 import  *
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
    ### 直接看 dtd_hdr_mix 的狀況
    #################################################################################################################################################################################################################
    ana_name = "1_ch"
    ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_s10_ch[:4],
                                                                   mask_L6_bce_s10_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "2_ep"
    ep_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_s10_ep[:8],
                                                                   mask_L6_bce_s10_ep[8:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ############################################
    ana_name = "3_noC"
    noC_and_add_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}-{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[mask_L6_bce_s10_noC_and_add[:3] + [mask_L6_bce_s10_ch[2]],
                                                                   mask_L6_bce_s10_noC_and_add[3:] + [mask_L6_bce_s10_ch[2]]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################