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
    # print("    kong_model2_dir:", kong_model2_dir)                                      ### 把 kong_model2 加進 sys.path
    #############################################################################################################################################################################################################
    '''
    import 主程式的東西
    '''
    #############################################################################################################################################################################################################
    kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
    # print("    kong_to_py_layer:", kong_to_py_layer)
    if  (kong_to_py_layer == 0): template_dir = ""
    elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][7:]  ### [7:] 是為了去掉 step1x_
    elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
    elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][7:] + "/" + code_exe_path_element[kong_layer + 2][5:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])  ### 前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子
    # print("    template_dir:", template_dir)
    ##########################################################################################################################################################################################################################################################################################
    ana_dir = template_dir
    ##########################################################################################################################################################################################################################################################################################
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


