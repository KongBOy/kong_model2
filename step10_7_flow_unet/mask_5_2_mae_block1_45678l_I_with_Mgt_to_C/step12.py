'''
目前只有 step12 一定需要切換資料夾到 該komg_model所在的資料夾 才能執行喔！
'''
if(__name__ == "__main__"):
    ### 把 kong_model2 加入 sys.path
    import os
    code_exe_path = os.path.realpath(__file__)                       ### 目前執行 step10_b.py 的 path
    code_exe_path_element = code_exe_path.split("\\")                ### 把 path 切分 等等 要找出 kong_model 在第幾層
    kong_layer = code_exe_path_element.index("kong_model2") + 1      ### 找出 kong_model2 在第幾層
    kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])  ### 定位出 kong_model2 的 dir
    import sys                                                       ### 把 kong_model2 加入 sys.path
    sys.path.append(kong_model2_dir)
    # print("step10b")
    # print("    code_exe_path:", code_exe_path)
    # print("    code_exe_path_element:", code_exe_path_element)
    # print("    kong_layer:", kong_layer)
    # print("    kong_model2_dir:", kong_model2_dir)
    ######################################################################################################################
    ### 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
    code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
    if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
        os.chdir(code_exe_dir)
    # print("current_path:", os.getcwd())
    ######################################################################################################################
    from step12_result_analyzer import Col_results_analyzer, Row_col_results_analyzer, Bm_Rec_exps_analyze
    from step11 import  *
    ##########################################################################################################################################################################################################################################################################################
    flow_ana_dir = "flow/I_with_Mgt_to_C_with_Mgt_to_F"
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
    ana_name = "2c_block1_flow_L345678_2-ch128,64,32,16,8,4,2,1_mae_s001"
    analyzer = Row_col_results_analyzer(ana_describe=f"{flow_ana_dir}/{ana_name}",
                                        ana_what="flow",
                                        row_col_results=mae_block1_flow_s001_L345678, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ######################################                                          
    ana_name = "2c_block1_rec_L45678_2-ch128,64,32,16,8,4,2,1_mae_s001"
    analyzer = Row_col_results_analyzer(ana_describe=f"{flow_ana_dir}/{ana_name}",
                                        ana_what="rec",
                                        row_col_results=mae_block1_rec_s001_L45678, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
