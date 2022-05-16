'''
目前只有 step12 一定需要切換資料夾到 該komg_model所在的資料夾 才能執行喔！
'''
if(__name__ == "__main__"):
    #############################################################################################################################################################################################################
    ### 把 kong_model2 加入 sys.path
    import os
    code_exe_path = os.path.realpath(__file__)                       ### 目前執行 step10_b.py 的 path
    code_exe_path_element = code_exe_path.split("\\")                ### 把 path 切分 等等 要找出 kong_model 在第幾層
    kong_layer = code_exe_path_element.index("kong_model2") + 1      ### 找出 kong_model2 在第幾層
    kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer])  ### 定位出 kong_model2 的 dir
    import sys                                                       ### 把 kong_model2 加入 sys.path
    sys.path.append(kong_model2_dir)
    # print(__file__.split("\\")[-1])
    # print("    code_exe_path:", code_exe_path)
    # print("    code_exe_path_element:", code_exe_path_element)
    # print("    kong_layer:", kong_layer)
    # print("    kong_model2_dir:", kong_model2_dir)
    #############################################################################################################################################################################################################
    from step12_result_analyzer import Row_col_results_analyzer
    from step11_L2345678 import  *
    #############################################################################################################################################################################################################
    kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
    # print("    kong_to_py_layer:", kong_to_py_layer)
    if  (kong_to_py_layer == 0): template_dir = ""
    elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][0:]  ### [7:] 是為了去掉 step1x_， 後來覺得好像改有意義的名字不去掉也行所以 改 0
    elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子， 後來覺得 自動排的順序也可以接受， 所以 改0
    elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])
    # print("    template_dir:", template_dir)
    #############################################################################################################################################################################################################
    ana_dir = template_dir
    #############################################################################################################################################################################################################
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
    ########
    # 1side_1
    ########
    ### 2side_1
    ana_name = "ch032_1side_1_2side_1_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_1_2side_1_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ########
    # 1side_2
    ########
    ### 2side_1
    ana_name = "ch032_1side_2_2side_1_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2_2side_1_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_2
    ana_name = "ch032_1side_2_2side_2_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2_2side_2_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ########
    # 1side_3
    ########
    ### 2side_1
    ana_name = "ch032_1side_3_2side_1_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3_2side_1_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_2
    ana_name = "ch032_1side_3_2side_2_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3_2side_2_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_3
    ana_name = "ch032_1side_3_2side_3_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3_2side_3_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ########
    # 1side_4
    ########
    ### 2side_1
    ana_name = "ch032_1side_4_2side_1_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4_2side_1_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_2
    ana_name = "ch032_1side_4_2side_2_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4_2side_2_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_3
    ana_name = "ch032_1side_4_2side_3_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4_2side_3_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    ### 2side_4
    ana_name = "ch032_1side_4_2side_4_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4_2side_4_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ########
    # 1side_5
    ########
    ### 2side_1
    ana_name = "ch032_1side_5_2side_1_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5_2side_1_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_2
    ana_name = "ch032_1side_5_2side_2_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5_2side_2_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_3
    ana_name = "ch032_1side_5_2side_3_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5_2side_3_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    ### 2side_4
    ana_name = "ch032_1side_5_2side_4_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5_2side_4_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_5
    ana_name = "ch032_1side_5_2side_5_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5_2side_5_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ########
    # 1side_6
    ########
    ### 2side_1
    ana_name = "ch032_1side_6_2side_1_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6_2side_1_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_2
    ana_name = "ch032_1side_6_2side_2_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6_2side_2_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_3
    ana_name = "ch032_1side_6_2side_3_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6_2side_3_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()

    ### 2side_4
    ana_name = "ch032_1side_6_2side_4_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6_2side_4_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_5
    ana_name = "ch032_1side_6_2side_5_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6_2side_5_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
    ### 2side_6
    ana_name = "ch032_1side_6_2side_6_34side_all"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6_2side_6_34side_all, show_in_img=True, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=256, img_w=256, fontsize= 8, title_fontsize=16, jump_to=82)\
                                            # .analyze_row_col_results_all_single_see(single_see_multiprocess=False, single_see_core_amount=6)\
                                            # .Gather_all_see_final_img()
