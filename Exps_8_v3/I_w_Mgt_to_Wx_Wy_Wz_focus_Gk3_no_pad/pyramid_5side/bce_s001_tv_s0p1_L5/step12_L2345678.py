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
    ##################################
    ### 5side0
    ##################################
    # "1" (3 3side以前的結果也順便一起看吧)  6 10 15 21 28 36 45 55
    # side1 OK 1
    ana_name = "ch032_1side_1__2_3_side_all__4side_0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_1__2_3_side_all__4side_0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_1__2_3_side_all__4side_1_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_1__2_3_side_all__4side_1_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 "3" (6 3side以前的結果也順便一起看吧)  10 15 21 28 36 45 55
    # side2 OK 4
    ana_name = "ch032_1side_2__2_3_side_all__4side_0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2__2_3_side_all__4side_0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_2__2_3_side_all__4side_1_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2__2_3_side_all__4side_1_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_2__2_3_side_all__4side_2_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2__2_3_side_all__4side_2_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()


    # 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
    # side3 OK 10
    ana_name = "ch032_1side_3__2_3_side_all__4side_0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_3__2_3_side_all__4side_1_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_1_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_3__2_3_side_all__4side_2_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_2_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_3__2_3_side_all__4side_3_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_3_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
    # side4 OK 20
    ana_name = "ch032_1side_4__2_3_side_all__4side_0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_1_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_1_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_2_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_2_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_3_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_3_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_4_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_4_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 "15" (21 3side以前的結果也順便一起看吧) 28 36 45 55
    # side5 OK 35
    ana_name = "ch032_1side_5__2_3_side_all__4side_0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_1_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_1_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_2_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_2_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_3_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_3_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_4_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_4_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_5_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_5_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_1_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_1_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_2_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_2_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_3_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_3_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_4_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_4_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_5_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_5_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s0_5s0"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s0_5s0, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    ##################################
    ### 5side1
    ##################################
    # "1" (3 3side以前的結果也順便一起看吧)  6 10 15 21 28 36 45 55
    # side1 OK 1
    ana_name = "ch032_1side_1__2_3_side_all__4side_1_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_1__2_3_side_all__4side_1_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 "3" (6 3side以前的結果也順便一起看吧)  10 15 21 28 36 45 55
    # side2 OK 4
    ana_name = "ch032_1side_2__2_3_side_all__4side_1_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2__2_3_side_all__4side_1_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_2__2_3_side_all__4side_2_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2__2_3_side_all__4side_2_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()


    # 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
    # side3 OK 10
    ana_name = "ch032_1side_3__2_3_side_all__4side_1_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_1_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_3__2_3_side_all__4side_2_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_2_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_3__2_3_side_all__4side_3_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_3_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
    # side4 OK 20
    ana_name = "ch032_1side_4__2_3_side_all__4side_1_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_1_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_2_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_2_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_3_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_3_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_4_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_4_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 "15" (21 3side以前的結果也順便一起看吧) 28 36 45 55
    # side5 OK 35
    ana_name = "ch032_1side_5__2_3_side_all__4side_1_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_1_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_2_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_2_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_3_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_3_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_4_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_4_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_5_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_5_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_1_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_1_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_2_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_2_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_3_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_3_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_4_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_4_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_5_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_5_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s1"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s1, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    ##################################
    ### 5side2
    ##################################
    # 1 "3" (6 3side以前的結果也順便一起看吧)  10 15 21 28 36 45 55
    # side2 OK 4
    ana_name = "ch032_1side_2__2_3_side_all__4side_2_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_2__2_3_side_all__4side_2_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()


    # 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
    # side3 OK 10
    ana_name = "ch032_1side_3__2_3_side_all__4side_2_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_2_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_3__2_3_side_all__4side_3_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_3_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
    # side4 OK 20
    ana_name = "ch032_1side_4__2_3_side_all__4side_2_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_2_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_3_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_3_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_4_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_4_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 "15" (21 3side以前的結果也順便一起看吧) 28 36 45 55
    # side5 OK 35
    ana_name = "ch032_1side_5__2_3_side_all__4side_2_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_2_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_3_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_3_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_4_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_4_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_5_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_5_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_2_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_2_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_3_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_3_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_4_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_4_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_5_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_5_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s2"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s2, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    ##################################
    ### 5side3
    ##################################
    # 1 3 "6" (10 3side以前的結果也順便一起看吧) 15 21 28 36 45 55
    # side3 OK 10
    ana_name = "ch032_1side_3__2_3_side_all__4side_3_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_3__2_3_side_all__4side_3_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
    # side4 OK 20
    ana_name = "ch032_1side_4__2_3_side_all__4side_3_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_3_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_4__2_3_side_all__4side_4_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_4_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 "15" (21 3side以前的結果也順便一起看吧) 28 36 45 55
    # side5 OK 35
    ana_name = "ch032_1side_5__2_3_side_all__4side_3_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_3_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_4_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_4_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_5_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_5_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_3_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_3_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_4_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_4_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_5_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_5_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s3"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s3, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    ##################################
    ### 5side4
    ##################################
    # 1 3 6 "10" (15 3side以前的結果也順便一起看吧) 21 28 36 45 55
    # side4 OK 20
    ana_name = "ch032_1side_4__2_3_side_all__4side_4_5s4"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_4__2_3_side_all__4side_4_5s4, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 "15" (21 3side以前的結果也順便一起看吧) 28 36 45 55
    # side5 OK 35
    ana_name = "ch032_1side_5__2_3_side_all__4side_4_5s4"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_4_5s4, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_5__2_3_side_all__4side_5_5s4"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_5_5s4, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_4_5s4"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_4_5s4, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_5_5s4"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_5_5s4, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s4"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s4, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    ##################################
    ### 5side5
    ##################################
    # 1 3 6 10 "15" (21 3side以前的結果也順便一起看吧) 28 36 45 55
    # side5 OK 35
    ana_name = "ch032_1side_5__2_3_side_all__4side_5_5s5"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_5__2_3_side_all__4side_5_5s5, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_5_5s5"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_5_5s5, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s5"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s5, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()

    ##################################
    ### 5side6
    ##################################
    # 1 3 6 10 15 "21" (28 3side以前的結果也順便一起看吧) 36 45 55
    # side6 OK 56
    ana_name = "ch032_1side_6__2_3_side_all__4side_6_5s6"
    analyzer = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                        ana_what_sees="see",
                                        ana_what="mask",
                                        row_col_results=ch032_1side_6__2_3_side_all__4side_6_5s6, show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)\
                                            .Gather_all_see_final_img()
