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
    from step12_result_analyzer import Col_exps_results_analyzer, Row_col_results_analyzer, Bm_Rec_exps_analyze
    from step11_6c_tv_bce_block1_45678l import  *
    #############################################################################################################################################################################################################
    kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
    # print("    kong_to_py_layer:", kong_to_py_layer)
    if  (kong_to_py_layer == 0): template_dir = ""
    elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][0:]  ### [7:] 是為了去掉 step1x_， 後來覺得好像改有意義的名字不去掉也行所以 改 0
    elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子， 後來覺得 自動排的順序也可以接受， 所以 改0
    elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])
    # print("    template_dir:", template_dir)
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
    ### 直接看 dtd_hdr_mix 的狀況
    #################################################################################################################################################################################################################
    ana_name = "L4_block1_1-tv_s001_bce_s001-1_ch"
    L4_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch[:4],
                                                                   tv_bce_L4_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ###################################################################
    ana_name = "L4_block1_1b1-ch032_tv_s001_bce_s001~100"  ### 6個結果
    L4_block1_tv_s001_bce_s001_100_chh032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch032_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_L4_block1_ch032_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b1-ch032_tv_s020_bce_s020~140"  ### 5個結果
    L4_block1_tv_s020_bce_s020_100_chh032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch032_tv_s020_bce_s020_140[:4],
                                                                   tv_bce_L4_block1_ch032_tv_s020_bce_s020_140[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b1-ch032_tv_s040_bce_s020~140"  ### 7個結果
    L4_block1_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch032_tv_s040_bce_s020_140[ :4],
                                                                   tv_bce_L4_block1_ch032_tv_s040_bce_s020_140[4: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b1-ch032_tv_s060_bce_s020~180"  ### 9個結果
    L4_block1_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch032_tv_s060_bce_s020_180[:3 ],
                                                                   tv_bce_L4_block1_ch032_tv_s060_bce_s020_180[3:6],
                                                                   tv_bce_L4_block1_ch032_tv_s060_bce_s020_180[6: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b1-ch032_tv_s080_bce_s020~180"  ### 9個結果
    L4_block1_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch032_tv_s080_bce_s020_180[:3 ],
                                                                   tv_bce_L4_block1_ch032_tv_s080_bce_s020_180[3:6],
                                                                   tv_bce_L4_block1_ch032_tv_s080_bce_s020_180[6: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b1-ch032_tv_s100_bce_s020~200"  ### 10個結果
    L4_block1_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch032_tv_s100_bce_s020_200[ :4],
                                                                   tv_bce_L4_block1_ch032_tv_s100_bce_s020_200[4:8],
                                                                   tv_bce_L4_block1_ch032_tv_s100_bce_s020_200[8: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ###################################################################
    ana_name = "L4_block1_1b2-ch032_tv_s01,20,40,60,80,100_bce_s020~200"
    L4_block1_tv_s020_bce_s020_100_chh032_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                   tv_bce_L4_block1_ch032_tv_s100_bce_s020_200,
                                                                   tv_bce_L4_block1_ch032_tv_s080_bce_s020_180,
                                                                   tv_bce_L4_block1_ch032_tv_s060_bce_s020_180,
                                                                   tv_bce_L4_block1_ch032_tv_s040_bce_s020_140,
                                                                   tv_bce_L4_block1_ch032_tv_s020_bce_s020_140,
                                                                   tv_bce_L4_block1_ch032_tv_s001_bce_s001_100,
                                                                   ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ###################################################################
    ana_name = "L4_block1_1b1-ch064_tv_s001_bce_s001~100"
    L4_block1_tv_s001_bce_s001_100_ch064_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch064_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_L4_block1_ch064_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b1-ch064_tv_s020_bce_s020~100"
    L4_block1_tv_s020_bce_s020_100_ch064_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch064_tv_s020_bce_s020_100[:3],
                                                                   tv_bce_L4_block1_ch064_tv_s020_bce_s020_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L4_block1_1b2-ch064_tv_s10,20_bce_s020~100"
    L4_block1_tv_s020_bce_s020_100_ch064_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch064_tv_s001_bce_s001_100[1:],
                                                                   tv_bce_L4_block1_ch064_tv_s020_bce_s020_100[:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "L5_block1_1-tv_s001_bce_s001-1_ch"
    L5_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L5_block1_ch[:4],
                                                                   tv_bce_L5_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                              # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L5_block1_1b-ch032_tv_s001_bce_s001~100"
    L5_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L5_block1_ch032_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_L5_block1_ch032_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "L6_block1_1-tv_s001_bce_s001-1_ch"
    L6_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L6_block1_ch[:4],
                                                                   tv_bce_L6_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                              # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "L6_block1_1b-ch016_tv_s001_bce_s001~100"
    L6_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L6_block1_ch016_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_L6_block1_ch016_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "L7_block1_1-tv_s001_bce_s001-1_ch"
    L7_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L7_block1_ch[:4],
                                                                   tv_bce_L7_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "L8_block1_1-tv_s001_bce_s001-1_ch"
    L8_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L8_block1_ch[:4],
                                                                   tv_bce_L8_block1_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "L4567_block1_2-tv_s001_bce_s001-1_ch"
    L8_block1_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{ana_dir}/0_ana_{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_L4_block1_ch[:-2],
                                                                   tv_bce_L5_block1_ch[:-2],
                                                                   tv_bce_L6_block1_ch[:-2],
                                                                   tv_bce_L7_block1_ch[:-2]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False, img_h=512, img_w=512)\
                                              .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
