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
    from step11c_6c_tv_bce_block1_45678l import  *
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
    ### 直接看 dtd_hdr_mix 的狀況
    #################################################################################################################################################################################################################
    ana_name = "6c_block1_4l_1-tv_s001_bce_s001-1_ch"
    block1_4l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch[:4],
                                                                   tv_bce_block1_4l_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ###################################################################
    ana_name = "6c_block1_4l_1b1-ch032_tv_s001_bce_s001~100"  ### 6個結果
    block1_4l_tv_s001_bce_s001_100_chh032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch032_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_block1_4l_ch032_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b1-ch032_tv_s020_bce_s020~140"  ### 5個結果
    block1_4l_tv_s020_bce_s020_100_chh032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch032_tv_s020_bce_s020_140[:4],
                                                                   tv_bce_block1_4l_ch032_tv_s020_bce_s020_140[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b1-ch032_tv_s040_bce_s020~140"  ### 7個結果
    block1_4l_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch032_tv_s040_bce_s020_140[ :4],
                                                                   tv_bce_block1_4l_ch032_tv_s040_bce_s020_140[4: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b1-ch032_tv_s060_bce_s020~180"  ### 9個結果
    block1_4l_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch032_tv_s060_bce_s020_180[:3 ],
                                                                   tv_bce_block1_4l_ch032_tv_s060_bce_s020_180[3:6],
                                                                   tv_bce_block1_4l_ch032_tv_s060_bce_s020_180[6: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b1-ch032_tv_s080_bce_s020~180"  ### 9個結果
    block1_4l_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch032_tv_s080_bce_s020_180[:3 ],
                                                                   tv_bce_block1_4l_ch032_tv_s080_bce_s020_180[3:6],
                                                                   tv_bce_block1_4l_ch032_tv_s080_bce_s020_180[6: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b1-ch032_tv_s100_bce_s020~200"  ### 10個結果
    block1_4l_tv_s040_bce_s020_140_ch032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch032_tv_s100_bce_s020_200[ :4],
                                                                   tv_bce_block1_4l_ch032_tv_s100_bce_s020_200[4:8],
                                                                   tv_bce_block1_4l_ch032_tv_s100_bce_s020_200[8: ]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ###################################################################
    ana_name = "6c_block1_4l_1b2-ch032_tv_s01,20,40,60,80,100_bce_s020~200"
    block1_4l_tv_s020_bce_s020_100_chh032_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[
                                                                   tv_bce_block1_4l_ch032_tv_s100_bce_s020_200,
                                                                   tv_bce_block1_4l_ch032_tv_s080_bce_s020_180,
                                                                   tv_bce_block1_4l_ch032_tv_s060_bce_s020_180,
                                                                   tv_bce_block1_4l_ch032_tv_s040_bce_s020_140,
                                                                   tv_bce_block1_4l_ch032_tv_s020_bce_s020_140,
                                                                   tv_bce_block1_4l_ch032_tv_s001_bce_s001_100,
                                                                   ], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ###################################################################
    ana_name = "6c_block1_4l_1b1-ch064_tv_s001_bce_s001~100"
    block1_4l_tv_s001_bce_s001_100_ch064_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch064_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_block1_4l_ch064_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b1-ch064_tv_s020_bce_s020~100"
    block1_4l_tv_s020_bce_s020_100_ch064_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch064_tv_s020_bce_s020_100[:3],
                                                                   tv_bce_block1_4l_ch064_tv_s020_bce_s020_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_4l_1b2-ch064_tv_s10,20_bce_s020~100"
    block1_4l_tv_s020_bce_s020_100_ch064_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch064_tv_s001_bce_s001_100[1:],
                                                                   tv_bce_block1_4l_ch064_tv_s020_bce_s020_100[:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "6c_block1_5l_1-tv_s001_bce_s001-1_ch"
    block1_5l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_5l_ch[:4],
                                                                   tv_bce_block1_5l_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_5l_1b-ch032_tv_s001_bce_s001~100"
    block1_5l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_5l_ch032_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_block1_5l_ch032_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "6c_block1_6l_1-tv_s001_bce_s001-1_ch"
    block1_6l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_6l_ch[:4],
                                                                   tv_bce_block1_6l_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                              # .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    ana_name = "6c_block1_6l_1b-ch016_tv_s001_bce_s001~100"
    block1_6l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_6l_ch016_tv_s001_bce_s001_100[:3],
                                                                   tv_bce_block1_6l_ch016_tv_s001_bce_s001_100[3:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "6c_block1_7l_1-tv_s001_bce_s001-1_ch"
    block1_7l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_7l_ch[:4],
                                                                   tv_bce_block1_7l_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "6c_block1_8l_1-tv_s001_bce_s001-1_ch"
    block1_8l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_8l_ch[:4],
                                                                   tv_bce_block1_8l_ch[4:]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
    ana_name = "6c_block1_4567l_2-tv_s001_bce_s001-1_ch"
    block1_8l_tv_s001_bce_s001_ch_analyze = Row_col_results_analyzer(ana_describe=f"{mask_ana_dir}/{ana_name}",
                                                  ana_what="mask",
                                                  row_col_results=[tv_bce_block1_4l_ch[:-2],
                                                                   tv_bce_block1_5l_ch[:-2],
                                                                   tv_bce_block1_6l_ch[:-2],
                                                                   tv_bce_block1_7l_ch[:-2]], show_in_img=False, show_gt_img=False, bgr2rgb=True, add_loss=False)\
                                            #   .analyze_row_col_results_all_single_see(single_see_multiprocess=True, single_see_core_amount=6)
    #################################################################################################################################################################################################################
