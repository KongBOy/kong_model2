from step11_b_result_obj_builder import Result_builder
import copy

compress_results = [] 

dir1 = "5_1_rect_mae136"
os_book_rect_1532_mae1 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-013834_rect_1532data_mae1_127.28" ).set_ana_plot_title("DGmae1").build()
os_book_rect_1532_mae3 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35to28_finish" ).set_ana_plot_title("DGmae3").build()
os_book_rect_1532_mae6 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242").set_ana_plot_title("DGmae6").build()
os_book_rect_1532_mae6_noD = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242").set_ana_plot_title("DGmae6").build()

dir3 = "5_3_just_G_mae136920"
os_book_just_G_mae1 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200525_222831-just_G-1532data_mae1_127.28"  ).set_ana_plot_title("just_G_mae1").build()
os_book_just_G_mae3 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200524-181909-just_G-1532data_mae3_128.242" ).set_ana_plot_title("just_G_mae3").build()
os_book_just_G_mae6 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200525_223733-just_G-1532data_mae6_128.246" ).set_ana_plot_title("just_G_mae6").build()
os_book_just_G_mae9 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200525_225555-just_G-1532data_mae9_127.51"  ).set_ana_plot_title("just_G_mae9").build()
os_book_just_G_mae20 = Result_builder().set_by_result_name(dir3+"/type7b_h500_w332_real_os_book-20200527_122823-just_G-1532data_mae20_128.246").set_ana_plot_title("just_G_mae20").build()

dir2 = "5_2_DG_vs_justG"
os_book_rect_D10        = copy.deepcopy(os_book_rect_1532_mae3); os_book_rect_D10.ana_plot_title = "rect_mae3_D1.0"
os_book_rect_D05        = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200527_073801-rect-1532data_D_0.5_128.245").set_ana_plot_title("rect_mae3_D0.5").build()
os_book_rect_D025       = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200602_233505-rect-1532data_D_0.25_128.245_epoch211").set_ana_plot_title("rect_mae3_D0.25").build()
os_book_rect_D01        = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200527_120840-rect-1532data_D_0.1_127.28" ).set_ana_plot_title("rect_mae3_D0.1").build()
os_book_rect_D00        = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200525-134838-rect-1532data_D_0.0"        ).set_ana_plot_title("rect_mae3_D0.0").build()
os_book_rect_D00_just_G = copy.deepcopy(os_book_just_G_mae3); os_book_rect_D00_just_G.ana_plot_title = "rect_mae3_D0.0_just_G"

dir4 = "5_4_bigger_smaller"
os_book_justG_bigger       = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200601_022935-just_G-1532data_bigger"     ).set_ana_plot_title("justG_mae3_bigger").build()
os_book_justG_normal       = copy.deepcopy(os_book_just_G_mae3); os_book_justG_normal.ana_plot_title = "justG_mae3_normal"
os_book_justG_smaller      = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200601_081803-just_G-1532data_smaller_128.246").set_ana_plot_title("justG_mae3_smaller").build()
os_book_justG_smaller2     = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200602_163853-just_G-1532data_mae3_127.48_smaller2_epoch425").set_ana_plot_title("justG_mae3_smaller2").build()

dir5 = "5_5_focus"
os_book_rect_nfocus        = copy.deepcopy(os_book_rect_1532_mae3); os_book_rect_nfocus.ana_plot_title = "no_focus_rect_mae3"
os_book_rect_focus         = Result_builder().set_by_result_name(dir5+"/type7b_h500_w332_real_os_book-20200601_224919-rect-1532data_mae3_focus_127.35_finish"        ).set_ana_plot_title("focus_rect_mae3").build()
os_book_justG_nfocus       = copy.deepcopy(os_book_just_G_mae3);os_book_justG_nfocus.ana_plot_title = "no_focus_justG_mae3"
os_book_justG_focus        = Result_builder().set_by_result_name(dir5+"/type7b_h500_w332_real_os_book-20200602_222505-just_G-1532data_mae3_focus_128.246_finish"     ).set_ana_plot_title("focus_justG_mae3").build()


dir6 = "5_6_400"
os_book_400_rect         = Result_builder().set_by_result_name(dir6+"/type7b_h500_w332_real_os_book-20200602_201009-rect-400data_mae3_127.35to51_finish" ).set_ana_plot_title("400_rect").build()
os_book_400_justG        = Result_builder().set_by_result_name(dir6+"/type7b_h500_w332_real_os_book-20200603_133217-just_G-400data_just_G_mae3_127.28"   ).set_ana_plot_title("400_justG").build()
os_book_1532_rect        = copy.deepcopy(os_book_rect_1532_mae3);os_book_justG_nfocus.ana_plot_title = "1532_rect"
os_book_1532_justG       = copy.deepcopy(os_book_just_G_mae3);os_book_justG_nfocus.ana_plot_title = "1532_justG"

# compress_results.append(os_book_rect_1532_mae1)
# compress_results.append(os_book_rect_1532_mae3)
# compress_results.append(os_book_rect_1532_mae6)
# compress_results.append(os_book_just_G_mae1)
# compress_results.append(os_book_just_G_mae3)
# compress_results.append(os_book_just_G_mae6)
# compress_results.append(os_book_just_G_mae9)
# compress_results.append(os_book_just_G_mae20)
# compress_results.append(os_book_rect_D10)
# compress_results.append(os_book_rect_D05)
# compress_results.append(os_book_rect_D00)
