from step11_b_result_obj_builder import Result_builder
import copy

compress_results = [] 

dir1 = "5_1_rect_mae136"
os_book_rect_1532_mae1 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-013834_rect_1532data_mae1_127.28" ).set_ana_plot_title("DGmae1").build()
os_book_rect_1532_mae3 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35to28_finish" ).set_ana_plot_title("DGmae3").build()
os_book_rect_1532_mae6 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242").set_ana_plot_title("DGmae6").build()
os_book_rect_1532_mae6_noD = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242").set_ana_plot_title("DGmae6").build()

dir3 = "5_3_justG_mae136920"
os_book_justG_mae1 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200525_222831-justG-1532data_mae1_127.28"  ).set_ana_plot_title("justG_mae1").build()
os_book_justG_mae3 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200524-181909-justG-1532data_mae3_128.242" ).set_ana_plot_title("justG_mae3").build()
os_book_justG_mae6 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200525_223733-justG-1532data_mae6_128.246" ).set_ana_plot_title("justG_mae6").build()
os_book_justG_mae9 = Result_builder().set_by_result_name (dir3+"/type7b_h500_w332_real_os_book-20200525_225555-justG-1532data_mae9_127.51"  ).set_ana_plot_title("justG_mae9").build()
os_book_justG_mae20 = Result_builder().set_by_result_name(dir3+"/type7b_h500_w332_real_os_book-20200527_122823-justG-1532data_mae20_128.246").set_ana_plot_title("justG_mae20").build()

dir2 = "5_2_DG_vs_justG"
os_book_rect_D10        = copy.deepcopy(os_book_rect_1532_mae3);   os_book_rect_D10.ana_plot_title = "rect_mae3_D1.0"
os_book_rect_D05        = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200527_073801-rect-1532data_D_0.5_128.245").set_ana_plot_title("rect_mae3_D0.5").build()
os_book_rect_D025       = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200602_233505-rect-1532data_D_0.25_128.245_epoch211").set_ana_plot_title("rect_mae3_D0.25").build()
os_book_rect_D01        = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200527_120840-rect-1532data_D_0.1_127.28" ).set_ana_plot_title("rect_mae3_D0.1").build()
os_book_rect_D00        = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200525-134838-rect-1532data_D_0.0"        ).set_ana_plot_title("rect_mae3_D0.0").build()
os_book_rect_D00_justG = copy.deepcopy(os_book_justG_mae3);   os_book_rect_D00_justG.ana_plot_title = "rect_mae3_D0.0_justG"

dir4 = "5_4_bigger_smaller"
os_book_justG_bigger       = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200615_030658-justG-1532data_mae3_SRbig_127.35"     ).set_ana_plot_title("justG_mae3_bigger").build()
os_book_justG_bigger_wrong = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200601_022935-justG-1532data_bigger_wrong"     ).set_ana_plot_title("justG_mae3_bigger_wrong").build()
os_book_justG_normal       = copy.deepcopy(os_book_justG_mae3);   os_book_justG_normal.ana_plot_title = "justG_mae3_normal"
os_book_justG_smaller      = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200601_081803-justG-1532data_smaller_128.246").set_ana_plot_title("justG_mae3_smaller").build()
os_book_justG_smaller2     = Result_builder().set_by_result_name(dir4+"/type7b_h500_w332_real_os_book-20200602_163853-justG-1532data_mae3_127.48_smaller2_epoch425").set_ana_plot_title("justG_mae3_smaller2").build()

dir5 = "5_5_focus"
os_book_rect_nfocus        = copy.deepcopy(os_book_rect_1532_mae3);   os_book_rect_nfocus.ana_plot_title = "no_focus_rect_mae3"
os_book_rect_focus         = Result_builder().set_by_result_name(dir5+"/type7b_h500_w332_real_os_book-20200601_224919-rect-1532data_mae3_focus_127.35_finish"        ).set_ana_plot_title("focus_rect_mae3").build()
os_book_justG_nfocus       = copy.deepcopy(os_book_justG_mae3);  os_book_justG_nfocus.ana_plot_title = "no_focus_justG_mae3"
os_book_justG_focus        = Result_builder().set_by_result_name(dir5+"/type7b_h500_w332_real_os_book-20200602_222505-justG-1532data_mae3_focus_128.246_finish"     ).set_ana_plot_title("focus_justG_mae3").build()


dir6 = "5_6_400"
os_book_400_rect         = Result_builder().set_by_result_name(dir6+"/type7b_h500_w332_real_os_book-20200602_201009-rect-400data_mae3_127.35to51_finish" ).set_ana_plot_title("400_rect").build()
os_book_400_justG        = Result_builder().set_by_result_name(dir6+"/type7b_h500_w332_real_os_book-20200603_133217-justG-400data_justG_mae3_127.28"   ).set_ana_plot_title("400_justG").build()
os_book_1532_rect        = copy.deepcopy(os_book_rect_1532_mae3);  os_book_1532_rect.ana_plot_title = "1532_rect"
os_book_1532_justG       = copy.deepcopy(os_book_justG_mae3);  os_book_1532_justG.ana_plot_title = "1532_justG"

dir7 = "5_7_first_k7_vs_k3"
os_book_GD_first_k7     = copy.deepcopy(os_book_rect_1532_mae3);  os_book_GD_first_k7.ana_plot_title = "GD_first_k7"
os_book_GD_first_k3     = Result_builder().set_by_result_name( dir7 + "/type7b_h500_w332_real_os_book-7_2-20200622_205312-justG-1532data_firstk3_127.246ep482"   ).set_ana_plot_title("GD_first_k3").build()
os_book_G_first_k7      = copy.deepcopy(os_book_justG_mae3);  os_book_G_first_k7.ana_plot_title = "G_first_k7"
os_book_G_first_k3      = Result_builder().set_by_result_name( dir7 + "/type7b_h500_w332_real_os_book-7_4-20200622_205606-rect-1532data_rect_firstk3_finish" ).set_ana_plot_title("G_first_k3").build()

dir8a = "5_8a_GD_mrf"
os_book_GD_no_mrf = copy.deepcopy(os_book_rect_1532_mae3);  os_book_GD_no_mrf.ana_plot_title = "GD_no_mrf"
os_book_GD_mrf_7        = Result_builder().set_by_result_name( dir8a + "/type7b_h500_w332_real_os_book-20200621_143739-mrf_rect-1532data_mrf_7-127.48_to_128.246_epoch514" ).set_ana_plot_title("GD_mrf_7").build()
os_book_GD_mrf_79       = Result_builder().set_by_result_name( dir8a + "/type7b_h500_w332_real_os_book-20200621_144146-mrf_rect-1532data_mrf_79_128.245_epoch511"          ).set_ana_plot_title("GD_mrf_79").build()
os_book_GD_mrf_replace7 = Result_builder().set_by_result_name( dir8a + "/type7b_h500_w332_real_os_book-20200621_123201-mrf_rect-1532data_mrf_replace7_127.35_epoch515"     ).set_ana_plot_title("GD_mrf_replace7").build()
os_book_GD_mrf_replace79 = Result_builder().set_by_result_name( dir8a + "/type7b_h500_w332_real_os_book-20200621_123636-mrf_rect-1532data_mrf_replace79_127.51_epoch514"   ).set_ana_plot_title("GD_mrf_replace79").build()

dir8b = "5_8b_G_mrf"
os_book_G_no_mrf = copy.deepcopy(os_book_justG_mae3);  os_book_G_no_mrf.ana_plot_title = "no_mrf"
os_book_G_mrf_7         = Result_builder().set_by_result_name( dir8b + "/type7b_h500_w332_real_os_book-5_8b_2-20200623_191110-justG-1532data_justG_mrf_7_128.245ep510"        ).set_ana_plot_title("G_mrf_7").build()
os_book_G_mrf_79        = Result_builder().set_by_result_name( dir8b + "/type7b_h500_w332_real_os_book-5_8b_3-20200624_002925-justG-1532data_justG_mrf_79_127.55ep479"        ).set_ana_plot_title("G_mrf_79").build()
os_book_G_mrf_replace7  = Result_builder().set_by_result_name( dir8b + "/type7b_h500_w332_real_os_book-5_8b_4-20200622_214400-justG-1532data_justG_mrf_replace7_127.48ep479"  ).set_ana_plot_title("G_mrf_replace7").build()
os_book_G_mrf_replace79 = Result_builder().set_by_result_name( dir8b + "/type7b_h500_w332_real_os_book-5_8b_5-20200623_210009-justG-1532data_justG_mrf_replace79_127.51ep566" ).set_ana_plot_title("G_mrf_replace79").build()

dir9 = "5_9_GD_D_train1_G_train_135"
os_book_D1G1 = copy.deepcopy(os_book_rect_focus);  os_book_D1G1.ana_plot_title = "D1_G1"
os_book_D1G3 = Result_builder().set_by_result_name( dir9 + "/type7b_h500_w332_real_os_book-20200611_222745-rect-1532data_mae3_focus_G03D01_finish" ).set_ana_plot_title("D1_G3").build()
os_book_D1G5 = Result_builder().set_by_result_name( dir9 + "/type7b_h500_w332_real_os_book-20200611_223121-rect-1532data_mae3_focus_G05D01_finish" ).set_ana_plot_title("D1_G5").build()

# compress_results.append(os_book_rect_1532_mae1)
# compress_results.append(os_book_rect_1532_mae3)
# compress_results.append(os_book_rect_1532_mae6)
# compress_results.append(os_book_justG_mae1)
# compress_results.append(os_book_justG_mae3)
# compress_results.append(os_book_justG_mae6)
# compress_results.append(os_book_justG_mae9)
# compress_results.append(os_book_justG_mae20)
# compress_results.append(os_book_rect_D10)
# compress_results.append(os_book_rect_D05)
# compress_results.append(os_book_rect_D00)
