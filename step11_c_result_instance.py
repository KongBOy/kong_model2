from step11_b_result_obj_builder import Result_builder
import copy

compress_results = [] 

dir1 = "5_rect_mae136"
os_book_rect_1532_mae1 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-013834_rect_1532data_mae1_127.28").set_ana_plot_title("mae1").build()
os_book_rect_1532_mae3 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35").set_ana_plot_title("mae3").build()
os_book_rect_1532_mae6 = Result_builder().set_by_result_name(dir1+"/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242").set_ana_plot_title("mae6").build()


dir2 = "5_just_G_mae1369"
os_book_just_G_mae1 = Result_builder().set_by_result_name (dir2+"/type7b_h500_w332_real_os_book-20200525_222831-just_G-1532data_mae1_127.28").set_ana_plot_title("just_G_mae1").build()
os_book_just_G_mae3 = Result_builder().set_by_result_name (dir2+"/type7b_h500_w332_real_os_book-20200524-181909-just_G-1532data_mae3_128.242").set_ana_plot_title("just_G_mae3").build()
os_book_just_G_mae6 = Result_builder().set_by_result_name (dir2+"/type7b_h500_w332_real_os_book-20200525_223733-just_G-1532data_mae6_128.246").set_ana_plot_title("just_G_mae6").build()
os_book_just_G_mae9 = Result_builder().set_by_result_name (dir2+"/type7b_h500_w332_real_os_book-20200525_225555-just_G-1532data_mae9_127.51").set_ana_plot_title("just_G_mae9").build()
os_book_just_G_mae20 = Result_builder().set_by_result_name(dir2+"/type7b_h500_w332_real_os_book-20200527_122823-just_G-1532data_mae20_128.246").set_ana_plot_title("just_G_mae20").build()


dir3 = "5_DG_vs_justG"
os_book_rect_D10 = copy.deepcopy(os_book_rect_1532_mae3); os_book_rect_D10.ana_plot_title = "rect_mae3_D1.0"
os_book_rect_D05 = Result_builder().set_by_result_name(dir3+"/type7b_h500_w332_real_os_book-20200527_073801-rect-1532data_D_0.5_128.245_epoch148").set_ana_plot_title("rect_mae3_D0.5").build()
os_book_rect_D00 = copy.deepcopy(os_book_just_G_mae3); os_book_rect_D00.ana_plot_title = "rect_mae3_D0.0"


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
