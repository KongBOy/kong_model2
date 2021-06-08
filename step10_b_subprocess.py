import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a_load_and_train_and_test.py"]
run = "build().run()"

#####################################################################################################################################################################################################################################################################################################################################
matplot_method_name = '"Save_as_matplot_visual"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
matplot_add_loss = True
matplot_bgr2rgb = True
matplot_see_core_amount = 7
matplot_single_see_core_amount = 2
matplot_result_do_result_print_msg = True
matplot_result_do_see_print_msg = True
matplot_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"

matplot_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"
matplot_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={matplot_method_name}, add_loss={matplot_add_loss}, bgr2rgb={matplot_bgr2rgb}, single_see_core_amount={matplot_single_see_core_amount}, see_print_msg={matplot_result_do_see_print_msg}, see_core_amount={matplot_see_core_amount}, result_print_msg={matplot_result_do_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
compress_method_name = '"Save_as_matplot_bm_rec_visual"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
compress_add_loss = True
compress_bgr2rgb = True
compress_see_core_amount = 7
compress_single_see_core_amount = 2
compress_result_do_result_print_msg = False
compress_result_do_see_print_msg = True
compress_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"

compress_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"
compress_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_result_do_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_do_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
calculate_visual_method_name = '"Calculate_and_Visual_SSIM_LD"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
calculate_visual_add_loss = True
calculate_visual_bgr2rgb = True
calculate_visual_see_core_amount = 1
calculate_visual_single_see_core_amount = 4
calculate_visual_result_do_result_print_msg = False
calculate_visual_result_do_see_print_msg = False
calculate_visual_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"

calculate_visual_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"
calculate_visual_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={calculate_visual_method_name}, add_loss={calculate_visual_add_loss}, bgr2rgb={calculate_visual_bgr2rgb}, single_see_core_amount={calculate_visual_single_see_core_amount}, see_print_msg={calculate_visual_result_do_see_print_msg}, see_core_amount={calculate_visual_see_core_amount}, result_print_msg={calculate_visual_result_do_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
visual_metric_method_name = '"Visual_SSIM_LD"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
visual_metric_add_loss = True
visual_metric_bgr2rgb = True
visual_metric_see_core_amount = 1
visual_metric_single_see_core_amount = 4
visual_metric_result_do_result_print_msg = False
visual_metric_result_do_see_print_msg = False
visual_metric_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"

visual_metric_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"
visual_metric_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={visual_metric_method_name}, add_loss={visual_metric_add_loss}, bgr2rgb={visual_metric_bgr2rgb}, single_see_core_amount={visual_metric_single_see_core_amount}, see_print_msg={visual_metric_result_do_see_print_msg}, see_core_amount={visual_metric_see_core_amount}, result_print_msg={visual_metric_result_do_result_print_msg})"


#####################################################################################################################################################################################################################################################################################################################################
Change_npz_dir_method_name = '"Change_npz_dir"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
Change_npz_dir_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={Change_npz_dir_method_name})"

# sb.run(same_command + [f"test2.{compress_1}"])

# ### hid_ch=64, 來測試 epoch系列 ##############################
# sb.run(same_command + [f"epoch050_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch100_bn_see_arg_T.{run}"])  ### 636
# sb.run(same_command + [f"epoch200_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch300_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch700_bn_see_arg_T.{run}"])

# sb.run(same_command + [f"old_ch128_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch032_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch016_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch008_bn_see_arg_T.{run}"])

# sb.run(same_command + [f"epoch050_new_shuf_bn_see_arg_T.{run}"])  ### 802
# sb.run(same_command + [f"epoch100_new_shuf_bn_see_arg_T.{run}"])  ### 1275
# sb.run(same_command + [f"epoch200_new_shuf_bn_see_arg_T.{run}"])  ### 1309
# sb.run(same_command + [f"epoch300_new_shuf_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch500_new_shuf_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch500_new_shuf_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"epoch700_new_shuf_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"epoch700_bn_see_arg_T_no_down .{calculate_visual_all}"])  ### 看看 lr 都不下降的效果

# sb.run(same_command + [f"old_ch128_new_shuf_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"old_ch032_new_shuf_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"old_ch016_new_shuf_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"old_ch008_new_shuf_bn_see_arg_F.{run}"])

# sb.run(same_command + [f"old_ch128_new_shuf_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch032_new_shuf_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch016_new_shuf_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch008_new_shuf_bn_see_arg_T.{run}"])


# sb.run(same_command + [f"ch64_bn04_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"ch64_bn04_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"ch64_bn08_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"ch64_bn08_bn_see_arg_T.{run}"])


# sb.run(same_command + [f"old_ch32_bn04_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch32_bn08_bn_see_arg_T.{run}"])
# sb.run(same_command + [f"old_ch32_bn16_bn_see_arg_T.{run}"])
# # ### sb.run(same_command + [f"blender_os_book_flow_unet_ch32_bn32.{compress_all}"])  ### 失敗
# # ### sb.run(same_command + [f"blender_os_book_flow_unet_ch32_bn64.{compress_all}"])  ### 失敗


# sb.run(same_command + [f"old_ch32_bn04_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"old_ch32_bn08_bn_see_arg_F.{run}"])
# sb.run(same_command + [f"old_ch32_bn16_bn_see_arg_F.{run}"])

##########################################################################################################################################################################################
### 4. epoch
# sb.run(same_command + [f"ch64_in_epoch060.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch080.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch100.{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch200.{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch220.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch240.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch260.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch280.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch300.{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch320.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch340.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch360.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch380.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch400.{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch420.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch440.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch460.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch480.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch500.{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch700.{calculate_visual_all}"])  ### 測試真的IN

## 4b.
# sb.run(same_command + [f"in_new_ch004_ep060.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch008_ep060.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch016_ep060.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch032_ep060.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch128_ep060.{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"in_new_ch004_ep100.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch008_ep100.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch016_ep100.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch032_ep100.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch128_ep100.{calculate_visual_all}"])  ### 測試真的IN

##########################################################################################################################################################################################
### 5
sb.run(same_command + [f"ch64_in_concat_A.{calculate_visual_all}"])  ### 看看Activation 完再concat的效果

##########################################################################################################################################################################################
# ### 6
sb.run(same_command + [f"unet_2l.{calculate_visual_all}"])
sb.run(same_command + [f"unet_3l.{calculate_visual_all}"])
sb.run(same_command + [f"unet_4l.{calculate_visual_all}"])
sb.run(same_command + [f"unet_5l.{calculate_visual_all}"])
sb.run(same_command + [f"unet_6l.{calculate_visual_all}"])
sb.run(same_command + [f"unet_7l.{calculate_visual_all}"])
sb.run(same_command + [f"unet_8l.{calculate_visual_all}"])

#############################################################################################################
### 7a
sb.run(same_command + [f"unet_8l_skip_use_add.{calculate_visual_all}"])
sb.run(same_command + [f"unet_7l_skip_use_add.{calculate_visual_all}"])
sb.run(same_command + [f"unet_6l_skip_use_add.{calculate_visual_all}"])
sb.run(same_command + [f"unet_5l_skip_use_add.{calculate_visual_all}"])
sb.run(same_command + [f"unet_4l_skip_use_add.{calculate_visual_all}"])
sb.run(same_command + [f"unet_3l_skip_use_add.{calculate_visual_all}"])
sb.run(same_command + [f"unet_2l_skip_use_add.{calculate_visual_all}"])

#############################################################################################################
### 7b
# sb.run(same_command + [f"unet_IN_7l_2to2noC     .{calculate_visual_all}"])
# sb.run(same_command + [f"unet_IN_7l_2to2noC_ch32.{calculate_visual_all}"])
# sb.run(same_command + [f"unet_IN_7l_2to3noC     .{calculate_visual_4te}"])  ### 3254
# sb.run(same_command + [f"unet_IN_7l_2to4noC     .{calculate_visual_all}"])
# sb.run(same_command + [f"unet_IN_7l_2to5noC     .{calculate_visual_all}"])
# sb.run(same_command + [f"unet_IN_7l_2to6noC     .{calculate_visual_all}"])  ### 3073
# sb.run(same_command + [f"unet_IN_7l_2to7noC     .{calculate_visual_all}"])  ### 2851
# sb.run(same_command + [f"unet_IN_7l_2to8noC     .{calculate_visual_all}"])  ### 2920


# sb.run(same_command + [f"unet_IN_7l_2to3noC_e020.{calculate_visual_all}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e040.{calculate_visual_all}"])  ### 測試真的IN  127.55
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e060.{calculate_visual_all}"])  ### 測試真的IN  127.35
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e080.{calculate_visual_all}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e100.{calculate_visual_all}"])  ### 測試真的IN  127.28

# sb.run(same_command + [f"unet_IN_7l_2to3noC_e120.{calculate_visual_all}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e140.{calculate_visual_all}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e160.{calculate_visual_all}"])  ### 測試真的IN  127.55
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e180.{calculate_visual_all}"])  ### 測試真的IN  127.55

#############################################################################################################
### 7c
sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_NO_relu   .{calculate_visual_all}"])   ### 127.35
sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_USErelu   .{calculate_visual_all}"])   ### 127.28
sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_USEsigmoid.{calculate_visual_all}"])   ### 127.35
sb.run(same_command + [f"unet_IN_7l_skip_use_cnn3_USErelu   .{calculate_visual_all}"])   ### 127.28
sb.run(same_command + [f"unet_IN_7l_skip_use_cnn3_USEsigmoid.{calculate_visual_all}"])   ### 127.28


#############################################################################################################
### 7d
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e060_wrong .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e100_wrong .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e060       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e100       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_sSE_e060       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_sSE_e100       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e060_wrong.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e100_wrong.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e060      .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e100      .{calculate_visual_all}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_sk_cSE_e060_wrong .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_cSE_e100_wrong .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_cSE_e060       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_cSE_e100       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_sSE_e060       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_sSE_e100       .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_scSE_e060_wrong.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_scSE_e100_wrong.{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_scSE_e060      .{calculate_visual_all}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_scSE_e100      .{calculate_visual_all}"])  ### 測試真的IN



##########################################################################################################################################################################################
### 8
### 之前 都跑錯的 mse 喔！將錯就錯， 下面重跑 正確的 mae 試試看 順便比較 mse/mae的效果
# sb.run(same_command + [f"t2_in_01_mo_01_gt_01_mse.{run}"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + [f"t1_in_01_mo_th_gt_01_mse.{run}"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + [f"t7_in_th_mo_th_gt_th_mse.{run}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t3_in_01_mo_th_gt_th_mse.{run}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t5_in_th_mo_th_gt_01_mse.{run}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t6_in_th_mo_01_gt_01_mse.{run}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t4_in_01_mo_01_gt_th_mse.{run}"])  ### 圖確認過覺得怪怪的 , 不想重train
# sb.run(same_command + [f"t8_in_th_mo_01_gt_th_mse.{run}"])  ### 已確認錯了, 隨然沒 train完，但前面的175 epoch 也幾乎一樣囉！, 不想重train


# ### 上面 應該是 沒改到loss所以才用mse，現在改用mae試試看
# sb.run(same_command + [f"t2_in_01_mo_01_gt_01_mae.{run}"])  ### ok
# sb.run(same_command + [f"t1_in_01_mo_th_gt_01_mae.{run}"])  ### 127.28 ok
# sb.run(same_command + [f"t3_in_01_mo_th_gt_th_mae.{run}"])  ### 127.28 ok
# sb.run(same_command + [f"t5_in_th_mo_th_gt_01_mae.{run}"])  ### 127.28 ok
# sb.run(same_command + [f"t6_in_th_mo_01_gt_01_mae.{run}"])  ### 127.28 ok
# sb.run(same_command + [f"t7_in_th_mo_th_gt_th_mae.{run}"])  ### 127.28 ok
# sb.run(same_command + [f"t4_in_01_mo_01_gt_th_mae.{run}"])  ### 127.35 ok
# sb.run(same_command + [f"t8_in_th_mo_01_gt_th_mae.{run}"])  ### 127.35 沒train完
##########################################################################################################################################################################################
### 9
# sb.run(same_command + [f"ch64_in_cnnNoBias_epoch060.{run}"])   ### 127.28

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
# sb.run(same_command + [f"rect_fk3_ch64_tfIN_resb_ok9_epoch500.{run}"])
# sb.run(same_command + [f"rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.{run}"])

# sb.run(same_command + [f"rect_2_level_fk3.{run}"])
# sb.run(same_command + [f"rect_3_level_fk3.{run}"])
# sb.run(same_command + [f"rect_4_level_fk3.{run}"])
# sb.run(same_command + [f"rect_5_level_fk3.{run}"])
# sb.run(same_command + [f"rect_6_level_fk3.{run}"])
# sb.run(same_command + [f"rect_7_level_fk3.{run}"])

# sb.run(same_command + [f"rect_2_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_3_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_4_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_5_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_6_level_fk3_ReLU.{run}"])  ### 127.28跑
# sb.run(same_command + [f"rect_7_level_fk3_ReLU.{run}"])  ### 127.28跑

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
# sb.run(same_command + [f"testest_big.{compress_10te}"])   ### 127.28
