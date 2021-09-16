import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a_load_and_train_and_test.py"]
run = "build().run()"
change_dir = "build()"  ### 因為我在 See 裡面 把 Change_dir 寫在 __init__ 裡， 所以 只要 build() 其實就有 Change_dir 的效果囉！

#####################################################################################################################################################################################################################################################################################################################################
flow_matplot_method_name = '"Save_as_flow_matplot_visual"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
flow_matplot_add_loss = True
flow_matplot_bgr2rgb = True
flow_matplot_see_core_amount = 7
flow_matplot_single_see_core_amount = 2
flow_matplot_result_print_msg = False
flow_matplot_see_print_msg = False
flow_matplot_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"

flow_matplot_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"
flow_matplot_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={flow_matplot_method_name}, add_loss={flow_matplot_add_loss}, bgr2rgb={flow_matplot_bgr2rgb}, single_see_core_amount={flow_matplot_single_see_core_amount}, see_print_msg={flow_matplot_see_print_msg}, see_core_amount={flow_matplot_see_core_amount}, result_print_msg={flow_matplot_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
compress_method_name = '"Save_as_bm_rec_matplot_visual"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
compress_add_loss = False
compress_bgr2rgb = True
### 應該是 CPU吃滿非常快的設定
# compress_see_core_amount = 7
# compress_single_see_core_amount = 2
### 別的東西再跑 不佔滿CPU的設定
compress_see_core_amount = 1
compress_single_see_core_amount = 8
### 不multiprocess 依序慢慢做 debug 用 的設定
# compress_see_core_amount = 1
# compress_single_see_core_amount = 1
compress_result_print_msg = False
compress_see_print_msg = False
compress_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"

compress_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"
compress_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={compress_method_name}, add_loss={compress_add_loss}, bgr2rgb={compress_bgr2rgb}, single_see_core_amount={compress_single_see_core_amount}, see_print_msg={compress_see_print_msg}, see_core_amount={compress_see_core_amount}, result_print_msg={compress_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
Calculate_SSIM_LD_method_name = '"Calculate_SSIM_LD"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
Calculate_SSIM_LD_see_core_amount = 1
Calculate_SSIM_LD_single_see_core_amount = 4
### 不multiprocess 依序慢慢做 debug 用 的設定
# Calculate_SSIM_LD_see_core_amount = 1
# Calculate_SSIM_LD_single_see_core_amount = 1
Calculate_SSIM_LD_result_print_msg = False
Calculate_SSIM_LD_see_print_msg = False
Calculate_SSIM_LD_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"

Calculate_SSIM_LD_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"
Calculate_SSIM_LD_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={Calculate_SSIM_LD_method_name}, single_see_core_amount={Calculate_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_SSIM_LD_see_print_msg}, see_core_amount={Calculate_SSIM_LD_see_core_amount}, result_print_msg={Calculate_SSIM_LD_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
Visual_SSIM_LD_method_name = '"Visual_SSIM_LD"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
Visual_SSIM_LD_add_loss = True
Visual_SSIM_LD_bgr2rgb = True
Visual_SSIM_LD_see_core_amount = 1
Visual_SSIM_LD_single_see_core_amount = 4
### 不multiprocess 依序慢慢做 debug 用 的設定
# Visual_SSIM_LD_see_core_amount = 1
# Visual_SSIM_LD_single_see_core_amount = 1
Visual_SSIM_LD_result_print_msg = False
Visual_SSIM_LD_see_print_msg = False
Visual_SSIM_LD_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"

Visual_SSIM_LD_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"
Visual_SSIM_LD_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={Visual_SSIM_LD_method_name}, add_loss={Visual_SSIM_LD_add_loss}, bgr2rgb={Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Visual_SSIM_LD_see_print_msg}, see_core_amount={Visual_SSIM_LD_see_core_amount}, result_print_msg={Visual_SSIM_LD_result_print_msg})"

#####################################################################################################################################################################################################################################################################################################################################
Calculate_and_Visual_SSIM_LD_method_name = '"Calculate_and_Visual_SSIM_LD"'  ### 兩層是因為 sb.run 會削掉一層 "" ～
Calculate_and_Visual_SSIM_LD_add_loss = True
Calculate_and_Visual_SSIM_LD_bgr2rgb = True
Calculate_and_Visual_SSIM_LD_see_core_amount = 1
Calculate_and_Visual_SSIM_LD_single_see_core_amount = 4
# Calculate_and_Visual_SSIM_LD_see_core_amount = 1
# Calculate_and_Visual_SSIM_LD_single_see_core_amount = 1
Calculate_and_Visual_SSIM_LD_result_print_msg = False
Calculate_and_Visual_SSIM_LD_see_print_msg = False
Calculate_and_Visual_SSIM_LD_all  = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=7, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_1    = f"build().result_obj.result_do_multiple_single_see(start_see=0, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_2    = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_3    = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_4    = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_8    = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_9    = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_10   = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"

Calculate_and_Visual_SSIM_LD_2te  = f"build().result_obj.result_do_multiple_single_see(start_see=1, see_amount=6, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_3te  = f"build().result_obj.result_do_multiple_single_see(start_see=2, see_amount=5, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_4te  = f"build().result_obj.result_do_multiple_single_see(start_see=3, see_amount=4, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_8te  = f"build().result_obj.result_do_multiple_single_see(start_see=4, see_amount=3, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_9te  = f"build().result_obj.result_do_multiple_single_see(start_see=5, see_amount=2, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"
Calculate_and_Visual_SSIM_LD_10te = f"build().result_obj.result_do_multiple_single_see(start_see=6, see_amount=1, see_method_name={Calculate_and_Visual_SSIM_LD_method_name}, add_loss={Calculate_and_Visual_SSIM_LD_add_loss}, bgr2rgb={Calculate_and_Visual_SSIM_LD_bgr2rgb}, single_see_core_amount={Calculate_and_Visual_SSIM_LD_single_see_core_amount}, see_print_msg={Calculate_and_Visual_SSIM_LD_see_print_msg}, see_core_amount={Calculate_and_Visual_SSIM_LD_see_core_amount}, result_print_msg={Calculate_and_Visual_SSIM_LD_result_print_msg})"


# sb.run(same_command + [f"test2.{compress_1}"])

### hid_ch=64, 來測試 epoch系列 ##############################
# sb.run(same_command + [f"epoch050_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"epoch100_bn_see_arg_T.{change_dir}"])  ### 636
# sb.run(same_command + [f"epoch200_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"epoch300_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"epoch700_bn_see_arg_T.{change_dir}"])

# sb.run(same_command + [f"old_ch128_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch032_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch016_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch008_bn_see_arg_T.{change_dir}"])

# sb.run(same_command + [f"epoch050_new_shuf_bn_see_arg_T.{change_dir}"])  ### 802
# sb.run(same_command + [f"epoch100_new_shuf_bn_see_arg_T.{change_dir}"])  ### 1275
# sb.run(same_command + [f"epoch200_new_shuf_bn_see_arg_T.{change_dir}"])  ### 1309
# sb.run(same_command + [f"epoch300_new_shuf_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"epoch500_new_shuf_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"epoch500_new_shuf_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"epoch700_new_shuf_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"epoch700_bn_see_arg_T_no_down .{change_dir}"])  ### 看看 lr 都不下降的效果

# sb.run(same_command + [f"old_ch128_new_shuf_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"old_ch032_new_shuf_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"old_ch016_new_shuf_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"old_ch008_new_shuf_bn_see_arg_F.{change_dir}"])

# sb.run(same_command + [f"old_ch128_new_shuf_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch032_new_shuf_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch016_new_shuf_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch008_new_shuf_bn_see_arg_T.{change_dir}"])


# sb.run(same_command + [f"ch64_bn04_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"ch64_bn04_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"ch64_bn08_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"ch64_bn08_bn_see_arg_T.{change_dir}"])


# sb.run(same_command + [f"old_ch32_bn04_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch32_bn08_bn_see_arg_T.{change_dir}"])
# sb.run(same_command + [f"old_ch32_bn16_bn_see_arg_T.{change_dir}"])
# # ### sb.run(same_command + [f"blender_os_book_flow_unet_ch32_bn32.{change_dir}"])  ### 失敗
# # ### sb.run(same_command + [f"blender_os_book_flow_unet_ch32_bn64.{change_dir}"])  ### 失敗


# sb.run(same_command + [f"old_ch32_bn04_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"old_ch32_bn08_bn_see_arg_F.{change_dir}"])
# sb.run(same_command + [f"old_ch32_bn16_bn_see_arg_F.{change_dir}"])

##########################################################################################################################################################################################
### 4. epoch
# sb.run(same_command + [f"ch64_in_epoch060.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch080.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch100.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch200.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch220.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch240.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch260.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch280.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch300.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch320.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch340.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch360.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch380.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch400.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch420.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch440.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch460.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch480.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_epoch500.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_epoch700.{change_dir}"])  ### 測試真的IN

### 4b.
# sb.run(same_command + [f"in_new_ch004_ep060.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch008_ep060.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch016_ep060.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch032_ep060.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch128_ep060.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"in_new_ch004_ep100.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch008_ep100.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch016_ep100.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch032_ep100.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"in_new_ch128_ep100.{change_dir}"])  ### 測試真的IN

##########################################################################################################################################################################################
### 5
# sb.run(same_command + [f"ch64_in_concat_A.{change_dir}"])  ### 看看Activation 完再concat的效果

##########################################################################################################################################################################################
### 6
# sb.run(same_command + [f"unet_2l.{change_dir}"])
# sb.run(same_command + [f"unet_3l.{change_dir}"])
# sb.run(same_command + [f"unet_4l.{change_dir}"])
# sb.run(same_command + [f"unet_5l.{change_dir}"])
# sb.run(same_command + [f"unet_6l.{change_dir}"])
# sb.run(same_command + [f"unet_7l.{change_dir}"])
# sb.run(same_command + [f"unet_8l.{change_dir}"])

#############################################################################################################
### 7a
# sb.run(same_command + [f"unet_8l_skip_use_add.{change_dir}"])
# sb.run(same_command + [f"unet_7l_skip_use_add.{change_dir}"])
# sb.run(same_command + [f"unet_6l_skip_use_add.{change_dir}"])
# sb.run(same_command + [f"unet_5l_skip_use_add.{change_dir}"])
# sb.run(same_command + [f"unet_4l_skip_use_add.{change_dir}"])
# sb.run(same_command + [f"unet_3l_skip_use_add.{change_dir}"])
# sb.run(same_command + [f"unet_2l_skip_use_add.{change_dir}"])

#############################################################################################################
### 7b
# sb.run(same_command + [f"unet_IN_7l_2to2noC     .{change_dir}"])
# sb.run(same_command + [f"unet_IN_7l_2to2noC_ch32.{change_dir}"])
# sb.run(same_command + [f"unet_IN_7l_2to3noC     .{change_dir}"])  ### 3254
# sb.run(same_command + [f"unet_IN_7l_2to4noC     .{change_dir}"])
# sb.run(same_command + [f"unet_IN_7l_2to5noC     .{change_dir}"])
# sb.run(same_command + [f"unet_IN_7l_2to6noC     .{change_dir}"])  ### 3073
# sb.run(same_command + [f"unet_IN_7l_2to7noC     .{change_dir}"])  ### 2851
# sb.run(same_command + [f"unet_IN_7l_2to8noC     .{change_dir}"])  ### 2920


# sb.run(same_command + [f"unet_IN_7l_2to3noC_e020.{change_dir}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e040.{change_dir}"])  ### 測試真的IN  127.55
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e060.{change_dir}"])  ### 測試真的IN  127.35
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e080.{change_dir}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e100.{change_dir}"])  ### 測試真的IN  127.28

# sb.run(same_command + [f"unet_IN_7l_2to3noC_e120.{change_dir}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e140.{change_dir}"])  ### 測試真的IN  127.28
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e160.{change_dir}"])  ### 測試真的IN  127.55
# sb.run(same_command + [f"unet_IN_7l_2to3noC_e180.{change_dir}"])  ### 測試真的IN  127.55

#############################################################################################################
### 7c
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_NO_relu   .{change_dir}"])   ### 127.35
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_USErelu   .{change_dir}"])   ### 127.28
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn1_USEsigmoid.{change_dir}"])   ### 127.35
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn3_USErelu   .{change_dir}"])   ### 127.28
# sb.run(same_command + [f"unet_IN_7l_skip_use_cnn3_USEsigmoid.{change_dir}"])   ### 127.28


#############################################################################################################
### 7d
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e060_wrong .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e100_wrong .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e060       .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_cSE_e100       .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_sSE_e060       .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_sSE_e100       .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e060_wrong.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e100_wrong.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e060      .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_2to3noC_sk_scSE_e100      .{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_sk_cSE_e060_wrong .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_cSE_e100_wrong .{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_sk_cSE_e060       .{change_dir}"])  ### 測試真的IN  finish
# sb.run(same_command + [f"ch64_in_sk_cSE_e100       .{change_dir}"])  ### 測試真的IN  finish
# sb.run(same_command + [f"ch64_in_sk_sSE_e060       .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_sSE_e100       .{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_sk_scSE_e060_wrong.{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_scSE_e100_wrong.{change_dir}"])  ### 測試真的IN

# sb.run(same_command + [f"ch64_in_sk_scSE_e060      .{change_dir}"])  ### 測試真的IN
# sb.run(same_command + [f"ch64_in_sk_scSE_e100      .{change_dir}"])  ### 測試真的IN



##########################################################################################################################################################################################
### 8
### 之前 都跑錯的 mse 喔！將錯就錯， 下面重跑 正確的 mae 試試看 順便比較 mse/mae的效果
# sb.run(same_command + [f"t2_in_01_mo_01_gt_01_mse.{change_dir}"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + [f"t1_in_01_mo_th_gt_01_mse.{change_dir}"])  ### 已確認錯了 , 重train 127.35
# sb.run(same_command + [f"t7_in_th_mo_th_gt_th_mse.{change_dir}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t3_in_01_mo_th_gt_th_mse.{change_dir}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t5_in_th_mo_th_gt_01_mse.{change_dir}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t6_in_th_mo_01_gt_01_mse.{change_dir}"])  ### 已確認錯了 , 重train 127.28
# sb.run(same_command + [f"t4_in_01_mo_01_gt_th_mse.{change_dir}"])  ### 圖確認過覺得怪怪的 , 不想重train
# sb.run(same_command + [f"t8_in_th_mo_01_gt_th_mse.{change_dir}"])  ### 已確認錯了, 隨然沒 train完，但前面的175 epoch 也幾乎一樣囉！, 不想重train


### 上面 應該是 沒改到loss所以才用mse，現在改用mae試試看
# sb.run(same_command + [f"t2_in_01_mo_01_gt_01_mae.{change_dir}"])  ### ok
# sb.run(same_command + [f"t1_in_01_mo_th_gt_01_mae.{change_dir}"])  ### 127.28 ok
# sb.run(same_command + [f"t3_in_01_mo_th_gt_th_mae.{change_dir}"])  ### 127.28 ok
# sb.run(same_command + [f"t5_in_th_mo_th_gt_01_mae.{change_dir}"])  ### 127.28 ok
# sb.run(same_command + [f"t6_in_th_mo_01_gt_01_mae.{change_dir}"])  ### 127.28 ok
# sb.run(same_command + [f"t7_in_th_mo_th_gt_th_mae.{change_dir}"])  ### 127.28 ok
# sb.run(same_command + [f"t4_in_01_mo_01_gt_th_mae.{change_dir}"])  ### 127.35 ok
# sb.run(same_command + [f"t8_in_th_mo_01_gt_th_mae.{change_dir}"])  ### 127.35 沒train完
##########################################################################################################################################################################################
### 9
# sb.run(same_command + [f"ch64_in_cnnNoBias_epoch060.{change_dir}"])   ### 127.28

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
# sb.run(same_command + [f"rect_fk3_ch64_tfIN_resb_ok9_epoch500.{change_dir}"])
# sb.run(same_command + [f"rect_fk3_ch64_tfIN_resb_ok9_epoch700_no_epoch_down.{change_dir}"])

# sb.run(same_command + [f"rect_2_level_fk3.{change_dir}"])
# sb.run(same_command + [f"rect_3_level_fk3.{change_dir}"])
# sb.run(same_command + [f"rect_4_level_fk3.{change_dir}"])
# sb.run(same_command + [f"rect_5_level_fk3.{change_dir}"])
# sb.run(same_command + [f"rect_6_level_fk3.{change_dir}"])
# sb.run(same_command + [f"rect_7_level_fk3.{change_dir}"])

# sb.run(same_command + [f"rect_2_level_fk3_ReLU.{change_dir}"])  ### 127.28跑
# sb.run(same_command + [f"rect_3_level_fk3_ReLU.{change_dir}"])  ### 127.28跑
# sb.run(same_command + [f"rect_4_level_fk3_ReLU.{change_dir}"])  ### 127.28跑
# sb.run(same_command + [f"rect_5_level_fk3_ReLU.{change_dir}"])  ### 127.28跑
# sb.run(same_command + [f"rect_6_level_fk3_ReLU.{change_dir}"])  ### 127.28跑
# sb.run(same_command + [f"rect_7_level_fk3_ReLU.{change_dir}"])  ### 127.28跑

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
### 1 試試看 mask
#############################  no-bg  ##################################
### 1. ch 結果超棒就直接結束了 沒有做其他嘗試
# sb.run(same_command + [f"mask_ch001_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_ch002_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_ch004_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_ch008_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_ch016_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_ch032_sigmoid_bce_ep060.{run}"])  ### 127.35跑

############################  have_bg  #################################
### 1. ch
# sb.run(same_command + [f"mask_have_bg_ch008_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_have_bg_ch016_sigmoid_bce_ep060.{run}"])  ### 127.35跑
# sb.run(same_command + [f"mask_have_bg_ch032_sigmoid_bce_ep060.{run}"])  ### 127.35跑
### 2. level
sb.run(same_command + [f"mask_have_bg_ch032_2l_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_3l_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_4l_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_5l_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_6l_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_sigmoid_bce_ep060.{run}"])  ### 127.35跑
### 3. 7l no-concat
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to2noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to3noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to4noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to5noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to6noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to7noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_have_bg_ch032_7l_2to8noC_sigmoid_bce_ep060.{run}"])  ### 127.35跑

##########################################################################################################################################################################################
##########################################################################################################################################################################################
##########################################################################################################################################################################################
# sb.run(same_command + [f"testest.{change_dir}"])   ### 127.28
# sb.run(same_command + [f"testest_big.{change_dir}"])   ### 127.28

### flow_matplot
### compress
### Calculate_SSIM_LD
### Visual_SSIM_LD
### Calculate_and_Visual_SSIM_LD
