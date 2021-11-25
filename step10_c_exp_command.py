'''
目前只有 step10b 一定需要切換資料夾到 該step10b所在的資料夾 才能執行喔！
'''

# import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

cmd_python_step10_a = ["python", "step10_a.py"]
run = "build().run()"

bm_rec_see_print_msg = False
bm_rec_result_print_msg = True
compress_and_bm_rec_all  = f"build().result_obj.result_do_all_single_see(start_see=0, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_1    = f"build().result_obj.result_do_all_single_see(start_see=0, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_2    = f"build().result_obj.result_do_all_single_see(start_see=1, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_3    = f"build().result_obj.result_do_all_single_see(start_see=2, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_4    = f"build().result_obj.result_do_all_single_see(start_see=3, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_8    = f"build().result_obj.result_do_all_single_see(start_see=4, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_9    = f"build().result_obj.result_do_all_single_see(start_see=5, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_10   = f"build().result_obj.result_do_all_single_see(start_see=6, see_amount=1 , see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"

compress_and_bm_rec_2te  = f"build().result_obj.result_do_all_single_see(start_see=1, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_3te  = f"build().result_obj.result_do_all_single_see(start_see=2, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_4te  = f"build().result_obj.result_do_all_single_see(start_see=3, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_8te  = f"build().result_obj.result_do_all_single_see(start_see=4, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_9te  = f"build().result_obj.result_do_all_single_see(start_see=5, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"
compress_and_bm_rec_10te = f"build().result_obj.result_do_all_single_see(start_see=6, see_amount=12, see_method_name='Save_as_bm_rec_matplot_visual', add_loss=False, bgr2rgb=True, see_core_amount=1, single_see_core_amount=6, see_print_msg={bm_rec_see_print_msg}, result_print_msg={bm_rec_result_print_msg}, jump_to=11)"


temp_coord_to_flow_all     = f"build().result_obj.result_do_all_single_see(start_see=0, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_1_to_4  = f"build().result_obj.result_do_all_single_see(start_see=0, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_1       = f"build().result_obj.result_do_all_single_see(start_see=0, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_2       = f"build().result_obj.result_do_all_single_see(start_see=1, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_3       = f"build().result_obj.result_do_all_single_see(start_see=2, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_4       = f"build().result_obj.result_do_all_single_see(start_see=3, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_8       = f"build().result_obj.result_do_all_single_see(start_see=4, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_9       = f"build().result_obj.result_do_all_single_see(start_see=5, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_10      = f"build().result_obj.result_do_all_single_see(start_see=6, see_amount=1 , see_method_name='Wrong_flow_rename_to_coord')"

temp_coord_to_flow_2te     = f"build().result_obj.result_do_all_single_see(start_see=1, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_3te     = f"build().result_obj.result_do_all_single_see(start_see=2, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_4te     = f"build().result_obj.result_do_all_single_see(start_see=3, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_8te     = f"build().result_obj.result_do_all_single_see(start_see=4, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_9te     = f"build().result_obj.result_do_all_single_see(start_see=5, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
temp_coord_to_flow_10te    = f"build().result_obj.result_do_all_single_see(start_see=6, see_amount=12, see_method_name='Wrong_flow_rename_to_coord')"
