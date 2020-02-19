from step1_c_move_map_visual import get_db_all_move

move_list = get_db_all_move("result-2088/move_map")
max_value = move_list.max() ### 228.015008014
min_value = move_list.min() ###-238.454509612

norm = ((move_list-min_value)/(max_value-min_value))*2-1
norm_max_value = norm.max()
norm_min_value = norm.min()

norm_back = (norm+1)/2 * (max_value-min_value) + min_value
morn_back_max = norm_back.max()
morn_back_min = norm_back.min()