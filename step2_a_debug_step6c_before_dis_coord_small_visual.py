import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import check_fig_ax_init, move_map_2D_arrow, move_map_3D_scatter

def step6_debug_c_Dis_coord_small_Move_map_Boundary_visual(dis_coord_small_m, move_map_m, start_xy_m,
                                                 boundary_value=0.95,
                                                 before_alpha=1.0,
                                                 after_alpha=1.0,
                                                 jump_r=4, jump_c=4,
                                                 fig=None, ax=None, ax_c=None, tight_layout=False):
    '''
    給 before方法用的：
        視覺化 Before方法 利用調整 dis_coord_m 完後的 dis_coord_small_m 就能直接當bm， 因為 small 可以確保bm 形成的 dis_img 是可控的能限定在某個範圍內
    '''
    before_alpha = 0.5
    after_alpha  = 0.7
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=4, ax_size=5.0, tight_layout=tight_layout)
    h_res, w_res = start_xy_m.shape[: 2]
    ##################################################################################################################
    '''圖0 2D箭頭視覺化 bm裡存的東西'''
    fig, ax, ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 storage array is still bm",
        jump_r=jump_r, jump_c=jump_c,
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        show_before_move_coord=True,  before_alpha=before_alpha,
        show_after_move_coord =False, after_alpha=after_alpha,
        fig=fig, ax=ax, ax_c=0)

    '''圖1 2D箭頭視覺化 bm裡存的東西 模擬移動後的結果'''
    fig, ax, ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 simulate bm move will get dis coord",
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        jump_r=jump_r, jump_c=jump_c,
        show_before_move_coord=True,  before_alpha=before_alpha,
        show_after_move_coord =True, after_alpha=after_alpha,
        fig=fig, ax=ax, ax_c=1)

    '''圖2 畫 3D scatter 分不同平面 比較好看 bm裡存的東西 模擬移動後的結果 '''
    fig, ax, ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up: dis_coord appearance is fm,\n bottom: appearance is bm",
        zticklabels=("bm", "", "", "fm"),
        jump_r=jump_r, jump_c=jump_c,
        before_C="orange", before_alpha=before_alpha, before_height=0,
        after_C = "blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax, ax_c=2)

    '''圖3 畫 3D scatter 分不同平面 比較好看 bm裡存的東西 模擬移動後的結果 + fm框框'''
    fig, ax, ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title=f"fm crop boundary={boundary_value}(same as ord_ratio) \n then do step7_dis_back_to_ord to \n get fm value",
        zticklabels=("bm", "", "", "fm"),
        jump_r=jump_r, jump_c=jump_c,
        boundary_value=boundary_value, boundary_C="blue", boundary_height=0.6, boundary_linewidth=2, boundary_fill=False, boundary_alpha=1,
        before_C="orange", before_alpha=before_alpha, before_height=0,
        after_C = "blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax, ax_c=3)
