import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import move_map_2D_arrow, move_map_3D_scatter


def step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_m, move_map_m, start_xy_m,
                                                     current_state = "",
                                                     jump_r=4, jump_c=4,
                                                     boundary_value=0,
                                                     before_alpha = 1.0,
                                                     after_alpha  = 1.0,
                                                     fig=None, ax=None, ax_c=None):
    '''
     給 Before/paper17 兩者都可以用的： 視覺化 dis_coord 調整的過程
    '''

    '''圖0,ax_c 3D scatter 分不同平面  bm裡存的東西 模擬移動後的結果 + valid框框'''
    fig, ax[0], _ = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title=f"{current_state} \n crop a boundary={boundary_value}",
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        jump_r=jump_r, jump_c=jump_c,
        boundary_value=boundary_value, boundary_C="black", boundary_linewidth=2, boundary_fill=False,
        show_before_move_coord=True,  before_alpha=before_alpha,
        show_after_move_coord =True, after_alpha=after_alpha,
        fig=fig, ax=ax[0], ax_c=ax_c)

    '''圖1,ax_c 畫 3D scatter 分不同平面  bm裡存的東西 模擬移動後的結果 + valid框框'''
    fig, ax[1], ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title=f"{current_state} \n crop a boundary={boundary_value}",
        zticklabels=("bm", "", "", "fm"),
        jump_r=jump_r, jump_c=jump_c,
        boundary_value=boundary_value, boundary_C="black", boundary_height=0.6, boundary_linewidth=2, boundary_alpha=1,
        before_C="orange", before_alpha=before_alpha, before_height=0,
        after_C ="blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax[1], ax_c=ax_c, ax_r=1, ax_rows=2)

    return fig, ax, ax_c
