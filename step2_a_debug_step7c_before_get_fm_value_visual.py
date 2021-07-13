import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import check_fig_ax_init, move_map_2D_arrow
from util import fill_nan

def step7_visual_util_b_before(see_inv_coord_m, see_inv_move_map_m, fm_nan_mask,
                        dst_coord_m, ord_ratio, ord_coord_m, see_coord_m, xy_m, fig=None, ax=None, ax_c=None):
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=2, ax_size=8, tight_layout=True)
    ##################################################################################################################
    see_coord_m = fill_nan(fm_nan_mask, see_coord_m)

    fig, ax, ax_c = move_map_2D_arrow(see_inv_move_map_m, start_xy_m=see_coord_m,
        arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
        fig_title="step7 fm appearance directly back to bm appearance",
        jump_r=2, jump_c=2,
        arrow_alpha=1.0,
        boundary_value=0.95, boundary_C="orange", boundary_linewidth=3,
        show_before_move_coord=True, before_alpha=0.8,
        show_after_move_coord =True, after_alpha=0.3,
        fig=fig, ax=ax, ax_c=ax_c)


    fig, ax, ax_c = move_map_2D_arrow(see_inv_move_map_m, start_xy_m=see_coord_m,
        arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
        fig_title="fm appearance move result appearance",
        jump_r=2, jump_c=2,
        arrow_alpha = 0.0,
        boundary_value=0.95, boundary_C="orange", boundary_linewidth=3,
        show_before_move_coord=True, before_alpha=0.0,
        show_after_move_coord =True, after_alpha=1.0,
        fig=fig, ax=ax, ax_c=ax_c)
