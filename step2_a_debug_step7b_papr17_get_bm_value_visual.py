import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import check_fig_ax_init, coord_m_2D_scatter, move_map_2D_arrow, move_map_3D_scatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import numpy as np

def step7_visual_util_b_paper17(see_inv_coord_m, see_inv_move_map_m,
                        dis_coord_m, ord_coord_m, see_coord_m, start_xy_m, ord_valid_mask,
                        boundary_value = 0.0, dis_rescale_again=1.0,
                        fig=None, ax=None, ax_c=None):
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=6, ax_size=5, tight_layout=True)
    ##################################################################################################################
    h_res, w_res = dis_coord_m.shape[:2]
    '''圖0'''
    before_alpha = 0.7
    after_alpha  = 0.3
    fig, ax, ax_c, ax3d = move_map_3D_scatter(see_inv_move_map_m, start_xy_m=start_xy_m,
        fig_title="step7 we want new bm(up) \n back to ord_valid_coord(bottom)", zticklabels=("new fm", "", "", "new bm"),
        jump_r=1, jump_c=1,
        before_height  =0.5, before_C  ="orange", before_alpha=0.8,
        after_height   =0,   after_C ="blue"    , after_alpha =0.8,
        fig=fig, ax=ax, ax_c=ax_c)
    ax3d.scatter(start_xy_m[..., 0], start_xy_m[..., 1], 0, c= 1 - ord_valid_mask, cmap="bwr", alpha=0.3, s=1)

    '''圖1'''
    fig, ax, ax_c, ax3d = move_map_3D_scatter(see_inv_move_map_m, start_xy_m=start_xy_m,
        fig_title=f"use dis_coord(dis_rescale_again=1/{dis_rescale_again}) \n back to boundary(new bm)", zticklabels=("", "", "boundary \n (new_bm)", "", "", "dis_coord"),
        jump_r=2, jump_c=2,
        before_height=0.5, before_C="orange" , before_alpha=0.8,
        after_height=0,    after_C = "blue", after_alpha =0.,
        fig=fig, ax=ax, ax_c=ax_c)
    ax3d.scatter(dis_coord_m[..., 0], dis_coord_m[..., 1], 1, c= np.arange(h_res * w_res), cmap="hsv", alpha=0.3, s=1)
    # boundary_height=0.5, boundary_value=boundary_value, boundary_C="orange", boundary_fill=True, boundary_alpha=0.5,

    '''圖2'''
    fig, ax, ax_c, ax3d = move_map_3D_scatter(see_inv_move_map_m, start_xy_m=start_xy_m,
        fig_title="then boundary(new bm) on dis_coord will \n back to ord_valid_coord(new fm)", zticklabels=("new fm", "", "new bm", "", "", "new_bm on \n dis_coord"),
        jump_r=2, jump_c=2,
        boundary_height=1.0, boundary_value=boundary_value, boundary_C="orange", boundary_fill=True, boundary_alpha=0.8,
        before_height=0.5, before_C="orange" , before_alpha=0.8,
        after_height=0,    after_C = "blue", after_alpha =0.8,
        fig=fig, ax=ax, ax_c=ax_c)
    ax3d.scatter(start_xy_m[..., 0], start_xy_m[..., 1]  , 0.0, c= 1 - ord_valid_mask, cmap="bwr", alpha=0.3, s=1)
    ax3d.scatter(dis_coord_m[..., 0], dis_coord_m[..., 1], 1.0, c= np.arange(h_res * w_res), cmap="hsv", alpha=0.2, s=1)


    '''圖3'''
    fig, ax, ax_c = coord_m_2D_scatter(dis_coord_m, fig_title=f"dis_coord(dis_rescale_again=1/{dis_rescale_again}) \n back to boundary={boundary_value}(new bm)", fig=fig, ax=ax, ax_c=ax_c)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    ax[ax_c - 1].add_patch( patches.Rectangle( (-boundary_value, -boundary_value), 2 * boundary_value, 2 * boundary_value, edgecolor='orange', linewidth=3, fill=False ))       ### 視覺化 boundary框框

    '''圖4'''
    before_alpha = 0.2
    after_alpha  = 0.8
    fig, ax, ax_c = move_map_2D_arrow(see_inv_move_map_m, start_xy_m=see_coord_m,
        arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
        fig_title="then bondary(new bm) will \n back to ord_valid_coord(new fm)\n then get new bm_value (ord_valid_coord)", jump_r=4, jump_c=4,
        boundary_value=boundary_value, boundary_C="orange", boundary_linewidth=3,
        show_before_move_coord=True, before_alpha=before_alpha,
        show_after_move_coord =True, after_alpha=after_alpha,
        fig=fig, ax=ax, ax_c=ax_c)

    '''圖5'''
    fig, ax, ax_c = coord_m_2D_scatter(see_inv_coord_m, fig_title="new bm inv coord(ord_valid_coord) appearance \n should same as ord_valid_coord appear.", fig=fig, ax=ax, ax_c=ax_c)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    ax[ax_c - 1].scatter(ord_coord_m[..., 0], ord_coord_m[..., 1], c = 1 - ord_valid_mask, s=1, cmap="bwr", alpha=0.2)  ### 視覺化 原始 移動後不超過boundary 的 ord_coord mask
