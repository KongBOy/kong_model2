import sys
sys.path.append("kong_util")

from kong_util.matplot_fig_ax_util import check_fig_ax_init, move_map_2D_arrow, move_map_3D_scatter
from step2_a_util import step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord_simulate

def step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual(dis_coord_big_m, move_map_m, start_xy_m,
                                                            boundary_value=0,
                                                            jump_r=4, jump_c=4,
                                                            before_alpha=0.5,
                                                            after_alpha =0.7,
                                                            fig=None, ax=None, ax_c=None, tight_layout=False):
    '''
    給 paper17方法用的：
        視覺化 paper17 利用調整 dis_coord_m 完後的 dis_coord_big_m 就能從中間抓出滿滿的 new_bm， 這new_bm 對應回的 new_fm 怎麼抓

    dis_coord_m： 主要想視覺畫的東西
    start_xy_m       ： 視覺化輔助用
    move_map_m ： 視覺化輔助用
    jump_r/c   ： 視覺化輔助用， 箭頭的密度
    '''
    # before_alpha = 0.5
    # after_alpha  = 0.7
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=2, fig_cols=7, ax_size=5.0, tight_layout=tight_layout)
    h_res, w_res = start_xy_m.shape[: 2]
    ord_valid_mask = step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord_simulate(dis_coord_big_m, boundary_value=boundary_value, visual=False)
    ##########################################################################################################################
    # dis_coord_method1_visual = method1(x=dis_coord_big_m[..., 0], y=dis_coord_big_m[..., 1], mask_ch=0)
    # ax[ax_c].imshow(dis_coord_method1_visual)
    # ax_c += 1
    ##########################################################################################################################
    '''圖0,0 2D箭頭視覺化 bm裡存的東西'''
    fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 storage array is still bm",
        jump_r=jump_r, jump_c=jump_c,
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        show_before_move_coord=True,  before_alpha=before_alpha,
        show_after_move_coord =False, after_alpha=after_alpha,
        fig=fig, ax=ax[0], ax_r=0, ax_c=0)

    '''圖0,1 2D箭頭視覺化 bm裡存的東西 模擬移動後的結果'''
    fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 simulate bm move will get dis coord",
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        jump_r=jump_r, jump_c=jump_c,
        show_before_move_coord=True,  before_alpha=before_alpha,
        show_after_move_coord =True, after_alpha=after_alpha,
        fig=fig, ax=ax[0], ax_c=1)

    '''圖0,2 畫 3D scatter 分不同平面 比較好看 bm裡存的東西 模擬移動後的結果 '''
    fig, ax[0], ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up: dis_coord appearance is still fm,\n bottom: appearance is still bm",
        zticklabels=("bm", "", "", "fm"),
        jump_r=2, jump_c=2,
        before_C="orange", before_alpha=before_alpha, before_height=0,
        after_C = "blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax[0], ax_c=2, ax_r=0, ax_rows=2)

    '''圖0,3 畫 3D scatter 分不同平面  bm裡存的東西 模擬移動後的結果 + valid框框'''
    fig, ax[0], ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up: crop a boundary as new bm, \n then bottom: becomes new fm, but not all",
        zticklabels=("new fm \n but not all", "", "", "fm \n grab area \n as new bm"),
        jump_r=2, jump_c=2,
        boundary_value=boundary_value, boundary_C="orange", boundary_height=0.6,
        before_C="red", before_alpha=before_alpha, before_height=0,
        after_C ="blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax[0], ax_c=3, ax_r=0, ax_rows=2)

    '''圖0,4 2D箭頭視覺化 移動的過程 + 移動後的結果 + valid框框'''
    fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 grab dis coord valid boundary as new bm",
        jump_r=jump_r, jump_c=jump_c,
        boundary_value=boundary_value, boundary_C="orange", boundary_linewidth=3,
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        show_before_move_coord=True, before_alpha=before_alpha, before_C="orange",
        show_after_move_coord =True, after_alpha=after_alpha,   after_C="blue",
        fig=fig, ax=ax[0], ax_c=4)

    '''圖0,5 2D箭頭視覺化 沒有超過 valid框框 的箭頭 '''
    fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="correspnd valid moves forms fm",
        jump_r=jump_r, jump_c=jump_c,
        arrow_C=1 - ord_valid_mask, arrow_cmap="bwr",  ### 1 - ord_valid_mask 是為了讓 mask 內畫藍色， mask外 畫紅色
        show_before_move_coord=False, before_alpha=before_alpha,
        show_after_move_coord =False, after_alpha=1.0, boundary_value=boundary_value,
        fig=fig, ax=ax[0], ax_c=5)

    '''圖0,6 畫 3D scatter 分不同平面  + valid框框'''
    fig, ax[0], ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up: crop a boundary as bm, \n then bottom: correspnd valid moves forms fm",
        zticklabels=("fm", "", "", "bm"),
        boundary_value=boundary_value, boundary_C="orange", boundary_height=0.6, boundary_fill=True,
        jump_r=2, jump_c=2,
        before_C= 1 - ord_valid_mask, before_cmap="bwr", before_alpha=before_alpha, before_height=0,  ### 1 - ord_valid_mask 是為了讓 mask 內畫藍色， mask外 畫紅色
        after_C = "blue"        ,                    after_alpha=0.3,   after_height =0.6,
        fig=fig, ax=ax[0], ax_c=6, ax_r=0, ax_rows=2)


    '''圖1,0 幾乎== 圖0,4， 只有 title改一下 '''
    fig, ax[1], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 grab dis coord valid boundary as new bm \n see detail at right figures",
        jump_r=jump_r, jump_c=jump_c,
        boundary_value=boundary_value, boundary_C="orange", boundary_linewidth=3,
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        show_before_move_coord=True, before_alpha=before_alpha, before_C="orange",
        show_after_move_coord =True, after_alpha=after_alpha,   after_C="blue",
        fig=fig, ax=ax[1], ax_c=0)


    '''圖1,1~1,5 step7 找 dis_coord_big_m valid的地方'''
    fig, ax[1], ax_c, _ = step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord_simulate(dis_coord_big_m, boundary_value=boundary_value, visual=True, fig=fig, ax=ax[1], ax_c=1)

    return fig, ax, ax_c
