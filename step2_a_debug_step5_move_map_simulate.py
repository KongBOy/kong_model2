import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import check_fig_ax_init, move_map_2D_moving_visual
from step2_a_util import wt_calculate

def step5_move_map_simulate_visual(start_xy_m, move_xy, d_abs_norm_m, alpha_fold=0.8, alpha_curl=2.0, fig=None, ax=None, ax_c=None):  ### alpha參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN
    '''
    用套用 move_map 後的 dis_coord 來 視覺化 move_map

    start_xy_m        ： 單純視覺化用的座標而已
    move_xy     ： 單純視覺化出來而已
    d_abs_norm_m： step3 xy_shifted_f 和 move_xy 做 cross 求出的 d_abs_norm 轉回的 map 形式 即 d_abs_nrom_m
    alpha_fold  ： 想模擬的 alpha_fold
    alpha_curl  ： 想模擬的 alpha_curl
    y_min/max   ： 因為要y軸顛倒， 3D把y軸顛倒的方法 目前只嘗試到用 .set_ylim( y_max, y_min )

    fig/ax/ax_c ： default 為 None， 代表要 建立新subplots
                   若 不是 None，在 fig上 畫上此function裡產生的圖
    '''
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=10, ax_size=5, tight_layout=True)
    ##########################################################################################################
    ### 取得 h_res/w_res
    h_res, w_res = start_xy_m.shape[:2]

    ### wt_fold/curl calculate
    wt_fold_m = wt_calculate(dis_type="fold", d_abs_norm=d_abs_norm_m, alpha=alpha_fold)
    wt_curl_m = wt_calculate(dis_type="curl", d_abs_norm=d_abs_norm_m, alpha=alpha_curl)
    move_map_fold_m = move_xy * wt_fold_m  ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘
    move_map_curl_m = move_xy * wt_curl_m  ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘

    fig, ax, ax_c = move_map_2D_moving_visual(move_map_fold_m, start_xy_m, fig_title="fold simulate", fig=fig, ax=ax, ax_c=ax_c)
    fig, ax, ax_c = move_map_2D_moving_visual(move_map_curl_m, start_xy_m, fig_title="curl simulate", fig=fig, ax=ax, ax_c=ax_c)
