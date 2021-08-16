import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import check_fig_ax_init, mesh3D_scatter_and_z0_plane
import numpy as np

from step2_a_util import wt_calculate

def step4_wt_simulate_visual(xy_shifted_m, d_abs_norm_m, alpha_fold_sim=0.8, alpha_curl_sim=2.0, fig=None, ax=None, ax_c=None, tight_layout=False):  ### alpha參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN
    '''
    分析 alpha/d 的關係 和 套用特定alpha(fold/curl都模擬) 看看 目前的 d_abs_norm_m 達到的 wt效果

    xy_shifted_m： step1,2 弄好的 xy_shifted_f 轉回的 map 形式 即 xy_shifted_m
    d_abs_norm_m： step3 xy_shifted_f 和 move_xy 做 cross 求出的 d_abs_norm_m 轉回的 map 形式 即 d_abs_nrom_m
    alpha_fold_sim  ： 想模擬的 alpha_fold_sim
    alpha_curl_sim  ： 想模擬的 alpha_curl_sim
    y_min/max   ： 因為要y軸顛倒， 3D把y軸顛倒的方法 目前只嘗試到用 .set_ylim( y_max, y_min )

    fig/ax/ax_c ： default 為 None， 代表要 建立新subplots
                   若 不是 None，在 fig上 畫上此function裡產生的圖
    '''
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=5, ax_size=5, tight_layout=tight_layout)

    ##########################################################################################################
    ### 變動alpha， 變動d， 看wt
    ### 嘗試以後 alpha > 0 感覺較正常， alpha == 0 時 不管 d 多少 wt 都為0， alpha < 0，儘管 alpha == -0.0， 只要 d 接近0 都會爆炸
    split = 50
    ### alpha_fold_sim == 0.8, 參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN, 區間就設這附近來看看
    # alpha_min = -0.1  ### alpha < 0，儘管 alpha == -0.0， 只要 d 接近0 都會爆炸
    # alpha_min =  0    ### alpha == 0 時 不管 d 多少 wt 都為0
    alpha_min =  0.1
    alpha_fold_min   = alpha_min
    alpha_fold_max   = 1
    alpha_fold_split = np.linspace(alpha_fold_min, alpha_fold_max, split)
    alpha_fold_map   = np.tile(alpha_fold_split, (split, 1))    ### shape 為 (30, 30)

    ### alpha_curl_sim == 2.0, 參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN, 區間就設這附近來看看
    alpha_curl_min   = alpha_min
    alpha_curl_max   = 5
    alpha_curl_split = np.linspace(alpha_curl_min, alpha_curl_max, split)
    alpha_curl_map = np.tile(alpha_curl_split, (split, 1))    ### shape 為 (30, 30)

    ### d_min/max
    d_abs_norm_min    = d_abs_norm_m.min()
    d_abs_norm_max    = d_abs_norm_m.max()
    d_abs_norm_split  = np.linspace(d_abs_norm_min, d_abs_norm_max, split)
    d_abs_norm_wt_map = np.tile(d_abs_norm_split, (split, 1)).T  ### shape 為 (30, 30)

    ### wt_fold/curl calculate
    wt_fold_map = wt_calculate(dis_type="fold", d_abs_norm=d_abs_norm_wt_map, alpha=alpha_fold_map)
    wt_curl_map = wt_calculate(dis_type="curl", d_abs_norm=d_abs_norm_wt_map, alpha=alpha_curl_map)


    ### wt_fold visual (scatter 無.T)
    fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
        x_m=alpha_fold_map, y_m=d_abs_norm_wt_map, z_m=wt_fold_map,
        fig_title="step4(fold sup) d:%.2f~%.2f - alpha:%.2f~%.2f" % (d_abs_norm_min, d_abs_norm_max, alpha_fold_min, alpha_fold_max),
        xlabel="alpha", ylabel="d", zlabel="wt_fold",
        cmap="viridis",
        y_flip=False, tight_layout=True,
        fig=fig, ax=ax, ax_c=ax_c )

    ### wt_fold visual (scatter 有.T)
    ###   .T 是為了顏色要對到！ 顏色跟畫點的順序有關所以要用.T， .T跟不.T畫出來的圖只有著色順序不同，剩下都一樣(可跟上圖做比較)！ xy軸互換沒效果喔！
    fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
        x_m=alpha_fold_map.T, y_m=d_abs_norm_wt_map.T, z_m=wt_fold_map.T,
        fig_title="step4(fold sup) d:%.2f~%.2f - alpha:%.2f~%.2f" % (d_abs_norm_min, d_abs_norm_max, alpha_fold_min, alpha_fold_max),
        xlabel="alpha", ylabel="d", zlabel="wt_fold",
        cmap="viridis",
        y_flip=False, tight_layout=True,
        fig=fig, ax=ax, ax_c=ax_c )

    ### wt_curl visual (scatter 有.T)
    ###   .T 是為了顏色要對到！ 顏色跟畫點的順序有關所以要用.T， .T跟不.T畫出來的圖只有著色順序不同，剩下都一樣(可跟上圖做比較)！ xy軸互換沒效果喔！
    fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
        x_m=alpha_curl_map.T, y_m=d_abs_norm_wt_map.T,  z_m=wt_curl_map.T,
        fig_title="step4(curl sup) d:%.2f~%.2f - alpha:%.2f~%.2f" % (d_abs_norm_min, d_abs_norm_max, alpha_curl_min, alpha_curl_max),
        xlabel="alpha", ylabel="d", zlabel="wt_curl",
        cmap="viridis",
        y_flip=False, tight_layout=True,
        fig=fig, ax=ax, ax_c=ax_c )


    ##########################################################################################################
    ### 套用某alpha， 模擬看看xy_shifted_f上的d 對應的 wt
    ### 套用 alpha_fold_sim to simulate wt_fold
    wt_fold_sim = wt_calculate(dis_type="fold", d_abs_norm=d_abs_norm_m, alpha=alpha_fold_sim)
    fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
        x_m=xy_shifted_m[..., 0], y_m=xy_shifted_m[..., 1],  z_m=wt_fold_sim,
        fig_title=f"step4.wt_fold, alpha={alpha_fold_sim}",
        xlabel="x", ylabel="y", zlabel="wt_fold",
        cmap="hsv",
        y_flip=True, tight_layout=True,
        fig=fig, ax=ax, ax_c=ax_c )

    ### 套用 alpha culr to simulate wt_curl
    wt_curl_sim = wt_calculate(dis_type="curl", d_abs_norm=d_abs_norm_m, alpha=alpha_curl_sim)
    fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
        x_m=xy_shifted_m[..., 0], y_m=xy_shifted_m[..., 1], z_m=wt_curl_sim,
        fig_title=f"step4.wt_curl, alpha={alpha_curl_sim}",
        xlabel="x", ylabel="y", zlabel="wt_curl",
        cmap="hsv",
        y_flip=True, tight_layout=True,
        fig=fig, ax=ax, ax_c=ax_c )

    return fig, ax, ax_c
