import sys
sys.path.append("kong_util")

from step0_access_path import data_access_path
import numpy as np
import matplotlib.pyplot as plt
from build_dataset_combine import Check_dir_exist_and_build
from util import get_xy_f_and_m, time_util, method1, build_blue_array
from matplot_fig_ax_util import change_into_img_2D_coord_ax, draw_3D_xy_plane_by_mesh_f, check_fig_ax_init, change_into_3D_coord_ax, mesh3D_scatter_and_z0_plane, move_map_2D_moving_visual
from matplot_fig_ax_util import coord_f_2D_scatter, coord_m_2D_scatter, move_map_1D_value, move_map_2D_arrow, move_map_3D_scatter
import matplotlib.patches as patches
import time
from scipy.interpolate import LinearNDInterpolator


debug_spyder_dict = dict()

def wt_calculate_before(dis_type, d_abs_norm, alpha):
    '''
    以前參考 https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
    已前的正mesh 是 y： 0~row-1, 切 row分
                   x： 0~col-1, 切 col分
                   這樣子算出的 d 較大， 所以 curl 的 d_abs_norm 要先 /100
    為了紀念留下來，麻煩不要用他，很多case 值都會超出範圍

    return wt的shape 取決於 d_abs_norm/alpha 的 shape
    '''
    if  (dis_type == "fold"): wt = alpha / (d_abs_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
    elif(dis_type == "curl"): wt = 1 - (d_abs_norm / 100 )**(alpha)
    return wt

def wt_calculate(dis_type, d_abs_norm, alpha):
    '''
    現在參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN
    現在的正mesh 是 y： -0.95~+0.95, 切 row分
                   x： -0.95~+0.95, 切 col分
    alpha： 建議 alpha > 0 ，
        1. 用step4_wt_simulate_visual 嘗試以後 alpha > 0 感覺較正常， alpha == 0 時 不管 d 多少 wt 都為0， alpha < 0，儘管 alpha == -0.0， 只要 d 接近0 都會 負很大
        2. alpha > 0 比較符合 paper09 DocUNet 的描述

    return 的 wt 的shape 為 你傳入的 d_abs_norm/alpha 的shape
    '''
    if  (dis_type == "fold"): wt = alpha / (d_abs_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
    elif(dis_type == "curl"): wt = 1 - (d_abs_norm )**(alpha)
    return wt

def step4_wt_simulate_visual(xy_shifted_m, d_abs_norm_m, alpha_fold_sim=0.8, alpha_curl_sim=2.0, fig=None, ax=None, ax_c=None):  ### alpha參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN
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
    fig, ax, ax_c = check_fig_ax_init(fig, ax, ax_c, fig_rows=1, fig_cols=5, ax_size=5, tight_layout=True)

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
    ###   .T 是為了顏色要對到！ 顏色跟畫點的順序有關所以要用.T， .T跟部.T畫出來的圖只有著色順序不同，剩下都一樣(可跟上圖做比較)！ xy軸互換沒效果喔！
    fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
        x_m=alpha_fold_map.T, y_m=d_abs_norm_wt_map.T, z_m=wt_fold_map.T,
        fig_title="step4(fold sup) d:%.2f~%.2f - alpha:%.2f~%.2f" % (d_abs_norm_min, d_abs_norm_max, alpha_fold_min, alpha_fold_max),
        xlabel="alpha", ylabel="d", zlabel="wt_fold",
        cmap="viridis",
        y_flip=False, tight_layout=True,
        fig=fig, ax=ax, ax_c=ax_c )

    ### wt_curl visual (scatter 有.T)
    ###   .T 是為了顏色要對到！ 顏色跟畫點的順序有關所以要用.T， .T跟部.T畫出來的圖只有著色順序不同，剩下都一樣(可跟上圖做比較)！ xy軸互換沒效果喔！
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


### 整個function都是用 image的方式來看(左上角(0,0)，x往右邊走增加，y往上面走增加)
def get_dis_move_map(x_min, x_max, y_min, y_max, w_res, h_res, vert_x, vert_y, move_x, move_y, dis_type="fold", alpha=50, debug=False):
    '''
    x/y_min/max：
        是用 np.linspace 喔！ x_min ~ x_max 就是真的到那個數字！ 不像 np.arange() 會是 x_min ~ x_max-1！
        所以如果要還原以前寫的東西 要記得 x_max-1 喔！
    w/h_res ： min ~ max 之間切多少格
    vert_x  ： 應該要 x_min <= vert_x <= x_max
    vert_y  ： 應該要 y_min <= vert_y <= y_max
    move_x/y： 雖然是沒有限定， 不過應該也是要在 min ~ max 之間會比較合理
    debug ： 建議要搭配 spyder 一起使用， 才能用 Variable Explore 看變數喔！
    '''
    if(debug): fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=12, ax_size=5, tight_layout=True)
    ################################################################################################################################
    ''' step1. 建立xy正mesh， 並 在正mesh中 選一個點當扭曲點
    xy_f         ： 正mesh                         ， 值域為 0~x_max, 0~y_max     , shape 為 (h_res * w_res, 2), f 是 flatten 的意思， xy_f[..., 0]是X座標， xy_f[..., 1]是Y座標
    vtex         ： 正mesh 中 的 一個座標           ， 值域在 0~x_max, 0~y_max 之間, shape 為 (1, 2)
    xy_shifted_f ： 位移到 以 vtex 為(0, 0) 的正mesh                    , shape 為 (h_res * w_res, 2)
    xyz_f_shifted： 位移到 以 vtex 為(0, 0) 的正mesh 在加上一維z， z全填0, shape 為 (h_res * w_res, 3)
    '''
    xy_f, xy_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思
    vtex = np.array([vert_x, vert_y])  ### 指定的扭曲點xy座標

    ### 位移整張mesh變成以扭曲點座標為原點
    xy_shifted_f  = xy_f - vtex
    ### 準備等等要算 d的材料：xyz_f_shifted 多加一個channel z，填0
    # xyz_f_shifted = np.zeros(shape=(h_res * w_res, 3))
    # xyz_f_shifted[:, 0:2] = xy_shifted_f

    ################################################################################################################################
    ''' step2.選一個移動向量 來 決定每個點要怎麼移動
        move_x       ： 一個值
        move_y       ： 一個值
        move_xy      ： [[move_x, move_y   ]]， shape為 (1, 2)
        move_xyz     ： [[move_x, move_y, 0]]， shape為 (1, 3)
        move_xyz_proc： [[move_x, move_y, 0],
                         [move_x, move_y, 0],
                         ...]                ， shape為 (h_res * w_res, 3)
    '''
    if(move_x == 0 and move_y == 0):  ### 兩者不能同時為0，要不然算外積好像有問題
        move_x = 0.00001
        move_y = 0.00001
    move_xy = np.array([[move_x, move_y]], dtype=np.float64)
    ### 研究完paper17 ### 研究完paper17 發現用二維就可做corss了！就不用多一個 z channel囉！
    # move_xyz      = np.array([move_x, move_y, 0])            ### 多一個channel z，填0
    # move_xyz_proc = np.tile( move_xyz, (h_res * w_res, 1) )  ### 擴張給每個點使用

    ''' step1,2 視覺化 xy 和 xy_shift 和 move_xy'''
    if(debug):
        fig, ax, ax_c = coord_f_2D_scatter(xy_f,         h_res, w_res, fig_title="step1.xy_mesh, step2. move_xy",       fig=fig, ax=ax, ax_c=ax_c)
        ax[ax_c - 1].arrow(vert_x, vert_y, move_x, move_y, color="black", length_includes_head=True, head_width=0.05)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！
        fig, ax, ax_c = coord_f_2D_scatter(xy_shifted_f, h_res, w_res, fig_title="step1.xy_mesh_shift, step2. move_xy", fig=fig, ax=ax, ax_c=ax_c)
        ax[ax_c - 1].arrow(   0  ,    0  , move_x, move_y, color="black", length_includes_head=True, head_width=0.05)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！


    ################################################################################################################################
    ''' step3.決定每個點的移動大小d，決定d的大小方式： 以扭曲點為中心的正mesh 和 move 做 cross product
        https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
        沒有說為什麼做這樣做！！！ paper是說觀察拉~~ 但也沒說怎麼觀察的 @口@
        但就是 以扭曲點為中心的正mesh 和 move 做 cross product， 會有下面兩個性質
            1. 扭曲點 到 正mesh該點的     距離   越近 d越小 最小為0
            2. 扭曲點 到 正mesh該點的 方向和move 越像 d越小 最小為0
        如果從結果來回去觀察，好像確實有這樣子的情形～

        d_raw_f   ： 剛做完 corss 最原始的結果， shape 為 (h_res * w_res, 3)
        d_raw     ： 只取 z維 的值， shape 維 (h_res * w_res, )
                        因為  以扭曲點為中心的正mesh每個點(當向量) 是二維， move向量 是 二維，
                        二維 跟 二維 的 向量做 cross， 其 結果向量的：
                            方向：一定在 第三維 且 兩維必0，
                            長度：就是 兩個向量 圍出的平行四邊形面積

        d_abs     ： d_raw 取絕對值！ 因為 d 是 distance 的概念， 只管大小， 不管方向～
        d_abs_norm： d向量的長度 除去 move向量的長度
    '''
    ### 研究完paper17 ### 研究完paper17 發現用二維就可做corss了！就不用多一個 z channel囉！
    # d_raw_f = np.cross(xyz_f_shifted, move_xyz_proc)    ### shape=(...,3)
    # d_raw = d_raw_f[:, 2]        ### shape=(...,1)，前兩個col一定是0，所以取第三個col，且 我們只在意"值"不在意"方向"，所以取絕對值
    d_raw = np.cross(xy_shifted_f, move_xy)  ### shape=(...,)
    d_abs = np.absolute(d_raw)               ### 為了視覺化 所以把 d_abs_norm 分兩步寫
    d_abs_norm = d_abs / d_abs.max()         ### 這比較符合 paper09 DocUNet 的描述， 可以讓 alpha 大的時候 變化較global， alpha 小的時候 變化較local
    # d_abs_norm = d_abs / (np.linalg.norm(move_xy, ord=2))  ### 這是網路上 兩個example 的寫法， 雖然跑起來都沒問題， 但比較不符合 paper09 DocUNet的描述！ norm2 就是算 move_xy向量的 長度 喔！

    ''' 從step3 之後 都用 flatten 轉回 map 的形式 囉！ '''
    xy_shifted_m = xy_shifted_f.reshape(h_res, w_res, 2)
    d_abs_norm_m = d_abs_norm.reshape(h_res, w_res, 1)


    ''' step3 視覺化 d'''
    if(debug):
        ### 研究完paper17 ### 研究完paper17 發現用二維就可做corss了！就不用多一個 z channel囉！
        # debug_spyder_dict["d_0_xyz_f_shifted"] = xyz_f_shifted
        # debug_spyder_dict["d_0_try_move_xyz_proc"] = move_xyz_proc
        debug_spyder_dict["d_1_d_raw"] = d_raw
        debug_spyder_dict["d_2_d_abs"] = d_abs
        debug_spyder_dict["d_3_move_xy"] = move_xy
        debug_spyder_dict["d_3_move_xy_norm2"] = np.linalg.norm(move_xy, ord=2)
        debug_spyder_dict["d_4_d_abs"]         = d_abs
        debug_spyder_dict["d_4_d_abs_norm"]    = d_abs_norm
        debug_spyder_dict["d_5_d_abs_norm_m"]  = d_abs_norm_m

        d_paper17_try = np.cross(xy_shifted_f, move_xy).reshape(-1, 1)
        d_paper17_try_norm = np.linalg.norm( np.cross(xy_shifted_f, move_xy).reshape(-1, 1) / np.linalg.norm(move_xy) , axis=1, keepdims=True)
        debug_spyder_dict["d_paper17_try_1"]                    = d_paper17_try
        debug_spyder_dict["d_paper17_try_2_vs_d_raw"]           = d_raw
        debug_spyder_dict["d_paper17_try_3_norm"]               = d_paper17_try_norm
        debug_spyder_dict["d_paper17_try_4_norm_vs_d_abs_norm"] = d_abs_norm

        ### 在 x_y_f_shifted 上 畫出 d_raw/ d_abs/ d_abs/norm
        fig, ax, ax_c, ax3d1 = mesh3D_scatter_and_z0_plane(x_m=xy_shifted_m[..., 0], y_m=xy_shifted_m[..., 1], z_m=d_raw,      fig_title="step3.d_raw",      xlabel='x', ylabel='y', zlabel='d_raw',      cmap="hsv", y_flip=True, scatter_alpha=0.1, plane_alpha=0.1, tight_layout=True, fig=fig, ax=ax, ax_c=ax_c)
        fig, ax, ax_c, ax3d2 = mesh3D_scatter_and_z0_plane(x_m=xy_shifted_m[..., 0], y_m=xy_shifted_m[..., 1], z_m=d_abs,      fig_title="step3.d_abs",      xlabel='x', ylabel='y', zlabel='d_abs',      cmap="hsv", y_flip=True, scatter_alpha=0.1, plane_alpha=0.1, tight_layout=True, fig=fig, ax=ax, ax_c=ax_c)
        fig, ax, ax_c, ax3d3 = mesh3D_scatter_and_z0_plane(x_m=xy_shifted_m[..., 0], y_m=xy_shifted_m[..., 1], z_m=d_abs_norm, fig_title="step3.d_abs_norm", xlabel='x', ylabel='y', zlabel='d_abs_norm', cmap="hsv", y_flip=True, scatter_alpha=0.1, plane_alpha=0.1, tight_layout=True, fig=fig, ax=ax, ax_c=ax_c)
        ax3d1.quiver(0, 0, 0, move_x, move_y, 0, color="black")
        ax3d2.quiver(0, 0, 0, move_x, move_y, 0, color="black")
        ax3d3.quiver(0, 0, 0, move_x, move_y, 0, color="black")
    #############################################################
    ''' step4. 根據dis_type計算wt， wt 的 shape 為 (h_res, w_res, 1)
            dis_type不管哪種，原則上都是 d越小 move越大，概念是要 離扭曲點越近移動越大。
            但是 "越大" 的 "越"的程度 折/捲 不一樣， 所以有 DocUNet paper 裡的兩個公式：
                折： wt = alpha / (d_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
                捲： wt = 1 - (d_norm )**(alpha)
                wt 的值域為 0~1， shape 為 (h_res * w_res, )
            細部的alpha參數可以看ppt，照這樣子做就有 折/捲 的效果
                alpha 值越大，扭曲越global(看起來效果小)
                alpha 值越小，扭曲越local(看起來效果大)
    '''
    wt_m = wt_calculate(dis_type=dis_type, d_abs_norm=d_abs_norm_m, alpha=alpha)

    '''step4. 根據dis_type 視覺化 wt'''
    if(debug):
        debug_spyder_dict["step4 wt"] = wt_m

        fig, ax, ax_c, ax3d = mesh3D_scatter_and_z0_plane(
            x_m=xy_shifted_m[..., 0], y_m=xy_shifted_m[..., 1],  z_m=wt_m,
            fig_title=f"step4.wt_{dis_type}, alpha={alpha}",
            xlabel="x", ylabel="y", zlabel=f"wt_{dis_type}",
            cmap="hsv",
            y_flip=True, tight_layout=True,
            fig=fig, ax=ax, ax_c=ax_c )

        '''simulate wt_fold/curl 兩種都模擬看看長怎樣'''
        step4_wt_simulate_visual(xy_shifted_m, d_abs_norm_m, alpha_fold_sim=0.8, alpha_curl_sim=2.0)


    ##########################################################################################################################
    '''step5. move 根據 wt 做權重, shape 為 (h_res, w_res, 2)'''
    move_map_m = move_xy * wt_m  ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘

    if(debug):
        debug_spyder_dict["move_map_m"] = move_map_m

        fig, ax, ax_c = move_map_1D_value(move_map_m=move_map_m, move_x=move_x, move_y=move_y, fig_title="step5. see move_map value", fig=fig, ax=ax, ax_c=ax_c)
        fig, ax, ax_c = move_map_2D_moving_visual(move_map_m=move_map_m, start_xy_m=xy_m, fig_title=f"step5. {dis_type}", fig=fig, ax=ax, ax_c=ax_c)

        '''simulate wt_fold/curl 兩種都模擬看看長怎樣'''
        step5_move_map_simulate_visual(xy_m, move_xy, d_abs_norm_m, alpha_fold=0.8, alpha_curl=2.0)

    ### 存debug圖
    if(debug):
        plt.savefig("debug_result")

    # plt.show()
    move_map_value_adjust_by_dst(move_map_m, xy_m, alpha, debug)

    return move_map_m  ### shape：(h_res, w_res, 2)


# def get_fm_like_paper17_or_blender():
'''
step6 繼續編號寫下去
'''
def move_map_value_adjust_by_dst(move_map_m, start_xy_m, alpha=50, debug=True):
    '''
    start_xy_m  ：
    move_map_m  ：

    debug ： 建議要搭配 spyder 一起使用， 才能用 Variable Explore 看變數喔！
    '''
    if(debug): fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=1, ax_size=5, tight_layout=True)
    ##########################################################################################################################
    h_res, w_res = start_xy_m.shape[: 2]

    '''算 step6a. start_xy_m 根據 move_map_m 移動， 就會得到 dis_coord_m'''
    dis_coord_m = start_xy_m + move_map_m  ### dis_coord
    # move_map_m  = dis_coord_m - start_xy_m
    if(debug):
        fig, ax, ax_c = step6_util_dis_coord__move_map__ord_valid_coord__visual(dis_coord_m, start_xy_m, move_map_m, valid_boundary=1, jump_r=4, jump_c=4, fig=None, ax=None, ax_c=None)
        debug_spyder_dict["step6a. dis_coord_m"] = dis_coord_m


    '''算 step6b. dis_coord_m 做些 平移， 得到 dis_coord_shifted_m， 記得回推一下 move_map_shifted_m 喔！'''
    dis_coord_x_m = dis_coord_m[..., 0]
    dis_coord_y_m = dis_coord_m[..., 1]
    dis_coord_xmin = dis_coord_x_m.min()
    dis_coord_xmax = dis_coord_x_m.max()
    dis_coord_ymin = dis_coord_y_m.min()
    dis_coord_ymax = dis_coord_y_m.max()
    dis_coord_xcenter = (dis_coord_xmin + dis_coord_xmax) / 2
    dis_coord_ycenter = (dis_coord_ymin + dis_coord_ymax) / 2
    dis_coord_shifted_m = dis_coord_m.copy()
    dis_coord_shifted_m[..., 0] -= dis_coord_xcenter
    dis_coord_shifted_m[..., 1] -= dis_coord_ycenter
    move_map_shifted_m = dis_coord_shifted_m - start_xy_m
    if(debug):
        fig, ax, ax_c = step6_util_dis_coord__move_map__ord_valid_coord__visual(dis_coord_shifted_m, start_xy_m, move_map_shifted_m, valid_boundary=1, jump_r=4, jump_c=4, fig=None, ax=None, ax_c=None)
        debug_spyder_dict["step6b. dis_coord_xcenter"]   = dis_coord_xcenter
        debug_spyder_dict["step6b. dis_coord_ycenter"]   = dis_coord_ycenter
        debug_spyder_dict["step6b. dis_coord_shifted_m"] = dis_coord_shifted_m


    '''算 step6c. dis_coord_shifted_m 做些 縮放， 記得回推一下 move_map_shifted_scaled_big_m 喔！'''
    dis_coord_shifted_x_m = dis_coord_shifted_m[..., 0]
    dis_coord_shifted_y_m = dis_coord_shifted_m[..., 1]
    dis_coord_shifted_xmin = dis_coord_shifted_x_m.min()
    dis_coord_shifted_xmax = dis_coord_shifted_x_m.max()
    dis_coord_shifted_ymin = dis_coord_shifted_y_m.min()
    dis_coord_shifted_ymax = dis_coord_shifted_y_m.max()
    raito = max(abs(dis_coord_shifted_xmin), abs(dis_coord_shifted_xmax), abs(dis_coord_shifted_ymin), abs(dis_coord_shifted_ymax))
    ### big， 自己要知道 要超過 see_coord_m
    dis_coord_shifted_scaled_big_m = dis_coord_shifted_m.copy()
    dis_coord_shifted_scaled_big_m *= raito * 1.2

    ### small， 自己要知道 不超過 see_coord_m
    dis_coord_shifted_scaled_small_m =  dis_coord_shifted_m.copy()
    dis_coord_shifted_scaled_small_m *= 0.9

    move_map_shifted_scaled_big_m   = dis_coord_shifted_scaled_big_m   - start_xy_m
    move_map_shifted_scaled_small_m = dis_coord_shifted_scaled_small_m - start_xy_m  ### 本身就不需要
    if(debug):
        fig, ax, ax_c = step6_util_dis_coord__move_map__ord_valid_coord__visual(dis_coord_shifted_scaled_big_m,   start_xy_m, move_map_shifted_scaled_big_m,   valid_boundary=1.00, jump_r=4, jump_c=4, fig=None, ax=None, ax_c=None)
        fig, ax, ax_c = step6_util_dis_coord__move_map__ord_valid_coord__visual(dis_coord_shifted_scaled_small_m, start_xy_m, move_map_shifted_scaled_small_m, valid_boundary=0.95, jump_r=4, jump_c=4, fig=None, ax=None, ax_c=None)
        debug_spyder_dict["step6c. raito"]                        = raito
        debug_spyder_dict["step6c. dis_coord_shifted_scaled_big_m"]   = dis_coord_shifted_scaled_big_m
        debug_spyder_dict["step6c. dis_coord_shifted_scaled_small_m"] = dis_coord_shifted_scaled_small_m


    ''' 在做完 縮放 後 的 dis_coord_shifted_scaled_big 上， 找valid區域， 並對應回原始move_map '''
    step7_dis_coord_valid_area_storage_inverse_move_map_is_bm(dis_coord_shifted_scaled_big_m, start_xy_m, fig=None, ax=None, ax_c=None)


    ''' 在做完 縮放 後 的 dis_coord_shifted_scaled_small '''
    step7_dis_coord_valid_area_storage_inverse_move_map_is_bm_before(dis_coord_shifted_scaled_small_m, start_xy_m, fig=None, ax=None, ax_c=None)

    plt.show()


def step6_util_dis_coord__move_map__ord_valid_coord__visual(dis_coord_m, start_xy_m, move_map_m, valid_boundary=1, jump_r=4, jump_c=4, fig=None, ax=None, ax_c=None):
    '''
    dis_coord_m： 主要想視覺畫的東西
    start_xy_m       ： 視覺化輔助用
    move_map_m ： 視覺化輔助用
    jump_r/c   ： 視覺化輔助用， 箭頭的密度
    '''
    before_alpha = 0.5
    after_alpha  = 0.7
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=2, fig_cols=7, ax_size=5.0, tight_layout=True)
    h_res, w_res = start_xy_m.shape[: 2]
    ##########################################################################################################################
    valid_mask = dis_coord_find_ord_valid_mask_and_ord_valid_coord(dis_coord_m, visual=False)

    # dis_coord_method1_visual = method1(x=dis_coord_m[..., 0], y=dis_coord_m[..., 1], mask_ch=0)
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
    fig, ax[0], ax_c = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up dis_coord appearance is fm,\n bottom appearance is bm",
        zticklabels=("bm", "", "", "fm"),
        jump_r=2, jump_c=2,
        before_C="orange", before_alpha=before_alpha, before_height=0,
        after_C = "blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax[0], ax_c=2, ax_r=0, ax_rows=2)

    '''圖0,3 畫 3D scatter 分不同平面  bm裡存的東西 模擬移動後的結果 + valid框框'''
    fig, ax[0], ax_c = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up crop a boundary as new bm, \n then bottom becomes new fm, but not all",
        zticklabels=("new fm \n but not all", "", "", "fm \n grab area \n as new bm"),
        jump_r=2, jump_c=2,
        valid_boundary=valid_boundary, boundary_C="orange", boundary_height=0.6,
        before_C="red", before_alpha=before_alpha, before_height=0,
        after_C ="blue",  after_alpha=after_alpha,   after_height =0.6,
        fig=fig, ax=ax[0], ax_c=3, ax_r=0, ax_rows=2)

    '''圖0,4 2D箭頭視覺化 移動的過程 + 移動後的結果 + valid框框'''
    fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 grab dis coord valid boundary as new bm",
        jump_r=jump_r, jump_c=jump_c,
        valid_boundary=valid_boundary, boundary_C="orange", boundary_linewidth=3,
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        show_before_move_coord=True, before_alpha=before_alpha, before_C="orange",
        show_after_move_coord =True, after_alpha=after_alpha,   after_C="blue",
        fig=fig, ax=ax[0], ax_c=4)

    '''圖0,5 2D箭頭視覺化 沒有超過 valid框框 的箭頭 '''
    fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="correspnd valid moves forms fm",
        jump_r=jump_r, jump_c=jump_c,
        arrow_C=1 - valid_mask, arrow_cmap="bwr",  ### 1 - valid_mask 是為了讓 mask 內畫藍色， mask外 畫紅色
        show_before_move_coord=False, before_alpha=before_alpha,
        show_after_move_coord =False, after_alpha=1.0, valid_boundary=valid_boundary,
        fig=fig, ax=ax[0], ax_c=5)

    '''圖0,6 畫 3D scatter 分不同平面  + valid框框'''
    fig, ax[0], ax_c = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
        fig_title="up crop a boundary as bm, \n then bottom correspnd valid moves forms fm",
        zticklabels=("fm", "", "", "bm"),
        valid_boundary=valid_boundary, boundary_C="orange", boundary_height=0.6, boundary_fill=True,
        jump_r=2, jump_c=2,
        before_C= 1 - valid_mask, before_cmap="bwr", before_alpha=before_alpha, before_height=0,  ### 1 - valid_mask 是為了讓 mask 內畫藍色， mask外 畫紅色
        after_C = "blue"        ,                    after_alpha=0.3,   after_height =0.6,
        fig=fig, ax=ax[0], ax_c=6, ax_r=0, ax_rows=2)


    '''圖1,0 幾乎== 圖0,4， 只有 title改一下 '''
    fig, ax[1], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
        fig_title="step6 grab dis coord valid boundary as new bm \n see detail at right figures",
        jump_r=jump_r, jump_c=jump_c,
        valid_boundary=valid_boundary, boundary_C="orange", boundary_linewidth=3,
        arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
        show_before_move_coord=True, before_alpha=before_alpha, before_C="orange",
        show_after_move_coord =True, after_alpha=after_alpha,   after_C="blue",
        fig=fig, ax=ax[1], ax_c=0)


    '''圖1,1~1,5 step7 找 dis_coord_m valid的地方'''
    fig, ax[1], ax_c, _ = dis_coord_find_ord_valid_mask_and_ord_valid_coord(dis_coord_m, visual=True, fig=fig, ax=ax[1], ax_c=1)


    return fig, ax, ax_c


def dis_coord_find_ord_valid_mask_and_ord_valid_coord(dis_coord_m, visual=False, fig=None, ax=None, ax_c=None):
    if(visual): fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=5, ax_size=5, tight_layout=True)
    ##################################################################################################################
    h_res, w_res = dis_coord_m.shape[: 2]

    canvas_mask_x        = np.zeros(shape=(h_res, w_res))  ### 分開來看 x valid的部分
    canvas_mask_y        = np.zeros(shape=(h_res, w_res))  ### 分開來看 y valid的部分
    canvas_mask_xy       = np.zeros(shape=(h_res, w_res))  ### 確實fm沒錯， 只抓 值在-1~1 之間的區域 會和 image_disorted_mask 一樣
    canvas_mask_contour  = np.zeros(shape=(h_res, w_res))  ### 確實fm沒錯， 只抓 值在-1~1 之間的區域 會和 image_disorted_mask 一樣
    canvas_dis_coord     = np.zeros(shape=(h_res, w_res, 3))
    dis_coord_visual     = method1(x=dis_coord_m[..., 0], y=dis_coord_m[..., 1] )
    for go_r, dis_coord_m_row in enumerate(dis_coord_m):
        for go_c, xy_coord in enumerate(dis_coord_m_row):
            if(-1.0 <= xy_coord[0] <= 1.0): canvas_mask_x[go_r, go_c] = 1
            if(-1.0 <= xy_coord[1] <= 1.0): canvas_mask_y[go_r, go_c] = 1
            if(-1.0 <= xy_coord[0] <= 1.0 and -1 <= xy_coord[1] <= 1.0):
                canvas_mask_xy   [go_r, go_c] += 1
                canvas_dis_coord [go_r, go_c] = dis_coord_visual[go_r, go_c]
            if(-1.0 <= xy_coord[0] <= 1.0 and -1.0 <= xy_coord[1] <= 1.0): canvas_mask_contour[go_r, go_c] += 1
            if(-1.3 <= xy_coord[0] <= 1.3 and -1.3 <= xy_coord[1] <= 1.3): canvas_mask_contour[go_r, go_c] += 1
            if(-1.5 <= xy_coord[0] <= 1.5 and -1.5 <= xy_coord[1] <= 1.5): canvas_mask_contour[go_r, go_c] += 1
    if(visual):
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_x valid between -1~+1");       ax[ax_c].imshow(canvas_mask_x,       cmap="gray", vmin=0, vmax=1);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_y valid between -1~+1");       ax[ax_c].imshow(canvas_mask_y,       cmap="gray", vmin=0, vmax=1);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_xy valid between -1~+1");      ax[ax_c].imshow(canvas_mask_xy,      cmap="gray", vmin=0, vmax=1);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_valid_contour between -1.5~+1.5"); ax[ax_c].imshow(canvas_mask_contour, cmap="gray", vmin=0, vmax=5);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_valid uv/fm"); ax[ax_c].imshow(canvas_dis_coord); ax_c += 1
        return fig, ax, ax_c, canvas_mask_xy
    else: return canvas_mask_xy


'''
step6 繼續編號寫下去
'''
# def get_inverse_

### dis_coord_big_m
def step7_dis_coord_valid_area_storage_inverse_move_map_is_bm(dis_coord_m, start_xy_m, fig=None, ax=None, ax_c=None):
    # valid_mask = dis_coord_find_ord_valid_mask_and_ord_valid_coord(dis_coord_m, visual=False)
    '''
    valid area： 我們現在是取跟 pytorch grid_sample() 一樣的 -1 ~ +1 的範圍是valid
    '''
    # fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=2, fig_cols=4, ax_size=5, tight_layout=True)
    ##################################################################################################################
    debug_spyder_dict["step7. dis_coord_big_m"] = dis_coord_m
    # fig, ax[0], ax_c = coord_m_2D_scatter(dis_coord_m, fig_title="dis_coord_big_m", fig=fig, ax=ax[0], ax_c=0)
    # ax[0, 0].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))
    # ax[0, 0].add_patch( patches.Rectangle( (-0.95, -0.95), 0.95 * 2, 0.95 * 2, edgecolor='green' , fill=False ))

    h_res, w_res = dis_coord_m.shape[:2]
    bm_w_res = 93  ### 可以自己調， 想應用到大圖就調大
    bm_h_res = 93  ### 可以自己調， 想應用到大圖就調大
    bm_xy_1_00f, bm_xy_1_00m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=bm_w_res, h_res=bm_h_res)

    xy_1_00f, xy_1_00m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=w_res, h_res=h_res)
    xy_0_95f, xy_0_95m = get_xy_f_and_m(x_min=-0.95, x_max=+0.95, y_min=-0.95, y_max=+0.95, w_res=w_res, h_res=h_res)

    # dis_coord_f = dis_coord_m.reshape(h_res * w_res, 2)

    ##################################################################################################################
    # import scipy.interpolate as spin
    ### 0.95 + move_map -> dis_coord -> back to 0.95, boundary grab 1.00， 因為這 boundary 是新的bm！ 可以自己取，取完以後要回去找 相對應 符合移動後在boundary 內的 ord 即可！
    ### 但是前面的這個 0.95 + move_map -> dis_coord -> back to 0.95 必須要對應到， 後面取的 boundary 裡面的值 才正確！
    ''' 方法1 錯的： 直接 對應回 -1~1(boundary) ， not match'''
    '''step7_util(dst_coord_m=dis_coord_m, dst_ratio=1.00, ord_coord_m=xy_1_00m, see_coord_m=bm_xy_1_00m, xy_m=start_xy_m)'''


    # bm_directly_1_00f = spin.griddata(dis_coord_f, xy_1_00f, bm_xy_1_00f, method="linear")                   ### 計算方法一(錯)： dis_coord_f不放大 直接對應回 -1~1(boundary)
    # bm_directly_1_00m = bm_directly_1_00f.reshape(bm_h_res, bm_w_res, 2)                                     ### flatten 轉 map

    # '''圖[0:2, 1]'''
    # bm_directly_1_00_move_map_m = bm_directly_1_00m - bm_xy_1_00m                                            ### 計算 move_map = dst - start
    # # coord_m_2D_scatter(dis_coord_m, fig=fig, ax=ax[0], ax_c=1)                                               ### 視覺化 完整 dis_coord
    # ax[0, 1].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))                  ### 視覺化 boundary框框
    # move_map_2D_arrow(bm_directly_1_00_move_map_m, start_xy_m=bm_xy_1_00m, arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
    #     fig_title="step7 dis_to_1_00_move_map", jump_r=4, jump_c=4,
    #     show_before_move_coord=True, before_alpha=0.1,
    #     show_after_move_coord =True, after_alpha=0.5,
    #     fig=fig, ax=ax[0], ax_c=1)

    # coord_m_2D_scatter(bm_directly_1_00m, fig_title="dis_to_1_00m_notOK", fig=fig, ax=ax[1], ax_c=1)                   ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    # ax[1, 1].scatter(xy_0_95m[..., 0], xy_0_95m[..., 1], c =valid_mask, s=1, cmap="gray", alpha=0.5)         ### 視覺化 原始 移動後不超過boundary 的 ord_coord mask
    # debug_spyder_dict["step7. spin.griddata bm_directly_1_00m"] = bm_directly_1_00m


    ##################################################################################################################
    ''' 方法2(paper17) ：把自己除0.95 放大一點 變成boundary的大小 再 對應回 -1~1(boundary) '''
    '''step7_util(dst_coord_m=dis_coord_m, dst_ratio=0.95, ord_coord_m=xy_1_00m, see_coord_m=bm_xy_1_00m, xy_m=start_xy_m)'''


    # bm_divede_then_1_00f = spin.griddata(dis_coord_f / 0.95, xy_1_00f, bm_xy_1_00f, method="linear")            ### 計算方法二(OK)： dis_coord_f 把自己除0.95 放大一點 變成boundary的大小 再 對應回 -1~1(boundary)
    # bm_divede_then_1_00m = bm_divede_then_1_00f.reshape(bm_h_res, bm_w_res, 2)                                  ### flatten 轉 map

    # '''圖[0:2, 2]'''
    # bm_divede_then_1_00_move_map_m = bm_divede_then_1_00m - bm_xy_1_00m                                         ### 計算 move_map = dst - start
    # # coord_m_2D_scatter(dis_coord_m / 0.95, fig=fig, ax=ax[0], ax_c=2)                                           ### 視覺化 完整 dis_coord( 放大一點/0.95)
    # ax[0, 2].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))                     ### 視覺化 boundary框框
    # move_map_2D_arrow(bm_divede_then_1_00_move_map_m, start_xy_m=bm_xy_1_00m, arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
    #     fig_title="step7 dis_bigger_then_to_1_00_move_map", jump_r=4, jump_c=4,
    #     show_before_move_coord=True, before_alpha=0.1,
    #     show_after_move_coord =True, after_alpha=0.5,
    #     fig=fig, ax=ax[0], ax_c=2)

    # coord_m_2D_scatter(bm_divede_then_1_00m, fig_title="dis_bigger_then_to_1_00m", fig=fig, ax=ax[1], ax_c=2)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    # ax[1, 2].scatter(xy_0_95m[..., 0], xy_0_95m[..., 1], c =valid_mask, s=1, cmap="gray", alpha=0.5)            ### 視覺化 原始 移動後不超過boundary 的 ord_coord mask
    # # ax[1, 2].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))
    # debug_spyder_dict["step7. spin.griddata bm_divede_then_1_00m_OK"] = bm_divede_then_1_00m


    ##################################################################################################################
    ''' 方法3： 直接 對應回 -0.95~0.95， 理論上來說應該是要這樣子， 因為我是從 -0.95~0.95走道 dis_coord， dis_coord應該要走回-0.95~+0.95， 實際上測試也確實如此 '''
    ### 但不大對，因為我的 valid area/boundary 是設定 -1~1
    step7_util(dst_coord_m=dis_coord_m, dst_ratio=1.00, ord_coord_m=xy_0_95m, see_coord_m=bm_xy_1_00m, xy_m=start_xy_m)
    # bm_directly_0_95f = spin.griddata(dis_coord_f, xy_0_95f, bm_xy_1_00f, method="linear")        ### 計算方法三(OK)： dis_coord_f 直接 對應回 -0.95~0.95
    # bm_directly_0_95m = bm_directly_0_95f.reshape(bm_h_res, bm_w_res, 2)                          ### flatten 轉 map

    # bm_directly_0_95_move_map_m = bm_directly_0_95m - bm_xy_1_00m                                 ### 計算 move_map = dst - start
    # # coord_m_2D_scatter(dis_coord_m, fig=fig, ax=ax[0], ax_c=3)                                    ### 視覺化 完整 dis_coord
    # ax[0, 3].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))       ### 視覺化 boundary框框
    # move_map_2D_arrow(bm_directly_0_95_move_map_m, bm_xy_1_00m, arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
    #     fig_title="step7 bm_directly_0_95_move_map", jump_r=4, jump_c=4,
    #     show_before_move_coord=True, before_alpha=0.1,
    #     show_after_move_coord =True, after_alpha=0.5,
    #     fig=fig, ax=ax[0], ax_c=3)

    # coord_m_2D_scatter(bm_directly_0_95m, fig_title="directly_0_95m_OK", fig=fig, ax=ax[1], ax_c=3)  ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    # ax[1, 3].scatter(xy_0_95m[..., 0], xy_0_95m[..., 1], c =valid_mask, s=1, cmap="gray", alpha=0.5)   ### 視覺化 原始 移動後不超過boundary 的 ord_coord mask
    # # ax[ax_c - 1].add_patch( patches.Rectangle( (-0.95, -0.95), 0.95 * 2, 0.95 * 2, edgecolor='green' , fill=False ))
    # debug_spyder_dict["step7. spin.griddata bm_directly_0_95m"] = bm_directly_0_95m

def step7_util(dst_coord_m, dst_ratio, ord_coord_m, see_coord_m, xy_m, fig=None, ax=None, ax_c=None):
    fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=4, ax_size=5, tight_layout=True)
    h_res, w_res = dst_coord_m.shape[:2]
    see_h_res, see_w_res = see_coord_m.shape[:2]
    ##################################################################################################################
    ord_coord_f = dst_coord_m.reshape(-1, 2)  ### dis
    dst_coord_f = ord_coord_m.reshape(-1, 2)  ### 0.95
    see_coord_f = see_coord_m.reshape(-1, 2)  ### 1.00
    import scipy.interpolate as spin
    inv_coord_f = spin.griddata(ord_coord_f / dst_ratio, dst_coord_f, see_coord_f, method="linear")  ### 計算
    inv_coord_m = inv_coord_f.reshape(see_h_res, see_w_res, 2)                                   ### flatten 轉 map
    inv_move_map = inv_coord_m - see_coord_m                                             ### 計算 move_map = dst - start
    # '''圖0'''
    # coord_m_2D_scatter(dst_coord_m / dst_ratio, fig_title="step7 (dis)ord_coord", fig=fig, ax=ax, ax_c=0)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord

    # '''圖1'''
    # move_map_3D_scatter(inv_move_map, start_xy_m=dst_coord_m / dst_ratio,
    #     fig_title="\n up dis_coord appearance is fm,\n bottom is bm", zticklabels=("bm", "", "", "fm"),
    #     jump_r=2, jump_c=2,
    #     fig=fig, ax=ax, ax_c=1)

    # '''圖2'''
    # coord_m_2D_scatter(dst_coord_m, fig_title="step7 util (dis)ord_coord", fig=fig, ax=ax, ax_c=2)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    # ax[2].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))       ### 視覺化 boundary框框
    # '''圖3'''
    # move_map_3D_scatter(inv_move_map, start_xy_m=dst_coord_m / dst_ratio,
    #     fig_title="up dis_coord appearance is fm,\n bottom is bm", zticklabels=("fm", "", "", "bm"),
    #     jump_r=2, jump_c=2,
    #     valid_boundary=1, boundary_height=0,
    #     before_height=0.5, after_height=0,
    #     fig=fig, ax=ax, ax_c=3)

    # '''圖4'''
    # move_map_2D_arrow(inv_move_map, see_coord_m, arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
    #     fig_title="step7 util bondary inv move", jump_r=4, jump_c=4,
    #     show_before_move_coord=True, before_alpha=0.1,
    #     show_after_move_coord =True, after_alpha=0.5,
    #     fig=fig, ax=ax, ax_c=4)
    # ax[4].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))       ### 視覺化 boundary框框

    # '''圖5'''
    # coord_m_2D_scatter(inv_coord_m, fig_title="step7 util bondary inv dst coord", fig=fig, ax=ax, ax_c=5)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    # valid_mask = dis_coord_find_ord_valid_mask_and_ord_valid_coord(dst_coord_m / dst_ratio, visual=False)
    # ax[5].scatter(xy_m[..., 0], xy_m[..., 1], c =valid_mask, s=1, cmap="gray", alpha=0.5)  ### 視覺化 原始 移動後不超過boundary 的 ord_coord mask

    valid_mask = dis_coord_find_ord_valid_mask_and_ord_valid_coord(dst_coord_m, visual=False)  ### 不用除 dst_ratio

    '''圖0'''
    before_alpha = 0.7; before_s = 3
    after_alpha  = 0.3; after_s = 1
    move_map_3D_scatter(inv_move_map, start_xy_m=dst_coord_m / dst_ratio,
        fig_title="up dis_coord appearance is fm,\n bottom is bm", zticklabels=("fm", "", "", "bm"),
        jump_r=2, jump_c=2,
        valid_boundary=1, boundary_height=0.5,
        before_height=0.5, before_alpha=before_alpha, before_s=before_s,
        after_height=0, after_alpha=after_alpha, after_s=after_s, after_C="blue",
        fig=fig, ax=ax, ax_c=0)

    '''圖1'''
    coord_m_2D_scatter(dst_coord_m / dst_ratio, fig_title="step7 util (dis)ord_coord", fig=fig, ax=ax, ax_c=1)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    ax[1].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))       ### 視覺化 boundary框框

    '''圖2'''
    before_alpha = 0.2
    after_alpha  = 0.8
    move_map_2D_arrow(inv_move_map, see_coord_m, arrow_C=None, arrow_cmap="gray",  ### 視覺化 boundary內 新bm 的 dis_coord/move_map/移動後的dis_coord
        fig_title="step7 util bondary inv move", jump_r=4, jump_c=4,
        show_before_move_coord=True, before_alpha=before_alpha,
        show_after_move_coord =True, after_alpha=after_alpha,
        fig=fig, ax=ax, ax_c=2)
    ax[2].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))       ### 視覺化 boundary框框

    '''圖3'''
    coord_m_2D_scatter(inv_coord_m, fig_title="step7 util bondary inv dst coord", fig=fig, ax=ax, ax_c=3)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    ax[3].scatter(xy_m[..., 0], xy_m[..., 1], c =valid_mask, s=1, cmap="gray", alpha=0.5)  ### 視覺化 原始 移動後不超過boundary 的 ord_coord mask



    # coord_m_2D_scatter(ord_coord_m, fig_title="step7 util (rec)dst_coord", fig=fig, ax=ax, ax_c=1)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord

    # coord_m_2D_scatter(dst_coord_m / dst_ratio, fig_title="step7 util ord_coord to dst_coord \n just see boundary", fig=fig, ax=ax, ax_c=2)             ### 視覺化 boundary內 新bm 的 移動後的dis_coord
    # ax[2].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))       ### 視覺化 boundary框框



### dis_coord_small_m
def step7_dis_coord_valid_area_storage_inverse_move_map_is_bm_before(dis_coord_m, start_xy_m, fig=None, ax=None, ax_c=None):
    '''
    valid area： 我們現在是取跟 pytorch grid_sample() 一樣的 -1 ~ +1 的範圍是valid
    '''
    # fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=4, ax_size=5, tight_layout=True)
    ##################################################################################################################
    debug_spyder_dict["step7. dis_coord_smallm"] = dis_coord_m
    # fig, ax, ax_c = coord_m_2D_scatter(dis_coord_m, fig_title="dis_coord_small_m", fig=fig, ax=ax, ax_c=ax_c)
    # ax[ax_c - 1].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))
    # ax[ax_c - 1].add_patch( patches.Rectangle( (-0.95, -0.95), 0.95 * 2, 0.95 * 2, edgecolor='green' , fill=False ))

    h_res, w_res = dis_coord_m.shape[:2]
    fm_w_res  = 93  ### 可以自己調， 想應用到大圖就調大
    fm_h_res  = 93  ### 可以自己調， 想應用到大圖就調大
    fm_xy_1_30f, fm_xy_1_30m = get_xy_f_and_m(x_min=-1.30, x_max=+1.30, y_min=-1.30, y_max=+1.30, w_res=fm_w_res, h_res=fm_h_res)
    fm_xy_1_23f, fm_xy_1_23m = get_xy_f_and_m(x_min=-1.23, x_max=+1.23, y_min=-1.23, y_max=+1.23, w_res=fm_w_res, h_res=fm_h_res)
    fm_xy_1_20f, fm_xy_1_20m = get_xy_f_and_m(x_min=-1.20, x_max=+1.20, y_min=-1.20, y_max=+1.20, w_res=fm_w_res, h_res=fm_h_res)
    fm_xy_1_00f, fm_xy_1_00m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=fm_w_res, h_res=fm_h_res)
    fm_xy_min_max_f, fm_xy_min_max_m = get_xy_f_and_m(x_min=dis_coord_m[..., 0].min(), x_max=dis_coord_m[..., 0].max(), y_min=dis_coord_m[..., 1].min(), y_max=dis_coord_m[..., 1].max(), w_res=fm_w_res, h_res=fm_h_res)

    xy_1_00f, xy_1_00m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=w_res, h_res=h_res)
    xy_0_95f, xy_0_95m = get_xy_f_and_m(x_min=-0.95, x_max=+0.95, y_min=-0.95, y_max=+0.95, w_res=w_res, h_res=h_res)

    # dis_coord_f = dis_coord_m.reshape(h_res * w_res, 2)
    ##################################################################################################################
    # import scipy.interpolate as spin
    ''' 方法0 ： 以前的fm'''
    ### 0.95 + move_map -> dis_coord -> back to 0.95, boundary grab 0.95， 因為沒有取新bm， 是直接 回去原本的地方， 原本的地方如果是 0.95， 回去也是0.95， boundary 也是原本的 0.95 囉！
    step7_util(dst_coord_m=dis_coord_m, dst_ratio=1.00, ord_coord_m=xy_0_95m, see_coord_m=fm_xy_1_00m, xy_m=start_xy_m)

    # using = 1.00
    # ### NOT OK
    # fm_before_f = spin.griddata(dis_coord_f, xy_0_95f, fm_xy_1_00f, method="linear")
    # fm_before_m = fm_before_f.reshape(fm_h_res, fm_w_res, 2)

    # fm_before_m_move_map_m = fm_before_m - dis_coord_m
    # fig, ax, ax_c = move_map_2D_arrow(fm_before_m_move_map_m, start_xy_m=dis_coord_m, arrow_C=None, arrow_cmap="gray",
    #     fig_title=f"step7 fm_before_{using}_move_map", jump_r=4, jump_c=4,
    #     show_before_move_coord=True, before_alpha=0.3,
    #     show_after_move_coord =False, after_alpha=0.5,
    #     fig=fig, ax=ax, ax_c=ax_c)
    # ax[ax_c - 1].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))
    # ax[ax_c - 1].add_patch( patches.Rectangle( (-0.95, -0.95), 0.95 * 2, 0.95 * 2, edgecolor='green' , fill=False ))

    # fig, ax, ax_c = move_map_2D_arrow(fm_before_m_move_map_m, start_xy_m=dis_coord_m, arrow_C=None, arrow_cmap="gray",
    #     fig_title=f"step7 fm_before_{using}_move_map", jump_r=4, jump_c=4,
    #     show_before_move_coord=True, before_alpha=0.3,
    #     show_after_move_coord =True, after_alpha=0.5,
    #     fig=fig, ax=ax, ax_c=ax_c)
    # ax[ax_c - 1].add_patch( patches.Rectangle( (-1, -1), 2, 2, edgecolor='blue' , fill=False ))
    # ax[ax_c - 1].add_patch( patches.Rectangle( (-0.95, -0.95), 0.95 * 2, 0.95 * 2, edgecolor='green' , fill=False ))


    # debug_spyder_dict[f"step7. spin.griddata fm_before_{using}_m"] = fm_before_m
    # debug_spyder_dict[f"step7. spin.griddata fm_before_{using}_m_move_map_m"] = fm_before_m_move_map_m


def step7_backup():
    pass
    # inverse_array = np.zeros(shape=(h_res, w_res, 3))  ### mask, x, y

    ### 根據我們定義的 valid area/boundary， 把 valid coord 轉換回 shape 為 (h_res, w_res) 的 array index
    # valid_min_x = -1
    # valid_max_x =  1
    # valid_min_y = -1
    # valid_max_y =  1
    # dis_coord_array_m =  dis_coord_m.copy()
    # dis_coord_array_m[..., 0] = (dis_coord_m[..., 0] - valid_min_x) / (valid_max_x - valid_min_x) * (w_res - 1)  ### (dis_coord_m + 1) / 2 * h_res
    # dis_coord_array_m[..., 1] = (dis_coord_m[..., 1] - valid_min_y) / (valid_max_y - valid_min_y) * (h_res - 1)  ### (dis_coord_m + 1) / 2 * h_res


    # for go_row, dis_coord_array_r in enumerate(dis_coord_array_m):
    #     for go_col, dis_coord_array_xy in dis_coord_array_r:
    #         dis_coord_x = dis_coord_array_xy[0]
    #         dis_coord_y = dis_coord_array_xy[1]
    #         if( 0 <= dis_coord_x < w_res and 0 <= dis_coord_y < h_res):
    #             inverse_array[int(dis_coord_y), int(dis_coord_x), 0] = 1
    #             inverse_array[int(dis_coord_y), int(dis_coord_x), 1] = dis_coord_x - move_map_m[go_row, go_col]
    #             inverse_array[int(dis_coord_y), int(dis_coord_x), 2] = dis_coord_y - move_map_m[go_row, go_col]



def step8(dis_coord_shifted_scaled_m):
    # ### 內插放大
    # ### 建立基礎物件
    # transfer_big_dis_coord = LinearNDInterpolator(start_xy_m.reshape(h_res * w_res, 2), dis_coord_m.reshape(h_res * w_res, 2))    ### 建立 LinearNDInterpolator物件，第一個參數放 正mesh 當作標, 第二個參數放 uv 當要被內差的值
    # transfer_big_move_map  = LinearNDInterpolator(start_xy_m.reshape(h_res * w_res, 2), move_map_m.reshape(h_res * w_res, 2))    ### 建立 LinearNDInterpolator物件，第一個參數放 正mesh 當作標, 第二個參數放 uv 當要被內差的值
    # ### 建立 放大正mesh
    # big_h_res = 4 * h_res
    # big_w_res = 4 * w_res
    # xy_big_f, xy_big_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=big_w_res, h_res=big_h_res)
    # ### 根據 放大正mesh 內插
    # big_dis_coord_f = transfer_big_dis_coord(xy_big_f)               ### 使用 LinearNDInterpolator物件，丟入 range一樣但中間切的格數多的正mesh，回傳 uv在隔數多時的樣子，跟上面的假設會依樣 -1.22... ~ +1.22...
    # big_dis_coord_m = big_dis_coord_f.reshape(big_h_res, big_w_res, 2)
    # big_move_map_f  = transfer_big_move_map(xy_big_f)               ### 使用 LinearNDInterpolator物件，丟入 range一樣但中間切的格數多的正mesh，回傳 uv在隔數多時的樣子，跟上面的假設會依樣 -1.22... ~ +1.22...
    # big_move_map_m  = big_move_map_f.reshape(big_h_res, big_w_res, 2)
    # ### 視覺化 valid ord_mesh
    # debug_spyder_dict["dis_coord_curl_f_big"] = big_dis_coord_f
    # debug_spyder_dict["dis_coord_curl_m_big"] = big_dis_coord_m

    # big_dis_coord_m = xy_big_m + big_move_map_m

    # fig, ax, ax_c = step6_util_dis_coord__move_map__ord_valid_coord__visual(big_dis_coord_m, xy_big_m, big_move_map_m, jump_r=16, jump_c=16)

    plt.show()



#######################################################################################################################################
#######################################################################################################################################
### 以下都是用取得參數以後，呼叫上面的本體做扭曲喔！
def get_rand_para(h_res, w_res, row, col, curl_probability, smooth=False):
    ratio = h_res / 4
    vert_x = np.random.randint(w_res)
    vert_y = np.random.randint(h_res)
    move_x = ((np.random.rand() - 0.5) * 0.9) * ratio  ### (-0.45~0.45) * ratio
    move_y = ((np.random.rand() - 0.5) * 0.9) * ratio  ### (-0.45~0.45) * ratio
    dis_type = np.random.rand(1)
    if(dis_type > curl_probability):
        dis_type = "fold"
        alpha = np.random.rand(1) * 50 + 50  ### 結果發現web上的還不錯
        if(smooth): alpha += 50   ### 老師想smooth一點，用step2_d嘗試 h=384,w=256時 fold的alpha +50還不錯
    #    alpha = (np.random.rand(1)*2.25 + 2.25)*ratio
    else:
        dis_type = "curl"
        alpha = (np.random.rand(1) / 2 + 0.5) * 1.7  ### ### 結果發現web上的還不錯，只有多rand部分除2
        if(smooth):
            alpha += 0.65  ### 老師想smooth一點，用step2_d嘗試 h=384,w=256時 curl的alpha 0.65還不錯，但別超過1.7
            if(alpha > 1.7): alpha = 1.7
    #    alpha = (np.random.rand(1)*0.045 + 0.045)*ratio

    return vert_x, vert_y, move_x, move_y, dis_type, alpha

### 只有 參數隨機產生， funciton 重複使用 上面寫好的function喔！
def distort_rand(dst_dir, start_index, amount, x_min, x_max, y_min, y_max, w_res, h_res, distort_time=None, curl_probability=0.3, move_x_thresh=40, move_y_thresh=55, smooth=False, write_npy=True):
    start_time = time.time()
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "distort_mesh_visuals")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "move_maps")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "distort_infos")

    move_maps = []
    for index in range(start_index, start_index + amount):
        dis_start_time = time.time()

        result_move_f = np.zeros(shape=(h_res * w_res, 2), dtype=np.float64)     ### 初始化 move容器
        if(distort_time is None): distort_time  = np.random.randint(4) + 1  ### 隨機決定 扭曲幾次

        for _ in range(distort_time):  ### 扭曲幾次
            while(True):  ### 如果扭曲太大的話，就重新取參數做扭曲，這樣就可以控制取到我想要的合理範圍
                vert_x, vert_y, move_x, move_y, dis_type, alpha = get_rand_para(h_res, w_res, curl_probability, smooth)  ### 隨機取得 扭曲參數
                # print("curl_probability",curl_probability, "dis_type",dis_type)
                move_f = get_dis_move_map( x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=vert_x, vert_y=vert_y, move_x=move_x, move_y=move_y , dis_type=dis_type, alpha=alpha, debug=False )  ### 用參數去得到 扭曲move_f
                max_move_x = abs(move_f[:, 0]).max()
                max_move_y = abs(move_f[:, 1]).max()
                if(max_move_x < move_x_thresh and max_move_y < move_y_thresh): break
                else: print("too much, continue random")
            result_move_f = result_move_f + move_f  ### 走到這就取到 在我覺得合理範圍的 move_f 囉，把我要的move_f加進去容器內～

            ### 紀錄扭曲參數
            if  (dis_type == "fold"): distrore_info_log(data_access_path + dst_dir + "/" + "distort_infos", index, h_res, w_res, distort_time, vert_x, vert_y, move_x, move_y, dis_type, alpha )
            elif(dis_type == "curl"): distrore_info_log(data_access_path + dst_dir + "/" + "distort_infos", index, h_res, w_res, distort_time, vert_x, vert_y, move_x, move_y, dis_type, alpha )


        ## 紀錄扭曲視覺化的結果
        # save_distort_mesh_visual(h_res, w_res, result_move_f, index)

        result_move_map = result_move_f.reshape(h_res, w_res, 2)  ### (..., 2)→(h_res, w_res, 2)
        result_move_map = result_move_map.astype(np.float32)
        if(write_npy) : np.save(data_access_path + dst_dir + "/" + "move_maps/%06i" % index, result_move_map)  ### 把move_map存起來，記得要轉成float32！
        print("%06i process 1 mesh cost time:" % index, "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time) )
        move_maps.append(result_move_map)
    return np.array(move_maps, dtype=np.float32)

### 用 step2_d去試 我想要的參數喔！ 測試結果還是不大像喔，要用step2_a_distort_page_and_pers.py 的 distort_more_like_page 才更像
def distort_like_page(dst_dir, start_index, x_min, x_max, y_min, y_max, w_res, h_res, write_npy=True):
    move_maps = []
    start_time = time.time()
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "distort_mesh_visuals")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "move_maps")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "distort_infos")

    ### 可以用step2_d去試 我想要的參數喔！
    distort_time = 1
    vert_y = 0
    move_x = 0
    dis_type = "curl"
    alpha = 1.5
    index = start_index

    for go_move_y in range(-13, 13):       ### 可以用step2_d去試 我想要的參數喔！
        for go_vert_x in range(120, 180):  ### 可以用step2_d去試 我想要的參數喔！
            dis_start_time = time.time()
            result_move_f = np.zeros(shape=(h_res * w_res, 2), dtype=np.float64)  ### 初始化 move容器
            move_f = get_dis_move_map( x_min, x_max, y_min, y_max, w_res, h_res, go_vert_x, vert_y, move_x, go_move_y , dis_type, alpha, debug=False )  ### 用參數去得到 扭曲move_f
            result_move_f = result_move_f + move_f  ### 把我要的move_f加進去容器內～
            distrore_info_log(data_access_path + dst_dir + "/" + "distort_infos", index, h_res, w_res, distort_time, go_vert_x, vert_y, move_x, go_move_y, dis_type, alpha )  ### 紀錄扭曲參數
            result_move_map = result_move_f.reshape(h_res, w_res, 2)  ### (..., 2)→(h_res, w_res, 2)
            result_move_map = result_move_map.astype(np.float32)
            # save_distort_mesh_visual(h_res, w_res, result_move_f, index)   ### 紀錄扭曲視覺化的結果
            np.save(data_access_path + dst_dir + "/" + "move_maps/%06i" % index, result_move_map)  ### 把move_map存起來，記得要轉成float32！
            print("%06i process 1 mesh cost time:" % index, "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time) )
            index += 1
            move_maps.append(result_move_map)
    return np.array(move_maps.astype(np.float32))

#######################################################################################################################################
#######################################################################################################################################
### 以下跟 紀錄、視覺化相關
def distrore_info_log(log_dir, index, h_res, w_res, distort_times, vert_x, vert_y, move_x, move_y, dis_type, alpha ):
    str_template = \
"\
vert_x=%i\n\
vert_y=%i\n\
move_x=%i\n\
move_y=%i\n\
dis_type=%s\n\
alpha=%i\n\
\n\
"
    with open(log_dir + "/" + "%06i-h_res=%i,w_res=%i,distort_times=%i.txt" % (index, h_res, w_res, distort_times), "a") as file_log:
        file_log.write(str_template % (vert_x, vert_y, move_x, move_y, dis_type, alpha))


def save_distort_mesh_visual(h_res, w_res, result_move_f, index):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 5)
    show_distort_mesh_visual(h_res, w_res, result_move_f, fig, ax)
    plt.savefig(data_access_path + "step2_flow_build/distort_mesh_visuals/%06i.png" % index)
    plt.close()

def show_distort_mesh_visual(x_min, x_max, y_min, y_max, w_res, h_res, move_f, fig, ax):
    xy_f, _ = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res)
    dis_coord = xy_f + move_f

    ax = change_into_img_2D_coord_ax(ax)
    ax_img = ax.scatter(dis_coord[:, 0], dis_coord[:, 1], c = np.arange(h_res * w_res), cmap="hsv", s=1)
    fig.colorbar(ax_img, ax=ax)

    debug_spyder_dict["dis_coordinate"] = dis_coord

def show_move_map_visual(move_map_m, ax):
    h_res, w_res = move_map_m.shape[:2]
    canvas = move_map_m / move_map_m.max()
    canvas = np.dstack( (canvas, np.ones(shape=(h_res, w_res, 1), dtype=np.float64) ) )
    ax.imshow(canvas)
    plt.show()


if(__name__ == "__main__"):
    ### 理解用，手動慢慢扭曲

    ### 印度那篇 move_map模擬成功 繼續往下模擬
    h_res    = 73
    w_res    = 73
    x_min    = -0.95
    x_max    = +0.95
    y_min    = -0.95
    y_max    = +0.95
    vert_x   = +0.441339
    vert_y   = -0.381496
    move_x   = +0.0
    move_y   = -0.116832
    dis_type = "curl"
    alpha    = 2.0
    move_map_curl_f = get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=vert_x, vert_y=vert_y, move_x=move_x, move_y=move_y, dis_type=dis_type, alpha=alpha, debug=True )  ### alpha:2~4

    #############################################################################################################################################
    ### 印度那篇 move_map模擬成功
    # h_res = 128
    # w_res = 128
    # x_min = -0.95
    # x_max = +0.95
    # y_min = -0.95
    # y_max = +0.95
    # vert_x = 0.441339
    # vert_y = -0.381496
    # move_f = get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=vert_x, vert_y=vert_y, move_x= 0.0, move_y= -0.116832, dis_type="curl", alpha= 2.0, debug=True )  ### alpha:2~4
    #############################################################################################################################################
    # row = 30
    # col = 30
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # move_f =          get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=col / 2, vert_y=row / 2, move_x=  -10, move_y=  10, dis_type="fold", alpha=  30, debug=True )  ### alpha:2~4
    # move_f = move_f + get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=      0, vert_y=     10, move_x= 3.5, move_y= 2.5, dis_type="fold", alpha=200, debug=True )

    # fig, ax = plt.subplots(1,1)
    # fig.set_size_inches(4, 5)
    # show_distort_mesh_visual(x_min, x_max, y_min, y_max, w_res, h_res, move_f,fig, ax)
    # plt.show()

    #############################################################################################################################################
    ### 隨機生成 256*256_2000張
    # dst_dir = "step2_build_flow_h=256,w=256_complex"
    # row=256
    # col=256
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # amount=2000
    # distort_rand(dst_dir=dst_dir, start_index=0, amount=2000, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55 )


    #############################################################################################################################################
    ### 隨機生成 384*256_2000張
    # dst_dir = "step2_build_flow_h=384,w=256_complex"
    # row=384
    # col=256
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # amount=2000
    # distort_rand(dst_dir=dst_dir, start_index=0, amount=amount, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55)
    ################################################
    #### 接續生成 頁面 扭曲影像，分開生成的原因是要 complex和complex+page 用的是相同的complex，所以page獨立生成，再把上面生成的結果複製一份，改名成complex+page，再把這裡生成的結果加進去
    # dst_dir = "step2_build_flow_h=384,w=256_page"
    # row=384
    # col=256
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # distort_like_page(dst_dir=dst_dir, start_index=2000 , x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res)  ### 目前寫死，固定生成60*26個 move_maps喔！
    #############################################################################################################################################
    ### 平滑多一點 384*256_1500張
    # dst_dir = "step2_build_flow_h=384,w=256_smooth_curl+fold"
    # row = 384
    # col = 256
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # amount = 450
    # distort_rand(dst_dir=dst_dir, start_index=amount * 0, amount=amount, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, distort_time=1, curl_probability=1.0, move_x_thresh=40, move_y_thresh=55, smooth=True )
    # distort_rand(dst_dir=dst_dir, start_index=amount * 1, amount=amount, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, distort_time=1, curl_probability=0.0, move_x_thresh=40, move_y_thresh=55, smooth=True )

    #############################################################################################################################################
    ### old應該要拿掉，有生成像page的60*26張，剩下用隨機補滿2000張
    # start_index = 0
    # amount = 60*26
    # row = 384#400#256#40*10#472 #40*10
    # col = 256#300#256#30*10#304 #30*10
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # dst_dir = "step2_flow_build_page"
    # distort_like_page(dst_dir=dst_dir, start_index=0    , x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res) ### 目前寫死，固定生成60*26個 move_maps喔！
    # distort_rand     (dst_dir=dst_dir, start_index=60*26, amount=2000-60*26, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res,distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55 )
