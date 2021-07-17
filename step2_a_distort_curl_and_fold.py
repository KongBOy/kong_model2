import sys
sys.path.append("kong_util")

import numpy as np
import matplotlib.pyplot as plt
from util import get_xy_f_and_m
from matplot_fig_ax_util import check_fig_ax_init, mesh3D_scatter_and_z0_plane, move_map_2D_moving_visual
from matplot_fig_ax_util import coord_f_2D_scatter, move_map_1D_value, move_map_2D_arrow


from step2_a_util                                                         import wt_calculate, step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord, apply_fm_to_get_dis_img, apply_bm_to_get_rec_img
from step2_a_debug_step4_wt_simulate                                      import step4_wt_simulate_visual
from step2_a_debug_step5_move_map_simulate                                import step5_move_map_simulate_visual
from step2_a_debug_step6a_dis_coord_adjust_visual                         import step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual
from step2_a_debug_step6b_papr17_dis_coord_big_and_ord_valid_coord_visual import step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual
from step2_a_debug_step6c_before_dis_coord_small_visual                   import step6_debug_c_Dis_coord_small_Move_map_Boundary_visual
from step2_a_debug_step7b_papr17_get_bm_value_visual                      import step7_visual_util_b_paper17
from step2_a_debug_step7c_before_get_fm_value_visual                      import step7_visual_util_c_before

debug_spyder_dict = dict()


''' step1~5 '''
### 整個function都是用 image的方式來看(左上角(0,0)，x往右邊走增加，y往上面走增加)
def get_dis_move_map(start_xy_m, vert_x, vert_y, move_x, move_y, dis_type="fold", alpha=50, debug=False):
    h_res, w_res = start_xy_m.shape[:2]
    start_xy_f = start_xy_m.reshape(-1, 2)
    '''
    start_xy_m    ： 正mesh，值域為 0~x_max, 0~y_max , shape 為 (h_res , w_res, 2), m 是 map_form 的意思， start_xy_f[..., 0]是X座標， start_xy_f[..., 1]是Y座標
    start_xy_f    ： 正mesh，值域為 0~x_max, 0~y_max , shape 為 (h_res * w_res, 2), f 是 flatten  的意思， start_xy_f[ : , 0]是X座標， start_xy_f[ : , 1]是Y座標
    vert_x  ： 應該要 x_min <= vert_x <= x_max
    vert_y  ： 應該要 y_min <= vert_y <= y_max
    move_x/y： 雖然是沒有限定， 不過應該也是要在 min ~ max 之間會比較合理
    dis_type： 有 fold 和 curl 可以選
    debug ： 建議要搭配 spyder 一起使用， 才能用 Variable Explore 看變數喔！
    '''
    if(debug): fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=12, ax_size=5, tight_layout=True)
    ################################################################################################################################
    ''' step1. 把正mesh 平移到 以 vert_xy 為中心的 正mesh
    vtex         ： 正mesh 中 的 一個座標           ， 值域在 0~x_max, 0~y_max 之間, shape 為 (1, 2)
    xy_shifted_f ： 位移到 以 vtex 為(0, 0) 的正mesh                    , shape 為 (h_res * w_res, 2)
    '''
    vtex = np.array([vert_x, vert_y])  ### 指定的扭曲點xy座標
    xy_shifted_f  = start_xy_f - vtex        ### 位移整張mesh變成以扭曲點座標為原點

    ################################################################################################################################
    ''' step2.選一個移動向量 來 決定每個點要怎麼移動
        move_x       ： 一個值
        move_y       ： 一個值
        move_xy      ： [[move_x, move_y   ]]， shape為 (1, 2)
    '''
    if(move_x == 0 and move_y == 0):  ### 兩者不能同時為0，要不然算外積好像有問題
        move_x = 0.00001
        move_y = 0.00001
    move_xy = np.array([[move_x, move_y]], dtype=np.float64)

    ''' step1,2 視覺化 xy 和 xy_shift 和 move_xy'''
    if(debug):
        fig, ax, ax_c = coord_f_2D_scatter(start_xy_f,         h_res, w_res, fig_title="step1.xy_mesh, step2. move_xy",       fig=fig, ax=ax, ax_c=ax_c)
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

        d_raw   ： 剛做完 corss 最原始的結果， shape 為 (h_res * w_res,)， 值為 兩xy向量形成的平行四邊形面積 往z軸跑 的值
                        因為  以扭曲點為中心的正mesh每個點(當向量) 是二維， move向量 是 二維，
                        二維 跟 二維 的 向量做 cross， 其 結果向量的：
                            方向：一定在 第三維 且 兩維必0，
                            長度：就是 兩個向量 圍出的平行四邊形面積

        d_abs     ： d_raw 取絕對值！ 因為 d 是 distance 的概念， 只管大小， 不管方向～
        d_abs_norm： d向量的長度 除去 move向量的長度
    '''
    ### 研究完paper17 ### 研究完paper17 發現用二維就可做corss了！ xy_shifted_f 和 move_xy 就不用多一個 z channel囉！
    d_raw = np.cross(xy_shifted_f, move_xy)  ### shape=(...,)
    d_abs = np.absolute(d_raw)               ### 為了視覺化 所以把 d_abs_norm 分兩步寫
    d_abs_norm = d_abs / d_abs.max()         ### 這比較符合 paper09 DocUNet 的描述， 可以讓 alpha 大的時候 變化較global， alpha 小的時候 變化較local
    # d_abs_norm = d_abs / (np.linalg.norm(move_xy, ord=2))  ### 這是網路上 兩個example 的寫法， 雖然跑起來都沒問題， 但比較不符合 paper09 DocUNet的描述！ norm2 就是算 move_xy向量的 長度 喔！

    ''' 從step3 之後 都用 flatten 轉回 map 的形式 囉！ '''
    xy_shifted_m = xy_shifted_f.reshape(h_res, w_res, 2)
    d_abs_norm_m = d_abs_norm.reshape(h_res, w_res, 1)


    ''' step3 視覺化 d'''
    if(debug):
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
    move_map_f = move_map_m.reshape(-1, 2)

    if(debug):
        debug_spyder_dict["move_map_m"] = move_map_m

        fig, ax, ax_c = move_map_1D_value(move_map_m=move_map_m, move_x=move_x, move_y=move_y, fig_title="step5. see move_map value", fig=fig, ax=ax, ax_c=ax_c)
        fig, ax, ax_c = move_map_2D_moving_visual(move_map_m=move_map_m, start_xy_m=start_xy_m, fig_title=f"step5. {dis_type}", fig=fig, ax=ax, ax_c=ax_c)

        '''simulate wt_fold/curl 兩種都模擬看看長怎樣'''
        step5_move_map_simulate_visual(start_xy_m, move_xy, d_abs_norm_m, alpha_fold=0.8, alpha_curl=2.0)

    ### 存debug圖
    if(debug):
        plt.savefig("debug_result")

    return move_map_f, move_map_m  ### shape：(h_res, w_res, 2)


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
''' step6 繼續編號寫下去 '''
def move_map_value_adjust_by_dis_coord_and_return_dis_coord(adjust_type,
                                                            adjust_ratio,
                                                            move_map_m, start_xy_m,
                                                            debug=True,
                                                            jump_r=4, jump_c=4,
                                                            boundary_value=0,
                                                            before_alpha=0.5,
                                                            after_alpha =0.7):
    '''
    共同都會對 dis_coord_m 做 "位移" 讓 dis_coord_m appearance 的正中心 在原點(0, 0)， 之後會再針對 adjust_type 來做 "縮放"
    如何調整move_map 的話是先 用正mesh 根據 move_map_m 移動後 得到的 dis_coord_m， 藉由調整 dis_coord 來 調整 move_map
    最後是return dis_coord_m， 如果想使用move_map_m 就使用 dis_coord - start_xy_m = move_map_m 囉！

    start_xy_m  ： 起始座標點         ， 用來計算 dis_coord_m 用的
    move_map_m  ： 移動的箭頭(不含座標)， 用來計算 dis_coord_m 用的
    adjust_type ：
         big    ： paper17的方法 ，把 dis_coord_m 值放大(超過 自訂的boundary)， 在 dis_coord_m 上取新的bm， 取 valid_ord_coord 當fm
        small   ： 以前的方法， 原本 dis_coord_m == move_map + ord_coord 直接當bm， 把 dis_coord_m 值縮小(限制在 自訂的boundary 內)， 縮小的dis_coord_m 的 appearance 就當fm， 之後會再把 fm 對應回 ord_coord 這樣子
        debug   ： 建議要搭配 spyder 一起使用， 才能用 Variable Explore 看變數喔！

    boundary_value：
      對paper17方法來說： 是要框出 new bm 要回到哪裡的範圍
      對before方法來說 ： 是要框出 ord_coord 的範圍， 所以 在製作 dis_coord_small 那邊 visual部分的boundary_value設定應為： boundary_value = ord_base_before
    '''
    if(debug): fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=2, fig_cols=5, ax_size=5, tight_layout=True)
    ##########################################################################################################################
    h_res, w_res = start_xy_m.shape[: 2]

    '''
    step6a. start_xy_m 根據 move_map_m 移動， 就會得到 dis_coord_m， 即 dis_coord_m = start_xy_m + move_map_m
    '''
    dis_coord_m = start_xy_m + move_map_m  ### dis_coord
    if(debug):
        '''圖0,0 2D箭頭視覺化 bm裡存的東西'''
        fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
            fig_title="step6 2D view \n storage array is still bm",
            jump_r=jump_r, jump_c=jump_c,
            arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
            show_before_move_coord=True,  before_alpha=before_alpha,
            show_after_move_coord =False, after_alpha=after_alpha,
            fig=fig, ax=ax[0], ax_c=0)

        '''圖0:1,1 3D scatter 分不同平面 原始dis_coord '''
        fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_m, move_map_m, start_xy_m,
                                                                              current_state="等等外面個別設定",
                                                                              jump_r=jump_r, jump_c=jump_c,
                                                                              boundary_value=0,
                                                                              before_alpha=before_alpha,
                                                                              after_alpha = after_alpha,
                                                                              fig=fig, ax=ax, ax_c=1)
        ax[0, 1].set_title("step6 2D view \n simulate bm move will get dis coord")
        ax[1, 1].set_title("step6 3D view\n up: dis_coord appearance is fm,\n bottom: appearance is bm")

        '''圖0:1,2 3D scatter 分不同平面 原始dis_coord '''
        fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_m, move_map_m, start_xy_m,
                                                                              current_state="dis_coord",
                                                                              jump_r=jump_r, jump_c=jump_c,
                                                                              boundary_value=boundary_value,
                                                                              before_alpha=before_alpha,
                                                                              after_alpha = after_alpha,
                                                                              fig=fig, ax=ax, ax_c=2)
        debug_spyder_dict["step6a. dis_coord_m"] = dis_coord_m
    #######################################################################################################################
    '''
    step6b. dis_coord_m 做些 平移( 平移整張 dis_coord_m 把 他的 中心點移到 原點)， 得到 dis_coord_shifted_m， 如果要用debug視覺化 記得回推一下 move_map_shifted_m 喔！
    '''
    dis_coord_x_m = dis_coord_m[..., 0]
    dis_coord_y_m = dis_coord_m[..., 1]
    dis_coord_xmin = dis_coord_x_m.min(); dis_coord_xmax = dis_coord_x_m.max()
    dis_coord_ymin = dis_coord_y_m.min(); dis_coord_ymax = dis_coord_y_m.max()
    dis_coord_xcenter = (dis_coord_xmin + dis_coord_xmax) / 2
    dis_coord_ycenter = (dis_coord_ymin + dis_coord_ymax) / 2
    dis_coord_shifted_m = dis_coord_m.copy()
    dis_coord_shifted_m[..., 0] -= dis_coord_xcenter
    dis_coord_shifted_m[..., 1] -= dis_coord_ycenter
    if(debug):
        move_map_shifted_m = dis_coord_shifted_m - start_xy_m
        '''圖0:1,3 3D scatter 分不同平面 原始dis_coord_scaled '''
        fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_shifted_m, move_map_shifted_m, start_xy_m,
                                                                              current_state="dis_coord_shifted",
                                                                              jump_r=jump_r, jump_c=jump_c,
                                                                              boundary_value=boundary_value,
                                                                              before_alpha=before_alpha,
                                                                              after_alpha = after_alpha,
                                                                              fig=fig, ax=ax, ax_c=3)
        debug_spyder_dict["step6b. dis_coord_xcenter"]   = dis_coord_xcenter
        debug_spyder_dict["step6b. dis_coord_ycenter"]   = dis_coord_ycenter
        debug_spyder_dict["step6b. dis_coord_shifted_m"] = dis_coord_shifted_m

    #######################################################################################################################
    '''
    step6c. dis_coord_shifted_m 做些 縮放， 如果要用debug視覺化 記得回推一下 move_map_shifted_scaled_m 喔！
    '''
    dis_coord_shifted_scaled_m =  dis_coord_shifted_m.copy()
    dis_coord_shifted_scaled_m *= adjust_ratio

    if(debug):
        '''圖0:1,4 3D scatter 分不同平面 原始dis_coord_shifted_scaled '''
        debug_spyder_dict["step6c. raito"] = adjust_ratio
        move_map_shifted_scaled_m   = dis_coord_shifted_scaled_m   - start_xy_m
        fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_shifted_scaled_m, move_map_shifted_scaled_m, start_xy_m,
                                                                            current_state=f"dis_coord_shifted_scaled_{adjust_type}",
                                                                            jump_r=jump_r, jump_c=jump_c,
                                                                            boundary_value=boundary_value,
                                                                            before_alpha=before_alpha,
                                                                            after_alpha =after_alpha,
                                                                            fig=fig, ax=ax, ax_c=4)
        ### adjust_type == "big" 是paper17的方法
        if(adjust_type == "big"):
            debug_spyder_dict[f"step6c. dis_coord_shifted_scaled_big_m"]   = dis_coord_shifted_scaled_m
            step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual(dis_coord_shifted_scaled_m,  move_map_shifted_scaled_m, start_xy_m,
                                                                                boundary_value=boundary_value,
                                                                                jump_r=4, jump_c=4,
                                                                                before_alpha=before_alpha,
                                                                                after_alpha =after_alpha,
                                                                                fig=None, ax=None, ax_c=None)
        ### adjust_type == "small" 是before的方法
        ###   以前的版本 本身 就不適合 套用 step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual 因為不需要 在 dis_coord_shifted_scaled_m 上取 新bm 和 找 ord_valid_coord，
        ###   而是直接用 dis_coord_shifted_scaled_m(縮小的)的appearance 當fm， 之後會再把 fm 對應回 ord_coord 這樣子，
        ###   所以 直接給 small 也寫一個視覺化囉！
        elif(adjust_type == "small"):
            debug_spyder_dict[f"step6c. dis_coord_shifted_scaled_small_m"]   = dis_coord_shifted_scaled_m
            step6_debug_c_Dis_coord_small_Move_map_Boundary_visual(dis_coord_shifted_scaled_m, move_map_shifted_scaled_m, start_xy_m,
                                                                  jump_r=jump_r, jump_c=jump_c,
                                                                  boundary_value=boundary_value,
                                                                  before_alpha=before_alpha,
                                                                  after_alpha =after_alpha,
                                                                  fig=None, ax=None, ax_c=None)
    return dis_coord_shifted_scaled_m


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
''' step7 繼續編號寫下去 '''
def step7a_dis_backto_ord_and_see_where(dis_coord_m, ord_coord_f, see_coord_f, img_w, img_h):
    dis_coord_f  = dis_coord_m.reshape(-1, 2)
    ##################################################################################################################
    import scipy.interpolate as spin
    see_inv_coord_f = spin.griddata(dis_coord_f, ord_coord_f, see_coord_f, method="cubic")  ### 計算， dst 為 dis_coord_big 的話 see_inv_coord_f 填滿滿， dst 為 dis_coord_small 的話 see_inv_coord_f 會有 nan
    see_inv_coord_m = see_inv_coord_f.reshape(img_h, img_w, 2)                                           ### flatten 轉 map
    return see_inv_coord_m


''' dis_coord_big_m '''
def step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_big_m, ord_base, see_base, img_w, img_h,
                                                                                                     debug=False, start_xy_base=1.0, dis_rescale_again=1.0):
    '''
    start_xy_m ： 視覺化用的， ord_coord 不一定 == start_xy 喔！ 因為 ord_coord 可能被 "再次rescale" 這樣子拉
    valid area(即boundary 、 see_coord的範圍 )： 因為是取 new bm 並反推 new fm， 所以new bm 的範圍可以取我們好處理的範圍， 我們現在是取跟 pytorch grid_sample() 一樣的 -1 ~ +1 的範圍是valid
    因為是在 dis_coord_big_m 抓 see_coord_m 一定會填滿(step6本來就要設定放大要超過 boundary(see_coord的範圍))， 所以 inv_see_coord_m 一定填得滿滿的 不會有nan
    '''
    h_res, w_res = dis_coord_big_m.shape[:2]
    ord_coord_f, ord_coord_m   = get_xy_f_and_m(x_min=-ord_base, x_max=+ord_base, y_min=-ord_base, y_max=+ord_base, w_res=w_res, h_res=h_res)  ### w_res, h_res
    see_coord_f, see_coord_m   = get_xy_f_and_m(x_min=-see_base, x_max=+see_base, y_min=-see_base, y_max=+see_base, w_res=img_w, h_res=img_h)  ### img_w, img_h
    see_inv_coord_m = step7a_dis_backto_ord_and_see_where(dis_coord_m=dis_coord_big_m, ord_coord_f=ord_coord_f, see_coord_f=see_coord_f, img_w=img_w, img_h=img_h)
    see_inv_move_map_m = see_inv_coord_m - see_coord_m    ### 計算 move_map = dst - start， 如果有nan 減完 仍為 nan ， dis_coord_big 的話 see_inv_coord_f 填滿滿

    new_bm = see_inv_coord_m
    new_fm = dis_coord_big_m  ### * ord_valid_mask[:, :, np.newaxis]  ### 就算不用mask遮住， pytorch 的 gridsample 還是可以運作喔！
    ord_valid_mask = step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord(dis_coord_big_m, boundary_value=ord_base, visual=False)
    ##################################################################################################################
    if(debug):
        _, start_xy_m = get_xy_f_and_m(x_min=-start_xy_base, x_max=+start_xy_base, y_min=-start_xy_base, y_max=+start_xy_base, w_res=img_w, h_res=img_h)  ### img_w, img_h
        step7_visual_util_b_paper17(see_inv_coord_m, see_inv_move_map_m,
                                    dis_coord_m=dis_coord_big_m, ord_coord_m=ord_coord_m, see_coord_m=see_coord_m,
                                    boundary_value=ord_base,
                                    start_xy_m=start_xy_m, ord_valid_mask=ord_valid_mask, dis_rescale_again=dis_rescale_again)
        debug_spyder_dict["step7. dis_coord_big_m"] = dis_coord_big_m
        debug_spyder_dict["step7. ord_valid_mask"] = ord_valid_mask
        debug_spyder_dict["step7. ord_coord_m"] = ord_coord_m
        debug_spyder_dict["step7. see_inv_coord_m"] = see_inv_coord_m
        debug_spyder_dict["step7. see_inv_move_map_m"] = see_inv_move_map_m
        debug_spyder_dict["step7. see_inv_move_map_m.isnan()"] = np.isnan(see_inv_move_map_m)
        debug_spyder_dict["step7. start_xy_m"] = start_xy_m
    return new_bm, new_fm, ord_valid_mask


##################################################################################################################
##################################################################################################################
''' dis_coord_small_m '''
def step7c_Before_Dis_coord_valid_area_is_Fm_and_inverse_backto_Ord_to_get_fm_value(dis_coord_small_m, ord_base, see_base, img_w, img_h,
                                                                                    debug=False, ord_ratio=1.0):
    '''
    valid area(即boundary 、 see_coord的範圍 )： 注意因為不是取new bm， 所以範圍不一定
    因為是在 dis_coord_small_m 抓 see_coord_m 一定 不會填滿(step6本來就要設定縮小 要在 boundary(see_coord的範圍) 裡面)， 所以 inv_see_coord_m 一定填不滿， 會有nan
    '''
    h_res, w_res = dis_coord_small_m.shape[:2]
    ord_coord_f, ord_coord_m   = get_xy_f_and_m(x_min=-ord_base, x_max=+ord_base, y_min=-ord_base, y_max=+ord_base, w_res=w_res, h_res=h_res)  ### w_res, h_res
    see_coord_f, see_coord_m   = get_xy_f_and_m(x_min=-see_base, x_max=+see_base, y_min=-see_base, y_max=+see_base, w_res=img_w, h_res=img_h)  ### img_w, img_h
    see_inv_coord_m = step7a_dis_backto_ord_and_see_where(dis_coord_m=dis_coord_small_m, ord_coord_f=ord_coord_f, see_coord_f=see_coord_f, img_w=img_w, img_h=img_h)
    see_inv_move_map_m = see_inv_coord_m - see_coord_m    ### 計算 move_map = dst - start， 如果有nan 減完 仍為 nan ， dis_coord_small 的話 see_inv_coord_f 會有 nan
    fm = see_inv_coord_m
    bm = dis_coord_small_m
    fm_nan_mask = 1 - np.isnan(see_inv_coord_m).astype(np.int32)[..., 0]

    if(debug):
        debug_spyder_dict["step7. bm == dis_coord_smallm"] = dis_coord_small_m
        debug_spyder_dict["step7. fm == see_inv_coord_m"]  = see_inv_coord_m
        debug_spyder_dict["step7. fm_nan_mask"] = fm_nan_mask
        step7_visual_util_c_before(see_inv_coord_m, see_inv_move_map_m, fm_nan_mask=fm_nan_mask,
                                    dis_coord_m=dis_coord_small_m, ord_ratio=ord_ratio, ord_coord_m=ord_coord_m, see_coord_m=see_coord_m)
    return fm, bm, fm_nan_mask

##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################


if(__name__ == "__main__"):
    import cv2
    from util import get_dir_certain_img
    # img_dir = "H:/Working/2 Blender/data_dir/0_ord/tex"
    # imgs = get_dir_certain_img(img_dir, certain_word="pr_Page_001.jpg", float_return=False)
    img_dir = "kong_util/img_data"
    imgs = get_dir_certain_img(img_dir, certain_word="rainbow.png", float_return=False)
    #######################################################################################################
    ord_img = imgs[0, :, :, ::-1]
    ord_img = cv2.resize(ord_img, (65, 65), interpolation=cv2.INTER_AREA)
    img_h, img_w = ord_img.shape[:2]

    ### 理解用，手動慢慢扭曲
    '''
    x/y_min/max：
        是用 np.linspace 喔！ x_min ~ x_max 就是真的到那個數字！ 不像 np.arange() 會是 x_min ~ x_max-1！
        所以如果要還原以前寫的東西 要記得 x_max-1 喔！
    w/h_res ： min ~ max 之間切多少格
    '''

    ### 印度那篇 move_map模擬成功 繼續往下模擬
    h_res    = img_h  ### 129  ### 77
    w_res    = img_w  ### 129  ### 77
    ord_ratio = 0.75  ### paper17主要是用來 rescale dis_coord 用的， 自己實作完覺得這個參數沒必要， 直接再 rescale的時候 給一個明確的數字不就好了， 在這邊控制 mesh縮放 的話 使用 LinearNDInterpolator 還要注意 怎麼對應 和 see_coord 超級麻煩， 如果我自己用的話設 1 就好了吧！
    x_min    = -1.00 * ord_ratio
    x_max    = +1.00 * ord_ratio
    y_min    = -1.00 * ord_ratio
    y_max    = +1.00 * ord_ratio
    vert_x   = +0.441339
    vert_y   = -0.381496
    move_x   = +0.0
    move_y   = -0.116832
    dis_type = "curl"
    alpha    = 2.0
    pytorch_gridsample_boundary = 1.00
    start_xy_f, start_xy_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思

    debug_1to5 = False
    debug_papr17 = True  ### False
    debug_before = True  ### False
    ##################################################################################################################
    ##################################################################################################################
    '''step1~5'''
    _, move_map_curl_m = get_dis_move_map(start_xy_m, vert_x=vert_x, vert_y=vert_y, move_x=move_x, move_y=move_y, dis_type=dis_type, alpha=alpha, debug=debug_1to5)  ### alpha:2~4

    '''step6 dis_coord_m = move_map_m + start_xy_m， 調整 dis_coord_m 變成 big 版本， 要超過 ord_ratio 喔！'''
    dis_coord_shifted_scaled_big_m   = move_map_value_adjust_by_dis_coord_and_return_dis_coord(adjust_type="big", adjust_ratio=1.5,  move_map_m=move_map_curl_m, start_xy_m=start_xy_m,
                                                                                               debug=debug_papr17, boundary_value=ord_ratio)
    '''step6 dis_coord_m = move_map_m + start_xy_m， 調整 dis_coord_m 變成 small 版本 不能超過 ord_ratio 喔！'''
    dis_coord_shifted_scaled_small_m   = move_map_value_adjust_by_dis_coord_and_return_dis_coord(adjust_type="small", adjust_ratio=0.8,  move_map_m=move_map_curl_m, start_xy_m=start_xy_m,
                                                                                                 debug=debug_before, boundary_value=ord_ratio)

    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ''' Paper17 方法4：
          最直覺的方法是 直接對應回原來的地方：
          ord_ratio + move_map -> dis_coord -> back to ord_ratio, boundary grab ord_ratio
          因為沒有取新bm， 是直接 回去原本的地方， 原本的地方如果是 ord_ratio， 回去也是ord_ratio， boundary 也是原本的 ord_ratio 囉！
          所以設定：
            ord_base_paper17 = ord_ratio
            see_base_paper17 = ord_base_paper17
          最後別忘了 把 fm, bm 從 ord_ratio 放大回 1.00 才對喔
    '''
    ord_base_paper17 = ord_ratio
    see_base_paper17 = ord_base_paper17

    ''' dis_coord_big 看有沒有需要 "再次縮放"， 通常是想 把 start_coord 的 -ord_ratio~+ord_ratio 轉回 -1~1， 不過這裡不想縮放， 設定1.00 就不會縮放了 '''
    dis_rescale_again  = 1.00  ### dis_coord_big 不變
    dis_coord_shifted_scaled_big_m /= dis_rescale_again  ### dis_coord_big 不變
    ord_base_paper17               /= dis_rescale_again  ### ord_ratio 不變， 代表可以 ord_ratio + move_map -> dis_coord -> back to ord_ratio
    see_base_paper17               = ord_base_paper17                ### boundary grab ord_ratio
    ############################################################################################################
    '''step7 在做完 再次縮放 後 的 dis_coord_shifted_scaled_big 上 找valid區域(-ord_ratio~ord_ratio) 當 new"Bm"， 並對應回原始 ord_coord(-ord_ratio~ord_ratio)'''
    new_bm, new_fm, ord_valid_mask  = step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_shifted_scaled_big_m,
                                    ord_base=ord_base_paper17, see_base=see_base_paper17, img_w=img_w, img_h=img_h,
                                    debug=debug_papr17, start_xy_base=ord_base_paper17,  dis_rescale_again=dis_rescale_again)
    new_bm = new_bm / ord_ratio  ### 放大回 -1~1 才正確喔！
    new_fm = new_fm / ord_ratio  ### 放大回 -1~1 才正確喔！
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ########################################################################################################################################################################
    ''' Before 方法2： see 取 -ord_ratio~ord_ratio應該才是對的， 最後還要把 fm, bm 從 0.95 放大回 1.00 才正確'''
    ### 0.95 + move_map -> dis_coord -> back to 0.95, boundary grab 0.95， 因為沒有取新bm， 是直接 回去原本的地方， 原本的地方如果是 0.95， 回去也是0.95， boundary 也是原本的 0.95 囉！
    ord_base_before  = ord_ratio
    see_base_before  = ord_base_before

    ''' step7 在做完 縮放 後 的 dis_coord_shifted_scaled_small 上 找valid區域(-ord_ratio~ord_ratio) 當 "Fm"， 並對應回原始move_map '''
    fm, bm, fm_nan_mask = step7c_Before_Dis_coord_valid_area_is_Fm_and_inverse_backto_Ord_to_get_fm_value(dis_coord_shifted_scaled_small_m,
                                ord_base=ord_base_before, see_base=see_base_before, img_w=img_w, img_h=img_h,
                                debug=debug_before, ord_ratio=ord_ratio)
    fm = fm / ord_ratio  ### 放大回 -1~1 才會正確對應
    bm = bm / ord_ratio  ### 放大回 -1~1 才會正確對應
    # plt.figure()
    # plt.imshow(fm_nan_mask)

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    ### 待處理 fm, bm 還要後處理一下 加一個mask channel 進去 就是 blender 的形式囉！
    ### 處理中 step7 整理一下 dis_coord, ord_coord, start_coord...
    ### 待處理 savefig 可以寫一下
    ### 待處理 step8 增加resolution的方法
    ### 待處理 step7 mask 應該要用 pytorch gridsample 執行完 拿到的mask 才最準確
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    ''' step9 apply_fm/bm'''
    dis_img = apply_fm_to_get_dis_img(ord_img, new_fm, visual=debug_papr17, before_title="Paper17_before_fm", after_title="Paper17_after_fm")
    rec_img = apply_bm_to_get_rec_img(dis_img, new_bm, visual=debug_papr17, before_title="Paper17_before_bm", after_title="Paper17_after_bm")
    dis_img = apply_fm_to_get_dis_img(ord_img, fm, visual=debug_before, before_title="Before_before_fm", after_title="Before_after_fm")
    rec_img = apply_bm_to_get_rec_img(dis_img, bm, visual=debug_before, before_title="Before_before_bm", after_title="Before_after_bm")

    ########################################################################################################
    plt.show()

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
    # move_f, _ = get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=vert_x, vert_y=vert_y, move_x= 0.0, move_y= -0.116832, dis_type="curl", alpha= 2.0, debug=True )  ### alpha:2~4
    #############################################################################################################################################
    # row = 30
    # col = 30
    # h_res = row
    # w_res = col
    # x_min = 0
    # x_max = w_res - 1
    # y_min = 0
    # y_max = h_res - 1
    # move_f, _ =          get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=col / 2, vert_y=row / 2, move_x=  -10, move_y=  10, dis_type="fold", alpha=  30, debug=True )  ### alpha:2~4
    # move_f, _ = move_f + get_dis_move_map(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=      0, vert_y=     10, move_x= 3.5, move_y= 2.5, dis_type="fold", alpha=200, debug=True )

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
