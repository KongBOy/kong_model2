import sys
sys.path.append("kong_util")

import numpy as np
import matplotlib.pyplot as plt
from util import get_xy_f_and_m
from matplot_fig_ax_util import check_fig_ax_init, mesh3D_scatter_and_z0_plane, move_map_2D_moving_visual
from matplot_fig_ax_util import coord_f_2D_scatter, move_map_1D_value, move_map_2D_arrow, move_map_3D_scatter


from step2_a_util                                                         import wt_calculate, step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord, apply_fm_to_get_dis_img, apply_bm_to_get_rec_img
from step2_a_debug_step4_wt_simulate                                      import step4_wt_simulate_visual
from step2_a_debug_step5_move_map_simulate                                import step5_move_map_simulate_visual
from step2_a_debug_step6a_dis_coord_adjust_visual                         import step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual
from step2_a_debug_step6b_papr17_dis_coord_big_and_ord_valid_coord_visual import step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual
from step2_a_debug_step6c_before_dis_coord_small_visual                   import step6_debug_c_Dis_coord_small_Move_map_Boundary_visual
from step2_a_debug_step7b_papr17_get_bm_value_visual                      import step7_visual_util_a_paper17
from step2_a_debug_step7c_before_get_fm_value_visual                      import step7_visual_util_b_before

debug_spyder_dict = dict()


''' step1~5 '''
### 整個function都是用 image的方式來看(左上角(0,0)，x往右邊走增加，y往上面走增加)
def get_dis_move_map(xy_m, vert_x, vert_y, move_x, move_y, dis_type="fold", alpha=50, debug=False):
    h_res, w_res = xy_m.shape[:2]
    xy_f = xy_m.reshape(-1, 2)
    '''
    xy_m    ： 正mesh，值域為 0~x_max, 0~y_max , shape 為 (h_res , w_res, 2), m 是 map_form 的意思， xy_f[..., 0]是X座標， xy_f[..., 1]是Y座標
    xy_f    ： 正mesh，值域為 0~x_max, 0~y_max , shape 為 (h_res * w_res, 2), f 是 flatten  的意思， xy_f[ : , 0]是X座標， xy_f[ : , 1]是Y座標
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
    xy_shifted_f  = xy_f - vtex        ### 位移整張mesh變成以扭曲點座標為原點

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
        fig, ax, ax_c = move_map_2D_moving_visual(move_map_m=move_map_m, start_xy_m=xy_m, fig_title=f"step5. {dis_type}", fig=fig, ax=ax, ax_c=ax_c)

        '''simulate wt_fold/curl 兩種都模擬看看長怎樣'''
        step5_move_map_simulate_visual(xy_m, move_xy, d_abs_norm_m, alpha_fold=0.8, alpha_curl=2.0)

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
                                                            boundary_value=1,
                                                            before_alpha=0.5,
                                                            after_alpha =0.7):
    '''
    共同都會對 dis_coord_m 做 "位移" 讓 dis_coord_m appearance 的正中心 在原點(0, 0)， 之後會再針對 adjust_type 來做 "縮放"
    如何調整move_map 的話是先 用正mesh 根據 move_map_m 移動後 得到的 dis_coord_m， 藉由調整 dis_coord 來 調整 move_map
    最後是return dis_coord_m， 如果想使用move_map_m 就使用 dis_coord - start_xy_m = move_map_m 囉！

    start_xy_m  ： 起始座標點         ， 用來計算 dis_coord_m 用的
    move_map_m  ： 移動的箭頭(不含座標)， 用來計算 dis_coord_m 用的
    adjust_type ：
        big   ： paper17的方法 ，把 dis_coord_m 值放大(超過 自訂的boundary)， 在 dis_coord_m 上取新的bm， 取 valid_ord_coord 當fm
        small ： 以前的方法， 原本 dis_coord_m == move_map + ord_coord 直接當bm， 把 dis_coord_m 值縮小(限制在 自訂的boundary 內)， 縮小的dis_coord_m 的 appearance 就當fm， 之後會再把 fm 對應回 ord_coord 這樣子
    debug ： 建議要搭配 spyder 一起使用， 才能用 Variable Explore 看變數喔！
    '''
    if(debug): fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=2, fig_cols=5, ax_size=5, tight_layout=True)
    ##########################################################################################################################
    h_res, w_res = start_xy_m.shape[: 2]

    '''算 step6a. start_xy_m 根據 move_map_m 移動， 就會得到 dis_coord_m'''
    dis_coord_m = start_xy_m + move_map_m  ### dis_coord
    if(debug):
        ##########################################################################################################################
        '''圖0,0 2D箭頭視覺化 bm裡存的東西'''
        fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
            fig_title="step6 2D view \n storage array is still bm",
            jump_r=jump_r, jump_c=jump_c,
            arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
            show_before_move_coord=True,  before_alpha=before_alpha,
            show_after_move_coord =False, after_alpha=after_alpha,
            fig=fig, ax=ax[0], ax_c=0)

        '''圖0,1 2D箭頭視覺化 bm裡存的東西 模擬移動後的結果'''
        fig, ax[0], ax_c = move_map_2D_arrow(move_map_m, start_xy_m=start_xy_m,
            fig_title="step6 2D view \n simulate bm move will get dis coord",
            arrow_C=None, arrow_cmap="gray",  ### arrow_C 為None 時 預設為 np.zeros(shape=(h_res, w_res))
            jump_r=jump_r, jump_c=jump_c,
            show_before_move_coord=True,  before_alpha=before_alpha,
            show_after_move_coord =True, after_alpha=after_alpha,
            fig=fig, ax=ax[0], ax_c=1)

        '''圖1,1 3D scatter 分不同平面 比較好看 bm裡存的東西 模擬移動後的結果 '''
        fig, ax[1], ax_c, _ = move_map_3D_scatter(move_map_m, start_xy_m=start_xy_m,
            fig_title="step6 3D view\n up: dis_coord appearance is fm,\n bottom: appearance is bm",
            zticklabels=("bm", "", "", "fm"),
            jump_r=jump_r, jump_c=jump_c,
            before_C="orange", before_alpha=before_alpha, before_height=0,
            after_C = "blue",  after_alpha=after_alpha,   after_height =0.6,
            fig=fig, ax=ax[1], ax_c=1, ax_r=1, ax_rows=2)

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
    '''算 step6b. dis_coord_m 做些 平移， 得到 dis_coord_shifted_m， 如果要用debug視覺化 記得回推一下 move_map_shifted_m 喔！'''
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
    '''算 step6c. dis_coord_shifted_m 做些 縮放， 如果要用debug視覺化 記得回推一下 move_map_shifted_scaled_m 喔！'''
    ### adjust_type == "big" 是paper17的方法
    # if(adjust_type == "big"):
    #     dis_coord_shifted_x_m = dis_coord_shifted_m[..., 0]
    #     dis_coord_shifted_y_m = dis_coord_shifted_m[..., 1]
    #     dis_coord_shifted_xmin = dis_coord_shifted_x_m.min()
    #     dis_coord_shifted_xmax = dis_coord_shifted_x_m.max()
    #     dis_coord_shifted_ymin = dis_coord_shifted_y_m.min()
    #     dis_coord_shifted_ymax = dis_coord_shifted_y_m.max()
    #     raito = max(abs(dis_coord_shifted_xmin), abs(dis_coord_shifted_xmax), abs(dis_coord_shifted_ymin), abs(dis_coord_shifted_ymax))
    #     ### big， 自己要知道 要超過 see_coord_m
    #     dis_coord_shifted_scaled_m = dis_coord_shifted_m.copy()
    #     dis_coord_shifted_scaled_m *= raito * 1.2

    # ### adjust_type == "small" 是before的方法
    # elif(adjust_type == "small"):
    #     ### small， 自己要知道 不超過 see_coord_m
    #     dis_coord_shifted_scaled_m =  dis_coord_shifted_m.copy()
    #     dis_coord_shifted_scaled_m *= 0.8
    dis_coord_shifted_scaled_m =  dis_coord_shifted_m.copy()
    dis_coord_shifted_scaled_m *= adjust_ratio

    if(debug):
        '''圖0:1,4 3D scatter 分不同平面 原始dis_coord_scaled_shifted_big '''
        debug_spyder_dict["step6c. raito"] = adjust_ratio
        move_map_shifted_scaled_m   = dis_coord_shifted_scaled_m   - start_xy_m
        fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_shifted_scaled_m, move_map_shifted_scaled_m, start_xy_m,
                                                                            current_state="dis_coord_shifted_scaled_big",
                                                                            jump_r=jump_r, jump_c=jump_c,
                                                                            boundary_value=boundary_value,
                                                                            before_alpha=before_alpha,
                                                                            after_alpha =after_alpha,
                                                                            fig=fig, ax=ax, ax_c=4)

    if(debug):
        ### adjust_type == "big" 是paper17的方法
        if(adjust_type == "big"):
            debug_spyder_dict[f"step6c. dis_coord_shifted_scaled_big_m"]   = dis_coord_shifted_scaled_m

            # move_map_shifted_scaled_big_m   = dis_coord_shifted_scaled_m   - start_xy_m
            # '''圖0:1,4 3D scatter 分不同平面 原始dis_coord_scaled_shifted_big '''
            # fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_shifted_scaled_m, move_map_shifted_scaled_big_m, start_xy_m,
            #                                                                     current_state="dis_coord_shifted_scaled_big",
            #                                                                     jump_r=jump_r, jump_c=jump_c,
            #                                                                     boundary_value=boundary_value,
            #                                                                     before_alpha=before_alpha,
            #                                                                     after_alpha =after_alpha,
            #                                                                     fig=fig, ax=ax, ax_c=4)
            step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual(dis_coord_shifted_scaled_m,  move_map_shifted_scaled_m, start_xy_m,
                                                                                boundary_value=1.00,
                                                                                jump_r=4, jump_c=4,
                                                                                before_alpha=before_alpha,
                                                                                after_alpha =after_alpha,
                                                                                fig=None, ax=None, ax_c=None)

        ### adjust_type == "small" 是before的方法
        elif(adjust_type == "small"):
            debug_spyder_dict[f"step6c. dis_coord_shifted_scaled_small_m"]   = dis_coord_shifted_scaled_m

            # move_map_shifted_scaled_small_m   = dis_coord_shifted_scaled_m   - start_xy_m
            # '''圖0:1,4 3D scatter 分不同平面 原始dis_coord_shifted_scaled_small '''
            # fig, ax, ax_c = step6_debug_a_Dis_coord_both_Move_map_Boundary_2D3Dvisual(dis_coord_shifted_scaled_m, move_map_shifted_scaled_small_m, start_xy_m,
            #                                                                          current_state="dis_coord_shifted_scaled_small",
            #                                                                          jump_r=jump_r, jump_c=jump_c,
            #                                                                          boundary_value=boundary_value,
            #                                                                          before_alpha=before_alpha,
            #                                                                          after_alpha =after_alpha,
            #                                                                          fig=fig, ax=ax, ax_c=4)

            ### 以前的版本 本身 就不適合 套用 step6_debug_b_Dis_coord_big_Move_map_Boundary_Ord_valid_coord_visual 因為不需要 在 dis_coord_shifted_scaled_m 上取 新bm 和 找 ord_valid_coord， 而是直接用 dis_coord_shifted_scaled_m(縮小的)的appearance 當fm， 之後會再把 fm 對應回 ord_coord 這樣子， 所以外面 如果 adjust_type==small時 debug 記得設定false
            step6_debug_c_Dis_coord_small_Move_map_Boundary_visual(dis_coord_shifted_scaled_m, move_map_shifted_scaled_m, start_xy_m,
                                                                  jump_r=jump_r, jump_c=jump_c,
                                                                  boundary_value=boundary_value,
                                                                  before_alpha=before_alpha,
                                                                  after_alpha =after_alpha,
                                                                  fig=None, ax=None, ax_c=None)

    #######################################################################################################################
    return dis_coord_shifted_scaled_m


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
''' step7 繼續編號寫下去 '''
def step7a_dst_backto_ord_and_see_where(dst_coord_m, ord_ratio, ord_coord_m, see_coord_m):
    see_h_res, see_w_res = see_coord_m.shape[:2]
    ord_coord_f = dst_coord_m.reshape(-1, 2)  ### dis
    dst_coord_f = ord_coord_m.reshape(-1, 2)  ### 0.95
    see_coord_f = see_coord_m.reshape(-1, 2)  ### 1.00
    ##################################################################################################################
    import scipy.interpolate as spin
    see_inv_coord_f = spin.griddata(ord_coord_f / ord_ratio, dst_coord_f, see_coord_f, method="linear")  ### 計算， dst 為 dis_coord_big 的話 see_inv_coord_f 填滿滿， dst 為 dis_coord_small 的話 see_inv_coord_f 會有 nan
    see_inv_coord_m = see_inv_coord_f.reshape(see_h_res, see_w_res, 2)                                   ### flatten 轉 map
    see_inv_move_map_m = see_inv_coord_m - see_coord_m    ### 計算 move_map = dst - start， 如果有nan 減完 仍為 nan
    return see_inv_coord_m, see_inv_move_map_m            ### dst 為 dis_coord_big 的話 see_inv_coord_f 填滿滿， dst 為 dis_coord_small 的話 see_inv_coord_f 會有 nan


''' dis_coord_big_m '''
def step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_big_m, ord_ratio, ord_base, see_base, img_w, img_h, start_xy_m, debug=False):
    '''
    因為是在 dis_coord_big_m 抓 see_coord_m 一定會填滿(step6本來就要設定放大要超過 boundary(see_coord的範圍))， 所以 inv_see_coord_m 一定填得滿滿的 不會有nan

    start_xy_m： 視覺化用的
    valid area(即boundary 、 see_coord的範圍 )： 因為是取 new bm 並反推 new fm， 所以new bm 的範圍可以取我們好處理的範圍， 我們現在是取跟 pytorch grid_sample() 一樣的 -1 ~ +1 的範圍是valid
    '''
    ##################################################################################################################
    ord_valid_mask = step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord(dis_coord_big_m, boundary_value=1.0, visual=False)
    debug_spyder_dict["step7. dis_coord_big_m"] = dis_coord_big_m

    h_res, w_res = dis_coord_big_m.shape[:2]
    # img_w = dis_coord_big_m.shape[1]  ### debug用 先調成跟 一開始 mesh res 一樣大
    # img_h = dis_coord_big_m.shape[0]  ### debug用 先調成跟 一開始 mesh res 一樣大
    # _, see_bm_xy_1_00_m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=img_w, h_res=img_h)
    # _, see_bm_xy_0_95_m = get_xy_f_and_m(x_min=-0.95, x_max=+0.95, y_min=-0.95, y_max=+0.95, w_res=img_w, h_res=img_h)

    # _, ord_xy_1_00m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=w_res, h_res=h_res)
    # _, ord_xy_0_95 = get_xy_f_and_m(x_min=-0.95, x_max=+0.95, y_min=-0.95, y_max=+0.95, w_res=w_res, h_res=h_res)

    _, ord_xy_m = get_xy_f_and_m(x_min=-ord_base, x_max=+ord_base, y_min=-ord_base, y_max=+ord_base, w_res=w_res, h_res=h_res)  ### w_res, h_res
    _, see_xy_m = get_xy_f_and_m(x_min=-see_base, x_max=+see_base, y_min=-see_base, y_max=+see_base, w_res=img_w, h_res=img_h)  ### img_w, img_h
    ##################################################################################################################
    see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(dst_coord_m=dis_coord_big_m, ord_ratio=ord_ratio, ord_coord_m=ord_xy_m,     see_coord_m=see_xy_m)
    if(debug):
       step7_visual_util_a_paper17(see_inv_coord_m, see_inv_move_map_m, dst_coord_m=dis_coord_big_m, ord_ratio=1.00, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m, xy_m=start_xy_m, ord_valid_mask=ord_valid_mask)
    ##################################################################################################################
    ### 0.95 + move_map -> dis_coord -> back to 0.95, boundary grab 1.00， 因為這 boundary 是新的bm！ 可以自己取，取完以後要回去找 相對應 符合移動後在boundary 內的 ord 即可！
    ### 但是前面的這個 0.95 + move_map -> dis_coord -> back to 0.95 必須要對應到， 後面取的 boundary 裡面的值 才正確！
    ''' 竟然對 方法1 錯的： 直接 對應回 -1~1(boundary) ， not match'''
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(dst_coord_m=dis_coord_big_m, ord_ratio=ord_ratio, ord_coord_m=ord_xy_m,     see_coord_m=see_xy_m)
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(dst_coord_m=dis_coord_big_m, ord_ratio=1.00, ord_coord_m=ord_xy_1_00m, see_coord_m=see_bm_xy_1_00_m)
    # new_bm = see_inv_coord_m
    # new_fm = dis_coord_big_m
    # if(debug):
    #    step7_visual_util_a_paper17(see_inv_coord_m, see_inv_move_map_m, dst_coord_m=dis_coord_big_m, ord_ratio=1.00, ord_coord_m=ord_xy_1_00m, see_coord_m=see_bm_xy_1_00_m, xy_m=start_xy_m, ord_valid_mask=ord_valid_mask)

    ##################################################################################################################
    ''' 錯 方法2(paper17) ：把自己除0.95 放大一點 變成boundary的大小 再 對應回 -1~1(boundary) '''
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(dst_coord_m=dis_coord_big_m, ord_ratio=ord_ratio, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m)
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(dst_coord_m=dis_coord_big_m, ord_ratio=0.95, ord_coord_m=ord_xy_1_00m, see_coord_m=see_bm_xy_1_00_m)
    # new_bm = see_inv_coord_m
    # new_fm = dis_coord_big_m  / ord_ratio  ### 方法2 改， 把 fm也放大就對了
    # if(debug):
    #   step7_visual_util_a_paper17(see_inv_coord_m, see_inv_move_map_m, dst_coord_m=dis_coord_big_m, ord_ratio=0.95, ord_coord_m=ord_xy_1_00m, see_coord_m=see_bm_xy_1_00_m, xy_m=start_xy_m, ord_valid_mask=ord_valid_mask)

    ##################################################################################################################
    ''' 錯，也許是see 改0.95才會對嗎? (try方法4) 方法3： 直接 對應回 -0.95~0.95， 理論上來說應該是要這樣子， 因為我是從 -0.95~0.95走道 dis_coord， dis_coord應該要走回-0.95~+0.95， 實際上測試也確實如此 '''
    ### 但不大對，因為我的 valid area/boundary 是設定 -1~1
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(  dst_coord_m=dis_coord_big_m, ord_ratio=ord_ratio, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m)
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(  dst_coord_m=dis_coord_big_m, ord_ratio=1.00, ord_coord_m=ord_xy_0_95, see_coord_m=see_bm_xy_1_00_m)
    # new_bm = see_inv_coord_m
    # new_fm = dis_coord_big_m
    ##################################################################################################################
    ''' 方法4： 方法3 把 see 改0.95， 最後把 fm, bm 從 0.95 放大回 1.00 就對了 '''
    ### 但不大對，因為我的 valid area/boundary 是設定 -1~1
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(  dst_coord_m=dis_coord_big_m, ord_ratio=ord_ratio, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m)
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(  dst_coord_m=dis_coord_big_m, ord_ratio=1.00, ord_coord_m=ord_xy_0_95, see_coord_m=see_bm_xy_0_95_m)
    # new_bm = see_inv_coord_m / ord_ratio
    # new_fm = dis_coord_big_m / ord_ratio
    if(debug):
        step7_visual_util_a_paper17(see_inv_coord_m, see_inv_move_map_m, dst_coord_m=dis_coord_big_m, ord_ratio=1.00, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m, xy_m=start_xy_m, ord_valid_mask=ord_valid_mask)
        debug_spyder_dict["step7 ord_valid_mask"] = ord_valid_mask
        debug_spyder_dict["step7 dst_coord_m"] = dis_coord_big_m
        debug_spyder_dict["step7 ord_coord_m"] = ord_xy_m
        debug_spyder_dict["step7 see_inv_coord_m"] = see_inv_coord_m
        debug_spyder_dict["step7 see_inv_move_map_m"] = see_inv_move_map_m
        debug_spyder_dict["step7 see_inv_move_map_m.isnan()"] = np.isnan(see_inv_move_map_m)
        debug_spyder_dict["step7 xy_m"] = xy_m
    ##################################################################################################################
    new_bm = see_inv_coord_m
    new_fm = dis_coord_big_m  ### * ord_valid_mask[:, :, np.newaxis]  ### 就算不用mask遮住， pytorch 的 gridsample 還是可以運作喔！
    return new_bm, new_fm, ord_valid_mask


##################################################################################################################
##################################################################################################################
''' dis_coord_small_m '''
def step7c_Before_Dis_coord_valid_area_is_Fm_and_inverse_backto_Ord_to_get_fm_value(dis_coord_small_m, ord_ratio, ord_base, see_base, img_w, img_h, start_xy_m, debug=False):
    '''
    因為是在 dis_coord_small_m 抓 see_coord_m 一定 不會填滿(step6本來就要設定縮小 要在 boundary(see_coord的範圍) 裡面)， 所以 inv_see_coord_m 一定填不滿， 會有nan

    valid area(即boundary 、 see_coord的範圍 )： 注意因為不是取new bm， 所以範圍不一定
    '''
    ##################################################################################################################
    debug_spyder_dict["step7. dis_coord_smallm"] = dis_coord_small_m

    h_res, w_res = dis_coord_small_m.shape[:2]
    # fm_w_res  = dis_coord_small_m.shape[1]  ### debug用 先調成跟 一開始 mesh res 一樣大
    # fm_h_res  = dis_coord_small_m.shape[0]  ### debug用 先調成跟 一開始 mesh res 一樣大
    ### 之前沒有想到 before 應該要用 dis_coord_small 還在 擴大 see 的範圍
    # _, see_fm_xy_1_30m = get_xy_f_and_m(x_min=-1.30, x_max=+1.30, y_min=-1.30, y_max=+1.30, w_res=img_w, h_res=img_h)
    # _, see_fm_xy_1_23m = get_xy_f_and_m(x_min=-1.23, x_max=+1.23, y_min=-1.23, y_max=+1.23, w_res=img_w, h_res=img_h)
    # _, see_fm_xy_1_20m = get_xy_f_and_m(x_min=-1.20, x_max=+1.20, y_min=-1.20, y_max=+1.20, w_res=img_w, h_res=img_h)

    ### 應該是錯的， 但細想下去好像沒錯， 只是在 轉換成真實座標時 要分 bm(-0.95~0.95) 和 fm(-1.00~1.00) 兩種方式轉換
    # _, see_fm_xy_1_00m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=img_w, h_res=img_h)
    ### 應該是對的， 轉換成真實座標時 bm/fm 可以統一用 -0.95~0.95
    # _, see_fm_xy_0_95_m = get_xy_f_and_m(x_min=-0.95, x_max=+0.95, y_min=-0.95, y_max=+0.95, w_res=img_w, h_res=img_h)

    # _, ord_xy_1_00_m = get_xy_f_and_m(x_min=-1.00, x_max=+1.00, y_min=-1.00, y_max=+1.00, w_res=w_res, h_res=h_res)
    # _, ord_xy_0_95_m = get_xy_f_and_m(x_min=-0.95, x_max=+0.95, y_min=-0.95, y_max=+0.95, w_res=w_res, h_res=h_res)

    _, ord_xy_m = get_xy_f_and_m(x_min=-ord_base, x_max=+ord_base, y_min=-ord_base, y_max=+ord_base, w_res=w_res, h_res=h_res)  ### w_res, h_res
    _, see_xy_m = get_xy_f_and_m(x_min=-see_base, x_max=+see_base, y_min=-see_base, y_max=+see_base, w_res=img_w, h_res=img_h)  ### img_w, img_h
    ##################################################################################################################
    ''' 方法1 ： see 取 -1~1 是錯的'''
    ### 0.95 + move_map -> dis_coord -> back to 0.95, boundary grab 0.95， 因為沒有取新bm， 是直接 回去原本的地方， 原本的地方如果是 0.95， 回去也是0.95， boundary 也是原本的 0.95 囉！
    # see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(         dst_coord_m=dis_coord_small_m, ord_ratio=1.00, ord_coord_m=ord_xy_0_95_m, see_coord_m=see_fm_xy_1_00m)
    # fm_nan_mask = 1 - np.isnan(see_inv_coord_m).astype(np.int32)[..., 0]
    # fm = see_inv_coord_m
    # bm = dis_coord_small_m
    # if(debug):
    #     step7_visual_util_b_before(see_inv_coord_m, see_inv_move_map_m, fm_nan_mask=fm_nan_mask, dst_coord_m=dis_coord_small_m, ord_ratio=1.00, ord_coord_m=ord_xy_0_95_m, see_coord_m=see_fm_xy_1_00m, xy_m=start_xy_m)

    ''' 方法2 ： see 取 -0.95~0.95應該才是對的， 最後把 fm, bm 從 0.95 放大回 1.00 '''
    ### 仔細思考這才對， 因為 ord_coord 在 放回原始img_array 是用0.95 當基準來做的， see_coord 如果用1.00當基準來做就不匹配(我猜 fm appearance 用 see用1.0 面積會縮小)， 應該要跟ord_coord用一樣的0.95才對
    see_inv_coord_m, see_inv_move_map_m = step7a_dst_backto_ord_and_see_where(         dst_coord_m=dis_coord_small_m, ord_ratio=ord_ratio, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m)
    fm_nan_mask = 1 - np.isnan(see_inv_coord_m).astype(np.int32)[..., 0]
    # fm = see_inv_coord_m   / ord_ratio    ### 放大回 -1~1 才會正確對應
    # bm = dis_coord_small_m / ord_ratio  ### 放大回 -1~1 才會正確對應
    if(debug):
        debug_spyder_dict["step7 fm_nan_mask"] = fm_nan_mask
        # see_coord_m = fill_nan(fm_nan_mask, see_coord_m)
        # debug_spyder_dict["step7 nan_mask_see_coord_m"] = see_coord_m
        step7_visual_util_b_before(see_inv_coord_m, see_inv_move_map_m, fm_nan_mask=fm_nan_mask, dst_coord_m=dis_coord_small_m, ord_ratio=1.00, ord_coord_m=ord_xy_m, see_coord_m=see_xy_m, xy_m=start_xy_m)
    ##################################################################################################################
    fm = see_inv_coord_m
    bm = dis_coord_small_m
    return fm, bm, fm_nan_mask

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
    ord_img = cv2.resize(ord_img, (256, 256), interpolation=cv2.INTER_AREA)
    img_h, img_w = ord_img.shape[:2]

    ### 理解用，手動慢慢扭曲
    '''
    x/y_min/max：
        是用 np.linspace 喔！ x_min ~ x_max 就是真的到那個數字！ 不像 np.arange() 會是 x_min ~ x_max-1！
        所以如果要還原以前寫的東西 要記得 x_max-1 喔！
    w/h_res ： min ~ max 之間切多少格
    '''

    ### 印度那篇 move_map模擬成功 繼續往下模擬
    h_res    = 129  ### 77
    w_res    = 129  ### 77
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
    xy_f, xy_m = get_xy_f_and_m(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思

    #########################################################################
    debug_1to5 = False
    debug_papr17 = False
    debug_before = False
    '''step1~5'''
    _, move_map_curl_m = get_dis_move_map(xy_m, vert_x=vert_x, vert_y=vert_y, move_x=move_x, move_y=move_y, dis_type=dis_type, alpha=alpha, debug=debug_1to5)  ### alpha:2~4

    ##################################################################################################################
    '''step6 dis_coord_m = move_map_m + start_xy_m， 調整 dis_coord_m 變成 big 版本'''
    dis_coord_shifted_scaled_big_m   = move_map_value_adjust_by_dis_coord_and_return_dis_coord(adjust_type="big", adjust_ratio=1.5,  move_map_m=move_map_curl_m, start_xy_m=xy_m, boundary_value=1.00     , debug=debug_papr17)
    ##################################################################################################################
    # ''' step7 在做完 縮放 後 的 dis_coord_shifted_scaled_big 上 找valid區域(-1.00~1.00) 當 "Bm"， 並對應回原始move_map '''
    # ''' 竟然對 方法1 錯的： 直接 對應回 -1~1(boundary) ， not match'''
    # new_bm, new_fm, ord_valid_mask   = step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_shifted_scaled_big_m,
    #                                 ord_ratio=1.00, ord_base=1.00, see_base=1.00, img_w=img_w, img_h=img_h, start_xy_m=xy_m, debug=debug_papr17)
    ##################################################################################################################
    # ''' 錯 方法2(paper17) ：把自己除0.95 放大一點 變成boundary的大小 再 對應回 -1~1(boundary) '''
    # new_bm, new_fm, ord_valid_mask   = step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_shifted_scaled_big_m,
    #                                 ord_ratio=ord_ratio, ord_base=1.00, see_base=1.00, img_w=img_w, img_h=img_h, start_xy_m=xy_m, debug=debug_papr17)
    # new_fm = new_fm / ord_ratio  ### 方法2 改， 把 fm也放大就對了
    ##################################################################################################################
    # ''' 錯，也許是see 改0.95才會對嗎? (try方法4) 方法3： 直接 對應回 -0.95~0.95， 理論上來說應該是要這樣子， 因為我是從 -0.95~0.95走道 dis_coord， dis_coord應該要走回-0.95~+0.95， 實際上測試也確實如此 '''
    # new_bm, new_fm, ord_valid_mask   = step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_shifted_scaled_big_m,
    #                                 ord_ratio=1.00, ord_base=ord_ratio, see_base=1.00, img_w=img_w, img_h=img_h, start_xy_m=xy_m, debug=debug_papr17)
    ##################################################################################################################
    ''' 方法4： 方法3 把 see 改0.95， 最後把 fm, bm 從 0.95 放大回 1.00 就對了 '''
    ### 但不大對，因為我的 valid area/boundary 是設定 -1~1
    new_bm, new_fm, ord_valid_mask   = step7b_Paper17_Dis_coord_valid_area_is_new_Bm_and_inverse_backto_Ord_valid_coord_to_get_bm_value(dis_coord_shifted_scaled_big_m,
                                    ord_ratio=1.00, ord_base=ord_ratio, see_base=ord_ratio, img_w=img_w, img_h=img_h, start_xy_m=xy_m, debug=debug_papr17)
    new_bm = new_bm / ord_ratio
    new_fm = new_fm / ord_ratio

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    '''step6 dis_coord_m = move_map_m + start_xy_m， 調整 dis_coord_m 變成 small 版本'''
    dis_coord_shifted_scaled_small_m = move_map_value_adjust_by_dis_coord_and_return_dis_coord(adjust_type="small", adjust_ratio=0.8, move_map_m=move_map_curl_m, start_xy_m=xy_m, boundary_value=ord_ratio, debug=debug_before)  ### 以前的版本 本身 就不適合 套用 step6_util， 因為不需要 在 dis_coord_shifted_scaled_m 上取 新bm 和 找 ord_valid_coord， 而是直接用 dis_coord_shifted_scaled_m(縮小的)的appearance 當fm， 之後會再把 fm 對應回 ord_coord 這樣子， 所以外面 如果 adjust_type==small時 debug 記得設定false
    ##################################################################################################################
    ''' step7 在做完 縮放 後 的 dis_coord_shifted_scaled_small 上 找valid區域(-ord_ratio~ord_ratio) 當 "Fm"， 並對應回原始move_map '''
    ''' 方法1 ： see 取 -1~1 是錯的'''
    ### 0.95 + move_map -> dis_coord -> back to 0.95, boundary grab 0.95， 因為沒有取新bm， 是直接 回去原本的地方， 原本的地方如果是 0.95， 回去也是0.95， boundary 也是原本的 0.95 囉！
    # fm, bm, fm_nan_mask = step7c_Before_Dis_coord_valid_area_is_Fm_and_inverse_backto_Ord_to_get_fm_value(dis_coord_shifted_scaled_small_m,
    #                             ord_ratio=1.00, ord_base=ord_ratio, see_base=1.00, img_w=img_w, img_h=img_h,
    #                             start_xy_m=xy_m, debug=debug_before)

    ''' 方法2 ： see 取 -0.95~0.95應該才是對的， 最後把 fm, bm 從 0.95 放大回 1.00 '''
    ### 仔細思考這才對， 因為 ord_coord 在 放回原始img_array 是用0.95 當基準來做的， see_coord 如果用1.00當基準來做就不匹配(我猜 fm appearance 用 see用1.0 面積會縮小)， 應該要跟ord_coord用一樣的0.95才對
    fm, bm, fm_nan_mask = step7c_Before_Dis_coord_valid_area_is_Fm_and_inverse_backto_Ord_to_get_fm_value(dis_coord_shifted_scaled_small_m,
                                ord_ratio=1.00, ord_base=ord_ratio, see_base=ord_ratio, img_w=img_w, img_h=img_h,
                                start_xy_m=xy_m, debug=debug_before)
    fm = fm / ord_ratio  ### 放大回 -1~1 才會正確對應
    bm = bm / ord_ratio  ### 放大回 -1~1 才會正確對應




    ### fm, bm 還要後處理一下 加一個mask channel 進去 就是 blender 的形式囉！
    ### step7 整理一下 dis_coord, ord_coord, start_coord...
    ### 測試 paper17/before 覺得錯的case是不是真的錯
    ### savefig 可以寫一下
    ''' step8 apply_fm/bm'''
    ########################################################################################################
    dis_img, fig, ax, ax_i = apply_fm_to_get_dis_img(ord_img, new_fm, visual=True, before_title="Paper17_before_fm", after_title="Paper17_after_fm")
    rec_img, fig, ax, ax_i = apply_bm_to_get_rec_img(dis_img, new_bm, visual=True, before_title="Paper17_before_bm", after_title="Paper17_after_bm")
    dis_img, fig, ax, ax_i = apply_fm_to_get_dis_img(ord_img, fm, visual=True, before_title="Before_before_fm", after_title="Before_after_fm")
    rec_img, fig, ax, ax_i = apply_bm_to_get_rec_img(dis_img, bm, visual=True, before_title="Before_before_bm", after_title="Before_after_bm")

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
