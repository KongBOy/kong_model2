import sys
sys.path.append("kong_util")

from step0_access_path import data_access_path
import numpy as np
import matplotlib.pyplot as plt
from build_dataset_combine import Check_dir_exist_and_build
from util import time_util
from matplot_fig_ax_3D_util import draw_3D_xy_plane_by_mesh_f
import time

debug_spyder_dict = dict()
### 參考連結：https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
def get_xy_f(row, col):  ### get_xy_flatten，拿到的map的shape：(..., 2)
    x = np.arange(col)
    x = np.tile(x, (row, 1))

#    y = np.arange(row-1, -1, -1) ### 就是這裡要改一下拉！不要抄網路的，網路的是用scatter的方式來看(左下角(0,0)，x往右增加，y往上增加)
    y = np.arange(row)  ### 改成這樣子 就是用image的方式來處理囉！(左上角(0,0)，x往右增加，y往上增加)
    y = np.tile(y, (col, 1)).T

    x_f = x.flatten()
    y_f = y.flatten()
    xy_f = np.array( [x_f, y_f], dtype=np.float64 )
    return xy_f.T

### 整個function都是用 image的方式來看(左上角(0,0)，x往右邊走增加，y往上面走增加)
def distort(row, col, vert_x, vert_y, move_x, move_y, dis_type="fold", alpha=50, debug=False):
    if(debug):
        fig_amount = 14
        fig, ax = plt.subplots(nrows=1, ncols=fig_amount)
        fig.set_size_inches(5 * fig_amount, 5)
        fig.tight_layout()
        ax_i = 0


    ''' step1. 建立xy正mesh， 並 在正mesh中 選一個點當扭曲點
    xy_f         ： 正mesh                         ， 值域為 0~row, 0~col     , shape 為 (row * col, 2), f 是 flatten 的意思， xy_f[..., 0]是X座標， xy_f[..., 1]是Y座標
    vtex         ： 正mesh 中 的 一個座標           ， 值域在 0~row, 0~col 之間, shape 為 (1, 2)
    xy_f_shifted ： 位移到 以 vtex 為(0, 0) 的正mesh
    xyz_f_shifted： 位移到 以 vtex 為(0, 0) 的正mesh 在加上一維z， z全填0
    '''
    xy_f = get_xy_f(row, col)  ### 拿到map的shape：(..., 2), f 是 flatten 的意思
    vtex = np.array([vert_x, vert_y])  ### 指定的扭曲點xy座標

    ### 位移整張mesh變成以扭曲點座標為原點
    xy_f_shifted  = xy_f - vtex
    ### 準備等等要算 d的材料：xyz_f_shifted 多加一個channel z，填0
    xyz_f_shifted = np.zeros(shape=(row * col, 3))
    xyz_f_shifted[:, 0:2] = xy_f_shifted

    '''視覺化 xy 和 xy_shift 和 move_xy'''
    if(debug):
        ### xy 視覺化
        ax[ax_i].set_title("step1.xy_mesh, step2. move_xy")
        ax[ax_i].scatter(xy_f[:, 0], xy_f[:, 1], c = np.arange(row * col), s=1, cmap="hsv")
        ax[ax_i].spines['right'].set_color('None')
        ax[ax_i].spines['top']  .set_color('None')
        ax[ax_i].xaxis.set_ticks_position('bottom')          # 設定bottom 為 x軸
        ax[ax_i].yaxis.set_ticks_position('left')            # 設定left   為 y軸
        ax[ax_i].spines['bottom'].set_position(('data', 0))  # 設定bottom x軸 位置(要丟tuple)
        ax[ax_i].spines['left']  .set_position(('data', 0))  # 設定left   y軸 位置(要丟tuple)
        ax[ax_i].scatter(vert_x, vert_y, c="r")
        ax[ax_i].arrow(vert_x, vert_y, move_x, move_y, color="black", length_includes_head=True, head_width=(col / 20) + 1)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
        ax_i += 1

        ### xy_shift 視覺化
        ax[ax_i].set_title("step1.xy_mesh_shift, step2. move_xy")
        ax[ax_i].scatter(xy_f_shifted[:, 0], xy_f_shifted[:, 1], c = np.arange(row * col), s=1, cmap="hsv")
        ax[ax_i].spines['right'].set_color('None')
        ax[ax_i].spines['top']  .set_color('None')
        ax[ax_i].xaxis.set_ticks_position('bottom')         # 設定bottom 為 x軸
        ax[ax_i].yaxis.set_ticks_position('left')           # 設定left   為 y軸
        ax[ax_i].spines['bottom'].set_position(('data', 0))  # 設定bottom x軸 位置(要丟tuple)
        ax[ax_i].spines['left']  .set_position(('data', 0))  # 設定left   y軸 位置(要丟tuple)
        ax[ax_i].arrow(0, 0, move_x, move_y, color="black", length_includes_head=True, head_width=(col / 20) + 1)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
        ax_i += 1
    ################################################################################################################################
    ''' step2.選一個移動向量 來 決定每個點要怎麼移動
        move_x       ： 一個值
        move_y       ： 一個值
        move_xy      ： [[move_x, move_y   ]]， shape為 (1, 2)
        move_xyz     ： [[move_x, move_y, 0]]， shape為 (1, 3)
        move_xyz_proc： [[move_x, move_y, 0],
                         [move_x, move_y, 0],
                         ...]                ， shape為 (row * col, 3)
    '''
    if(move_x == 0 and move_y == 0):  ### 兩者不能同時為0，要不然算外積好像有問題
        move_x = 0.00001
        move_y = 0.00001
    move_xy = np.array([[move_x, move_y]], dtype=np.float64)
    move_xyz      = np.array([move_x, move_y, 0])        ### 多一個channel z，填0
    move_xyz_proc = np.tile( move_xyz, (row * col, 1) )  ### 擴張給每個點使用


    ################################################################################################################################
    ''' step3.決定每個點的移動大小d，決定d的大小方式： 以扭曲點為中心的正mesh 和 move 做 cross product
        https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
        沒有說為什麼做這樣做！！！ paper是說觀察拉~~ 但也沒說怎麼觀察的 @口@
        但就是 以扭曲點為中心的正mesh 和 move 做 cross product， 會有下面兩個性質
            1. 扭曲點 到 正mesh該點的     距離   越近 d越小 最小為0
            2. 扭曲點 到 正mesh該點的 方向和move 越像 d越小 最小為0
        如果從結果來回去觀察，好像確實有這樣子的情形～

        d_raw_f   ： 剛做完 corss 最原始的結果， shape 為 (row * col, 3)
        d_raw     ： 只取 z維 的值， shape 維 (row * col, )
                        因為  以扭曲點為中心的正mesh每個點(當向量) 是二維， move向量 是 二維，
                        二維 跟 二維 的 向量做 cross， 其 結果向量的：
                            方向：一定在 第三維 且 兩維必0，
                            長度：就是 兩個向量 圍出的平行四邊形面積

        d_abs     ： d_raw 取絕對值，但看印度那篇發現其實不用！
        d_abs_norm： d向量的長度 除去 move向量的長度
    '''
    d_raw_f = np.cross(xyz_f_shifted, move_xyz_proc)    ### shape=(...,3)
    d_raw = d_raw_f[:, 2]
    d_abs = np.absolute(d_raw_f[:, 2])                  ### shape=(...,1)，前兩個col一定是0，所以取第三個col，且 我們只在意"值"不在意"方向"，所以取絕對值
    # d_abs = d_raw  ### 不用 abs 試試看
    d_abs_norm = d_abs / (np.linalg.norm(move_xy, ord=2))  ### norm2 就是算 move_xy向量的 長度 喔！
    # d_abs_norm = d_abs_norm / d_abs_norm.max() ### 最後嘗試了以後覺得效果不好，所以還是用回原本的 除 move_xy的向量長度


    '''視覺化 xy 和 xy_shift 和 move_xy'''
    if(debug):
        # print("d_abs.max()", d_abs.max())
        # print("d_abs.min()", d_abs.min())
        # print("d_abs.shape", d_abs.shape)
        # print("d_abs[0]", d_abs[0])

        # print("d_abs_norm.max()", d_abs_norm.max())
        # print("d_abs_norm.min()", d_abs_norm.min())
        # print("d_abs_norm.shape", d_abs_norm.shape)
        # print("d_abs_norm[0]", d_abs_norm[0])
        debug_spyder_dict["xyz_f_shifted"] = xyz_f_shifted
        debug_spyder_dict["move_xyz_proc"] = move_xyz_proc
        debug_spyder_dict["d_0_d_raw"] = d_raw
        debug_spyder_dict["d_1_d_abs"] = d_abs
        debug_spyder_dict["d_2_move_xy_norm2"] = np.linalg.norm(move_xy, ord=2)
        debug_spyder_dict["d_3_d_abs_norm"] = d_abs_norm

        ### 在 x_y_f_shifted 上 畫出 d_raw/ d_abs/ d_abs/norm
        ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d"); ax3d.set_title("step3.d_raw");      ax[ax_i].remove(); ax3d.scatter(xy_f_shifted[:, 0], xy_f_shifted[:, 1], d_raw      , c = np.arange(row * col), s=1, cmap="hsv"); draw_3D_xy_plane_by_mesh_f(ax3d, row, col, mesh_x_f=xy_f_shifted[:, 0], mesh_y_f=xy_f_shifted[:, 1], z=0, alpha=0.5 ); ax_i += 1
        ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d"); ax3d.set_title("step3.d_abs");      ax[ax_i].remove(); ax3d.scatter(xy_f_shifted[:, 0], xy_f_shifted[:, 1], d_abs      , c = np.arange(row * col), s=1, cmap="hsv"); draw_3D_xy_plane_by_mesh_f(ax3d, row, col, mesh_x_f=xy_f_shifted[:, 0], mesh_y_f=xy_f_shifted[:, 1], z=0, alpha=0.5 ); ax3d.plot((0, move_x), (0, move_y), (0, 0) , c="b"); ax_i += 1
        ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d"); ax3d.set_title("step3.d_abs_norm"); ax[ax_i].remove(); ax3d.scatter(xy_f_shifted[:, 0], xy_f_shifted[:, 1], d_abs_norm , c = np.arange(row * col), s=1, cmap="hsv"); draw_3D_xy_plane_by_mesh_f(ax3d, row, col, mesh_x_f=xy_f_shifted[:, 0], mesh_y_f=xy_f_shifted[:, 1], z=0, alpha=0.5 ); ax3d.plot((0, move_x), (0, move_y), (0, 0) , c="b"); ax_i += 1
    #############################################################
    ''' step4. d越小移動越大，概念是要 離扭曲點越近移動越大。
            但是 "越大" 的 "越"的程度 折/捲 不一樣， 所以有 DocUNet paper 裡的兩個公式：
                折： wt = alpha / (d_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
                捲： wt = 1 - (d_norm )**(alpha)
                wt 的值域為 0~1， shape 為 (row * col, )
            細部的alpha參數可以看ppt，照這樣子做就有 折/捲 的效果
                alpha 值越大，扭曲越global(看起來效果小)
                alpha 值越小，扭曲越local(看起來效果大)
    '''
    if dis_type == "fold":
        wt = alpha / (d_abs_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
    elif dis_type == "curl":
        wt = 1 - (d_abs_norm / 100 )**(alpha)
    # print("dis_type:",dis_type)
    # print("wt.max()",wt.max())
    # print("wt.min()",wt.min())

    '''視覺化 wt'''
    if(debug):
        ##########################################################################################################
        ### 變動alpha， 變動d， 看wt
        split = 30
        ### alpha_fold
        alpha_fold_min       = 0
        alpha_fold_max       = 50
        alpha_fold_split     = np.linspace(alpha_fold_min, alpha_fold_max, split)
        alpha_fold_wt_para_v = np.tile(alpha_fold_split, (split, 1))    ### shape 為 (30, 30)

        ### alpha_curl
        alpha_curl_min       = 0
        alpha_curl_max       = 50
        alpha_curl_split     = np.linspace(alpha_curl_min, alpha_curl_max, split)
        alpha_curl_wt_para_v = np.tile(alpha_curl_split, (split, 1))    ### shape 為 (30, 30)

        ### d_min/max
        d_abs_norm_min = d_abs_norm.min()
        d_abs_norm_max = d_abs_norm.max()
        d_abs_norm_split = np.linspace(d_abs_norm_min, d_abs_norm_max, split)
        d_abs_norm_wt_para_v = np.tile(d_abs_norm_split, (split, 1)).T  ### shape 為 (30, 30)

        ### wt_fold calculate
        wt_fold_para_v = alpha_fold_wt_para_v / (d_abs_norm_wt_para_v + alpha_fold_wt_para_v + 0.00001)  ### +0.00001是怕分母為零
        ### wt_curl calculate
        wt_curl_para_v = 1 - (d_abs_norm_wt_para_v / 100 )**(alpha_curl_wt_para_v)

        ### wt_fold visual
        ax[ax_i].remove()
        ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d")
        ax3d.set_title("step4.d_alpha_fold")
        ax3d.set_xlabel("alpha")
        ax3d.set_ylabel("d")
        ax3d.scatter(alpha_fold_wt_para_v, d_abs_norm_wt_para_v, wt_fold_para_v , c = np.arange(split * split), s=1, cmap="hsv")
        draw_3D_xy_plane_by_mesh_f(ax3d, split, split, mesh_x_f=alpha_fold_wt_para_v.flatten(), mesh_y_f=d_abs_norm_wt_para_v.flatten() , z=0, alpha=0.5 )
        ax_i += 1

        ### wt_curl visual
        ax[ax_i].remove()
        ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d")
        ax3d.set_title("step4.d_alpha_curl")
        ax3d.set_xlabel("alpha")
        ax3d.set_ylabel("d")
        ax3d.scatter(alpha_fold_wt_para_v, d_abs_norm_wt_para_v, wt_curl_para_v , c = np.arange(split * split), s=1, cmap="hsv")
        draw_3D_xy_plane_by_mesh_f(ax3d, split, split, mesh_x_f=alpha_fold_wt_para_v.flatten(), mesh_y_f=d_abs_norm_wt_para_v.flatten() , z=0, alpha=0.5 )
        ax_i += 1


        ##########################################################################################################
        ### 固定alpha， 看xy_f_shifted上的d 對應的 wt
        alpha_fold = 50
        alpha_curl = 0.6
        wt_fold = alpha / (d_abs_norm + alpha_fold + 0.00001)  ### +0.00001是怕分母為零
        wt_curl = 1 - (d_abs_norm / 100 )**(alpha_curl)

        ax[ax_i].remove(); ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d"); ax3d.set_title("step4.wt_fold"); ax3d.scatter(xy_f_shifted[:, 0], xy_f_shifted[:, 1], wt_fold , c = np.arange(row * col), s=1, cmap="hsv"); ax3d.plot((0, move_x), (0, move_y), (0, 0) , c="b"); draw_3D_xy_plane_by_mesh_f(ax3d, row, col, mesh_x_f=xy_f_shifted[:, 0], mesh_y_f=xy_f_shifted[:, 1], z=0, alpha=0.5 ); ax_i += 1
        ax[ax_i].remove(); ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d"); ax3d.set_title("step4.wt_curl"); ax3d.scatter(xy_f_shifted[:, 0], xy_f_shifted[:, 1], wt_curl , c = np.arange(row * col), s=1, cmap="hsv"); ax3d.plot((0, move_x), (0, move_y), (0, 0) , c="b"); draw_3D_xy_plane_by_mesh_f(ax3d, row, col, mesh_x_f=xy_f_shifted[:, 0], mesh_y_f=xy_f_shifted[:, 1], z=0, alpha=0.5 ); ax_i += 1

    ##########################################################################################################################
    ### step5 根據 wt 做 move
    mesh_move = move_xy * np.expand_dims(wt, axis=1)  ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘

    if(debug):
        debug_spyder_dict["mesh_move"] = mesh_move

        ax[ax_i].set_title("step5.see mesh_move value")
        ax[ax_i].scatter(mesh_move[:, 0], mesh_move[:, 1], c = np.arange(row * col), s=1, cmap="hsv")
        ax[ax_i].spines['right'].set_color('None')
        ax[ax_i].spines['top']  .set_color('None')
        ax[ax_i].xaxis.set_ticks_position('bottom')  # 設定bottom 為 x軸
        ax[ax_i].yaxis.set_ticks_position('left')    # 設定left   為 y軸
        # ax[ax_i].arrow(0, 0, move_x, move_y, color="black", length_includes_head=True, head_width=(col / 20) + 1)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
        ax_i += 1


        '''視覺化 移動後的結果'''
        ax[ax_i].set_title("step5. dis_fold_coord")
        mesh_move_fold = move_xy * np.expand_dims(wt_fold, axis=1)  ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘
        show_distort_mesh_visual(row, col, mesh_move_fold, fig, ax[ax_i]); ax_i += 1

        ax[ax_i].set_title("step5. dis_curl_coord")
        mesh_move_curl = move_xy * np.expand_dims(wt_curl, axis=1)  ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘
        show_distort_mesh_visual(row, col, mesh_move_curl, fig, ax[ax_i]); ax_i += 1

        '''用視覺化 移動的過程'''
        ### 2D 箭頭 效果不好，可能是因為箭頭太密， 他的箭頭出不來， jump 調開 也出不來
        jump = 3
        ax[ax_i].set_title("step5.move 2D_visual")
        ax[ax_i].quiver(xy_f[::jump, 0], xy_f[::jump, 1], mesh_move_fold[::jump, 0], mesh_move_fold[::jump, 1])
        ax[ax_i].set_xlim(-30, 60)
        ax_i += 1

        ### 3D 來試試看
        ax3d = fig.add_subplot(1, fig_amount, ax_i + 1, projection="3d")
        ax3d.set_title("step5.move 3D_visual")
        ax3d.set_zlim(0, 1)
        ax[ax_i].remove()
        ax_i += 1

        ### 畫 3D 箭頭 效果不好， 分不同平面 畫箭頭 發現 箭頭太密看不到平面ˊ口ˋ
        # ax3d.quiver(xy_f[::jump, 0], xy_f[::jump, 1], 0,  mesh_move_fold[::jump, 0], mesh_move_fold[::jump, 1], 0.5, cmap="hsv")

        ### 畫 3D scatter 分不同平面 沒有箭頭的中間段 效果比較好
        ax3d.scatter(xy_f[::jump, 0]                            , xy_f[::jump, 1]                            , 0   , c = np.arange((row * col) // jump), s=1, cmap="hsv"); ax_i += 1
        ax3d.scatter(xy_f[::jump, 0] + mesh_move_fold[::jump, 0], xy_f[::jump, 1] + mesh_move_fold[::jump, 1], 0.5 , c = np.arange((row * col) // jump), s=1, cmap="hsv"); ax_i += 1


        debug_spyder_dict["quiver_xy_f"] = xy_f
        debug_spyder_dict["quiver_mesh_move_fold"] = mesh_move_fold
        debug_spyder_dict["quiver_xy_f[:, 0]"] = xy_f[:, 0]
        debug_spyder_dict["quiver_xy_f[:, 1]"] = xy_f[:, 1]
        debug_spyder_dict["quiver_mesh_move_fold[:, 0]"] = mesh_move_fold[:, 0]
        debug_spyder_dict["quiver_mesh_move_fold[:, 1]"] = mesh_move_fold[:, 1]


    # if(debug):
    #     ###########################################################################################
    #     fig, ax = plt.subplots(1, 2)
    #     d_map = d_abs_norm.reshape(row, col)
    #     ax[0].set_title("d_map")
    #     ax0_img = ax[0].imshow(d_map)  ### 因為有用到imshow，所以會自動設定 左上角(0,0)
    #     ax[0].scatter(vert_x, vert_y, c="r")    ### 畫出扭曲點
    #     ax[0].arrow(vert_x, vert_y, move_x, move_y, color="black", length_includes_head=True, head_width=(col / 20) + 1)  ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
    #     fig.colorbar(ax0_img, ax=ax[0])
    #     fig.set_size_inches(8, 4)
    #     ### 第二張子圖 plt是用scatter(左下角(0,0))，所以畫完圖後 y軸要上下顛倒一下，才會符合imaged看的方式(左上角(0,0))
    #     show_distort_mesh_visual(row, col, mesh_move, fig, ax[1])

    #     # print("move_xy", move_xy)
    #     # print("wt", wt)
    #     # print("mesh_move", mesh_move)
    #     # print("move_xy.shape", move_xy.shape)
    #     # print("wt.shape", wt.shape)
    #     # print("mesh_move.shape", mesh_move.shape)
    #     plt.show()
    ###########################################################################################

    ### 備用程式碼
    # xy_v            = xy_f.reshape(row, col, 2)
    # xy_v_shifted    = xy_f_shifted.reshape(row, col, 2)
    # xyz_v_shifted   = xyz_f_shifted.reshape(row, col, 3)
    # move_xyz_proc_v = move_xyz_proc.reshape(row, col, 3)
    # d_raw_v         = d_raw_f.reshape(row, col, 3)
    if(debug):
        plt.savefig("debug_result")
        plt.show()
    return mesh_move  ### shape：(..., 2)
#######################################################################################################################################
#######################################################################################################################################
### 以下都是用取得參數以後，呼叫上面的本體做扭曲喔！
def get_rand_para(row, col, curl_probability, smooth=False):
    ratio = row / 4
    vert_x = np.random.randint(col)
    vert_y = np.random.randint(row)
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
def distort_rand(dst_dir=".", start_index=0, amount=2000, row=40, col=30, distort_time=None, curl_probability=0.3, move_x_thresh=40, move_y_thresh=55, smooth=False, write_npy=True):
    start_time = time.time()
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "distort_mesh_visuals")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "move_maps")
    Check_dir_exist_and_build(data_access_path + dst_dir + "/" + "distort_infos")

    move_maps = []
    for index in range(start_index, start_index + amount):
        dis_start_time = time.time()

        result_move_f = np.zeros(shape=(row * col, 2), dtype=np.float64)     ### 初始化 move容器
        if(distort_time is None): distort_time  = np.random.randint(4) + 1  ### 隨機決定 扭曲幾次

        for _ in range(distort_time):  ### 扭曲幾次
            while(True):  ### 如果扭曲太大的話，就重新取參數做扭曲，這樣就可以控制取到我想要的合理範圍
                vert_x, vert_y, move_x, move_y, dis_type, alpha = get_rand_para(row, col, curl_probability, smooth)  ### 隨機取得 扭曲參數
                # print("curl_probability",curl_probability, "dis_type",dis_type)
                move_f = distort( row, col, vert_x, vert_y, move_x, move_y , dis_type, alpha, debug=False )  ### 用參數去得到 扭曲move_f
                max_move_x = abs(move_f[:, 0]).max()
                max_move_y = abs(move_f[:, 1]).max()
                if(max_move_x < move_x_thresh and max_move_y < move_y_thresh): break
                else: print("too much, continue random")
            result_move_f = result_move_f + move_f  ### 走到這就取到 在我覺得合理範圍的 move_f 囉，把我要的move_f加進去容器內～

            ### 紀錄扭曲參數
            if  (dis_type == "fold"): distrore_info_log(data_access_path + dst_dir + "/" + "distort_infos", index, row, col, distort_time, vert_x, vert_y, move_x, move_y, dis_type, alpha )
            elif(dis_type == "curl"): distrore_info_log(data_access_path + dst_dir + "/" + "distort_infos", index, row, col, distort_time, vert_x, vert_y, move_x, move_y, dis_type, alpha )


        ## 紀錄扭曲視覺化的結果
        # save_distort_mesh_visual(result_move_f, index)

        result_move_map = result_move_f.reshape(row, col, 2)  ### (..., 2)→(row, col, 2)
        result_move_map = result_move_map.astype(np.float32)
        if(write_npy) : np.save(data_access_path + dst_dir + "/" + "move_maps/%06i" % index, result_move_map)  ### 把move_map存起來，記得要轉成float32！
        print("%06i process 1 mesh cost time:" % index, "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time) )
        move_maps.append(result_move_map)
    return np.array(move_maps, dtype=np.float32)

### 用 step2_d去試 我想要的參數喔！ 測試結果還是不大像喔，要用step2_a_distort_page_and_pers.py 的 distort_more_like_page 才更像
def distort_like_page(dst_dir, start_index, row, col, write_npy=True):
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
            result_move_f = np.zeros(shape=(row * col, 2), dtype=np.float64)  ### 初始化 move容器
            move_f = distort( row, col, go_vert_x, vert_y, move_x, go_move_y , dis_type, alpha, debug=False )  ### 用參數去得到 扭曲move_f
            result_move_f = result_move_f + move_f  ### 把我要的move_f加進去容器內～
            distrore_info_log(data_access_path + dst_dir + "/" + "distort_infos", index, row, col, distort_time, go_vert_x, vert_y, move_x, go_move_y, dis_type, alpha )  ### 紀錄扭曲參數
            result_move_map = result_move_f.reshape(row, col, 2)  ### (..., 2)→(row, col, 2)
            result_move_map = result_move_map.astype(np.float32)
            # save_distort_mesh_visual(result_move_f, index)   ### 紀錄扭曲視覺化的結果
            np.save(data_access_path + dst_dir + "/" + "move_maps/%06i" % index, result_move_map)  ### 把move_map存起來，記得要轉成float32！
            print("%06i process 1 mesh cost time:" % index, "%.3f" % (time.time() - dis_start_time), "total_time:", time_util(time.time() - start_time) )
            index += 1
            move_maps.append(result_move_map)
    return np.array(move_maps.astype(np.float32))

#######################################################################################################################################
#######################################################################################################################################
### 以下跟 紀錄、視覺化相關
def distrore_info_log(log_dir, index, row, col, distort_times, vert_x, vert_y, move_x, move_y, dis_type, alpha ):
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
    with open(log_dir + "/" + "%06i-row=%i,col=%i,distort_times=%i.txt" % (index, row, col, distort_times), "a") as file_log:
        file_log.write(str_template % (vert_x, vert_y, move_x, move_y, dis_type, alpha))

def show_distort_mesh_visual(row, col, move_f, fig, ax):
    xy_f = get_xy_f(row, col)
    xy_f = xy_f + move_f
    ax.invert_yaxis()  ### 整張圖上下顛倒

    ax_img = ax.scatter(xy_f[:, 0], xy_f[:, 1], c = np.arange(row * col), cmap="hsv", s=1)
    ax.spines['right'].set_color('None')
    ax.spines['top']  .set_color('None')
    ax.xaxis.set_ticks_position('bottom')         # 設定bottom 為 x軸
    ax.yaxis.set_ticks_position('left')           # 設定left   為 y軸
    ax.spines['bottom'].set_position(('data', 0))  # 設定bottom x軸 位置(要丟tuple)
    ax.spines['left']  .set_position(('data', 0))  # 設定left   y軸 位置(要丟tuple)
    fig.colorbar(ax_img, ax=ax)

def save_distort_mesh_visual(result_move_f, index):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 5)
    show_distort_mesh_visual(row, col, result_move_f, fig, ax)
    plt.savefig(data_access_path + "step2_flow_build/distort_mesh_visuals/%06i.png" % index)
    plt.close()

def show_move_map_visual(move_map, ax):
    canvas = move_map / move_map.max()
    canvas = np.dstack( (canvas, np.ones(shape=(row, col, 1), dtype=np.float64) ) )
    ax.imshow(canvas)
    plt.show()


if(__name__ == "__main__"):
    ### 理解用，手動慢慢扭曲
    row = 30
    col = 30

    move_f =          distort(row, col, vert_x=col / 2, vert_y=row / 2, move_x=  -10, move_y=  10, dis_type="fold", alpha=  30, debug=True )  ### alpha:2~4
    # move_f = move_f + distort(row, col, vert_x=      0, vert_y=     10, move_x= 3.5, move_y= 2.5, dis_type="fold", alpha=200, debug=True )

    # fig, ax = plt.subplots(1,1)
    # fig.set_size_inches(4, 5)
    # show_distort_mesh_visual(row,col,move_f,fig, ax)
    # plt.show()

    #############################################################################################################################################
    ### 隨機生成 256*256_2000張
    # dst_dir = "step2_build_flow_h=256,w=256_complex"
    # row=256
    # col=256
    # amount=2000
    # distort_rand(dst_dir=dst_dir, start_index=0, amount=2000, row=row, col=col,distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55 )


    #############################################################################################################################################
    ### 隨機生成 384*256_2000張
    # dst_dir = "step2_build_flow_h=384,w=256_complex"
    # row=384
    # col=256
    # amount=2000
    # distort_rand(dst_dir=dst_dir, start_index=0, amount=amount, row=row, col=col,distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55 )
    ################################################
    #### 接續生成 頁面 扭曲影像，分開生成的原因是要 complex和complex+page 用的是相同的complex，所以page獨立生成，再把上面生成的結果複製一份，改名成complex+page，再把這裡生成的結果加進去
    # dst_dir = "step2_build_flow_h=384,w=256_page"
    # row=384
    # col=256
    # distort_like_page(dst_dir=dst_dir, start_index=2000    , row=row, col=col) ### 目前寫死，固定生成60*26個 move_maps喔！
    #############################################################################################################################################
    ### 平滑多一點 384*256_1500張
    # dst_dir = "step2_build_flow_h=384,w=256_smooth_curl+fold"
    # row = 384
    # col = 256
    # amount = 450
    # distort_rand(dst_dir=dst_dir, start_index=amount * 0, amount=amount, row=row, col=col, distort_time=1, curl_probability=1.0, move_x_thresh=40, move_y_thresh=55, smooth=True )
    # distort_rand(dst_dir=dst_dir, start_index=amount * 1, amount=amount, row=row, col=col, distort_time=1, curl_probability=0.0, move_x_thresh=40, move_y_thresh=55, smooth=True )

    #############################################################################################################################################
    ### old應該要拿掉，有生成像page的60*26張，剩下用隨機補滿2000張
    # start_index = 0
    # amount = 60*26
    # row = 384#400#256#40*10#472 #40*10
    # col = 256#300#256#30*10#304 #30*10
    # dst_dir = "step2_flow_build_page"
    # distort_like_page(dst_dir=dst_dir, start_index=0    , row=row, col=col) ### 目前寫死，固定生成60*26個 move_maps喔！
    # distort_rand     (dst_dir=dst_dir, start_index=60*26, amount=2000-60*26, row=row, col=col,distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55 )
