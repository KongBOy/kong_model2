import sys
sys.path.append("kong_util")

from build_dataset_combine import Check_dir_exist_and_build
from util import time_util
from matplot_fig_ax_util import change_into_img_2D_coord_ax

from step0_access_path import data_access_path
from step2_a_distort_curl_and_fold import get_dis_move_map, get_xy_f_and_m
import time
import numpy as np
import matplotlib.pyplot as plt

debug_spyder_dict = {}

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
                move_f, _ = get_dis_move_map( x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, w_res=w_res, h_res=h_res, vert_x=vert_x, vert_y=vert_y, move_x=move_x, move_y=move_y , dis_type=dis_type, alpha=alpha, debug=False )  ### 用參數去得到 扭曲move_f
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
            move_f, _ = get_dis_move_map( x_min, x_max, y_min, y_max, w_res, h_res, go_vert_x, vert_y, move_x, go_move_y , dis_type, alpha, debug=False )  ### 用參數去得到 扭曲move_f
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
