from step0_access_path import access_path
import numpy as np 
import matplotlib.pyplot as plt 
from build_dataset_combine import Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir
from util import time_util
import time
### 參考連結：https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
def get_xy_f(row,col): ### get_xy_flatten，拿到的map的shape：(..., 2)
    x = np.arange(col)
    x = np.tile(x,(row,1))
    
#    y = np.arange(row-1, -1, -1) ### 就是這裡要改一下拉！不要抄網路的，網路的是用scatter的方式來看(左下角(0,0)，x往右增加，y往上增加)
    y = np.arange(row) ### 改成這樣子 就是用image的方式來處理囉！(左上角(0,0)，x往右增加，y往上增加)
    y = np.tile(y,(col,1)).T
    
    x_f = x.flatten()
    y_f = y.flatten()
    xy_f = np.array( [x_f, y_f], dtype=np.float64 )
    return xy_f.T

### 整個function都是用 image的方式來看(左上角(0,0)，x往右邊走增加，y往上面走增加)
def distorte(row, col, vert_x, vert_y, move_x, move_y, dis_type="fold", alpha=50, debug=False):
    xy_f = get_xy_f(row, col) ### 拿到map的shape：(..., 2)

    ### step1.選一個點當扭曲點
    vtex = np.array([vert_x, vert_y])      ### 指定的扭曲點xy座標
    xy_f_shifted  = xy_f - vtex ### 位移整張mesh變成以扭曲點座標為原點
    # 準備等等要算 d的材料：xy_f_shifted 多加一個channel z，填0
    xyz_f = np.zeros(shape=(row*col, 3)) 
    xyz_f[:, 0:2] = xy_f_shifted
    
    ### step2.選一個移動向量 來 決定每個點要怎麼移動
    if(move_x==0 and move_y==0): ### 兩者不能同時為0，要不然算外積好像有問題
        move_x = 0.00001 
        move_y = 0.00001
    move_xy = np.array([[move_x,move_y]],dtype=np.float64) 
    move_xyz      = np.array([move_x, move_y, 0])    ### 多一個channel z，填0
    move_xyz_proc = np.tile( move_xyz, (row*col,1) ) ### 擴張給每個點使用
    
    ### step3.決定每個點的移動大小d，決定d的大小方式：離扭曲點越近d越小 且 以扭曲點為原點，走到該點的向量 越像(正反方向) 移動向量的話 d越小，
    ###   所以就用  位移後的mesh(就剛好是以扭曲點為原點，走到該點的向量) 和 移動向量 做 外積了(方向和 移動向量 越像，算出來的值越小)
    d = np.cross(xyz_f, move_xyz_proc)   ###shape=(...,3)
    d = np.absolute(d[:, 2]) ###shape=(...,1)，前兩個col一定是0，所以取第三個col，且 我們只在意"值"不在意"方向"，所以取絕對值
    d = d / (np.linalg.norm(move_xy, ord=2)) ### norm2 就是算 move_xy向量的 長度 喔！
    # d = d / d.max() ### 最後嘗試了以後覺得效果不好，所以還是用回原本的 除 move_xy的向量長度
    # print("dis_type:",dis_type)
    # print("d.max()",d.max())
    # print("d.min()",d.min())
    
    #############################################################
    ### step4. d越小移動越大，概念是要 離扭曲點越近移動越大。細部的alpha參數可以看ppt，照這樣子做就有 折/捲 的效果
    ### wt算除來的值介於0~1之間，
    ### alpha 值越大，扭曲越global(看起來效果小)
    ### alpha 值越小，扭曲越local(看起來效果大)
    if dis_type == "fold":
        wt = alpha / (d + alpha+0.00001) ### +0.00001是怕分母為零
    elif dis_type == "curl":
        wt = 1 - (d/100 )**(alpha)
    # print("wt.max()",wt.max())
    # print("wt.min()",wt.min())

    mesh_move = move_xy * np.expand_dims(wt, axis=1) ### 移動量*相應的權重，wt的shape是(...,)，move_xy的shape是(...,2)，所以wt要expand一下成(...,1)才能相乘
    

    if(debug):
    ###########################################################################################
        fig, ax = plt.subplots(1,2)
        d_map = d.reshape(row,col)
        ax[0].set_title("d_map")
        ax0_img = ax[0].imshow(d_map) ### 因為有用到imshow，所以會自動設定 左上角(0,0)
        ax[0].scatter(vert_x, vert_y, c="r")    ### 畫出扭曲點
        ax[0].arrow(vert_x,vert_y ,move_x, move_y, color="white",length_includes_head=True,head_width=(col/20)+1) ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
        fig.colorbar(ax0_img, ax=ax[0])
        fig.set_size_inches(8,4)
        ### 第二張子圖 plt是用scatter(左下角(0,0))，所以畫完圖後 y軸要上下顛倒一下，才會符合imaged看的方式(左上角(0,0))
        show_distorted_mesh_visual(row, col, mesh_move,fig, ax[1])
        plt.show()
    ###########################################################################################
    return mesh_move ### shape：(..., 2)
#######################################################################################################################################
#######################################################################################################################################
### 以下都是用取得參數以後，呼叫上面的本體做扭曲喔！
def get_rand_para(row, col, curl_probability):
    ratio = row/4
    vert_x = np.random.randint(col)
    vert_y = np.random.randint(row)
    move_x = ((np.random.rand()-0.5)*0.9)*ratio ### (-0.45~0.45) * ratio
    move_y = ((np.random.rand()-0.5)*0.9)*ratio ### (-0.45~0.45) * ratio
    dis_type = np.random.rand(1)
    if(dis_type > curl_probability):
        dis_type="fold"
        alpha = np.random.rand(1) * 50 + 50 ### 結果發現web上的還不錯
    #    alpha = (np.random.rand(1)*2.25 + 2.25)*ratio
    else:
        dis_type="curl"
        alpha = (np.random.rand(1)/2 + 0.5)*1.7 ### ### 結果發現web上的還不錯，只有多rand部分除2
    #    alpha = (np.random.rand(1)*0.045 + 0.045)*ratio

    return vert_x, vert_y, move_x, move_y, dis_type, alpha

### 只有 參數隨機產生， funciton 重複使用 上面寫好的function喔！
def distort_rand(start_index=0, amount=2000, row=40, col=30, distort_time=None, curl_probability=0.3, move_x_thresh=40, move_y_thresh=55):
    start_time = time.time()
    for index in range(start_index, start_index+amount):
        dis_start_time = time.time()

        result_move_f = np.zeros(shape=(row*col,2), dtype=np.float64)     ### 初始化 move容器
        if(distort_time is None): distort_time  = np.random.randint(4)+1  ### 隨機決定 扭曲幾次

        for _ in range(distort_time): ### 扭曲幾次
            while(True): ### 如果扭曲太大的話，就重新取參數做扭曲，這樣就可以控制取到我想要的合理範圍
                vert_x, vert_y, move_x, move_y, dis_type, alpha = get_rand_para(row, col, curl_probability)  ### 隨機取得 扭曲參數
                move_f = distorte( row, col, vert_x, vert_y, move_x, move_y , dis_type, alpha, debug=False ) ### 用參數去得到 扭曲move_f
                max_move_x = abs(move_f[:,0]).max()
                max_move_y = abs(move_f[:,1]).max()
                if(max_move_x < move_x_thresh and max_move_y < move_y_thresh):break
                else:print("too much, continue random")
            result_move_f = result_move_f + move_f ### 走到這就取到 在我覺得合理範圍的 move_f 囉，把我要的move_f加進去容器內～

            ### 紀錄扭曲參數
            if (dis_type == "fold"):distrore_info_log(access_path+"step2_flow_build/distorte_infos", index, row, col, distort_time, vert_x, vert_y, move_x, move_y, dis_type, alpha )
            elif dis_type == "curl":distrore_info_log(access_path+"step2_flow_build/distorte_infos", index, row, col, distort_time, vert_x, vert_y, move_x, move_y, dis_type, alpha )


        ## 紀錄扭曲視覺化的結果
        # save_distorted_mesh_visual(result_move_f, index)

        result_move_map = result_move_f.reshape(row,col,2) ### (..., 2)→(row, col, 2)
        np.save(access_path+"step2_flow_build/move_maps/%06i"%index,result_move_map.astype(np.float32)) ### 把move_map存起來，記得要轉成float32！
        print("%06i process 1 mesh cost time:"%index, "%.3f"%(time.time()-dis_start_time), "total_time:", time_util(time.time()-start_time) )


### 用 step2_d去試 我想要的參數喔！
def distort_like_page(start_index, row, col):
    start_time = time.time()
    
    ### 可以用step2_d去試 我想要的參數喔！
    distort_time = 1
    vert_y = 0
    move_x = 0
    dis_type = "curl"
    alpha = 1.5
    index = start_index

    for go_move_y in range(-13,13):      ### 可以用step2_d去試 我想要的參數喔！
        for go_vert_x in range(120,180): ### 可以用step2_d去試 我想要的參數喔！
            dis_start_time = time.time()
            result_move_f = np.zeros(shape=(row*col,2), dtype=np.float64) ### 初始化 move容器
            move_f = distorte( row, col, go_vert_x, vert_y, move_x, go_move_y , dis_type, alpha, debug=False ) ### 用參數去得到 扭曲move_f
            result_move_f = result_move_f + move_f ### 把我要的move_f加進去容器內～
            distrore_info_log(access_path+"step2_flow_build/distorte_infos", index, row, col, distort_time, go_vert_x, vert_y, move_x, go_move_y, dis_type, alpha ) ### 紀錄扭曲參數
            result_move_map = result_move_f.reshape(row,col,2) ### (..., 2)→(row, col, 2)
            # save_distorted_mesh_visual(result_move_f, index)   ### 紀錄扭曲視覺化的結果
            np.save(access_path+"step2_flow_build/move_maps/%06i"%index, result_move_map.astype(np.float32)) ### 把move_map存起來，記得要轉成float32！
            print("%06i process 1 mesh cost time:"%index, "%.3f"%(time.time()-dis_start_time), "total_time:", time_util(time.time()-start_time) )
            index += 1

#######################################################################################################################################
#######################################################################################################################################
### 以下跟 紀錄、視覺化相關
def distrore_info_log(log_dir, index, row, col, distorte_times, vert_x, vert_y, move_x, move_y, dis_type, alpha ):
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
    with open(log_dir + "/" + "%06i-row=%i,col=%i,distorte_times=%i.txt"%(index,row,col,distorte_times),"a") as file_log:
        file_log.write(str_template%(vert_x, vert_y, move_x, move_y, dis_type, alpha))

def show_distorted_mesh_visual(row,col, move_f,fig, ax):
    xy_f = get_xy_f(row,col)
    xy_f = xy_f + move_f
    ax.set_title("distorted_mesh_visual")
    ax_img = ax.scatter(xy_f[:,0],xy_f[:,1],c = np.arange(row*col),cmap="brg")
    fig.colorbar(ax_img,ax=ax)
    ax = ax.invert_yaxis() ### 整張圖上下顛倒

def save_distorted_mesh_visual(result_move_f, index):
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(4, 5)
    show_distorted_mesh_visual(row,col, result_move_f, fig, ax)
    plt.savefig(access_path+"step2_flow_build/distorted_mesh_visuals/%06i.png"%index)
    plt.close()

def show_move_map_visual(move_map, ax):
    canvas = move_map / move_map.max()
    canvas = np.dstack( (canvas, np.ones(shape=(row,col,1), dtype=np.float64) )  ) 
    ax.imshow(canvas)
    plt.show()

if(__name__=="__main__"):
    # access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！
    # start_index = 60*26
    # amount = 2000
    start_index = 0
    amount = 60*26
    row = 384#400#256#40*10#472 #40*10
    col = 256#300#256#30*10#304 #30*10

    ### 理解用，手動慢慢扭曲
#    move_f =          distorte( row, col, x=col/2, y=row/2, move_x= col/2, move_y= row/2, dis_type="fold", alpha=2, debug=True )  ### alpha:2~4
#    move_f = move_f + distorte( row, col, x= 0, y=10, move_x=3.5, move_y= 2.5, dis_type="fold", alpha=200, debug=True )

#    fig, ax = plt.subplots(1,1)
#    fig.set_size_inches(4, 5)
#    show_distorted_mesh_visual(row,col,move_f,fig, ax)
#    plt.show()
    ##############################################################################################
    ### 隨機生成
    Check_dir_exist_and_build_new_dir(access_path+"step2_flow_build/distorted_mesh_visuals")
    Check_dir_exist_and_build_new_dir(access_path+"step2_flow_build/move_maps")
    Check_dir_exist_and_build_new_dir(access_path+"step2_flow_build/distorte_infos")

    distort_like_page(start_index=0, row=row, col=col)
    distort_rand(start_index=60*26, amount=2000-60*26, row=row, col=col,distort_time=1, curl_probability=0.5, move_x_thresh=40, move_y_thresh=55 )

    
