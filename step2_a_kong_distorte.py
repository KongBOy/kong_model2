import numpy as np 
import matplotlib.pyplot as plt 
from build_dataset_combine import Check_dir_exist_and_build_new_dir

### 參考連結：https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
def get_xy_f(row,col): ### get_xy_flatten，拿到的map的shape：(..., 2)
    x = np.arange(col)
    x = np.tile(x,(row,1))
    
#    y = np.arange(row-1, -1, -1) ### 就是這裡要改一下拉！不要抄網路的，網路的是用scatter的方式來看(左下角(0,0)，x往右增加，y往上增加)
    y = np.arange(row) ### 改成這樣子 就是用image的方式來處理囉！(左上角(0,0)，x往右增加，y往上增加)
    y = np.tile(y,(col,1)).T
    
    x_f = x.flatten()
    y_f = y.flatten()
    xy_f = np.array( [x_f, y_f] )
    return xy_f.T

### 整個function都是用 image的方式來看(左上角(0,0)，x往右邊走增加，y往上面走增加)
def distorte(row, col, x, y, move_x, move_y, curve_type="fold", alpha_fold=50, alpha_curl=1.95,debug=False):
    xy_f = get_xy_f(row, col) ### 拿到map的shape：(..., 2)

    ### step1.選一個點當扭曲點
    vtex = np.array([x,y])      ### 指定的扭曲點xy座標
    xy_f_shifted  = xy_f - vtex ### 位移整張mesh變成以扭曲點座標為原點
    # 準備等等要算 d的材料：xy_f_shifted 多加一個channel z，填0
    xyz_f = np.zeros(shape=(row*col, 3)) 
    xyz_f[:, 0:2] = xy_f_shifted
    
    ### step2.選一個移動向量 來 決定每個點要怎麼移動
    move_xy = np.array([[move_x,move_y]],dtype=np.float64) 
    move_xyz      = np.array([move_x, move_y, 0])    ### 多一個channel z，填0
    move_xyz_proc = np.tile( move_xyz, (row*col,1) ) ### 擴張給每個點使用
    
    ### step3.決定每個點的移動大小d，決定d的大小方式：離扭曲點越近d越小 且 以扭曲點為原點，走到該點的向量 越像(正反方向) 移動向量的話 d越小，
    ###   所以就用  位移後的mesh(就剛好是以扭曲點為原點，走到該點的向量) 和 移動向量 做 外積了(方向和 移動向量 越像，算出來的值越小)
    d = np.cross(xyz_f, move_xyz_proc)   ###shape=(...,3)
    d = np.absolute(d[:, 2]) ###shape=(...,1)，前兩個col一定是0，所以取第三個col，且 我們只在意"值"不在意"方向"，所以取絕對值
    d = d / (np.linalg.norm(move_xy, ord=2)) ### norm2 就是算 move_xy向量的 長度 喔！
#    d = d / d.max() ### 最後嘗試了以後覺得效果不好，所以還是用回原本的 除 move_xy的向量長度
    # print("curve_type:",curve_type)
    # print("d.max()",d.max())
    # print("d.min()",d.min())
    
    #############################################################
    ### step4. d越小移動越大，概念是要 離扭曲點越近移動越大。細部的alpha參數可以看ppt，照這樣子做就有 折/捲 的效果
    ### wt算除來的值介於0~1之間，
    ### alpha 值越大，扭曲越global(看起來效果小)
    ### alpha 值越小，扭曲越local(看起來效果大)
    if curve_type == "fold":
        wt = alpha_fold / (d + alpha_fold+0.00001)
    elif curve_type == "curl":
        wt = 1 - (d/100 )**alpha_curl
    # print("wt.max()",wt.max())
    # print("wt.min()",wt.min())
    mesh_move = move_xy * np.expand_dims(wt, axis=1)
    

    if(debug):
    ###########################################################################################
        fig, ax = plt.subplots(1,2)
        d_map = d.reshape(row,col)
        ax[0].set_title("d_map")
        ax0_img = ax[0].imshow(d_map) ### 因為有用到imshow，所以會自動設定 左上角(0,0)
        ax[0].scatter(x, y, c="r")    ### 畫出扭曲點
        ax[0].arrow(x,y ,move_x, move_y, color="white",length_includes_head=True,head_width=(col/20)+1) ### 移動向量，然後箭頭化的方式是(x,y,dx,dy)！ 不是(x1,y1,x2,y2)！head_width+1是怕col太小除完變0
        fig.colorbar(ax0_img, ax=ax[0])
        fig.set_size_inches(8,4)
        ### 第二張子圖 plt是用scatter(左下角(0,0))，所以畫完圖後 y軸要上下顛倒一下，才會符合imaged看的方式(左上角(0,0))
        show_distorted_mesh(row, col, mesh_move,fig, ax[1])
        plt.show()
    ###########################################################################################
    return mesh_move

### 只有 參數隨機產生， funciton 重複使用 上面寫好的function喔！
def distorte_rand(row=40, col=30, distort_times=None, curl_probability=0.3, index=0):
    ratio = row/4
    result_move_f = np.zeros(shape=(row*col,2), dtype=np.float64)
    if(distort_times is None):
        distort_times  = np.random.randint(4)+1  ### 隨機決定 扭曲幾次
    for _ in range(distort_times):
        x = np.random.randint(col)
        y = np.random.randint(row)
        move_x = ((np.random.rand()-0.5)*0.9)*ratio ### (-0.45~0.45) * ratio
        move_y = ((np.random.rand()-0.5)*0.9)*ratio ### (-0.45~0.45) * ratio
        curve_type = np.random.rand(1)
        if(curve_type > curl_probability):
            curve_type="fold"
            alpha_fold = np.random.rand(1) * 50 + 50 ### 結果發現web上的還不錯
        #    alpha_fold = (np.random.rand(1)*2.25 + 2.25)*ratio
            move_f = distorte( row, col, x, y, move_x, move_y , curve_type, alpha_fold=alpha_fold, debug=False )
        else:
            curve_type="curl"
            alpha_curl = (np.random.rand(1)/2 + 0.5)*1.7 ### ### 結果發現web上的還不錯，只有多rand部分除2
        #    alpha_curl = (np.random.rand(1)*0.045 + 0.045)*ratio
            move_f = distorte( row, col, x, y, move_x, move_y , curve_type, alpha_curl=alpha_curl, debug=False )
        result_move_f = result_move_f + move_f


        ### 紀錄扭曲參數
        if(curve_type == "fold"):    
            distrore_info_log("step2_flow_build/distorte_info", index, row, col, distort_times, x, y, move_x, move_y, curve_type, alpha_fold, )
        elif curve_type == "curl":
            distrore_info_log("step2_flow_build/distorte_info", index, row, col, distort_times, x, y, move_x, move_y, curve_type, alpha_curl, )

    return result_move_f

def shift_distort_move_f(move_f): ### 調整move_f，讓最小的位移量為0，意思就是不會有負數的狀況！好處是可以設-1為背景～
    ### 注意喔！move_map位移量最小是(0,0) 不代表 影像配合move_map 位移後 在 完全切得剛剛好的情況下 其 左上角的座標是(0,0)喔！
    ### 因為有可能 偏上面的 往下移動 或者 偏左邊的點 往右邊移動喔～
    ### 然後除非剛剛好 位移後 左上角的座標是(0,0)，否則 我們不能手動切成左上角剛剛好是座標(0,0)，因為這樣就代表 位移量 最小不是(0,0)了！
    ### 所以不要在這邊嘗試 shift 成 影像配合move_map 位移後 在 完全切得剛剛好的情況下 其 左上角的座標是(0,0)喔！
    shift_move_f = move_f.copy() 
    left = move_f[:,0].min()
    top  = move_f[:,1].min()
    
    shift_move_f[:,0] += left*-1
    shift_move_f[:,1] += top *-1
    return shift_move_f
    
    
    
def distrore_info_log(log_dir, index, row, col, distorte_times, x, y, move_x, move_y, curve_type, alpha, ):
    str_template = \
"\
x=%i\n\
y=%i\n\
move_x=%i\n\
move_y=%i\n\
curve_type=%s\n\
alpha=%i\n\
\n\
"
    with open(log_dir + "/" + "distort_info_%06i-row=%i,col=%i,distorte_times=%i.txt"%(index,row,col,distorte_times),"a") as file_log:
        file_log.write(str_template%(x, y, move_x, move_y, curve_type, alpha))

def show_distorted_mesh(row,col, move_f,fig, ax):
    xy_f = get_xy_f(row,col)
    xy_f = xy_f + move_f
    ax.set_title("distorted_mesh")
    ax_img = ax.scatter(xy_f[:,0],xy_f[:,1],c = np.arange(row*col),cmap="brg")
    fig.colorbar(ax_img,ax=ax)
    ax = ax.invert_yaxis() ### 整張圖上下顛倒

def show_move_map_visual(move_map, ax):
    canvas = move_map / move_map.max()
    canvas = np.dstack( (canvas, np.ones(shape=(row,col,1), dtype=np.float64) )  ) 
    ax.imshow(canvas)
    plt.show()

if(__name__=="__main__"):
    row = 256#40*10#472 #40*10
    col = 256#30*10#304 #30*10

    ### 理解用，手動慢慢扭曲
#    move_f =          distorte( row, col, x=col/2, y=row/2, move_x= col/2, move_y= row/2, curve_type="fold", alpha_fold=2, debug=True )  ### alpha_fold:2~4
#    move_f = move_f + distorte( row, col, x= 0, y=10, move_x=3.5, move_y= 2.5, curve_type="fold", alpha_fold=200, debug=True )

#    fig, ax = plt.subplots(1,1)
#    fig.set_size_inches(4, 5)
#    show_distorted_mesh(row,col,move_f,fig, ax)
#    plt.show()
    ##############################################################################################
    ### 隨機生成
    Check_dir_exist_and_build_new_dir("step2_flow_build/distorted_mesh")
    Check_dir_exist_and_build_new_dir("step2_flow_build/move_map")
    Check_dir_exist_and_build_new_dir("step2_flow_build/distorte_info")

    import time
    start_time = time.time()
    for index in range(1000):
        dis_start_time = time.time()
        result_move_f = distorte_rand(row,col, index=index, distort_times=1 ,curl_probability=0.9)
        # print("result_move_f.shape",result_move_f.shape) ### (..., 2)

        result_move_f = shift_distort_move_f(result_move_f) ### 調整move_f，讓最小的位移量為0，意思就是不會有負數的狀況！好處是可以設-1為背景～
        print(result_move_f[:,0].min(), result_move_f[:,1].min()) ### 確認位移量為0，意思就是不會有負數的狀況！好處是可以設-1為背景～

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(4, 5)
        show_distorted_mesh(row,col, result_move_f, fig, ax)
        # plt.show()
        plt.savefig("step2_flow_build/distorted_mesh/%06i.png"%index)
        plt.close()
        
        result_move_map = result_move_f.reshape(row,col,2) ### ### (..., 2)→(row, col, 2)
        for go_row in range(row):
            for go_col in range(col):
                if result_move_map[go_row,go_col].sum() == 0: ### 如果不是背景的話，都會有小小的移動量來跟背景做區別，前景背景的區別在rec_move_map的時候會用到！ 
                    result_move_map[go_row,go_col] += 0.00001 
        np.save("step2_flow_build/move_map/%06i"%index,result_move_map.astype(np.float32))
        print("%06i process 1 mesh cost time:"%index, "%.3f"%(time.time()-dis_start_time), "total_time:", "%.3f"%(time.time()-start_time))
    ##############################################################################################
#    fig, ax = plt.subplots(1,1)
#    fig.set_size_inches(4, 5)
#    
#    move_map   = np.load("move_map.npy")
#    canvas = move_map.copy()
##    move_map2img[:,:,1] = move_map2img[:,:,1] ### scatter 和 image 看的方式轉換：y差一個負號
#    canvas = move_map / move_map.max()
#    canvas = np.dstack( (canvas, np.ones(shape=(row,col,1), dtype=np.float64) )  ) 
#    ax.imshow(canvas)
#    plt.show()
#    
#    plt.figure()
#    fig, ax = plt.subplots(nrows=1, ncols=1)
#    show_move_map_visual(move_map2img, ax)