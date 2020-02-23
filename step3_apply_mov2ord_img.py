from build_dataset_combine import Check_dir_exist_and_build
from util import get_dir_move, get_dir_img, method2, get_xy_map
import numpy as np 
import cv2

def get_l_r_t_d(kernel_size):
    l = t = (kernel_size+1)//2 -1
    r = d = kernel_size//2 +1
    return l,r,t,d


def emulate_apply_and_find_boundary(img, move_map): ### 模擬apply後的結果 來找右下角的 boundary
    ### 注意喔！move_map位移量最小是(0,0) 不代表 影像配合move_map 位移後 在 完全切得剛剛好的情況下 其 左上角的座標是(0,0)喔！
    ### 因為有可能 偏上面的 往下移動 或者 偏左邊的點 往右邊移動喔～
    ### 然後除非剛剛好 位移後 左上角的座標是(0,0)，否則 我們不能手動切成左上角剛剛好是座標(0,0)，因為這樣就代表 位移量 最小不是(0,0)了！
    ### 所以不要在這邊 把影像的 左邊、上面 切得剛剛好喔！
    row, col = img.shape[:2]
    x, y = get_xy_map(row, col)

    x = x + move_map[...,0]
    y = y + move_map[...,1]
    right = int(x.max())+1
    down = int(y.max())+1
    return down, right 

    
def search_mask_have_hole(dis_msk, hole_size=1):
    pag_msk = dis_msk.copy()
    pag_msk[pag_msk>1] = 1
    row, col = pag_msk.shape[:2]
    
    ### 用來過濾 前景的filter
    check_foreground_size = hole_size+2+2+2
    ch_l, ch_r, ch_t, ch_d = get_l_r_t_d(check_foreground_size)
    check_foreground_ok = check_foreground_size**2*0.91 ### 要超過check_foreground_ok才算事前景，沒超過就算背景
    
    ### 用來檢查 page上有沒有洞的filter
    check_page_size = hole_size+2
    pa_l, pa_r, pa_t, pa_d = get_l_r_t_d(check_page_size)
    page     = check_page_size**2
    hole     = (hole_size)//2+1
    check_page_ok  = page - hole ### 要超過page_ok才算是完整頁面(沒有洞)，沒超過就算有洞
    
#    debug用，視覺化前景 和 偵測到哪邊有洞
#    hole_around_visual = np.zeros_like(pag_msk)
#    foreground_visual = np.zeros_like(pag_msk)
    around_have_hole_amount=0
    for go_row in range(row):
        for go_col in range(col):
#            ### debug用，視覺化前景
#            if(pag_msk[go_row-ch_t:go_row+ch_d, go_col-ch_l:go_col+ch_r].sum() > check_foreground_ok):
#                foreground_visual[go_row,go_col] +=1
            if( (pag_msk[go_row-ch_t:go_row+ch_d, go_col-ch_l:go_col+ch_r].sum() > check_foreground_ok) and  ### 要確定是 頁面 還是 背景
                (pag_msk[go_row-pa_t:go_row+pa_d, go_col-pa_l:go_col+pa_r].sum() <= check_page_ok ) ): ### 要超過page_ok才算沒有洞，沒超過就算有洞
#                print("row,col around have hole", go_row, go_col, go_col, go_row)
#                hole_around_visual[go_row,go_col] +=1  ## debug用，視覺化偵測到哪邊有洞
                around_have_hole_amount +=1
    print("around_have_hole_amount",around_have_hole_amount )
#    cv2.imwrite("step2_flow_build/%s-3a-hole_around_visual.bmp"%(name),hole_around_visual*70)
#    cv2.imwrite("step2_flow_build/%s-3a-foreground_visual.bmp"%(name),foreground_visual*70)
    return around_have_hole_amount




def apply_move(img, move_map, name="0", write_to_step3=False):
    row, col = move_map.shape[:2] ### row, col 由 step2產生的flow 決定
    img = cv2.resize(img, (col, row), interpolation=cv2.INTER_NEAREST) 

    ksize = 3
    move_y_max, move_x_max = emulate_apply_and_find_boundary(img, move_map) ### 模擬apply後的結果 來找右下角的 boundary
    # move_x = move_map[..., 0]
    # move_x_max = abs(move_x.max())
    # move_x_min = abs(move_x.min())
    # move_y = move_map[..., 1]
    # move_y_max = abs(move_y.max())
    # move_y_min = abs(move_y.min())


    ### 初始化各個會用到的canvas
    dis_h = move_y_max
    dis_w = move_x_max 
    # dis_h = int( np.around(move_y_min + row + move_y_max) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    # dis_w = int( np.around(move_x_min + col + move_x_max) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    dis_img  = np.zeros(shape=(dis_h,dis_w,3), dtype=np.float64)
    rec_mov  = np.zeros(shape=(dis_h,dis_w,2), dtype=np.float64)
    dis_msk  = np.zeros(shape=(dis_h,dis_w),   dtype=np.uint8)
    
    ### 把原始影像扭曲，並取得 rec_mov
    for go_row in range(row):
        for go_col in range(col):
            dst_x = go_col + int(move_map[go_row,go_col,0] ) ### 找出位移後的x，因為array的存取要int，所以把move_map的值(float)強制轉int
            dst_y = go_row + int(move_map[go_row,go_col,1] ) ### 找出位移後的y，因為array的存取要int，所以把move_map的值(float)強制轉int

            dis_img[dst_y,dst_x,:] += img[go_row,go_col,:]
            rec_mov[dst_y,dst_x,:] += move_map[go_row,go_col,:]*-1

            dis_msk[dst_y,dst_x]   += 1

            # ### 我已經設定成如果不是背景的話，都會有小小的移動量來跟背景做區別，所以.sum!=0就代表前景囉！前景才需做移動！
            # if(move_map[go_row,go_col].sum() != 0):
            #     dst_x = go_col + int(move_map[go_row,go_col,0] + move_x_min) ### 現在的起點是(move_x_min, move_y_min)，所以要位移一下
            #     dst_y = go_row + int(move_map[go_row,go_col,1] + move_y_min) ### 現在的起點是(move_x_min, move_y_min)，所以要位移一下

            #     dis_img[dst_y,dst_x,:] += img[go_row,go_col,:]
            #     rec_mov[dst_y,dst_x,:] += move_map[go_row,go_col,:]*-1
            #     if(rec_mov[dst_y,dst_x].sum() == 0): ### 如果位移量剛好xy全0時，加一個小小的移動量來跟背景做區別
            #         rec_mov[dst_y,dst_x] += 0.00001
            #         print("here all zero")
            #     dis_msk[dst_y,dst_x]   += 1

            
            # print(dst_y,dst_x)
            # cv2.imshow("img",img)
            # cv2.imshow("Mask.bmp",(dis_msk*70).astype(np.uint8))
            # cv2.imshow("dis_img",dis_img.astype(np.uint8))
            # cv2.waitKey()


    for go_row in range(dis_img.shape[0]):
        for go_col in range(dis_img.shape[1]):
            if dis_msk[go_row,go_col] > 1: ### 扭曲的過程中可能有 多點移到相同的點會被加多次，把他跟加的次數除回來
                dis_img[go_row,go_col] = dis_img[go_row,go_col] / dis_msk[go_row, go_col]      
                rec_mov[go_row,go_col] = rec_mov[go_row,go_col] / dis_msk[go_row, go_col]

    
    

    ### 視覺化
    if(write_to_step3):
        move_map_visual = method2(move_map[...,0], move_map[...,1],color_shift=1)
        cv2.imwrite("step3_apply_flow_result/%s-1-I.bmp"%(name), img)
        cv2.imwrite("step3_apply_flow_result/%s-2-q.jpg"%(name), move_map_visual)
        cv2.imwrite("step3_apply_flow_result/%s-3a2-I1.jpg"%(name),dis_img)
        cv2.imwrite("step3_apply_flow_result/%s-3a3-Mask.bmp"%(name),dis_msk*70)
        np.save("step3_apply_flow_result/%s-3a3-Mask"%(name),dis_msk)
                
    ####################################################################################################################
    #### 扭曲影像 空洞的地方補起來
    search_mask_have_hole_count = 0
    while(search_mask_have_hole(dis_msk) > 0 and search_mask_have_hole_count < 10):
        dis_img_ref = dis_img.copy() ### 要把 參考img 跟 處理中img 分開 喔！
        dis_msk_ref = dis_msk.copy() ### 要把 參考msk 跟 處理中msk 分開 喔！
        rec_mov_ref = rec_mov.copy() ### 要把 參考rec_mov 跟 處理中rec_mov 分開 喔！
        for go_row in range(dis_h):
            for go_col in range(dis_w):
                if(dis_msk_ref[go_row,go_col]==0 and dis_msk_ref[go_row-1:go_row+2,go_col-1:go_col+2].sum() >=6 ):
                    # print(go_row,go_col)
                    l,r,t,d = get_l_r_t_d(ksize)
                    msk_sum = dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r].sum()
                    
                    dis_img[go_row,go_col,0] =  (dis_img_ref[go_row-t:go_row+d,go_col-l:go_col+r,0]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    dis_img[go_row,go_col,1] =  (dis_img_ref[go_row-t:go_row+d,go_col-l:go_col+r,1]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    dis_img[go_row,go_col,2] =  (dis_img_ref[go_row-t:go_row+d,go_col-l:go_col+r,2]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    
                    rec_mov[go_row,go_col,0] = (rec_mov_ref[go_row-t:go_row+d,go_col-l:go_col+r,0]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)
                    rec_mov[go_row,go_col,1] = (rec_mov_ref[go_row-t:go_row+d,go_col-l:go_col+r,1]*dis_msk_ref[go_row-t:go_row+d,go_col-l:go_col+r]).sum()/(msk_sum)

                    dis_msk[go_row,go_col] += 1
        search_mask_have_hole_count += 1
    if(write_to_step3):
        cv2.imwrite("step3_apply_flow_result/%s-3a1-I1-patch.bmp"%(name), dis_img.astype(np.uint8))
        cv2.imwrite("step3_apply_flow_result/%s-3a4-Mask-patch.bmp"%(name), dis_msk*70)
        np.save("step3_apply_flow_result/%s-3a4-Mask-patch"%(name),dis_msk)
        
        np.save("step3_apply_flow_result/%s-3b-rec_mov_map"%(name),rec_mov.astype(np.float32))
        rec_mov_visual = method2(rec_mov[:,:,0],rec_mov[:,:,1],1)
        cv2.imwrite("step3_apply_flow_result/%s-3b-rec_mov_visual.jpg"%(name), rec_mov_visual)
        print("dis_msk.max()",dis_msk.max())
    

    return dis_img.copy(), rec_mov.copy()

if(__name__=="__main__"):
    import time
    start_time = time.time()
    # img = cv2.imread("step1_pabe/book.jpg")
    img_list  = get_dir_img("step1_page")
    img_amount = len(img_list)
    move_list = get_dir_move("step2_flow_build/move_map")


    Check_dir_exist_and_build("step3_apply_flow_result")
    start_index = 0 ### 這是用在 如果不小心中斷，可以用這設定從哪裡開始
    for i, move_map in enumerate(move_list[start_index:]):
        img = img_list[np.random.randint(img_amount)]
        apply_start_time = time.time()
        name = "%06i"%(i+start_index)
        dis_img, rec_mov = apply_move(img, move_map, name, write_to_step3=True)
        print("%06i process 1 mesh cost time:"%(i+start_index), "%.3f"%(time.time()-apply_start_time), "total_time:", "%.3f"%(time.time()-start_time))


# rec_img, dis_mov = apply_move(dis_img, rec_mov, name="rec")

#cv2.waitKey()
#cv2.destroyAllWindows()
