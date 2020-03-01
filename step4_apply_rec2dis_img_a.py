from step0_access_path import access_path
import cv2
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build
from util import method2, get_dir_certain_img, get_dir_certain_move
from step3_apply_mov2ord_img import apply_move

# access_path = "D:/Users/user/Desktop/db/" ### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！

dis_imgs = get_dir_certain_img( access_path+"step3_apply_flow_result", "3a1-I1-patch.bmp")
rec_movs = get_dir_certain_move(access_path+"step3_apply_flow_result", "3b-rec_mov_map.npy")


result_dir = access_path+"step4_apply_rec_flow_result"
Check_dir_exist_and_build(result_dir)


data_amount = len(dis_imgs)
for i in range(data_amount):
    print("doing %06i test image"%i)
    ### 為了方便看 整個還原的流程，把 dis_img 和 rec_move_map 也寫一份進來
    rec_move_map_visual = method2(rec_movs[i][..., 0], rec_movs[i][...,1], color_shift=1)
    cv2.imwrite( result_dir + "/" + "%06i-4a-dis_img.jpg"%i,dis_imgs[i].astype(np.uint8))
    cv2.imwrite( result_dir + "/" + "%06i-4b-rec_move_map_visual.jpg"%i,rec_move_map_visual)
    # cv2.waitKey(0)

    # ### 真的apply進去
    # rec_img , rec_debug_move = apply_move(dis_imgs[i], rec_movs[i])
    # cv2.imwrite( result_dir + "/" + "%06i-4c-rec_img.jpg"%i,rec_img.astype(np.uint8))


    ### 真的apply進去，要看 base_xy版
    rec_img , rec_debug_move, move_x_min, move_y_min = apply_move(dis_imgs[i], rec_movs[i], return_base_xy=True)
    print(move_x_min, move_y_min)
    cv2.imwrite( result_dir + "/" + "%06i-4c-rec_img.jpg"%i,rec_img.astype(np.uint8))

    ### rec_debug_move 視覺化 並 存起來
    rec_debug_move_bgr = method2(rec_debug_move[...,0], rec_debug_move[...,1], 1)
    cv2.imwrite(result_dir + "/%06i-4d-rec_debug_move_map.bmp"%i, rec_debug_move_bgr) ### 視覺化的結果存起來
    np.save(result_dir + "/%06i-4e_rec_debug_move_map"%i, rec_debug_move) ### 存起來