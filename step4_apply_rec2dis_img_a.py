import cv2
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build
from util import method2, get_dir_certain_img, get_dir_certain_move
from step3_apply_mov2ord_img import apply_move


dis_imgs = get_dir_certain_img( "step3_apply_flow_result", "3a1-I1-patch.bmp")
rec_movs = get_dir_certain_move("step3_apply_flow_result", "3b-rec_mov_map.npy")


result_dir = "step4_apply_rec_flow_result"
Check_dir_exist_and_build(result_dir)


data_amount = len(dis_imgs)
for i in range(data_amount):
    ### 為了方便看 整個還原的流程，把 dis_img 和 rec_move_map 也寫一份進來
    rec_move_map_visual = method2(rec_movs[i][..., 0], rec_movs[i][...,1], color_shift=1)
    cv2.imwrite( result_dir + "/" + "%06i-4a-dis_img.jpg"%i,dis_imgs[i].astype(np.uint8))
    cv2.imwrite( result_dir + "/" + "%06i-4b-rec_move_map_visual.jpg"%i,rec_move_map_visual)
    # cv2.waitKey(0)

    ### 真的apply進去
    rec_img , rec_recheck_mov = apply_move(dis_imgs[i], rec_movs[i])
    cv2.imwrite( result_dir + "/" + "%06i-4c-rec_img.jpg"%i,rec_img.astype(np.uint8))
