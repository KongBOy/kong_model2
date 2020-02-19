import cv2
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build
from util import method2
from step2_apply_mov2ord_img_sucess import apply_move
Check_dir_exist_and_build("step4_result")

for i in range(100):
    dis_img = cv2.imread("step3_result/%06i-3b-I1-patch.bmp"%i)
    rec_move_map = np.load("step3_result/%06i-4-rec_mov_map.npy"%i)
    rec_move_map_visual = method2(rec_move_map[...,0,], rec_move_map[...,1],color_shift=2)
    cv2.imwrite("step4_result/%06i-5a-dis_img.jpg"%i,dis_img.astype(np.uint8))
    cv2.imwrite("step4_result/%06i-5b-rec_move_map_visual.jpg"%i,rec_move_map_visual)
    # cv2.waitKey(0)

    rec_img , rec_recheck_mov = apply_move(dis_img, rec_move_map)
    cv2.imwrite("step4_result/%06i-5c-rec_img.jpg"%i,rec_img.astype(np.uint8))
