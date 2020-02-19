import cv2
import numpy as np 
from build_dataset_combine import Check_dir_exist_and_build
from util import method2
from step2_apply_mov2ord_img_sucess import apply_move
Check_dir_exist_and_build("step5_result_result")

for i in range(10):
    dis_img = cv2.imread("step5_result/%02i_ord_distorted_img.jpg"%i)
    rec_move_map = np.load("step5_result/%02i_rec_move_map.npy"%i)
    rec_move_map_visual = method2(rec_move_map[...,0,], rec_move_map[...,1],color_shift=2)
    cv2.imwrite("step5_result_result/%02i_5a-dis_img.jpg"%i,dis_img.astype(np.uint8))
    cv2.imwrite("step5_result_result/%02i_5b-rec_move_map_visual.jpg"%i,rec_move_map_visual)
    # cv2.waitKey(0)

    rec_img , rec_recheck_mov = apply_move(dis_img, rec_move_map)
    cv2.imwrite("step5_result_result/%02i_5c-rec_img.jpg"%i,rec_img.astype(np.uint8))