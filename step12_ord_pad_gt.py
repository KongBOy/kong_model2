from step0_access_path import access_path
from util import get_dir_certain_img, get_max_move_xy_from_certain_move
from build_dataset_combine import Check_dir_exist_and_build_new_dir
import numpy as np 
import cv2


result_dir = access_path+"step12_ord_pad_gt"
Check_dir_exist_and_build_new_dir( result_dir )

imgs = get_dir_certain_img("step3_apply_flow_result","1-I.bmp")


### 本來想要padding，但後來發現好像不用，所以其實直接拿step3裡的 1-I.bmp也行喔！
### 初始化各個會用到的canvas
row=256
col=256
move_x_max, move_y_max = get_max_move_xy_from_certain_move("step3_apply_flow_result","2-q.npy")

dis_h = int( np.around(move_y_max + row + move_y_max) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
dis_w = int( np.around(move_x_max + col + move_x_max) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
dis_img  = np.zeros(shape=(dis_h,dis_w,3), dtype=np.uint8)
dis_imgs = np.tile(dis_img,(2000,1,1,1))

move_x_max = int(move_x_max)
move_y_max = int(move_y_max)
dis_imgs[:, move_y_max:move_y_max+row, move_x_max:move_x_max+col, :] = imgs

for i, dis_img in enumerate(dis_imgs):
    # cv2.imwrite(result_dir + "/" + "%06i_img-pad.bmp"%i, dis_img)
    cv2.imwrite(result_dir + "/" + "%06i_img.bmp"%i, imgs[i])
    # cv2.waitKey()
    # cv2.destroyAllWindows()


#dst_x = go_col + int(move_map[go_row,go_col,0] + move_x_max) ### 現在的起點是(move_x_max, move_y_max)，所以要位移一下
#dst_y = go_row + int(move_map[go_row,go_col,1] + move_y_max) ### 現在的起點是(move_x_max, move_y_max)，所以要位移一下