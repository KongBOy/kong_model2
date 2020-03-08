from step0_access_path import access_path
from util import get_dir_certain_img, get_max_move_xy_from_certain_move
from build_dataset_combine import Check_dir_exist_and_build_new_dir
import numpy as np 
import cv2

pad_result = True ### pad 和 沒pad 的差別只有：1.存結果的地方不同、2.unet_rec_img 有沒有pad成 H,W 的大小

#######################################################################
step12_result_dir = access_path+"step12_ord_pad_gt"
if(pad_result):
    step12_result_dir = access_path+"step12_ord_pad_gt_pad"
Check_dir_exist_and_build_new_dir( step12_result_dir )


### 本來想要padding，但後來發現好像不用，所以其實直接拿step3裡的 1-I.bmp也行喔！
### 但後來老師想用dis_img加入rect2訓練，所以還是需要pad一下，step3有空改寫的時候要多存pad的過程喔~
imgs = get_dir_certain_img(access_path+"step3_apply_flow_result","1-I.bmp")


if(pad_result):
    ### 初始化各個會用到的canvas
    move_x_max, move_y_max = get_max_move_xy_from_certain_move(access_path+"step3_apply_flow_result","2-q.npy")
    row=256
    col=256
    img_pad_h = int( np.around(move_y_max + row + move_y_max) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    img_pad_w = int( np.around(move_x_max + col + move_x_max) ) ### np.around 是四捨五入，然後因為要丟到shape裡所以轉int
    img_pad  = np.zeros(shape=(img_pad_h,img_pad_w,3), dtype=np.uint8)
    img_pads = np.tile(img_pad,(2000,1,1,1))

    int_move_x_max = int(move_x_max)
    int_move_y_max = int(move_y_max)
    img_pads[:, int_move_y_max:int_move_y_max+row, int_move_x_max:int_move_x_max+col, :] = imgs
    imgs = img_pads

for i, img in enumerate(imgs):
    cv2.imwrite(step12_result_dir + "/" + "%06i_img.bmp"%i, img)
    
    # cv2.waitKey()
    # cv2.destroyAllWindows()

