from util import get_dir_move, get_max_move_xy_from_numpy, get_dir_certain_move, get_dir_certain_img, method2, get_max_move_xy_from_certain_move
import numpy as np 
import cv2


dis_imgs = get_dir_certain_img("step3_apply_flow_result","3a1-I1-patch")
dis_img = dis_imgs[0]
moves = get_dir_certain_move("step3_apply_flow_result","2-q")
move_map = moves[0]

max_move_x, max_move_y = get_max_move_xy_from_certain_move("step3_apply_flow_result","2-q")

name = "000000"

row, col = move_map.shape[:2]

rec_img = np.zeros(shape=(row,col,3))

for go_row in range(row):
    for go_col in range(col):
        x = int(go_col + move_map[go_row, go_col, 0] + max_move_x) 
        y = int(go_row + move_map[go_row, go_col, 1] + max_move_y)  
        rec_img[go_row, go_col,:] = dis_img[y, x,:]

        
cv2.imshow("rec_img", rec_img.astype(np.uint8))
cv2.imwrite("rec_img.png", rec_img.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()




### 原來要用 inverse flow 的版本還是留一下好了
# row, col = img.shape[:2]

# rec_img = np.zeros_like(img)

# for go_row in range(row):
#     for go_col in range(col):
#         x = int(go_col + move_map[go_row, go_col, 0] + move_x_min) 
#         y = int(go_row + move_map[go_row, go_col, 1] + move_y_min)  

#         rec_img[go_row, go_col,:] = dis_img[y, x,:]
# cv2.imshow("rec_img", rec_img)
# cv2.imwrite("rec_img.png", rec_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
