import sys 
sys.path.append("kong_util")


### 把 see 000~031 都做成影片
from video_from_img import Video_combine_from_imgs, Video_combine_from_dir
from build_dataset_combine import Save_as_jpg
for i in range(32):
    ord_dir = r"F:\Users\Lin_server\Desktop\0 data_dir\result\type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2\see-%03i\matplot_visual"%i
    Save_as_jpg(ord_dir, ord_dir,delete_ord_file=True)
    Video_combine_from_dir(ord_dir, ord_dir, "combine_jpg.avi")


# import numpy as np 
# from util import matplot_visual_one_row_imgs
# import cv2

# ord_dir = r"F:\Users\Lin_server\Desktop\0 data_dir\result\type5c-real_have_see-no_bg-gt-gray3ch_20200428-011344_model5_rect2\see-%03i"%(0)
# from util import get_dir_img
# imgs = get_dir_img(ord_dir, float_return=False)

# img1 = np.ones(shape=(500,404,3), dtype = np.uint8)
# img2 = np.ones(shape=(472,304,3), dtype = np.uint8)*125
# img3 = np.ones(shape=(384,256,3), dtype = np.uint8)*125
# titles = ["distorted_img","distorted_img"]
# imgs = [imgs[0],imgs[1],imgs[2]]

# # cv2.imshow("123", img1)
# # cv2.imshow("456", img2)
# matplot_visual_one_row_imgs(titles, imgs)
# cv2.waitKey()
