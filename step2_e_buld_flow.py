import sys 
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build
import numpy as np 
import time
from step0_access_path import data_access_path
from step2_a_distort_curl_and_fold import distort_rand
from step2_a_distort_page_and_pers import distort_just_perspect, distort_just_page


class temp_dataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data  = test_data
        self.train_data_amount = len(self.train_data)
        self.test_data_amount  = len(self.test_data)
    
    def pers_augment(self, pers_moves):
        pers_amount = len(pers_moves)
        self.train_data = np.tile(self.train_data, (pers_amount,1,1,1))
        self.test_data  = np.tile(self.test_data,  (pers_amount,1,1,1))

        aug_train_pers_moves = np.tile(pers_moves, (self.train_data_amount,1,1,1))
        aug_test_pers_moves  = np.tile(pers_moves, (self.test_data_amount,1,1,1))

        self.train_data += aug_train_pers_moves
        self.test_data  += aug_test_pers_moves 

        return self


# fold_moves = np.tile(fold_moves, (pers_amount,1,1,1))
# page_moves = np.tile(page_moves, (pers_amount,1,1,1))

### 平滑多一點 384*256_1500張
row=384
col=256
amount=55

dst_dir = "這裡隨便，因為write_npy這裡設False"
curl_moves = distort_rand        (dst_dir=dst_dir, start_index=amount*0, amount=amount, row=row, col=col,distort_time=1, curl_probability=1.0, move_x_thresh=40, move_y_thresh=55, smooth=True, write_npy=False )
fold_moves = distort_rand        (dst_dir=dst_dir, start_index=amount*1, amount=amount, row=row, col=col,distort_time=1, curl_probability=0.0, move_x_thresh=40, move_y_thresh=55, smooth=True, write_npy=False )
page_moves = distort_just_page   (dst_dir=dst_dir, start_index=amount*2, row=384, col=256, repeat=1, write_npy=False ) ### repeat是為了要讓 同種style 有repeat種 頁面內容

pers_moves = distort_just_perspect(dst_dir=dst_dir, start_index=    0   , row=384, col=256, write_npy=False)

curl_train_amount = 50
fold_train_amount = 50
page_train_amount = 50
curl_db = temp_dataset(curl_moves[:curl_train_amount], curl_moves[curl_train_amount:]).pers_augment(pers_moves)
fold_db = temp_dataset(fold_moves[:fold_train_amount], fold_moves[fold_train_amount:]).pers_augment(pers_moves)
page_db = temp_dataset(page_moves[:page_train_amount], page_moves[page_train_amount:]).pers_augment(pers_moves)

train_moves = np.concatenate([curl_db.train_data, fold_db.train_data, page_db.train_data])
test_moves  = np.concatenate([curl_db.test_data,  fold_db.test_data,  page_db.test_data ])


dst_dir = "step2_build_flow_h=384,w=256_smooth-curl+fold_and_page"
Check_dir_exist_and_build(data_access_path + dst_dir + "/"+"move_maps")

index = 0
for move in train_moves:
    np.save(data_access_path + dst_dir + "/" + "move_maps/%06i_train"%(index), move) ### 把move_map存起來，記得要轉成float32！
    index += 1

for move in test_moves:
    np.save(data_access_path + dst_dir + "/" + "move_maps/%06i_test"%(index), move) ### 把move_map存起來，記得要轉成float32！
    index += 1