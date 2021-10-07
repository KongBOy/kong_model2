import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name


'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''

class See_info:
    '''
    See 是 最直接、最基本 model output的東西，在各個model裡面都應該有寫 自己的 generate_see
    而這邊只是 讀取 training 過程中生成的 See 這樣子囉~~
    '''
    def __init__(self, result_read_dir, result_write_dir, see_name):
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.see_name = see_name

        self.result_read_dir  = result_read_dir
        self.result_write_dir = result_write_dir

        self.see_read_dir  = self.result_read_dir  + "/" + self.see_name
        self.see_write_dir = self.result_write_dir + "/" + self.see_name

        self.in_use_range = "0~1"
        self.gt_use_range = "0~1"

    def get_see_base_info(self):
        ''' 我有把 see_file_amount 這個attr拿掉囉！ 因為 應用的時候需要一直 -1, -2, -3 很煩， -1, -2, -3 分別代表什麼數字都直接定義清楚這樣子拉！'''
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.see_jpg_names        = get_dir_certain_file_name(self.see_read_dir, certain_word=".jpg")
        self.in_img_path          = self.see_read_dir + "/" + self.see_jpg_names[0]
        self.gt_flow_v_path       = self.see_read_dir + "/" + self.see_jpg_names[1]
        self.rec_hope_path        = self.see_read_dir + "/" + self.see_jpg_names[2]

        ### 因為 See_flow_visual 和 See_bm_rec 要用到， 所以從 See_flow_visual 提升上來囉！
        self.see_flow_epoch_jpg_names      = get_dir_certain_file_name(self.see_read_dir, certain_word="epoch", certain_ext=".jpg")
        self.flow_epoch_jpg_read_paths = [self.see_read_dir + "/" + epoch_jpg_name for epoch_jpg_name in self.see_flow_epoch_jpg_names]  ### 沒有 write_paths，因為這是 predict_flow_visual， 是從model 訓練過程產生的， 後處理不會產生！ 就是不會做ewrite的動作囉！就不用write_path拉！
        self.flow_epoch_jpg_amount     = len(self.see_flow_epoch_jpg_names)

        ### 不確定合不合理， 目前就先暫時用 see_flow_epoch_jpg_names -1(去掉epoch0) 來代表 現在已經train了幾個epochs囉！ 即 trained_epochs， see_flow_epoch_jpg_names 用了覺得不合理再換吧～
        self.trained_epoch       = self.flow_epoch_jpg_amount - 1  ### 去掉epoch0
