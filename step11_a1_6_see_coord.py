from step11_a0_see_base import See_info


import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_names, move_dir_certain_file

import datetime
# import pdb

'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''

class See_coord(See_info):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_coord, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                          ..._write_dir
        """
        self.old1_coord_read_dir   = self.see_read_dir
        self.old1_coord_write_dir  = self.see_write_dir
        ##########################################################################################
        # self.coord_read_dir      = self.see_read_dir  + "/1_coord"  ### 之後也許東西多起來會需要獨立資料夾， 先寫起來放
        # self.coord_write_dir     = self.see_write_dir + "/1_coord"  ### 之後也許東西多起來會需要獨立資料夾， 先寫起來放
        self.coord_read_dir      = self.see_read_dir
        self.coord_write_dir     = self.see_write_dir

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_coord_dir(print_msg=True)

    ### 先寫起來放著
    def Change_coord_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_coord_dir, Current See:{self.see_name}")
        move_dir_certain_file(self.old1_coord_read_dir,  certain_word="epoch", certain_ext=".bmp", dst_dir=self.coord_read_dir,  print_msg=print_msg)
        move_dir_certain_file(self.old1_coord_write_dir, certain_word="epoch", certain_ext=".bmp", dst_dir=self.coord_write_dir, print_msg=print_msg)

    ### 給下一步 bm_rec 用的
    def get_coord_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.coord_names  = get_dir_certain_file_names(self.coord_read_dir , certain_word="epoch", certain_ext=".bmp")
        self.coord_read_paths  = [self.coord_read_dir + "/" + name for name in self.coord_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.coord_amount = len(self.coord_read_paths)
        self.trained_epoch  = self.coord_amount - 1  ### 去掉epoch0

# class See(See_flow_visual, See_rec_metric, See_mask):
# class See(See_flow_visual, See_npy_to_npz, See_bm_rec, See_rec_metric, See_mask):
#     def __init__(self, result_read_dir, result_write_dir, see_name):
#         super(See, self).__init__(result_read_dir, result_write_dir, see_name)
