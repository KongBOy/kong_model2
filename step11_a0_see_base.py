import sys
sys.path.append("kong_util")
from kong_util.util import get_dir_certain_file_names

from step06_a_datas_obj import Range

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
    def __init__(self, result_obj, see_name):
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.see_name = see_name

        self.result_obj = result_obj
        self.result_read_dir  = result_obj.result_read_dir
        self.result_write_dir = result_obj.result_write_dir

        self.see_read_dir  = self.result_read_dir  + "/" + self.see_name
        self.see_write_dir = self.result_write_dir + "/" + self.see_name

    def get_name_savely(self, search_dir_1, search_dir_2=".", certain_word=".", certain_ext=".", print_msg=False):
        names = get_dir_certain_file_names(search_dir_1, certain_word=certain_word, certain_ext=certain_ext, print_msg=print_msg)
        if(len(names) > 0): return names[-1]

        ### 如果 search_dir_1 找不到就去 search_dir_2 找
        names = get_dir_certain_file_names(search_dir_2, certain_word=certain_word, certain_ext=certain_ext, print_msg=print_msg)
        if(len(names) > 0): return names[-1]

        ### 如果 search_dir_1, 2 就代表真的找不到了
        if(print_msg):
            print(f"{search_dir_1} 找不到 {certain_word}{certain_ext} 字眼的檔案")
            print(f"{search_dir_2} 找不到 {certain_word}{certain_ext} 字眼的檔案")
            return None

    def get_path_savely(self, search_dir_1, search_dir_2=".", certain_word=".", certain_ext=".", print_msg=False):
        names = get_dir_certain_file_names(search_dir_1, certain_word=certain_word, certain_ext=certain_ext, print_msg=print_msg)
        if(len(names) > 0): return f"{search_dir_1}/{names[-1]}"

        ### 如果 search_dir_1 找不到就去 search_dir_2 找
        names = get_dir_certain_file_names(search_dir_2, certain_word=certain_word, certain_ext=certain_ext, print_msg=print_msg)
        if(len(names) > 0): return f"{search_dir_2}/{names[-1]}"

        ### 如果 search_dir_1, 2 就代表真的找不到了
        if(print_msg):
            print(f"{search_dir_1} 找不到 {certain_word}{certain_ext} 字眼的檔案")
            print(f"{search_dir_2} 找不到 {certain_word}{certain_ext} 字眼的檔案")
            return None

    def get_see_base_info(self):
        ''' 我有把 see_file_amount 這個attr拿掉囉！ 因為 應用的時候需要一直 -1, -2, -3 很煩， -1, -2, -3 分別代表什麼數字都直接定義清楚這樣子拉！'''
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        # self.see_jpg_names        = get_dir_certain_file_names(self.see_read_dir, certain_word=".jpg")
        # self.in_img_path          = self.see_read_dir + "/" + self.see_jpg_names[0]
        # print("self.see_read_dir:", self.see_read_dir)
        self.in_img_path  = self.get_path_savely(self.see_read_dir, certain_word="in_img")

        self.dis_img_path = self.get_path_savely(self.see_read_dir, certain_word="dis_img")
        if(self.dis_img_path is None): self.dis_img_path = self.get_path_savely(self.see_read_dir, certain_word="in_img")  ### 為了相容以前沒有寫 dis_img 的 result

        self.rec_hope_path    = self.get_path_savely(self.see_read_dir, certain_word="rec_hope", certain_ext=".jpg")  ### search_dir_1 設 self.flow_v_write_dir 是因為 有可能 npy_to_npz 完 馬上就接著做
        if(self.rec_hope_path is None): self.rec_hope_path = self.get_path_savely(self.see_read_dir, certain_word="rec_hope", certain_ext=".png")  ### 為了相容以前沒有寫 dis_img 的 result
        # print("self.in_img_path" , self.in_img_path)
        # print("self.dis_img_path", self.dis_img_path)
        # self.in_img_names = get_dir_certain_file_names(self.see_read_dir, certain_word="in_img")
        # if(len(self.in_img_names) > 0): self.in_img_path = f"{self.see_read_dir}/{self.in_img_names[0]}"
        # else: print(f"{self.see_read_dir} 找不到 gt_flow.npz")


        ### 不確定合不合理， 目前就先暫時用 flow_ep_jpg_names -1(去掉epoch0) 來代表 現在已經train了幾個epochs囉！ 即 trained_epochs， flow_ep_jpg_names 用了覺得不合理再換吧～
        # self.trained_epoch       = self.flow_ep_jpg_amount - 1  ### 去掉epoch0
