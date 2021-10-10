from step11_a0_see_base import See_info


import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name, move_dir_certain_file

import datetime
# import pdb

'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''

class See_mask(See_info):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_mask, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                          ..._write_dir
        """
        self.old1_mask_read_dir   = self.see_read_dir
        self.old1_mask_write_dir  = self.see_write_dir
        ##########################################################################################
        # self.mask_read_dir      = self.see_read_dir  + "/1_mask"  ### 之後也許東西多起來會需要獨立資料夾， 先寫起來放
        # self.mask_write_dir     = self.see_write_dir + "/1_mask"  ### 之後也許東西多起來會需要獨立資料夾， 先寫起來放
        self.mask_read_dir      = self.see_read_dir
        self.mask_write_dir     = self.see_write_dir

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_mask_dir(print_msg=True)

    ### 先寫起來放著
    def Change_mask_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_mask_dir, Current See:{self.see_name}")
        move_dir_certain_file(self.old1_mask_read_dir,  certain_word="epoch", certain_ext=".bmp", dst_dir=self.mask_read_dir,  print_msg=print_msg)
        move_dir_certain_file(self.old1_mask_write_dir, certain_word="epoch", certain_ext=".bmp", dst_dir=self.mask_write_dir, print_msg=print_msg)

    ### 給下一步 bm_rec 用的
    def get_mask_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.mask_names  = get_dir_certain_file_name(self.mask_read_dir , certain_word="epoch", certain_ext=".bmp")
        self.mask_read_paths  = [self.mask_read_dir + "/" + name for name in self.mask_names]  ### 目前還沒用到～　所以也沒有寫 write_path 囉！

        self.mask_amount = len(self.mask_read_paths)
        self.trained_epoch  = self.mask_amount - 1  ### 去掉epoch0

# class See(See_flow_visual, See_rec_metric, See_mask):
# class See(See_flow_visual, See_npy_to_npz, See_bm_rec, See_rec_metric, See_mask):
#     def __init__(self, result_read_dir, result_write_dir, see_name):
#         super(See, self).__init__(result_read_dir, result_write_dir, see_name)



if(__name__ == "__main__"):
    from step0_access_path import result_read_path, result_write_path
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_3b_4-20210306_231628-flow_unet-ch32_bn_16", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_005-train")
    # try_npy_to_npz.npy_to_npz_comapre()
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)

    test_see = See(result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-testest", result_write_dir=result_write_path + "result/5_14_flow_unet/type8_blender_os_book-testest", see_name="see_001-real")
    test_see.Calculate_SSIM_LD(epoch=0, single_see_core_amount=8)
