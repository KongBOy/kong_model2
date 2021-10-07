from step11_a0_see_base import See_info

from step0_access_path import JPG_QUALITY, CORE_AMOUNT_NPY_TO_NPZ, CORE_AMOUNT_BM_REC_VISUAL, CORE_AMOUNT_FIND_LTRD_AND_CROP, CORE_AMOUNT_SAVE_AS_JPG
from step0_access_path import Syn_write_to_read_dir

import sys
sys.path.append("kong_util")
from util import get_dir_certain_file_name, move_dir_certain_file
from build_dataset_combine import Check_dir_exist_and_build
from multiprocess_util import multi_processing_interface

import time
import numpy as np
from tqdm import tqdm
import os

import datetime
# import pdb

'''
繼承關係(我把它設計成 有一種 做完 前面才能做後面的概念)：
See_info -> See_npy_to_npz -> See_bm_rec -> See_rec_metric
          ↘ See_flow_visual

後來 覺得不需要 用繼承來限定 做完 前面坐後面的概念， 覺得多餘， 所以 統一都繼承 See_info，
這樣也可以See到各個檔案
'''

class See_npy_to_npz(See_info):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See_npy_to_npz, self).__init__(result_read_dir, result_write_dir, see_name)
        """
        __init__：放 Dir：..._read_dir
                         ..._write_dir
        """
        self.old1_see_npz_read_dir   = self.see_read_dir
        self.old1_see_npz_write_dir  = self.see_write_dir
        self.old2_see_npz_read_dir   = self.see_read_dir  + "/npz"
        self.old2_see_npz_write_dir  = self.see_write_dir + "/npz"

        ### 給 npy轉npz 用的
        self.npy_read_dir      = self.see_read_dir
        self.npy_write_dir     = self.see_write_dir

        ### 給 bm_rec 用的
        self.npz_read_dir       = self.see_read_dir  + "/1_npz"
        self.npz_write_dir      = self.see_write_dir + "/1_npz"

        ### 資料夾的位置有改 保險起見加一下，久了確定 放的位置都更新了 可刪這行喔
        # self.Change_npz_dir(print_msg=True)

    def Change_npz_dir(self, print_msg=False):  ### Change_dir 寫這 而不寫在 外面 是因為 see資訊 是要在 class See 裡面才方便看的到，所以 在這邊寫多個function 比較好寫，外面傳參數 要 exp.result.sees[]..... 很麻煩想到就不想寫ˊ口ˋ
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Change_npz_dir, Current See:{self.see_name}")
        # move_dir_certain_file(self.old1_see_npz_read_dir,  certain_word="epoch", certain_ext=".npz", dst_dir=self.npz_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old1_see_npz_write_dir, certain_word="epoch", certain_ext=".npz", dst_dir=self.npz_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old1_see_npz_read_dir,  certain_word="0b-gt", certain_ext=".npz", dst_dir=self.npz_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old1_see_npz_write_dir, certain_word="0b-gt", certain_ext=".npz", dst_dir=self.npz_write_dir, print_msg=print_msg)
        # move_dir_certain_file(self.old2_see_npz_read_dir,  certain_word="epoch", certain_ext=".npz", dst_dir=self.npz_read_dir,  print_msg=print_msg)
        # move_dir_certain_file(self.old2_see_npz_write_dir, certain_word="epoch", certain_ext=".npz", dst_dir=self.npz_write_dir, print_msg=print_msg)

    ### 給自己 npy轉npz 用的
    def get_npy_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        self.npy_names            = get_dir_certain_file_name(self.npy_read_dir, certain_word=".npy")
        self.npy_read_paths       = [self.npy_read_dir  + "/" + npy_name for npy_name in self.npy_names]  ### 沒有 write_paths，因為式 npy轉npz， 不會有寫npy的動作， 雖然下面的 compare 會寫一點npy， 但也因為 有用 .replace() 所以用 see_npy_name.replace() 較保險這樣子！
        self.npy_amount           = len(self.npy_read_paths)

    ### 給下一步 bm_rec 用的
    def get_npz_info(self):
        """
        get_info：放 ..._names
                      ├ ..._read_paths
                      └ ..._write_paths
                     file_amount
        """
        ### 有包含 gt_flow 的 list喔！ 第一個 放的是 gt_flow
        print("self.npz_read_dir~~~~~~~~~~~~~~~~~", self.npz_read_dir)
        self.npz_names            = get_dir_certain_file_name(self.npz_read_dir, certain_word=".npz")
        self.flow_gt_npz_path         = self.npz_read_dir + "/" + self.npz_names[0]

        ### 不包含 gt_flow 的 list喔！
        self.npz_epoch_names      = get_dir_certain_file_name(self.npz_read_dir, certain_word="epoch", certain_ext=".npz")
        self.npz_epoch_read_paths = [self.npz_read_dir + "/" + epoch_npz_name for epoch_npz_name in self.npz_epoch_names]    ### 沒有 write_paths，同上 ，既然已經沒有 self.see_npz_write_paths， 當然更不會有 self.see_npz_epoch_write_paths 拉！

        self.npz_epoch_amount     = len(self.npz_epoch_read_paths)

    def Npy_to_npz(self, single_see_core_amount=8, see_print_msg=False):   ### 因為有刪東西的動作，覺得不要multiprocess比較安全~~
        """
        把 See 資料夾內的.npy改存成.npz，存完會把.npy刪除喔～
        """
        ### See_method 第零ａ部分：顯示開始資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: doing Npy_to_npz, Current See:{self.see_name}")
        start_time = time.time()

        ### See_method 第一部分：建立資料夾：不能 build_new_dir，因為原本的 .npy 因為容量太大 ， 轉完 .npz， 最後會把.npy刪掉！ 因此如果 第二次以上執行 就不會有.npy了 無法重建.npz！ 所以不能把 .npz資料夾刪掉重建喔！
        Check_dir_exist_and_build(self.npz_write_dir)

        ### See_method 第二部分：取得see資訊
        self.get_npy_info()


        ### See_method 第三部分：主要做的事情在這裡， 如果要有想設計平行處理的功能 就要有 1.single_see_core_amount 和 2.下面的if/elif/else 和 3._see_method 前兩個參數要為 start_index, task_amount 相關詞喔！
        if(self.npy_amount > 0):
            if(single_see_core_amount == 1):  ### single_see_core_amount 大於1 代表 單核心跑， 就重新導向 最原始的function囉 把 see內的任務 依序完成！
                self._npy_to_npz(start_index=0, amount=self.npy_amount)
            elif(single_see_core_amount  > 1):  ### single_see_core_amount 大於1 代表 多核心跑， 丟進 multiprocess_interface 把 see內的任務 切段 平行處理囉
                multi_processing_interface(core_amount=CORE_AMOUNT_NPY_TO_NPZ, task_amount=self.npy_amount, task=self._npy_to_npz, print_msg=see_print_msg)  ### 因為和 bm_rec 的動作包一起， 外面指定的 single_see_core_amount 是比較適合 bm_rec 的， 所以 npy_to_npz 就用 在 step0 統一指定的 CORE數囉！
            else:
                print("single_see_core_amount 設定錯誤， 需要 >= 1 的數字才對喔！ == 1 代表see內任務單核心跑， > 1 代表see內任務多核心跑")

        ### See_method 第四部分：後處理～沒事情就空白拉

        ### See_method 第五部分：如果 write 和 read 資料夾不同，把 write完的結果 同步回 read資料夾喔！
        if(self.see_write_dir != self.see_read_dir):  ### 因為接下去的任務需要 此任務的結果， 如果 read/write 資料夾位置不一樣， write完的結果 copy 一份 放回read， 才能讓接下去的動作 有 東西 read 喔！
            Syn_write_to_read_dir(write_dir=self.npz_write_dir, read_dir=self.npz_read_dir, build_new_dir=False)

        ### See_method 第零b部分：顯示結束資訊 和 計時
        print(datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S"), f"See level: finish Npy_to_npz, Current See:{self.see_name}, cost time:{time.time() - start_time}")

    ### See_method 第三部分：主要做的事情在這裡
    def _npy_to_npz(self, start_index, amount):
        for see_npy_name in tqdm(self.npy_names[start_index:start_index + amount]):  ### 因為有用 .replace()， 對see_npy_name.replace() 較保險， 所以這邊用 see_npy_name 而不用 see_npy_path！
            npy = np.load(self.see_read_dir + "/" + see_npy_name)
            np.savez_compressed(self.npz_write_dir + "/" + see_npy_name.replace(".npy", ".npz"), npy)
            os.remove(self.see_read_dir + "/" + see_npy_name)
            # print(self.see_read_dir + "/" + see_npy_name, "delete ok")
            # npz = np.load(self.see_read_dir + "/" + see_npy_name.replace(".npy", ".npz"))  ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
            # print((npy - npz["arr_0"]).sum())                                              ### 已用這兩行確認 npz 壓縮式 無失真的！值完全跟npy一樣喔！
        ### 不要想 邊生圖 邊 npy轉npz了，原因寫在 _Draw_bm_rec_matplot_visual 上面

    def npy_to_npz_comapre(self):
        self.get_npy_info()

        ### load_3_load_50_npy
        start_time = time.time()
        for go_name, see_npy_path in enumerate(self.npy_read_paths):
            np.load(see_npy_path)   ### 344 MB
        load_3_time = time.time() - start_time
        print("load_3_load_50_npy ok")

        ### save_3_save_50_npy
        npys = []
        for go_name, see_npy_path in enumerate(self.npy_read_paths):
            npys.append(np.load(see_npy_path))   ### 344 MB
        start_time = time.time()
        for go_name, npy in enumerate(npys):
            np.save(self.see_npy_paths[go_name], npy)
        save_3_time = time.time() - start_time
        print("save_3_save_50_npy ok")

        ### save_2_save_50_npz
        start_time = time.time()
        for go_name, npy in enumerate(npys):
            np.savez_compressed(self.see_write_dir + "/" + self.npy_names[go_name].replace(".npy", ""), npy)
        save_2_time = time.time() - start_time
        print("save_2_save_50_npz ok")

        ### load_2_load_50_npz
        start_time = time.time()
        for go_name, see_npy_name in enumerate(self.npy_names):
            np.load(self.see_read_dir + "/" + see_npy_name.replace(".npy", ".npz"))   ### 344 MB
        load_2_time = time.time() - start_time
        print("load_2_load_50_npz ok")

        ### save_1_save_1_npz_conatin_50npy
        start_time = time.time()
        np.savez_compressed(self.see_write_dir + "/" + "000_try_npz", np.array(npys))
        save_1_time = time.time() - start_time
        print("save_1_save_1_npz_conatin_50npy ok")

        ### load_1_load_1_npz_conatin_50npy
        start_time = time.time()
        big_npz = np.load(self.see_read_dir + "/" + "000_try_npz.npz")
        print(big_npz["arr_0"].shape)   ### 小心！要有使用他，才會真的去load資料喔！
        load_1_time = time.time() - start_time
        print("load_1_load_1_npz_conatin_50npy ok")
        print("")
        os.remove(self.see_read_dir + "/" + "000_try_npz.npz")  ### 只是用來看讀取寫入速度而已，沒有真的要用，所以測試完後記得要刪掉喔！


        print("save_1_save_1_npz_conatin_50npy:", save_1_time)
        print("save_2_save_50_npz:", save_2_time)
        print("save_3_save_50_npy:", save_3_time)
        print("load_1_load_1_npz_conatin_50npy:", load_1_time)
        print("load_2_load_50_npz:", load_2_time)
        print("load_3_load_50_npy:", load_3_time)
        """
        save_1_save_1_npz_conatin_50npy: 7.14643120765686
        save_2_save_50_npz: 7.625092267990112
        save_3_save_50_npy: 0.1186835765838623 ---------------->用這個
        load_1_load_1_npz_conatin_50npy: 1.0990545749664307
        load_2_load_50_npz: 0.2214348316192627 ---------------->用這個
        load_3_load_50_npy: 0.31119656562805176
        """
        print("finish")
