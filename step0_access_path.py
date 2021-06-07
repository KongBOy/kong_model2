### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！
data_access_path    = "G:/0 data_dir/"    ### 通常是 讀取速度較快的SSD，127.35 是 400GB SSD

result_read_path    = "D:/0 data_dir/"    ### 通常是 大容量的機械式硬碟，127.35 是 2T 機械式硬碟
result_write_path   = "G:/0 data_dir/"    ### 通常是 有碎片也沒差的SSD，127.35 是 400GB SSD， 弄完再剪下貼上 到 大容量的硬碟

analyze_access_path = "G:/0 data_dir/"    ### 通常是 有碎片也沒差的SSD，127.35 是 400GB SSD， 弄完再剪下貼上 到 大容量的硬碟

JPG_QUALITY = 30
CORE_AMOUNT = 7
CORE_AMOUNT_NPY_TO_NPZ = 6
CORE_AMOUNT_BM_REC_VISUAL = 8  ### 8  ### 14  ### 500
CORE_AMOUNT_SAVE_AS_JPG = 6  ### 12         ### Save_as_jpg
CORE_AMOUNT_FIND_LTRD_AND_CROP = 6  ### 12  ### Find_ltrd_and_crop

import sys
sys.path.append("kong_util")
from build_dataset_combine import Check_dir_exist_and_build_new_dir
import os

def Syn_write_to_read_dir(write_dir, read_dir, print_msg=False):
    """
    為了 HDD 不產生磁碟碎片，
    我在 train完的後處理 會儲存在 SSD 裡面， 此時是 write 在 SSD，
    這樣會和 從倉庫讀取來源資料的 read 不同(通常儲存在大容量 HDD)，
    為了怕 write完的東西 會是 下個步驟的 read，所以要有這個 同步method 把 write 完的結果 copy 一份回 read 喔！

    # shutil.copytree(write_dir, read_dir)        ### 已測試，產生碎片
    # shutil.copy(path, path)                     ### 已測試，在檔案大又多的時候產生碎片
    """
    Check_dir_exist_and_build_new_dir(read_dir)   ### 在 read 的地方建立 存結果的資料夾，目前覺得 如果已存在 要 刪掉重建，之後想改可再改喔
    write_dir = write_dir.replace("/", "\\") + "\\"
    read_dir  = read_dir .replace("/", "\\") + "\\"
    command   = f'xcopy "{write_dir}" "{read_dir}" /Y /Q'  ### 複製資料夾內的檔案(不包含子資料夾，子資料夾也想複製的話加/E，但我覺得不要，因為如果複製 上層資料夾， 會重複複製到很多次相同子資料夾， 浪費時間)
    ### /Y 預設覆蓋檔案的複製、/Q 不顯示複製的檔案、/E 子資料夾也會複製過去
    ### 我找了很多資料，就是弄不掉 已複製 ... 個檔案，就算了吧～
    os.system(command)
    if(print_msg):
        print("(src)write_dir:", write_dir)
        print("(dst) read_dir:", read_dir)
        print("   use command:", command)
        print("")

    ### 留幾版在刪掉，用一個個檔案複製的方式，但後來發現 用cmd指令 可直接copy 資料夾！且複寫也沒問題～　就用上面的方式了～
    ### 但覺得下面的觀念還不錯，就留一下囉
    # file_names   = get_dir_certain_file_name(write_dir, ".jpg")
    # file_names  += get_dir_certain_file_name(write_dir, ".png")
    # file_names  += get_dir_certain_file_name(write_dir, ".avi")
    # file_names  += get_dir_certain_file_name(write_dir, ".npy")
    # file_names  += get_dir_certain_file_name(write_dir, ".npz")

    # file_names  += get_dir_certain_file_name(write_dir, "checkpoint")
    # file_names  += get_dir_certain_file_name(write_dir, "ckpt")
    # file_names  += get_dir_certain_file_name(write_dir, ".v2")

    # file_names  += get_dir_certain_file_name(write_dir, ".txt")
    # file_names  += get_dir_certain_file_name(write_dir, ".py")

    # for file_name in file_names:
    #     write_path = write_dir + "/" + file_name
    #     read_path  = read_dir  + "/" + file_name
    #     print("write_path:", write_path)
    #     print("read_path :", read_path)
    #     print('copy "' + write_path + '" "' + read_path + '" /Y')
    #     os.system("copy '" + write_path + "' '" + read_path + "' /Y")
    #     os.system('copy "' + write_path + '" "' + read_path + '"')
    #     shutil.copy(write_path, read_path)
