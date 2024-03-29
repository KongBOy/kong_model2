### 定位出 kong_model2
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])    ### 定位出 kong_model2 的 dir
import sys
sys.path.append(kong_model2_dir + "/kong_util")
# print("step0")
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
######################################################################################################################
### 後面直接補上 "/"囉，就不用再 +"/"+，自己心裡知道就好！
Kong_Doc3D_Dir   = "D:/data_dir/"    ### 通常是 讀取速度較快的SSD，127.35 是 400GB SSD
# Data_Access_Dir  = Kong_Doc3D_Dir
Data_Access_Dir  = f"{kong_model2_dir}/data_dir/"    ### 通常是 讀取速度較快的SSD，127.35 是 400GB SSD

# Result_Read_Dir  = "H:/data_dir/"    ### 通常是 大容量的機械式硬碟，127.35 是 2T 機械式硬碟
# Result_Write_Dir = "H:/data_dir/"    ### 通常是 有碎片也沒差的SSD，127.35 是 400GB SSD， 弄完再剪下貼上 到 大容量的硬碟
Result_Read_Dir  = f"{kong_model2_dir}/data_dir/"    ### 通常是 大容量的機械式硬碟，127.35 是 2T 機械式硬碟
Result_Write_Dir = f"{kong_model2_dir}/data_dir/"    ### 通常是 有碎片也沒差的SSD，127.35 是 400GB SSD， 弄完再剪下貼上 到 大容量的硬碟

# Analyze_Read_Dir   =  "H:/data_dir/"
# Analyze_Write_Dir  =  "H:/data_dir/"
Analyze_Read_Dir  =  f"{kong_model2_dir}/data_dir/"
Analyze_Write_Dir = f"{kong_model2_dir}/data_dir/"

JPG_QUALITY = 30
CORE_AMOUNT = 6
CORE_AMOUNT_NPY_TO_NPZ = 2
CORE_AMOUNT_BM_REC_VISUAL = 8  ### 8  ### 14  ### 500
CORE_AMOUNT_WM_VISUAL = 5  ### 8  ### 14  ### 500
CORE_AMOUNT_SAVE_AS_JPG = 6  ### 12         ### Save_as_jpg
CORE_AMOUNT_FIND_LTRD_AND_CROP = 6  ### 12  ### Find_ltrd_and_crop

Change_name_used_Result_Read_Dirs = [
    "K:/data_dir/",  ### 6T
    "P:/data_dir/",  ### 4T
    "H:/data_dir/",  ### 2T
    "F:/kong_model2/data_dir/",  ### 400GB
    Result_Write_Dir,  ### 別台電腦再 kong_model2 裡面的result
]


def Syn_write_to_read_dir(write_dir, read_dir, build_new_dir=False, copy_sub_dir=False, print_msg=True):
    from kong_util.build_dataset_combine import Check_dir_exist_and_build, Check_dir_exist_and_build_new_dir
    from kong_util.util                  import Visit_sub_dir_include_self_and_get_dir_paths

    import os
    """
    為了 HDD 不產生磁碟碎片，
    我在 train完的後處理 會儲存在 SSD 裡面， 此時是 write 在 SSD，
    這樣會和 從倉庫讀取來源資料的 read 不同(通常儲存在大容量 HDD)，
    為了怕 write完的東西 會是 下個步驟的 read，所以要有這個 同步method 把 write 完的結果 copy 一份回 read 喔！

    # shutil.copytree(write_dir, read_dir)        ### 已測試，產生碎片
    # shutil.copy(path, path)                     ### 已測試，在檔案大又多的時候產生碎片

    目前的 see_method 對應要不要用 build_new_dir： 應該只有 npy_to_npz這種 會把原始檔案 刪掉的 method 不能build_new_dir 囉！
         True,  flow_matplot
         False, Npy_to_npz
         True,  bm_rec
         True,  Calculate_SSIM_LD
         True,  Visual_SSIM_LD
    """
    ######################################################################################################################################################################################################################################################
    ### build_new_dir 的探討：
    ### 1.在 read 的地方建立 存結果的資料夾，目前覺得 如果已存在 蓋過即可(前提是 "data" / "visual" 要分開放喔！)
    ### 1. "data" / "visual" 要分開放喔！ 會有 "中斷後重新啟動" 殘檔問題
    ###     之前會覺得 不能 new_dir 的原因是 因為 我 "data" / "visual" 沒有分開放！
    ###     舉例：像我的 SSIM/LD 的 .npy 和 visual 放同個資料夾，.npy算完，換做visual來到這行同步， 如果是build_new_dir 就會把算好的.npy刪掉囉！因為放在同個資料夾！ 同步的時候會互相影響！
    ###           如果 不build_new_dir， 要考慮到 "中斷後重新執行" 的問題， "visual" 的 .png轉.jpg 如果中斷 很大可能有png殘檔， 重新開始 算.npy 算完要同步 .npy 就會同步到 visual 的png殘檔囉！ 就算visual做完同步也不會覆蓋到，因為png已轉jpg，蓋不到！
    ###     如果硬要用 其實 remove_dir_certain_file_name 在同步前把殘檔刪乾淨，但寫起來麻煩 不想找自己麻煩ˊ口ˋ
    ###     其實只要把 "data" / "visual" 分開 資料夾放！ 就 不會有 同步的時候 同步到別人東西 的問題
    ###     補充一下我的 .npz(data) 和 .npz_visual(visual) 放一起囉，但這沒問題是因為，我的visual 是在 training過程中做，training後完全不會再做visual的動作囉，就沒 "中斷後重新啟動"問題！但要小心 不使用build_new_dir， 會有 "父子資料夾問題"喔！
    ### 2.一定 不要build_new_dir！ 會有 "父子資料夾的問題"
    ###     因為要考慮到 父子資料夾 的問題喔！像是 .npz(父 see_010-test ) 和 bm/rec(子 see_010-test/bm_rec_matplot_visual)
    ###     如果同步父時 會把 子全部刪掉囉！
    ###     只要不 build_new_dir 就算 父子資料夾也沒問題！
    ###         1.因為 這樣就不會刪到子了
    ###         2.因為 windows 的 copy 不會複製子資料夾， 所以 同步父的時候不會同步子！
    ###         3.同步是在 最後執行的，所以 一定可以覆蓋到 要同步的資料，比如 bm/rec，一定是在全變成.jpg後才會執行同步，不會有.png和.jpg混雜的情形(這情形是在 "data" / "visual" 放一起， "visual" 的 .png轉.jpg＂中斷後重新執行"， "data" 可能會 同步到 "visual" 的.png)
    ### 3. 向 npy_to_npz 這種 "會把 原始檔案刪除" 的動作 要很小心，
    ###      在 初始狀態時 一定要把 之前已處理完的結果 複製到 write_dir， 要不然 原始檔案已刪了， 第二次執行的時候 在 write_dir 中就不會存在， 同步的時候 就會把原始的 read_dir 的檔案也刪掉囉！
    ###      下面有畫圖更好理解：
    ###      https://drive.google.com/file/d/1hHc1oUggKMTJUPTaEkVUvJbjlPeIBZiu/view?usp=sharing
    # Check_dir_exist_and_build(read_dir)

    ### 會想build_new_dir 的原因是因為 如果我有改 epoch 檔名， 用 build_new_dir 才會把舊結果刪除！
    ### 但是以上問題其實都可以解，
    ### 不對，3很難解
    ###    1解： data 和 visual 分開存就好啦
    ###    2解： 同步時要有順序，先從父開始同步，在同步子就行了， 不要先同步子 再同步父喔！這樣 再同步父的時候會把 子已經同步完的結果 刪光光 這樣子拉～
    ###    3解： 記得 複製 已處理過的檔案 進 write_dir 就好了！
    ### 但是 3解不合理， 因為 我 SSD 的結果處理完 正常就刪掉了， 如果要再重新處理一次時 還需要從 HDD 先copy 一份過來，也太麻煩了吧！不合理呀！
    # Check_dir_exist_and_build_new_dir(read_dir)
    ######################################################################################################################################################################################################################################################
    ### 最後總結，加一個 參數 控制 要不要build_new_dir吧ˊ口ˋ ，default用 最保險的 不要build_new_dir
    ###   目前的 see_method 對應要不要用 build_new_dir： 應該只有 npy_to_npz這種 會把原始檔案 刪掉的 method 不能build_new_dir 囉！
    ###      True,  flow_matplot
    ###      False, Npy_to_npz
    ###      True,  bm_rec
    ###      True,  Calculate_SSIM_LD
    ###      True,  Visual_SSIM_LD
    if(build_new_dir): Check_dir_exist_and_build_new_dir (read_dir)
    else:              Check_dir_exist_and_build         (read_dir)
    ######################################################################################################################################################################################################################################################
    ######################################################################################################################################################################################################################################################

    write_dir = write_dir.replace("/", "\\") + "\\"
    read_dir  = read_dir .replace("/", "\\") + "\\"
    if(print_msg):
        print("(src)write_dir:", write_dir)
        print("(dst) read_dir:", read_dir)
    
    ### 看一下 子資料夾 最長 的名字 會到多長
    src_dirs = []
    Visit_sub_dir_include_self_and_get_dir_paths(src_dir=write_dir, dir_containor=src_dirs)
    src_dir_lens = [len(src_dir) for src_dir in src_dirs]
    src_dir_len_max = max(src_dir_lens)

    ### 如果 子資料夾 最長的名字 <255 用 xcopy，比較不會有碎片， >255 沒辦法用xcopy， 就只能用 shutil 囉
    if(src_dir_len_max <= 255):
        command   = f'xcopy "{write_dir}" "{read_dir}" /Y /Q'  ### 複製資料夾內的檔案(不包含子資料夾，子資料夾也想複製的話加/E，但我覺得不要，因為如果複製 上層資料夾， 會重複複製到很多次相同子資料夾， 浪費時間)
        if(copy_sub_dir): command += " /E"  ### 如果有需要複製子資料夾， 再自己把 參數 copy_sub_dir 設True 囉！
        ### /Y 預設覆蓋檔案的複製、/Q 不顯示複製的檔案、/E 子資料夾也會複製過去
        ### 我找了很多資料，就是弄不掉 已複製 ... 個檔案，就算了吧～
        os.system(command)
        if(print_msg):
            print("   use command:", command)
            print("")
    else:
        print(f"(dst) read_dir > 255 個字， 沒辦法只能用 shutil.copyfile() 這種產生碎片的 copy囉")

        ### 走訪 (src)write_dir 內所有的資料夾
        from kong_util.util import Visit_sub_dir_include_self_and_get_dir_paths
        src_dirs = []
        Visit_sub_dir_include_self_and_get_dir_paths(write_dir, dir_containor=src_dirs)

        ### 定位 和 建出 所有 (dst)read_dir 資料夾， 因為用 shutil.copyfile 需要 dst 資料夾存在
        dst_dirs = []
        for src_dir in src_dirs:
            dst_dirs.append( src_dir.replace(write_dir, read_dir) )
            Check_dir_exist_and_build(dst_dirs[-1])

        ### 抓出所有的 src_dirs 內部的所有檔案path
        src_file_paths = []
        for src_dir in src_dirs:
            src_dir_files = os.listdir(src_dir)
            if(len(src_dir_files) > 0):
                src_file_paths += [src_dir + "/" + file_name for file_name in src_dir_files if(os.path.isfile(src_dir + "/" + file_name))]

        ### 定位所有的 dst_dirs 內部的所有檔案path
        dst_file_paths = []
        for src_file_path in src_file_paths:
            dst_file_paths.append(src_file_path.replace(write_dir, read_dir))

        import shutil
        for go_path, src_file_path in enumerate(src_file_paths):
            shutil.copyfile(src_file_paths[go_path], dst_file_paths[go_path])


        # shutil.copytree(write_dir, read_dir)

        # command   = f'robocopy "{write_dir}" "{read_dir}" /E'  ### 複製資料夾內的檔案(不包含子資料夾，子資料夾也想複製的話加/E，但我覺得不要，因為如果複製 上層資料夾， 會重複複製到很多次相同子資料夾， 浪費時間)

    ### 留幾版在刪掉，用一個個檔案複製的方式，但後來發現 用cmd指令 可直接copy 資料夾！且複寫也沒問題～　就用上面的方式了～
    ### 但覺得下面的觀念還不錯，就留一下囉
    # file_names   = get_dir_certain_file_names(write_dir, ".jpg")
    # file_names  += get_dir_certain_file_names(write_dir, ".png")
    # file_names  += get_dir_certain_file_names(write_dir, ".avi")
    # file_names  += get_dir_certain_file_names(write_dir, ".npy")
    # file_names  += get_dir_certain_file_names(write_dir, ".npz")

    # file_names  += get_dir_certain_file_names(write_dir, "checkpoint")
    # file_names  += get_dir_certain_file_names(write_dir, "ckpt")
    # file_names  += get_dir_certain_file_names(write_dir, ".v2")

    # file_names  += get_dir_certain_file_names(write_dir, ".txt")
    # file_names  += get_dir_certain_file_names(write_dir, ".py")

    # for file_name in file_names:
    #     write_path = write_dir + "/" + file_name
    #     read_path  = read_dir  + "/" + file_name
    #     print("write_path:", write_path)
    #     print("read_path :", read_path)
    #     print('copy "' + write_path + '" "' + read_path + '" /Y')
    #     os.system("copy '" + write_path + "' '" + read_path + "' /Y")
    #     os.system('copy "' + write_path + '" "' + read_path + '"')
    #     shutil.copy(write_path, read_path)
