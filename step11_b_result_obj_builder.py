from step0_access_path import result_read_path, result_write_path
from step06_a_datas_obj import DB_C
from step11_a1_see_obj import See
from step11_a2_result_obj import Result
import datetime

class Result_init_builder:
    def __init__(self, result=None):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if(result is None):
            self.result = Result()
        else:
            self.result = result

    def build(self):
        print(f"Result build finish, can use {self.result.result_name}")
        return self.result

class Result_sees_builder(Result_init_builder):
    def _build_sees(self, sees_ver, in_use_range, gt_use_range):
        if  (sees_ver == "sees_ver1"):
            self.result.sees = [See(self.result.result_read_dir, self.result.result_write_dir, "see-%03i" % see_num) for see_num in range(32)]
        elif(sees_ver == "sees_ver2"):
            self.result.sees = [
                See(self.result.result_read_dir, self.result.result_write_dir, "see_000-test_emp"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_001-test_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_002-test_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_003-test_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_004-test_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_005-test_img"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_006-test_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_007-test_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_008-test_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_009-test_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_010-test_lin"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_011-test_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_012-test_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_013-test_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_014-test_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_015-test_str"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_016-train_emp"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_017-train_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_018-train_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_019-train_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_020-train_img"), See(self.result.result_read_dir, self.result.result_write_dir, "see_021-train_img"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_022-train_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_023-train_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_024-train_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_025-train_lin"), See(self.result.result_read_dir, self.result.result_write_dir, "see_026-train_lin"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_027-train_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_028-train_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_029-train_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_030-train_str"), See(self.result.result_read_dir, self.result.result_write_dir, "see_031-train_str")]
        elif(sees_ver == "sees_ver3"):
            self.result.sees = [
                See(self.result.result_read_dir, self.result.result_write_dir, "see_000-test_lt1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_001-test_lt2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_002-test_lt3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_003-test_lt4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_004-test_rt1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_005-test_rt2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_006-test_rt3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_007-test_rt4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_008-test_ld1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_009-test_ld2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_010-test_ld3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_011-test_ld4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_012-test_rd1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_013-test_rd2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_014-test_rd3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_015-test_rd4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_016-train_lt1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_017-train_lt2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_018-train_lt3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_019-train_lt4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_020-train_rt1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_021-train_rt2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_022-train_rt3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_023-train_rt4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_024-train_ld1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_025-train_ld2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_026-train_ld3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_027-train_ld4"),
                See(self.result.result_read_dir, self.result.result_write_dir, "see_028-train_rd1"), See(self.result.result_read_dir, self.result.result_write_dir, "see_029-train_rd2"), See(self.result.result_read_dir, self.result.result_write_dir, "see_030-train_rd3"), See(self.result.result_read_dir, self.result.result_write_dir, "see_031-train_rd4")]
        elif(sees_ver == "sees_ver4_blender"):
            # self.result.sees = [See(self.result.result_read_dir, self.result.result_write_dir, "see_001-real") , See(self.result.result_read_dir, self.result.result_write_dir, "see_002-real") , See(self.result.result_read_dir, self.result.result_write_dir, "see_003-real"), See(self.result.result_read_dir, self.result.result_write_dir, "see_004-real"),
            #                     See(self.result.result_read_dir, self.result.result_write_dir, "see_005-train"), See(self.result.result_read_dir, self.result.result_write_dir, "see_006-train"), See(self.result.result_read_dir, self.result.result_write_dir, "see_007-train"), See(self.result.result_read_dir, self.result.result_write_dir, "see_008-train"),
            #                     See(self.result.result_read_dir, self.result.result_write_dir, "see_009-test") , See(self.result.result_read_dir, self.result.result_write_dir, "see_010-test") , See(self.result.result_read_dir, self.result.result_write_dir, "see_011-test") , See(self.result.result_read_dir, self.result.result_write_dir, "see_012-test")]
            self.result.sees = [See(self.result.result_read_dir, self.result.result_write_dir, "see_001-real") , See(self.result.result_read_dir, self.result.result_write_dir, "see_002-real") , See(self.result.result_read_dir, self.result.result_write_dir, "see_003-real"), See(self.result.result_read_dir , self.result.result_write_dir, "see_004-real"),
                                See(self.result.result_read_dir, self.result.result_write_dir, "see_008-train"),
                                See(self.result.result_read_dir, self.result.result_write_dir, "see_009-test") , See(self.result.result_read_dir, self.result.result_write_dir, "see_010-test") ]

        self.result.see_amount = len(self.result.sees)
        # self.result.see_file_amount = self.result.sees[0].see_file_amount   ### 覺得 see 已經有 see_file_amount了，result 就不需要這attr了， 想用 要知道 要去 sees[...] 取 喔！
        # print("3. at see", self.result.result_name, ", self.result.gt_use_range~~~~~~~~~~~~~~~", self.result.gt_use_range)
        for see in self.result.sees:  ### 設定 in/gt_use_range，生圖 才會跟 gt 的前處理一致喔！
            see.in_use_range = in_use_range
            see.gt_use_range = gt_use_range

class Result_train_builder(Result_sees_builder):
    ###     3b.用result_name 裡面的 DB_CATEGORY 來決定sees_ver
    def _use_result_name_find_sees_ver(self):
        db_c = self.result.result_name.split("/")[-1].split("-")[0]  ### "/"是為了抓底層資料夾，"-"是為了抓 DB_CATEGORY
        sees_ver = ""
        if  (db_c in [DB_C.type5c_real_have_see_no_bg.value,
                      DB_C.type5d_real_have_see_have_bg.value,
                      DB_C.type6_h_384_w_256_smooth_curl_fold_and_page.value  ]): sees_ver = "sees_ver2"
        elif(db_c in [DB_C.type7_h472_w304_real_os_book.value,
                      DB_C.type7b_h500_w332_real_os_book.value]):                 sees_ver = "sees_ver3"
        elif(db_c in [DB_C.type8_blender_os_book.value]):                         sees_ver = "sees_ver4_blender"
        else: sees_ver = "sees_ver1"
        return sees_ver

    ### 設定方式一：用exp_obj來設定
    def set_by_exp(self, exp):
        '''
        step1,2： 用 exp 資訊(describe_mid, describe_end) 和 時間日期 來組成 result_name，
                  設定好 result_name 後 會在 呼叫 step3 來設定 result細節(ckpt/logs dir、see_version、in/gt_use_range)喔！
        '''
        ### step0. (step1.包成function而已) 用 exp 資訊(describe_mid, describe_end) 和 時間日期 來組成 result_name
        def _get_result_name_by_exp(exp):
            import datetime
            ### 自動決定 result_name，再去做進一步設定
            result_name_element = [exp.db_obj.category.value]
            if(exp.describe_mid is not None): result_name_element += [exp.describe_mid]
            result_name_element += [self.current_time, exp.model_obj.model_name.value]
            if(exp.describe_end is not None): result_name_element += [exp.describe_end]
            result_name = "-".join(result_name_element)  ### result資料夾，裡面放checkpoint和tensorboard資料夾
            return exp.exp_dir + "/" + result_name

        ### step1.用 exp 資訊(describe_mid, describe_end) 和 時間日期 來組成 result_name
        self.result.result_name  = _get_result_name_by_exp(exp)

        ### step2.決定好 result_name 後，用result_name來設定Result，
        self.set_by_result_name(self.result.result_name, in_use_range=exp.in_use_range, gt_use_range=exp.gt_use_range)
        return self

    ### 設定方式二：直接給 result_name來設定( result_name格式可以參考 _get_result_name_by_exp )
    def set_by_result_name(self, result_name, in_use_range, gt_use_range):
        '''
        step3abc. 完全手動設定 result_name 和 result細節(ckpt/logs dir、see_version/sees、in/gt_use_range)
        '''
        ### step3a.用result_name 來設定ckpt, logs 的資料夾
        self.result.result_name = result_name  ### 如果他被包在某個資料夾，該資料夾也算名字喔！ex：5_justG_mae1369/type7b_h500_w332_real_os_book-20200525_225555-justG-1532data_mae9_127.35_copy
        self.result.result_read_dir  = result_read_path  + "result/" + result_name
        self.result.result_write_dir = result_write_path + "result/" + result_name
        self.result.ckpt_read_dir    = self.result.result_read_dir + "/ckpt"
        self.result.ckpt_write_dir   = self.result.result_write_dir + "/ckpt"
        self.result.logs_read_dir  = self.result.result_read_dir + "/logs"
        self.result.logs_write_dir = self.result.result_write_dir + "/logs"
        self.result.train_code_read_dir  = self.result.result_read_dir  + f"/train_code_{self.current_time}"
        self.result.train_code_write_dir = self.result.result_write_dir + f"/train_code_{self.current_time}"
        self.result.test_dir = self.result.result_write_dir + "/test"

        ### step3b. 設定 in/gt_use_range，這步一定要在 建立 sees 前面做喔！這樣 sees 才知道怎麼設 in/gt_use_range
        self.result.in_use_range = in_use_range
        self.result.gt_use_range = gt_use_range
        # print("2. self.result.gt_use_range", self.result.gt_use_range)

        ### step3c.用result_name 來決定sees_ver， 再用 in/gt_use_range 去建立sees
        self.result.sees_ver = self._use_result_name_find_sees_ver()
        self._build_sees(self.result.sees_ver, self.result.in_use_range, self.result.gt_use_range)



        ### 給 ana_describe，這是給 step12 用的，default 直 設 result.describe_end
        self.result.ana_describe = result_name.split("-")[-1]
        return self

class Result_plot_builder(Result_train_builder):
    def set_ana_describe(self, ana_describe):
        self.result.ana_describe = ana_describe
        return self

class Result_builder(Result_plot_builder): pass
