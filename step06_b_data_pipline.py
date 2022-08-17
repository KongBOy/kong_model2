

from step06_a_datas_obj import VALUE_RANGE, Range
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.keras.backend.set_floatx('float32')  ### 這步非常非常重要！用了才可以加速！

from kong_util.util import get_dir_dir_names

# import pdb
####################################################################################################
class norm_and_resize_mapping_util():
    def set_hole_norm(self):
        self.norm_use_what_min = self.db_range.min
        self.norm_use_what_max = self.db_range.max

    def set_ch_norm(self):
        ch_mins = tf.constant([ self.db_ch_ranges[0].min,
                                self.db_ch_ranges[1].min,
                                self.db_ch_ranges[2].min ], dtype=tf.float32)
        ch_mins = tf.reshape(ch_mins, shape=(1, 1, 3))
        ch_maxs = tf.constant([ self.db_ch_ranges[0].max,
                                self.db_ch_ranges[1].max,
                                self.db_ch_ranges[2].max ], dtype=tf.float32)
        ch_maxs = tf.reshape(ch_maxs, shape=(1, 1, 3))

        self.norm_use_what_min = ch_mins
        self.norm_use_what_max = ch_maxs


    def _resize(self, data):
        data = tf.image.resize(data, self.img_resize, method=tf.image.ResizeMethod.AREA)
        return data

    def _norm_img_to_tanh(self, img):  ### 因為用tanh，所以把值弄到 [-1, 1]
        img = (img / 127.5) - 1
        return img

    def _norm_img_to_01(self, img):
        img = img / 255.0
        return img

    def _norm_to_01_by_max_min_val(self, data):  ### 因為用tanh，所以把值弄到 [-1, 1]
        # data = (data - self.db_range.min) / (self.db_range.max - self.db_range.min)
        data = (data - self.norm_use_what_min) / (self.norm_use_what_max - self.norm_use_what_min)

        # print("self.db_range.max", self.db_range.max)
        # print("self.db_range.min", self.db_range.min)
        return data

    def _norm_to_tanh_by_max_min_val(self, data):  ### 因為用tanh，所以把值弄到 [-1, 1]
        data = self._norm_to_01_by_max_min_val(data) * 2 - 1
        return data

    def check_use_range_and_db_range_then_normalize_data_by_use_what_min_max(self, data):
        if  (self.use_range == self.db_range): pass  ### 此狀況比如 uv值方面blender都幫我們弄好了：值在 0~1 之間，所以不用normalize～
        elif(self.use_range == VALUE_RANGE.zero_to_one   .value): data = self._norm_to_01_by_max_min_val   (data)
        elif(self.use_range == VALUE_RANGE.neg_one_to_one.value): data = self._norm_to_tanh_by_max_min_val (data)  ### resize 和 值弄到 -1~1，假設(99%正確拉懶得考慮其他可能ˊ口ˋ)blender 的 flow一定0~1，所以不等於發生時 一定是要弄成 -1~1
        elif(self.use_range == VALUE_RANGE.img_range     .value): print("img 的 in/gt range 設錯囉！ 不能夠直接用 0~255 的range 來train 模型喔~~")   ### 防呆一下
        elif(self.use_range is None): print("tf_data 忘記設定 in/use_gt_range 了！，你可能會看到 Dataset.zip() 的錯誤喔 ~ ")              ### 防呆一下
        return data


####################################################################################################
####################################################################################################
class img_mapping_util(norm_and_resize_mapping_util):
    def step0a_load_byte_img(self, file_name):
        byte_img = tf.io.read_file(file_name)
        return byte_img

    '''
    寫得這麼難看，是因為 dataset.map() 不能傳參數呀~~ 網路上查到用lambda 傳參數 可跑過，但是無法autograph！
    所以就乖乖寫得這麼難看囉！
    '''
    def step0b_decode_bmp(self, byte_img): return tf.image.decode_bmp(byte_img)
    def step0b_decode_jpg(self, byte_img): return tf.image.decode_jpeg(byte_img)
    def step0b_decode_png(self, byte_img): return tf.image.decode_png(byte_img)
    def step0b_decode_gif(self, byte_img): return tf.image.decode_gif(byte_img)[0]  ### gif 好像比較特殊，我猜是 THWC, T是時間


    def step1_uint8(self, img):
        img  = tf.cast(img, tf.uint8)  ### 一定要在 resize 之後 再 cast 成 uint8 喔！ 因為 resize 完會變 float32
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉

    def step1_uint8_resize(self, img):
        img = self._resize(img)
        img  = tf.cast(img, tf.uint8)  ### 一定要在 resize 之後 再 cast 成 uint8 喔！ 因為 resize 完會變 float32
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉


    def step1_float32_resize_and_to_01(self, img):
        img  = tf.cast(img, tf.float32)
        img = self._norm_img_to_01(img)
        img = self._resize(img)
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉


    def step1_float32_resize_and_to_tanh(self, img):
        img  = tf.cast(img, tf.float32)
        img = self._norm_img_to_tanh(img)
        img = self._resize(img)
        return img[..., :3]  ### png有四個channel，第四個是透明度用不到所以只拿前三個channel囉


####################################################################################################
####################################################################################################
class flow_wc_mapping_util(norm_and_resize_mapping_util):
    def _step0_load_knpy(self, file_name):
        '''
        uv(flow):
            ch1 : mask
            ch2 : y
            ch3 : x
        wc:
            ch1：z
            ch2：y
            ch3：x
        '''
        flow = tf.io.read_file(file_name)
        flow = tf.io.decode_raw(flow , tf.float32)
        flow = tf.cast(flow, tf.float32)
        return flow

    def step1_flow_load(self, file_name):
        raw_data = self._step0_load_knpy(file_name)
        F = tf.reshape(raw_data, [self.db_h, self.db_w, 3])  ### ch1:mask, ch2:y, ch3:x
        return F

    def step2_M_bin_and_C_norm_by_use_what_min_max_no_mul_M_wrong_but_OK(self, F):
        '''
        Wrong 但 為了 相容性 仍保留～
        normalize 完以後 mask 以外的區域不保證為0，
        flow剛好是因為 本身值就接近 0~1，所以才會 剛好 mask 外 幾乎為0，
        但 wc 就不是如此囉！
        '''
        M     = F[..., 0:1]
        C     = F[..., 1:3]
        M_pre = self._mask_binarization(M)
        C_pre = self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max(C)
        F_pre = tf.concat([M_pre, C_pre], axis=-1)
        F_pre = self._resize(F_pre)
        return F_pre

    def step2_M_bin_and_C_norm_by_use_what_min_max_then_mul_M_right(self, F):
        ''' 
        Right 的方法
        normalize 完以後 mask 以外的區域不保證為0， 所以應該還要再 * mask， 以確保 mask 外為0
        這是寫完 wc 後才發現的問題， wc 是一定要 * mask，　因為 wc 本身值 不接近 0~1， normalize 完後 mask外有值！
        flow 因為本身值就接近 0~1， 所以乘了看起來沒效果， 但理論上還是乘一下比較保險喔！
        因為不乘的話 mask 外 有可能有雜雜的非0 存在喔～
        '''
        M     = F[..., 0:1]
        C     = F[..., 1:3]
        M_pre = self._mask_binarization(M)
        C_pre = self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max(C)
        C_pre = C_pre * M  ### 多了這一步，其他都跟 wrong 一樣
        F_pre = tf.concat([M_pre, C_pre], axis=-1)
        F_pre = self._resize(F_pre)
        return F_pre


    def step2_flow_extract_just_mask(self, F):
        M = F[..., 0:1]
        return M

    ###############################################################################################
    ###############################################################################################
    def step1_wc_load(self, file_name):
        raw_data = self._step0_load_knpy(file_name)
        wc = tf.reshape(raw_data, [self.db_h, self.db_w, 4])  ### ch1:z, ch2:y, ch3:x, ch4:後處理成mask了
        return wc   #[..., :3]


    def step2_M_bin_and_W_norm_by_use_what_min_max_no_mul_M_wrong(self, wc):
        ''' Wrong 但 為了 相容性 仍保留～
        normalize 完以後 mask 以外的區域不保證為0，
        flow剛好是因為 本身值就接近 0~1，所以才會 剛好 mask 外 幾乎為0，
        但 wc 就不是如此囉！
        '''
        M      = wc[..., 3:4]
        M_pre  = self._mask_binarization(M)
        wc     = wc[...,  :3]
        wc_pre = self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max(wc)
        W_pre = tf.concat([wc_pre, M_pre], axis=-1)
        W_pre = self._resize(W_pre)
        return W_pre

    def step2_M_bin_and_W_norm_by_use_what_min_max_then_mul_M_right(self, wc):
        ''' Right 的方法
        normalize 完以後 mask 以外的區域不保證為0， 所以應該還要再 * mask， 以確保 mask 外為0
        這是寫完 wc 後才發現的問題， wc 是一定要 * mask，　因為 wc 本身值 不接近 0~1， normalize 完後 mask外有值！
        flow 因為本身值就接近 0~1， 所以乘了看起來沒效果， 但理論上還是乘一下比較保險喔！
        因為不乘的話 mask 外 有可能有雜雜的非0 存在喔～
        '''
        M      = wc[..., 3:4]
        M_pre  = self._mask_binarization(M)
        wc     = wc[...,  :3]
        wc_pre = self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max(wc)
        wc_w_M_pre = wc_pre * M_pre  ### 多了這一步，其他都跟 wrong 一樣
        W_w_M_pre = tf.concat([wc_w_M_pre, M_pre], axis=-1)
        W_w_M_pre = self._resize(W_w_M_pre)
        return W_w_M_pre

    def step2_M_bin_and_W_norm_by_use_what_min_max_then_mul_M_right_only_for_doc3d_x_value_reverse(self, wc):
        ''' Right 的方法
        normalize 完以後 mask 以外的區域不保證為0， 所以應該還要再 * mask， 以確保 mask 外為0
        這是寫完 wc 後才發現的問題， wc 是一定要 * mask，　因為 wc 本身值 不接近 0~1， normalize 完後 mask外有值！
        flow 因為本身值就接近 0~1， 所以乘了看起來沒效果， 但理論上還是乘一下比較保險喔！
        因為不乘的話 mask 外 有可能有雜雜的非0 存在喔～
        '''
        M      = wc[..., 3:4]
        M_pre  = self._mask_binarization(M)
        wc     = wc[...,  :3]
        wc_pre = self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max(wc)
        ''' 多這邊而已 開始'''
        wz_pre = wc_pre[..., 0:1]
        wy_pre = wc_pre[..., 1:2]
        wx_pre = wc_pre[..., 2:3]
        wx_pre = 1. - wx_pre  ### doc3d x value 是反的喔！
        wc_pre = tf.concat([wz_pre, wy_pre, wx_pre], axis=-1)
        ''' 多這邊而已 結束'''

        wc_w_M_pre = wc_pre * M_pre  ### 多了這一步，其他都跟 wrong 一樣
        W_w_M_pre = tf.concat([wc_w_M_pre, M_pre], axis=-1)
        W_w_M_pre = self._resize(W_w_M_pre)
        return W_w_M_pre

class mask_mapping_util(norm_and_resize_mapping_util):
    def _3ch_get_1ch(self, img): return img[..., 0]

    def _mask_binarization(self, mask):
        threshold = 0.8
        mask = tf.where(mask > threshold, 1, 0)
        mask = tf.cast(mask, tf.float32)
        return mask

'''
以上： 定義 mapping function
以下： 使用 mapping function
所以兩部分沒辦法融合 成一個function， 只要用mapping的方式， 一定是分成兩個part， 一個定義， 一個使用
'''
####################################################################################################
####################################################################################################
class tf_Data_element:
    def __init__(self):
        self.ord = None
        self.pre = None
### 下面的 tf_Data_element_factory_builder/Factory 都是為了要建 tf_Data_element_factory 這個物件喔！
### 把img_db 包成class 是因為 tf.data.Dataset().map(f)的這個f，沒有辦法丟參數壓！所以只好包成class，把要傳的參數當 data_member囉！ 另一方面是好管理、好閱讀～
class tf_Data_element_factory(img_mapping_util, flow_wc_mapping_util, mask_mapping_util):
    def __init__(self):  ### file_format 是 bmp/jpg喔！
        self.ord_dir = None
        '''
        db_range: DB 本身的 range(由 db_obj 決定)
        use_range: 進網路前想用的 range(由 exp_obj 決定)
        '''

        ### img類型要存的
        self.file_format = None
        self.img_resize = None

        self.db_h = None
        self.db_w = None
        self.db_range  = None
        self.use_range = None
        self.db_ch_ranges = None

        self.doc3d_subdirs = None

        ### 跟上面不大一樣， 是從內部執行中才決定， 不是從 builder 設定的喔
        self.norm_use_what_min = None
        self.norm_use_what_max = None

        ### 以下兩個是最重要需要求得的
        self.tf_data_element = tf_Data_element()
        # self.ord_db = None  ### 原始讀進來的影像db
        # self.pre_db = None  ### 進 generator 之前 做resize 和 值變-1~1的處理 後的db

    ####################################################################################################
    def _build_file_name_db(self):
        ### debug用
        # from tqdm import tqdm
        # print("ord_dir", self.ord_dir)

        ### 如果使用 doc3d 資料集 在 訓練時 可以指定 要用哪一個 subdir
        if("doc3d" in self.ord_dir.lower() and "train" in self.ord_dir.lower()):
            ### 如果沒有指定 doc3d_subdirs， 自動去 doc3d/train/.../... 裡面看 有哪些 subdirs
            if(self.doc3d_subdirs is None): self.doc3d_subdirs = get_dir_dir_names(self.ord_dir)
            # print("self.doc3d_subdirs", self.doc3d_subdirs)

            ### 去 指定的 subdir 抓 file_names
            for go_sub, doc3d_subdir in enumerate(self.doc3d_subdirs):
                if(go_sub == 0): tf_file_names =                           tf.data.Dataset.list_files(self.ord_dir + f"/{doc3d_subdir}/" + "*." + self.file_format, shuffle=False)
                else:            tf_file_names = tf_file_names.concatenate(tf.data.Dataset.list_files(self.ord_dir + f"/{doc3d_subdir}/" + "*." + self.file_format, shuffle=False))

            ### debug用
            # for _ in tqdm(tf_file_names): pass
            return tf_file_names

        else:
            return tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.file_format, shuffle=False)

    def step1_file_paths_to_names(self, file_path):
        return (tf.strings.split(file_path, "\\")[-1])

    ####################################################################################################
    def build_name_db(self):
        tf_data_element = tf_Data_element()
        file_names = self._build_file_name_db()
        tf_data_element.ord = file_names.map(self.step1_file_paths_to_names)
        tf_data_element.pre = tf_data_element.ord  ### 好像用不到不過為求統一性，還是指定一下好了
        return tf_data_element

    ####################################################################################################
    def build_img_db(self):
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = tf_Data_element()
        ### 測map速度用， 這兩行純讀檔 不map， 要用的時候拿掉這兩行註解， 把兩行外有座map動作的地方都註解調， 結論是map不怎麼花時間， 是shuffle 的 buffer_size 設太大 花時間！
        # tf_data_element = tf_Data_element()
        # tf_data_element.ord = tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.file_format, shuffle=False)
        # tf_data_element.pre = tf.data.Dataset.list_files(self.ord_dir + "/" + "*." + self.file_format, shuffle=False)

        file_names = self._build_file_name_db()
        byte_imgs = file_names.map(self.step0a_load_byte_img)

        if  (self.file_format == "bmp"): decoded_imgs = byte_imgs.map(self.step0b_decode_bmp)
        elif(self.file_format == "jpg"): decoded_imgs = byte_imgs.map(self.step0b_decode_jpg)
        elif(self.file_format == "png"): decoded_imgs = byte_imgs.map(self.step0b_decode_png)
        elif(self.file_format == "gif"): decoded_imgs = byte_imgs.map(self.step0b_decode_gif)

        tf_data_element.ord = decoded_imgs.map(self.step1_uint8)
        ### 測試 use_range 有沒有設成功
        # print("self.use_range:", self.use_range)
        # print("VALUE_RANGE.neg_one_to_one.value:", VALUE_RANGE.neg_one_to_one.value, self.use_range == VALUE_RANGE.neg_one_to_one.value)
        # print("VALUE_RANGE.zero_to_one.value:", VALUE_RANGE.zero_to_one.value, self.use_range == VALUE_RANGE.zero_to_one.value)
        if  (self.use_range == Range(-1,   1)): tf_data_element.pre = decoded_imgs.map(self.step1_float32_resize_and_to_tanh)
        elif(self.use_range == Range( 0,   1)): tf_data_element.pre = decoded_imgs.map(self.step1_float32_resize_and_to_01)
        elif(self.use_range == Range( 0, 255)): tf_data_element.pre = decoded_imgs.map(self.step1_uint8_resize)
        elif(self.use_range is None): print("tf_data 忘記設定 db_in/use_gt_range 或 rec_hope_range 了！，你可能會看到 Dataset.zip() 的錯誤喔 ~ ")
        return tf_data_element

    ### dtd
    def _dtd_bg_augmentation(self, img):
        chance = tf.random.uniform(shape=[1])
        ### dtd背景
        if(chance > 0.3):
            img_resized       = tf.image.resize(img, [200, 200], method=tf.image.ResizeMethod.AREA)  ### 參考 DewarpNet 的 augmentation方式
            img_resized_tiled = tf.tile(img_resized, (3, 3, 1))
            bg                = img_resized_tiled[:self.img_resize[0], :self.img_resize[1], :]
            ### use_range 已經在 build_img_db 裡面處理好了， 這裡已經是對的 range 囉！

        ### 單色背景
        elif(chance < 0.3 and chance > 0.2):
            c = tf.random.uniform(shape=(1, 1, 3))
            bg = tf.ones(shape=[self.img_resize[0], self.img_resize[1], 3], dtype=tf.float32) * c  ### use_range 為  0 ~ 1
            if  (self.use_range == Range(-1,   1)): bg = bg * 2 - 1                                ### use_range 為 -1 ~ 1
            elif(self.use_range == Range( 0, 255)): bg = bg * 255                                  ### use_range 為  0 ~ 255
        ### 最暗背景
        else:
            bg = tf.zeros(shape=[self.img_resize[0], self.img_resize[1], 3], dtype=tf.float32)  ### use_range 為  0 ~ 1 和 0 ~ 255 都適用
            if(self.use_range == Range(-1,   1)): bg = bg * 2 - 1                               ### use_range 為 -1 ~ 1 時，最小為 -1

        return bg

    def build_dtd_img_db(self):
        img_db = self.build_img_db()
        img_db.pre = img_db.pre.map(self._dtd_bg_augmentation)
        return img_db

    #####################################################################################
    def _build_flow_base_db(self):
        tf_data_element = tf_Data_element()
        tf_data_element.ord = self._build_file_name_db()
        tf_data_element.ord = tf_data_element.ord.map(self.step1_flow_load)

        tf_data_element.pre = self._build_file_name_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step1_flow_load)

        ### default 為了相容性配合以前使用 整個 db 的 min, max
        self.set_hole_norm()

        return tf_data_element

    def build_mov_db(self):
        ''' 以前 誤會 flow 是 move map 時期的版本 '''
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_flow_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max)
        return tf_data_element

    ###  mask+coord(先y再x) 3ch合併 的形式 (最原本的)
    def build_flow_db(self):
        ''' 一次抓 flow 三個 channel '''
        ### 測map速度用， 這兩行純讀檔 不map， 要用的時候拿掉這兩行註解， 把兩行外有座map動作的地方都註解調， 結論是map不怎麼花時間， 是shuffle 的 buffer_size 設太大 花時間！
        # tf_data_element = tf_Data_element()
        # tf_data_element.ord = tf.data.Dataset.list_files(self.ord_dir + "/" + "*.knpy", shuffle=False)
        # tf_data_element.pre = tf.data.Dataset.list_files(self.ord_dir + "/" + "*.knpy", shuffle=False)
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_flow_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.check_use_range_and_db_range_then_normalize_data_by_use_what_min_max)
        return tf_data_element

    ###  mask1ch, coord(先y再x) 2ch 的形式
    def build_F_db_by_MC_hole_norm_no_mul_M_wrong_but_OK(self):
        ''' 因為 flow 去掉 M 後的 max=0.9980217, min=0.0， 幾乎就 0~1了， 
        normalize 完以後 mask 外的區域 也幾乎為0不便，
        所以之前忘記做這個train下去還是沒問題的，
        目前就先繼續錯下去好像也沒差，
        等告一段落時再統一改！
        '''
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_flow_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_C_norm_by_use_what_min_max_no_mul_M_wrong_but_OK)
        return tf_data_element

    def build_F_db_by_MC_hole_norm_then_mul_M_right(self):
        ''' 因為 flow 去掉 M 後的 max=0.9980217, min=0.0， 幾乎就 0~1了， 
        normalize 完以後 mask 外的區域 也幾乎為0不便，
        所以之前忘記做這個train下去還是沒問題的，
        目前就先繼續錯下去好像也沒差，
        等告一段落時再統一改！
        '''
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_flow_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_C_norm_by_use_what_min_max_then_mul_M_right)
        return tf_data_element

    #####################################################################################
    def _build_wc_base_db(self):
        tf_data_element = tf_Data_element()
        tf_data_element.ord = self._build_file_name_db()
        tf_data_element.ord = tf_data_element.ord.map(self.step1_wc_load)

        tf_data_element.pre = self._build_file_name_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step1_wc_load)
        return tf_data_element

    def build_W_db_by_MW_hole_norm_no_mul_M_wrong(self):
        ''' Wrong 但 為了 相容性 仍保留～
        normalize 完以後 mask 以外的區域不保證為0，
        flow剛好是因為 本身值就接近 0~1，所以才會 剛好 mask 外 幾乎為0，
        但 wc 就不是如此囉！
        '''
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_wc_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_W_norm_by_use_what_min_max_no_mul_M_wrong)
        return tf_data_element

    def build_W_db_by_MW_hole_norm_then_mul_M_right(self):
        ''' Right 的方法
        normalize 完以後 mask 以外的區域不保證為0， 所以應該還要再 * mask， 以確保 mask 外為0
        這是寫完 wc 後才發現的問題， wc 是一定要 * mask，　因為 wc 本身值 不接近 0~1， normalize 完後 mask外有值！
        flow 因為本身值就接近 0~1， 所以乘了看起來沒效果， 但理論上還是乘一下比較保險喔！
        因為不乘的話 mask 外 有可能有雜雜的非0 存在喔～
        '''
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_wc_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_W_norm_by_use_what_min_max_then_mul_M_right)
        return tf_data_element

    def build_W_db_by_MW_hole_norm_then_mul_M_right_only_for_doc3d_x_value_reverse(self):
        ''' Right 的方法
        normalize 完以後 mask 以外的區域不保證為0， 所以應該還要再 * mask， 以確保 mask 外為0
        這是寫完 wc 後才發現的問題， wc 是一定要 * mask，　因為 wc 本身值 不接近 0~1， normalize 完後 mask外有值！
        flow 因為本身值就接近 0~1， 所以乘了看起來沒效果， 但理論上還是乘一下比較保險喔！
        因為不乘的話 mask 外 有可能有雜雜的非0 存在喔～
        '''
        ### 使用hole_norm
        self.set_hole_norm()

        tf_data_element = self._build_wc_base_db()
        # tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_W_norm_by_use_what_min_max_then_mul_M_right)
        tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_W_norm_by_use_what_min_max_then_mul_M_right_only_for_doc3d_x_value_reverse)
        return tf_data_element

    def build_W_db_by_MW_ch_norm_then_mul_M_right(self):
        ''' Right 的方法
        normalize 完以後 mask 以外的區域不保證為0， 所以應該還要再 * mask， 以確保 mask 外為0
        這是寫完 wc 後才發現的問題， wc 是一定要 * mask，　因為 wc 本身值 不接近 0~1， normalize 完後 mask外有值！
        flow 因為本身值就接近 0~1， 所以乘了看起來沒效果， 但理論上還是乘一下比較保險喔！
        因為不乘的話 mask 外 有可能有雜雜的非0 存在喔～
        '''
        ### 使用ch_norm
        self.set_ch_norm()

        tf_data_element = self._build_wc_base_db()
        tf_data_element.pre = tf_data_element.pre.map(self.step2_M_bin_and_W_norm_by_use_what_min_max_then_mul_M_right)
        return tf_data_element

    #####################################################################################
    ###  mask1ch， 目前好像沒用到， 這是在用 車子db 測試的mask時候 用的， 測試完後好像就沒再用到了
    def build_mask_db(self):
        tf_data_element = tf_Data_element()
        file_names = self._build_file_name_db()
        if(self.file_format != "knpy"):
            byte_imgs = file_names.map(self.step0a_load_byte_img)
            ### 處理 format
            if  (self.file_format == "jpg"): decoded_imgs = byte_imgs.map(self.step0b_decode_jpg)
            elif(self.file_format == "gif"): decoded_imgs = byte_imgs.map(self.step0b_decode_gif)

            ### 處理 range
            ### db_range先寫 0~255 的case， 有遇到 db_range 0~1的case的話再去加寫
            if(self.db_range == VALUE_RANGE.img_range.value):
                tf_data_element.ord = decoded_imgs.map(self.step1_uint8_resize)
                if  (self.use_range == VALUE_RANGE.neg_one_to_one.value): tf_data_element.pre = decoded_imgs.map(self.step1_float32_resize_and_to_tanh)
                elif(self.use_range == VALUE_RANGE.zero_to_one.value):    tf_data_element.pre = decoded_imgs.map(self.step1_float32_resize_and_to_01)
                elif(self.use_range == VALUE_RANGE.img_range.value):      tf_data_element.pre = decoded_imgs.map(self.step1_uint8_resize)
                elif(self.use_range is None): print("tf_data 忘記設定 in/use_gt_range 或 rec_hope_range 了！，你可能會看到 Dataset.zip() 的錯誤喔 ~ ")
        elif(self.file_format == "knpy"):
            self.build_flow_db()
            tf_data_element.ord = tf_data_element.ord.map(self.step2_flow_extract_just_mask)
            tf_data_element.pre = tf_data_element.pre.map(self.step2_flow_extract_just_mask)

        ### 處理channel數
        if(tf_data_element.pre.element_spec.shape[2] == 3): tf_data_element.pre = tf_data_element.pre.map(self._3ch_get_1ch)
        return tf_data_element


####################################################################################################
####################################################################################################
class tf_Data_element_factory_builder():
    def __init__(self, tf_pipline_factory=None):
        if(tf_pipline_factory is None): self.tf_pipline_factory = tf_Data_element_factory()
        else:                   self.tf_pipline_factory = tf_pipline_factory

    ### 建立empty tf_pipline_factory
    def build(self):
        return self.tf_pipline_factory

    def set_factory(self, ord_dir, file_format, img_resize, db_h, db_w, db_range, use_range, db_ch_ranges=None, doc3d_subdirs=None):
        print("DB ord_dir:", ord_dir)
        self.tf_pipline_factory.ord_dir      = ord_dir
        self.tf_pipline_factory.file_format  = file_format
        self.tf_pipline_factory.img_resize   = img_resize
        self.tf_pipline_factory.db_h         = db_h
        self.tf_pipline_factory.db_w         = db_w
        self.tf_pipline_factory.db_range     = db_range
        self.tf_pipline_factory.use_range    = use_range
        self.tf_pipline_factory.db_ch_ranges = db_ch_ranges
        self.tf_pipline_factory.doc3d_subdirs = doc3d_subdirs
        return self


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
class tf_Data:   ### 以上 以下 都是為了設定這個物件
    def __init__(self):
        self.db_obj           = None
        self.batch_size       = None
        self.train_shuffle    = None

        self.img_resize       = None

        self.doc3d_subdirs = None

        self.use_in_range = None
        self.use_gt_range = None
        self.use_rec_hope_range = None
        self.use_DewarpNet_result_range = None

        # self.train_name_db    = None
        # self.train_in_db      = None
        # self.train_in_db_pre  = None
        # self.train_gt_db      = None
        # self.train_gt_db_pre  = None
        self.train_db_combine = None
        self.train_amount     = None

        # self.test_name_db     = None
        # self.test_in_db       = None
        # self.test_in_db_pre   = None
        # self.test_gt_db       = None
        # self.test_gt_db_pre   = None
        self.test_db_combine = None
        self.test_amount      = None

        # self.see_name_db     = None
        # self.see_in_db        = None
        # self.see_in_db_pre    = None
        # self.see_gt_db        = None
        # self.see_gt_db_pre    = None
        self.see_amount       = None

        # self.rec_hope_train_db     = None
        # self.rec_hope_train_db_pre = None
        # self.rec_hope_test_db      = None
        # self.rec_hope_test_db_pre  = None
        # self.rec_hope_see_db       = None
        # self.rec_hope_see_db_pre   = None

        self.train_name_db = tf_Data_element()
        self.train_in_db   = tf_Data_element()
        self.train_gt_db   = tf_Data_element()
        self.train_in2_db  = tf_Data_element()
        self.train_gt2_db  = tf_Data_element()
        self.train_db_combine = None

        self.test_name_db = tf_Data_element()
        self.test_in_db   = tf_Data_element()
        self.test_gt_db   = tf_Data_element()
        self.test_in2_db  = tf_Data_element()
        self.test_gt2_db  = tf_Data_element()
        self.test_db_combine = None

        self.see_name_db = tf_Data_element()
        self.see_in_db   = tf_Data_element()
        self.see_gt_db   = tf_Data_element()
        self.see_in2_db  = tf_Data_element()
        self.see_gt2_db  = tf_Data_element()

        self.rec_hope_train_db = tf_Data_element()
        self.rec_hope_test_db  = tf_Data_element()
        self.rec_hope_see_db   = tf_Data_element()

        self.dtd = tf_Data_element()

        ### DewarpNet_result 不會有 train_db
        self.DewarpNet_result_test = tf_Data_element()
        self.DewarpNet_result_see  = tf_Data_element()

        ### 最主要是再 step7 unet generate image 時用到，但我覺得可以改寫！所以先註解掉了！
        # self.in_format          = None
        # self.gt_format          = None
