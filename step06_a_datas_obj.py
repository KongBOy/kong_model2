"""
直接看 DB 的狀況並且記錄下來
"""
from step0_access_path import data_access_path
from enum import Enum
import copy

class Range():
    def __init__(self, min=None, max=None):
        if(min is not None): self.min = float(min)
        if(max is not None): self.max = float(max)
    def __eq__(self, that):
        if not isinstance(that, Range): return False
        return self.min == that.min and self.max == that.max
    def __str__(self):
        return f"min:{self.min}, max:{self.max}"

class DB_CATEGORY(Enum):
    '''
    第一層資料夾的名字
    '''
    type0_h_384_w_256_page                        = "type0_h=384,w=256_page"
    type1_h_256_w_256_complex                     = "type1_h=256,w=256_complex"
    type2_h_384_w_256_complex                     = "type2_h=384,w=256_complex"
    type3_h_384_w_256_complex_page                = "type3_h=384,w=256_complex+page"
    type4_h_384_w_256_complex_page_more_like      = "type4_h=384,w=256_complex+page_more_like"
    type5c_real_have_see_no_bg                    = "type5c_real_have_see_no_bg"
    type5d_real_have_see_have_bg                  = "type5d_real_have_see_have_bg"
    type6_h_384_w_256_smooth_curl_fold_and_page   = "type6_h384_w256-smooth-curl_fold_page"
    type7_h472_w304_real_os_book                  = "type7_h472_w304_real_os_book"
    type7b_h500_w332_real_os_book                 = "type7b_h500_w332_real_os_book"
    type8_blender                                 = "type8_blender"
    type9_try_segmentation                        = "type9_try_segmentation"


class DB_NAME(Enum):
    '''
    第二層資料夾的名字
    '''
    complex_move_map       = "complex_move_map"
    complex_gt_ord_pad     = "complex_gt_ord_pad"
    complex_gt_ord         = "complex_gt_ord"

    complex_page_move_map  = "complex+page"
    complex_page_ord_pad   = "complex+page..."
    complex_page_ord       = "complex+page..."

    complex_page_more_like_move_map = "complex+page_more_like_move_map"
    complex_page_more_like_ord_pad  = "complex+page_more_like_ord_pad"
    complex_page_more_like_ord      = "complex+page_more_like_ord"

    smooth_complex_page_more_like_move_map  = "smooth-curl+fold_and_page_move_map"
    smooth_complex_page_more_like_ord_pad   = "smooth-curl+fold_and_page_ord_pad"
    smooth_complex_page_more_like_ord       = "smooth-curl+fold_and_page_ord"

    # complex_page_more_like_ord 和上面一樣
    wei_book                              = "smooth-curl+fold_and_page_ord"
    wei_book_and_complex_page_more_like   = "smooth-curl+fold_and_page_ord"

    no_bg_gt_color     = "no_bg-gt_color"
    no_bg_gt_gray3ch   = "no_bg-gt_gray3ch"
    have_bg_gt_color   = "have_bg-gt_color"
    have_bg_gt_gray3ch = "have_bg-gt_gray3ch"

    os_book_400data        = "os_book_400data"
    os_book_1532data       = "os_book_1532data"
    os_book_1532data_focus = "os_book_1532data_focus"
    os_book_1532data_big   = "os_book_1532data_big"
    os_book_800data        = "os_book_800data"

    ### Blender 系列
    blender_os_hw756       = "os_hw756"
    blender_os_hw768       = "os_hw768"

    blender_os_hw512_have_bg                       = "os_hw512_have_bg"
    blender_os_and_paper_hw512_have_bg             = "os_and_paper_hw512_hdr_bg"
    blender_os_and_paper_hw512_have_dtd_bg         = "os_and_paper_hw512_dtd_bg"
    blender_os_and_paper_hw512_have_dtd_hdr_mix_bg = "os_and_paper_hw512_dtd_hdr_mix_bg"
    os_and_paper_hw512_dtd_hdr_bg_I_to_W_w_M       = "os_and_paper_hw512_dtd_hdr_bg_I_to_W_w_M"

    ### 網路上載的 測試mask 的 DB
    car_db_try_segmentation = "car_db_try_segmentation"

class DB_GET_METHOD(Enum):
    no_detail = ""
    in_dis_gt_move_map = "in_dis_gt_move_map"
    in_dis_gt_ord_pad  = "in_dis_gt_ord_pad"
    in_dis_gt_ord      = "in_dis_gt_ord"
    in_rec_gt_ord      = "in_rec_gt_ord"
    test_indicate      = "test_indicate"

    in_dis_gt_flow     = "in_dis_gt_flow"

    in_dis_gt_mask_coord = "in_dis_gt_mask_coord"

    in_img_gt_mask   = "in_img_gt_mask"

    in_dis_gt_wc           = "in_dis_gt_wc"
    in_dis_gt_wc_try_mul_M = "in_dis_gt_wc_try_mul_M"

    in_wc_gt_flow    = "in_wc_gt_flow"  ### train_in 除了wc外會多抓 dis_img 來 讓 F 可以做 bm_rec喔！

class VALUE_RANGE(Enum):
    zero_to_one    = Range( 0,   1)
    img_range      = Range( 0, 255)
    neg_one_to_one = Range(-1,   1)

####################################################################################################################################
class Datasets():  ### 以上 以下 都是為了要設定這個物件
    # def __init__(self, category, db_name, get_method, h, w):
    def __init__(self):
        ### (必須)basic
        self.category = None
        self.db_name  = None
        self.get_method = None
        self.h = None   ### 用來幫助決定 img_resize 用的，跟 資料夾名稱的命名目前無關喔！因為要有關連太複雜了～
        self.w = None   ### 用來幫助決定 img_resize 用的，跟 資料夾名稱的命名目前無關喔！因為要有關連太複雜了～
        self.db_dir = None
        ''''''
        ### (必須)dir
        self.train_in_dir = None
        self.train_gt_dir = None
        self.test_in_dir  = None
        self.test_gt_dir  = None
        self.see_in_dir   = None
        self.see_gt_dir   = None

        self.train_in2_dir = None
        self.train_gt2_dir = None
        self.test_in2_dir  = None
        self.test_gt2_dir  = None
        self.see_in2_dir   = None
        self.see_gt2_dir   = None

        self.rec_hope_train_dir = None
        self.rec_hope_test_dir  = None
        self.rec_hope_see_dir   = None
        ''''''
        self.in_dir_name  = None
        self.gt_dir_name  = None

        self.in2_dir_name  = None
        self.gt2_dir_name  = None
        self.test_db_name = "test"  ### 目前只覺得 test_db 可以 在事後替換
        ''''''
        ### (必須)format
        ### 最主要是再 step7 unet generate image 時用到，但我覺得那邊可以改寫！改成記bmp/jpg了！
        self.in_format  = None  ### 本來指定 img/move_map(但因有用get_method，已經可區分img/move_map的動作)，現在覺得指定 bmp/jpg好了
        self.gt_format  = None
        self.in2_format  = None
        self.gt2_format  = None
        self.rec_hope_format  = None
        ''''''
        self.db_in_range = None
        self.db_gt_range = None
        self.db_in2_range = None
        self.db_gt2_range = None
        self.db_rec_hope_range = None

        ### (不必須)detail
        ''''''
        self.see_version = None
        ''''''
        self.have_train = True
        self.have_see   = False
        self.have_rec_hope = False

        ### 這裡的東西我覺得跟 db_obj 相關性 沒有比 tf_data還來的大，所以把這幾個attr移到tf_data裡囉！
        ### 一切的一切最後就是要得到這個 data_dict
        ### 這些資訊要從 datapipline 那邊再設定
        # self.batch_size    = 1
        # self.img_resize    = None
        # self.train_shuffle = True
        # self.data_dict     = None ### 這個變成tf_data物件了！

    def __str__(self):
        print("category:%s, db_name:%s, get_method:%s, h:%i, w:%i" % (self.category.value, self.db_name.value, self.get_method.value, self.h, self.w))
        print("train_in_dir:%s," % self.train_in_dir)
        print("train_gt_dir:%s," % self.train_gt_dir)
        print("test_in_dir:%s," % self.test_in_dir)
        print("test_gt_dir:%s," % self.test_gt_dir)
        print("see_in_dir:%s," % self.see_in_dir)
        print("see_gt_dir:%s," % self.see_gt_dir)
        print("---------------------------------------------")
        if( self.train_in2_dir is not None): print("train_in2_dir:%s," % self.train_in2_dir)
        if( self.train_gt2_dir is not None): print("train_gt2_dir:%s," % self.train_gt2_dir)
        if( self.test_in2_dir  is not None): print("test_in2_dir:%s,"  % self.test_in2_dir)
        if( self.test_gt2_dir  is not None): print("test_gt2_dir:%s,"  % self.test_gt2_dir)
        if( self.see_in2_dir   is not None): print("see_in2_dir:%s,"   % self.see_in2_dir)
        if( self.see_gt2_dir   is not None): print("see_gt2_dir:%s,"   % self.see_gt2_dir)
        print()
        print("rec_hope_train_dir:%s," % self.rec_hope_train_dir)
        print("rec_hope_test_dir :%s," % self.rec_hope_test_dir)
        print("rec_hope_see_dir  :%s," % self.rec_hope_see_dir)
        print("in_format:%s, gt_format:%s, rec_hope_format:%s" % (self.in_format, self.gt_format, self.rec_hope_format))
        return ""
####################################################################################################################################

class Dataset_init_builder:
    def __init__(self, db=None):
        if(db is None): self.db = Datasets()
        else: self.db = db

    def build(self):
        print(f"DB_builder build finish")
        return self.db

class Dataset_basic_builder(Dataset_init_builder):
    def set_basic(self, category, db_name, get_method, h, w):
        self.db.category   = category
        self.db.db_name    = db_name
        self.db.get_method = get_method
        self.db.h          = h
        self.db.w          = w
        self.db.db_dir     = data_access_path + "datasets/" + self.db.category.value + "/" + self.db.db_name.value
        return self

    def set_get_method(self, get_method):
        self.db.get_method = get_method
        return self

class Dataset_dir_builder(Dataset_basic_builder):
    def set_dir_manually(self, train_in_dir =None, train_gt_dir =None, test_in_dir =None, test_gt_dir =None, see_in_dir =None, see_gt_dir =None, rec_hope_train_dir=None, rec_hope_test_dir=None, rec_hope_see_dir=None,
                               train_in2_dir=None, train_gt2_dir=None, test_in2_dir=None, test_gt2_dir=None, see_in2_dir=None, see_gt2_dir=None):
        self.db.train_in_dir = train_in_dir
        self.db.train_gt_dir = train_gt_dir
        self.db.test_in_dir  = test_in_dir
        self.db.test_gt_dir  = test_gt_dir
        self.db.see_in_dir   = see_in_dir
        self.db.see_gt_dir   = see_gt_dir

        self.db.train_in2_dir = train_in2_dir
        self.db.train_gt2_dir = train_gt2_dir
        self.db.test_in2_dir  = test_in2_dir
        self.db.test_gt2_dir  = test_gt2_dir
        self.db.see_in2_dir   = see_in2_dir
        self.db.see_gt2_dir   = see_gt2_dir

        self.db.rec_hope_train_dir = rec_hope_train_dir
        self.db.rec_hope_test_dir  = rec_hope_test_dir
        self.db.rec_hope_see_dir   = rec_hope_see_dir
        return self

    def set_dir_by_basic(self):
        in_dir_name  = "未指定"
        in2_dir_name = "未指定"
        gt_dir_name  = "未指定"
        gt2_dir_name = "未指定"
        if  (self.db.get_method == DB_GET_METHOD.in_dis_gt_move_map):
            in_dir_name = "dis_imgs"
            gt_dir_name = "move_maps"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_ord_pad):
            in_dir_name = "dis_imgs"
            gt_dir_name = "gt_ord_pad_imgs"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_ord):
            in_dir_name = "dis_imgs"
            gt_dir_name = "gt_ord_imgs"
        elif(self.db.get_method == DB_GET_METHOD.in_rec_gt_ord):
            in_dir_name = "unet_rec_imgs"
            gt_dir_name = "gt_ord_imgs"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_flow or
             self.db.get_method == DB_GET_METHOD.in_dis_gt_mask_coord):
            in_dir_name = "dis_imgs"
            gt_dir_name = "flows"
        elif(self.db.get_method == DB_GET_METHOD.in_img_gt_mask):
            in_dir_name = "in_imgs"
            gt_dir_name = "gt_masks"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_wc or self.db.get_method == DB_GET_METHOD.in_dis_gt_wc_try_mul_M):
            in_dir_name = "dis_imgs"
            gt_dir_name = "wcs"
        elif(self.db.get_method == DB_GET_METHOD.in_wc_gt_flow):
            in_dir_name = "wcs"
            in2_dir_name = "dis_imgs"
            gt_dir_name = "flows"


        self.db.train_in_dir = self.db.db_dir + "/train/" + in_dir_name
        self.db.train_gt_dir = self.db.db_dir + "/train/" + gt_dir_name
        self.db.test_in_dir  = self.db.db_dir + f"/{self.db.test_db_name}/" + in_dir_name
        self.db.test_gt_dir  = self.db.db_dir + f"/{self.db.test_db_name}/" + gt_dir_name
        self.db.see_in_dir   = self.db.db_dir + "/see/"   + in_dir_name
        self.db.see_gt_dir   = self.db.db_dir + "/see/"   + gt_dir_name

        self.db.train_in2_dir = self.db.db_dir + "/train/" + in2_dir_name
        self.db.train_gt2_dir = self.db.db_dir + "/train/" + gt2_dir_name
        self.db.test_in2_dir  = self.db.db_dir + f"/{self.db.test_db_name}/" + in2_dir_name
        self.db.test_gt2_dir  = self.db.db_dir + f"/{self.db.test_db_name}/" + gt2_dir_name
        self.db.see_in2_dir   = self.db.db_dir + "/see/"   + in2_dir_name
        self.db.see_gt2_dir   = self.db.db_dir + "/see/"   + gt2_dir_name

        self.db.rec_hope_train_dir = self.db.db_dir + "/train/rec_hope"
        self.db.rec_hope_test_dir  = self.db.db_dir + f"/{self.db.test_db_name}/rec_hope"
        self.db.rec_hope_see_dir   = self.db.db_dir + "/see/rec_hope"
        return self

    def reset_test_db_name(self, test_db_name):
        self.db.test_db_name      = test_db_name
        self.set_dir_by_basic()
        return self

class Dataset_format_builder(Dataset_dir_builder):
    def set_in_gt_format_and_range(self, in_format="bmp", gt_format="bmp", rec_hope_format="jpg", db_in_range=None, db_gt_range=None, db_rec_hope_range=None,
                                         in2_format=None, gt2_format=None, db_in2_range=None, db_gt2_range=None,):
        """
        設定 bmp, npy, knpy, ...... 等等的 資料格式
        """
        self.db.in_format  = in_format
        self.db.gt_format  = gt_format
        self.db.in2_format = in2_format
        self.db.gt2_format = gt2_format
        self.db.rec_hope_format = rec_hope_format

        self.db.db_in_range       = db_in_range
        self.db.db_gt_range       = db_gt_range
        self.db.db_in2_range      = db_in2_range
        self.db.db_gt2_range      = db_gt2_range
        self.db.db_rec_hope_range = db_rec_hope_range

        return self

class Dataset_detail_builder(Dataset_format_builder):
    def set_detail(self, have_train=True, have_see=False, have_rec_hope=False, see_version=None):
        """
        資料集 裡面有沒有 train/see 資料夾，有的時候因為懶所以還沒建就想用了這樣子
        """
        self.db.have_train    = have_train
        self.db.have_see      = have_see
        self.db.have_rec_hope = have_rec_hope
        self.db.see_version   = see_version
        return self

class Dataset_builder(Dataset_detail_builder): pass


### Enum改短名字的概念
DB_C = DB_CATEGORY
DB_N = DB_NAME
DB_GM = DB_GET_METHOD
### 直接先建好 obj 給外面import囉！
type5c_real_have_see_no_bg_gt_color          = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg, DB_N.no_bg_gt_color , DB_GM.in_dis_gt_ord,                                                   h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", gt_format="bmp", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver2")
type6_h384_w256_smooth_curl_fold_page        = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page     , DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", gt_format="npy", db_in_range=Range(0, 255)                           ).set_detail(have_train=True, have_see=True, see_version="sees_ver2")
type7_h472_w304_real_os_book_400data         = Dataset_builder().set_basic(DB_C.type7_h472_w304_real_os_book                    , DB_N.os_book_400data                       , DB_GM.in_dis_gt_ord,      h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver3")
type7b_h500_w332_real_os_book_1532data       = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book                   , DB_N.os_book_1532data                      , DB_GM.in_dis_gt_ord,      h=500, w=332).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver3")
type7b_h500_w332_real_os_book_1532data_focus = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book                   , DB_N.os_book_1532data_focus                , DB_GM.in_dis_gt_ord,      h=500, w=332).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver3")
type7b_h500_w332_real_os_book_1532data_big   = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book                   , DB_N.os_book_1532data_big                  , DB_GM.in_dis_gt_ord,      h=600, w=396).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver3")
type7b_h500_w332_real_os_book_400data        = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book                   , DB_N.os_book_400data                       , DB_GM.in_dis_gt_ord,      h=500, w=332).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver3")
type7b_h500_w332_real_os_book_800data        = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book                   , DB_N.os_book_800data                       , DB_GM.in_dis_gt_ord,      h=500, w=332).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True, see_version="sees_ver3")
################################################################################################################################################################################################################################################################################################################################################################################################################################
type8_blender_os_book_756                    = Dataset_builder().set_basic(DB_C.type8_blender,          DB_N.blender_os_hw756,                                 DB_GM.in_dis_gt_flow,       h=756, w=756)  .set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy",                      db_in_range=Range(0, 255), db_gt_range=Range(0,   1)                               ).set_detail(have_train=True, have_see=True,                     see_version="sees_ver4_blender")
type9_try_segmentation                       = Dataset_builder().set_basic(DB_C.type9_try_segmentation, DB_N.car_db_try_segmentation,                          DB_GM.in_img_gt_mask,       h=1280, w=1918).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="gif",                       db_in_range=Range(0, 255), db_gt_range=Range(0, 255)                               ).set_detail(have_train=True, have_see=False,                    see_version="sees_ver4_blender")
### 1
type8_blender_os_book_768                    = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_hw768,                               DB_GM.in_dis_gt_flow,       h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
type9_mask_flow                              = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_hw768,                               DB_GM.in_dis_gt_mask_coord, h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
### 2
type8_blender_os_book_512_have_bg            = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_hw512_have_bg,                       DB_GM.in_dis_gt_flow,       h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
type9_mask_flow_have_bg                      = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_hw512_have_bg,                       DB_GM.in_dis_gt_mask_coord, h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
### 3
type8_blender_os_and_paper_hw512_have_bg     = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_and_paper_hw512_have_bg,             DB_GM.in_dis_gt_flow,       h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
type9_mask_flow_have_bg_and_paper            = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_and_paper_hw512_have_bg,             DB_GM.in_dis_gt_mask_coord, h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
### 4
type8_blender_os_and_paper_hw512_have_dtd_bg = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_and_paper_hw512_have_dtd_bg,         DB_GM.in_dis_gt_flow,       h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
type9_mask_flow_have_bg_dtd_and_paper        = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.blender_os_and_paper_hw512_have_dtd_bg,         DB_GM.in_dis_gt_mask_coord, h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
### 5
type8_blender_os_and_paper_hw512_have_dtd_hdr_mix_bg = Dataset_builder().set_basic(DB_C.type8_blender   , DB_N.blender_os_and_paper_hw512_have_dtd_hdr_mix_bg, DB_GM.in_dis_gt_flow,       h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
type9_mask_flow_have_bg_dtd_hdr_mix_and_paper        = Dataset_builder().set_basic(DB_C.type8_blender   , DB_N.blender_os_and_paper_hw512_have_dtd_hdr_mix_bg, DB_GM.in_dis_gt_mask_coord, h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy", rec_hope_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 1), db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")

type8_blender_wc                             = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.os_and_paper_hw512_dtd_hdr_bg_I_to_W_w_M,           DB_GM.in_dis_gt_wc,         h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy",                        db_in_range=Range(0, 255), db_gt_range=Range(-0.13532962, 0.1357405) ).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")
type8_blender_wc_try_mul_M                   = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.os_and_paper_hw512_dtd_hdr_bg_I_to_W_w_M,           DB_GM.in_dis_gt_wc_try_mul_M, h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", gt_format="knpy",                        db_in_range=Range(0, 255), db_gt_range=Range(-0.13532962, 0.1357405) ).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")

type8_blender_wc_flow                        = Dataset_builder().set_basic(DB_C.type8_blender           , DB_N.os_and_paper_hw512_dtd_hdr_bg_I_to_W_w_M,           DB_GM.in_wc_gt_flow,        h=512, w=512).set_dir_by_basic().set_in_gt_format_and_range(in_format="knpy", gt_format="knpy", in2_format="png", db_in_range=Range(-0.13532962, 0.1357405), db_gt_range=Range(0,   1), db_in2_range=(0, 255) ).set_detail(have_train=True, have_see=True, have_rec_hope=True, see_version="sees_ver4_blender")

if(__name__ == "__main__"):
    db = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg,   DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", gt_format="bmp", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    db = Dataset_builder().set_basic(DB_C.type7_h472_w304_real_os_book, DB_N.os_book_400data,  DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    db = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book,DB_N.os_book_1532data, DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg", db_in_range=Range(0, 255), db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    print(type8_blender_os_book_768.build())
    print(type9_try_segmentation.build())
    print(type8_blender_wc_flow.build())
    # db_complex_1_pure_unet = Datasets(DB_CATEGORY.type1_h_256_w_256_complex, DB_GET_METHOD.in_dis_gt_move_map   , h=256, w=256 )
    # db_complex_2_pure_rect = Datasets(DB_CATEGORY.type1_h_256_w_256_complex, DB_GET_METHOD.in_dis_gt_ord_pad_img, h=256, w=256 )
    # db_complex_3_pure_rect = Datasets(DB_CATEGORY.type1_h_256_w_256_complex, DB_GET_METHOD.in_rec_gt_ord_img    , h=256, w=256 )
