from step0_access_path import access_path
from enum import Enum, auto

class DB_CATEGORY(Enum):
    type0_h_384_w_256_page                        = "type0_h=384,w=256_page"
    type1_h_256_w_256_complex                     = "type1_h=256,w=256_complex"
    type2_h_384_w_256_complex                     = "type2_h=384,w=256_complex"
    type3_h_384_w_256_complex_page                = "type3_h=384,w=256_complex+page"
    type4_h_384_w_256_complex_page_more_like      = "type4_h=384,w=256_complex+page_more_like"
    type5c_real_have_see_no_bg_gt_color_gray3ch   = "type5c-real_have_see-no_bg-gt-color&gray3ch"
    type5d_real_have_see_have_bg_gt_color_gray3ch = "type5d-real_have_see-have_bg-gt_color&gray3ch"
    type6_h_384_w_256_smooth_curl_fold_and_page   = "type6_h384_w256-smooth-curl_fold_page"
    type7_h472_w304_real_os_book                  = "type7_h472_w304_real_os_book"
    type7b_h500_w332_real_os_book                 = "type7b_h500_w332_real_os_book"
    type8_blender_os_book                         = "type8_blender_os_book"


class DB_NAME(Enum):
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

    blender_os_hw756       = "blender_os_hw756"

class DB_GET_METHOD(Enum):
    no_detail = ""
    in_dis_gt_move_map = "in_dis_gt_move_map"
    in_dis_gt_ord_pad  = "in_dis_gt_ord_pad"
    in_dis_gt_ord      = "in_dis_gt_ord"
    in_rec_gt_ord      = "in_rec_gt_ord"
    test_indicate      = "test_indicate"

    in_dis_gt_flow     = "in_dis_gt_flow"

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
        ### (必須)dir
        self.train_in_dir = None
        self.train_gt_dir = None
        self.test_in_dir  = None
        self.test_gt_dir  = None
        self.see_in_dir   = None
        self.see_gt_dir   = None
        ### (必須)type
        ### 最主要是再 step7 unet generate image 時用到，但我覺得那邊可以改寫！改成記bmp/jpg了！
        self.in_type  = None  ### 本來指定 img/move_map(但因有用get_method，已經可區分img/move_map的動作)，現在覺得指定 bmp/jpg好了
        self.gt_type  = None
        self.see_type = None
        ### (不必須)detail
        self.have_train = True
        self.have_see   = False

        ### 這裡的東西我覺得跟 db_obj 相關性 沒有比 tf_data還來的大，所以把這幾個attr一道tf_data裡囉！
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
        print("in_type:%s, gt_type:%s, see_type:%s" % (self.in_type, self.gt_type, self.see_type))
        return ""
####################################################################################################################################

class Dataset_init_builder:
    def __init__(self, db=None):
        if(db is None): self.db = Datasets()
        else: self.db = db

    def build(self):
        return self.db

class Dataset_basic_builder(Dataset_init_builder):
    def set_basic(self, category, db_name, get_method, h, w):
        self.db.category   = category
        self.db.db_name    = db_name
        self.db.get_method = get_method
        self.db.h          = h
        self.db.w          = w
        self.db.db_dir     = access_path + "datasets/" + self.db.category.value + "/" + self.db.db_name.value
        return self

class Dataset_dir_builder(Dataset_basic_builder):
    def set_dir_manually(self, train_in_dir=None, train_gt_dir=None, test_in_dir=None, test_gt_dir=None, see_in_dir=None, see_gt_dir=None):
        self.db.train_in_dir = train_in_dir
        self.db.train_gt_dir = train_gt_dir
        self.db.test_in_dir  = test_in_dir
        self.db.test_gt_dir  = test_gt_dir
        self.db.see_in_dir   = see_in_dir
        self.db.see_gt_dir   = see_gt_dir
        return self

    def set_dir_by_basic(self):
        in_dir_name = ""
        gt_dir_name = ""
        if  (self.db.get_method == DB_GET_METHOD.in_dis_gt_move_map) :
            in_dir_name = "dis_imgs"
            gt_dir_name = "move_maps"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_ord_pad)  :
            in_dir_name = "dis_imgs"
            gt_dir_name = "gt_ord_pad_imgs"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_ord)      :
            in_dir_name = "dis_imgs"
            gt_dir_name = "gt_ord_imgs"
        elif(self.db.get_method == DB_GET_METHOD.in_rec_gt_ord)      :
            in_dir_name = "unet_rec_imgs"
            gt_dir_name = "gt_ord_imgs"
        elif(self.db.get_method == DB_GET_METHOD.in_dis_gt_flow)      :
            in_dir_name = "dis_imgs"
            gt_dir_name = "flows"


        self.db.train_in_dir = self.db.db_dir + "/train/" + in_dir_name
        self.db.train_gt_dir = self.db.db_dir + "/train/" + gt_dir_name
        self.db.test_in_dir  = self.db.db_dir + "/test/"  + in_dir_name
        self.db.test_gt_dir  = self.db.db_dir + "/test/"  + gt_dir_name
        self.db.see_in_dir   = self.db.db_dir + "/see/"   + in_dir_name
        self.db.see_gt_dir   = self.db.db_dir + "/see/"   + gt_dir_name
        return self

class Dataset_type_builder(Dataset_dir_builder):
    def set_in_gt_type(self, in_type="bmp", gt_type="bmp", see_type="bmp"):
        self.db.in_type  = in_type
        self.db.gt_type  = gt_type
        self.db.see_type = see_type
        return self

class Dataset_detail_builder(Dataset_type_builder):
    def set_detail(self, have_train=True, have_see=False):
        self.db.have_train    = have_train
        self.db.have_see      = have_see
        return self

class Dataset_builder(Dataset_detail_builder): pass


### Enum改短名字的概念
DB_C = DB_CATEGORY
DB_N = DB_NAME
DB_GM = DB_GET_METHOD
### 直接先建好 obj 給外面import囉！
type5c_real_have_see_no_bg_gt_color          = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color_gray3ch, DB_N.no_bg_gt_color        , DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_type(in_type="bmp", gt_type="bmp", see_type="bmp").set_detail(have_train=True, have_see=True).build()
type6_h384_w256_smooth_curl_fold_page        = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page, DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_type(in_type="bmp", gt_type="npy", see_type="npy").set_detail(have_train=True, have_see=True).build()
type7_h472_w304_real_os_book_400data         = Dataset_builder().set_basic(DB_C.type7_h472_w304_real_os_book               , DB_N.os_book_400data       , DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
type7b_h500_w332_real_os_book_1532data       = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book              , DB_N.os_book_1532data      , DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
type7b_h500_w332_real_os_book_1532data_focus = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book              , DB_N.os_book_1532data_focus, DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
type7b_h500_w332_real_os_book_1532data_big   = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book              , DB_N.os_book_1532data_big  , DB_GM.in_dis_gt_ord, h=600, w=396).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
type7b_h500_w332_real_os_book_400data        = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book              , DB_N.os_book_400data       , DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
type7b_h500_w332_real_os_book_800data        = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book              , DB_N.os_book_800data       , DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
type8_blender_os_book                        = Dataset_builder().set_basic(DB_C.type8_blender_os_book                      , DB_N.blender_os_hw756      , DB_GM.in_dis_gt_flow, h=756, w=756).set_dir_by_basic().set_in_gt_type(in_type="png", gt_type="knpy", see_type=None).set_detail(have_train=True, have_see=False).build()


if(__name__=="__main__"):
    db = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color_gray3ch, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_type(in_type="bmp", gt_type="bmp", see_type="bmp").set_detail(have_train=True, have_see=True).build()
    db = Dataset_builder().set_basic(DB_C.type7_h472_w304_real_os_book, DB_N.os_book_400data, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
    db = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book, DB_N.os_book_1532data, DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_type(in_type="jpg", gt_type="jpg", see_type="jpg").set_detail(have_train=True, have_see=True).build()
    print(type8_blender_os_book)
    # db_complex_1_pure_unet = Datasets(DB_CATEGORY.type1_h_256_w_256_complex, DB_GET_METHOD.in_dis_gt_move_map   , h=256, w=256 )
    # db_complex_2_pure_rect = Datasets(DB_CATEGORY.type1_h_256_w_256_complex, DB_GET_METHOD.in_dis_gt_ord_pad_img, h=256, w=256 )
    # db_complex_3_pure_rect = Datasets(DB_CATEGORY.type1_h_256_w_256_complex, DB_GET_METHOD.in_rec_gt_ord_img    , h=256, w=256 )


