from step11_b_result_obj_builder import Result_builder
import copy
"""
這個.py主要是要 把 系列的exp 從step10a 裡面抽出來，包成 exps 給 step12 用

系列之間的 dependency 是有順序性的，：
應該都會是 某系列做完，覺得哪邊可以改，再坐下一個系列
如果發現 要用的還沒宣告，應該是自己系列的設計有問題
ex:可能有 兩個不同系列的東西混到一起來看之類的，這樣的話就區分出兩個系列即可拉~~

group寫法1(但缺點很大目前不採用)：用Result_bulder()重新在這裡建構
    優點：速度快，因為只建構Result，不需要跟 Exp一樣 有db, model, board, loss, ...複雜的建構過程
    缺點：不好維護，因為我result_dir可能會改名字，這樣子 我 step10_a 改完result_name，這裡又要跟著改一次！太麻煩且易改錯！
"""
dir04_01 = "04_01_bg_&_gt_color"
have_bg_gt_gray_mae3  = Result_builder().set_by_result_name(dir04_01 + "/type5d_real_have_see_have_bg-gt_gray3ch-20200428-152656_model5_rect2-have_bg_gt_gray", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("have_bg_gt_gray").build()
have_bg_gt_color_mae3 = Result_builder().set_by_result_name(dir04_01 + "/type5d_real_have_see_have_bg-gt_color-20200428-153059_model5_rect2-have_bg_gt_color", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("have_bg_gt_color").build()
no_bg_gt_color_mae3   = Result_builder().set_by_result_name(dir04_01 + "/type5c_real_have_see_no_bg-gt_color-20200428-132611_model5_rect2-no_bg_gt_color", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("no_bg_gt_color").build()
no_bg_gt_gray_mae3    = Result_builder().set_by_result_name(dir04_01 + "/type5c_real_have_see_no_bg-gt_gray3ch-20200428-011344_model5_rect2-no_bg_gt_gray", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("no_bg_gt_gray").build()
# have_bg_gt_gray_mae3  = Result("04_01_bg_&_gt_color/type5d-real_have_see-have_bg-gt_gray3ch_20200428-152656_model5_rect2", r_describe="have_bg_gt_gray")
# have_bg_gt_color_mae3 = Result("04_01_bg_&_gt_color/type5d-real_have_see-have_bg-gt_color_20200428-153059_model5_rect2"  , r_describe="have_bg_gt_color")
# no_bg_gt_color_mae3   = Result("04_01_bg_&_gt_color/type5c-real_have_see-no_bg-gt-color_20200428-132611_model5_rect2"    , r_describe="no_bg_gt_color")
# no_bg_gt_gray_mae3    = Result("04_01_bg_&_gt_color/type5c_real_have_see_no_bg-gt_gray3ch_20200428-011344_model5_rect2"  , r_describe="no_bg_gt_gray")


dir04_02 = "04_02_no_mrf_mae136"
no_mrf_mae1 = Result_builder().set_by_result_name(dir04_02 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200506-064552_model5_rect2-no_mrf_mae1", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("no_mrf_mae1").build()
no_mrf_mae3 = Result_builder().set_by_result_name(dir04_02 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200428-011344_model5_rect2-no_mrf_mae3", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("no_mrf_mae3").build()
no_mrf_mae6 = Result_builder().set_by_result_name(dir04_02 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200506-065346_model5_rect2-no_mrf_mae6", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("no_mrf_mae6").build()
# no_mrf_mae1 = Result("04_02_no_mrf_mae136/type5c_real_have_see_no_bg-gt_gray3ch_20200506-064552_model5_rect2" , r_describe="no_mrf_mae1")
# no_mrf_mae3 = Result("04_02_no_mrf_mae136/type5c_real_have_see_no_bg-gt_gray3ch_20200428-011344_model5_rect2" , r_describe="no_mrf_mae3")
# no_mrf_mae6 = Result("04_02_no_mrf_mae136/type5c_real_have_see_no_bg-gt_gray3ch_20200506-065346_model5_rect2" , r_describe="no_mrf_mae6")


dir04_03 = "04_03_mrf_mae136"
mrf_7_9_1 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-190344_model6_mrf_rect2-mrf_7_9_mae1", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_7_9_mae1").build()
mrf_7_9_3 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200429-145226_model6_mrf_rect2-mrf_7_9_mae3", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_7_9_mae3").build()
mrf_7_9_6 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231036_model6_mrf_rect2-mrf_7_9_mae6", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_7_9_mae6").build()
mrf_7_11_1 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-190955_model6_mrf_rect2-mrf_7_11_mae1", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_7_11_mae1").build()
mrf_7_11_3 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200429-150505_model6_mrf_rect2-mrf_7_11_mae3", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_7_11_mae3").build()
mrf_7_11_6 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231336_model6_mrf_rect2-mrf_7_11_mae6", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_7_11_mae6").build()
mrf_9_11_1 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-190837_model6_mrf_rect2-mrf_9_11_mae1", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_9_11_mae1").build()
mrf_9_11_3 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200429-145548_model6_mrf_rect2-mrf_9_11_mae3", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_9_11_mae3").build()
mrf_9_11_6 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231249_model6_mrf_rect2-mrf_9_11_mae6", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_9_11_mae6").build()
mrf_13579_1 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-191110_model6_mrf_rect2-mrf_13579_mae1", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_13579_mae1").build()
mrf_13579_3 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200428-154149_model6_mrf_rect2-mrf_13579_mae3", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_13579_mae3").build()
mrf_13579_6 = Result_builder().set_by_result_name(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231530_model6_mrf_rect2-mrf_13579_mae6", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("mrf_13579_mae6").build()
# mrf_7_9_1   = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-190344_model6_mrf_rect2" , r_describe="mrf_7_9_mae1")
# mrf_7_9_3   = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200429-145226_model6_mrf_rect2" , r_describe="mrf_7_9_mae3")
# mrf_7_9_6   = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231036_model6_mrf_rect2" , r_describe="mrf_7_9_mae6")
# mrf_7_11_1  = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-190955_model6_mrf_rect2" , r_describe="mrf_7_11_mae1")
# mrf_7_11_3  = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200429-150505_model6_mrf_rect2" , r_describe="mrf_7_11_mae3")
# mrf_7_11_6  = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231336_model6_mrf_rect2" , r_describe="mrf_7_11_mae6")
# mrf_9_11_1  = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-190837_model6_mrf_rect2" , r_describe="mrf_9_11_mae1")
# mrf_9_11_3  = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200429-145548_model6_mrf_rect2" , r_describe="mrf_9_11_mae3")
# mrf_9_11_6  = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231249_model6_mrf_rect2" , r_describe="mrf_9_11_mae6")
# mrf_13579_1 = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200504-191110_model6_mrf_rect2" , r_describe="mrf_13579_mae1")
# mrf_13579_3 = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200428-154149_model6_mrf_rect2" , r_describe="mrf_13579_mae3")
# mrf_13579_6 = Result(dir04_03 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200501-231530_model6_mrf_rect2" , r_describe="mrf_13579_mae6")


dir04_04 = "04_04_mrf_replace7"
mrf_replace7_use7 = Result_builder().set_by_result_name(dir04_04 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200507-105001_model6_mrf_rect2-replace7_use7",   in_use_range="0~1", gt_use_range="0~1").set_ana_describe("replace7_use7").build()
mrf_replace7_use5_7 = Result_builder().set_by_result_name(dir04_04 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200507-105739_model6_mrf_rect2-replace7_use5+7", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("replace7_use5+7").build()
mrf_replace7_use7_9 = Result_builder().set_by_result_name(dir04_04 + "/type5c_real_have_see_no_bg-gt_gray3ch_20200507-110022_model6_mrf_rect2-replace7_use7+9", in_use_range="0~1", gt_use_range="0~1").set_ana_describe("replace7_use7+9").build()
# mrf_replace7_use7   = Result("04_04_mrf_replace7/type5c_real_have_see_no_bg-gt_gray3ch_20200507-105001_model6_mrf_rect2" , r_describe="replace7_use7")
# mrf_replace7_use5_7 = Result("04_04_mrf_replace7/type5c_real_have_see_no_bg-gt_gray3ch_20200507-105739_model6_mrf_rect2" , r_describe="replace7_use5+7")
# mrf_replace7_use7_9 = Result("04_04_mrf_replace7/type5c_real_have_see_no_bg-gt_gray3ch_20200507-110022_model6_mrf_rect2" , r_describe="replace7_use7+9")

### 分析 bg 和 gt_color
bg_and_gt_color_results = [have_bg_gt_gray_mae3, have_bg_gt_color_mae3, no_bg_gt_color_mae3, no_bg_gt_gray_mae3]

### 分析 mrf loss mae的比例
mrf_r_c_results = [
                        [no_mrf_mae1, mrf_7_9_1, mrf_7_11_1, mrf_9_11_1, mrf_13579_1],
                        [no_mrf_mae3, mrf_7_9_3, mrf_7_11_3, mrf_9_11_3, mrf_13579_3],
                        [no_mrf_mae6, mrf_7_9_6, mrf_7_11_6, mrf_9_11_6, mrf_13579_6]
                    ]

### 分析 mrf 取代 第一層7
mrf_replace7_results = [no_mrf_mae3, mrf_replace7_use7, mrf_replace7_use5_7, mrf_replace7_use7_9]



dir01 = "05_01_rect_mae136"
os_book_rect_1532_mae1 = Result_builder().set_by_result_name(dir01 + "/type7b_h500_w332_real_os_book-20200524-013834_rect_1532data_mae1_127.28", in_use_range="-1~1", gt_use_range="-1~1")           .set_ana_describe("DGmae1").build()
os_book_rect_1532_mae3 = Result_builder().set_by_result_name(dir01 + "/type7b_h500_w332_real_os_book-20200524-012601-rect-1532data_mae3_127.35to28_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("DGmae3").build()
os_book_rect_1532_mae6 = Result_builder().set_by_result_name(dir01 + "/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242", in_use_range="-1~1", gt_use_range="-1~1")          .set_ana_describe("DGmae6").build()
os_book_rect_1532_mae6_noD = Result_builder().set_by_result_name(dir01 + "/type7b_h500_w332_real_os_book-20200524-014129_rect_1532data_mae6_128.242", in_use_range="-1~1", gt_use_range="-1~1")      .set_ana_describe("DGmae6").build()

dir03 = "05_03_justG_mae136920"
os_book_justG_mae1  = Result_builder().set_by_result_name(dir03 + "/type7b_h500_w332_real_os_book-20200525_222831-justG-1532data_mae1_127.28", in_use_range="-1~1", gt_use_range="-1~1")  .set_ana_describe("justG_mae1").build()
os_book_justG_mae3  = Result_builder().set_by_result_name(dir03 + "/type7b_h500_w332_real_os_book-20200524-181909-justG-1532data_mae3_128.242", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("justG_mae3").build()
os_book_justG_mae6  = Result_builder().set_by_result_name(dir03 + "/type7b_h500_w332_real_os_book-20200525_223733-justG-1532data_mae6_128.246", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("justG_mae6").build()
os_book_justG_mae9  = Result_builder().set_by_result_name(dir03 + "/type7b_h500_w332_real_os_book-20200525_225555-justG-1532data_mae9_127.51", in_use_range="-1~1", gt_use_range="-1~1")  .set_ana_describe("justG_mae9").build()
os_book_justG_mae20 = Result_builder().set_by_result_name(dir03 + "/type7b_h500_w332_real_os_book-20200527_122823-justG-1532data_mae20_128.246", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("justG_mae20").build()

dir02 = "05_02_DG_vs_justG"
os_book_rect_D10        = copy.deepcopy(os_book_rect_1532_mae3);   os_book_rect_D10.ana_describe = "rect_mae3_D1.0"
os_book_rect_D05        = Result_builder().set_by_result_name(dir02 + "/type7b_h500_w332_real_os_book-20200527_073801-rect-1532data_D_0.5_128.245", in_use_range="-1~1", gt_use_range="-1~1")          .set_ana_describe("rect_mae3_D0.5").build()
os_book_rect_D025       = Result_builder().set_by_result_name(dir02 + "/type7b_h500_w332_real_os_book-20200602_233505-rect-1532data_D_0.25_128.245_epoch211", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("rect_mae3_D0.25").build()
os_book_rect_D01        = Result_builder().set_by_result_name(dir02 + "/type7b_h500_w332_real_os_book-20200527_120840-rect-1532data_D_0.1_127.28", in_use_range="-1~1", gt_use_range="-1~1")           .set_ana_describe("rect_mae3_D0.1").build()
os_book_rect_D00        = Result_builder().set_by_result_name(dir02 + "/type7b_h500_w332_real_os_book-20200525-134838-rect-1532data_D_0.0", in_use_range="-1~1", gt_use_range="-1~1")                  .set_ana_describe("rect_mae3_D0.0").build()
os_book_rect_D00_justG = copy.deepcopy(os_book_justG_mae3);   os_book_rect_D00_justG.ana_describe = "rect_mae3_D0.0_justG"

dir04 = "05_04_bigger_smaller"
os_book_justG_bigger       = Result_builder().set_by_result_name(dir04 + "/type7b_h500_w332_real_os_book-20200615_030658-justG-1532data_mae3_SRbig_127.35", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("justG_mae3_bigger").build()
os_book_justG_bigger_wrong = Result_builder().set_by_result_name(dir04 + "/type7b_h500_w332_real_os_book-20200601_022935-justG-1532data_bigger_wrong", in_use_range="-1~1", gt_use_range="-1~1")      .set_ana_describe("justG_mae3_bigger_wrong").build()
os_book_justG_normal       = copy.deepcopy(os_book_justG_mae3);   os_book_justG_normal.ana_describe = "justG_mae3_normal"
os_book_justG_smaller      = Result_builder().set_by_result_name(dir04 + "/type7b_h500_w332_real_os_book-20200601_081803-justG-1532data_smaller_128.246", in_use_range="-1~1", gt_use_range="-1~1")              .set_ana_describe("justG_mae3_smaller").build()
os_book_justG_smaller2     = Result_builder().set_by_result_name(dir04 + "/type7b_h500_w332_real_os_book-20200602_163853-justG-1532data_mae3_127.48_smaller2_epoch425", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("justG_mae3_smaller2").build()

dir05 = "05_05_focus"
os_book_rect_nfocus        = copy.deepcopy(os_book_rect_1532_mae3);   os_book_rect_nfocus.ana_describe = "no_focus_rect_mae3"
os_book_rect_focus         = Result_builder().set_by_result_name(dir05 + "/type7b_h500_w332_real_os_book-20200601_224919-rect-1532data_mae3_focus_127.35_finish", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("focus_rect_mae3").build()
os_book_justG_nfocus       = copy.deepcopy(os_book_justG_mae3);  os_book_justG_nfocus.ana_describe = "no_focus_justG_mae3"
os_book_justG_focus        = Result_builder().set_by_result_name(dir05 + "/type7b_h500_w332_real_os_book-20200602_222505-justG-1532data_mae3_focus_128.246_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("focus_justG_mae3").build()


dir06 = "05_06_400"
os_book_400_rect         = Result_builder().set_by_result_name(dir06 + "/type7b_h500_w332_real_os_book-20200602_201009-rect-400data_mae3_127.35to51_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("400_rect").build()
os_book_400_justG        = Result_builder().set_by_result_name(dir06 + "/type7b_h500_w332_real_os_book-20200603_133217-justG-400data_justG_mae3_127.28", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("400_justG").build()
os_book_1532_rect        = copy.deepcopy(os_book_rect_1532_mae3);  os_book_1532_rect.ana_describe = "1532_rect"
os_book_1532_justG       = copy.deepcopy(os_book_justG_mae3);  os_book_1532_justG.ana_describe = "1532_justG"

dir07 = "05_07_first_k7_vs_k3"
os_book_GD_first_k7     = copy.deepcopy(os_book_rect_1532_mae3);  os_book_GD_first_k7.ana_describe = "GD_first_k7"
os_book_GD_first_k3     = Result_builder().set_by_result_name(dir07 + "/type7b_h500_w332_real_os_book-7_4-20200622_205606-rect-1532data_rect_firstk3_finish", in_use_range="-1~1", gt_use_range="-1~1")  .set_ana_describe("G_first_k3").build()
os_book_G_first_k7      = copy.deepcopy(os_book_justG_mae3);  os_book_G_first_k7.ana_describe = "G_first_k7"
os_book_G_first_k3      = Result_builder().set_by_result_name(dir07 + "/type7b_h500_w332_real_os_book-7_2-20200622_205312-justG-1532data_firstk3_127.246finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("GD_first_k3").build()


dir08a = "05_08a_GD_mrf"
os_book_GD_no_mrf = copy.deepcopy(os_book_rect_1532_mae3);  os_book_GD_no_mrf.ana_describe = "GD_no_mrf"
os_book_GD_mrf_7         = Result_builder().set_by_result_name(dir08a + "/type7b_h500_w332_real_os_book-20200621_143739-mrf_rect-1532data_mrf_7-127.48_to_128.246_epoch514", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("GD_mrf_7").build()
os_book_GD_mrf_79        = Result_builder().set_by_result_name(dir08a + "/type7b_h500_w332_real_os_book-20200621_144146-mrf_rect-1532data_mrf_79_128.245_epoch511", in_use_range="-1~1", gt_use_range="-1~1")         .set_ana_describe("GD_mrf_79").build()
os_book_GD_mrf_replace7  = Result_builder().set_by_result_name(dir08a + "/type7b_h500_w332_real_os_book-20200621_123201-mrf_rect-1532data_mrf_replace7_127.35_epoch515", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("GD_mrf_replace7").build()
os_book_GD_mrf_replace79 = Result_builder().set_by_result_name(dir08a + "/type7b_h500_w332_real_os_book-20200621_123636-mrf_rect-1532data_mrf_replace79_127.51_epoch514", in_use_range="-1~1", gt_use_range="-1~1")   .set_ana_describe("GD_mrf_replace79").build()

# ##################################################################################################################################################################
dir08b = "05_08b_1_G_mrf"
os_book_G_no_mrf = copy.deepcopy(os_book_justG_mae3);  os_book_G_no_mrf.ana_describe = "G_no_mrf"
os_book_G_mrf_7         = Result_builder().set_by_result_name(dir08b + "/type7b_h500_w332_real_os_book-5_8b_2-20200623_191110-justG-1532data_justG_mrf_7_128.245ep510", in_use_range="-1~1", gt_use_range="-1~1")       .set_ana_describe("G_mrf_7").build()
os_book_G_mrf_79        = Result_builder().set_by_result_name(dir08b + "/type7b_h500_w332_real_os_book-5_8b_3-20200624_002925-justG-1532data_justG_mrf_79_127.55ep479", in_use_range="-1~1", gt_use_range="-1~1")       .set_ana_describe("G_mrf_79").build()
os_book_G_mrf_replace7  = Result_builder().set_by_result_name(dir08b + "/type7b_h500_w332_real_os_book-5_8b_4-20200622_214400-justG-1532data_justG_mrf_replace7_127.48ep479", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("G_mrf_replace7").build()
os_book_G_mrf_replace79 = Result_builder().set_by_result_name(dir08b + "/type7b_h500_w332_real_os_book-5_8b_5-20200623_210009-justG-1532data_justG_mrf_replace79_127.51ep566", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf_replace79").build()

### 8b_2. G_no_mrf_firstk3 vs G + mrf  7 /5 /3 + first_k3
dir08b_2 = "05_08b_2_G_mrf357_k3"
os_book_G_no_mrf_firstk7 = copy.deepcopy(os_book_justG_mae3); os_book_G_no_mrf_firstk7.ana_describe = "G_no_mrf_firstk7"
os_book_G_no_mrf_firstk3 = copy.deepcopy(os_book_G_first_k3); os_book_G_no_mrf_firstk3.ana_describe = "G_no_mrf_firstk3"
os_book_G_mrf_7_firstk7  = copy.deepcopy(os_book_G_mrf_7);    os_book_G_mrf_7_firstk7 .ana_describe = "G_mrf_7_firstk7"
os_book_G_mrf_7_firstk3 = Result_builder().set_by_result_name(dir08b_2 + "/type7b_h500_w332_real_os_book-5_8b_2b-20200626_055221-justG_mrf7_k3-128.51_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf7_firstk3").build()
os_book_G_mrf_5_firstk3 = Result_builder().set_by_result_name(dir08b_2 + "/type7b_h500_w332_real_os_book-5_8b_2c-20200628_081636-justG_mrf5_k3-128.51_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf5_firstk3").build()
os_book_G_mrf_3_firstk3 = Result_builder().set_by_result_name(dir08b_2 + "/type7b_h500_w332_real_os_book-5_8b_2d-20200628_091752-justG_mrf3_k3-128.48finish", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("G_mrf3_firstk3").build()

### 8b_3. G + mrf 97/75/53 + first_k3
dir08b_3 = "05_08b_3_G_mrf97,75,53"
os_book_G_no_mrf_firstk7 = copy.deepcopy(os_book_justG_mae3); os_book_G_no_mrf_firstk7.ana_describe = "G_no_mrf_firstk7"
os_book_G_no_mrf_firstk3 = copy.deepcopy(os_book_G_first_k3); os_book_G_no_mrf_firstk3.ana_describe = "G_no_mrf_firstk3"
os_book_G_mrf_79_firstk7 = copy.deepcopy(os_book_G_mrf_79);   os_book_G_mrf_79_firstk7.ana_describe = "G_mrf_79_firstk7"
os_book_G_mrf_79_firstk3 = Result_builder().set_by_result_name(dir08b_3 + "/type7b_h500_w332_real_os_book-5_8b_3b-20200626_055435-justG_mrf79-128.246ep528", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf79_firstk3").build()
os_book_G_mrf_57_firstk3 = Result_builder().set_by_result_name(dir08b_3 + "/type7b_h500_w332_real_os_book-5_8b_3c-20200628_082724-justG_mrf57-128.246ep568", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf57_firstk3").build()
os_book_G_mrf_35_firstk3 = Result_builder().set_by_result_name(dir08b_3 + "/type7b_h500_w332_real_os_book-5_8b_3d-20200628_094532-justG_mrf35-127.35ep671", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("G_mrf35_firstk3").build()

### 8b_4. G + mrf replace 7/5/3
dir08b_4 = "05_08b_4_G_mrf_replace357"
os_book_G_no_mrf         = copy.deepcopy(os_book_justG_mae3);     os_book_G_no_mrf.ana_describe = "G_no_mrf"
os_book_G_mrf_replace7   = copy.deepcopy(os_book_G_mrf_replace7); os_book_G_mrf_replace7.ana_describe = "G_mrf_replace7"
os_book_G_mrf_replace5   = Result_builder().set_by_result_name(dir08b_4 + "/type7b_h500_w332_real_os_book-5_8b_4b-20200626_052721-justG_mrf_replace5-127.35", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf_replace5").build()
os_book_G_mrf_replace3   = Result_builder().set_by_result_name(dir08b_4 + "/type7b_h500_w332_real_os_book-5_8b_4c-20200626_053735-justG_mrf_replace3-127.48", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf_replace3").build()

### 8b_5. G + mrf replace 97/75/53
dir08b_5 = "05_08b_5_G_mrf_replace57,35"
os_book_G_no_mrf         = copy.deepcopy(os_book_justG_mae3);      os_book_G_no_mrf       .ana_describe = "G_no_mrf"
os_book_G_mrf_replace79  = copy.deepcopy(os_book_G_mrf_replace79); os_book_G_mrf_replace79.ana_describe = "G_mrf_replace79"
os_book_G_mrf_replace57  = Result_builder().set_by_result_name(dir08b_5 + "/type7b_h500_w332_real_os_book-5_8b_5b-20200626_084855-justG_mrf_replace79-128.55ep57_to127.28ep600", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("G_mrf_replace57").build()
os_book_G_mrf_replace35  = Result_builder().set_by_result_name(dir08b_5 + "/type7b_h500_w332_real_os_book-5_8b_5c-20200626_054200-justG_mrf_replace35-128.28ep687", in_use_range="-1~1", gt_use_range="-1~1")             .set_ana_describe("G_mrf_replace35").build()

### 8c_G_mrf_3,4branch 135,357,3579
dir08c = "05_08c_Gk3_mrf_3,4branch"
os_book_G_mrf_35_firstk3 = copy.deepcopy(os_book_G_mrf_35_firstk3); os_book_G_mrf_35_firstk3.ana_describe = "Gk3_mrf35"
os_book_G_mrf_135 = Result_builder().set_by_result_name (dir08c + "/type7b_h500_w332_real_os_book-5_8c2_Gmrf135-20200701_192915-justGk3_mrf135-128.246_finish", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk3_mrf135").build()
os_book_G_mrf_357 = Result_builder().set_by_result_name (dir08c + "/type7b_h500_w332_real_os_book-5_8c3_Gmrf357-20200701_192639-justGk3_mrf357-127.51_finish", in_use_range="-1~1", gt_use_range="-1~1")  .set_ana_describe("Gk3_mrf357").build()
os_book_G_mrf_3579 = Result_builder().set_by_result_name(dir08c + "/type7b_h500_w332_real_os_book-5_8c4_Gmrf3579-20200701_192955-justGk3_mrf3579-127.28_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_mrf3579").build()

### 8d_GD_mrf_3,4branch 135,357,3579
dir08d = "05_08d_Gk3D_mrf_3,4branch"
os_book_GD_mrf_35   = Result_builder().set_by_result_name(dir08d + "/type7b_h500_w332_real_os_book-5_8d1_Gmrf35-20200707_175514-rect_mrf35_Gk3_DnoC_k4-127.55finish", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("Gk3_mrf35").build()
os_book_GD_mrf_135  = Result_builder().set_by_result_name(dir08d + "/type7b_h500_w332_real_os_book-5_8d2_Gmrf135-20200705_040804-rect_mrf135_Gk3_DnoC_k4-128.246_finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_DnoCK4_mrf135").build()
os_book_GD_mrf_357  = Result_builder().set_by_result_name(dir08d + "/type7b_h500_w332_real_os_book-5_8d3_Gmrf357-20200705_040906-rect_mrf357_Gk3_DnoC_k4-127.51finish", in_use_range="-1~1", gt_use_range="-1~1")  .set_ana_describe("Gk3_DnoCK4_mrf357").build()
os_book_GD_mrf_3579 = Result_builder().set_by_result_name(dir08d + "/type7b_h500_w332_real_os_book-5_8d4_Gmrf3579-20200705_040953-rect_mrf3579_Gk3_DnoC_k4-127.28finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_DnoCK4_mrf3579").build()



##################################################################################################################################################################
### 9a. Gk7_D_concat_try_and_k3_4try
dir09a = "05_09a_Gk7_D_concat_try_and_k3_4try"
os_book_Gk7_D_concat_k4    = copy.deepcopy(os_book_rect_1532_mae3); os_book_Gk7_D_concat_k4.ana_describe = "Gk7_D_concat_k4"
os_book_Gk7_D_concat_k3    = Result_builder().set_by_result_name(dir09a + "/type7b_h500_w332_real_os_book-5_9a_2-20200630_060217-rect_Gk7_D_concat_k3-127.51ep618", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("Gk7_D_concat_k3").build()
os_book_Gk7_D_no_concat_k4 = Result_builder().set_by_result_name(dir09a + "/type7b_h500_w332_real_os_book-5_9a_3-20200630_060549-rect_Gk7_D_no_concat_k4-128.246ep588", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk7_D_no_concat_k4").build()
os_book_Gk7_D_no_concat_k3 = Result_builder().set_by_result_name(dir09a + "/type7b_h500_w332_real_os_book-5_9a_4-20200630_061213-rect_Gk7_D_no_concat_k3-127.28ep607", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk7_D_no_concat_k3").build()
os_book_Gk7_no_D           = copy.deepcopy(os_book_G_first_k7); os_book_Gk7_no_D.ana_describe = "Gk7_no_D"

### 9b. Gk3_D_concat_try_and_k3_4try
dir09b = "05_09b_Gk3_D_concat_try_and_k3_4try"
os_book_Gk3_D_concat_k4    = Result_builder().set_by_result_name(dir09b + "/type7b_h500_w332_real_os_book-5_9b_1-20200726_075333-rect_Gk3_D_concat_k4", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("Gk3_D_concat_k4").build()
os_book_Gk3_D_concat_k3    = Result_builder().set_by_result_name(dir09b + "/type7b_h500_w332_real_os_book-5_9b_2-20200726_075415-rect_Gk3_D_concat_k3", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("Gk3_D_concat_k3").build()
os_book_Gk3_D_no_concat_k4 = Result_builder().set_by_result_name(dir09b + "/type7b_h500_w332_real_os_book-5_9b_3-20200711_002538-rect_Gk3_D_no_concat_k4-127.55finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_D_no_concat_k4").build()
os_book_Gk3_D_no_concat_k3 = Result_builder().set_by_result_name(dir09b + "/type7b_h500_w332_real_os_book-5_9b_4-20200711_002752-rect_Gk3_D_no_concat_k3-127.48finish", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk3_D_no_concat_k3").build()
os_book_Gk3_no_D           = copy.deepcopy(os_book_G_first_k3); os_book_Gk3_no_D.ana_describe = "Gk3_no_D"

##################################################################################################################################################################
dir10 = "05_10_GD_D_train1_G_train_135"
os_book_D1G1 = copy.deepcopy(os_book_Gk3_D_no_concat_k4);  os_book_D1G1.ana_describe = "D1_G1"
os_book_D1G3 = Result_builder().set_by_result_name(dir10 + "/type7b_h500_w332_real_os_book-5_10_2-20200727_155142-rect_Gk3_train3_Dk4_no_concat", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("D1_G3").build()
os_book_D1G5 = Result_builder().set_by_result_name(dir10 + "/type7b_h500_w332_real_os_book-5_10_3-20200729_153512-rect_Gk3_train5_Dk4_no_concat", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("D1_G5").build()

##################################################################################################################################################################
dir11 = "05_11_no_res"
os_book_Gk3_res        = copy.deepcopy(os_book_G_first_k3);  os_book_Gk3_res.ana_describe = "Gk3_res"
os_book_Gk3_no_res     = Result_builder().set_by_result_name(dir11 + "/type7b_h500_w332_real_os_book-5_11_1-20200711_000930-justGk3_no_res-127.51finish", in_use_range="-1~1", gt_use_range="-1~1")        .set_ana_describe("justG_fk3_no_res").build()

os_book_G_mrf_357_res    = copy.deepcopy(os_book_G_mrf_357);  os_book_G_mrf_357_res.ana_describe = "Gk3_mrf357_res"
os_book_G_mrf_357_no_res = Result_builder().set_by_result_name(dir11 + "/type7b_h500_w332_real_os_book-5_11_3-20200711_001249-justG_mrf357_no_res-128.246finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_mrf357_no_res").build()


os_book_Gk3_Dk4_no_concat_res    = copy.deepcopy(os_book_Gk3_D_no_concat_k4);  os_book_Gk3_Dk4_no_concat_res.ana_describe = "Gk3_Dk4_no_concat_res"
os_book_Gk3_Dk4_no_concat_no_res = Result_builder().set_by_result_name(dir11 + "/type7b_h500_w332_real_os_book-5_11_2-20200711_001122-rect_Gk3_no_res_D_no_concat-127.28finish", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_Dk4_no_concat_no_res").build()
##################################################################################################################################################################
dir12 = "05_12_resb_num"
os_book_Gk3_resb0     = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_1-20200903_102550-justGk3_resb00-127.48", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk3_resb_0").build()
os_book_Gk3_resb1     = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_2-20200903_150017-justGk3_resb01-127.35", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk3_resb_1").build()
os_book_Gk3_resb3     = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_3-20200908_230930-justGk3_resb03-127.55", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk3_resb_3").build()
os_book_Gk3_resb5     = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_4-20200903_102828-justGk3_resb05-128.246", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_resb_5").build()
os_book_Gk3_resb7     = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_5-20200908_104030-justGk3_resb07-127.28", in_use_range="-1~1", gt_use_range="-1~1") .set_ana_describe("Gk3_resb_7").build()
os_book_Gk3_resb9     = copy.deepcopy(os_book_G_first_k3); os_book_Gk3_resb9.ana_describe = "Gk3_resb_9"
os_book_Gk3_resb11    = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_7-20200908_103946-justGk3_resb11-127.51-ep512", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_resb_11").build()
os_book_Gk3_resb20    = Result_builder().set_by_result_name(dir12 + "/type7b_h500_w332_real_os_book-5_12_8-20200903_102218-justGk3_resb20-127.51", in_use_range="-1~1", gt_use_range="-1~1")      .set_ana_describe("Gk3_resb_20").build()

##################################################################################################################################################################
dir13 = "05_13_coord_conv"
os_book_Gk3_no_coord_conv         = copy.deepcopy(os_book_G_first_k3);  os_book_Gk3_no_coord_conv.ana_describe = "Gk3_no_coord"
os_book_Gk3_coord_conv_first      = Result_builder().set_by_result_name(dir13 + "/type7b_h500_w332_real_os_book-5_13_1-20200824_033322-justG-127.35_first", in_use_range="-1~1", gt_use_range="-1~1")    .set_ana_describe("Gk3_first").build()
os_book_Gk3_coord_conv_first_end  = Result_builder().set_by_result_name(dir13 + "/type7b_h500_w332_real_os_book-5_13_2-20200827_010445-justG-127.35_first_end", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_first_end").build()
os_book_Gk3_coord_conv_all        = Result_builder().set_by_result_name(dir13 + "/type7b_h500_w332_real_os_book-5_13_3-20200827_000519-justG-127.35_all", in_use_range="-1~1", gt_use_range="-1~1")      .set_ana_describe("Gk3_all").build()

os_book_Gk3_mrf_357_no_coord_conv       = copy.deepcopy(os_book_G_mrf_357);  os_book_Gk3_mrf_357_no_coord_conv.ana_describe = "Gk3_mrf357_no_coord"
os_book_Gk3_mrf357_coord_conv_first     = Result_builder().set_by_result_name(dir13 + "/type7b_h500_w332_real_os_book-5_13_2-20200824_033403-justG_mrf357-127.28-first", in_use_range="-1~1", gt_use_range="-1~1")   .set_ana_describe("Gk3_mrf357_first").build()
# os_book_Gk3_mrf357_coord_conv_first_end = Result_builder().set_by_result_name(dir13 + "/待訓練").set_ana_describe("Gk3_Dk4_no_concat_no_res").build()
os_book_Gk3_mrf357_coord_conv_all       = Result_builder().set_by_result_name(dir13 + "/type7b_h500_w332_real_os_book-5_13_2-20200827_000320-justG_mrf357-127.28_all_ep581", in_use_range="-1~1", gt_use_range="-1~1").set_ana_describe("Gk3_mrf357_all").build()

# compress_results.append(os_book_rect_1532_mae1)
# compress_results.append(os_book_rect_1532_mae3)
# compress_results.append(os_book_rect_1532_mae6)
# compress_results.append(os_book_justG_mae1)
# compress_results.append(os_book_justG_mae3)
# compress_results.append(os_book_justG_mae6)
# compress_results.append(os_book_justG_mae9)
# compress_results.append(os_book_justG_mae20)
# compress_results.append(os_book_rect_D10)
# compress_results.append(os_book_rect_D05)
# compress_results.append(os_book_rect_D00)

##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
from step10_a_load_and_train_and_test import *
"""
group寫法2：from step10_a_load_and_train_and_test import * 直接包 exps
補充：無法直接 from step10_a import * 直接處理，
    因為裡面包含太多其他物件了！光要抽出自己想用的 exp物件就是一大工程覺得~
    還有我也不知道要怎麼 直接用 ，也是要一個個 名字 打出來 才能用，名字 都打出來了，不如就直接 包成exps 囉！
"""

### copy的示範
# ch_064_300 = copy.deepcopy(epoch300_bn_see_arg_T.build()); ch_064_300.ana_describe = "flow_unet-ch64_300"
# ch_064_700 = copy.deepcopy(epoch700_bn_see_arg_T.build()); ch_064_700.ana_describe = "flow_unet-ch64_700"

### 0 old vs new shuffle
###    想比較是因為bn==1的情況下 old 和 new shuffle 理論上是一樣的，實際上也是差萬分之幾而已，所以先把old收起來囉，想刪記得只刪see就好
###################################################################################################
### 0_1 epoch old_shuffle
epoch_old_shuf_exps  = [
    epoch050_bn_see_arg_T.build(),
    epoch100_bn_see_arg_T.build(),
    epoch200_bn_see_arg_T.build(),
    epoch300_bn_see_arg_T.build(),
    ### jump epoch500 真的是失誤， 因為無法重現 old shuffle，所以真的無法
    epoch700_bn_see_arg_T.build()]


# ### 0_2 epoch new_shuffle
epoch_new_shuf_exps  = [
    epoch050_new_shuf_bn_see_arg_T.build(),
    epoch100_new_shuf_bn_see_arg_T.build(),
    epoch200_new_shuf_bn_see_arg_T.build(),
    epoch300_new_shuf_bn_see_arg_T.build(),
    epoch500_new_shuf_bn_see_arg_T.build(),
    epoch700_new_shuf_bn_see_arg_T.build()]

### 0_3 epoch old vs new shuffle
epoch_old_new_shuf_exps  = [
    epoch050_bn_see_arg_T.build(),
    epoch050_new_shuf_bn_see_arg_T.build(),
    epoch100_bn_see_arg_T.build(),
    epoch100_new_shuf_bn_see_arg_T.build(),
    epoch200_bn_see_arg_T.build(),
    epoch200_new_shuf_bn_see_arg_T.build(),
    epoch300_bn_see_arg_T.build(),
    epoch300_new_shuf_bn_see_arg_T.build(),
    ### jump epoch500
    ### jump epoch500_new_shuf_bn_see_arg_T ，因為上面jumpg保持統一性，儘管有也jump
    epoch700_bn_see_arg_T.build(),
    epoch700_new_shuf_bn_see_arg_T.build()]

### 0_4 ch old_shuffle
ch_old_shuf_exps = [
    old_ch128_bn_see_arg_T.build(),
    ### jump ch064 真的是失誤， 因為無法重現 old shuffle，所以真的無法
    old_ch032_bn_see_arg_T.build(),
    old_ch016_bn_see_arg_T.build(),
    old_ch008_bn_see_arg_T.build()]

### 0_5 ch new_shuffle
ch064_new_shuf_bn_see_arg_T = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch064_new_shuf_bn_see_arg_T.result_obj.ana_describe = "ch064_new_shuf_bn_see_arg_T"
ch_new_shuf_exps = [
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ch064_new_shuf_bn_see_arg_T,
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build()]

### 0_6 ch  old vs new shuffle
ch_old_new_shuf_exps = [
    old_ch128_bn_see_arg_T.build(),
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ### jump ch064
    ### jump ch064，因為上面jumpg保持統一性，儘管有也jump
    old_ch032_bn_see_arg_T.build(),
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build()]
###################################################################################################
###################################################################################################
###################################################################################################
### 1 epoch
epoch_exps  = [
    epoch050_new_shuf_bn_see_arg_T.build(),
    epoch100_new_shuf_bn_see_arg_T.build(),
    epoch200_new_shuf_bn_see_arg_T.build(),
    epoch300_new_shuf_bn_see_arg_T.build(),
    epoch500_new_shuf_bn_see_arg_T.build(),
    epoch500_new_shuf_bn_see_arg_F.build(),  ### 順便比一個 bn 設錯會怎麼樣
    epoch700_new_shuf_bn_see_arg_T.build(),
    epoch700_bn_see_arg_T_no_down.build(), ]   ### 如果 lr 都不下降 會怎麼樣

epoch020_500_exps  = [
    unet_IN_7l_2to3noC_e020.build(),  # 0
    unet_IN_7l_2to3noC_e040.build(),
    unet_IN_7l_2to3noC_e060.build(),
    unet_IN_7l_2to3noC_e080.build(),
    unet_IN_7l_2to3noC_e100.build(),
    unet_IN_7l_2to3noC_e120.build(),
    unet_IN_7l_2to3noC_e140.build(),  ## 6
    unet_IN_7l_2to3noC_e160.build(),
    unet_IN_7l_2to3noC_e180.build(),
    ch64_in_epoch200.build(),
    ch64_in_epoch220.build(),
    ch64_in_epoch240.build(),
    ch64_in_epoch260.build(),  ### 12
    ch64_in_epoch280.build(),
    ch64_in_epoch300.build(),
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),
    ch64_in_epoch380.build(),  ### 18
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),  ### 24
    ]

epoch100_500_exps  = [
    unet_IN_7l_2to3noC_e100.build(),  # 0
    unet_IN_7l_2to3noC_e120.build(),
    unet_IN_7l_2to3noC_e140.build(),
    unet_IN_7l_2to3noC_e160.build(),
    unet_IN_7l_2to3noC_e180.build(),
    ch64_in_epoch200.build(),
    ch64_in_epoch220.build(),  ###  6
    ch64_in_epoch240.build(),
    ch64_in_epoch260.build(),
    ch64_in_epoch280.build(),
    ch64_in_epoch300.build(),
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),  ### 13
    ch64_in_epoch380.build(),
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),  ### 20
    ]

epoch200_500_exps  = [
    ch64_in_epoch200.build(),
    ch64_in_epoch220.build(),
    ch64_in_epoch240.build(),
    ch64_in_epoch260.build(),
    ch64_in_epoch280.build(),
    ch64_in_epoch300.build(),
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),
    ch64_in_epoch380.build(),
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),
    ]

epoch300_500_exps  = [
    ch64_in_epoch320.build(),
    ch64_in_epoch340.build(),
    ch64_in_epoch360.build(),
    ch64_in_epoch380.build(),
    ch64_in_epoch400.build(),
    ch64_in_epoch420.build(),
    ch64_in_epoch440.build(),
    ch64_in_epoch460.build(),
    ch64_in_epoch480.build(),
    ch64_in_epoch500.build(),
    ]

# # ### 1_2 ch，但 bn_see_arg_F，所以結果圖醜，收起來，留 old_ch032_new_shuf_bn_see_arg_F 是為了給 bn 比較用
# old_ch128_new_shuf_bn_see_arg_F
# ch064_new_shuf_bn_see_arg_F = copy.deepcopy(epoch500_new_shuf_bn_see_arg_F.build()); ch064_new_shuf_bn_see_arg_F.result_obj.ana_describe = "ch064_new_shuf_bn_see_arg_F"
# old_ch032_new_shuf_bn_see_arg_F
# old_ch016_new_shuf_bn_see_arg_F
# old_ch008_new_shuf_bn_see_arg_F
###################################################################################################
###################################################################################################
### 2 ch
ch_exps = [
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ch064_new_shuf_bn_see_arg_T,
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build(),]

### 這 epoch_no_down_vs_ch_exps 真的很重要，
### 理解到了 training loss 在 0.000x 上閜跑是沒有什麼差的，
### 因為 ch008 跟 epoch700_no_down 的 training loss 最後是一樣的！
### 但是 在 real 影像上 是截然不同的喔！ epoch700_no_down 比 ch008 好很多！
### 所以不能只看 training loss，一定要直接看 real影像 處理的效果
epoch_no_down_vs_ch_exps = [
    old_ch128_new_shuf_bn_see_arg_T.build(),
    ch064_new_shuf_bn_see_arg_T,
    old_ch032_new_shuf_bn_see_arg_T.build(),
    old_ch016_new_shuf_bn_see_arg_T.build(),
    old_ch008_new_shuf_bn_see_arg_T.build(),
    epoch700_bn_see_arg_T_no_down.build(), ]
###################################################################################################
###################################################################################################
### 3 bn
###   3_1. ch64 只能 bn 1, 4, 8，覺得不夠明顯
ch64_bn01_bn_see_arg_T = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch64_bn01_bn_see_arg_T.result_obj.ana_describe = "ch64_bn01_bn_see_arg_T"
bn_ch64_exps_bn_see_arg_T = [
    ch64_bn01_bn_see_arg_T,
    ch64_bn04_bn_see_arg_T.build(),
    ch64_bn08_bn_see_arg_T.build(), ]

###   3_2. ch32 就能 bn 1, 4, 8, 16
ch32_bn01_bn_see_arg_T = copy.deepcopy(old_ch032_new_shuf_bn_see_arg_T.build()); ch32_bn01_bn_see_arg_T.result_obj.ana_describe = "ch32_bn01_bn_see_arg_T"
bn_ch32_exps_bn_see_arg_T = [
    ch32_bn01_bn_see_arg_T,
    old_ch32_bn04_bn_see_arg_T.build(),
    old_ch32_bn08_bn_see_arg_T.build(),
    old_ch32_bn16_bn_see_arg_T.build(), ]

###   3_3. ch32 bn1, 4, 8, 16 see_arg 設 True/False 來比較看看，ch64的 bn數比較少就跳過囉~~
ch64_bn01_bn_see_arg_F = copy.deepcopy(epoch500_new_shuf_bn_see_arg_F.build()); ch64_bn01_bn_see_arg_F.result_obj.ana_describe = "ch64_bn01_bn_see_arg_F"
bn_ch64_exps_bn_see_arg_F_and_T = [
    ch64_bn01_bn_see_arg_F,
    ch64_bn01_bn_see_arg_T,
    ch64_bn04_bn_see_arg_F.build(),
    ch64_bn04_bn_see_arg_T.build(),
    ch64_bn08_bn_see_arg_F.build(),
    ch64_bn08_bn_see_arg_T.build(), ]

###   3_4. ch32 bn1, 4, 8, 16 see_arg 設 True/False 來比較看看，ch64的 bn數比較少就跳過囉~~
ch32_bn01_bn_see_arg_F = copy.deepcopy(old_ch032_new_shuf_bn_see_arg_F.build()); ch32_bn01_bn_see_arg_F.result_obj.ana_describe = "ch32_bn01_bn_see_arg_F"
bn_ch32_exps_bn_see_arg_F_and_T = [
    ch32_bn01_bn_see_arg_F,
    ch32_bn01_bn_see_arg_T,
    old_ch32_bn04_bn_see_arg_F.build(),
    old_ch32_bn04_bn_see_arg_T.build(),
    old_ch32_bn08_bn_see_arg_F.build(),
    old_ch32_bn08_bn_see_arg_T.build(),
    old_ch32_bn16_bn_see_arg_F.build(),
    old_ch32_bn16_bn_see_arg_T.build(), ]

###################################################################################################
###################################################################################################
### 4 bn_in
###   4_1. in 的 batch_size一定只能等於1 所以拿 epoch500 來比較，也像看train 久一點的效果，所以就多train 一個 epoch700 的 並拿相應的 bn來比較
###   不管 epoch500, 700 都是 in 比 bn 好！
ch64_bn_epoch500 = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch64_bn_epoch500.result_obj.ana_describe = "ch64_bn_epoch500"
ch64_bn_epoch700 = copy.deepcopy(epoch700_new_shuf_bn_see_arg_T).build(); ch64_bn_epoch700.result_obj.ana_describe = "ch64_bn_epoch700"
bn_in_size1_exps = [
    ch64_bn_epoch500,
    ch64_in_epoch500.build(),
    ch64_bn_epoch700,
    ch64_in_epoch700.build(),
]


### 4_2. in vs bn batch_size > 1
###   batch_size 越大，效果越差
ch64_1_bn01 = copy.deepcopy(epoch500_new_shuf_bn_see_arg_T.build()); ch64_1_bn01.result_obj.ana_describe = "ch64_1_bn01"
ch64_2_in01 = copy.deepcopy(ch64_in_epoch500.build());               ch64_2_in01.result_obj.ana_describe = "ch64_2_in01"
ch64_3_bn04 = copy.deepcopy(ch64_bn04_bn_see_arg_T.build());         ch64_3_bn04.result_obj.ana_describe = "ch64_3_bn04"
ch64_4_bn08 = copy.deepcopy(ch64_bn08_bn_see_arg_T.build());         ch64_4_bn08.result_obj.ana_describe = "ch64_4_bn08"
bn_in_sizen_exps = [
    ch64_1_bn01,
    ch64_2_in01,
    ch64_3_bn04,
    ch64_4_bn08,
]
###################################################################################################
###################################################################################################
### 5 unet concat Activation vs concat BN，都是epoch500 且 lr有下降， 先不管 concatA loss 相當於表現差的哪種結果 只放兩個exp比較
###    concat_A 的效果比較好，且從架構圖上已經看出來 確實 concat_A 較合理
ch64_in_concat_B = copy.deepcopy(ch64_in_epoch500.build()); ch64_in_concat_B.result_obj.ana_describe = "ch64_in_concat_B"
in_concat_AB = [
    ch64_in_concat_A.build(),
    ch64_in_concat_B,
]

###################################################################################################
###################################################################################################
### 6 unet level 2~7
###   想看看 差一層 差多少，先不管 8_layer 表現好 的 相當於 哪種結果
###   最後發現 層越少效果越差，但 第8層 效果跟 第7層 差不多， 沒辦法試 第9層 因為要512的倍數
unet_layers = [
    unet_2l.build(),
    unet_3l.build(),
    unet_4l.build(),
    unet_5l.build(),
    unet_6l.build(),
    unet_7l.build(),
    unet_8l.build(),
]


###################################################################################################
### 7a_1 unet 的concat 改成 add 的效果如何，效果超差
unet_skip_use_add = [
    unet_8l_skip_use_add.build(),
    unet_7l_skip_use_add.build(),
    unet_6l_skip_use_add.build(),
    unet_5l_skip_use_add.build(),
    unet_4l_skip_use_add.build(),
    unet_3l_skip_use_add.build(),
    unet_2l_skip_use_add.build(),
]

### 7a_2 unet 的concat vs add 的效果如何，concat好，add不好
unet_skip_use_concat_vs_add = [
    unet_7l.build(),
    unet_7l_skip_use_add.build(),
    unet_6l.build(),
    unet_6l_skip_use_add.build(),
    unet_5l.build(),
    unet_5l_skip_use_add.build(),
]


### 7b unet 的 第一層不concat 來跟 前面還不錯的結果比較
###    train loss 在 全接~前兩個skip省略 表現差不多，
###    see來看的話 train/test 都差不多，但在 real 前兩個skip 的結果 看起來都比 全接好！ 且 覺得 2to3noC 比 2to2noC 更好些！
###    2to4noC 在 real3 表現差，且 2to4noC 之後 邊緣 的部分就越做越差囉～
unet_IN_7l_all_C_ch64_in_epoch500 = copy.deepcopy(ch64_in_epoch500.build()); unet_IN_7l_all_C_ch64_in_epoch500.result_obj.ana_describe = "1a-unet_IN_7l_all_C_ch64_in_epoch500"  ### 當初的train_code沒寫好沒有存到 model用的 code
# unet_IN_7l_all_C_ch64_bn_epoch500 = copy.deepcopy(ch64_bn_epoch500.build()); unet_IN_7l_all_C_ch64_bn_epoch500.result_obj.ana_describe = "1b-unet_IN_7l_all_C_ch64_bn_epoch500"  ### 從4_1就知道bn沒有in好，所以就不用這個了
# unet_IN_7l_all_C_unet_7l          = copy.deepcopy(unet_7l.build()); unet_IN_7l_all_C_unet_7l.result_obj.ana_describe = "1c-unet_IN_7l_all_C_unet_7l"  ### 他的loss好像最低，但沒有train完
unet_IN_7l_2to2noC       .build().result_obj.ana_describe = "2a-unet_IN_7l_2to2noC"
# unet_IN_7l_2to2noC_ch32.build().result_obj.ana_describe = "2b-unet_IN_7l_2to2noC_ch32"  ### ch32 效果比較沒那麼好，先註解跳過不看
unet_IN_7l_2to3noC       .build().result_obj.ana_describe = "3-unet_IN_7l_2to3noC"
unet_IN_7l_2to4noC       .build().result_obj.ana_describe = "4-unet_IN_7l_2to4noC"
unet_IN_7l_2to5noC       .build().result_obj.ana_describe = "5-unet_IN_7l_2to5noC"
unet_IN_7l_2to6noC       .build().result_obj.ana_describe = "6-unet_IN_7l_2to6noC"
unet_IN_7l_2to7noC       .build().result_obj.ana_describe = "7-unet_IN_7l_2to7noC"
# unet_IN_7l_2to8noC               .result_obj.ana_describe = "8-unet_IN_7l_2to8noC"   ### 當初訓練就怪怪的，先跳過！

unet_skip_noC = [
    unet_IN_7l_all_C_ch64_in_epoch500,
    # unet_IN_7l_all_C_ch64_bn_epoch500,
    # unet_IN_7l_all_C_unet_7l,
    unet_IN_7l_2to2noC,
    # unet_IN_7l_2to2noC_ch32,
    unet_IN_7l_2to3noC,
    unet_IN_7l_2to4noC,
    unet_IN_7l_2to5noC,
    unet_IN_7l_2to6noC,
    unet_IN_7l_2to7noC,
    # unet_IN_7l_2to8noC,   ### 好像train壞掉怪怪的
]


### 7c unet 的 第一層不concat 來跟 前面還不錯的結果比較
unet_IN_7l_skip_clean = copy.deepcopy(ch64_in_epoch500.build()); unet_IN_7l_skip_clean.result_obj.ana_describe = "1-unet_IN_7l_skip_clean"  ### 當初的train_code沒寫好沒有存到 model用的 code

unet_IN_7l_skip_use_cnn1_NO_relu     .build().result_obj.ana_describe = "2-unet_IN_7l_skip_use_cnn1_NO_relu"
unet_IN_7l_skip_use_cnn1_USErelu     .build().result_obj.ana_describe = "3a-unet_IN_7l_skip_use_cnn1_USErelu"
unet_IN_7l_skip_use_cnn1_USEsigmoid  .build().result_obj.ana_describe = "3b-unet_IN_7l_skip_use_cnn1_USEsigmoid"
unet_IN_7l_skip_use_cnn3_USErelu     .build().result_obj.ana_describe = "4a-unet_IN_7l_skip_use_cnn3_USErelu"
unet_IN_7l_skip_use_cnn3_USEsigmoid  .build().result_obj.ana_describe = "4b-unet_IN_7l_skip_use_cnn3_USEsigmoid"

unet_skip_use_cnn = [
    unet_IN_7l_skip_clean,
    unet_IN_7l_skip_use_cnn1_NO_relu,
    unet_IN_7l_skip_use_cnn1_USErelu,
    unet_IN_7l_skip_use_cnn1_USEsigmoid,
    unet_IN_7l_skip_use_cnn3_USErelu,
    unet_IN_7l_skip_use_cnn3_USEsigmoid,
]


### 7d.看看 UNet 的 skip 用 cSE/ sSE/ csSE 試試看
unet_7l_skip_SE = [
    ch64_in_epoch060          .build(),
    ch64_in_sk_cSE_e060_wrong .build(),
    ch64_in_sk_sSE_e060       .build(),
    ch64_in_sk_scSE_e060_wrong.build(),
]

ch64_2to3noC_sk_no_e060         = copy.deepcopy(unet_IN_7l_2to3noC_e060.build()); ch64_2to3noC_sk_no_e060         .result_obj.ana_describe = "1-ch64_2to3noC_sk_no_e060"  ### 當初的train_code沒寫好沒有存到 model用的 code
ch64_2to3noC_sk_cSE_e060_wrong  = ch64_2to3noC_sk_cSE_e060_wrong .build();        ch64_2to3noC_sk_cSE_e060_wrong  .result_obj.ana_describe = "2-ch64_2to3noC_sk_cSE_e060_wrong"
ch64_2to3noC_sk_sSE_e060        = ch64_2to3noC_sk_sSE_e060       .build();        ch64_2to3noC_sk_sSE_e060        .result_obj.ana_describe = "3-ch64_2to3noC_sk_sSE_e060"
ch64_2to3noC_sk_scSE_e060_wrong = ch64_2to3noC_sk_scSE_e060_wrong.build();        ch64_2to3noC_sk_scSE_e060_wrong .result_obj.ana_describe = "4-ch64_2to3noC_sk_scSE_e060_wrong"
unet_7l_2to3noC_skip_SE = [
    ch64_2to3noC_sk_no_e060,
    ch64_2to3noC_sk_cSE_e060_wrong,
    ch64_2to3noC_sk_sSE_e060,
    ch64_2to3noC_sk_scSE_e060_wrong,
]

###################################################################################################
###################################################################################################
### 8_1 unet 的 range(mae)
unet_range_mae = [
    t1_in_01_mo_th_gt_01_mae.build(),
    t2_in_01_mo_01_gt_01_mae.build(),
    t3_in_01_mo_th_gt_th_mae.build(),
    t4_in_01_mo_01_gt_th_mae.build(),
    t5_in_th_mo_th_gt_01_mae.build(),
    t6_in_th_mo_01_gt_01_mae.build(),
    t7_in_th_mo_th_gt_th_mae.build(),
    t8_in_th_mo_01_gt_th_mae.build(),
]

unet_range_mae_good = [
    t1_in_01_mo_th_gt_01_mae.build(),
    t2_in_01_mo_01_gt_01_mae.build(),
    t5_in_th_mo_th_gt_01_mae.build(),
    t6_in_th_mo_01_gt_01_mae.build(),
]

unet_range_mae_ok = [
    t3_in_01_mo_th_gt_th_mae.build(),
    t7_in_th_mo_th_gt_th_mae.build(),
]

unet_range_mse = [
    t1_in_01_mo_th_gt_01_mse.build(),
    t2_in_01_mo_01_gt_01_mse.build(),
    t3_in_01_mo_th_gt_th_mse.build(),
    t4_in_01_mo_01_gt_th_mse.build(),
    t5_in_th_mo_th_gt_01_mse.build(),
    t6_in_th_mo_01_gt_01_mse.build(),
    t7_in_th_mo_th_gt_th_mse.build(),
    t8_in_th_mo_01_gt_th_mse.build(),
]

########################################################################################################################
###########################################################################
### 另一個架構
rect_layers_wrong_lrelu = [
    rect_2_level_fk3.build(),
    rect_3_level_fk3.build(),
    rect_4_level_fk3.build(),
    rect_5_level_fk3.build(),
    rect_6_level_fk3.build(),
    rect_7_level_fk3.build(),
]

rect_layers_right_relu = [
    rect_2_level_fk3_ReLU.build(),
    rect_3_level_fk3_ReLU.build(),
    rect_4_level_fk3_ReLU.build(),
    rect_5_level_fk3_ReLU.build(),
    rect_6_level_fk3_ReLU.build(),
    rect_7_level_fk3_ReLU.build(),
]

rect_fk3_ch64_tfIN_resb_ok9_epoch500
rect_7_level_fk7

########################################################################################################################

test1  ### 少資料測試 method 有沒有寫對
