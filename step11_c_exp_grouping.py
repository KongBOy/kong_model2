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