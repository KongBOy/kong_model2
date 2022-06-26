#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
code_dir = "\\".join(code_exe_path_element[:-1])
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
sys.path.append(code_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    code_dir:", code_dir)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
#############################################################################################################################################################################################################
kong_to_py_layer = len(code_exe_path_element) - 1 - kong_layer  ### 中間 -1 是為了長度轉index
# print("    kong_to_py_layer:", kong_to_py_layer)
if  (kong_to_py_layer == 0): template_dir = ""
elif(kong_to_py_layer == 2): template_dir = code_exe_path_element[kong_layer + 1][0:]  ### [7:] 是為了去掉 step1x_， 後來覺得好像改有意義的名字不去掉也行所以 改 0
elif(kong_to_py_layer == 3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:]  ### [5:] 是為了去掉 mask_ ，前面的 mask_ 是為了python 的 module 不能 數字開頭， 隨便加的這樣子， 後來覺得 自動排的順序也可以接受， 所以 改0
elif(kong_to_py_layer >  3): template_dir = code_exe_path_element[kong_layer + 1][0:] + "/" + code_exe_path_element[kong_layer + 2][0:] + "/" + "/".join(code_exe_path_element[kong_layer + 3: -1])
# print("    template_dir:", template_dir)  ### 舉例： template_dir: 7_mask_unet/5_os_book_and_paper_have_dtd_hdr_mix_bg_tv_s04_mae
#############################################################################################################################################################################################################
exp_dir = template_dir
#############################################################################################################################################################################################################
### Basic builder
from step06_a_datas_obj import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder

### Model_builder
from step09_3side_L5 import *

# rm_paths = [path for path in sys.path if code_dir in path]
# for rm_path in rm_paths: sys.path.remove(rm_path)
# rm_moduless = [module for module in sys.modules if "step09" in module]
# for rm_module in rm_moduless: del sys.modules[rm_module]


### Exp_builder
'''
p20_L5_woDiv
p20_L5_wiDiv
p20_L5_3UNet

Mae_s001_Sob_k05_s001_EroM
Mae_s001_Sob_k15_s001_EroM
Mae_s001_Sob_k25_s001_EroM
Mae_s001_Sob_k35_s001_EroM
'''
##### I_w_M_to_W
##### k05, 15, 25, 35
### woDiv
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok    .ch096.wiColorJ.Add2Loss.Sob_k05_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k05_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok    .ch096.wiColorJ.Add2Loss.Sob_k15_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k15_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok    .ch096.wiColorJ.Add2Loss.Sob_k25_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k25_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok    .ch096.wiColorJ.Add2Loss.Sob_k35_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k35_s001_EroM
### wiDiv
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok_div.ch032.wiColorJ.Add2Loss.Sob_k05_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k05_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok_div.ch032.wiColorJ.Add2Loss.Sob_k15_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k15_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok_div.ch032.wiColorJ.Add2Loss.Sob_k25_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k25_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_focus_Zok_div.ch032.wiColorJ.Add2Loss.Sob_k35_s001_EroM_Mae_s001.pyr_Tcrop255_p20_j15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k35_s001_EroM
### 3UNet
from Exps_7_v3.doc3d.I_w_M_to_Wxyz_focus_Z_ok.pyr_Tcrop255_pad20_jit15.Sob_k05_s001_EroM_Mae_s001.pyr_2s.L5.step10_a                     import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k05_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_Wxyz_focus_Z_ok.pyr_Tcrop255_pad20_jit15.Sob_k15_s001_EroM_Mae_s001.pyr_2s.L5.step10_a                     import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k15_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_Wxyz_focus_Z_ok.pyr_Tcrop255_pad20_jit15.Sob_k25_s001_EroM_Mae_s001.pyr_2s.L5.step10_a                     import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k25_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_Wxyz_focus_Z_ok.pyr_Tcrop255_pad20_jit15.Sob_k35_s001_EroM_Mae_s001.pyr_2s.L5.step10_a                     import ch032_1side_6__2side_6 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k35_s001_EroM

##### k03, 05, 09, 11
### 3UNet
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__3UNet__Mae_s001_Sob_k03_s001 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k03_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__3UNet__Mae_s001_Sob_k09_s001 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k09_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__3UNet__Mae_s001_Sob_k11_s001 as I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k11_s001_EroM
### woDiv
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__woDiv__Mae_s001_Sob_k03_s001 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k03_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__woDiv__Mae_s001_Sob_k09_s001 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k09_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__woDiv__Mae_s001_Sob_k11_s001 as I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k11_s001_EroM
### wiDiv
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__wiDiv__Mae_s001_Sob_k03_s001 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k03_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__wiDiv__Mae_s001_Sob_k09_s001 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k09_s001_EroM
from Exps_7_v3.doc3d.I_w_M_to_W_Comb.step10_a import L5_ch032_2blk__wiDiv__Mae_s001_Sob_k11_s001 as I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k11_s001_EroM


##### W_w_M_to_C
### 2UNet
'''
先只用
Mae_s001_Sob_k05_s001_EroM
'''
from Exps_7_v3.doc3d.W_w_Mgt_to_Cx_Cy_focus_Z_ok.Sob_k05_s001_EroM_Mae_s001.pyr_Tcrop255_pad20_jit15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae
# from Exps_7_v3.doc3d.W_w_Mgt_to_Cx_Cy_focus_Z_ok.Sob_k15_s001_EroM_Mae_s001.pyr_Tcrop255_pad20_jit15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k15_Mae
# from Exps_7_v3.doc3d.W_w_Mgt_to_Cx_Cy_focus_Z_ok.Sob_k25_s001_EroM_Mae_s001.pyr_Tcrop255_pad20_jit15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k25_Mae
# from Exps_7_v3.doc3d.W_w_Mgt_to_Cx_Cy_focus_Z_ok.Sob_k35_s001_EroM_Mae_s001.pyr_Tcrop255_pad20_jit15.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k35_Mae

#############################################################################################################################################################################################################

###############################################################################################################################
use_db_obj = type8_blender_kong_doc3d_v2
use_loss_obj = [G_mae_s001_loss_info_builder.set_loss_target("UNet_z").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_y").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_x").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cx").copy(), G_mae_s001_loss_info_builder.set_loss_target("UNet_Cy").copy()]  ### z, y, x 順序是看 step07_b_0b_Multi_UNet 來對應的喔

#############################################################
### 為了resul_analyze畫空白的圖，建一個empty的 Exp_builder
empty = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_6_p20_L5_woDiv.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="為了resul_analyze畫空白的圖，建一個empty的 Exp_builder")
#############################################################
DewarpNet = Exp_builder().set_basic("train", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=ch032_pyramid_1side_6__2side_6_p20_L5_woDiv.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="DewarpNet")
#############################################################
### p20
'''
woDiv
    Sob_k05_Mae
    Sob_k15_Mae
    Sob_k25_Mae
    Sob_k35_Mae
wiDiv
    Sob_k05_Mae
    Sob_k15_Mae
    Sob_k25_Mae
    Sob_k35_Mae
3UNet
    Sob_k05_Mae
    Sob_k15_Mae
    Sob_k25_Mae
    Sob_k35_Mae
配 Sob_k05_Mae
'''
### woDIv train錯
p20_L5_woDiv_Sob_k03_Mae__Sob_k05_Mae_ch032 = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv_ch032, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k03_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k03_Mae__Sob_k05_Mae_ch032")
p20_L5_woDiv_Sob_k09_Mae__Sob_k05_Mae_ch032 = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv_ch032, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k09_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k09_Mae__Sob_k05_Mae_ch032")
p20_L5_woDiv_Sob_k11_Mae__Sob_k05_Mae_ch032 = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv_ch032, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k11_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k11_Mae__Sob_k05_Mae_ch032")

### woDIv
p20_L5_woDiv_Sob_k05_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k05_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k05_Mae__Sob_k05_Mae_ch032")
p20_L5_woDiv_Sob_k15_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k15_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k15_Mae__Sob_k05_Mae")
p20_L5_woDiv_Sob_k25_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k25_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k25_Mae__Sob_k05_Mae")
p20_L5_woDiv_Sob_k35_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_woDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_woDiv_Mae_s001_Sob_k35_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-woDiv_Sob_k35_Mae__Sob_k05_Mae")
### wiDIv
p20_L5_wiDiv_Sob_k03_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k03_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k03_Mae__Sob_k05_Mae")
p20_L5_wiDiv_Sob_k05_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k05_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k05_Mae__Sob_k05_Mae")
p20_L5_wiDiv_Sob_k09_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k09_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k09_Mae__Sob_k05_Mae")
p20_L5_wiDiv_Sob_k11_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k11_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k11_Mae__Sob_k05_Mae")
p20_L5_wiDiv_Sob_k15_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k15_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k15_Mae__Sob_k05_Mae")
p20_L5_wiDiv_Sob_k25_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k25_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k25_Mae__Sob_k05_Mae")
p20_L5_wiDiv_Sob_k35_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_wiDiv, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_wiDiv_Mae_s001_Sob_k35_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-wiDiv_Sob_k35_Mae__Sob_k05_Mae")
### 3UNet
p20_L5_3UNet_Sob_k03_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k03_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k03_Mae__Sob_k05_Mae")
p20_L5_3UNet_Sob_k05_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k05_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k05_Mae__Sob_k05_Mae")
p20_L5_3UNet_Sob_k09_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k09_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k09_Mae__Sob_k05_Mae")
p20_L5_3UNet_Sob_k11_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k11_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k11_Mae__Sob_k05_Mae")
p20_L5_3UNet_Sob_k15_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k15_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k15_Mae__Sob_k05_Mae")
p20_L5_3UNet_Sob_k25_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k25_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k25_Mae__Sob_k05_Mae")
p20_L5_3UNet_Sob_k35_Mae__Sob_k05_Mae = Exp_builder().set_basic("test_Kong_Crop_p60_gt_DewarpNet_p60_then_Use_KModel5_FBA", use_db_obj, ch032_pyramid_1side_6__2side_6_p20_L5_3UNet, use_loss_obj, exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end="not decide yet") .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_multi_model_reload_exp_builders_dict(I_to_Wx_Wy_Wz=I_w_M_to_W_p20_2s_L5_3UNet_Mae_s001_Sob_k35_s001_EroM    , W_to_Cx_Cy=W_w_M_to_C_Tcrop255_p20_2s_L5_Sob_k05_Mae).set_result_name(result_name="p20_L5-3UNet_Sob_k35_Mae__Sob_k05_Mae")
#############################################################
if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        p20_L5_woDiv_Sob_k05_Mae__Sob_k05_Mae.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
