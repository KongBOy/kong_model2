#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
### 把 kong_model2 加入 sys.path
import os
code_exe_path = os.path.realpath(__file__)                   ### 目前執行 step10_b.py 的 path
code_exe_path_element = code_exe_path.split("\\")            ### 把 path 切分 等等 要找出 kong_model 在第幾層
kong_layer = code_exe_path_element.index("kong_model2")      ### 找出 kong_model2 在第幾層
kong_model2_dir = "\\".join(code_exe_path_element[:kong_layer + 1])  ### 定位出 kong_model2 的 dir
import sys                                                   ### 把 kong_model2 加入 sys.path
sys.path.append(kong_model2_dir)
# print(__file__.split("\\")[-1])
# print("    code_exe_path:", code_exe_path)
# print("    code_exe_path_element:", code_exe_path_element)
# print("    kong_layer:", kong_layer)
# print("    kong_model2_dir:", kong_model2_dir)
###############################################################################################################################################################################################################
# 按F5執行時， 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～ 才可 import step10_a.py 喔！
code_exe_dir = os.path.dirname(code_exe_path)   ### 目前執行 step10_b.py 的 dir
if(os.getcwd() != code_exe_dir):                ### 如果 不是在 step10_b.py 的資料夾， 自動幫你切過去～
    os.chdir(code_exe_dir)
# print("current_path:", os.getcwd())
###############################################################################################################################################################################################################
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Bce_s001             .pyr_2s.L5.step10_a import empty                  as empty

### L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Bce_s001             .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Mae_s001             .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Mae_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k05_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k05_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k07_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k07_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k09_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k09_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k11_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k11_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k13_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k13_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k15_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k15_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k17_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k17_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k19_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k19_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k21_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k21_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k23_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k23_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k25_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k25_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k27_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k27_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k29_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k29_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k31_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k31_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k33_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k33_L5
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k35_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k35_L5

from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k05_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k05_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k07_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k07_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k09_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k09_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k11_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k11_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k13_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k13_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k15_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k15_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k17_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k17_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k19_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k19_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k21_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k21_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k23_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k23_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k25_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k25_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k27_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k27_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k29_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k29_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k31_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k31_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k33_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k33_Bce_L5
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k35_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k35_Bce_L5

### L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Bce_s001             .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k05_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k05_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k07_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k07_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k09_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k09_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k11_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k11_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k13_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k13_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k15_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k15_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k17_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k17_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k19_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k19_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k21_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k21_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k23_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k23_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k25_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k25_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k27_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k27_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k29_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k29_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k31_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k31_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k33_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k33_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Sob_k35_s001         .pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k35_L6

# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k05_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k05_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k07_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k07_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k09_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k09_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k11_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k11_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k13_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k13_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k15_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k15_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k17_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k17_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k19_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k19_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k21_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k21_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k23_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k23_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k25_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k25_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k27_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k27_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k29_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k29_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k31_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k31_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k33_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k33_Bce_L6
# from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.wiColorJ.pyr_Tcrop255_p60_j15.Add2Loss.Sob_k35_s001_Bce_s001.pyr_2s.L6.step10_a import ch032_1side_7__2side_7 as Tcrop255_p20_Sob_k35_Bce_L6

### L5
# Tcrop255_P20_Sob_L5 = [
    # [Tcrop255_p20_Bce_L5    , Tcrop255_p20_Mae_L5    , empty                  , empty                  , empty               , ] ,
    # [Tcrop255_p20_Sob_k05_L5, Tcrop255_p20_Sob_k07_L5, Tcrop255_p20_Sob_k09_L5, Tcrop255_p20_Sob_k11_L5, Tcrop255_p20_Sob_k13_L5, ] ,
    # [Tcrop255_p20_Sob_k15_L5, Tcrop255_p20_Sob_k17_L5, Tcrop255_p20_Sob_k19_L5, Tcrop255_p20_Sob_k21_L5, Tcrop255_p20_Sob_k23_L5, ] ,
    # [Tcrop255_p20_Sob_k25_L5, Tcrop255_p20_Sob_k27_L5, Tcrop255_p20_Sob_k29_L5, Tcrop255_p20_Sob_k31_L5, Tcrop255_p20_Sob_k33_L5, ] ,
# ]

Tcrop255_P20_Sob_add_Bce_L5 = [
    [Tcrop255_p20_Bce_L5        , Tcrop255_p20_Mae_L5        , empty                      , empty                      , empty                   , ] ,
    [Tcrop255_p20_Sob_k05_Bce_L5, Tcrop255_p20_Sob_k07_Bce_L5, Tcrop255_p20_Sob_k09_Bce_L5, Tcrop255_p20_Sob_k11_Bce_L5, Tcrop255_p20_Sob_k13_Bce_L5, ] ,
    [Tcrop255_p20_Sob_k15_Bce_L5, Tcrop255_p20_Sob_k17_Bce_L5, Tcrop255_p20_Sob_k19_Bce_L5, Tcrop255_p20_Sob_k21_Bce_L5, Tcrop255_p20_Sob_k23_Bce_L5, ] ,
    [Tcrop255_p20_Sob_k25_Bce_L5, Tcrop255_p20_Sob_k27_Bce_L5, Tcrop255_p20_Sob_k29_Bce_L5, Tcrop255_p20_Sob_k31_Bce_L5, Tcrop255_p20_Sob_k33_Bce_L5, ] ,
]

# ### L6
# Tcrop255_P20_Sob_L6 = [
#     [Tcrop255_p20_Bce_L6    , empty                  , empty                  , empty                  , empty                  , ] ,
#     [Tcrop255_p20_Sob_k05_L6, Tcrop255_p20_Sob_k07_L6, Tcrop255_p20_Sob_k09_L6, Tcrop255_p20_Sob_k11_L6, Tcrop255_p20_Sob_k13_L6, ] ,
#     [Tcrop255_p20_Sob_k15_L6, Tcrop255_p20_Sob_k17_L6, Tcrop255_p20_Sob_k19_L6, Tcrop255_p20_Sob_k21_L6, Tcrop255_p20_Sob_k23_L6, ] ,
#     [Tcrop255_p20_Sob_k25_L6, Tcrop255_p20_Sob_k27_L6, Tcrop255_p20_Sob_k29_L6, Tcrop255_p20_Sob_k31_L6, Tcrop255_p20_Sob_k33_L6, ] ,
# ]

# Tcrop255_P20_Sob_add_Bce_L6 = [
#     [Tcrop255_p20_Bce_L6        , empty                      , empty                      , empty                      , empty                      , ] ,
#     [Tcrop255_p20_Sob_k05_Bce_L6, Tcrop255_p20_Sob_k07_Bce_L6, Tcrop255_p20_Sob_k09_Bce_L6, Tcrop255_p20_Sob_k11_Bce_L6, Tcrop255_p20_Sob_k13_Bce_L6, ] ,
#     [Tcrop255_p20_Sob_k15_Bce_L6, Tcrop255_p20_Sob_k17_Bce_L6, Tcrop255_p20_Sob_k19_Bce_L6, Tcrop255_p20_Sob_k21_Bce_L6, Tcrop255_p20_Sob_k23_Bce_L6, ] ,
#     [Tcrop255_p20_Sob_k25_Bce_L6, Tcrop255_p20_Sob_k27_Bce_L6, Tcrop255_p20_Sob_k29_Bce_L6, Tcrop255_p20_Sob_k31_Bce_L6, Tcrop255_p20_Sob_k33_Bce_L6, ] ,
# ]
