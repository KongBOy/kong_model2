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

from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Bce_s001             .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Bce_s001             .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k05_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k05
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k15_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k15
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k25_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k25
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k35_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k35
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k05_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k05_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k15_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k15_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k25_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k25_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad20_jit15.Sob_k35_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p20_Sob_k35_Bce

from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Bce_s001             .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k05_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k05
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k15_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k15
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k25_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k25
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k35_s001         .pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k35
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k05_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k05_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k15_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k15_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k25_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k25_Bce
from Exps_7_v3.doc3d.I_to_M_Gk3_no_pad.pyr_Tcrop255_pad60_jit15.Sob_k35_s001_Bce_s001.pyr_2s.L5.step10_a import ch032_1side_6__2side_6 as Tcrop255_p60_Sob_k35_Bce

Tcrop255_Pad_Loss_analyze = [
    [Tcrop255_p20_Bce        , empty                   , empty                   , empty                   , ] + [Tcrop255_p60_Bce        , empty                   , empty                   , empty                   , ],
    [Tcrop255_p20_Sob_k05    , Tcrop255_p20_Sob_k15    , Tcrop255_p20_Sob_k25    , Tcrop255_p20_Sob_k35    , ] + [Tcrop255_p60_Sob_k05    , Tcrop255_p60_Sob_k15    , Tcrop255_p60_Sob_k25    , Tcrop255_p60_Sob_k35    , ],
    [Tcrop255_p20_Sob_k05_Bce, Tcrop255_p20_Sob_k15_Bce, Tcrop255_p20_Sob_k25_Bce, Tcrop255_p20_Sob_k35_Bce, ] + [Tcrop255_p60_Sob_k05_Bce, Tcrop255_p60_Sob_k15_Bce, Tcrop255_p60_Sob_k25_Bce, Tcrop255_p60_Sob_k35_Bce, ],
]
