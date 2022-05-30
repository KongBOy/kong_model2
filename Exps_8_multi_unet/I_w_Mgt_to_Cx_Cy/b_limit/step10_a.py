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

from step06_a_datas_obj import *
from step09_g1_multi_unet2_obj_I_w_Mgt_to_Cx_Cy import *
from step10_a2_loss_info_obj import *
from step10_b2_exp_builder import Exp_builder

rm_paths = [path for path in sys.path if code_dir in path]
for rm_path in rm_paths: sys.path.remove(rm_path)
rm_moduless = [module for module in sys.modules if "step09" in module]
for rm_module in rm_moduless: del sys.modules[rm_module]

#############################################################################################################################################################################################################
'''
exp_dir 是 決定 result_dir 的 "上一層"資料夾 名字喔！ exp_dir要巢狀也沒問題～
比如：exp_dir = "6_mask_unet/自己命的名字"，那 result_dir 就都在：
    6_mask_unet/自己命的名字/result_a
    6_mask_unet/自己命的名字/result_b
    6_mask_unet/自己命的名字/...
'''

use_db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper
#############################################################
I_to_Cx_L4_ch128_lim_and_I_to_Cy_L4_ch128_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L4_ch128_lim_and_I_to_Cy_L4_ch128_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L4_ch128_lim_and_I_to_Cy_L4_ch128_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L4_ch128_block1_&&_I_to_Cy_L4_ch128_block1-20211228_150315")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)

I_to_Cx_L5_ch128_lim_and_I_to_Cy_L5_ch128_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L5_ch128_lim_and_I_to_Cy_L5_ch128_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L5_ch128_lim_and_I_to_Cy_L5_ch128_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
I_to_Cx_L5_ch064_lim_and_I_to_Cy_L5_ch064_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L5_ch064_lim_and_I_to_Cy_L5_ch064_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L5_ch064_lim_and_I_to_Cy_L5_ch064_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

I_to_Cx_L6_ch128_lim_and_I_to_Cy_L6_ch128_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L6_ch128_lim_and_I_to_Cy_L6_ch128_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L6_ch128_lim_and_I_to_Cy_L6_ch128_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L6_ch128_block1_&&_I_to_Cy_L6_ch128_block1-20211228_071656")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
I_to_Cx_L6_ch064_lim_and_I_to_Cy_L6_ch064_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L6_ch064_lim_and_I_to_Cy_L6_ch064_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L6_ch064_lim_and_I_to_Cy_L6_ch064_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
I_to_Cx_L6_ch032_lim_and_I_to_Cy_L6_ch032_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L6_ch032_lim_and_I_to_Cy_L6_ch032_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L6_ch032_lim_and_I_to_Cy_L6_ch032_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

I_to_Cx_L7_ch128_lim_and_I_to_Cy_L7_ch128_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L7_ch128_lim_and_I_to_Cy_L7_ch128_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L7_ch128_lim_and_I_to_Cy_L7_ch128_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L7_ch128_block1_&&_I_to_Cy_L7_ch128_block1-20211228_071254")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
I_to_Cx_L7_ch064_lim_and_I_to_Cy_L7_ch064_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L7_ch064_lim_and_I_to_Cy_L7_ch064_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L7_ch064_lim_and_I_to_Cy_L7_ch064_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L7_ch064_block1_&&_I_to_Cy_L7_ch064_block1-20211228_071340")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
I_to_Cx_L7_ch032_lim_and_I_to_Cy_L7_ch032_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L7_ch032_lim_and_I_to_Cy_L7_ch032_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L7_ch032_lim_and_I_to_Cy_L7_ch032_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
I_to_Cx_L7_ch016_lim_and_I_to_Cy_L7_ch016_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L7_ch016_lim_and_I_to_Cy_L7_ch016_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L7_ch016_lim_and_I_to_Cy_L7_ch016_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

I_to_Cx_L8_ch128_lim_and_I_to_Cy_L8_ch128_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L8_ch128_lim_and_I_to_Cy_L8_ch128_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L8_ch128_lim_and_I_to_Cy_L8_ch128_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L8_ch128_block1_&&_I_to_Cy_L8_ch128_block1-20211228_150346")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
I_to_Cx_L8_ch064_lim_and_I_to_Cy_L8_ch064_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L8_ch064_lim_and_I_to_Cy_L8_ch064_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L8_ch064_lim_and_I_to_Cy_L8_ch064_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L8_ch064_block1_&&_I_to_Cy_L8_ch064_block1-20211228_071805")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
I_to_Cx_L8_ch032_lim_and_I_to_Cy_L8_ch032_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L8_ch032_lim_and_I_to_Cy_L8_ch032_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L8_ch032_lim_and_I_to_Cy_L8_ch032_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="I_to_Cx_L8_ch032_block1_&&_I_to_Cy_L8_ch032_block1-20211228_071750")  #.change_result_name_v4_Remove_sig_out(run_change=True, print_msg=True)  #.change_result_name_v3_to_v4_Remove_db_name(run_change=True, print_msg=True)
I_to_Cx_L8_ch016_lim_and_I_to_Cy_L8_ch016_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L8_ch016_lim_and_I_to_Cy_L8_ch016_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L8_ch016_lim_and_I_to_Cy_L8_ch016_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")
I_to_Cx_L8_ch008_lim_and_I_to_Cy_L8_ch008_lim_ep060 = Exp_builder().set_basic("test_real_photo_paper2", use_db_obj, I_to_Cx_L8_ch008_lim_and_I_to_Cy_L8_ch008_lim, [G_mae_s001_loss_info_builder.set_loss_target("UNet1"), G_mae_s001_loss_info_builder.set_loss_target("UNet2")], exp_dir=exp_dir, code_exe_path=code_exe_path, describe_end=I_to_Cx_L8_ch008_lim_and_I_to_Cy_L8_ch008_lim.kong_model.model_describe) .set_train_args(epochs= 60).set_train_in_gt_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_result_name(result_name="")

if(__name__ == "__main__"):
    print("build exps cost time:", time.time() - start_time)
    if len(sys.argv) < 2:
        ############################################################################################################
        ### 直接按 F5 或打 python step10_b1_exp_obj_load_and_train_and_test.py，後面沒有接東西喔！才不會跑到下面給 step10_b_subprocss.py 用的程式碼~~~
        I_to_Cx_L4_ch128_lim_and_I_to_Cy_L4_ch128_lim_ep060.build().run()
        # print('no argument')
        sys.exit()

    ### 以下是給 step10_b_subprocess.py 用的，相當於cmd打 python step10_b1_exp_obj_load_and_train_and_test.py 某個exp.build().run()
    eval(sys.argv[1])
