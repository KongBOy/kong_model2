import os
import sys
sys.path.append("SIFTflow")
from kong_use_evalUnwarp_sucess import use_DewarpNet_eval

ord_dir = os.getcwd()  ### step1 紀錄 目前的主程式資料夾
os.chdir("SIFTflow")   ### step2 跳到 SIFTflow資料夾裡面
print(os.getcwd())

use_DewarpNet_eval(path1="", path2="")  ### step3 執行 SIFTflow資料夾裡面 的 kong_use_evalUnwarp_sucess.use_DewarpNet_eval 來執行 kong_evalUnwarp_sucess.m

os.chdir(ord_dir)  ### step4 跳回 主程式資料夾
print(os.getcwd())
