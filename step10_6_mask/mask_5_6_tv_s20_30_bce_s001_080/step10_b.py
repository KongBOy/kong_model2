'''
目前只有 step10b 一定需要切換資料夾到 該step10b所在的資料夾 才能執行喔！
'''

import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a.py"]
run = "build().run()"

############################  have_bg  #################################
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s001.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s020.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s040.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s20_bce_s080.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s001.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s020.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s040.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_6l_ep060_tv_s30_bce_s080.{run}"])  ### 127.35跑