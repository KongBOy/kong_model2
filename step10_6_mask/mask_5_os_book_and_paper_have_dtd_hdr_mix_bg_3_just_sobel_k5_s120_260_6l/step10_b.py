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
### 1a. ch
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s120_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s140_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s160_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s180_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s200_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s220_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s240_6l_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_h_bg_ch032_sig_sobel_k5_s260_6l_ep060.{run}"])  ### 127.35跑
