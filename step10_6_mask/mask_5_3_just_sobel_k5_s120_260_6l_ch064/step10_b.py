'''
目前只有 step10b 一定需要切換資料夾到 該step10b所在的資料夾 才能執行喔！
'''

import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

cmd_python_step10_a = ["python", "step10_a.py"]
run = "build().run()"

############################  have_bg  #################################
### 1a. ch
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s120_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s140_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s160_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s180_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s200_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s220_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s240_L6_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_sobel_k5_s260_L6_ep060.{run}"])  ### 127.35跑
