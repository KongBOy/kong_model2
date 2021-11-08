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
sb.run(same_command + [f"ch016_sig_6l_tv_s001_bce_s020_ep060.{run}"])
sb.run(same_command + [f"ch016_sig_6l_tv_s001_bce_s040_ep060.{run}"])
sb.run(same_command + [f"ch016_sig_6l_tv_s001_bce_s060_ep060.{run}"])
sb.run(same_command + [f"ch016_sig_6l_tv_s001_bce_s080_ep060.{run}"])
sb.run(same_command + [f"ch016_sig_6l_tv_s001_bce_s100_ep060.{run}"])
