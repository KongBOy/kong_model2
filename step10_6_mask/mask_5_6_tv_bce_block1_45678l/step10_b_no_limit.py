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
sb.run(same_command + [f"ch128_no_limit_sig_4l_ep060.{run}"])

sb.run(same_command + [f"ch064_no_limit_sig_5l_ep060.{run}"])
sb.run(same_command + [f"ch128_no_limit_sig_5l_ep060.{run}"])

sb.run(same_command + [f"ch032_no_limit_sig_6l_ep060.{run}"])
sb.run(same_command + [f"ch064_no_limit_sig_6l_ep060.{run}"])
sb.run(same_command + [f"ch128_no_limit_sig_6l_ep060.{run}"])

sb.run(same_command + [f"ch016_no_limit_sig_7l_ep060.{run}"])
sb.run(same_command + [f"ch032_no_limit_sig_7l_ep060.{run}"])
sb.run(same_command + [f"ch064_no_limit_sig_7l_ep060.{run}"])

sb.run(same_command + [f"ch008_no_limit_sig_8l_ep060.{run}"])
sb.run(same_command + [f"ch016_no_limit_sig_8l_ep060.{run}"])
sb.run(same_command + [f"ch032_no_limit_sig_8l_ep060.{run}"])
