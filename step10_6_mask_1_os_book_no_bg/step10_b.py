import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a.py"]
run = "build().run()"

#############################  no-bg  ##################################
### 1. ch 結果超棒就直接結束了 沒有做其他嘗試
sb.run(same_command + [f"mask_ch001_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_ch002_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_ch004_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_ch008_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_ch016_sigmoid_bce_ep060.{run}"])  ### 127.35跑
sb.run(same_command + [f"mask_ch032_sigmoid_bce_ep060.{run}"])  ### 127.35跑
