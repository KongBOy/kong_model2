'''
目前只有 step10b 一定需要切換資料夾到 該step10b所在的資料夾 才能執行喔！
'''

import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a.py"]
run = "build().run()"

#### 3l ############################################################################################
# sb.run(same_command + [f"l3_ch128_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch128_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch128_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch128_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch128_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch128_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch064_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch064_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch064_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch064_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch064_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch064_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch032_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch032_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch032_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch032_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch032_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch032_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch016_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch016_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch016_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch016_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch016_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch016_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch008_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch008_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch008_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch008_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch008_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch008_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch004_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch004_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch004_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch004_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch004_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch004_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch002_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch002_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch002_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch002_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch002_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch002_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l3_ch001_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch001_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch001_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch001_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch001_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l3_ch001_sig_ep060_bce_s100.{run}"])  ### 127.27
#### 4l ############################################################################################
# sb.run(same_command + [f"l4_ch064_sig_ep060_bce_s001.{run}"])
# sb.run(same_command + [f"l4_ch064_sig_ep060_bce_s020.{run}"])
# sb.run(same_command + [f"l4_ch064_sig_ep060_bce_s040.{run}"])
# sb.run(same_command + [f"l4_ch064_sig_ep060_bce_s060.{run}"])
# sb.run(same_command + [f"l4_ch064_sig_ep060_bce_s080.{run}"])
# sb.run(same_command + [f"l4_ch064_sig_ep060_bce_s100.{run}"])

# sb.run(same_command + [f"l4_ch032_sig_ep060_bce_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"l4_ch032_sig_ep060_bce_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"l4_ch032_sig_ep060_bce_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"l4_ch032_sig_ep060_bce_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"l4_ch032_sig_ep060_bce_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"l4_ch032_sig_ep060_bce_s100.{run}"])  ### 127.27

# sb.run(same_command + [f"l4_ch016_sig_ep060_bce_s001.{run}"])  ### 127.37
# sb.run(same_command + [f"l4_ch016_sig_ep060_bce_s020.{run}"])  ### 127.37
# sb.run(same_command + [f"l4_ch016_sig_ep060_bce_s040.{run}"])  ### 127.37
# sb.run(same_command + [f"l4_ch016_sig_ep060_bce_s060.{run}"])  ### 127.37
# sb.run(same_command + [f"l4_ch016_sig_ep060_bce_s080.{run}"])  ### 127.37
# sb.run(same_command + [f"l4_ch016_sig_ep060_bce_s100.{run}"])  ### 127.37

# sb.run(same_command + [f"l4_ch008_sig_ep060_bce_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch008_sig_ep060_bce_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch008_sig_ep060_bce_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch008_sig_ep060_bce_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch008_sig_ep060_bce_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch008_sig_ep060_bce_s100.{run}"])  ### 127.49

# sb.run(same_command + [f"l4_ch004_sig_ep060_bce_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch004_sig_ep060_bce_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch004_sig_ep060_bce_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch004_sig_ep060_bce_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch004_sig_ep060_bce_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch004_sig_ep060_bce_s100.{run}"])  ### 127.49

# sb.run(same_command + [f"l4_ch002_sig_ep060_bce_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch002_sig_ep060_bce_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch002_sig_ep060_bce_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch002_sig_ep060_bce_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch002_sig_ep060_bce_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch002_sig_ep060_bce_s100.{run}"])  ### 127.49

# sb.run(same_command + [f"l4_ch001_sig_ep060_bce_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch001_sig_ep060_bce_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch001_sig_ep060_bce_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch001_sig_ep060_bce_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch001_sig_ep060_bce_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"l4_ch001_sig_ep060_bce_s100.{run}"])  ### 127.49

#### 5l ############################################################################################
# sb.run(same_command + [f"l5_ch032_sig_ep060_bce_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch032_sig_ep060_bce_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch032_sig_ep060_bce_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch032_sig_ep060_bce_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch032_sig_ep060_bce_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch032_sig_ep060_bce_s100.{run}"])  ### 127.55

# sb.run(same_command + [f"l5_ch016_sig_ep060_bce_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch016_sig_ep060_bce_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch016_sig_ep060_bce_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch016_sig_ep060_bce_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch016_sig_ep060_bce_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch016_sig_ep060_bce_s100.{run}"])  ### 127.55

# sb.run(same_command + [f"l5_ch008_sig_ep060_bce_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch008_sig_ep060_bce_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch008_sig_ep060_bce_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch008_sig_ep060_bce_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch008_sig_ep060_bce_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch008_sig_ep060_bce_s100.{run}"])  ### 127.55

# sb.run(same_command + [f"l5_ch004_sig_ep060_bce_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch004_sig_ep060_bce_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch004_sig_ep060_bce_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch004_sig_ep060_bce_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch004_sig_ep060_bce_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch004_sig_ep060_bce_s100.{run}"])  ### 127.55

# sb.run(same_command + [f"l5_ch002_sig_ep060_bce_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch002_sig_ep060_bce_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch002_sig_ep060_bce_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch002_sig_ep060_bce_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch002_sig_ep060_bce_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch002_sig_ep060_bce_s100.{run}"])  ### 127.55

# sb.run(same_command + [f"l5_ch001_sig_ep060_bce_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch001_sig_ep060_bce_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch001_sig_ep060_bce_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch001_sig_ep060_bce_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch001_sig_ep060_bce_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"l5_ch001_sig_ep060_bce_s100.{run}"])  ### 127.55

#### 6l ############################################################################################
# sb.run(same_command + [f"l6_ch016_sig_ep060_bce_s001.{run}"])
# sb.run(same_command + [f"l6_ch016_sig_ep060_bce_s020.{run}"])
# sb.run(same_command + [f"l6_ch016_sig_ep060_bce_s040.{run}"])
# sb.run(same_command + [f"l6_ch016_sig_ep060_bce_s060.{run}"])
# sb.run(same_command + [f"l6_ch016_sig_ep060_bce_s080.{run}"])
# sb.run(same_command + [f"l6_ch016_sig_ep060_bce_s100.{run}"])

# sb.run(same_command + [f"l6_ch008_sig_ep060_bce_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch008_sig_ep060_bce_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch008_sig_ep060_bce_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch008_sig_ep060_bce_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch008_sig_ep060_bce_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch008_sig_ep060_bce_s100.{run}"])  ### 127.28

# sb.run(same_command + [f"l6_ch004_sig_ep060_bce_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch004_sig_ep060_bce_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch004_sig_ep060_bce_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch004_sig_ep060_bce_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch004_sig_ep060_bce_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch004_sig_ep060_bce_s100.{run}"])  ### 127.28

# sb.run(same_command + [f"l6_ch002_sig_ep060_bce_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch002_sig_ep060_bce_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch002_sig_ep060_bce_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch002_sig_ep060_bce_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch002_sig_ep060_bce_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch002_sig_ep060_bce_s100.{run}"])  ### 127.28

# sb.run(same_command + [f"l6_ch001_sig_ep060_bce_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch001_sig_ep060_bce_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch001_sig_ep060_bce_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch001_sig_ep060_bce_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch001_sig_ep060_bce_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"l6_ch001_sig_ep060_bce_s100.{run}"])  ### 127.28

#### 7l ############################################################################################
# sb.run(same_command + [f"l7_ch008_sig_ep060_bce_s001.{run}"])
# sb.run(same_command + [f"l7_ch008_sig_ep060_bce_s020.{run}"])
# sb.run(same_command + [f"l7_ch008_sig_ep060_bce_s040.{run}"])
# sb.run(same_command + [f"l7_ch008_sig_ep060_bce_s060.{run}"])
# sb.run(same_command + [f"l7_ch008_sig_ep060_bce_s080.{run}"])
# sb.run(same_command + [f"l7_ch008_sig_ep060_bce_s100.{run}"])
#### 8l ############################################################################################
# sb.run(same_command + [f"l8_ch004_sig_ep060_bce_s001.{run}"])
# sb.run(same_command + [f"l8_ch004_sig_ep060_bce_s020.{run}"])
# sb.run(same_command + [f"l8_ch004_sig_ep060_bce_s040.{run}"])
# sb.run(same_command + [f"l8_ch004_sig_ep060_bce_s060.{run}"])
# sb.run(same_command + [f"l8_ch004_sig_ep060_bce_s080.{run}"])
# sb.run(same_command + [f"l8_ch004_sig_ep060_bce_s100.{run}"])
