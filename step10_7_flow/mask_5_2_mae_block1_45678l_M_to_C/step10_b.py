'''
目前只有 step10b 一定需要切換資料夾到 該step10b所在的資料夾 才能執行喔！
'''

import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

same_command = ["python", "step10_a.py"]
run = "build().run()"

#### l2 ############################################################################################
# sb.run(same_command + [f"L2_ch128_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch128_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch128_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch128_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch128_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch064_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch064_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch064_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch064_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch064_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch032_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch032_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch032_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch032_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch032_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch016_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch016_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch016_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch016_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch008_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch008_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L2_ch008_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L2_ch008_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L2_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L2_ch004_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch004_mae_s100.{run}"])

# sb.run(same_command + [f"L2_ch002_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch002_mae_s100.{run}"])

# sb.run(same_command + [f"L2_ch001_mae_s001.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s020.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s040.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s060.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s080.{run}"])
# sb.run(same_command + [f"L2_ch001_mae_s100.{run}"])
#### 3l ############################################################################################
# sb.run(same_command + [f"L3_ch128_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch128_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch128_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch128_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch128_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch064_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch064_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch064_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch064_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch064_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch032_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch032_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch032_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch032_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch032_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch016_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch016_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch016_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch016_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch008_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch008_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L3_ch008_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L3_ch008_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L3_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L3_ch004_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch004_mae_s100.{run}"])

# sb.run(same_command + [f"L3_ch002_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch002_mae_s100.{run}"])

# sb.run(same_command + [f"L3_ch001_mae_s001.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s020.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s040.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s060.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s080.{run}"])
# sb.run(same_command + [f"L3_ch001_mae_s100.{run}"])
#### 4l ############################################################################################
# sb.run(same_command + [f"L4_ch064_mae_s001.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s020.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s040.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s060.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s080.{run}"])
# sb.run(same_command + [f"L4_ch064_mae_s100.{run}"])
### E_relu
# sb.run(same_command + [f"L4_ch064_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L4_ch064_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L4_ch064_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L4_ch032_mae_s001.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s020.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s040.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s060.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s080.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s100.{run}"])  ### 127.27
### E_relu
# sb.run(same_command + [f"L4_ch032_mae_s001_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s020_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s040_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s060_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s080_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s100_E_relu .{run}"])  ### 127.27
### no_Bias
# sb.run(same_command + [f"L4_ch032_mae_s001_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s020_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s040_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s060_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s080_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch032_mae_s100_no_Bias.{run}"])  ### 127.27
### coord_conv
# sb.run(same_command + [f"L4_ch032_mae_s001_coord_conv.{run}"])  ### 127.27
####################
# sb.run(same_command + [f"L4_ch016_mae_s001.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s020.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s040.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s060.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s080.{run}"])  ### 127.37
# sb.run(same_command + [f"L4_ch016_mae_s100.{run}"])  ### 127.37
### E_relu
# sb.run(same_command + [f"L4_ch016_mae_s001_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s020_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s040_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s060_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s080_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s100_E_relu .{run}"])  ### 127.27
### no_Bias
# sb.run(same_command + [f"L4_ch016_mae_s001_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s020_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s040_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s060_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s080_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch016_mae_s100_no_Bias.{run}"])  ### 127.27
### coord_conv
# sb.run(same_command + [f"L4_ch016_mae_s001_coord_conv.{run}"])  ### 127.27
####################
# sb.run(same_command + [f"L4_ch008_mae_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch008_mae_s100.{run}"])  ### 127.49
### E_relu
# sb.run(same_command + [f"L4_ch008_mae_s001_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s020_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s040_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s060_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s080_E_relu .{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s100_E_relu .{run}"])  ### 127.27
### no_Bias
# sb.run(same_command + [f"L4_ch008_mae_s001_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s020_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s040_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s060_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s080_no_Bias.{run}"])  ### 127.27
# sb.run(same_command + [f"L4_ch008_mae_s100_no_Bias.{run}"])  ### 127.27
### coord_conv
# sb.run(same_command + [f"L4_ch008_mae_s001_coord_conv.{run}"])  ### 127.27
####################
# sb.run(same_command + [f"L4_ch004_mae_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch004_mae_s100.{run}"])  ### 127.49
### E_relu
# sb.run(same_command + [f"L4_ch004_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L4_ch004_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L4_ch004_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L4_ch002_mae_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch002_mae_s100.{run}"])  ### 127.49
####################
# sb.run(same_command + [f"L4_ch001_mae_s001.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s020.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s040.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s060.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s080.{run}"])  ### 127.49
# sb.run(same_command + [f"L4_ch001_mae_s100.{run}"])  ### 127.49

#### 5l ############################################################################################
# sb.run(same_command + [f"L5_ch032_mae_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch032_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch032_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch032_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch032_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L5_ch016_mae_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch016_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch016_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch016_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L5_ch008_mae_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch008_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch008_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch008_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L5_ch004_mae_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch004_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch004_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch004_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch004_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L5_ch002_mae_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch002_mae_s100.{run}"])  ### 127.55
### E_relu
# sb.run(same_command + [f"L5_ch002_mae_s001_E_relu.{run}"])
### no_Bias
# sb.run(same_command + [f"L5_ch002_mae_s001_no_Bias.{run}"])
### coord_conv
# sb.run(same_command + [f"L5_ch002_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L5_ch001_mae_s001.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s020.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s040.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s060.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s080.{run}"])  ### 127.55
# sb.run(same_command + [f"L5_ch001_mae_s100.{run}"])  ### 127.55

#### 6l ############################################################################################
### coord_conv
# sb.run(same_command + [f"L6_ch064_mae_s001_coord_conv.{run}"])
####################
### coord_conv
# sb.run(same_command + [f"L6_ch032_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L6_ch016_mae_s001.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s020.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s040.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s060.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s080.{run}"])
# sb.run(same_command + [f"L6_ch016_mae_s100.{run}"])
### coord_conv
# sb.run(same_command + [f"L6_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L6_ch008_mae_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch008_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L6_ch004_mae_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch004_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch004_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L6_ch002_mae_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch002_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch002_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L6_ch001_mae_s001.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s020.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s040.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s060.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s080.{run}"])  ### 127.28
# sb.run(same_command + [f"L6_ch001_mae_s100.{run}"])  ### 127.28
### coord_conv
# sb.run(same_command + [f"L6_ch001_mae_s001_coord_conv.{run}"])

#### 7l ############################################################################################
### coord_conv
# sb.run(same_command + [f"L7_ch032_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L7_ch016_mae_s001.{run}"])
### coord_conv
# sb.run(same_command + [f"L7_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L7_ch008_mae_s001.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s020.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s040.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s060.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s080.{run}"])
# sb.run(same_command + [f"L7_ch008_mae_s100.{run}"])
### coord_conv
# sb.run(same_command + [f"L7_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L7_ch004_mae_s001.{run}"])
### coord_conv
# sb.run(same_command + [f"L7_ch004_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L7_ch002_mae_s001.{run}"])
### coord_conv
# sb.run(same_command + [f"L7_ch002_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L7_ch001_mae_s001.{run}"])
#### 8l ############################################################################################
### coord_conv
# sb.run(same_command + [f"L8_ch016_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L8_ch008_mae_s001.{run}"])
### coord_conv
# sb.run(same_command + [f"L8_ch008_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L8_ch004_mae_s001.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s020.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s040.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s060.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s080.{run}"])
# sb.run(same_command + [f"L8_ch004_mae_s100.{run}"])
### coord_conv
# sb.run(same_command + [f"L8_ch004_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L8_ch002_mae_s001.{run}"])
### coord_conv
# sb.run(same_command + [f"L8_ch002_mae_s001_coord_conv.{run}"])
####################
# sb.run(same_command + [f"L8_ch001_mae_s001.{run}"])
### coord_conv
# sb.run(same_command + [f"L8_ch001_mae_s001_coord_conv.{run}"])