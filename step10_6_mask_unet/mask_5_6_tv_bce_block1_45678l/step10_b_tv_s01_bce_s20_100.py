'''
目前只有 step10b 一定需要切換資料夾到 該step10b所在的資料夾 才能執行喔！
'''

import subprocess as sb
### 沒有作用~~ 切不過去喔~~
# sb.run(["conda.bat", "deactivate"])
# sb.run(["conda.bat", "activate", "tensorflow_cpu"])

cmd_python_step10_a = ["python", "step10_a.py"]
run = "build().run()"


# sb.run(cmd_python_step10_a + [f"ch016_sig_L6_ep060_tv_s001_bce_s020.{run}"])
# sb.run(cmd_python_step10_a + [f"ch016_sig_L6_ep060_tv_s001_bce_s040.{run}"])
# sb.run(cmd_python_step10_a + [f"ch016_sig_L6_ep060_tv_s001_bce_s060.{run}"])
# sb.run(cmd_python_step10_a + [f"ch016_sig_L6_ep060_tv_s001_bce_s080.{run}"])
# sb.run(cmd_python_step10_a + [f"ch016_sig_L6_ep060_tv_s001_bce_s100.{run}"])

# sb.run(cmd_python_step10_a + [f"ch032_sig_L5_ep060_tv_s001_bce_s020.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L5_ep060_tv_s001_bce_s040.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L5_ep060_tv_s001_bce_s060.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L5_ep060_tv_s001_bce_s080.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L5_ep060_tv_s001_bce_s100.{run}"])

# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s001_bce_s020.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s001_bce_s040.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s001_bce_s060.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s001_bce_s080.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s001_bce_s100.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s020_bce_s020.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s020_bce_s040.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s020_bce_s060.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s020_bce_s080.{run}"])
# sb.run(cmd_python_step10_a + [f"ch064_sig_L4_ep060_tv_s020_bce_s100.{run}"])

# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s001_bce_s020.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s001_bce_s040.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s001_bce_s060.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s001_bce_s080.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s001_bce_s100.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s020.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s040.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s060.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s080.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s100.{run}"])
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s120.{run}"])  ###127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s020_bce_s140.{run}"])  ###127.29

# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s120.{run}"])  ### 127.35
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s140.{run}"])  ### 127.35
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s100.{run}"])  ### 127.35
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s080.{run}"])  ### 127.35
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s060.{run}"])  ### 127.35
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s040.{run}"])  ### 127.35
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s040_bce_s020.{run}"])  ### 127.35

# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s140.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s120.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s100.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s160.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s180.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s080.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s060.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s040.{run}"])  ### 還沒 127.28
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s060_bce_s020.{run}"])  ### 還沒 127.28

# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s160.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s140.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s120.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s180.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s100.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s080.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s060.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s040.{run}"])  ### 還沒 127.29
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s080_bce_s020.{run}"])  ### 還沒 127.29

# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s180.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s160.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s140.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s200.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s120.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s100.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s080.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s060.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s040.{run}"])  ### 還沒 127.55
# sb.run(cmd_python_step10_a + [f"ch032_sig_L4_ep060_tv_s100_bce_s020.{run}"])  ### 還沒 127.55