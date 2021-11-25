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
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch128_sig_bce_ep060.{run}"])  ### 127.37跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_bce_ep060.{run}"])  ### 127.37跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch016_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch008_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch004_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch002_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch001_sig_bce_ep060.{run}"])  ### 127.35跑
### 1b. ch and epoch
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch128_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch064_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch016_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch008_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch004_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch002_sig_bce_ep200.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch001_sig_bce_ep200.{run}"])  ### 127.35跑
### 2. level
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L2_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L3_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L4_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L5_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L8_sig_bce_ep060.{run}"])  ### 127.35跑
### 3. no-concat
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to2noC_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to3noC_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to4noC_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to5noC_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to6noC_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to7noC_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_2to8noC_sig_bce_ep060.{run}"])  ### 127.35跑
### 4. skip use add
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L2_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L3_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L4_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L5_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L6_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L7_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
sb.run(cmd_python_step10_a + [f"mask_h_bg_ch032_L8_skipAdd_sig_bce_ep060.{run}"])  ### 127.35跑
