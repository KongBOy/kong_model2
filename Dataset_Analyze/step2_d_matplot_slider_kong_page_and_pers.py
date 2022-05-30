import sys
sys.path.append("kong_util")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

from kong_util.util import get_xy_map
from step2_a_distort_page_and_pers import build_page_move_map, build_perspective_move_map


fig, ax = plt.subplots()
fig.set_size_inches(5, 8)
fig.subplots_adjust(left=0.13, bottom=0.20)
plt.xlim(128 - 200 * 0.8, 128 + 200 * 0.8)
plt.ylim(192 - 300 * 0.8, 192 + 300 * 0.8)

axcolor = 'lightgoldenrodyellow'  ### slide bar 的背景顏色：金黃淡亮色
bar_h = 0.02  ### bar的高度
bar_w = 0.75  ### bar的寬度
bar_x_start = 0.15  ### bar左上角的x
bar_y_start = 0.02  ### bar左上角的y

### 我想要看的參數
dis_type = "pers"
# row = 400
# col = 300
row = 384
col = 256
top_curl    = 0
down_curl   = 0
lr_shift    = 0
ratio_col_t = 0
ratio_row   = 0
ratio_col_d = 0

################################################################################################################################
### 在畫布上 畫出TextBox 並 填入想要的字
title = plt.axes([0.3, 0.9, 0.3, 0.05], facecolor=axcolor)
TextBox(title, "", "row=%i, col=%i" % (row, col),)

###################################################
### Slide bar 部分
### 1.圖 → 2.Slide → Slide功能
### 1.在 畫布上 畫出 bar圖
ratio_col_t_ax  = plt.axes([bar_x_start, bar_y_start * 7, bar_w, bar_h], facecolor=axcolor)
ratio_row_ax    = plt.axes([bar_x_start, bar_y_start * 6, bar_w, bar_h], facecolor=axcolor)
ratio_col_d_ax  = plt.axes([bar_x_start, bar_y_start * 5, bar_w, bar_h], facecolor=axcolor)
top_curl_ax     = plt.axes([bar_x_start, bar_y_start * 4, bar_w, bar_h], facecolor=axcolor)
down_curl_ax    = plt.axes([bar_x_start, bar_y_start * 3, bar_w, bar_h], facecolor=axcolor)
lr_shift_ax     = plt.axes([bar_x_start, bar_y_start * 2, bar_w, bar_h], facecolor=axcolor)

### 1. → 2. 把 bar 跟 滑動bar 做鏈結
### Slider( 畫布上的bar圖, 顯示在畫布上的字, 最小值, 最大值, 初始直, 走一步的單位  )
ratio_col_t_sl  = Slider(ratio_col_t_ax , 'ratio_col_t' , 0  , 1 , valinit=1, valstep=0.01)
ratio_row_sl    = Slider(ratio_row_ax   , 'ratio_row'   , 0  , 1 , valinit=1, valstep=0.01)
ratio_col_d_sl  = Slider(ratio_col_d_ax , 'ratio_col_d' , 0  , 1 , valinit=1, valstep=0.01)
top_curl_sl     = Slider(top_curl_ax    , 'top_curl'    , 0  , 100 , valinit=0, valstep=1)
down_curl_sl    = Slider(down_curl_ax   , 'down_curl'   , 0  , 100 , valinit=0, valstep=1)
lr_shift_sl     = Slider(lr_shift_ax    , 'lr_shift'    , 0  , 100 , valinit=0, valstep=1)

### 3.Slide功能
### 初始化 一些 等等要用到的東西
x, y = get_xy_map(int(row), int(col))  ### 從 0~row 或 0~col
xy_map = np.dstack((x, y))
ax_img = ax.scatter(x, y)
ax = ax.invert_yaxis()
# ax_img = ax.scatter(xy_map[:,0], xy_map[:,1], c = np.arange(row*col), cmap="brg") ### 有漸層顏色，但是很慢所以註解掉了

### 定義 滑動bar時要做的事情
def apply_move():
    global row, col, top_curl, down_curl, lr_shift, ratio_col_t, ratio_row, ratio_col_d, dis_type
    ratio_col_t = ratio_col_t_sl.val
    ratio_row   = ratio_row_sl  .val
    ratio_col_d = ratio_col_d_sl.val
    top_curl    = top_curl_sl   .val
    down_curl   = down_curl_sl  .val
    lr_shift    = lr_shift_sl   .val

    # global row, col, top_curl, down_curl, lr_shift, ratio_col_t, dis_type, alpha
    if  (dis_type == "pers"): move = build_perspective_move_map(int(row), int(col), ratio_col_t, ratio_row, ratio_col_d)
    elif(dis_type == "page"): move = build_page_move_map       (int(row), int(col), top_curl, down_curl, lr_shift)

    proc_xy_map = xy_map + move
    ax_img.set_offsets(proc_xy_map.reshape(-1, 2))  ### 限定要放 flatten的形式喔！ [..., 2]
    print("lr_shift_max = ", abs(move[..., 0]).max(), ", ratio_col_t_max = ", abs(move[..., 1]).max(), ", dis_type=", dis_type)

def update(val):
    apply_move()


### 2. -> 3. Slide功能 跟 Slide bar做連結
ratio_col_t_sl .on_changed(update)
ratio_row_sl   .on_changed(update)
ratio_col_d_sl .on_changed(update)
top_curl_sl    .on_changed(update)
down_curl_sl   .on_changed(update)
lr_shift_sl    .on_changed(update)
###################################################
### Reset按鈕部分
### 1.圖 → 2.按鈕 → 3.按鈕功能 做連結
reset_ax = plt.axes([0.4, 0.01, 0.2, 0.03], facecolor=axcolor)  ### 1.畫圖出來
reset_btn = Button(reset_ax, "Reset")  ### 1. -> 2. 圖和button做連結
def reset(event):  ### 3.定義功能
    if  (dis_type == "pers"):
        ratio_col_t_sl.reset()
        ratio_row_sl  .reset()
        ratio_col_d_sl.reset()

    elif(dis_type == "page"):
        top_curl_sl   .reset()
        down_curl_sl  .reset()
        lr_shift_sl   .reset()

reset_btn.on_clicked(reset)  ### 2. -> 3. 把功能和button做連結

## Apply按鈕部分
## 1.圖 → 2.按鈕 → 3.按鈕功能 做連結
apply_ax = plt.axes([0.2, 0.01, 0.2, 0.03], facecolor=axcolor)  ### 1.畫圖出來
apply_btn = Button(apply_ax, "Apply")  ### 1. -> 2. 圖和button做連結
def Apply(event):  ### 3.定義功能
    apply_move()
apply_btn.on_clicked(Apply)  ### 2. -> 3. 把功能和button做連結


## Apply按鈕部分
## 1.圖 → 2.按鈕 → 3.按鈕功能 做連結
stick_ax = plt.axes([0.6, 0.01, 0.2, 0.03], facecolor=axcolor)  ### 1.畫圖出來
stick_btn = Button(stick_ax, "Stick")  ### 1. -> 2. 圖和button做連結
def Stick(event):  ### 3.定義功能
    global row, col, top_curl, down_curl, lr_shift, ratio_col_t, dis_type, alpha
    if  (dis_type == "pers"): move = build_perspective_move_map(int(row), int(col), ratio_col_t, ratio_row, ratio_col_d)
    elif(dis_type == "page"): move = build_page_move_map       (int(row), int(col), top_curl, down_curl, lr_shift)
    global xy_map
    xy_map = xy_map + move

stick_btn.on_clicked(Stick)  ### 2. -> 3. 把功能和button做連結

###################################################
### 左邊 交換type部分
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('pers', 'page'), active=0)
def change_type(label):
    global dis_type
    dis_type = label

radio.on_clicked(change_type)

plt.show()
