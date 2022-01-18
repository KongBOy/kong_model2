import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

from step2_a_distort_curl_and_fold import get_xy_f_and_m, get_dis_move_map

fig, ax = plt.subplots()
fig.set_size_inches(6, 8)
plt.subplots_adjust(left=0.13, bottom=0.20)
plt.xlim(-100, 400)
plt.ylim(-100, 500)

axcolor = 'lightgoldenrodyellow'  ### slide bar 的背景顏色：金黃淡亮色
bar_h = 0.02  ### bar的高度
bar_w = 0.75  ### bar的寬度
bar_x_start = 0.15  ### bar左上角的x
bar_y_start = 0.02  ### bar左上角的y

### 我想要看的參數
curve_type = "curl"
# row = 400
# col = 300
# row = 256
# col = 256
row = 384
col = 256
vert_x = 0
vert_y = 0
move_x = 0
move_y = 0
alpha = 0

################################################################################################################################
### 在畫布上 畫出TextBox 並 填入想要的字
title = plt.axes([0.3, 0.9, 0.3, 0.05], facecolor=axcolor)
TextBox(title, "", "row=%i, col=%i" % (row, col),)

###################################################
### Slide bar 部分
### 1.圖 → 2.Slide → Slide功能
### 1.在 畫布上 畫出 bar圖
vert_x_ax = plt.axes([bar_x_start, bar_y_start * 7, bar_w, bar_h], facecolor=axcolor)
vert_y_ax = plt.axes([bar_x_start, bar_y_start * 6, bar_w, bar_h], facecolor=axcolor)
move_x_ax = plt.axes([bar_x_start, bar_y_start * 5, bar_w, bar_h], facecolor=axcolor)
move_y_ax = plt.axes([bar_x_start, bar_y_start * 4, bar_w, bar_h], facecolor=axcolor)
alpha_c_ax  = plt.axes([bar_x_start, bar_y_start * 3, bar_w, bar_h], facecolor=axcolor)
alpha_f_ax  = plt.axes([bar_x_start, bar_y_start * 2, bar_w, bar_h], facecolor=axcolor)

### 1. → 2. 把 bar 跟 滑動bar 做鏈結
### Slider( 畫布上的bar圖, 顯示在畫布上的字, 最小值, 最大值, 初始直, 走一步的單位  )
vert_y_sl = Slider(vert_y_ax , 'vert_y', 0  , row  , valinit=0, valstep=0.1)
vert_x_sl = Slider(vert_x_ax , 'vert_x', 0  , col  , valinit=0, valstep=0.1)
move_x_sl = Slider(move_x_ax , 'move_x', -28.80, 28.80, valinit=0, valstep=0.1)
move_y_sl = Slider(move_y_ax , 'move_y', -28.80, 28.80, valinit=0, valstep=0.1)
alpha_c_sl  = Slider(alpha_c_ax  , 'alpha_c' , 0.85, 1.70 * 2 , valinit=0.85, valstep=0.01)
alpha_f_sl  = Slider(alpha_f_ax  , 'alpha_f' , 0.2   , 150 , valinit=0, valstep=0.1)

### 3.Slide功能
### 初始化 一些 等等要用到的東西
xy_f, _ = get_xy_f_and_m(x_min=0, x_max=int(col) - 1, y_min=0, y_max=int(row) - 1, w_res=int(col), h_res=int(row))
ax_img = ax.scatter(xy_f[:, 0], xy_f[:, 1])
ax = ax.invert_yaxis()
# ax_img = ax.scatter(xy_f[:,0], xy_f[:,1], c = np.arange(row*col), cmap="brg") ### 有漸層顏色，但是很慢所以註解掉了

### 定義 滑動bar時要做的事情
def apply_move():
    global row, col, vert_x, vert_y, move_x, move_y, curve_type, alpha
    vert_x = vert_x_sl.val
    vert_y = vert_y_sl.val
    move_x = move_x_sl.val
    move_y = move_y_sl.val
    if  (curve_type == "curl"): alpha = alpha_c_sl.val
    elif(curve_type == "fold"): alpha = alpha_f_sl.val
    # global row, col, vert_x, vert_y, move_x, move_y, curve_type, alpha
    if  (curve_type == "curl"): move_f, _ = get_dis_move_map(int(row), int(col), int(vert_x), int(vert_y), move_x, move_y, curve_type, alpha=alpha)
    elif(curve_type == "fold"): move_f, _ = get_dis_move_map(int(row), int(col), int(vert_x), int(vert_y), move_x, move_y, curve_type, alpha=alpha)
    proc_xy_f = xy_f + move_f
    ax_img.set_offsets(proc_xy_f)  ### 限定要放 flatten的形式喔！ [..., 2]
    print("move_x_max = ", abs(move_f[:, 0]).max(), ", move_y_max = ", abs(move_f[:, 1]).max(), ", curve_type=", curve_type, ", alpha=", alpha)

def update(val):
    apply_move()


### 2. -> 3. Slide功能 跟 Slide bar做連結
vert_x_sl.on_changed(update)
vert_y_sl.on_changed(update)
move_x_sl.on_changed(update)
move_y_sl.on_changed(update)
alpha_c_sl.on_changed(update)
alpha_f_sl.on_changed(update)
###################################################
### Reset按鈕部分
### 1.圖 → 2.按鈕 → 3.按鈕功能 做連結
reset_ax = plt.axes([0.4, 0.01, 0.2, 0.03], facecolor=axcolor)  ### 1.畫圖出來
reset_btn = Button(reset_ax, "Reset")  ### 1. -> 2. 圖和button做連結
def reset(event):  ### 3.定義功能
    vert_x_sl.reset()
    vert_y_sl.reset()
    move_x_sl.reset()
    move_y_sl.reset()
    # alpha_c_sl.reset()
    if  (curve_type == "curl"): alpha_c_sl.reset()
    elif(curve_type == "fold"): alpha_f_sl.reset()
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
    global row, col, vert_x, vert_y, move_x, move_y, curve_type, alpha
    if  (curve_type == "curl"): move_f, _ = get_dis_move_map(int(row), int(col), int(vert_x), int(vert_y), move_x, move_y, curve_type, alpha)
    elif(curve_type == "fold"): move_f, _ = get_dis_move_map(int(row), int(col), int(vert_x), int(vert_y), move_x, move_y, curve_type, alpha)
    global xy_f
    xy_f = xy_f + move_f
stick_btn.on_clicked(Stick)  ### 2. -> 3. 把功能和button做連結

###################################################
### 左邊 交換type部分
rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('fold', 'curl'), active=1)
def change_type(label):
    global curve_type
    curve_type = label
radio.on_clicked(change_type)

plt.show()
