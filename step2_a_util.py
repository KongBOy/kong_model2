import sys
sys.path.append("kong_util")

from matplot_fig_ax_util import check_fig_ax_init
import numpy as np
from util import method1

def wt_calculate_before(dis_type, d_abs_norm, alpha):
    '''
    以前參考 https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
    已前的正mesh 是 y： 0~row-1, 切 row分
                   x： 0~col-1, 切 col分
                   這樣子算出的 d 較大， 所以 curl 的 d_abs_norm 要先 /100
    為了紀念留下來，麻煩不要用他，很多case 值都會超出範圍

    return wt的shape 取決於 d_abs_norm/alpha 的 shape
    '''
    if  (dis_type == "fold"): wt = alpha / (d_abs_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
    elif(dis_type == "curl"): wt = 1 - (d_abs_norm / 100 )**(alpha)
    return wt

def wt_calculate(dis_type, d_abs_norm, alpha):
    '''
    現在參考 paper17印度 用的 https://github.com/XiyanLiu/AGUN
    現在的正mesh 是 y： -0.95~+0.95, 切 row分
                   x： -0.95~+0.95, 切 col分
    alpha： 建議 alpha > 0 ，
        1. 用step4_wt_simulate_visual 嘗試以後 alpha > 0 感覺較正常， alpha == 0 時 不管 d 多少 wt 都為0， alpha < 0，儘管 alpha == -0.0， 只要 d 接近0 都會 負很大
        2. alpha > 0 比較符合 paper09 DocUNet 的描述

    return 的 wt 的shape 為 你傳入的 d_abs_norm/alpha 的shape
    '''
    if  (dis_type == "fold"): wt = alpha / (d_abs_norm + alpha + 0.00001)  ### +0.00001是怕分母為零
    elif(dis_type == "curl"): wt = 1 - (d_abs_norm )**(alpha)
    return wt


''' 給 paper17方法用的 核心： 利用調整後的 dis_coord_m 中間 一個boundary(new_bm)， 這new_bm 對應回的 new_fm 怎麼抓 '''
def step7b_dis_coord_big_find_ord_valid_mask_and_ord_valid_coord(dis_coord_big_m, boundary_value=1.0, visual=False, fig=None, ax=None, ax_c=None):
    if(visual): fig, ax, ax_c = check_fig_ax_init(fig=fig, ax=ax, ax_c=ax_c, fig_rows=1, fig_cols=5, ax_size=5, tight_layout=True)
    ##################################################################################################################
    h_res, w_res = dis_coord_big_m.shape[: 2]

    canvas_mask_x        = np.zeros(shape=(h_res, w_res))  ### 分開來看 x valid的部分
    canvas_mask_y        = np.zeros(shape=(h_res, w_res))  ### 分開來看 y valid的部分
    canvas_mask_xy       = np.zeros(shape=(h_res, w_res))  ### 確實fm沒錯， 只抓 值在-1~1 之間的區域 會和 image_disorted_mask 一樣
    canvas_mask_contour  = np.zeros(shape=(h_res, w_res))  ### 確實fm沒錯， 只抓 值在-1~1 之間的區域 會和 image_disorted_mask 一樣
    canvas_dis_coord     = np.zeros(shape=(h_res, w_res, 3))
    dis_coord_visual     = method1(x=dis_coord_big_m[..., 0], y=dis_coord_big_m[..., 1] )
    for go_r, dis_coord_m_row in enumerate(dis_coord_big_m):
        for go_c, xy_coord in enumerate(dis_coord_m_row):
            if(-boundary_value <= xy_coord[0] <= boundary_value): canvas_mask_x[go_r, go_c] = 1
            if(-boundary_value <= xy_coord[1] <= boundary_value): canvas_mask_y[go_r, go_c] = 1
            if(-boundary_value <= xy_coord[0] <= boundary_value and -1 <= xy_coord[1] <= boundary_value):
                canvas_mask_xy   [go_r, go_c] += 1
                canvas_dis_coord [go_r, go_c] = dis_coord_visual[go_r, go_c]
            if(-boundary_value       <= xy_coord[0] <= boundary_value       and -boundary_value       <= xy_coord[1] <= boundary_value      ): canvas_mask_contour[go_r, go_c] += 1
            if(-boundary_value - 0.1 <= xy_coord[0] <= boundary_value + 0.1 and -boundary_value - 0.1 <= xy_coord[1] <= boundary_value + 0.1): canvas_mask_contour[go_r, go_c] += 1
            if(-boundary_value - 0.2 <= xy_coord[0] <= boundary_value + 0.2 and -boundary_value - 0.2 <= xy_coord[1] <= boundary_value + 0.2): canvas_mask_contour[go_r, go_c] += 1
    if(visual):
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_x valid between -1~+1");       ax[ax_c].imshow(canvas_mask_x,       cmap="gray", vmin=0, vmax=1);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_y valid between -1~+1");       ax[ax_c].imshow(canvas_mask_y,       cmap="gray", vmin=0, vmax=1);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_xy valid between -1~+1");      ax[ax_c].imshow(canvas_mask_xy,      cmap="gray", vmin=0, vmax=1);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_valid_contour between -1.5~+1.5"); ax[ax_c].imshow(canvas_mask_contour, cmap="gray", vmin=0, vmax=5);  ax_c += 1
        ax[ax_c].set_title("step6.dis_coord boundary correspond \n ord_valid uv/fm"); ax[ax_c].imshow(canvas_dis_coord); ax_c += 1
        return fig, ax, ax_c, canvas_mask_xy
    else: return canvas_mask_xy

def apply_flow(data, flow, visual=False, before_title=None, after_title=None, fig=None, ax=None, ax_c=None):
    if(visual):
        fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=2, ax_size=5, tight_layout=True)
        ax[ax_c].imshow(data)
        if(before_title is not None): ax[ax_c].set_title(before_title)
        ax_c += 1
    '''
    data:
        img：  HWC(HW3), dtype == uint8
        mask： HWC(HW1), dtype == float， 可以丟 np.ones() 搭配 fm， 就會得到 fm mask 囉！
    flow: HWC(HW2), dtype == float
    '''
    data_type = ""
    if  (data.ndim == 2): data = data[..., np.newaxis]
    if  (data.shape[2] == 1): data_type = "mask"
    elif(data.shape[2] == 3): data_type = "img"
    ########################################################################################################



    ########################################################################################################
    ### numpy(進 pytorch 前處理)
    data_t = data.astype(float).transpose((2, 0, 1))[np.newaxis, ...]  ### NCHW(1CHW), dtype == float
    flow_t = np.expand_dims(flow, 0)  ### NHWC(1HW2), dtype == float

    ########################################################################################################
    ### tensor
    import torch.nn.functional as F
    import torch
    data_t = torch.from_numpy(data_t)
    flow_t      = torch.from_numpy(flow_t)
    result_t = F.grid_sample(input=data_t, grid=flow_t)

    ########################################################################################################
    ### numpy
    if  (data_type == "img" ): result = result_t.numpy()[0].transpose(1, 2, 0).astype(np.uint8)  ### HWC, dtype == uint8
    elif(data_type == "mask"): result = result_t.numpy()[0].transpose(1, 2, 0)                   ### HWC, dtype == float

    if(visual):
        ax[ax_c].imshow(result)
        if(after_title is not None): ax[ax_c].set_title(after_title)
        ax_c += 1
        return result, fig, ax, ax_c
    else:
        return result


def apply_fm_to_get_dis_img(ord_img, fm, visual=False, before_title=None, after_title=None, fig=None, ax=None, ax_c=None):
    '''
    ord_img： HWC(HW3), dtype == uint8
    fm     ： HWC(HW2), dtype == float
    return dis_img
    '''
    return apply_flow(data=ord_img, flow=fm, visual=visual, before_title=before_title, after_title=after_title, fig=fig, ax=ax, ax_c=ax_c)

def apply_bm_to_get_rec_img(dis_img, bm, visual=False, before_title=None, after_title=None, fig=None, ax=None, ax_c=None):
    '''
    dis_img： HWC(HW3), dtype == uint8
    bm     ： HWC(HW2), dtype == float
    return rec_img
    '''
    return apply_flow(data=dis_img, flow=bm, visual=visual, before_title=before_title, after_title=after_title, fig=fig, ax=ax, ax_c=ax_c)

# def apply_fm_to_get_dis_img(ord_img, fm, visual=False, fig=None, ax=None, ax_c=None):
#     '''
#     ord_img： HWC(HW3), dtype == uint8
#     fm     ： HWC(HW2), dtype == float
#     bm     ： HWC(HW2), dtype == float
#     return dis_img
#     '''
#     if(visual):
#         fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=2, ax_size=5, tight_layout=True)
#         ax[ax_c].imshow(ord_img); ax_c += 1

#     ########################################################################################################
#     ### numpy(進 pytorch 前處理)
#     ord_img_t = ord_img.astype(float).transpose((2, 0, 1))[np.newaxis, ...]  ### NCHW(13HW), dtype == float
#     fm_t = np.expand_dims(fm, 0)  ### NHWC(1HW2), dtype == float

#     ########################################################################################################
#     ### tensor
#     import torch.nn.functional as F
#     import torch
#     ord_img_t = torch.from_numpy(ord_img_t)
#     fm_t      = torch.from_numpy(fm_t)
#     dis_img_t = F.grid_sample(input=ord_img_t, grid=fm_t)

#     ########################################################################################################
#     ### numpy
#     dis_img = dis_img_t.numpy()[0].transpose(1, 2, 0).astype(np.uint8)  ### HWC, dtype == uint8
#     if(visual):
#         ax[ax_c].imshow(dis_img)
#         ax_c += 1
#         return dis_img, fig, ax, ax_c
#     else:
#         return dis_img

# def apply_bm_to_get_dis_img(dis_img, bm, visual=False, fig=None, ax=None, ax_c=None):
#     '''
#     dis_img： HWC(HW3), dtype == uint8
#     bm     ： HWC(HW2), dtype == float
#     return rec_img
#     '''
#     if(visual):
#         fig, ax, ax_c = check_fig_ax_init(fig=None, ax=None, ax_c=None, fig_rows=1, fig_cols=2, ax_size=5, tight_layout=True)
#         ax[ax_c].imshow(dis_img); ax_c += 1

#     ########################################################################################################
#     ### numpy(進 pytorch 前處理)
#     dis_img_t = dis_img.astype(float).transpose((2, 0, 1))[np.newaxis, ...]  ### NCHW(13HW), dtype == float
#     bm_t = np.expand_dims(bm, 0)  ### NHWC(1HW2), dtype == float

#     ########################################################################################################
#     ### tensor
#     import torch.nn.functional as F
#     import torch
#     dis_img_t = torch.from_numpy(dis_img_t)
#     bm_t      = torch.from_numpy(bm_t)
#     rec_img_t = F.grid_sample(input=dis_img_t, grid=bm_t)

#     ########################################################################################################
#     ### numpy
#     rec_img = rec_img_t.numpy()[0].transpose(1, 2, 0).astype(np.uint8)  ### HWC, dtype == uint8
#     if(visual):
#         ax[ax_c].imshow(rec_img)
#         ax_c += 1
#         return rec_img, fig, ax, ax_c
#     else:
#         return rec_img
