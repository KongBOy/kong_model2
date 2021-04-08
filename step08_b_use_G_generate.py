import sys
sys.path.append("kong_util")

import matplotlib.pyplot as plt
import cv2
from build_dataset_combine import Check_dir_exist_and_build, Save_as_jpg, method1
from util import matplot_visual_single_row_imgs
import numpy as np


### 用 網路 生成 影像
def generate_img_results(model_G, in_img_pre, gt_use_range):
    rect       = model_G(in_img_pre, training=True)  ### 把影像丟進去model生成還原影像
    # print("rect_back before max, min:", rect_back.numpy().max(), rect_back.numpy().min())  ### 測試 拉range 有沒有拉對
    if  (gt_use_range == "-1~1"): rect_back  = ((rect[0].numpy() + 1) * 125).astype(np.uint8)   ### 把值從 -1~1轉回0~255 且 dtype轉回np.uint8
    elif(gt_use_range == "0~1"):  rect_back  = ( rect[0].numpy() * 255).astype(np.uint8)   ### 把值從 -1~1轉回0~255 且 dtype轉回np.uint8
    # print("rect_back after max, min:", rect_back.numpy().max(), rect_back.numpy().min())  ### 測試 拉range 有沒有拉對
    return rect_back  ### 注意訓練model時是用tf來讀img，為rgb的方式訓練，所以生成的是rgb的圖喔！


### 這是一張一張進來的，沒有辦法跟 Result 裡面的 see 生成法合併，要的話就是把這裡matplot部分去除，用result裡的see生成matplot圖囉！
def generate_img_sees(model_G, see_index, in_img, in_img_pre, gt_img, epoch=0, result_obj=None, see_reset_init=False):
    '''
    如果之後有需要 in/gt_use_range，可以從result_obj裡面拿喔，就用 result_obj.in/gt_use_range 即可
    '''
    rect_back = generate_img_results(model_G, in_img_pre, result_obj.gt_use_range)

    see_dir  = result_obj.sees[see_index].see_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    plot_dir = see_dir + "/" + "matplot_visual"    ### 每個 see資料夾 內都有一個matplot_visual 存 in_img, rect, gt_img 併起來好看的結果

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_dir)   ### 建立 see資料夾
        Check_dir_exist_and_build(plot_dir)  ### 建立 see資料夾/matplot_visual資料夾
        cv2.imwrite(see_dir + "/" + "0a-in_img.jpg", in_img[0].numpy())   ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_dir + "/" + "0b-gt_img.jpg", gt_img[0].numpy())  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(see_dir + "/" + "epoch_%04i.jpg" % epoch, rect_back[:, :, ::-1])  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！

    ### matplot_visual的部分，記得因為用 matplot 所以要 bgr轉rgb，但是因為有用matplot_visual_single_row_imgs，裡面會bgr轉rgb了，所以這裡不用轉囉！
    ### 這部分要記得做！在 train_step3 的 self.result_obj.Draw_loss_during_train(epoch, self.epochs) 才有畫布可以畫loss！
    result_obj.sees[see_index].save_as_matplot_visual_during_train(epoch)

    # imgs = [in_img, rect_back, gt_img]  ### 把 in_img, rect_back, gt_img 包成list
    # titles = ['Input Image', 'rect Image', 'Ground Truth']  ### 設定 title要顯示的字
    # matplot_visual_single_row_imgs(img_titles=titles, imgs=imgs, fig_title="epoch_%04i"%epoch, dst_dir=plot_dir ,file_name="epoch=%04i"%epoch, bgr2rgb=False)
    # Save_as_jpg(plot_dir, plot_dir,delete_ord_file=True)   ### matplot圖存完是png，改存成jpg省空間


#######################################################################################################################################
def generate_flow_results(model_G, in_img_pre, gt_use_range, training=False):
    flow      = model_G(in_img_pre, training=training)
    # print("flow before max, min:", flow.numpy().max(), flow.numpy().min())  ### 測試 拉range 有沒有拉對
    if(gt_use_range == "-1~1"): flow = (flow + 1) / 2
    # print("flow after max, min:", flow.numpy().max(), flow.numpy().min())  ### 測試 拉range 有沒有拉對
    return flow[0]


def generate_flow_sees_without_rec(model_G, see_index, in_img, in_img_pre, gt_flow, epoch=0, result_obj=None, training=True, see_reset_init=True):
    '''
    如果有需要 in/gt_use_range，可以從result_obj裡面拿喔，就用 result_obj.in/gt_use_range 即可
    '''
    flow = generate_flow_results(model_G, in_img_pre, result_obj.gt_use_range, training=training)
    gt_flow = gt_flow[0]
    flow_visual = method1(flow[..., 2], flow[..., 1])[..., ::-1] * 255.
    gt_flow_visual = method1(gt_flow[..., 2], gt_flow[..., 1])[..., ::-1] * 255.


    see_dir  = result_obj.sees[see_index].see_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_dir)   ### 建立 see資料夾
        cv2.imwrite(see_dir + "/" + "0a-in_img.jpg", in_img[0].numpy())   ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_dir + "/" + "0b-gt_a_gt_flow.jpg", gt_flow_visual)  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save(see_dir + "/" + "0b-gt_a_gt_flow", gt_flow)  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    np.save(    see_dir + "/" + "epoch_%04i_a_flow"            % epoch, flow)      ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_dir + "/" + "epoch_%04i_a_flow_visual.jpg" % epoch, flow_visual)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！
    # cv2.imwrite(see_dir + "/" + "epoch_%04i_b_in_rec_img.jpg" % epoch     , in_rec_img)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！

    ### matplot_visual的部分，記得因為用 matplot 所以要 bgr轉rgb，但是因為有用matplot_visual_single_row_imgs，裡面會bgr轉rgb了，所以這裡不用轉囉！
    ### 這部分要記得做！在 train_step3 的 self.result_obj.Draw_loss_during_train(epoch, self.epochs) 才有畫布可以畫loss！
    ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了
    # result_obj.sees[see_index].save_as_matplot_visual_during_train(epoch, bgr2rgb=True)
