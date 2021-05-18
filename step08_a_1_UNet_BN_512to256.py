import sys
sys.path.append("kong_util")
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, BatchNormalization, Concatenate
from util import method2
import matplotlib.pyplot as plt
import time
from build_dataset_combine import Check_dir_exist_and_build, Save_as_jpg
from matplot_fig_ax_util import Matplot_single_row_imgs  # matplot_visual_single_row_imgs
import cv2
from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec2


### 參考 DewarpNet 的 train_wc 用的 UNet
### 所有 pytorch BN 裡面有兩個參數的設定不確定～： affine=True, track_running_stats=True，目前思考覺得改道tf2全拿掉也可以
### 目前 總共用7層，所以size縮小 2**7 ，也就是 1/128 這樣子！例如256*256*3丟進去，最中間的feature map長寬2*2*512喔！
class Generator512to256(tf.keras.models.Model):
    def __init__(self, out_ch=3, **kwargs):
        super(Generator512to256, self).__init__(**kwargs)
        self.conv1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv1")  #,bias=False) ### in_channel:3

        self.lrelu2 = LeakyReLU(alpha=0.2, name="lrelu2")
        self.conv2  = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2")  #,bias=False) ### in_channel:64
        self.bn2    = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn2")  ### b_in_channel:128

        self.lrelu3 = LeakyReLU(alpha=0.2, name="lrelu3")
        self.conv3  = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv3")  #,bias=False) ### in_channel:128
        self.bn3    = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn3")  ### b_in_channel:256

        self.lrelu4 = LeakyReLU(alpha=0.2, name="lrelu4")
        self.conv4  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv4")  #,bias=False) ### in_channel:256
        self.bn4    = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn4")  ### b_in_channel:512

        self.lrelu5 = LeakyReLU(alpha=0.2, name="lrelu5")
        self.conv5  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv5")  #,bias=False) ### in_channel:512
        self.bn5    = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn5")  ### b_in_channel:512

        self.lrelu6 = LeakyReLU(alpha=0.2, name="lrelu6")
        self.conv6  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv6")  #,bias=False) ### in_channel:512
        self.bn6    = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn6")  ### b_in_channel:512

        ###################
        # 最底層
        self.lrelu7 = LeakyReLU(alpha=0.2, name="lrelu7")
        self.conv7  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv7")  #,bias=False) ### in_channel:512

        self.relu7t = ReLU(name="relu7t")
        self.conv7t = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv7t")  #,bias=False) ### in_channel:512
        self.bn7t   = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn7t")  ### b_in_channel:512
        self.concat7 = Concatenate(name="concat7")
        ###################

        self.relu6t = ReLU(name="relu6t")
        self.conv6t = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv6t")  #,bias=False) ### in_channel:1024
        self.bn6t   = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn6t")  ### b_in_channel:512
        self.concat6 = Concatenate(name="concat6")

        self.relu5t = ReLU(name="relu5t")
        self.conv5t = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv5t")  #,bias=False) ### in_channel:1024
        self.bn5t   = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn5t")  ### b_in_channel:512
        self.concat5 = Concatenate(name="concat5")

        self.relu4t = ReLU(name="relu4t")
        self.conv4t = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv4t")  # ,bias=False) ### in_channel:1024
        self.bn4t = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn4t")  # b_in_channel:256
        self.concat4 = Concatenate(name="concat4")

        self.relu3t = ReLU(name="relu3t")
        self.conv3t = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv3t")  # ,bias=False) ### in_channel:512
        self.bn3t = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn3t")  # b_in_channel:128
        self.concat3 = Concatenate(name="concat3")


        self.relu2t = ReLU(name="relu2t")
        self.conv2t = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="same", name="conv2t")  #,bias=False) ### in_channel:256
        self.bn2t   = BatchNormalization(epsilon=1e-05, momentum=0.1, name="bn2t")  ### b_in_channel:64
        self.concat2 = Concatenate(name="concat2")


        self.relu1t = ReLU(name="relu1t")
        self.conv_out = Conv2D(out_ch, kernel_size=(4, 4), strides=(1, 1), padding="same", name="conv_out")  ### in_channel:128
        # (4): Tanh()

    def call(self, input_tensor, training=True):
        x = self.conv1(input_tensor)

        skip2 = x
        x = self.lrelu2(skip2)
        x = self.conv2(x)
        x = self.bn2(x, training)

        skip3 = x
        x = self.lrelu3(skip3)
        x = self.conv3(x)
        x = self.bn3(x, training)

        skip4 = x
        x = self.lrelu4(skip4)
        x = self.conv4(x)
        x = self.bn4(x, training)

        skip5 = x
        x = self.lrelu5(skip5)
        x = self.conv5(x)
        x = self.bn5(x, training)

        skip6 = x
        x = self.lrelu6(skip6)
        x = self.conv6(x)
        x = self.bn6(x, training)
        ###############################
        skip7 = x
        x = self.lrelu7(skip7)
        x = self.conv7(x)

        x = self.relu7t(x)
        x = self.conv7t(x)
        x = self.bn7t(x, training)
        # x = self.concat7([skip7,x])
        x = self.concat7([x, skip7])
        ###############################
        x = self.relu6t(x)
        x = self.conv6t(x)
        x = self.bn6t(x, training)
        # x = self.concat6([skip6,x])
        x = self.concat6([x, skip6])

        x = self.relu5t(x)
        x = self.conv5t(x)
        x = self.bn5t(x, training)
        # x = self.concat5([skip5,x])
        x = self.concat5([x, skip5])


        x = self.relu4t(x)
        x = self.conv4t(x)
        x = self.bn4t(x, training)
        # x = self.concat4([skip4,x])
        x = self.concat4([x, skip4])


        x = self.relu3t(x)
        x = self.conv3t(x)
        x = self.bn3t(x, training)
        # x = self.concat3([skip3,x])
        x = self.concat3([x, skip3])


        x = self.relu2t(x)
        x = self.conv2t(x)
        x = self.bn2t(x, training)
        # x = self.concat2([skip2,x])
        x = self.concat2([x, skip2])

        x = self.relu1t(x)
        x = self.conv_out(x)
        return tf.nn.tanh(x)


    def model(self, x):  ### 看summary用的
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))

#######################################################################################################################################
def generator_loss(gen_output, target):
    target = tf.cast(target, tf.float32)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return l1_loss

@tf.function()
def train_step(model_obj, in_dis_img, gt_move_map, board_obj):
    with tf.GradientTape() as gen_tape:
        gen_output = model_obj.generator(in_dis_img, training=True)
        gen_l1_loss  = generator_loss(gen_output, gt_move_map)

    generator_gradients     = gen_tape.gradient(gen_l1_loss, model_obj.generator.trainable_variables)
    model_obj.generator_optimizer.apply_gradients(zip(generator_gradients, model_obj.generator.trainable_variables))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    board_obj.losses["gen_l1_loss"](gen_l1_loss)

#######################################################################################################################################
# def generate_results( model, test_input, test_gt, max_train_move, min_train_move,  epoch=0, result_write_dir="."):
def generate_results(model_G, in_img_pre, max_train_move, min_train_move):
    move_map      = model_G(in_img_pre, training=True)
    move_map_back = ((move_map - min_train_move) / (max_train_move - min_train_move)) * 2 - 1
    in_img_back = ((in_img_pre[0].numpy() + 1) * 125).astype(np.uint8)  ### 把值從 -1~1轉回0~255 且 dtype轉回np.uint8
    return move_map_back, in_img_back


def generate_sees(model_G, see_index, in_img_pre, gt_move_map, max_train_move, min_train_move, max_db_move_x, max_db_move_y, epoch=0, result_obj=None, see_reset_init=False):
    move_map_back, in_img_back = generate_results(model_G, in_img_pre, max_train_move, min_train_move)
    ### 我們不要存move_map_back.npy，存move_map_visual.jpg
    move_map_back_visual = method2(move_map_back[..., 0], move_map_back[..., 1], 1)
    gt_move_map_visual = method2(gt_move_map[..., 0], gt_move_map[..., 1], 1)
    in_rec_img = apply_move_to_rec2(in_img_back, move_map_back, max_train_move, min_train_move)
    gt_rec_img = apply_move_to_rec2(in_img_back, gt_move_map, max_train_move, min_train_move)

    see_write_dir  = result_obj.sees[see_index].see_write_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置
    plot_dir = see_write_dir + "/" + "matplot_visual"        ### 每個 see資料夾 內都有一個matplot_visual 存 in_img, rect, gt_img 併起來好看的結果

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_write_dir)   ### 建立 see資料夾
        Check_dir_exist_and_build(plot_dir)  ### 建立 see資料夾/matplot_visual資料夾
        cv2.imwrite(see_write_dir + "/" + "0a-in_img.jpg", in_img_back)   ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_a_gt_move_map.jpg", gt_move_map_visual)  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        cv2.imwrite(see_write_dir + "/" + "0b-gt_b_gt_rec_img.jpg", gt_rec_img)  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_a_move_map_visual.jpg" % epoch, move_map_back_visual)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！
    cv2.imwrite(see_write_dir + "/" + "epoch_%04i_b_in_rec_img.jpg" % epoch     , in_rec_img)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！

    imgs = [in_img_back, move_map_back_visual, gt_move_map_visual, in_rec_img, gt_rec_img]  ### 把 in_img_back, rect_back, gt_img 包成list
    titles = ['Input Image', 'gen_move_map', 'gt_move_map', 'gen_rec_img', 'gt_rec_img', 'Ground Truth']  ### 設定 title要顯示的字

    ### 改完但還沒有測試喔~~
    single_row_imgs = Matplot_single_row_imgs(imgs=imgs, img_titles=titles, fig_title="epoch_%04i" % epoch, bgr2rgb=False, add_loss=False)
    single_row_imgs.Save_fig(dst_dir=plot_dir, epoch=epoch, epoch_name="epoch")
    # matplot_visual_single_row_imgs(img_titles=titles, imgs=imgs, fig_title="epoch_%04i" % epoch, dst_dir=plot_dir, file_name="epoch=%04i" % epoch, bgr2rgb=False)
    # Save_as_jpg(plot_dir, plot_dir, delete_ord_file=True)   ### matplot圖存完是png，改存成jpg省空間

    # result_obj.sees[see_index].save_as_matplot_visual_during_train(epoch)

    # plt.figure(figsize=(20,6))
    # display_list = [test_input[0], test_gt[0], prediction[0]]
    # title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.title(title[i])
    #     # getting the pixel values between [0, 1] to plot it.
    #     if(i==0):
    #         plt.imshow(display_list[i] * 0.5 + 0.5)
    #     else:
    #         back = (display_list[i]+1)/2 * (max_train_move-min_train_move) + min_train_move
    #         back_bgr = method2(back[...,0], back[...,1],1)
    #         plt.imshow(back_bgr)
    #     plt.axis('off')
    # # plt.show()
    # plt.savefig(result_write_dir + "/" + "epoch_%04i-result.png"%epoch)
    # plt.close()
    # print("sample image cost time:", time.time()-sample_start_time)




### test_g_in_db 還是要，因為要給generator生成還是需要他這樣子～
### db_dir 和 db_name 主要是為了拿 mac_db_move_xy 和 maxmin_train_move
def test_visual(test_dir_name, model_dict, data_dict, start_index=0):
    from build_dataset_combine import Check_dir_exist_and_build
    import numpy as np
    from util import  get_dir_move, get_dir_certain_img


    ### 建立放結果的資料夾
    test_plot_dir = test_dir_name + "/" + "plot_result"
    Check_dir_exist_and_build(test_plot_dir)

    ### test已經做好的資料
    g_move_maps = get_dir_move       (test_dir_name)
    g_rec_imgs  = get_dir_certain_img(test_dir_name, "g_rec_img.bmp").astype(np.uint8)


    ### 用來給 apply_rec_move的max_db_move_x/y
    max_db_move_x = model_dict["max_db_move_x"]
    max_db_move_y = model_dict["max_db_move_y"]


    col_img_num = 6
    ax_bigger = 2
    if  ("test_gt_db" in data_dict.keys() and data_dict["gt_type"] == "move_map"): col_img_num = 5
    elif("test_gt_db" in data_dict.keys() and data_dict["gt_type"] == "img")     : col_img_num = 4
    elif("test_gt_db" not in data_dict.keys())                                  : col_img_num = 3

    print("col_img_num", col_img_num)

    # for i, (test_input, test_gt) in enumerate( zip( data_dict["test_in_db"], data_dict["test_gt_db"] ) ):
    for i, test_input in enumerate(data_dict["test_in_db"]) :
        if("test_gt_db" in data_dict.keys()):
            test_gt = data_dict["test_gt_db"][i]

        if(i < start_index): continue  ### 可以用這個控制從哪個test開始做
        print("test_visual %06i" % i)

        fig, ax = plt.subplots(1, col_img_num)
        fig.set_size_inches(col_img_num * 2.1 * ax_bigger, col_img_num * ax_bigger)  ### 2200~2300可以放4張圖，配500的高度，所以一張圖大概550~575寬，500高，但為了好計算還是用 500寬配500高好了！
        plot_i = 0

        ### 圖. test_input
        test_input = test_input[0].numpy()  ### 這是沒有resize過的！recover是要用這個來做喔！不是用test_in_db_pre resize過的test_input來做！[0]是因為建tf.dataset時有用batch
        # ax[plot_i].imshow(test_input.astype(np.uint8))
        ax[plot_i].imshow(test_input)
        ax[plot_i].set_title("distorted_img")
        plot_i += 1

        ### 圖. G predict的 move_map(test時已經做過直接拿來用)
        g_move_map = g_move_maps[i]
        g_move_bgr = method2(g_move_map[..., 0], g_move_map[..., 1], 1)  ### method2回傳的是 uint8的array喔！
        ax[plot_i].imshow(g_move_bgr)
        ax[plot_i].set_title("predict_dis_flow")
        plot_i += 1

        ### 圖. test_gt(如果有 test_gt_db的話)
        ### 我的 data_dict["test_gt_db"] 本身就是 numpy了！當初忘記把它變tensorflow dataset，但誤打誤撞也剛剛好發現不需要變喔！
        if("test_gt_db" in data_dict.keys() and data_dict["gt_type"] == "move_map"):
            test_gt_bgr = method2(test_gt[:, :, 0], test_gt[:, :, 1])
            ax[plot_i].imshow(test_gt_bgr.astype(np.uint8))
            ax[plot_i].set_title("gt_rec")
            plot_i += 1

        ### 圖. G 的 rec_img(test時已經做過直接拿來用)
        g_rec_img = g_rec_imgs[i, :, :, ::-1]
        ax[plot_i].imshow(g_rec_img)
        ax[plot_i].set_title("predict_rec")
        plot_i += 1


        ### 圖. test_input 配 test_gt(如果gt_type為move_map的話) 來做 rec～
        if("test_gt_db" in data_dict.keys() and data_dict["gt_type"] == "move_map"):
            gt_rec_img = apply_move_to_rec2(test_input, test_gt, max_db_move_x, max_db_move_y)
            ax[plot_i].imshow(gt_rec_img.astype(np.uint8))
            ax[plot_i].set_title("gt_rec")
            plot_i += 1

        ### 原圖
        if("test_gt_db" in data_dict.keys() and data_dict["gt_type"] == "img"):
            ax[plot_i].imshow(test_gt[0])
            ax[plot_i].set_title("ord_img")
            # plot_i += 1 ### 最後一張圖了，不用再加囉

        # plt.show()
        plt.savefig(test_plot_dir + "/" + "index%02i-result.png" % i)
        plt.close()


#######################################################################################################################################
if(__name__ == "__main__"):
    import numpy as np

    generator = Generator512to256()  # 建G
    in_img = np.ones(shape=(1, 384, 384, 3), dtype=np.float32)  # 建 假資料
    gt_img = np.ones(shape=(1, 192, 192, 3), dtype=np.float32)  # 建 假資料
    start_time = time.time()  # 看資料跑一次花多少時間
    y = generator(in_img)
    print(y)
    print("cost time", time.time() - start_time)
