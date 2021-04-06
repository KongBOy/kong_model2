from step08_a_1_UNet import Generator

import sys
sys.path.append("kong_util")
import tensorflow as tf
from util import method1
from build_dataset_combine import Check_dir_exist_and_build
import cv2
import numpy as np
import pdb

#######################################################################################################################################
def generator_loss(gen_output, target):
    target = tf.cast(target, tf.float32)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return l1_loss

@tf.function()
def train_step(model_obj, in_dis_img, gt_flow, board_obj):
    with tf.GradientTape() as gen_tape:
        gen_output = model_obj.generator(in_dis_img, training=True)
        gen_l1_loss  = generator_loss(gen_output, gt_flow)

    generator_gradients     = gen_tape.gradient(gen_l1_loss, model_obj.generator.trainable_variables)
    model_obj.optimizer_G.apply_gradients(zip(generator_gradients, model_obj.generator.trainable_variables))

    ### 把值放進 loss containor裡面，在外面才會去算 平均後 才畫出來喔！
    board_obj.losses["gen_l1_loss"](gen_l1_loss)

#######################################################################################################################################
# def generate_results( model, test_input, test_gt, max_train_move, min_train_move,  epoch=0, result_dir="."):
def generate_results(model_G, in_img_pre, training=False):
    flow      = model_G(in_img_pre, training=training)
    in_img_back = (in_img_pre[0].numpy() * 255).astype(np.uint8)  ### 把值從 0~1轉回0~255 且 dtype轉回np.uint8
    return flow[0], in_img_back


def generate_sees_without_rec(model_G, see_index, in_img_pre, gt_flow, epoch=0, result_obj=None, training=True, see_reset_init=True):
    flow, in_img_back = generate_results(model_G, in_img_pre, training=training)
    gt_flow = gt_flow[0]
    flow_visual = method1(flow[..., 2], flow[..., 1])[..., ::-1] * 255.
    gt_flow_visual = method1(gt_flow[..., 2], gt_flow[..., 1])[..., ::-1] * 255.


    see_dir  = result_obj.sees[see_index].see_dir  ### 每個 see 都有自己的資料夾 存 model生成的結果，先定出位置

    if(epoch == 0 or see_reset_init):  ### 第一次執行的時候，建立資料夾 和 寫一些 進去資料夾比較好看的東西
        Check_dir_exist_and_build(see_dir)   ### 建立 see資料夾
        cv2.imwrite(see_dir + "/" + "0a-in_img.jpg", in_img_back)   ### 寫一張 in圖進去，進去資料夾時比較好看，0a是為了保證自動排序會放在第一張
        cv2.imwrite(see_dir + "/" + "0b-gt_a_gt_flow.jpg", gt_flow_visual)  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
        np.save(see_dir + "/" + "0b-gt_a_gt_flow", gt_flow)  ### 寫一張 gt圖進去，進去資料夾時比較好看，0b是為了保證自動排序會放在第二張
    np.save(    see_dir + "/" + "epoch_%04i_a_flow"            % epoch, flow)      ### 我覺得不可以直接存npy，因為太大了！但最後為了省麻煩還是存了，相對就減少see的數量來讓總大小變小囉～
    cv2.imwrite(see_dir + "/" + "epoch_%04i_a_flow_visual.jpg" % epoch, flow_visual)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！
    # cv2.imwrite(see_dir + "/" + "epoch_%04i_b_in_rec_img.jpg" % epoch     , in_rec_img)  ### 把 生成影像存進相對應的資料夾，因為 tf訓練時是rgb，生成也是rgb，所以用cv2操作要轉bgr存才對！

    ### matplot_visual的部分，記得因為用 matplot 所以要 bgr轉rgb，但是因為有用matplot_visual_single_row_imgs，裡面會bgr轉rgb了，所以這裡不用轉囉！
    ### 這部分要記得做！在 train_step3 的 self.result_obj.Draw_loss_during_train(epoch, self.epochs) 才有畫布可以畫loss！
    ### 目前覺得好像也不大會去看matplot_visual，所以就先把這註解掉了
    # result_obj.sees[see_index].save_as_matplot_visual_during_train(epoch, bgr2rgb=True)


#######################################################################################################################################
if(__name__ == "__main__"):
    import time
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import DB_C, DB_N, DB_GM
    from step06_b_data_pipline import Dataset_builder, tf_Data_builder
    from step08_c_model_obj import MODEL_NAME, KModel_builder
    from step09_board_obj import Board_builder

    generator = Generator(out_ch=2)  # 建G

    db_obj = Dataset_builder().set_basic(DB_C.type8_blender_os_book                      , DB_N.blender_os_hw768      , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_type(in_type="png", gt_type="knpy", see_type=None).set_detail(have_train=True, have_see=True).build()
    model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_flow_unet()
    tf_data = tf_Data_builder().set_basic(db_obj, 1 , train_shuffle=False).set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    board_obj = Board_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_by_model_name(model_obj.model_name).build()  ###step3 建立tensorboard，只有train 和 train_reload需要
    ###     step2 訓練
    for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        model_obj.train_step(model_obj, train_in_pre, train_gt_pre, board_obj)

    # in_img = np.ones(shape=(1, 768, 768, 3), dtype=np.float32)  # 建 假資料
    # gt_img = np.ones(shape=(1, 768, 768, 2), dtype=np.float32)  # 建 假資料
    # start_time = time.time()  # 看資料跑一次花多少時間
    # y = generator(in_img)
    # print(y)
    # print("cost time", time.time() - start_time)
