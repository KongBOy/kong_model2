from step08_a_2_Rect2 import Generator
from step08_a_4_Flow_UNet import generator_loss, train_step, generate_results, generate_sees_without_rec

if(__name__ == "__main__"):
    import time
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import DB_C, DB_N, DB_GM
    from step06_b_data_pipline import Dataset_builder, tf_Data_builder
    from step08_c_model_obj import MODEL_NAME, KModel_builder
    from step09_board_obj import Board_builder


    db_obj = Dataset_builder().set_basic(DB_C.type8_blender_os_book, DB_N.blender_os_hw768 , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_type(in_type="png", gt_type="knpy", see_type=None).set_detail(have_train=True, have_see=True).build()
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_rect_IN_ch64).build_flow_rect()
    tf_data = tf_Data_builder().set_basic(db_obj, 1 , train_shuffle=False).set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    board_obj = Board_builder().set_logs_dir_and_summary_writer(logs_dir="abc").build_by_model_name(model_obj.model_name).build()  ###step3 建立tensorboard，只有train 和 train_reload需要
    ###     step2 訓練
    for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        model_obj.train_step(model_obj, train_in_pre, train_gt_pre, board_obj)

    # generator = Generator(out_ch=2)  # 建G
    # in_img = np.ones(shape=(1, 768, 768, 3), dtype=np.float32)  # 建 假資料
    # gt_img = np.ones(shape=(1, 768, 768, 2), dtype=np.float32)  # 建 假資料
    # start_time = time.time()  # 看資料跑一次花多少時間
    # y = generator(in_img)
    # print(y)
    # print("cost time", time.time() - start_time)
