import tensorflow as tf
from step08_a_2_Rect2 import Generator, mae_kong

@tf.function
def train_step(model_obj, dis_img, gt_img, board_obj=None):
    with tf.GradientTape(persistent=True) as tape:
        rec_img = model_obj.generator(dis_img)
        loss_rec = mae_kong(rec_img, gt_img, lamb=tf.constant(3., tf.float32))  ### 40 調回 3

    grad_G  = tape.gradient(loss_rec, model_obj.generator.trainable_weights)
    model_obj.optimizer_G.apply_gradients(zip(grad_G, model_obj.generator.trainable_weights))

    if(board_obj is not None):
        board_obj.losses["loss_rec"](loss_rec)


if __name__ == "__main__":
    import numpy as np

    dis_img = np.ones(shape=(1, 500, 332, 3), dtype=np.float32)
    gt_img = np.ones(shape=(1, 500, 332, 3), dtype=np.float32)
    gen = Generator()
    rec_img = gen(dis_img)
    train_step(gen, dis_img, gt_img)
    print("finish")
