import tensorflow as tf
from step7_kong_model2_UNet_512to256 import Generator512to256
from step8_kong_model5_Rect2 import Discriminator
from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec_tf
class Unet_GAN(tf.keras.models.Model):
    def __init__(self,  **kwargs):
        super(Unet_GAN, self).__init__()
        self.generator = Generator512to256(out_channel=2)
        self.discriminator = Discriminator()
        self.max_train_move = 10 ### 過程中要 把move_map值弄到-1~1還原時需要
        self.min_train_move = 10 ### 過程中要 把move_map值弄到-1~1還原時需要
        self.max_db_move_x  = 10 ### 過程中要 用g_move_map_back 還原時出 g_rec_img需要
        self.max_db_move_y  = 10 ### 過程中要 用g_move_map_back 還原時出 g_rec_img需要
        
         

        
    def call(self, dis_img, dis_img_pre,  gt_img):
        move_map = self.generator(dis_img_pre)
        move_map_back = (move_map[0,:,:]+1)/2 * (self.max_train_move-self.min_train_move) + self.min_train_move ### 把 -1~1 轉回原始的值域
        unet_rec_img = apply_move_to_rec_tf(dis_img[0], move_map_back, self.max_db_move_x, self.max_db_move_y)
        return unet_rec_img
                
        # fake_score = self.discriminator(unet_rec_img, g_rec_img)
        # real_score = self.discriminator(unet_rec_img, gt_img)
        # return g_rec_img, fake_score, real_score


@tf.function
def mse_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float32)):
    loss = tf.reduce_mean( tf.math.square( tensor1 - tensor2 ) )
    return loss * lamb

@tf.function
def mae_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float32)):
    loss = tf.reduce_mean( tf.math.abs( tensor1 - tensor2 ) )
    return loss * lamb

# @tf.function
def train_step(unet_gan, dis_img, dis_img_pre, gt_img, optimizer_G, optimizer_D, summary_writer, epoch ):
    with tf.GradientTape(persistent=True) as tape:
        unet_rec_img = unet_gan(dis_img, pre_in_img, gt_img)
        print("unet_rec_img.shape", unet_rec_img.shape)
        print("gt_img.shape", gt_img.shape)
        loss_rec = mae_kong(unet_rec_img, gt_img, lamb=tf.constant(40.,tf.float32))
        # print("loss_rec", loss_rec)
        g_total_loss = loss_rec 

        
    # grad_G     = tape.gradient(g_total_loss, unet_gan.generator.    trainable_weights)
    # optimizer_G.apply_gradients( zip(grad_G, unet_gan.generator.    trainable_weights )  )
    # with summary_writer.as_default():
    #     tf.summary.scalar('1_loss_rec', loss_rec, step=epoch)
    #     tf.summary.scalar('3_g_total_loss', g_total_loss, step=epoch)

    
if(__name__=="__main__"):
    import numpy as np 
    dis_img    = np.ones(shape=(1, 492  ,336   ,3 ),dtype=np.float32)
    pre_in_img = np.ones(shape=(1, 384*2,256*2 ,3 ),dtype=np.float32)
    gt_img     = np.ones(shape=(1, 384  ,256   ,3 ),dtype=np.float32)
    
    unet_gan = Unet_GAN()
    unet_rec_img = unet_gan(dis_img, pre_in_img, gt_img)
    print("unet_rec_img.shape", unet_rec_img.shape)
    
    optimizer_G = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_D = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    summary_writer = tf.summary.create_file_writer( "step8_model_tensorboard_test" ) ### 建tensorboard，這會自動建資料夾喔！
    
    train_step(unet_gan, dis_img, pre_in_img, gt_img, optimizer_G, optimizer_D, summary_writer, epoch=0 )
    print("finish~")
    