from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# from tensorflow_addons.layers import InstanceNormalization
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

def instance_norm(in_x, name="instance_norm"):    
    depth = in_x.get_shape()[3]
    scale = tf.Variable(tf.random.normal(shape=[depth],mean=1.0, stddev=0.02), dtype=tf.float32)
    #print(scale)
    offset = tf.Variable(tf.zeros(shape=[depth]))
    mean, variance = tf.nn.moments(in_x, axes=[1,2], keepdims=True)
    # print("mean",mean)
    # print("variance",variance)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (in_x-mean)*inv
    return scale*normalized + offset


class InstanceNorm_kong(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(InstanceNorm_kong, self).__init__(**kwargs)

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02), dtype=tf.float32)
        self.offset = self.add_weight("offset", shape = [depth], initializer=tf.constant_initializer(0.0), dtype=tf.float32 )

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
        epsilon = tf.constant(1e-5,dtype=tf.float32)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, ks=3, s=1):
        super(ResBlock, self).__init__()
        self.ks = ks
        self.conv_1 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c1   = InstanceNorm_kong()
        self.conv_2 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c2   = InstanceNorm_kong()
    
    def call(self, input_tensor):
        p = int( (self.ks-1)/2 )
        x = tf.pad( input_tensor, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        x = tf.pad( x, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_2(x)
        x = self.in_c2(x)
        return x + input_tensor

class Discriminator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.concat = Concatenate(name = "D_concat")

        self.conv_1 = Conv2D(64  ,   kernel_size=4, strides=2, padding="same")
        self.leaky_lr1 = LeakyReLU(alpha=0.2)

        self.conv_2 = Conv2D(64*2,   kernel_size=4, strides=2, padding="same")
        self.in_c2   = InstanceNorm_kong()
        self.leaky_lr2 = LeakyReLU(alpha=0.2)

        self.conv_3 = Conv2D(64*4,   kernel_size=4, strides=2, padding="same")
        self.in_c3   = InstanceNorm_kong()
        self.leaky_lr3 = LeakyReLU(alpha=0.2)

        self.conv_4 = Conv2D(64*8,   kernel_size=4, strides=2, padding="same")
        self.in_c4   = InstanceNorm_kong()
        self.leaky_lr4 = LeakyReLU(alpha=0.2)

        self.conv_map = Conv2D(1   ,   kernel_size=4, strides=1, padding="same")

        
    def call(self, dis_img, gt_img):
        # print("dis_img",dis_img.shape)
        # print("gt_img",gt_img.shape)
        concat_img = self.concat([dis_img, gt_img])
        # print("concat_img",concat_img.shape)
        x = self.conv_1(concat_img)

        x = self.leaky_lr1(x)
        # x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_2(x)
        x = self.in_c2(x)
        x = self.leaky_lr2(x)
        # x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_3(x)
        x = self.in_c3(x)
        x = self.leaky_lr3(x)
        # x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_4(x)
        x = self.in_c4(x)
        x = self.leaky_lr4(x)
        # x = tf.nn.leaky_relu(x, alpha=0.2)

        return self.conv_map(x)

class Generator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Generator, self).__init__(**kwargs)
        self.conv1   = Conv2D(64  ,   kernel_size=7, strides=1, padding="valid")
        self.in_c1   = InstanceNorm_kong()
        self.conv2   = Conv2D(64*2,   kernel_size=3, strides=2, padding="same")
        self.in_c2   = InstanceNorm_kong()
        self.conv3   = Conv2D(64*4,   kernel_size=3, strides=2, padding="same")
        self.in_c3   = InstanceNorm_kong()

        self.resb1   = ResBlock(c_num=64*4)
        self.resb2   = ResBlock(c_num=64*4)
        self.resb3   = ResBlock(c_num=64*4)
        self.resb4   = ResBlock(c_num=64*4)
        self.resb5   = ResBlock(c_num=64*4)
        self.resb6   = ResBlock(c_num=64*4)
        self.resb7   = ResBlock(c_num=64*4)
        self.resb8   = ResBlock(c_num=64*4)
        self.resb9   = ResBlock(c_num=64*4)

        self.convT1  = Conv2DTranspose(64*2, kernel_size=3, strides=2, padding="same")
        self.in_cT1  = InstanceNorm_kong()
        self.convT2  = Conv2DTranspose(64  , kernel_size=3, strides=2, padding="same")
        self.in_cT2  = InstanceNorm_kong()
        self.convRGB = Conv2D(3  ,   kernel_size=7, strides=1, padding="valid")

    def call(self, input_tensor):
        x = tf.pad(input_tensor, [[0,0], [3,3], [3,3], [0,0]], "REFLECT")

        ### c1
        x = self.conv1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        ### c2
        x = self.conv2(x)
        x = self.in_c2(x)
        x = tf.nn.relu(x)
        ### c1
        x = self.conv3(x)
        x = self.in_c3(x)
        x = tf.nn.relu(x)

        x = self.resb1(x)
        x = self.resb2(x)
        x = self.resb3(x)
        x = self.resb4(x)
        x = self.resb5(x)
        x = self.resb6(x)
        x = self.resb7(x)
        x = self.resb8(x)
        x = self.resb9(x)

        x = self.convT1(x)
        x = self.in_cT1(x)
        x = tf.nn.relu(x)

        x = self.convT2(x)
        x = self.in_cT2(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], "REFLECT")
        x_RGB = self.convRGB(x)
        return tf.nn.tanh(x_RGB)



class MRFBlock(tf.keras.layers.Layer):
    def __init__(self, c_num):
        super(MRFBlock, self).__init__()
        self.conv_11 = Conv2D( c_num, kernel_size=1, strides=1, padding="same")
        self.in_c11  = InstanceNorm_kong()
        self.conv_12 = Conv2D( c_num, kernel_size=1, strides=1, padding="same")
        self.in_c12  = InstanceNorm_kong()

        self.conv_31 = Conv2D( c_num, kernel_size=3, strides=1, padding="same")
        self.in_c31  = InstanceNorm_kong()
        self.conv_32 = Conv2D( c_num, kernel_size=3, strides=1, padding="same")
        self.in_c32  = InstanceNorm_kong()

        self.conv_51 = Conv2D( c_num, kernel_size=5, strides=1, padding="same")
        self.in_c51  = InstanceNorm_kong()
        self.conv_52 = Conv2D( c_num, kernel_size=5, strides=1, padding="same")
        self.in_c52  = InstanceNorm_kong()

        self.conv_71 = Conv2D( c_num, kernel_size=7, strides=1, padding="same")
        self.in_c71  = InstanceNorm_kong()
        self.conv_72 = Conv2D( c_num, kernel_size=7, strides=1, padding="same")
        self.in_c72  = InstanceNorm_kong()

        self.conv_91 = Conv2D( c_num, kernel_size=9, strides=1, padding="same")
        self.in_c91  = InstanceNorm_kong()
        self.conv_92 = Conv2D( c_num, kernel_size=9, strides=1, padding="same")
        self.in_c92  = InstanceNorm_kong()

        self.concat = Concatenate(name="concat_MRF")
        
    def call(self, input_tensor):
        x1 = self.conv_11(input_tensor)
        return x1
        # x1 = self.in_c11(x1)
        # x1 = tf.nn.relu(x1)
        # x1 = self.conv_12(x1)
        # x1 = self.in_c12(x1)
        # x1 = tf.nn.relu(x1)

        # x3 = self.conv_31(input_tensor)
        # x3 = self.in_c31(x3)
        # x3 = tf.nn.relu(x3)
        # x3 = self.conv_32(x3)
        # x3 = self.in_c32(x3)
        # x3 = tf.nn.relu(x3)
        
        # x5 = self.conv_51(input_tensor)
        # x5 = self.in_c51(x5)
        # x5 = tf.nn.relu(x5)
        # x5 = self.conv_52(x5)
        # x5 = self.in_c52(x5)
        # x5 = tf.nn.relu(x5)

        # x7 = self.conv_71(input_tensor)
        # x7 = self.in_c71(x7)
        # x7 = tf.nn.relu(x7)
        # x7 = self.conv_72(x7)
        # x7 = self.in_c72(x7)
        # x7 = tf.nn.relu(x7)

        # x9 = self.conv_91(input_tensor)
        # x9 = self.in_c91(x9)
        # x9 = tf.nn.relu(x9)
        # x9 = self.conv_92(x9)
        # x9 = self.in_c92(x9)
        # x9 = tf.nn.relu(x9)

        # x_concat = self.concat([x1, x3, x5, x7, x9])
        # return x_concat




class Rect2(tf.keras.models.Model):
    def __init__(self):
        super(Rect2, self).__init__()
        self.mrfb = MRFBlock(c_num=64)
        self.generator = Generator()
        self.discriminator = Discriminator()
    def call(self, dis_img, gt_img):
        mrfb_result = self.mrfb(dis_img)
        g_rec_img = self.generator(mrfb_result)
        # g_rec_img = self.generator(dis_img)
        # print("dis_img   before D", dis_img.shape)
        # print("g_rec_img before D", g_rec_img.shape)
        # print("gt_img    before D", gt_img.shape)
        fake_score = self.discriminator(dis_img, g_rec_img)
        real_score = self.discriminator(dis_img, gt_img)
        return g_rec_img, fake_score, real_score



class MRF_Rect2(tf.keras.models.Model):
    def __init__(self):
        super(MRF_Rect2, self).__init__()
        self.mrfb = MRFBlock(c_num=64)
        self.generator = Generator()
        self.discriminator = Discriminator()

    def call(self, dis_img, gt_img):
        mrfb_result = self.mrfb(dis_img)
        g_rec_img = self.generator(mrfb_result)
        
        # g_rec_img = self.generator(dis_img)
        fake_score = self.discriminator(dis_img, g_rec_img)
        real_score = self.discriminator(dis_img, gt_img)
        return g_rec_img, fake_score, real_score



@tf.function
def mse_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float32)):
    loss = tf.reduce_mean( tf.math.square( tensor1 - tensor2 ) )
    return loss * lamb

@tf.function
def mae_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float32)):
    loss = tf.reduce_mean( tf.math.abs( tensor1 - tensor2 ) )
    return loss * lamb

@tf.function
def train_step(rect2, dis_img, gt_img, optimizer_G, optimizer_D, summary_writer, epoch ):
    with tf.GradientTape(persistent=True) as tape:
        print("here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ in train_step1")
        print("dis_img",dis_img.shape)
        print("gt_img",gt_img.shape)
        g_rec_img, fake_score, real_score = rect2(dis_img, gt_img)
        print("here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ in train_step2")

        loss_rec = mae_kong(g_rec_img, gt_img, lamb=tf.constant(40.,tf.float32))
        loss_g2d = mse_kong(fake_score, tf.ones_like(fake_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32))
        g_total_loss = loss_rec + loss_g2d
        print("here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ in train_step3")

        loss_d_fake = mse_kong( fake_score, tf.zeros_like(fake_score, dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        loss_d_real = mse_kong( real_score, tf.ones_like (real_score, dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        d_total_loss = (loss_d_real+loss_d_fake)/2
        print("here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ in train_step4")
        
    grad_D = tape.gradient(d_total_loss, rect2.discriminator.trainable_weights)
    grad_G = tape.gradient(g_total_loss, rect2.generator.    trainable_weights)
    optimizer_D.apply_gradients( zip(grad_D, rect2.discriminator.trainable_weights )  )
    optimizer_G.apply_gradients( zip(grad_G, rect2.generator.    trainable_weights )  )
    print("here~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ in train_step5")
    with summary_writer.as_default():
        tf.summary.scalar('1_loss_rec', loss_rec, step=epoch)
        tf.summary.scalar('2_loss_g2d', loss_g2d, step=epoch)
        tf.summary.scalar('3_g_total_loss', g_total_loss, step=epoch)
        tf.summary.scalar('4_loss_d_fake' , loss_d_fake,  step=epoch)
        tf.summary.scalar('5_loss_d_real' , loss_d_real,  step=epoch)
        tf.summary.scalar('6_d_total_loss', d_total_loss, step=epoch)
    

import time
import matplotlib.pyplot as plt
def generate_images( model, dis_img, gt_img,  epoch=0, result_dir="."):
    sample_start_time = time.time()
    rect2 = model(dis_img, training=True)

    plt.figure(figsize=(20,6))
    display_list = [dis_img[0], rect2[0], gt_img[0]]
    title = ['Input Image', 'rect2 Image', 'Ground Truth']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
    
        plt.axis('off')
    # plt.show()
    plt.savefig(result_dir + "/" + "epoch_%02i-result.png"%epoch)
    plt.close()
    print("sample image cost time:", time.time()-sample_start_time)


#######################################################################################################################
#######################################################################################################################
### testing 的部分 ####################################################################################################
def test_visual(test_dir_name,  data_dict,  start_index=0):
    from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec
    import matplotlib.pyplot as plt
    from util import get_dir_img
    from build_dataset_combine import Check_dir_exist_and_build
    import numpy as np 

    ### 建立放結果的資料夾
    test_plot_dir = test_dir_name + "/" + "/plot_result"
    Check_dir_exist_and_build(test_plot_dir)
    
    ### test已經做好的資料  
    g_imgs  = get_dir_img(test_dir_name) 


    col_img_num = 3
    ax_bigger = 2
    for i, (in_img, gt_img) in enumerate( zip(data_dict["test_in_db"],data_dict["test_gt_db"]) )  :
    # for i in range(data_amount):
        if(i < start_index): continue ### 可以用這個控制從哪個test開始做
        print("test_visual %06i"%i)
        
        fig, ax = plt.subplots(1,col_img_num)
        fig.set_size_inches(col_img_num*2.1 *ax_bigger, col_img_num*ax_bigger) ### 2200~2300可以放4張圖，配500的高度，所以一張圖大概550~575寬，500高，但為了好計算還是用 500寬配500高好了！

        ### 圖. unet_rec_img
        ax[0].imshow(in_img[0])
        ax[0].set_title("in_img")

        ### 圖. rect2恢復unet_rec_img的結果
        g_img = g_imgs[i,:,:,::-1]
        ax[1].imshow(g_img)
        ax[1].set_title("rect2_rec_img")

        ### 圖. gt影像
        ax[2].imshow(gt_img[0])
        ax[2].set_title("gt_img")
        plt.savefig(test_plot_dir + "/" + "index%02i-result.png"%i)
        # plt.show()
        plt.close()
    ######################################################################################################################
    ######################################################################################################################


if(__name__ == "__main__"):
    import numpy as np
    import matplotlib.pyplot as plt
    # generator = Generator()
    # img_g = np.ones( shape=(1,256,256,3), dtype=np.float32)
    # out_g = generator(img_g)
    # plt.imshow(out_g[0,...])
    # plt.show()
    # print("out_g.numpy()",out_g.numpy())

    # discriminator = Discriminator()
    # img_d1 = np.ones(shape=(1,256,256,3),dtype=np.float32)
    # img_d2 = np.ones(shape=(1,256,256,3),dtype=np.float32)
    # out_d = discriminator(img_d1, img_d2)
    # plt.imshow(out_d[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
    # plt.colorbar()
    # plt.show()
    # print("out_d.numpy()",out_d.numpy())

    rect2 = Rect2()
    dis_img = np.ones( shape=(1,496,336,3), dtype=np.float32)
    gt_img  = np.ones( shape=(1,496,336,3), dtype=np.float32)
    optimizer_G = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_D = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    summary_writer = tf.summary.create_file_writer( "temp_logs_dir" ) ### 建tensorboard，這會自動建資料夾喔！
    train_step(rect2, dis_img, gt_img, optimizer_G, optimizer_D, summary_writer, 0)
    # train_step(rect2, dis_img, gt_img, optimizer_G, optimizer_D, summary_writer, 0)


    img_resize = (494+2,336) ### dis_img(in_img的大小)的大小且要是4的倍數
    from step6_data_pipline import get_1_pure_unet_db  , \
                               get_2_pure_rect2_dataset
    access_path = "F:/Users/Lin_server/Desktop/0 db/"
    db_dir  = access_path+"datasets"
    db_name = "2_pure_rect2_page_h=384,w=256" 
    BATCH_SIZE = 1

    data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize )
    for n, (input_image, target) in enumerate( zip(data_dict["train_in_db_pre"], data_dict["train_gt_db_pre"]) ):
        g_rec_img, fake_score, real_score = rect2(input_image, target)
        train_step(rect2    , input_image, target, optimizer_G, optimizer_D, summary_writer, n)

    # print(rect2.generator(dis_img))
    print("finish")