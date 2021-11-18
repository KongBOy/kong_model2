import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose, Concatenate, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
# from tensorflow_addons.layers import InstanceNormalization
tf.keras.backend.set_floatx('float32')  ### 這步非常非常重要！用了才可以加速！

# def instance_norm(in_x, name="instance_norm"):
#     depth = in_x.get_shape()[3]
#     scale = tf.Variable(tf.random.normal(shape=[depth],mean=1.0, stddev=0.02), dtype=tf.float32)
#     #print(scale)
#     offset = tf.Variable(tf.zeros(shape=[depth]))
#     mean, variance = tf.nn.moments(in_x, axes=[1,2], keepdims=True)
#     # print("mean",mean)
#     # print("variance",variance)
#     epsilon = 1e-5
#     inv = tf.math.rsqrt(variance + epsilon)
#     normalized = (in_x-mean)*inv
#     return scale*normalized + offset

class CoordConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CoordConv, self).__init__(**kwargs)

    def call(self, input_tensor):
        height = input_tensor.shape[1]
        width  = input_tensor.shape[2]
        x = tf.range(start=0, limit=width, dtype=tf.float32)
        x = tf.reshape(x, [1, -1])
        x = tf.tile(x, [height, 1])
        x = tf.expand_dims(x, axis=-1)
        x = x / (width - 1)
        x = x * 2 - 1
        # print(x)

        y = tf.range(start=0, limit=height, dtype=tf.float32)
        y = tf.reshape(y, [-1, 1])
        y = tf.tile(y, [1, width])
        y = tf.expand_dims(y, axis=-1)
        y = y / (height - 1)
        y = y * 2 - 1

        # print(y)

        yx = tf.concat([y, x], axis=-1)
        yx = tf.expand_dims(yx, axis=0)
        # print(yx)

        return tf.concat([input_tensor, yx], axis=-1)

class InstanceNorm_kong(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNorm_kong, self).__init__(**kwargs)

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale  = self.add_weight("scale", shape=[depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02), dtype=tf.float32)
        self.offset = self.add_weight("offset", shape=[depth], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1, 2], keepdims=True)
        epsilon = tf.constant(1e-5, dtype=tf.float32)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv

        return self.scale * normalized + self.offset
        # return tf.matmul(input, self.kernel)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, use_what_IN=InstanceNorm_kong, ks=3, s=1, use_res_learning=True, coord_conv=False, **kwargs):
        super(ResBlock, self).__init__()
        self.ks = ks
        self.use_res_learning = use_res_learning
        self.coord_conv = coord_conv
        # if(self.coord_conv): self.coord_conv_res_layer1 = CoordConv()
        self.conv_1 = Conv2D(c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c1 = use_what_IN()
        # if(self.coord_conv): self.coord_conv_res_layer2 = CoordConv()
        self.conv_2 = Conv2D(c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c2 = use_what_IN()

    def call(self, input_tensor):
        p = int((self.ks - 1) / 2)
        # if(self.coord_conv): input_tensor = self.coord_conv_res_layer1(input_tensor)
        x = tf.pad(input_tensor, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        x = self.conv_1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)

        # if(self.coord_conv): x = self.coord_conv_res_layer2(x)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        x = self.conv_2(x)
        x = self.in_c2(x)
        if(self.use_res_learning): return x + input_tensor[..., :]
        else: return x

class Discriminator(tf.keras.models.Model):
    def __init__(self, D_first_concat=True, use_what_IN=InstanceNorm_kong, D_kernel_size=4, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.D_first_concat = D_first_concat
        if(self.D_first_concat):
            self.concat = Concatenate()


        self.D_kernel_size = D_kernel_size

        self.conv_1 = Conv2D(64    , kernel_size=self.D_kernel_size, strides=2, padding="same")
        self.leaky_lr1 = LeakyReLU(alpha=0.2)

        self.conv_2 = Conv2D(64 * 2, kernel_size=self.D_kernel_size, strides=2, padding="same")
        self.in_c2   = use_what_IN()
        self.leaky_lr2 = LeakyReLU(alpha=0.2)

        self.conv_3 = Conv2D(64 * 4, kernel_size=self.D_kernel_size, strides=2, padding="same")
        self.in_c3   = use_what_IN()
        self.leaky_lr3 = LeakyReLU(alpha=0.2)

        self.conv_4 = Conv2D(64 * 8, kernel_size=self.D_kernel_size, strides=2, padding="same")
        self.in_c4   = use_what_IN()
        self.leaky_lr4 = LeakyReLU(alpha=0.2)

        self.conv_map = Conv2D(1   , kernel_size=self.D_kernel_size, strides=1, padding="same")


    def call(self, dis_img, gt_img):
        # print("dis_img",dis_img.shape)
        # print("gt_img",gt_img.shape)

        if(self.D_first_concat):
            concat_img = self.concat([dis_img, gt_img])
            # print("concat_img",concat_img.shape)
            x = self.conv_1(concat_img)
        else:
            x = self.conv_1(dis_img)
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

### 應該是參考 CycleGAN 的 Generator
class Generator(tf.keras.models.Model):
    def __init__(self, first_k3=False, hid_ch=64, true_IN=True, mrfb=None, mrf_replace=False, coord_conv=False, use_res_learning=True, resb_num=9, out_tanh=True, out_ch=3, **kwargs):
        super(Generator, self).__init__(**kwargs)
        ############################################################################
        self.coord_conv = coord_conv

        ############################################################################
        ### 架構 mrfb(如果 mrfb 外面有傳mrfb進來的話)
        self.mrfb = mrfb
        self.mrf_replace = mrf_replace
        if(self.mrfb is None and self.mrf_replace is True):
            print("設定錯誤，沒有mrfb要怎麼 用mrfb取代第一層呢~")
            return
        ########################################################################################################################################################
        ### 架構本體網路囉！
        self.first_k3 = first_k3
        self.first_k = 7
        if(self.first_k3): self.first_k = 3
        ########################################################################################################################################################
        ### 還是想實驗看看 tensorflow_addon 跟自己寫的 IN 有沒有差
        self.true_IN = true_IN
        self.use_what_IN = InstanceNorm_kong  ### 原本架構使用InstanceNorm_kong，用它來當 default IN
        if(self.true_IN): self.use_what_IN = InstanceNormalization
        ########################################################################################################################################################
        ### 想實驗看看 output 是 tanh 和 sigmoid 的效果，out_tanh=False 就是用 sigmoid
        self.out_tanh = out_tanh


        if(self.coord_conv): self.coord_conv_layer1 = CoordConv()
        if(self.mrf_replace is False):  ### 如果沒有用 mrf 來取代第一層，就用普通的conv
            self.conv1   = Conv2D(filters=hid_ch  , kernel_size=self.first_k, strides=1, padding="valid")
        self.in_c1   = self.use_what_IN()

        if(self.coord_conv): self.coord_conv_layer2 = CoordConv()
        self.conv2   = Conv2D(filters=hid_ch * 2, kernel_size=3, strides=2, padding="same")
        self.in_c2   = self.use_what_IN()

        if(self.coord_conv): self.coord_conv_layer3 = CoordConv()
        self.conv3   = Conv2D(filters=hid_ch * 4, kernel_size=3, strides=2, padding="same")
        self.in_c3   = self.use_what_IN()

        self.use_res_learning = use_res_learning
        self.resb_num = resb_num
        self.resbs = []
        for go_r in range(resb_num): self.resbs.append(ResBlock(c_num=hid_ch * 4, use_what_IN=self.use_what_IN, use_res_learning=self.use_res_learning, coord_conv=coord_conv))


        if(self.coord_conv): self.coord_conv_layer4 = CoordConv()
        self.convT1  = Conv2DTranspose(filters=hid_ch * 2, kernel_size=3, strides=2, padding="same")
        self.in_cT1  = self.use_what_IN()
        if(self.coord_conv): self.coord_conv_layer5 = CoordConv()
        self.convT2  = Conv2DTranspose(filters=hid_ch  , kernel_size=3, strides=2, padding="same")
        self.in_cT2  = self.use_what_IN()
        if(self.coord_conv): self.coord_conv_layer6 = CoordConv()
        self.convRGB = Conv2D(filters=out_ch  , kernel_size=self.first_k, strides=1, padding="valid")

        self.tanh = Activation(tf.nn.tanh)

    def call(self, input_tensor, training=None):  ### 這裡的training只是為了介面統一，實際上沒用到喔，因為IN不需要指定 train/test mode
        if(self.coord_conv):
            input_tensor = self.coord_conv_layer1(input_tensor)

        # print("input_tensor.shape", input_tensor.shape)

        first_pad_size = int((self.first_k - 1) / 2)
        if(self.mrfb is not None):  ### 看有沒有用 mrf
            x = self.mrfb(input_tensor)
            if(self.mrf_replace is False):  ### 看mrf要不要取代第一層，  如果沒有用 mrf取代第一層， 就用普通的conv然後手動鏡射padding
                x = tf.pad(x , [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")
        else:  ### 沒有用mrf，第一層就用普通的conv然後手動鏡射padding
            x = tf.pad(input_tensor, [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")

        ### c1
        if(self.mrf_replace is False):
            x = self.conv1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        ### c2

        if(self.coord_conv): x = self.coord_conv_layer2(x)
        x = self.conv2(x)
        x = self.in_c2(x)
        x = tf.nn.relu(x)
        ### c1
        if(self.coord_conv): x = self.coord_conv_layer3(x)
        x = self.conv3(x)
        x = self.in_c3(x)
        x = tf.nn.relu(x)

        for go_r in range(self.resb_num):
            x = self.resbs[go_r](x)


        if(self.coord_conv): x = self.coord_conv_layer4(x)
        x = self.convT1(x)
        x = self.in_cT1(x)
        x = tf.nn.relu(x)

        if(self.coord_conv): x = self.coord_conv_layer5(x)
        x = self.convT2(x)
        x = self.in_cT2(x)
        x = tf.nn.relu(x)

        if(self.coord_conv): x = self.coord_conv_layer6(x)
        x = tf.pad(x, [[0, 0], [first_pad_size, first_pad_size], [first_pad_size, first_pad_size], [0, 0]], "REFLECT")
        x_RGB = self.convRGB(x)
        return self.tanh(x_RGB)


class MRFBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, use_what_IN=InstanceNorm_kong, use1=False, use3=False, use5=False, use7=False, use9=False, **kwargs):
        super(MRFBlock, self).__init__()
        self.use1 = use1
        self.use3 = use3
        self.use5 = use5
        self.use7 = use7
        self.use9 = use9
        self.branch_amount = 0

        if(self.use1):
            self.conv_11 = Conv2D(c_num, kernel_size=1, strides=1, padding="same")
            self.in_c11  = use_what_IN()
            self.conv_12 = Conv2D(c_num, kernel_size=1, strides=1, padding="same")
            self.in_c12  = use_what_IN()
            self.branch_amount += 1

        if(self.use3):
            self.conv_31 = Conv2D(c_num, kernel_size=3, strides=1, padding="same")
            self.in_c31  = use_what_IN()
            self.conv_32 = Conv2D(c_num, kernel_size=3, strides=1, padding="same")
            self.in_c32  = use_what_IN()
            self.branch_amount += 1

        if(self.use5):
            self.conv_51 = Conv2D(c_num, kernel_size=5, strides=1, padding="same")
            self.in_c51  = use_what_IN()
            self.conv_52 = Conv2D(c_num, kernel_size=5, strides=1, padding="same")
            self.in_c52  = use_what_IN()
            self.branch_amount += 1

        if(self.use7):
            self.conv_71 = Conv2D(c_num, kernel_size=7, strides=1, padding="same")
            self.in_c71  = use_what_IN()
            self.conv_72 = Conv2D(c_num, kernel_size=7, strides=1, padding="same")
            self.in_c72  = use_what_IN()
            self.branch_amount += 1

        if(self.use9):
            self.conv_91 = Conv2D(c_num, kernel_size=9, strides=1, padding="same")
            self.in_c91  = use_what_IN()
            self.conv_92 = Conv2D(c_num, kernel_size=9, strides=1, padding="same")
            self.in_c92  = use_what_IN()
            self.branch_amount += 1

        if  (self.branch_amount == 0):
            print("設定錯誤！use13579至少一個要True喔！")
            return
        elif(self.branch_amount > 1):
            self.concat = Concatenate()

    def call(self, input_tensor):
        concat_list = []
        if(self.use1):
            x1 = self.conv_11(input_tensor)
            x1 = self.in_c11(x1)
            x1 = tf.nn.relu(x1)
            x1 = self.conv_12(x1)
            x1 = self.in_c12(x1)
            x1 = tf.nn.relu(x1)
            concat_list.append(x1)

        if(self.use3):
            x3 = self.conv_31(input_tensor)
            x3 = self.in_c31(x3)
            x3 = tf.nn.relu(x3)
            x3 = self.conv_32(x3)
            x3 = self.in_c32(x3)
            x3 = tf.nn.relu(x3)
            concat_list.append(x3)

        if(self.use5):
            x5 = self.conv_51(input_tensor)
            x5 = self.in_c51(x5)
            x5 = tf.nn.relu(x5)
            x5 = self.conv_52(x5)
            x5 = self.in_c52(x5)
            x5 = tf.nn.relu(x5)
            concat_list.append(x5)

        if(self.use7):
            x7 = self.conv_71(input_tensor)
            x7 = self.in_c71(x7)
            x7 = tf.nn.relu(x7)
            x7 = self.conv_72(x7)
            x7 = self.in_c72(x7)
            x7 = tf.nn.relu(x7)
            concat_list.append(x7)

        if(self.use9):
            x9 = self.conv_91(input_tensor)
            x9 = self.in_c91(x9)
            x9 = tf.nn.relu(x9)
            x9 = self.conv_92(x9)
            x9 = self.in_c92(x9)
            x9 = tf.nn.relu(x9)
            concat_list.append(x9)


        # x_concat = self.concat([x1, x3, x5, x7, x9])
        if  (self.branch_amount  > 1): x_out = self.concat(concat_list)
        elif(self.branch_amount == 1): x_out = concat_list[0]
        return x_out





class Rect2(tf.keras.models.Model):
    def __init__(self, gen_obj=Generator(), dis_obj=Discriminator(), **kwargs):
        super(Rect2, self).__init__()
        self.generator     = gen_obj
        self.discriminator = dis_obj

    def call(self, dis_img, gt_img):
        g_rec_img = self.generator(dis_img)
        fake_score = self.discriminator(dis_img, g_rec_img)
        real_score = self.discriminator(dis_img, gt_img)
        return g_rec_img, fake_score, real_score

    def model(self, dis_img, gt_img):
        return tf.keras.models.Model(inputs=[dis_img, gt_img], outputs=self.call(dis_img, gt_img))


#######################################################################################################################
#######################################################################################################################
### testing 的部分 ####################################################################################################
def test_visual(test_dir_name, data_dict, start_index=0):

    from step4_apply_rec2dis_img_b_use_move_map import apply_move_to_rec
    import matplotlib.pyplot as plt
    from util import get_dir_imgs
    from build_dataset_combine import Check_dir_exist_and_build
    import numpy as np

    ### 建立放結果的資料夾
    test_plot_dir = test_dir_name + "/" + "/plot_result"
    Check_dir_exist_and_build(test_plot_dir)

    ### test已經做好的資料
    g_imgs  = get_dir_imgs(test_dir_name)


    col_img_num = 3
    ax_bigger = 2
    # for i in range(data_amount):
    for i, (in_img, gt_img) in enumerate(zip(data_dict["test_in_db"], data_dict["test_gt_db"]))  :
        if(i < start_index): continue  ### 可以用這個控制從哪個test開始做
        print("test_visual %06i" % i)

        fig, ax = plt.subplots(1, col_img_num)
        fig.set_size_inches(col_img_num * 2.1 * ax_bigger, col_img_num * ax_bigger)  ### 2200~2300可以放4張圖，配500的高度，所以一張圖大概550~575寬，500高，但為了好計算還是用 500寬配500高好了！

        ### 圖. unet_rec_img
        ax[0].imshow(in_img[0])
        ax[0].set_title("in_img")

        ### 圖. rect恢復unet_rec_img的結果
        g_img = g_imgs[i, :, :, ::-1]
        ax[1].imshow(g_img)
        ax[1].set_title("rect_rec_img")

        ### 圖. gt影像
        ax[2].imshow(gt_img[0])
        ax[2].set_title("gt_img")
        plt.savefig(test_plot_dir + "/" + "index%02i-result.png" % i)
        # plt.show()
        plt.close()
    ######################################################################################################################
    ######################################################################################################################


#######################################################################################################################################
if(__name__ == "__main__"):
    ### 直接用 假資料 嘗試 model 跑不跑得過
    import numpy as np
    import matplotlib.pyplot as plt
    generator = Generator()
    img_g = np.ones( shape=(1, 256, 256, 3), dtype=np.float32)
    out_g = generator(img_g)
    plt.imshow(out_g[0, ...])
    plt.show()
    print("out_g.numpy()", out_g.numpy().shape)

    discriminator = Discriminator()
    img_d1 = np.ones(shape=(1, 256, 256, 3), dtype=np.float32)
    img_d2 = np.ones(shape=(1, 256, 256, 3), dtype=np.float32)
    out_d = discriminator(img_d1, img_d2)
    plt.imshow(out_d[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    plt.show()
    print("out_d.numpy()", out_d.numpy().shape)

    rect = Rect2()
    dis_img = np.ones(shape=(1, 496, 336, 3), dtype=np.float32)
    gt_img  = np.ones(shape=(1, 496, 336, 3), dtype=np.float32)
    rect(dis_img, gt_img)

#######################################################################################################################################
    ### 嘗試 真的 load tf_data 進來 train 看看
    from tqdm import tqdm
    from step06_a_datas_obj import DB_C, DB_N, DB_GM
    from step06_b_data_pipline import Dataset_builder, tf_Data_builder
    from step09_d_KModel_builder import KModel_builder, MODEL_NAME
    from step09_b_loss_info_obj import Loss_info_builder

    ### 1. model_obj
    model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).use_rect2(first_k3=False, g_train_many=False)

    ### 2. db_obj 和 tf_data
    db_obj = Dataset_builder().set_basic(DB_C.type7b_h500_w332_real_os_book , DB_N.os_book_800data      , DB_GM.in_dis_gt_ord, h=500, w=332).set_dir_by_basic().set_in_gt_format_and_range(in_format="jpg", gt_format="jpg").set_detail(have_train=True, have_see=True).build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1 , train_shuffle=False).set_data_use_range(in_use_range="-1~1", gt_use_range="-1~1").set_img_resize(model_obj.model_name).build_by_db_get_method().build()

    ### 3. loss_info_obj
    GAN_mae_loss_info = Loss_info_builder().build_gan_loss().build_gan_loss_containors().build()

    ### 4. 跑起來試試看
    for n, (_, train_in_pre, _, train_gt_pre) in enumerate(tqdm(tf_data.train_db_combine)):
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_obj=GAN_mae_loss_info)
    print("finish")



#######################################################################################################################################
    ### 以前舊的東西，之後沒用到舊刪掉囉！
    # img_resize = (494 + 2, 336)  ### dis_img(in_img的大小)的大小且要是4的倍數
    # from step06_b_data_pipline import get_1_pure_unet_db  , get_2_pure_rect2_dataset

    # data_access_path = "F:/Users/Lin_server/Desktop/0 db/"
    # db_dir  = data_access_path + "datasets"
    # db_name = "2_pure_rect2_page_h=384,w=256"
    # BATCH_SIZE = 1

    # data_dict = get_2_pure_rect2_dataset(db_dir=db_dir, db_name=db_name, batch_size=BATCH_SIZE, img_resize=img_resize)
    # for n, (input_image, target) in enumerate(zip(data_dict["train_in_db_pre"], data_dict["train_gt_db_pre"])):
    #     g_rec_img, fake_score, real_score = rect2(input_image, target)
    #     train_step(rect2    , input_image, target, optimizer_G, optimizer_D, summary_writer, n)

    # print(rect2.generator(dis_img))
