import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, BatchNormalization, Concatenate
from util import method2
import matplotlib.pyplot as plt 
import time


### 所有 pytorch BN 裡面有兩個參數的設定不確定～： affine=True, track_running_stats=True，目前思考覺得改道tf2全拿掉也可以
### 目前 總共用7層，所以size縮小 2**7 ，也就是 1/128 這樣子！例如256*256*3丟進去，最中間的feature map長寬2*2*512喔！
class Generator(tf.keras.models.Model):
    def __init__(self,out_channel=3, **kwargs):
        super(Generator,self).__init__(**kwargs)
        self.conv1 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv1") #,bias=False) ### in_channel:3

        self.lrelu2 = LeakyReLU(alpha=0.2,name="lrelu2")
        self.conv2  = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv2") #,bias=False) ### in_channel:64
        self.bn2    = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn2") ### b_in_channel:128

        self.lrelu3 = LeakyReLU(alpha=0.2,name="lrelu3")
        self.conv3  = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv3") #,bias=False) ### in_channel:128
        self.bn3    = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn3") ### b_in_channel:256

        self.lrelu4 = LeakyReLU(alpha=0.2,name="lrelu4")
        self.conv4  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv4") #,bias=False) ### in_channel:256
        self.bn4    = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn4") ### b_in_channel:512

        self.lrelu5 = LeakyReLU(alpha=0.2,name="lrelu5")
        self.conv5  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv5") #,bias=False) ### in_channel:512
        self.bn5    = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn5") ### b_in_channel:512

        self.lrelu6 = LeakyReLU(alpha=0.2,name="lrelu6")
        self.conv6  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv6") #,bias=False) ### in_channel:512
        self.bn6    = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn6") ### b_in_channel:512

        ###################
        # 最底層
        self.lrelu7 = LeakyReLU(alpha=0.2,name="lrelu7")
        self.conv7  = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv7") #,bias=False) ### in_channel:512

        self.relu7t = ReLU(name="relu7t")
        self.conv7t = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv7t") #,bias=False) ### in_channel:512
        self.bn7t   = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn7t") ### b_in_channel:512
        self.concat7 = Concatenate(name="concat7")
        ###################

        self.relu6t = ReLU(name="relu6t")
        self.conv6t = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv6t") #,bias=False) ### in_channel:1024
        self.bn6t   = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn6t") ### b_in_channel:512
        self.concat6 = Concatenate(name="concat6")

        self.relu5t = ReLU(name="relu5t")
        self.conv5t = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv5t") #,bias=False) ### in_channel:1024
        self.bn5t   = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn5t") ### b_in_channel:512
        self.concat5 = Concatenate(name="concat5")

        self.relu4t = ReLU(name="relu4t")
        self.conv4t = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv4t") #,bias=False) ### in_channel:1024
        self.bn4t   = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn4t") ### b_in_channel:256
        self.concat4 = Concatenate(name="concat4")

        self.relu3t = ReLU(name="relu3t")
        self.conv3t = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv3t") #,bias=False) ### in_channel:512
        self.bn3t   = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn3t") ### b_in_channel:128
        self.concat3 = Concatenate(name="concat3")


        self.relu2t = ReLU(name="relu2t")
        self.conv2t = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv2t") #,bias=False) ### in_channel:256
        self.bn2t   = BatchNormalization(epsilon=1e-05, momentum=0.1,name="bn2t") ### b_in_channel:64
        self.concat2 = Concatenate(name="concat2")


        self.relu1t = ReLU(name="relu1t")
        self.conv1t = Conv2DTranspose(out_channel, kernel_size=(4, 4), strides=(2, 2), padding="same",name="conv1t") ### in_channel:128
        # (4): Tanh()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)

        skip2 = x
        x = self.lrelu2(skip2)
        x = self.conv2(x)
        x = self.bn2(x)
        
        skip3 = x
        x = self.lrelu3(skip3)
        x = self.conv3(x)
        x = self.bn3(x)

        skip4 = x
        x = self.lrelu4(skip4)
        x = self.conv4(x)
        x = self.bn4(x)

        skip5 = x
        x = self.lrelu5(skip5)
        x = self.conv5(x)
        x = self.bn5(x)

        skip6 = x
        x = self.lrelu6(skip6)
        x = self.conv6(x)
        x = self.bn6(x)
        ###############################
        skip7 = x
        x = self.lrelu7(skip7)
        x = self.conv7(x)

        x = self.relu7t(x)
        x = self.conv7t(x)
        x = self.bn7t(x)
        # x = self.concat7([skip7,x])
        x = self.concat7([x,skip7])
        ###############################
        x = self.relu6t(x)
        x = self.conv6t(x)
        x = self.bn6t(x)
        # x = self.concat6([skip6,x])
        x = self.concat6([x,skip6])

        x = self.relu5t(x)
        x = self.conv5t(x)
        x = self.bn5t(x)
        # x = self.concat5([skip5,x])
        x = self.concat5([x,skip5])


        x = self.relu4t(x)
        x = self.conv4t(x)
        x = self.bn4t(x)
        # x = self.concat4([skip4,x])
        x = self.concat4([x,skip4])


        x = self.relu3t(x)
        x = self.conv3t(x)
        x = self.bn3t(x)
        # x = self.concat3([skip3,x])
        x = self.concat3([x,skip3])


        x = self.relu2t(x)
        x = self.conv2t(x)
        x = self.bn2t(x)
        # x = self.concat2([skip2,x])
        x = self.concat2([x,skip2])

        x = self.relu1t(x)
        x = self.conv1t(x)
        return tf.nn.tanh(x)
        
    
    def model(self, x):
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x) )
        
def generator_loss(gen_output, target):
    target = tf.cast(target,tf.float32)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return l1_loss

#######################################################################################################################################
@tf.function()
def train_step(generator, generator_optimizer, summary_writer, input_image, target, epoch):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        gen_l1_loss  = generator_loss( gen_output, target)

    generator_gradients     = gen_tape.gradient(gen_l1_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)




#######################################################################################################################################
def generate_images( model, test_input, test_label, max_train_move, min_train_move,  epoch=0, result_dir="."):
    sample_start_time = time.time()
    prediction = model(test_input, training=True)

    plt.figure(figsize=(20,6))
    display_list = [test_input[0], test_label[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        if(i==0):
            plt.imshow(display_list[i] * 0.5 + 0.5)
        else:
            back = (display_list[i]+1)/2 * (max_train_move-min_train_move) + min_train_move
            back_bgr = method2(back[...,0], back[...,1],1)
            plt.imshow(back_bgr)
        plt.axis('off')
    # plt.show()
    plt.savefig(result_dir + "/" + "epoch_%02i-result.png"%epoch)
    plt.close()
    print("sample image cost time:", time.time()-sample_start_time)


#######################################################################################################################################
if(__name__=="__main__"):
    import time
    import numpy as np 

    generator = Generator()  ### 建G
    img = np.ones(shape=(1,256,256,3), dtype= np.float32) ### 建 假資料
    start_time = time.time() ### 看資料跑一次花多少時間
    y= generator(img)
    print(y)
    print("cost time", time.time()- start_time)
