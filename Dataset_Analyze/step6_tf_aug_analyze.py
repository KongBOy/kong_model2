import matplotlib.pyplot as plt
from   matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import tensorflow as tf

def analyze_tf_aug(img, Mask):
    if(len(img .shape) == 4): img  = img[0]
    if(len(Mask.shape) == 4): Mask = Mask[0]

    base_size = 3.5
    nrows = 1
    ncols = 5
    l_margin = 0.05
    r_margin = 0.95
    w_step = 1 / ncols - 0.01

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=( base_size * ncols, base_size * nrows ))
    plt.subplots_adjust(left=l_margin, bottom=0.2, right=r_margin, top=0.9, wspace=0.20, hspace=0.35)
    ax[0].imshow(img)
    ax[1].imshow(img)
    ax[2].imshow(img)
    ax[3].imshow(img)
    ax[4].imshow(img)
    axcolor = 'lightgoldenrodyellow'  ### slide bar 的背景顏色：金黃淡亮色

    place_1_x = 0.05
    place_2_x = place_1_x + w_step
    place_3_x = place_2_x + w_step
    place_4_x = place_3_x + w_step
    place_5_x = place_4_x + w_step

    place_all_y = 0.02

    place_w = 0.13
    place_h = 0.05

    bright_place   = plt.axes([place_1_x              , place_all_y, place_w    , place_h], facecolor=axcolor)
    contrast_place = plt.axes([place_2_x + 0.01       , place_all_y, place_w    , place_h], facecolor=axcolor)  ### 字串太長， +0.0x 才不會字重疊
    hue_place      = plt.axes([place_3_x              , place_all_y, place_w    , place_h], facecolor=axcolor)
    saturate_place = plt.axes([place_4_x + 0.01       , place_all_y, place_w    , place_h], facecolor=axcolor)  ### 字串太長， +0.0x 才不會字重疊
    reset_place    = plt.axes([place_5_x              , place_all_y, place_w / 2, place_h], facecolor=axcolor)
    random_place   = plt.axes([place_5_x + place_w / 2, place_all_y, place_w / 2, place_h], facecolor=axcolor)
    # plt.show()  ### 這裡show可以看一下大致上的位置， 但如果要用下面的slide 這裡的show 要註解調喔

    bright_sl   = Slider(bright_place   , 'bright'   , -1  , 1  , valinit=0, valstep=0.01)
    contrast_sl = Slider(contrast_place , 'contrast' ,  0  , 5  , valinit=1, valstep=0.01)
    hue_sl      = Slider(hue_place      , 'hue'      ,  0  , 1  , valinit=1, valstep=0.01)
    saturate_sl = Slider(saturate_place , 'saturate' , -2  , 5  , valinit=1, valstep=0.01)
    reset_btn   = Button(reset_place    , "reset")
    random_btn  = Button(random_place   , "random")

    def apply_bright   (val):
        ax[0].imshow(tf.image.adjust_brightness (img    , delta             = val))
        apply_mix()
    def apply_contrast (val):
        ax[1].imshow(tf.image.adjust_contrast   (img    , contrast_factor   = val))
        apply_mix()
    def apply_hue      (val):
        ax[2].imshow(tf.image.adjust_hue        (img    , delta             = val) * Mask + img * (1 - Mask))
        apply_mix()
    def apply_saturate (val):
        ax[3].imshow(tf.image.adjust_saturation (img    , saturation_factor = val))
        apply_mix()

    def apply_mix():
        bright_val   = bright_sl.val
        contrast_val = contrast_sl.val
        hue_val      = hue_sl.val
        saturate_val = saturate_sl.val

        mix_img = tf.image.adjust_brightness (img    , delta             = bright_val)
        mix_img = tf.image.adjust_contrast   (mix_img, contrast_factor   = contrast_val)
        mix_img = tf.image.adjust_hue        (mix_img, delta             = hue_val)
        mix_img = tf.image.adjust_saturation (mix_img, saturation_factor = saturate_val)
        ax[4].imshow(mix_img)

    def apply_reset(event):
        bright_sl.reset()
        contrast_sl.reset()
        hue_sl.reset()
        saturate_sl.reset()
        apply_mix()

    def apply_random(event):
        bright_sl   .set_val(tf.random.uniform(shape=[1], minval=-0.5, maxval=0.5, dtype=tf.float32).numpy()[0])
        contrast_sl .set_val(tf.random.uniform(shape=[1], minval= 0.5, maxval=3.0, dtype=tf.float32).numpy()[0])
        hue_sl      .set_val(tf.random.uniform(shape=[1], minval= 0.1, maxval=0.9, dtype=tf.float32).numpy()[0])
        saturate_sl .set_val(tf.random.uniform(shape=[1], minval= 0.0, maxval=4.0, dtype=tf.float32).numpy()[0])
        print("bright_sl.val   =", bright_sl.val)
        print("contrast_sl.val =", contrast_sl.val)
        print("hue_sl.val      =", hue_sl.val)
        print("saturate_sl.val =", saturate_sl.val)
        apply_mix()

    bright_sl   .on_changed(apply_bright)
    contrast_sl .on_changed(apply_contrast)
    hue_sl      .on_changed(apply_hue)
    saturate_sl .on_changed(apply_saturate)
    reset_btn   .on_clicked(apply_reset)
    random_btn  .on_clicked(apply_random)
    plt.show()
    plt.close()

if(__name__ == "__main__"):
    import cv2
    import numpy as np

    img = cv2.imread(r"F:\kong_model2\debug_data\1_1_1-pr_Page_141-PZU0001.png")
    img = img / 255.0
    uv  = np.load(r"F:\kong_model2\debug_data\1_1_1-pr_Page_141-PZU0001.npy")
    mask = uv[..., 0:1]
    analyze_tf_aug(img=img, Mask=mask)
