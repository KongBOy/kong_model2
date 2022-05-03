import tensorflow as tf
from step07_a_unet_component import Conv_Blocks, UNet_down, UNet_up, Use_what_acti
from tensorflow.keras.layers import Activation, Concatenate, Conv2D
from tensorflow_addons.layers import InstanceNormalization

### 參考 DewarpNet 的 train_wc 用的 UNet
### 所有 pytorch BN 裡面有兩個參數的設定不確定～： affine=True, track_running_stats=True，目前思考覺得改道tf2全拿掉也可以
### 目前 總共用7層，所以size縮小 2**7 ，也就是 1/128 這樣子！例如256*256*3丟進去，最中間的feature map長寬2*2*512喔！
class Generator(tf.keras.models.Model):
    def __init__(self, hid_ch=64, depth_level=7, out_ch=3, no_concat_layer=0,
                 kernel_size=4, strides=2, padding="same", norm="in",
                 d_acti="lrelu", u_acti="relu", unet_acti="tanh",
                 use_bias=True,
                 conv_block_num=0,
                 skip_op=None, skip_merge_op="concat",
                 ch_upper_bound=512,
                 coord_conv=False,

                 d_amount=1,
                 bottle_divide = False,
                 out_conv_block = False,  ### 2020/02/15重新審視一次 架構， 覺得 conv_transpose完應該要再接 conv_block 會更好吧~~
                 concat_before_down = False,  ### 2020/02/16重新審視一次 架構， 覺得 應該用做完conv_blocks來skip到decoder比較對吧~~
                 #  out_tanh=True,
                 #  skip_use_add=False, skip_use_cSE=False, skip_use_sSE=False, skip_use_scSE=False, skip_use_cnn=False, skip_cnn_k=3, skip_use_Acti=None,
                 **tf_kwargs):
        self.debug = False  ### 手動設定吧
        '''
        d_acti: lrelu/ relu
        u_acti: relu/ lrelu
        unet_acti: tanh/ sigmoid
        skip_op: cse/ sse/ scse
        '''
        if(depth_level < 2):
            print("UNet 不可以小於 2層， 因為如果只有 1層 沒辦法做 skip connection")
            exit()
        super(Generator, self).__init__(**tf_kwargs)
        self.kernel_size = kernel_size
        self.depth_level = depth_level
        self.hid_ch = hid_ch
        self.padding = padding
        self.norm = norm
        self.no_concat_layer = no_concat_layer
        self.unet_out_ch = out_ch
        self.unet_acti = unet_acti
        self.use_bias  = use_bias
        self.conv_block_num = conv_block_num
        self.d_amount  = d_amount
        self.bottle_divide = bottle_divide
        self.ch_upper_bound = ch_upper_bound
        self.coord_conv = coord_conv
        self.u_acti = u_acti

        self.out_conv_block     = out_conv_block  ### 2020/02/15重新審視一次 架構， 覺得 conv_transpose完應該要再接 conv_block 會更好吧~~
        self.concat_before_down = concat_before_down  ### 2020/02/16重新審視一次 架構， 覺得 應該用做完conv_blocks來skip到decoder比較對吧~~

        self.common_kwargs = dict(kernel_size=kernel_size, strides=strides, padding=padding, norm=norm,
                    #   d_acti=d_acti, u_acti=u_acti,
                      use_bias=use_bias,
                      coord_conv=coord_conv,
                    #   skip_op=skip_op,
                    #   skip_merge_op=skip_merge_op
                      )
        ### 確保 self.conv_block_num 是 list 的狀態
        if(type(self.conv_block_num) != type(tf.python.training.tracking.data_structures.ListWrapper([]))):  ### 用self 來接住 list 的話 tensorflow 會自動轉成 ListWrapper (別忘記我有繼承 tensorflow.keras.models.Model 喔～～)
            if  (out_conv_block is True):  self.conv_block_num = [self.conv_block_num] * (self.depth_level * 2) + [self.conv_block_num]
            elif(out_conv_block is False): self.conv_block_num = [self.conv_block_num] * (self.depth_level * 2) + [0]
        # print("self.conv_block_num:", self.conv_block_num)

        ### 定義 Down 架構
        ### 最基本(比如最少層depth_level=2)的一定有 top, bottle
        self.d_top    = UNet_down(at_where="top"   ,
                                  in_ch =self.Get_Layer_hid_ch(to_L=1, ch_upper_bound=ch_upper_bound),
                                  out_ch=self.Get_Layer_hid_ch(to_L=1, ch_upper_bound=ch_upper_bound),
                                  acti=d_acti,
                                  conv_block_num=self.conv_block_num[0],
                                  name="D_0->1_top",
                                  **self.common_kwargs)  ### Layer 0 -> 1， to_L=1 代表 走進 第1層
        self.d_middles = {}
        if(depth_level >= 3):
            for i in range(depth_level - 2):  ### -2 是 -top 和 -bottle 共兩層
                layer_id = i + 1 + 1  ### +1 是 index轉layer_id， 再+1 是因為前面有top層。 middle 至少 一定從 走入Layer2開始(Down) 或 從Layer2開始返回(Up)
                d_middle_name = f"D_{layer_id-1}->{layer_id}_middle"
                self.d_middles[d_middle_name] = UNet_down(at_where="middle",
                                                          in_ch =self.Get_Layer_hid_ch(to_L=layer_id - 1, ch_upper_bound=ch_upper_bound),
                                                          out_ch=self.Get_Layer_hid_ch(to_L=layer_id, ch_upper_bound=ch_upper_bound),
                                                          acti=d_acti,
                                                          conv_block_num=self.conv_block_num[layer_id - 1],
                                                          name=d_middle_name,
                                                          **self.common_kwargs )
        self.d_bottle = UNet_down(at_where="bottle",
                                  in_ch =self.Get_Layer_hid_ch(to_L=depth_level - 1, ch_upper_bound=ch_upper_bound),
                                  out_ch=self.Get_Layer_hid_ch(to_L=depth_level, ch_upper_bound=ch_upper_bound),
                                  acti=d_acti,
                                  conv_block_num=self.conv_block_num[self.depth_level - 1],
                                  name=f"D_{depth_level-1}->{depth_level}_bottle",
                                  **self.common_kwargs)

        ### 定義 Up 架構
        self.up_arch_dict = {}
        self.Up_arch_define()


        ############################################################################################################################################################
        if(self.unet_acti == "tanh"):    self.tanh    = Activation(tf.nn.tanh,    name="out_tanh")
        if(self.unet_acti == "sigmoid"): self.sigmoid = Activation(tf.nn.sigmoid, name="out_sigmoid")

    def Up_arch_define(self):
        for go_dec in range(self.d_amount):
            self.up_arch_dict[f"u{go_dec}_bottle" ] = UNet_up  (at_where="bottle",
                                                                in_ch =self.Get_Layer_hid_ch(to_L=self.depth_level    , ch_upper_bound=self.ch_upper_bound),
                                                                out_ch=self.Get_Layer_hid_ch(to_L=self.depth_level - 1, ch_upper_bound=self.ch_upper_bound),
                                                                acti=self.u_acti,
                                                                conv_block_num=self.conv_block_num[self.depth_level],
                                                                name=f"U_{self.depth_level}->{self.depth_level-1}_bottle",
                                                                **self.common_kwargs)  ### 因為是返回上一層， 所以 -1
            self.up_arch_dict[f"u{go_dec}_middles"] = {}
            if(self.depth_level >= 3):
                for i in range(self.depth_level - 2 - 1, 0 - 1, -1):  ### -2 是 -top 和 -bottle 共兩層， 最後的 start 和 stop 都 -1 是因為讓 range step 是 負一 要 -1 才會是我要的效果
                    '''
                    以 depth_level==5 舉例， 
                    我想要的range效果是跑  2,      1,      0，
                    layer_id 會+2 變      4,      3,      2，
                    u_middle_name 為      4->3 和 3->2 和 2->1
                    '''
                    layer_id = i + 1 + 1  ### +1 是 index轉layer_id， 再+1 是因為前面有top層。 middle 至少 一定從 走入Layer2開始(Down) 或 從Layer2開始返回(Up)， 這邊的 +2 不能和 for 裡面的 -2 消掉喔！ 因為 for 裡是 代表跑幾次！ 不能消！
                    # print("layer_id", layer_id)  ### debug 用
                    u_middle_name = f"U_{layer_id}->{layer_id-1}_middle"
                    # u_middle_name = f"{6-layer_id}U_{layer_id}->{layer_id-1}_middle"  ### 這可以照順序排，不過以前train的 網路 名字會對不起來無法reload QAQ
                    self.up_arch_dict[f"u{go_dec}_middles"][u_middle_name] = UNet_up  (at_where="middle",
                                                                                       in_ch =self.Get_Layer_hid_ch(to_L=layer_id    , ch_upper_bound=self.ch_upper_bound),
                                                                                       out_ch=self.Get_Layer_hid_ch(to_L=layer_id - 1, ch_upper_bound=self.ch_upper_bound),
                                                                                       acti=self.u_acti,
                                                                                       conv_block_num=self.conv_block_num[-layer_id - 1],  ### -1 是因為現在有新增 out_conv_block 在最尾巴， 所以原本的index要多-1
                                                                                       name=u_middle_name,
                                                                                       **self.common_kwargs)
            if(self.out_conv_block is False):
                self.up_arch_dict[f"u{go_dec}_top"] = UNet_up  (at_where="top",
                                                                in_ch =self.Get_Layer_hid_ch(to_L=1, ch_upper_bound=self.ch_upper_bound),
                                                                out_ch=self.Get_Layer_hid_ch(to_L=0, ch_upper_bound=self.ch_upper_bound),
                                                                acti=self.u_acti,
                                                                conv_block_num=self.conv_block_num[-1 - 1],  ### -1 是因為現在有新增 out_conv_block 在最尾巴， 所以原本的index要多-1
                                                                name="U_1->0_top",
                                                                **self.common_kwargs)  ### Layer 1 -> 0， to_L=0 代表 返回 第0層
            elif(self.out_conv_block is True):
                self.up_arch_dict[f"u{go_dec}_top"] = UNet_up  (at_where="top",
                                                                in_ch =self.hid_ch,
                                                                out_ch=self.hid_ch,
                                                                acti=self.u_acti,
                                                                conv_block_num=self.conv_block_num[-1 - 1],  ### -1 是因為現在有新增 out_conv_block 在最尾巴， 所以原本的index要多-1
                                                                name="U_1->0_top",
                                                                **self.common_kwargs)  ### Layer 1 -> 0， to_L=0 代表 返回 第0層

                self.up_arch_dict[f"u{go_dec}_top_out_IN"]     = InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform", name="U_0_top_out_IN")

                self.up_arch_dict[f"u{go_dec}_top_out_concat"] = Concatenate(name="U_0_top_out_concat")

                self.up_arch_dict[f"u{go_dec}_top_out_Acti"]   = Use_what_acti(self.u_acti)

                self.up_arch_dict[f"u{go_dec}_top_out_conv_blocks"]  = Conv_Blocks(in_ch  = self.hid_ch,
                                                                             out_ch = self.hid_ch,

                                                                             kernel_size=self.kernel_size,
                                                                             strides=1,
                                                                             padding="same",
                                                                             norm=self.norm,
                                                                             acti=self.u_acti,
                                                                             use_bias=self.use_bias,
                                                                             conv_block_num=self.conv_block_num[-1],
                                                                             coord_conv=self.coord_conv,
                                                                             name = "U_0_top_out_conv_blocks")

                self.up_arch_dict[f"u{go_dec}_top_out_1x1conv"]  = Conv2D(self.unet_out_ch, kernel_size=1, strides=1, padding=self.padding, use_bias=self.use_bias, name="U_0_top_out_1x1conv")

            # self.d_bottle = UNet_down(at_where="bottle", out_ch=min(hid_ch * 2**(depth_level - 1)    , 512), name=f"D{depth_level} bottle")  ### L0(3), L1(hid_ch*2**0), L2(hid_ch*2**1), ..., L2(hid_ch*2**depth_level - 1)
            # self.u_bottle = UNet_up  (at_where="bottle", out_ch=min(hid_ch * 2**(depth_level - 1 - 1), 512), name=f"U{depth_level} bottle")  ### L0(3), L1(hid_ch*2**0), L2(hid_ch*2**1), ..., L2(hid_ch*2**depth_level - 1)， 因為是返回上一層， out_ch 2的冪次要再 -1

    def Get_Layer_hid_ch(self, to_L, ch_upper_bound=512):
        # print("Get_Layer_hid_ch ch_upper_bound", ch_upper_bound)
        '''
        L_0 (3 或 1),
        L_1 (hid_ch*2**0),
        L_2 (hid_ch*2**1),
        L_3 (hid_ch*2**2),
        ...,
        L_depth_level (hid_ch*2**depth_level - 1)
        '''
        if(to_L == 0): return self.unet_out_ch
        else:          return min(self.hid_ch * 2**(to_L - 1), ch_upper_bound)

    def call(self, input_tensor, training=None):
        skips = []

        #####################################################
        ### Down top
        x, x_after_down, x_before_down = self.d_top(input_tensor)
        if(self.out_conv_block is True):  skips.append(x_before_down)  ### 如果有用 out_conv_block的話 記得也要多他的 skip 喔！ 而他的skip 一定只能用 x_before_down 不能用 after_down， 要不shape對不到
        if(self.debug):  print(f"{self.d_top.name} x_before_down: {x_before_down[0, 0, 0, :3]}")  ### debug 用

        if  (self.concat_before_down is False): skips.append(x_after_down)
        elif(self.concat_before_down is True):  skips.append(x_before_down)
        if(self.debug):
            if  (self.concat_before_down is False): print(f"{self.d_top.name} x_after_down : {x_after_down[0, 0, 0, :3]}")  ### debug 用
            elif(self.concat_before_down is True):  print(f"{self.d_top.name} x_before_down: {x_before_down[0, 0, 0, :3]}")  ### debug 用

        if(self.out_conv_block is True and self.concat_before_down is True):
            skips.pop()  ### pop 的原因打在下面
            if(self.debug):
                print("out_conv_block     為 True 時 已經 append 一次 x_before_down，")
                print("concat_before_down 為 True 時 又會 append 一次 x_before_down， 就重複了，")
                print("所以 這邊 skips pop 一次， 拿掉重複的 x_before_down～")

        ### Down middle
        for name, d_middle in self.d_middles.items():
            x, x_after_down, x_before_down = d_middle(x)
            if  (self.concat_before_down is False): skips.append(x_after_down)
            elif(self.concat_before_down is True):  skips.append(x_before_down)
            if(self.debug):
                if  (self.concat_before_down is False): print(f"{name} x_after_down : {x_after_down[0, 0, 0, :3]}")  ### debug 用
                elif(self.concat_before_down is True):  print(f"{name} x_before_down: {x_before_down[0, 0, 0, :3]}")  ### debug 用
        ### Down bottle
        # print(self.d_bottle.name)  ### debug 用
        x_bottle, x_after_down, x_before_down = self.d_bottle(x)  ### down 的 bottle沒有需要用到 x_after_down
        if  (self.concat_before_down is False): pass
        elif(self.concat_before_down is True):  skips.append(x_before_down)
        if(self.debug):
            if  (self.concat_before_down is False): print(f"{self.d_bottle.name} x_after_down : {x_after_down[0, 0, 0, :3]}")  ### debug 用
            elif(self.concat_before_down is True):  print(f"{self.d_bottle.name} x_before_down: {x_before_down[0, 0, 0, :3]}")  ### debug 用

        ### 檢查一下 skips 裡面的東西對不對
        if(self.debug):
            for i, skip in enumerate(skips): print(f"skip{i}.shape: {skip.shape}, {skip[0, 0, 0, :3]}")
        #####################################################
        ### Bottle divide， 此寫法也相容 不divide的情況喔
        n, h, w, c = x_bottle.shape
        c_div = c // self.d_amount
        x_bottle_divs = []
        for dec_i in range(self.d_amount):
            x_bottle_divs.append( x_bottle[..., c_div * dec_i : c_div * (dec_i + 1)] )
            if(self.debug): print(f"dec_i:{dec_i}, bottle_feature:{x_bottle_divs[-1][0, 0, 0, :3]}")  ### debug 用
        #####################################################
        outs = []
        if(self.debug): print("self.d_amount", self.d_amount)
        for go_dec in range(self.d_amount):
            skip_id = -1  ### 倒數第一個

            ### Up bottle
            if(self.no_concat_layer >= self.depth_level - 1):  ### no_concat 的 Case
                if(self.debug): print(f"u{go_dec}_bottle ", self.up_arch_dict[f"u{go_dec}_bottle"].name, f"feature {x_bottle_divs[go_dec][0, 0, 0, :3]} no concat")  ### debug 用
                feature_up = self.up_arch_dict[f"u{go_dec}_bottle"](x_bottle_divs[go_dec])
            else:  ### concat 的 Case
                if(self.debug): print(f"u{go_dec}_bottle ", self.up_arch_dict[f"u{go_dec}_bottle"].name, f"feature {x_bottle_divs[go_dec][0, 0, 0, :3]} concat with {skips[skip_id][0, 0, 0, :3]}")  ### debug 用
                feature_up = self.up_arch_dict[f"u{go_dec}_bottle"](x_bottle_divs[go_dec], skips[skip_id])
            skip_id -= 1  ### 接著繼續倒數下一個

            ### Up middle
            for go, (name, u_middle) in enumerate(list(self.up_arch_dict[f"u{go_dec}_middles"].items())):
                layer_id = self.depth_level - 1 - go
                if (layer_id <= self.no_concat_layer):
                    ### no_concat 的 Case
                    if(self.debug): print(f"u{go_dec}_middles", f"{name} feature {feature_up[0, 0, 0, :3]} no concat")  ### debug 用
                    feature_up = u_middle(feature_up)
                else:
                    ### concat 的 Case
                    if(self.debug): print(f"u{go_dec}_middles", f"{name} feature {feature_up[0, 0, 0, :3]} concat with {skips[skip_id][0, 0, 0, :3]}")  ### debug 用
                    feature_up = u_middle(feature_up, skips[skip_id])
                skip_id -= 1  ### 接著繼續倒數下一個
            ### Up top
            feature_up = self.up_arch_dict[f"u{go_dec}_top"](feature_up)  ### up 的 top 沒有 skip
            if(self.debug): print(f"u{go_dec}_top    ", self.up_arch_dict[f"u{go_dec}_top"].name, "no concat")  ### debug 用

            ### 如果要用 v3 的 輸出前要接 Conv_Blocks 的 Case
            if(self.out_conv_block is True):
                ### IN
                feature_up = self.up_arch_dict[f"u{go_dec}_top_out_IN"](feature_up)

                ### 看要不要 Concat
                if(self.no_concat_layer >= 1):
                    ### no_concat 的 Case
                    if(self.debug): print(f"u{go_dec}_top_out_conv_blocks", self.up_arch_dict[f"u{go_dec}_top_out_conv_blocks"].name, "no concat")  ### debug 用
                    feature_up = self.up_arch_dict[f"u{go_dec}_top_out_conv_blocks"](feature_up)
                elif(self.no_concat_layer == 0):
                    ### concat 的 Case
                    if(self.debug): print(f"u{go_dec}_top_out_conv_blocks", self.up_arch_dict[f"u{go_dec}_top_out_conv_blocks"].name, f"concat with {skips[skip_id][0, 0, 0, :3]}")  ### debug 用
                    b, h, w, c = feature_up.shape  ### 因為想嘗試 no_pad， 所以 pred 可能 size 會跟 gt 差一點點， 就以 pred為主喔！
                    feature_up = self.up_arch_dict[f"u{go_dec}_top_out_concat"]([skips[skip_id][:, :h, :w, :], feature_up])

                ### Activation
                feature_up = self.up_arch_dict[f"u{go_dec}_top_out_Acti"](feature_up)

                ### Conv_Blocks
                feature_up = self.up_arch_dict[f"u{go_dec}_top_out_conv_blocks"](feature_up)

                ### 1x1Conv
                feature_up = self.up_arch_dict[f"u{go_dec}_top_out_1x1conv"](feature_up)

            ### 有可能有多個Decoder， 所以做完的結果 append 到 outs
            outs.append(feature_up)

        #####################################################
        acti_outs = []
        for out in outs:
            if  (self.unet_acti == "tanh"):    acti_outs.append(self.tanh(out))
            elif(self.unet_acti == "sigmoid"): acti_outs.append(self.sigmoid(out))

        if(len(acti_outs) == 1): return acti_outs[0]  ### 如果list內容物只有一個， 相當於 x = [ 一個 ], x = [一個]  在外面他不會自動解開！要解開再return 喔！
        else:                    return acti_outs     ### 如果list內容物一個以上， 相當於 x, y = [ 一個, 一個 ], x = 一個, y = 一個 ， 會自動解開～　所以整個list 回傳也沒問題！

        # #####################################################
        # if(self.bottle_divide is False):
        #     x1_bottle = x_bottle
        #     x2_bottle = x_bottle
        #     x3_bottle = x_bottle
        # else:
        #     n, h, w, c = x_bottle.shape
        #     # print("n, h, w, c~~~~~~~~~~~~~~~~", n, h, w, c)
        #     c_div = c // self.d_amount
        #     x1_bottle  = x_bottle[..., c_div * 0 : c_div * (0 + 1)]  ### dec_id = 0
        #     x2_bottle  = x_bottle[..., c_div * 1 : c_div * (1 + 1)]  ### dec_id = 1
        #     x3_bottle  = x_bottle[..., c_div * 2 : c_div * (2 + 1)]  ### dec_id = 2

        # #####################################################
        # skip_id = self.depth_level - 1 - 1  ### 第一個 -1 是因為 top_bottle 沒有skip， 第二個 -1 是因為 數量 轉 index
        # if(self.d_amount >= 1):
        #     ### Up bottle
        #     # print(self.u_bottle.name)  ### debug 用
        #     if(self.no_concat_layer >= self.depth_level - 1): x1 = self.u_bottle(x1_bottle)
        #     else:                                             x1 = self.u_bottle(x1_bottle, skips[skip_id])
        #     skip_id -= 1

        #     ### Up middle
        #     for go, (name, u_middle) in enumerate(list(self.u_middles.items())):
        #         # print("up mid name:", name)  ### debug 用
        #         layer_id = self.depth_level - 1 - go
        #         if (layer_id <= self.no_concat_layer): x1 = u_middle(x1)
        #         else:                                  x1 = u_middle(x1, skips[skip_id])
        #         skip_id -= 1
        #     ### Up top
        #     # print(self.u_top.name)  ### debug 用
        #     x1 = self.u_top(x1)  ### up 的 top 沒有 skip
        # #####################################################
        # skip_id = self.depth_level - 1 - 1  ### 第一個 -1 是因為 top_bottle 沒有skip， 第二個 -1 是因為 數量 轉 index
        # if(self.d_amount >= 2):
        #     ### Up bottle
        #     if(self.no_concat_layer >= self.depth_level - 1): x2 = self.u_bottle2(x2_bottle)
        #     else:                                             x2 = self.u_bottle2(x2_bottle, skips[skip_id])
        #     skip_id -= 1

        #     ### Up middle
        #     for go, (name, u_middle2) in enumerate(list(self.u_middle2s.items())):
        #         # print("up mid name:", name)  ### debug 用
        #         layer_id = self.depth_level - 1 - go
        #         if (layer_id <= self.no_concat_layer): x2 = u_middle2(x2)
        #         else:                                  x2 = u_middle2(x2, skips[skip_id])
        #         skip_id -= 1
        #     ### Up top
        #     # print(self.u_top.name)  ### debug 用
        #     x2 = self.u_top2(x2)  ### up 的 top 沒有 skip
        # #####################################################
        # skip_id = self.depth_level - 1 - 1  ### 第一個 -1 是因為 top_bottle 沒有skip， 第二個 -1 是因為 數量 轉 index
        # if(self.d_amount >= 3):
        #     ### Up bottle
        #     if(self.no_concat_layer >= self.depth_level - 1): x3 = self.u_bottle3(x3_bottle)
        #     else:                                             x3 = self.u_bottle3(x3_bottle, skips[skip_id])
        #     skip_id -= 1

        #     ### Up middle
        #     for go, (name, u_middle3) in enumerate(list(self.u_middle3s.items())):
        #         # print("up mid name:", name)  ### debug 用
        #         layer_id = self.depth_level - 1 - go
        #         if (layer_id <= self.no_concat_layer): x3 = u_middle3(x3)
        #         else:                                  x3 = u_middle3(x3, skips[skip_id])
        #         skip_id -= 1
        #     ### Up top
        #     # print(self.u_top.name)  ### debug 用
        #     x3 = self.u_top3(x3)  ### up 的 top 沒有 skip
        # #####################################################
        # ### UNet out
        # if(self.d_amount == 1):
        #     if  (self.unet_acti == "tanh"):    return self.tanh(x1)
        #     elif(self.unet_acti == "sigmoid"): return self.sigmoid(x1)
        # elif(self.d_amount == 2):
        #     if  (self.unet_acti == "tanh"):    return self.tanh(x1), self.tanh(x2)
        #     elif(self.unet_acti == "sigmoid"): return self.sigmoid(x1), self.sigmoid(x2)
        # elif(self.d_amount == 3):
        #     if  (self.unet_acti == "tanh"):    return self.tanh(x1), self.tanh(x2), self.tanh(x3)
        #     elif(self.unet_acti == "sigmoid"): return self.sigmoid(x1), self.sigmoid(x2), self.sigmoid(x3)


if(__name__ == "__main__"):
    import numpy as np
    import time
    from kong_util.tf_model_util import Show_model_layer_names, Show_model_weights
    # data = np.ones(shape=(1, 512, 512, 3), dtype=np.float32)
    # start_time = time.time()  # 看資料跑一次花多少時間
    # # test_g = Generator(hid_ch=64, depth_level=7, use_bias=False)
    # test_g = Generator(hid_ch= 128, depth_level=4, out_ch=1, unet_acti="sigmoid", conv_block_num=1, ch_upper_bound= 2**14)
    # test_g(data)
    # print("cost time", time.time() - start_time)
    # test_g.summary()
    # print(test_g(data))



    ############################################################################################################################
    ### 嘗試 真的 load tf_data 進來 train 看看
    import numpy as np
    from tqdm import tqdm
    from step06_a_datas_obj import *
    from step06_d_tf_Data_builder import tf_Data_builder
    from step10_a2_loss_info_obj import *
    from step09_c_train_step import *

    # from step09_e2_mask_unet2_obj import *
    # from step09_e3_flow_unet2_obj_I_to_C import *
    # from step09_e4_flow_unet2_obj_Mgt_to_C import *
    from step09_e5_flow_unet2_obj_I_w_Mgt_to_C import *


    # model_obj = flow_unet2_ch032_tanh_L7
    # model_obj = flow_unet2_ch128_sig_L7
    # model_obj = flow_unet2_ch064_sig_L7
    # model_obj = flow_unet2_ch032_sig_L7
    # model_obj = flow_unet2_ch016_sig_L7
    # model_obj = flow_unet2_ch008_sig_L7
    # model_obj = flow_unet2_ch004_sig_L7
    # model_obj = flow_unet2_ch002_sig_L7
    # model_obj = flow_unet2_ch001_sig_L7

    # model_obj = flow_unet2_L2_ch32_sig
    # model_obj = flow_unet2_L3_ch32_sig
    # model_obj = flow_unet2_L4_ch32_sig
    # model_obj = flow_unet2_L5_ch32_sig
    # model_obj = flow_unet2_L6_ch32_sig
    # model_obj = flow_unet2_L7_ch32_sig
    # model_obj = flow_unet2_L8_ch32_sig

    # model_obj = flow_unet2_IN_L7_ch32_2to2noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to3noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to4noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to5noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to6noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to7noC_sig
    # model_obj = flow_unet2_IN_L7_ch32_2to8noC_sig

    # model_obj = flow_unet2_L8_skip_use_add_sig
    # model_obj = flow_unet2_L7_skip_use_add_sig
    # model_obj = flow_unet2_L6_skip_use_add_sig
    # model_obj = flow_unet2_L5_skip_use_add_sig
    # model_obj = flow_unet2_L4_skip_use_add_sig
    # model_obj = flow_unet2_L3_skip_use_add_sig
    # model_obj = flow_unet2_L2_skip_use_add_sig

    model_obj = flow_unet2_block1_ch016_sig_L6

    model_obj = model_obj.build()  ### 可替換成 上面 想測試的 model

    ### 2. db_obj 和 tf_data
    db_obj  = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    tf_data = tf_Data_builder().set_basic(db_obj, 1, train_shuffle=False).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).set_img_resize(( 512, 512) ).build_by_db_get_method().build()

    ### 3. loss_info_obj
    loss_info_objs = [G_mae_s001_loss_info_builder.build()]
    ### 4. 跑起來試試看
    for n, (train_in, train_in_pre, train_gt, train_gt_pre, _) in enumerate(tqdm(tf_data.train_db_combine)):
        # print("train_in.numpy().min():", train_in.numpy().min())
        # print("train_in.numpy().max():", train_in.numpy().max())
        # print("train_in_pre.numpy().min():", train_in_pre.numpy().min())
        # print("train_in_pre.numpy().max():", train_in_pre.numpy().max())
        model_obj.train_step(model_obj=model_obj, in_data=train_in_pre, gt_data=train_gt_pre, loss_info_objs=loss_info_objs)
        if(n ==  0):
            model_obj.generator.summary()
            Show_model_weights(model_obj.generator)
        if(n == 10):
            model_obj.generator.save_weights("debug_data/try_save/weights")
            iter10 = model_obj.generator.layers[0].weights[1]
            print("iter10:", iter10)
        if(n == 20):
            iter20 = model_obj.generator.layers[0].weights[1]
            print("iter20:", iter20)
            model_obj.generator.load_weights("debug_data/try_save/weights")
            iter20_load10 = model_obj.generator.layers[0].weights[1]
            print("iter20_load10:", iter20_load10)
