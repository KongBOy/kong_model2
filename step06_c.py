class tf_Data_init_builder:
    def __init__(self, tf_data=None):
        if(tf_data is None): self.tf_data = tf_Data()
        else:                self.tf_data = tf_data

    def set_basic(self, db_obj, batch_size=1, train_shuffle=True):
        self.tf_data.db_obj        = db_obj
        self.tf_data.batch_size    = batch_size
        self.tf_data.train_shuffle = train_shuffle
        return self

    def set_data_use_range(self, use_in_range, use_gt_range, use_rec_hope_range=Range(0, 255)):
    # def set_data_use_range(self, use_in_range=Range(0, 1), use_gt_range=Range(0, 1), use_rec_hope_range=Range(0, 255), in_min=None, in_max=None, gt_min=None, gt_max=None, rec_hope_min=None, rec_hope_max=None):
        self.tf_data.use_in_range = use_in_range
        self.tf_data.use_gt_range = use_gt_range
        self.tf_data.use_rec_hope_range = use_rec_hope_range
        # self.tf_data.in_max       = in_max
        # self.tf_data.in_min       = in_min
        # self.tf_data.gt_min       = gt_min
        # self.tf_data.gt_max       = gt_max
        # self.tf_data.rec_hope_min = rec_hope_min
        # self.tf_data.rec_hope_max = rec_hope_max
        return self

    def set_img_resize(self, img_resize):
        self.tf_data.img_resize = img_resize
        # '''
        # tuple / list 或 <enum 'MODEL_NAME'>
        # '''
        # # print(type(img_resize))

        # ### tuple / list 的 case
        # if(type(img_resize) == type( () ) or type(img_resize) == type( [] )):
        #     self.tf_data.img_resize = img_resize
        # ### <enum 'MODEL_NAME'> 的 case
        # else:
        #     # print("doing tf_data resize according model_name")
        #     # print("self.tf_data.db_obj.h = ", self.tf_data.db_obj.h)
        #     # print("self.tf_data.db_obj.w = ", self.tf_data.db_obj.w)
        #     # print("math.ceil(self.tf_data.db_obj.h / 128) * 128 = ", math.ceil(self.tf_data.db_obj.h / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
        #     # print("math.ceil(self.tf_data.db_obj.w / 128) * 128 = ", math.ceil(self.tf_data.db_obj.w / 128) * 128 )  ### move_map的話好像要用floor再*2的樣子，覺得算了應該也不會再用那個了就直接改掉了
        #     if  ("unet" in img_resize.value):
        #         self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h / 128) * 128 , math.ceil(self.tf_data.db_obj.w / 128) * 128)  ### 128的倍數，且要是gt_img的兩倍大喔！
        #     elif("rect" in img_resize.value or "justG" in img_resize.value):
        #         self.tf_data.img_resize = (math.ceil(self.tf_data.db_obj.h / 4) * 4, math.ceil(self.tf_data.db_obj.w / 4) * 4)  ### dis_img(in_img的大小)的大小且要是4的倍數
        return self

    def build_by_db_get_method(self):
        self.train_in_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_in_dir, file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.train_gt_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_gt_dir, file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()
        self.test_in_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_in_dir,  file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.test_gt_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_gt_dir,  file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()
        self.see_in_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_in_dir,   file_format=self.tf_data.db_obj.in_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in_range, use_range=self.tf_data.use_in_range).build()
        self.see_gt_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_gt_dir,   file_format=self.tf_data.db_obj.gt_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt_range, use_range=self.tf_data.use_gt_range).build()

        if("未指定" not in self.tf_data.db_obj.train_in2_dir): self.train_in2_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_in2_dir, file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.train_gt2_dir): self.train_gt2_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.train_gt2_dir, file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()
        if("未指定" not in self.tf_data.db_obj.test_in2_dir ): self.test_in2_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_in2_dir,  file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.test_gt2_dir ): self.test_gt2_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.test_gt2_dir,  file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()
        if("未指定" not in self.tf_data.db_obj.see_in2_dir  ): self.see_in2_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_in2_dir,   file_format=self.tf_data.db_obj.in2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_in2_range, use_range=self.tf_data.use_in_range).build()
        if("未指定" not in self.tf_data.db_obj.see_gt2_dir  ): self.see_gt2_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.see_gt2_dir,   file_format=self.tf_data.db_obj.gt2_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_gt2_range, use_range=self.tf_data.use_gt_range).build()

        self.rec_hope_train_factory = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_train_dir, file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_test_factory  = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_test_dir,  file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()
        self.rec_hope_see_factory   = tf_Data_element_factory_builder().set_factory(self.tf_data.db_obj.rec_hope_see_dir,   file_format=self.tf_data.db_obj.rec_hope_format, img_resize=self.tf_data.img_resize, db_h=self.tf_data.db_obj.h, db_w=self.tf_data.db_obj.w, db_range=self.tf_data.db_obj.db_rec_hope_range, use_range=self.tf_data.use_rec_hope_range).build()


        if    (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_move_map):
            self.build_by_in_dis_gt_move_map()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord or
               self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_ord_pad or
               self.tf_data.db_obj.get_method == DB_GM.in_rec_gt_ord):
            self.build_by_in_img_and_gt_img_db()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_flow):          self.build_by_in_dis_gt_flow_or_wc(get_what="flow")
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_mask_coord):    self.build_by_in_dis_gt_mask_coord()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc):            self.build_by_in_dis_gt_flow_or_wc(get_what="wc")
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc_try_mul_M):  self.build_by_in_dis_gt_wc_try_mul_M()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_img_gt_mask):          self.build_by_in_img_gt_mask()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_wc_gt_flow):           self.build_by_in_wc_gt_flow()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_wc_gt_flow_try_mul_M): self.build_by_in_wc_gt_flow_try_mul_M()
        elif  (self.tf_data.db_obj.get_method == DB_GM.in_dis_gt_wc_flow_try_mul_M): self.build_by_in_dis_gt_wc_flow_try_mul_M()

        return self



    def _build_train_test_in_img_db(self):
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_in_db   = self.train_in_factory.build_img_db()
        self.tf_data.train_name_db = self.train_in_factory.build_name_db()

        self.tf_data.test_in_db    = self.test_in_factory.build_img_db()
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

    def _train_in_gt_and_test_in_gt_combine_then_train_shuffle(self):
        ### 先 zip 再 batch == 先 batch 再 zip (已經實驗過了，詳細內容看 try_lots 的 try10_資料pipline囉)
        ### train_in,gt 打包
        self.tf_data.train_db_combine = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in_db.pre,
                                                             self.tf_data.train_gt_db.ord, self.tf_data.train_gt_db.pre,
                                                             self.tf_data.train_name_db.ord))
        ### test_in,gt 打包
        self.tf_data.test_db_combine  = tf.data.Dataset.zip((self.tf_data.test_in_db.ord, self.tf_data.test_in_db.pre,
                                                             self.tf_data.test_gt_db.ord, self.tf_data.test_gt_db.pre,
                                                             self.tf_data.test_name_db.ord))
        ### 先shuffle(只有 train_db 需要) 再 batch
        ### train shuffle
        # if(self.tf_data.train_shuffle): self.tf_data.train_db_combine = self.tf_data.train_db_combine.shuffle(int(self.tf_data.train_amount / 2))  ### shuffle 的 buffer_size 太大會爆記憶體，嘗試了一下大概 /1.8 左右ok這樣子~ 但 /2 應該比較保險！
        if(self.tf_data.train_shuffle): self.tf_data.train_db_combine = self.tf_data.train_db_combine.shuffle(200)  ### 在 kong_model2_lots_try/try10_資料pipline 有很棒的模擬， 忘記可以去參考看看喔！
        ### train 取 batch 和 prefetch(因為有做訓練，訓練中就可以先取下個data了)
        self.tf_data.train_db_combine = self.tf_data.train_db_combine.batch(self.tf_data.batch_size)       ### shuffle完 打包成一包包 batch
        self.tf_data.train_db_combine = self.tf_data.train_db_combine.prefetch(-1)                         ### -1 代表 AUTOTUNE：https://www.tensorflow.org/api_docs/python/tf/data#AUTOTUNE，我自己的觀察是 他會視系統狀況自動調速度， 所以比較不會 cuda load failed 這樣子 ~~bb
        # self.tf_data.train_db_combine = self.tf_data.train_db_combine.prefetch(self.tf_data.train_amount)  ### 反正就一個很大的數字， 可以穩定的用最高速跑， 但如果有再做別的事情可能會 cuda load failed 這樣子~ 如果設一個很大的數字， 觀察起來是會盡可能全速跑， 如果再做其他事情(看YT) 可能會　cuda error 喔～
        ### test 取 batch
        self.tf_data.test_db_combine  = self.tf_data.test_db_combine.batch(self.tf_data.batch_size)   ### shuffle完 打包成一包包 batch


class tf_Data_in_dis_gt_move_map_builder(tf_Data_init_builder):
    def build_by_in_dis_gt_move_map(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()


        ### 在拿move_map db 之前，要先去抓 max/min train_move，我是設計放 train_gt_dir 下的.npy，如果怕混淆 要改放.txt之類的都可以喔！
        ### 決定還是放在上一層好了，因為下面會用 get_db_amount 是算檔案數量的，雖然是去in_dir抓影像跟gt_dir沒關係，但還是怕有意外(以後忘記之類的)～放外面最安全囉！
        ### 且放外面容易看到可以提醒自己有這東西的存在覺得ˊ口ˋ
        if(os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy") and
           os.path.isfile(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")):
            self.tf_data.gt_max = np.load(self.tf_data.db_obj.train_gt_dir + "/../max_train_move.npy")
            self.tf_data.gt_min = np.load(self.tf_data.db_obj.train_gt_dir + "/../min_train_move.npy")
        else:  ### 如果.npy不存在，就去重新找一次 max/min train_move，找完也順便存一份給之後用囉！
            print("因為現在已經存成.knpy，沒辦法抓 max/min train_move 囉！麻煩先去以前的dataset撈出來啦！")
            ### 偷懶可以把 .npy 放同個資料夾，把註解拿掉就可以順便求囉！只是因為這有return，所以還是要重新執行一次才會完整跑完喔～
            # move_maps = get_dir_moves(self.tf_data.db_obj.train_gt_dir)
            # self.tf_data.gt_max = move_maps.max()
            # self.tf_data.gt_min = move_maps.min()
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../max_train_move", self.tf_data.max_train_move)
            # np.save(self.tf_data.db_obj.train_gt_dir+"/../min_train_move", self.tf_data.min_train_move)
            # print("self.tf_data.max_train_move",self.tf_data.max_train_move)
            # print("self.tf_data.min_train_move",self.tf_data.min_train_move)
            return

        self.tf_data.train_gt_db = self.train_gt_factory.build_mov_db()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_mov_db()


        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            self.tf_data.see_gt_db   = self.see_gt_factory.build_mov_db()

            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method2

        # take_num = 5
        # print(self.tf_data.train_max)
        # print(self.tf_data.train_min)
        # for i, (img, img_pre, move, move_pre) in enumerate(self.tf_data.train_db_combine.take(take_num)):     ### 想看test 的部分用這行 且 註解掉上行
        #     print("i",i)
        #     fig, ax = plt.subplots(1,4)
        #     fig.set_size_inches(15,5)
        #     ax_i = 0
        #     img = tf.cast(img[0], tf.uint8)
        #     ax[ax_i].imshow(img)
        #     print(img.numpy().dtype)

        #     ax_i += 1
        #     img_pre_back = (img_pre[0]+1.)*127.5
        #     img_pre_back = tf.cast(img_pre_back, tf.int32)
        #     ax[ax_i].imshow(img_pre_back)

        #     ax_i += 1
        #     move_bgr = method2(move[0,...,0], move[0,...,1])
        #     ax[ax_i].imshow(move_bgr)

        #     ax_i += 1
        #     # move_back = (move[0]+1)/2 * (train_max-train_min) + train_min  ### 想看train的部分用這行 且 註解掉下行
        #     move_back = (move_pre[0]+1)/2 * (self.tf_data.train_max-self.tf_data.train_min) + self.tf_data.train_min    ### 想看test 的部分用這行 且 註解掉上行
        #     move_back_bgr = method2(move_back[...,0], move_back[...,1],1)
        #     ax[ax_i].imshow(move_back_bgr)
        #     plt.show()
        #     plt.close()
        #########################################################################################################################################
        return self


class tf_Data_in_dis_gt_img_builder(tf_Data_in_dis_gt_move_map_builder):
    def build_by_in_img_and_gt_img_db(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        self.tf_data.train_gt_db = self.train_gt_factory.build_img_db()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_img_db()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        #########################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     train_in     = train_in[0]      ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0]  ### 值 0. ~ 1.
        #     train_gt     = train_gt[0]      ### 值 0  ~ 255
        #     train_gt_pre = train_gt_pre[0]  ### 值 0. ~ 1.

        #     fig, ax = plt.subplots(1, 4)
        #     fig.set_size_inches(15, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()
        #########################################################

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_img_db()

            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)
        return self
    ############################################################


class tf_Data_in_dis_gt_flow_or_wc_builder(tf_Data_in_dis_gt_img_builder):
    def build_by_in_dis_gt_flow_or_wc(self, get_what="flow"):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        if  (get_what == "flow"): self.tf_data.train_gt_db = self.train_gt_factory.build_flow_db()
        elif(get_what == "wc"  ): self.tf_data.train_gt_db = self.train_gt_factory.build_M_W_db_wrong()

        if  (get_what == "flow"): self.tf_data.test_gt_db = self.test_gt_factory.build_flow_db()
        elif(get_what == "wc"):   self.tf_data.test_gt_db = self.test_gt_factory.build_M_W_db_wrong()


        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # print('self.tf_data.train_in_db',self.tf_data.train_in_db)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db_pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db_pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            if  (get_what == "flow"): self.tf_data.see_gt_db = self.see_gt_factory.build_flow_db()
            elif(get_what == "wc"):   self.tf_data.see_gt_db = self.see_gt_factory.build_M_W_db_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)


        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db_pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1

        # # for i in enumerate(self.tf_data.train_gt_db): pass
        # # print("train_gt_finish")

        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(5)):
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1 train_in"    ] = train_in[0].numpy()
        #     debug_dict[f"{i}--2-2 train_in_pre"] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 train_Mgt"    ] = train_gt[0][0].numpy()
        #     debug_dict[f"{i}--2-4 train_Mgt_pre"] = train_gt_pre[0][0].numpy()
        #     debug_dict[f"{i}--2-5 train_Wgt"    ] = train_gt[1][0].numpy()
        #     debug_dict[f"{i}--2-6 train_Wgt_pre"] = train_gt_pre[1][0].numpy()

        #     if(get_what == "flow"):
        #         train_gt_visual     = method1(train_gt[0, ..., 2]    , train_gt[0, ..., 1])
        #         train_gt_pre_visual = method1(train_gt_pre[0, ..., 2], train_gt_pre[0, ..., 1])

        #         fig, ax = plt.subplots(1, 4)
        #         fig.set_size_inches(15, 5)
        #         ax[0].imshow(train_in[0])
        #         ax[1].imshow(train_in_pre[0])
        #         ax[2].imshow(train_gt_visual)
        #         ax[3].imshow(train_gt_pre_visual)
        #         ax[2].imshow(train_gt[0])
        #         ax[3].imshow(train_gt_pre[0])

        #     elif(get_what == "wc"):
        #         fig, ax = plt.subplots(2, 5)
        #         fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #         ### ord vs pre
        #         ax[0, 0].imshow(train_in    [0])
        #         ax[0, 1].imshow(train_in_pre[0])

        #         ### W_ord vs W_pre
        #         ax[0, 2].imshow(train_gt    [0, ..., :3])
        #         ax[0, 3].imshow(train_gt_pre[0, ..., :3])

        #         ### Wx, Wy, Wz 看一下長什麼樣子
        #         ax[1, 0].imshow(train_gt_pre[0, ..., 0])
        #         ax[1, 1].imshow(train_gt_pre[0, ..., 1])
        #         ax[1, 2].imshow(train_gt_pre[0, ..., 2])

        #         ### M_ord vs M_pre
        #         ax[1, 3].imshow(train_gt    [0, ..., 3])
        #         ax[1, 4].imshow(train_gt_pre[0, ..., 3])
        #     fig.tight_layout()
        #     plt.show()
        #########################################################################################################################################
        return self

    def build_by_in_dis_gt_wc_try_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()
        # self.tf_data.train_gt_db = self.train_gt_factory.build_M_W_db_wrong()  ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        # self.tf_data.test_gt_db  = self.test_gt_factory.build_M_W_db_wrong()   ### 可以留著原本的 用 下面的視覺化來比較一下 then_mul_M 的差異
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_W_db_right()
        self.tf_data.test_gt_db  = self.test_gt_factory.build_M_W_db_right()
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()
        # print('self.tf_data.train_in_db',self.tf_data.train_in_db.ord)
        # print('self.tf_data.train_in_db_pre',self.tf_data.train_in_db.pre)
        # print('self.tf_data.train_gt_db',self.tf_data.train_gt_db.ord)
        # print('self.tf_data.train_gt_db_pre',self.tf_data.train_gt_db.pre)

        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_gt_db   = self.see_gt_factory.build_M_W_db_right()
            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # from util import method1
            # for i in enumerate(self.tf_data.train_gt_db): pass
            # print("train_gt_finish")

            # for i, (see_in, see_in_pre, see_gt, see_gt_pre, name) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                         self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1),
            #                                                                                         self.tf_data.see_name_db.ord))):
            #     debug_dict[f"{i}--1-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--1-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--1-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--1-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--2-1 see_in"    ]  = see_in    [0].numpy()
            #     debug_dict[f"{i}--2-2 see_in_pre"]  = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--2-3 see_Mgt"    ] = see_gt    [0].numpy()
            #     debug_dict[f"{i}--2-4 see_Mgt_pre"] = see_gt_pre[0].numpy()
            #     debug_dict[f"{i}--2-5 see_Wgt"    ] = see_gt    [0].numpy()
            #     debug_dict[f"{i}--2-6 see_Wgt_pre"] = see_gt_pre[0].numpy()

            #     fig, ax = plt.subplots(2, 5)
            #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
            #     ### ord vs pre
            #     ax[0, 0].imshow(see_in    [0])
            #     ax[0, 1].imshow(see_in_pre[0])

            #     ### W_ord vs W_pre
            #     ax[0, 2].imshow(see_gt    [0, ..., :3])
            #     ax[0, 3].imshow(see_gt_pre[0, ..., :3])

            #     ### Wx, Wy, Wz 看一下長什麼樣子
            #     ax[1, 0].imshow(see_gt_pre[0, ..., 0])
            #     ax[1, 1].imshow(see_gt_pre[0, ..., 1])
            #     ax[1, 2].imshow(see_gt_pre[0, ..., 2])

            #     ### M_ord vs M_pre
            #     Mgt     = see_gt    [0, ..., 3:4]
            #     Mgt_pre = see_gt_pre[0, ..., 3:4]
            #     ax[1, 3].imshow(see_gt    [0, ..., 3:4])
            #     ax[1, 4].imshow(see_gt_pre[0, ..., 3:4])

            #     in_pre_w_Mgt_pre = see_in_pre[0] * Mgt_pre
            #     ax[0, 4].imshow(in_pre_w_Mgt_pre)

            #     fig.tight_layout()
            #     plt.show()
            ##########################################################################################################################################

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()

            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################


        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # import os
        # for i in enumerate(self.tf_data.train_gt_db.ord): pass
        # print("train_gt_finish")

        # # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(5)):
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine):
        #     # if(i < 620): continue
        #     # print(i)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1 train_in"    ]  = train_in    [0].numpy()
        #     debug_dict[f"{i}--2-2 train_in_pre"]  = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 train_Mgt"    ] = train_gt    [0].numpy()
        #     debug_dict[f"{i}--2-4 train_Mgt_pre"] = train_gt_pre[0].numpy()
        #     debug_dict[f"{i}--2-5 train_Wgt"    ] = train_gt    [0].numpy()
        #     debug_dict[f"{i}--2-6 train_Wgt_pre"] = train_gt_pre[0].numpy()

        #     fig, ax = plt.subplots(2, 5)
        #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #     ### ord vs pre
        #     in_ord = train_in    [0]
        #     in_pre = train_in_pre[0]
        #     ax[0, 0].imshow(in_ord)
        #     ax[0, 1].imshow(in_pre)

        #     ### W_ord vs W_pre
        #     W_ord = train_gt    [0, ..., :3]
        #     W_pre = train_gt_pre[0, ..., :3]
        #     ax[0, 2].imshow(W_ord)
        #     ax[0, 3].imshow(W_pre)

        #     ### Wx, Wy, Wz 看一下長什麼樣子
        #     Wx_pre = train_gt_pre[0, ..., 0]
        #     Wy_pre = train_gt_pre[0, ..., 1]
        #     Wz_pre = train_gt_pre[0, ..., 2]
        #     ax[1, 0].imshow(Wx_pre)
        #     ax[1, 1].imshow(Wy_pre)
        #     ax[1, 2].imshow(Wz_pre)

        #     ### M_ord vs M_pre
        #     Mgt     = train_gt    [0, ..., 3:4]
        #     Mgt_pre = train_gt_pre[0, ..., 3:4]
        #     ax[1, 3].imshow(Mgt)
        #     ax[1, 4].imshow(Mgt_pre)

        #     ### W_pre * M
        #     W_pre_w_Mgt_pre = W_pre * Mgt_pre
        #     ax[0, 4].imshow(W_pre_w_Mgt_pre)

        #     fig.tight_layout()
        #     if(os.path.isdir(self.tf_data.db_obj.check_train_gt_dir) is False): os.makedirs(self.tf_data.db_obj.check_train_gt_dir)
        #     plt.show()
        #     # plt.savefig(f"{self.tf_data.db_obj.check_train_gt_dir}/" + "%05i" % (i + 1) )
        #     # plt.close()
        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # import os
        # # for i in enumerate(self.tf_data.train_gt_db.ord): pass
        # # print("train_gt_finish")

        # # for i, (test_in, test_in_pre, test_gt, test_gt_pre, name) in enumerate(self.tf_data.test_db_combine.take(5)):
        # for i, (test_in, test_in_pre, test_gt, test_gt_pre, name) in enumerate(self.tf_data.test_db_combine):
        #     debug_dict[f"{i}--1-1 test_in"    ] = test_in
        #     debug_dict[f"{i}--1-2 test_in_pre"] = test_in_pre
        #     debug_dict[f"{i}--1-3 test_gt"    ] = test_gt
        #     debug_dict[f"{i}--1-4 test_gt_pre"] = test_gt_pre

        #     debug_dict[f"{i}--2-1 test_in"    ]  = test_in    [0].numpy()
        #     debug_dict[f"{i}--2-2 test_in_pre"]  = test_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3 test_Mgt"    ] = test_gt    [0].numpy()
        #     debug_dict[f"{i}--2-4 test_Mgt_pre"] = test_gt_pre[0].numpy()
        #     debug_dict[f"{i}--2-5 test_Wgt"    ] = test_gt    [0].numpy()
        #     debug_dict[f"{i}--2-6 test_Wgt_pre"] = test_gt_pre[0].numpy()

        #     fig, ax = plt.subplots(2, 5)
        #     fig.set_size_inches(4.5 * 5, 4.5 * 2)
        #     ### ord vs pre
        #     ax[0, 0].imshow(test_in    [0])
        #     ax[0, 1].imshow(test_in_pre[0])

        #     ### W_ord vs W_pre
        #     ax[0, 2].imshow(test_gt    [0, ..., :3])
        #     ax[0, 3].imshow(test_gt_pre[0, ..., :3])

        #     ### Wx, Wy, Wz 看一下長什麼樣子
        #     ax[1, 0].imshow(test_gt_pre[0, ..., 0])
        #     ax[1, 1].imshow(test_gt_pre[0, ..., 1])
        #     ax[1, 2].imshow(test_gt_pre[0, ..., 2])

        #     ### M_ord vs M_pre
        #     ax[1, 3].imshow(test_gt    [0, ..., 3])
        #     ax[1, 4].imshow(test_gt_pre[0, ..., 3])
        #     fig.tight_layout()
        #     if os.path.isdir(self.tf_data.db_obj.check_test_gt_dir) is False: os.makedirs(self.tf_data.db_obj.check_test_gt_dir)
        #     # plt.show()
        #     plt.savefig(f"{self.tf_data.db_obj.check_test_gt_dir}/" + "%05i" % (i + 1) )
        #     plt.close()
        #########################################################################################################################################
        return self

class tf_Data_in_dis_gt_mask_coord_builder(tf_Data_in_dis_gt_flow_or_wc_builder):
    def build_by_in_dis_gt_mask_coord(self):
        '''
        in_ord: dis_img, shape = (1, ord_h, ord_w, 3), value:0 ~255
        in_pre: dis_img, shape = (1,  db_h,  db_w, 3), value:0 ~  1
        gt_ord: flow,    shape = (1,  db_h , db_w, 3), value:0 ~  1, ch0: mask, ch1:y, ch2:x
        gt_pre: flow,    shape = (1,  db_h , db_w, 3), value:0 ~  1, ch0: mask, ch1:y, ch2:x
        '''
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_C_db_wrong()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.test_gt_db = self.test_gt_factory.build_M_C_db_wrong()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine):
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0].numpy()
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()
        #     # ####################
        #     # ### 確認一下 tight_crop 的效果如何， mask 有沒有被 reflect pad 到
        #     # from step08_b_use_G_generate_0_util import Tight_crop
        #     # from kong_util.build_dataset_combine import Check_dir_exist_and_build
        #     # print(name)
        #     # Mgt_pre     = train_gt_pre[0, ..., 0:1].numpy()
        #     # dis_img_pre = train_in_pre[0].numpy()
        #     # flow_pre    = train_gt_pre[0].numpy()
        #     # tight_crop = Tight_crop(pad_size= 60, resize=(256, 256))
        #     # crop_dis_img_pre , boundary = tight_crop(dis_img_pre, Mgt_pre)
        #     # crop_flow_pre    , boundary = tight_crop(flow_pre   , Mgt_pre)

        #     # plt_img = 3
        #     # fig, ax = plt.subplots(nrows=1, ncols=plt_img, figsize=(5 * plt_img, 5))
        #     # ax[0].imshow(dis_img_pre)
        #     # ax[1].imshow(crop_dis_img_pre)
        #     # ax[2].imshow(crop_flow_pre[..., 0:1])

        #     # import cv2
        #     # debug_dir = r"C:\Users\TKU\Desktop\kong_model2\debug_data\doc3d_tight_crop_check"
        #     # Check_dir_exist_and_build(debug_dir)
        #     # cv2.imwrite(f"{debug_dir}/%06i_dis_img.png"      % i, (dis_img_pre * 255.).astype(np.uint8))
        #     # cv2.imwrite(f"{debug_dir}/%06i_mask.png"         % i, (flow_pre[..., 0:1] * 255.).astype(np.uint8))
        #     # cv2.imwrite(f"{debug_dir}/%06i_dis_img_crop.png" % i, (crop_dis_img_pre.numpy() * 255.).astype(np.uint8))
        #     # cv2.imwrite(f"{debug_dir}/%06i_mask_crop.png"    % i, (crop_flow_pre[..., 0:1].numpy() * 255.).astype(np.uint8))
        #     # plt.show()
        #     # print("finish")
        #     # # tight_crop.reset_jit()  ### 測試看看沒設定 jit_scale 會不會跳出錯誤訊息


        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0]
        #     train_in_pre      = train_in_pre[0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(1, 6)
        #     fig.set_size_inches(30, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt_mask)
        #     ax[3].imshow(train_gt_pre_mask)
        #     ax[4].imshow(train_gt_move_visual)
        #     ax[5].imshow(train_gt_pre_move_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            self.tf_data.see_gt_db   = self.see_gt_factory.build_M_C_db_wrong()

            self.tf_data.see_amount  = get_db_amount(self.tf_data.db_obj.see_in_dir)
            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

            #     from step08_b_use_G_generate_0_util import Tight_crop
            #     gt_mask_pre = see_gt_pre[..., 0:1]
            #     tight_crop = Tight_crop(pad_size=20, resize=(256, 256))
            #     crop_see_in    , _ = tight_crop(see_in    , gt_mask_pre)
            #     crop_see_in_pre, _ = tight_crop(see_in_pre, gt_mask_pre)
            #     crop_see_gt    , _ = tight_crop(see_gt    , gt_mask_pre)
            #     crop_see_gt_pre, _ = tight_crop(see_gt_pre, gt_mask_pre)
            #     tight_crop.reset_jit()  ### 測試看看沒設定 jit_scale 會不會跳出錯誤訊息

            #     debug_dict[f"{i}--5-1 crop_see_in"    ] = crop_see_in
            #     debug_dict[f"{i}--5-2 crop_see_in_pre"] = crop_see_in_pre
            #     debug_dict[f"{i}--5-3 crop_see_gt"    ] = crop_see_gt
            #     debug_dict[f"{i}--5-4 crop_see_gt_pre"] = crop_see_gt_pre

            #     debug_dict[f"{i}--6-1  crop_see_in"     ] = crop_see_in[0].numpy()
            #     debug_dict[f"{i}--6-2  crop_see_in_pre" ] = crop_see_in_pre[0].numpy()
            #     debug_dict[f"{i}--6-3a crop_see_gt_mask"] = crop_see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--6-3b crop_see_gt_move"] = crop_see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--6-4a crop_see_gt_pre_mask"] = crop_see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--6-4b crop_see_gt_pre_move"] = crop_see_gt_pre[0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

class tf_Data_in_wc_gt_flow_builder(tf_Data_in_dis_gt_mask_coord_builder):
    def build_by_in_wc_gt_flow(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_M_W_db_wrong()

        self.tf_data.test_in_db    = self.test_in_factory.build_M_W_db_wrong()
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()

        ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
        self.tf_data.train_in2_db     = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db      = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db.ord  = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in2_db.ord))
        self.tf_data.test_in_db .ord  = tf.data.Dataset.zip((self.tf_data.test_in_db.ord , self.tf_data.test_in2_db .ord))
        self.tf_data.train_in_db.pre  = tf.data.Dataset.zip((self.tf_data.train_in_db.pre, self.tf_data.train_in2_db.pre))
        self.tf_data.test_in_db .pre  = tf.data.Dataset.zip((self.tf_data.test_in_db.pre , self.tf_data.test_in2_db .pre))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount    = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount     = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_C_db_wrong()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_M_C_db_wrong()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     ''' 注意這裡的train_in 有多 dis_img 喔！
        #            train_in[0] 是 wc,      shape=(N, H, W, C)
        #            train_in[1] 是 dis_img, shape=(N, H, W, C)
        #     '''
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in[0]  ### [0]第一個是 取 wc, [1] 是取 dis_img
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0][0]  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     train_in_pre      = train_in_pre[0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(1, 6)
        #     fig.set_size_inches(30, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt_mask)
        #     ax[3].imshow(train_gt_pre_mask)
        #     ax[4].imshow(train_gt_move_visual)
        #     ax[5].imshow(train_gt_pre_move_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_M_W_db_wrong()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
            self.tf_data.see_in2_db = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db.ord  = tf.data.Dataset.zip((self.tf_data.see_in_db.ord, self.tf_data.see_in2_db.ord))
            self.tf_data.see_in_db.pre  = tf.data.Dataset.zip((self.tf_data.see_in_db.pre, self.tf_data.see_in2_db.pre))

            self.tf_data.see_gt_db  = self.see_gt_factory.build_M_C_db_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

    def build_by_in_wc_gt_flow_try_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_M_W_db_right()

        self.tf_data.test_in_db   = self.test_in_factory.build_M_W_db_right()
        self.tf_data.test_name_db = self.test_in_factory.build_name_db()

        ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
        self.tf_data.train_in2_db = self.train_in2_factory.build_img_db()
        self.tf_data.test_in2_db  = self.test_in2_factory .build_img_db()
        self.tf_data.train_in_db.ord = tf.data.Dataset.zip((self.tf_data.train_in_db.ord, self.tf_data.train_in2_db.ord))
        self.tf_data.test_in_db .ord = tf.data.Dataset.zip((self.tf_data.test_in_db .ord, self.tf_data.test_in2_db .ord))
        self.tf_data.train_in_db.pre = tf.data.Dataset.zip((self.tf_data.train_in_db.pre, self.tf_data.train_in2_db.pre))
        self.tf_data.test_in_db .pre = tf.data.Dataset.zip((self.tf_data.test_in_db .pre, self.tf_data.test_in2_db .pre))

        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount  = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_M_C_db_wrong()
        self.tf_data.test_gt_db  = self.test_gt_factory .build_M_C_db_wrong()

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     ''' 注意這裡的train_in 有多 dis_img 喔！
        #            train_in[0] 是 wc,      shape=(N, H, W, C)
        #            train_in[1] 是 dis_img, shape=(N, H, W, C)
        #     '''
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"    ] = train_in[0]  ### [0]第一個是 取 wc, [1] 是取 dis_img
        #     debug_dict[f"{i}--1-2 train_in_pre"] = train_in_pre[0]
        #     debug_dict[f"{i}--1-3 train_gt"    ] = train_gt
        #     debug_dict[f"{i}--1-4 train_gt_pre"] = train_gt_pre

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in[0][0].numpy()  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0][0].numpy()
        #     debug_dict[f"{i}--2-3a train_gt_mask"] = train_gt[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-3b train_gt_move"] = train_gt[0, ..., 1:3].numpy()
        #     debug_dict[f"{i}--2-4a train_gt_pre_mask"] = train_gt_pre[0, ..., 0:1].numpy()
        #     debug_dict[f"{i}--2-4b train_gt_pre_move"] = train_gt_pre[0, ..., 1:3].numpy()

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0][0]  ### [0]第一個是 取 wc, [1] 是取 dis_img， 第二個[0]是取 batch
        #     train_in_pre      = train_in_pre[0][0]
        #     train_gt_mask     = train_gt    [0, ..., 0:1].numpy()
        #     train_gt_pre_mask = train_gt_pre[0, ..., 0:1].numpy()
        #     train_gt_move     = train_gt    [0, ..., 1:3].numpy()
        #     train_gt_pre_move = train_gt_pre[0, ..., 1:3].numpy()
        #     train_gt_move_visual     = method1(train_gt_move[..., 1]    , train_gt_move[..., 0])
        #     train_gt_pre_move_visual = method1(train_gt_pre_move[..., 1], train_gt_pre_move[..., 0])

        #     ### 檢查 gt_mask 是否 == gt_pre_mask
        #     #  # print( "train_gt_mask == train_gt_pre_mask:", (train_gt_mask == train_gt_pre_mask).astype(np.uint8).sum() == train_gt_mask.shape[0] * train_gt_mask.shape[1])

        #     fig, ax = plt.subplots(3, 5)
        #     fig.set_size_inches(25, 15)
        #     ax[0, 0].imshow(train_in)
        #     ax[0, 1].imshow(train_in_pre)
        #     ax[0, 2].imshow(train_in_pre[..., 0:1])
        #     ax[0, 3].imshow(train_in_pre[..., 1:2])
        #     ax[0, 4].imshow(train_in_pre[..., 2:3])

        #     ax[1, 0].imshow(train_gt_mask)
        #     ax[1, 1].imshow(train_gt_pre_mask)
        #     ax[1, 2].imshow(train_gt_move_visual)

        #     ax[2, 0].imshow(train_gt_pre_move[..., 0:1])
        #     ax[2, 1].imshow(train_gt_pre_move[..., 1:2])
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            self.tf_data.see_in_db   = self.see_in_factory.build_M_W_db_right()
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()

            ''' 這裡的 train_in2_db 是 dis_img， 只是為了讓 F 來做 bm_rec 來 visualize 而已， 不會丟進去model裡面， 所以 不需要 train_in2_db_pre 喔！ 更不需要 zip 了'''
            self.tf_data.see_in2_db = self.see_in2_factory .build_img_db()
            self.tf_data.see_in_db.ord  = tf.data.Dataset.zip((self.tf_data.see_in_db.ord, self.tf_data.see_in2_db.ord))
            self.tf_data.see_in_db.pre  = tf.data.Dataset.zip((self.tf_data.see_in_db.pre, self.tf_data.see_in2_db.pre))

            self.tf_data.see_gt_db = self.see_gt_factory.build_M_C_db_wrong()

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！ 這要用sypder開才看的到喔
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.ord.batch(1), self.tf_data.see_in_db.pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.ord.batch(1), self.tf_data.see_gt_db.pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_gt"    ] = see_gt
            #     debug_dict[f"{i}--3-4 see_gt_pre"] = see_gt_pre

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_gt_mask"] = see_gt[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-3b see_gt_move"] = see_gt[0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-4a see_gt_pre_mask"] = see_gt_pre[0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-4b see_gt_pre_move"] = see_gt_pre[0, ..., 1:3].numpy()

            #     # ####### Become doc3d
            #     # ##### 弄成 doc3D 的 448*448， 用任何一個能抓到 flow/wc 的 get_method 都可以寫， 只是上次是用這個寫 update_wc， 就乾脆沿用繼續用這個 get_method 來寫囉～
            #     # from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
            #     # import cv2
            #     # wc      = see_in[0][0, ..., 0:3].numpy()
            #     # dis_img = see_in[1][0].numpy()
            #     # uv      = see_gt[0].numpy()
            #     # mask    = see_gt[0][..., 0:1].numpy()
            #     # ### 視覺化一下
            #     # fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(5 * 6, 5))
            #     # ax[0].imshow(wc[..., 0])
            #     # ax[1].imshow(wc[..., 1])
            #     # ax[2].imshow(wc[..., 2])
            #     # ax[3].imshow(uv[..., 0])
            #     # ax[4].imshow(uv[..., 1])
            #     # ax[5].imshow(uv[..., 2])
            #     # plt.tight_layout()
            #     # # plt.show()

            #     # dis_img_doc3d = dis_img[..., ::-1]  ### tf2 讀出來是 rgb， cv2存圖是bgr， 所以要 rgb2bgr

            #     # wc_ch_min = np.array([0.0        , -0.13532962, -0.080751580]).reshape(1, 1, 3)
            #     # wc_ch_max = np.array([0.039187048,  0.1357405 ,  0.07755918 ]).reshape(1, 1, 3)
            #     # wc_01 = (wc - wc_ch_min) / (wc_ch_max - wc_ch_min)

            #     # wc_ch_min_doc3d = np.array([-0.67187124, -1.2280148, -1.2410645]).reshape(1, 1, 3)
            #     # wc_ch_max_doc3d = np.array([ 0.63452387,  1.2387834,  1.2485291]).reshape(1, 1, 3)
            #     # wc_doc3d = wc_01 * (wc_ch_max_doc3d - wc_ch_min_doc3d) + wc_ch_min_doc3d

            #     # uv_doc3d   = cv2.resize(uv,       (448, 448)).astype(np.float32)
            #     # wc_doc3d   = cv2.resize(wc_doc3d, (448, 448)).astype(np.float32)
            #     # mask_doc3d = cv2.resize(mask,     (448, 448)).reshape(448, 448, 1)
            #     # W_w_M_doc3d = np.concatenate((wc_doc3d, mask_doc3d), axis=-1).astype(np.float32)


            #     # become_doc3d = "F:/kong_model2/debug_data/DB_become_doc3d/see"
            #     # become_doc3d_dis_img_dir    = f"{become_doc3d}/0_dis_img"
            #     # become_doc3d_uv_npy_dir     = f"{become_doc3d}/1_uv-1_npy"
            #     # become_doc3d_uv_knpy_dir    = f"{become_doc3d}/1_uv-3_knpy"
            #     # become_doc3d_wc_npy_dir     = f"{become_doc3d}/2_wc-1_npy"
            #     # become_doc3d_W_w_M_npy_dir  = f"{become_doc3d}/2_wc-4_W_w_M_npy"
            #     # become_doc3d_W_w_M_knpy_dir = f"{become_doc3d}/2_wc-5_W_w_M_knpy"
            #     # Check_dir_exist_and_build(become_doc3d_dis_img_dir    )
            #     # Check_dir_exist_and_build(become_doc3d_uv_npy_dir    )
            #     # Check_dir_exist_and_build(become_doc3d_uv_knpy_dir   )
            #     # Check_dir_exist_and_build(become_doc3d_wc_npy_dir    )
            #     # Check_dir_exist_and_build(become_doc3d_W_w_M_npy_dir )
            #     # Check_dir_exist_and_build(become_doc3d_W_w_M_knpy_dir)
            #     # see_name = "see_%03i" % (i + 1)
            #     # doc3d_dis_img_path    = become_doc3d_dis_img_dir    + "/" + see_name + ".png"
            #     # doc3d_uv_npy_path     = become_doc3d_uv_npy_dir     + "/" + see_name + ".npy"
            #     # doc3d_uv_knpy_path    = become_doc3d_uv_knpy_dir    + "/" + see_name + ".knpy"
            #     # doc3d_wc_npy_path     = become_doc3d_wc_npy_dir     + "/" + see_name + ".npy"
            #     # doc3d_W_w_M_npy_path  = become_doc3d_W_w_M_npy_dir  + "/" + see_name + ".npy"
            #     # doc3d_W_w_M_knpy_path = become_doc3d_W_w_M_knpy_dir + "/" + see_name + ".knpy"

            #     # cv2.imwrite(doc3d_dis_img_path, dis_img_doc3d)

            #     # np.save(doc3d_uv_npy_path, uv_doc3d)
            #     # Save_npy_path_as_knpy(src_path=doc3d_uv_npy_path, dst_path=doc3d_uv_knpy_path)

            #     # np.save(doc3d_wc_npy_path, wc_doc3d)

            #     # np.save(doc3d_W_w_M_npy_path, W_w_M_doc3d)
            #     # Save_npy_path_as_knpy(src_path=doc3d_W_w_M_npy_path, dst_path=doc3d_W_w_M_knpy_path)

            # #     # ##### flow 的 mask 是更新後還不錯的mask， 把他抓出來 更新 W 的 mask
            # #     # from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
            # #     # update_M_dir = "F:/kong_model2/debug_data/DB_update_wc_mask"
            # #     # npy_dir  = update_M_dir + "/npy"
            # #     # knpy_dir = update_M_dir + "/knpy"
            # #     # Check_dir_exist_and_build(update_M_dir)
            # #     # Check_dir_exist_and_build(npy_dir)
            # #     # Check_dir_exist_and_build(knpy_dir)
            # #     # W_w_M  = see_in[0][0].numpy()
            # #     # M_good = see_gt[0, ..., 0:1].numpy()
            # #     # W = W_w_M[..., 0:3]
            # #     # W_w_M_good = np.concatenate([W, M_good], axis=-1)
            # #     # npy_path  = f"{npy_dir}/{i + 1}_W_w_M_good.npy"
            # #     # knpy_path = f"{knpy_dir}/{i + 1}_W_w_M_good.knpy"
            # #     # np.save(npy_path, W_w_M_good)
            # #     # Save_npy_path_as_knpy(src_path=npy_path, dst_path=knpy_path)
            # #     # ### 視覺化一下
            # #     # # fig, ax = plt.subplots(nrows=1, ncols=2)
            # #     # # ax[0].imshow(W_w_M [..., 3])
            # #     # # ax[1].imshow(M_good[..., 0])
            # #     # # plt.show()
            # #     # print("see finish")

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # from kong_util.build_dataset_combine import Check_dir_exist_and_build, Save_npy_path_as_knpy
            # import cv2
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.ord):
            #     ####### Become doc3d
            #     rec_hope = rec_hope_see.numpy()[..., ::-1]  ### tf2 讀出來是 rgb， cv2存圖是bgr， 所以要 rgb2bgr

            #     become_doc3d = "F:/kong_model2/debug_data/DB_become_doc3d/see"
            #     become_doc3d_rec_hope_dir     = f"{become_doc3d}/0_rec_hope"
            #     Check_dir_exist_and_build(become_doc3d_rec_hope_dir    )
            #     see_name = "see_%03i" % (i + 1)
            #     doc3d_rec_hope_path     = become_doc3d_rec_hope_dir     + "/" + see_name + ".png"
            #     cv2.imwrite(doc3d_rec_hope_path, rec_hope)

            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope)
            #     # plt.show()
            #     # plt.close()
            ##########################################################################################################################################
        return self

class tf_Data_in_dis_gt_wc_flow_builder(tf_Data_in_wc_gt_flow_builder):
    def build_by_in_dis_gt_wc_flow_try_mul_M(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        ### train_in
        self.tf_data.train_name_db = self.train_in_factory .build_name_db()
        self.tf_data.train_in_db   = self.train_in_factory .build_img_db()
        ### test_in
        self.tf_data.test_name_db  = self.test_in_factory.build_name_db()
        self.tf_data.test_in_db    = self.test_in_factory.build_img_db()


        ### 設定一下 train_amount，在 shuffle 計算 buffer 大小 的時候會用到， test_amount 忘記會不會用到了， 反正我就copy past 以前的程式碼， 有遇到再來補吧
        self.tf_data.train_amount  = get_db_amount(self.tf_data.db_obj.train_in_dir)
        self.tf_data.test_amount   = get_db_amount(self.tf_data.db_obj.test_in_dir)

        ### train_gt
        self.tf_data.train_gt_db  = self.train_gt_factory .build_M_W_db_right()
        self.tf_data.train_gt2_db = self.train_gt2_factory.build_M_C_db_wrong()
        self.tf_data.train_gt_db.ord = tf.data.Dataset.zip((self.tf_data.train_gt_db.ord, self.tf_data.train_gt2_db.ord))
        self.tf_data.train_gt_db.pre = tf.data.Dataset.zip((self.tf_data.train_gt_db.pre, self.tf_data.train_gt2_db.pre))

        ### test_gt
        self.tf_data.test_gt_db  = self.test_gt_factory .build_M_W_db_right()
        self.tf_data.test_gt2_db = self.test_gt2_factory.build_M_C_db_wrong()
        self.tf_data.test_gt_db.ord = tf.data.Dataset.zip((self.tf_data.test_gt_db.ord, self.tf_data.test_gt2_db.ord    ))
        self.tf_data.test_gt_db.pre = tf.data.Dataset.zip((self.tf_data.test_gt_db.pre, self.tf_data.test_gt2_db.pre))
        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        ##########################################################################################################################################
        ### 勿刪！用來測試寫得對不對！
        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre, name) in enumerate(self.tf_data.train_db_combine.take(3)):
        #     ''' 注意這裡的train_in 有多 dis_img 喔！
        #            train_in[0] 是 wc,      shape=(N, H, W, C)
        #            train_in[1] 是 dis_img, shape=(N, H, W, C)
        #     '''
        #     # if(  i == 0 and self.tf_data.train_shuffle is True) : print("first shuffle finish, cost time:"   , time.time() - start_time)
        #     # elif(i == 0 and self.tf_data.train_shuffle is False): print("first no shuffle finish, cost time:", time.time() - start_time)
        #     debug_dict[f"{i}--1-1 train_in"      ] = train_in
        #     debug_dict[f"{i}--1-2 train_in_pre"  ] = train_in_pre
        #     debug_dict[f"{i}--1-3 train_gt_W"    ] = train_gt[0]      ### [0]是 取 wc, [1] 是取 flow
        #     debug_dict[f"{i}--1-4 train_gt_W_pre"] = train_gt_pre[0]  ### [0]是 取 wc, [1] 是取 flow
        #     debug_dict[f"{i}--1-3 train_gt_F"    ] = train_gt[1]      ### [0]是 取 wc, [1] 是取 flow
        #     debug_dict[f"{i}--1-4 train_gt_F_pre"] = train_gt_pre[1]  ### [0]是 取 wc, [1] 是取 flow

        #     debug_dict[f"{i}--2-1  train_in"     ] = train_in    [0]
        #     debug_dict[f"{i}--2-2  train_in_pre" ] = train_in_pre[0].numpy()
        #     debug_dict[f"{i}--2-3a train_Mgt"]     = train_gt    [0][0, ..., 3:4].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-3b train_Wgt"]     = train_gt    [0][0, ..., 0:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-4a train_Mgt_pre"] = train_gt_pre[0][0, ..., 3:4].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-4b train_Wgt_pre"] = train_gt_pre[0][0, ..., 0:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-5a train_Mgt"]     = train_gt    [1][0, ..., 0:1].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-5b train_Cgt"]     = train_gt    [1][0, ..., 1:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-6a train_Mgt_pre"] = train_gt_pre[1][0, ..., 0:1].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch
        #     debug_dict[f"{i}--2-6b train_Cgt_pre"] = train_gt_pre[1][0, ..., 1:3].numpy()  ### [0]第一個是 取 wc, [1] 是取 flow 第二個[0]是取 batch

        #     # breakpoint()
        #     ### 用 matplot 視覺化， 也可以順便看一下 真的要使用data時， 要怎麼抓資料才正確
        #     train_in          = train_in[0]
        #     train_in_pre      = train_in_pre[0]

        #     train_Mgt_at_W      = train_gt    [0][0, ..., 3:4].numpy()
        #     train_Mgt_pre_at_W  = train_gt_pre[0][0, ..., 3:4].numpy()
        #     train_Wgt           = train_gt    [0][0, ..., 0:3].numpy()
        #     train_Wgt_pre       = train_gt_pre[0][0, ..., 0:3].numpy()

        #     train_Mgt      = train_gt    [1][0, ..., 0:1].numpy()
        #     train_Mgt_pre  = train_gt_pre[1][0, ..., 0:1].numpy()
        #     train_Cgt      = train_gt    [1][0, ..., 1:3].numpy()
        #     train_Cgt_pre  = train_gt_pre[1][0, ..., 1:3].numpy()

        #     train_Fgt_visual     = method1(train_Cgt[..., 1]    , train_Cgt[..., 0])
        #     train_Fgt_pre_visual = method1(train_Cgt_pre[..., 1], train_Cgt_pre[..., 0])

        #     fig, ax = plt.subplots(3, 4)
        #     fig.set_size_inches(20, 15)
        #     ax[0, 0].imshow(train_in)
        #     ax[0, 1].imshow(train_in_pre)

        #     ax[1, 0].imshow(train_Mgt_at_W)
        #     ax[1, 1].imshow(train_Mgt_pre_at_W)
        #     ax[1, 2].imshow(train_Wgt)
        #     ax[1, 3].imshow(train_Wgt_pre)

        #     ax[2, 0].imshow(train_Mgt)
        #     ax[2, 1].imshow(train_Mgt_pre)
        #     ax[2, 2].imshow(train_Fgt_visual)
        #     ax[2, 3].imshow(train_Fgt_pre_visual)
        #     fig.tight_layout()
        #     plt.show()

        ##########################################################################################################################################
        if(self.tf_data.db_obj.have_see):
            ### see_in
            self.tf_data.see_name_db = self.see_in_factory.build_name_db()
            self.tf_data.see_in_db   = self.see_in_factory.build_img_db()

            ### see_gt
            self.tf_data.see_gt_db  = self.see_gt_factory .build_M_W_db_right()
            self.tf_data.see_gt2_db = self.see_gt2_factory.build_M_C_db_wrong()
            self.tf_data.see_gt_db.ord = tf.data.Dataset.zip((self.tf_data.see_gt_db.ord, self.tf_data.see_gt2_db.ord))
            self.tf_data.see_gt_db.pre = tf.data.Dataset.zip((self.tf_data.see_gt_db.pre, self.tf_data.see_gt2_db.pre))

            self.tf_data.see_amount    = get_db_amount(self.tf_data.db_obj.see_in_dir)

            ###########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # for i, (see_in, see_in_pre, see_gt, see_gt_pre) in enumerate(tf.data.Dataset.zip((self.tf_data.see_in_db.batch(1), self.tf_data.see_in_db_pre.batch(1),
            #                                                                                   self.tf_data.see_gt_db.batch(1), self.tf_data.see_gt_db_pre.batch(1)))):
            #     debug_dict[f"{i}--3-1 see_in"    ] = see_in
            #     debug_dict[f"{i}--3-2 see_in_pre"] = see_in_pre
            #     debug_dict[f"{i}--3-3 see_Wgt"    ] = see_gt    [0]
            #     debug_dict[f"{i}--3-4 see_Wgt_pre"] = see_gt_pre[0]
            #     debug_dict[f"{i}--3-5 see_Fgt"    ] = see_gt    [1]
            #     debug_dict[f"{i}--3-6 see_Fgt_pre"] = see_gt_pre[1]

            #     debug_dict[f"{i}--4-1  see_in"     ] = see_in[0].numpy()
            #     debug_dict[f"{i}--4-2  see_in_pre" ] = see_in_pre[0].numpy()
            #     debug_dict[f"{i}--4-3a see_Mgt"]     = see_gt    [0][0, ..., 3:4].numpy()
            #     debug_dict[f"{i}--4-3b see_Wgt"]     = see_gt    [0][0, ..., 0:3].numpy()
            #     debug_dict[f"{i}--4-4a see_Mgt_pre"] = see_gt_pre[0][0, ..., 3:4].numpy()
            #     debug_dict[f"{i}--4-4b see_Wgt_pre"] = see_gt_pre[0][0, ..., 0:3].numpy()
            #     debug_dict[f"{i}--4-5a see_Mgt"]     = see_gt    [1][0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-5b see_Cgt"]     = see_gt    [1][0, ..., 1:3].numpy()
            #     debug_dict[f"{i}--4-6a see_Mgt_pre"] = see_gt_pre[1][0, ..., 0:1].numpy()
            #     debug_dict[f"{i}--4-6b see_Cgt_pre"] = see_gt_pre[1][0, ..., 1:3].numpy()

        if(self.tf_data.db_obj.have_rec_hope):
            self.tf_data.rec_hope_train_db = self.rec_hope_train_factory.build_img_db()
            self.tf_data.rec_hope_test_db  = self.rec_hope_test_factory .build_img_db()
            self.tf_data.rec_hope_see_db   = self.rec_hope_see_factory  .build_img_db()


            self.tf_data.rec_hope_train_amount = get_db_amount(self.tf_data.db_obj.rec_hope_train_dir)
            self.tf_data.rec_hope_test_amount  = get_db_amount(self.tf_data.db_obj.rec_hope_test_dir)
            self.tf_data.rec_hope_see_amount   = get_db_amount(self.tf_data.db_obj.rec_hope_see_dir)

            ##########################################################################################################################################
            ### 勿刪！用來測試寫得對不對！
            # import matplotlib.pyplot as plt
            # for i, rec_hope_see in enumerate(self.tf_data.rec_hope_see_db.pre.take(5)):
            #     fig, ax = plt.subplots(nrows=1, ncols=1)
            #     ax.imshow(rec_hope_see[0])
            #     plt.show()
            #     plt.close()
            ##########################################################################################################################################
        return self

class tf_Data_in_img_gt_mask_builder(tf_Data_in_dis_gt_wc_flow_builder):
    def build_by_in_img_gt_mask(self):
        ##########################################################################################################################################
        ### 整理程式碼後發現，所有模型的 輸入都是 dis_img呀！大家都一樣，寫成一個function給大家call囉， 會建立 train_in_img_db 和 test_in_img_db
        self._build_train_test_in_img_db()


        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.train_gt_db = self.train_gt_factory.build_mask_db()

        ### 拿到 gt_masks_db 的 train dataset，從 檔名 → tensor
        self.tf_data.test_gt_db  = self.test_gt_factory.build_mask_db()

        print("self.tf_data.train_in_db.ord", self.tf_data.train_in_db.ord)
        print("self.tf_data.train_in_db.pre", self.tf_data.train_in_db.pre)
        print("self.tf_data.train_gt_db.ord", self.tf_data.train_gt_db.ord)
        print("self.tf_data.train_gt_db.pre", self.tf_data.train_gt_db.pre)

        ##########################################################################################################################################
        ### 整理程式碼後發現，train_in,gt combine 和 test_in,gt combine 及 之後的shuffle 大家都一樣，寫成一個function給大家call囉
        self._train_in_gt_and_test_in_gt_combine_then_train_shuffle()

        # import matplotlib.pyplot as plt
        # from util import method1
        # for i, (train_in, train_in_pre, train_gt, train_gt_pre) in enumerate(self.tf_data.train_db_combine):
        #     # print(train_in.numpy().shape)       ### (10, 768, 768, 3)
        #     train_in     = train_in[0]          ### 值 0  ~ 255
        #     train_in_pre = train_in_pre[0]      ### 值 0. ~ 1.
        #     print("train_in", train_in.numpy().dtype)       ### uint8
        #     print("train_in", train_in.numpy().shape)       ### (h, w, 3)
        #     print("train_in", train_in.numpy().min())       ### 0
        #     print("train_in", train_in.numpy().max())       ### 255
        #     print("train_in_pre", train_in_pre.numpy().dtype)   ### float32
        #     print("train_in_pre", train_in_pre.numpy().shape)   ### (h, w, 3)
        #     print("train_in_pre", train_in_pre.numpy().min())   ### 0.0
        #     print("train_in_pre", train_in_pre.numpy().max())   ### 1.0

        #     # print(train_gt.numpy().shape)       ### (10, 768, 768, 3)
        #     train_gt     = train_gt[0]          ### 值 0. ~ 1.
        #     train_gt_pre = train_gt_pre[0]      ### 值 0. ~ 1.
        #     print("train_gt", train_gt.numpy().dtype)       ### float32
        #     print("train_gt", train_gt.numpy().shape)       ### (h, w, 3)
        #     print("train_gt", train_gt.numpy().min())       ### 0.0
        #     print("train_gt", train_gt.numpy().max())       ### 1.0
        #     print("train_gt_pre", train_gt_pre.numpy().dtype)   ### float32
        #     print("train_gt_pre", train_gt_pre.numpy().min())   ### 0.0
        #     print("train_gt_pre", train_gt_pre.numpy().max())   ### 1.0


        #     fig, ax = plt.subplots(1, 4)
        #     fig.set_size_inches(15, 5)
        #     ax[0].imshow(train_in)
        #     ax[1].imshow(train_in_pre)
        #     ax[2].imshow(train_gt)
        #     ax[3].imshow(train_gt_pre)
        #     plt.show()

        '''
        還沒弄see
        '''
        return self

class tf_Data_builder(tf_Data_in_img_gt_mask_builder):
    def build(self):
        print(f"TF_data_builder build finish")
        return self.tf_data

########################################################################################################################################
### 因為我們還需要根據 使用的model(其實就是看model_name) 來決定如何resize，所以就不在這邊 先建構好 許多tf_data物件囉！

if(__name__ == "__main__"):
    from step09_d_KModel_builder_combine_step789 import MODEL_NAME, KModel_builder
    from step06_a_datas_obj import *

    start_time = time.time()

    # db_obj = Dataset_builder().set_basic(DB_C.type5c_real_have_see_no_bg_gt_color, DB_N.no_bg_gt_gray3ch, DB_GM.in_dis_gt_ord, h=472, w=304).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="bmp", db_gt_range=Range(0, 255)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.rect).build_by_model_name()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=batch_size-1, train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    # db_obj = Dataset_builder().set_basic(DB_C.type6_h_384_w_256_smooth_curl_fold_and_page, DB_N.smooth_complex_page_more_like_move_map, DB_GM.in_dis_gt_move_map, h=384, w=256).set_dir_by_basic().set_in_gt_format_and_range(in_format="bmp", db_in_range=Range(0, 255), gt_format="...", db_gt_range=Range(...)).set_detail(have_train=True, have_see=True).build()
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.unet).build_unet()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize( model_obj.model_name).build_by_db_get_method().build()

    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    ''' mask_flow 3ch合併 的形式'''
    # db_obj = Dataset_builder().set_basic(DB_C.type8_blender                      , DB_N.blender_os_hw768      , DB_GM.in_dis_gt_flow, h=768, w=768).set_dir_by_basic().set_in_gt_format_and_range(in_format="png", db_in_range=Range(0, 255), gt_format="knpy", db_gt_range=Range(0, 1), rec_hope_format="jpg", db_rec_hope_range=Range(0, 255)).set_detail(have_train=True, have_see=True, have_rec_hope=True).build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(-1, 1), use_gt_range=Range(-1, 1), use_rec_hope_range=Range(0, 255)).build_by_db_get_method().build()

    '''in_img, gt_mask'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_try_segmentation.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type9_mask_flow_have_bg_dtd_hdr_mix_and_paper.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=10 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet).hook_build_and_gen_op()
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_wc_flow_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_kong_doc3d_in_W_and_I_gt_F.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=True).set_img_resize(( 256, 256) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    db_obj = type8_blender_kong_doc3d_in_I_gt_MC.build()
    print(db_obj)
    model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 448, 448) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    # print("here")
    # img1 = tf_data.train_db_combine.take(10)


    ### 雖然來源相同， 不過兩個iter是獨立運作的
    # db1 = tf_data.train_db_combine
    # db2 = tf_data.train_db_combine

    # iter1 = iter(db1)
    # iter2 = iter(db2)

    # data1 = next(iter1)
    # data2 = next(iter2)
    # plt.figure()
    # plt.imshow(data1[0][1][0][..., :3])
    # plt.figure()
    # plt.imshow(data2[0][1][0][..., :3])
    # plt.show()
    # print("data1 == data2", data1)


    # print(len(img1))
    # print("here~ ", img1)
    # img2 = tf_data.train_db_combine.take(10)
    # print("here~~", img2)
    # ### 有shuffle蒔 從這裡知道 take 不花時間， 有點像定義動作的概念

    # ### 有shuffle蒔 從下面知道， 真正用for去取資料的時候才花時間～ 用for 才是真正去取資料喔！ take()不是， take()只是定義動作而已～
    # plt.figure()
    # for data in img1:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here1")

    # plt.figure()
    # for data in img2:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here2")

    # plt.figure()
    # for data in img2:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here3")
    # it1 = iter(img1)
    # it2 = iter(img2)

    # data = next(it2)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # data = next(it2)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # data = next(it1)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])

    # plt.figure()
    # for data in img1:
    #     plt.imshow(data[0][1][0][..., :3])
    #     break
    # print("here1")

    # plt.figure()
    # it1 = iter(img1)
    # data = next(it1)
    # plt.figure()
    # plt.imshow(data[0][1][0][..., :3])
    # print("here1")

    # i = 0
    # it1 = iter(img1.repeat(2))
    # while(True):
    #     data = next(it1)
    #     i = i + 1
    #     print(i)

    # plt.show()
    ''' mask1ch, flow 2ch合併 的形式'''
    ### 這裡為了debug方便 train_shuffle 設 False喔， 真的在train時應該有設True
    # db_obj = type8_blender_dis_wc_flow_try_mul_M.build()
    # print(db_obj)
    # model_obj = KModel_builder().set_model_name(MODEL_NAME.flow_unet)
    # tf_data = tf_Data_builder().set_basic(db_obj, batch_size=1 , train_shuffle=False).set_img_resize(( 512, 512) ).set_data_use_range(use_in_range=Range(0, 1), use_gt_range=Range(0, 1)).build_by_db_get_method().build()

    print(time.time() - start_time)
    print("finish")
