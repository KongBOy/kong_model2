from step11_a1_1_see_npy_to_npz import See_npy_to_npz
from step11_a1_2_see_flow       import See_flow_visual
from step11_a1_3_see_bm_rec     import See_bm_rec
from step11_a1_4_see_rec_metric import See_rec_metric
from step11_a1_5_see_mask       import See_mask
from step11_a1_6_see_wc         import See_wc
# class See(See_flow_visual, See_rec_metric, See_mask):
class See(See_flow_visual, See_npy_to_npz, See_bm_rec, See_rec_metric, See_mask, See_wc):
    def __init__(self, result_obj, see_name):
        super(See, self).__init__(result_obj, see_name)


    def _rename_wrong_to_right_path(self, wrong_file_path, wrong_word, right_word,  print_msg=False):
        import os
        import shutil
        '''
        舉例：
            wrong_file_path  : C:/Users/CVML/Desktop/kong_model2/data_dir/result/7_mask_unet/5_2_mae_block1_45678l/type8_blender_os_book-2_L2_ch001-flow_unet2-block1_L2_ch001_mae_s001-20211119_013628_copy/see_001-real/0b-gt_a_gt_flow.jpg
            name_wrong_word : 0b-gt_a_gt_flow
            name_right_word : 0b-gt_b_gt_flow
        '''
        wrong_name_dir, wrong_name = os.path.split(wrong_file_path)

        if(os.path.exists(wrong_file_path)):
            right_file_name = wrong_name.replace(wrong_word, right_word)
            right_file_path = wrong_name_dir + "/" + right_file_name
            shutil.move(wrong_file_path, right_file_path)
            if(print_msg):
                print("wrong_file_path:", wrong_file_path , "rename to" )
                print("right_file_path:", right_file_path , "finish~")
            return right_file_path
        else:
            print(f"{wrong_file_path} 檔案不存在，也許已經rename過了， 或本來就不存在")
            return None

    def Wrong_flow_rename_to_coord(self):
        import shutil
        import numpy as np
        import cv2
        import tensorflow as tf
        '''
        存錯， 沒有 coord 和 flow 區分的概念時寫的錯的東西， 要把 coord 和 mask concat 起來才是 flow 喔！
        '''
        ### See_method 第二部分：取得see資訊
        self.get_see_base_info()  ### 取得 結果內的 某個see資料夾 內的所有影像 檔名 和 數量
        self.get_flow_info()
        self.get_npy_info()

        ### fake_see 補過來
        if("real" in self.see_name):
            fake_see_path = f"C:/Users/CVML/Desktop/kong_model2/data_dir/datasets/type8_blender_os_book/blender_os_and_paper_hw512_have_dtd_hdr_mix_bg/see/fake_see/1_npy/0_{self.see_name}_fakeGT.npy"
            shutil.copy(fake_see_path, self.flow_gt_npy_path)
        elif(self.see_name == "see_008-train"):
            db_see_path = f"C:/Users/CVML/Desktop/kong_model2/data_dir/datasets/type8_blender_os_book/blender_os_and_paper_hw512_have_dtd_hdr_mix_bg/see/flows/1_uv-0904.knpy"
            flow = tf.io.read_file(db_see_path)
            flow = tf.io.decode_raw(flow , tf.float32)
            flow = tf.reshape(flow, [512, 512, 3])
            flow = flow.numpy()
            np.save(self.flow_gt_npy_path, flow)
        elif(self.see_name == "see_009-test"):
            db_see_path = f"C:/Users/CVML/Desktop/kong_model2/data_dir/datasets/type8_blender_os_book/blender_os_and_paper_hw512_have_dtd_hdr_mix_bg/see/flows/1_uv-0935.knpy"
            flow = tf.io.read_file(db_see_path)
            flow = tf.io.decode_raw(flow , tf.float32)
            flow = tf.reshape(flow, [512, 512, 3])
            flow = flow.numpy()
            np.save(self.flow_gt_npy_path, flow)
        elif(self.see_name == "see_010-test"):
            db_see_path = f"C:/Users/CVML/Desktop/kong_model2/data_dir/datasets/type8_blender_os_book/blender_os_and_paper_hw512_have_dtd_hdr_mix_bg/see/flows/1_uv-0997.knpy"
            flow = tf.io.read_file(db_see_path)
            flow = tf.io.decode_raw(flow , tf.float32)
            flow = tf.reshape(flow, [512, 512, 3])
            flow = flow.numpy()
            np.save(self.flow_gt_npy_path, flow)

        in_img_src_dir = "C:/Users/CVML/Desktop/kong_model2/data_dir/datasets/type8_blender_os_book/blender_os_and_paper_hw512_have_dtd_hdr_mix_bg/see/dis_imgs"
        in_img_src_path = ""
        if  (self.see_name == "see_001-real"):  in_img_src_path = in_img_src_dir +  "/0_3_1 black_bg IMG_6105.png"
        elif(self.see_name == "see_002-real"):  in_img_src_path = in_img_src_dir +  "/0_3_2 black_bg IMG_6107.png"
        elif(self.see_name == "see_003-real"):  in_img_src_path = in_img_src_dir +  "/0_3_3 black_bg IMG_6108.png"
        elif(self.see_name == "see_004-real"):  in_img_src_path = in_img_src_dir +  "/0_3_4 black_bg IMG_6109.png"
        elif(self.see_name == "see_008-train"): in_img_src_path = in_img_src_dir +  "/test_0_image-0904.png"
        elif(self.see_name == "see_009-test"):  in_img_src_path = in_img_src_dir +  "/test_0_image-0935.png"
        elif(self.see_name == "see_010-test"):  in_img_src_path = in_img_src_dir +  "/test_0_image-0997.png"

        in_img_dst_dir  = self.see_read_dir
        in_img_dst_path = in_img_dst_dir + "/0a-in_img.jpg"
        in_img = cv2.imread(in_img_src_path)
        in_img_resize = cv2.resize(in_img, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(in_img_dst_path, in_img_resize)
        # shutil.copy(in_img_src_path, in_img_dst_path)

        ### 弄出： "0b-gt_a_gt_mask.npy"
        ### 弄出： "0b-gt_a_gt_mask.jpg"
        gt_flow = np.load(self.flow_gt_npy_path)
        # print("gt_flow.shape:", gt_flow.shape)
        gt_mask = gt_flow[..., 0:1]
        np.save    (f"{self.flow_v_read_dir}/0b-gt_a_gt_mask", gt_mask)
        cv2.imwrite(f"{self.flow_v_read_dir}/0b-gt_a_gt_mask.jpg", (gt_mask * 255).astype(np.uint8))
        # print("gt_mask.max()", gt_mask.max())


        ### "0b-gt_a_gt_flow" 改成 "0b-gt_b_gt_flow.npy"
        _              = self._rename_wrong_to_right_path(wrong_file_path=self.flow_gt_npy_path,   wrong_word="0b-gt_a_gt_flow", right_word="0b-gt_b_gt_flow")
        right_jpg_path = self._rename_wrong_to_right_path(wrong_file_path=self.gt_flow_jpg_path, wrong_word="0b-gt_a_gt_flow", right_word="0b-gt_b_gt_flow")
        _              = self._rename_wrong_to_right_path(wrong_file_path=f"{self.see_read_dir}/0a-in_gt_mask.jpg"  , wrong_word="0a-in_gt_mask.jpg", right_word="0a2-in_gt_mask.jpg")
        _              = self._rename_wrong_to_right_path(wrong_file_path=f"{self.see_read_dir}/0a-in_img.jpg"      , wrong_word="0a-in_img.jpg",     right_word="0a1-in_img.jpg")
        from step08_b_use_G_generate_0_util import flow_or_coord_visual_op
        gt_flow_visual = flow_or_coord_visual_op(gt_flow)
        cv2.imwrite(right_jpg_path, gt_flow_visual)


        ### epoch_flow
        for go_npy, coord_npy_epoch_path in enumerate(self.npy_read_paths):  ### 雖然檔名是flow 但其實是 coord 喔！
            coord = np.load(coord_npy_epoch_path)
            if(coord.shape[2] == 2):
                flow = np.concatenate((gt_mask, coord), axis=2)
                flow_visual = flow_or_coord_visual_op(data=flow)
                cv2.imwrite(self.flow_ep_jpg_read_paths[go_npy], flow_visual.astype(np.uint8))
                np.save(coord_npy_epoch_path, flow)

            else:
                print("coord.shape[2]:", coord.shape[2], "可能已經 變成flow了喔，pass")




if(__name__ == "__main__"):
    from step0_access_path import Result_Read_Path, Result_Write_Path
    # try_npy_to_npz = See( result_read_dir=Result_Read_Path + "result/5_14_flow_unet/type8_blender_os_book-5_14_3b_4-20210306_231628-flow_unet-ch32_bn_16", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=Result_Read_Path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=Result_Read_Path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_005-train")
    # try_npy_to_npz.npy_to_npz_comapre()
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)

    test_see = See(result_read_dir=Result_Read_Path + "result/5_14_flow_unet/type8_blender_os_book-testest", result_write_dir=Result_Write_Path + "result/5_14_flow_unet/type8_blender_os_book-testest", see_name="see_001-real")
    test_see.Calculate_SSIM_LD(epoch=0, single_see_core_amount=8)
