from step11_a1_1_see_npy_to_npz import See_npy_to_npz
from step11_a1_2_see_flow       import See_flow_visual
from step11_a1_3_see_bm_rec     import See_bm_rec
from step11_a1_4_see_rec_metric import See_rec_metric
from step11_a1_5_see_mask       import See_mask
# class See(See_flow_visual, See_rec_metric, See_mask):
class See(See_flow_visual, See_npy_to_npz, See_bm_rec, See_rec_metric, See_mask):
    def __init__(self, result_read_dir, result_write_dir, see_name):
        super(See, self).__init__(result_read_dir, result_write_dir, see_name)



if(__name__ == "__main__"):
    from step0_access_path import result_read_path, result_write_path
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_3b_4-20210306_231628-flow_unet-ch32_bn_16", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_001-real")
    # try_npy_to_npz = See( result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-5_14_1_6-20210308_100044-flow_unet-new_shuf_epoch700", see_name="see_005-train")
    # try_npy_to_npz.npy_to_npz_comapre()
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)
    # try_npy_to_npz.Npy_to_npz(multiprocess=True)

    test_see = See(result_read_dir=result_read_path + "result/5_14_flow_unet/type8_blender_os_book-testest", result_write_dir=result_write_path + "result/5_14_flow_unet/type8_blender_os_book-testest", see_name="see_001-real")
    test_see.Calculate_SSIM_LD(epoch=0, single_see_core_amount=8)
