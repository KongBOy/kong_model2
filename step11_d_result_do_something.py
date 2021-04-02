# from step11_c_result_instance import compress_results
from step11_c_result_instance import rec_bm_results
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if(__name__ == "__main__"):
    ### matplot_visual   ### 覺得我也不會看所以就省了ˊ口ˋ
    # for result in compress_results:
    #     result.save_all_single_see_as_matplot_visual_multiprocess(add_loss=True)

    ### bm, rec，會順便把 npy 轉 npz 喔！
    for go_result, rec_bm_result in enumerate(rec_bm_results):
        if(go_result == 0):  ### 這個 if 是給 中途不小心中斷，可以從某個指定的 see_num 開始做，要手動把中斷的放第一個result拉
            # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=0, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=1, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=2, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=3, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=4, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=5, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
            rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=6, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
        else:   ### 中斷的作完，剩下就是作全部囉！
            rec_bm_result.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)


        # rec_bm_result.save_all_single_see_as_matplot_bm_rec_visual_at_final_epoch(add_loss=False, bgr2rgb=True)
