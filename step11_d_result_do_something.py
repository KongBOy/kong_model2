from step11_c_result_instance import compress_results,  rec_bm_results

if(__name__ == "__main__"):
    ### matplot_visual   ### 覺得我也不會看所以就省了ˊ口ˋ
    # for result in compress_results:
    #     result.save_all_single_see_as_matplot_visual_multiprocess(add_loss=True)

    ### bm, rec，會順便把 npy 轉 npz 喔！
    for rec_bm_result in rec_bm_results:
        # rec_bm_result.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)

        # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=7, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
        # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=8, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
        # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=9, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
        # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=10, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)
        # rec_bm_result.save_single_see_as_matplot_bm_rec_visual(see_num=11, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)


        rec_bm_result.save_all_single_see_as_matplot_bm_rec_visual_at_final_epoch(add_loss=False, bgr2rgb=True)
