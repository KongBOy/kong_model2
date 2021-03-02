from step11_c_result_instance import compress_results, rec_bm_results

if(__name__ == "__main__"):
    # for result in compress_results:
    #     result.save_all_single_see_as_matplot_visual_multiprocess(add_loss=True)

    for rec_bm_result in rec_bm_results:
        rec_bm_result.save_all_single_see_as_matplot_bm_rec_visual(start_index=0, amount=12, add_loss=False, bgr2rgb=True, single_see_multiprocess=True)