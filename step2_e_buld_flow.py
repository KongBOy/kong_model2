




### 平滑多一點 384*256_1500張
dst_dir = "step2_build_flow_h=384,w=256_smooth_curl+fold"
row=384
col=256
amount=450
distort_rand(dst_dir=dst_dir, start_index=amount*0, amount=amount, row=row, col=col,distort_time=1, curl_probability=1.0, move_x_thresh=40, move_y_thresh=55, smooth=True )
distort_rand(dst_dir=dst_dir, start_index=amount*1, amount=amount, row=row, col=col,distort_time=1, curl_probability=0.0, move_x_thresh=40, move_y_thresh=55, smooth=True )
