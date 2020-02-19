def distrore_info_log(log_dir, index, row, col, distorte_times, x, y, move_x, move_y, curve_type, alpha, ):
    str_template = \
"\
x=%i\n\
y=%i\n\
move_x=%i\n\
move_y=%i\n\
curve_type=%s\n\
alpha=%i\n\
\n\
"

    with open(log_dir + "/" + "distort_info_%06i-row=%i,col=%i,distorte_times=%i.txt"%(index,row,col,distorte_times),"a") as file_log:
        file_log.write(str_template%(x, y, move_x, move_y, curve_type, alpha))
