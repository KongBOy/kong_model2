import mask_5_2_mae_block1_45678l_M_to_C.step10_a as mae_block1
#################################################################################################################################################################################################################################################################################################################################################################################################
### L2345678_ord
mae_block1_flow_s001_L2345678 = [
    [mae_block1.L2_ch128_mae_s001.build().result_obj,
     mae_block1.L2_ch064_mae_s001.build().result_obj,
     mae_block1.L2_ch032_mae_s001.build().result_obj,
     mae_block1.L2_ch016_mae_s001.build().result_obj,
     mae_block1.L2_ch008_mae_s001.build().result_obj,
     mae_block1.L2_ch004_mae_s001.build().result_obj,
     mae_block1.L2_ch002_mae_s001.build().result_obj,
     mae_block1.L2_ch001_mae_s001.build().result_obj],

    [mae_block1.L3_ch128_mae_s001.build().result_obj,
     mae_block1.L3_ch064_mae_s001.build().result_obj,
     mae_block1.L3_ch032_mae_s001.build().result_obj,
     mae_block1.L3_ch016_mae_s001.build().result_obj,
     mae_block1.L3_ch008_mae_s001.build().result_obj,
     mae_block1.L3_ch004_mae_s001.build().result_obj,
     mae_block1.L3_ch002_mae_s001.build().result_obj,
     mae_block1.L3_ch001_mae_s001.build().result_obj],

    [mae_block1.L4_ch064_mae_s001.build().result_obj,
     mae_block1.L4_ch032_mae_s001.build().result_obj,
     mae_block1.L4_ch016_mae_s001.build().result_obj,
     mae_block1.L4_ch008_mae_s001.build().result_obj,
     mae_block1.L4_ch004_mae_s001.build().result_obj,
     mae_block1.L4_ch002_mae_s001.build().result_obj,
     mae_block1.L4_ch001_mae_s001.build().result_obj],

    [mae_block1.L5_ch032_mae_s001.build().result_obj,
     mae_block1.L5_ch016_mae_s001.build().result_obj,
     mae_block1.L5_ch008_mae_s001.build().result_obj,
     mae_block1.L5_ch004_mae_s001.build().result_obj,
     mae_block1.L5_ch002_mae_s001.build().result_obj,
     mae_block1.L5_ch001_mae_s001.build().result_obj],

    [mae_block1.L6_ch016_mae_s001.build().result_obj,
     mae_block1.L6_ch008_mae_s001.build().result_obj,
     mae_block1.L6_ch004_mae_s001.build().result_obj,
     mae_block1.L6_ch002_mae_s001.build().result_obj,
     mae_block1.L6_ch001_mae_s001.build().result_obj],

    [mae_block1.L7_ch016_mae_s001.build().result_obj,
     mae_block1.L7_ch008_mae_s001.build().result_obj,
     mae_block1.L7_ch004_mae_s001.build().result_obj,
     mae_block1.L7_ch002_mae_s001.build().result_obj,
     mae_block1.L7_ch001_mae_s001.build().result_obj
    ],
    [mae_block1.L8_ch008_mae_s001.build().result_obj,
     mae_block1.L8_ch004_mae_s001.build().result_obj,
     mae_block1.L8_ch002_mae_s001.build().result_obj,
     mae_block1.L8_ch001_mae_s001.build().result_obj],
]


#################################################################################################################################################################################################################################################################################################################################################################################################
mae_block1_rec_s001_L45678 = [
### L3 以前直接跳過 因為發現幾乎無法 rec #######################################################################
    [mae_block1.L8_ch128_mae_s001.build().result_obj,
     mae_block1.L8_ch064_mae_s001.build().result_obj,
     mae_block1.L8_ch032_mae_s001.build().result_obj,
     mae_block1.L8_ch016_mae_s001.build().result_obj,
     mae_block1.L8_ch008_mae_s001.build().result_obj,
     mae_block1.L8_ch004_mae_s001.build().result_obj,
     mae_block1.L8_ch002_mae_s001.build().result_obj,
     mae_block1.L8_ch001_mae_s001.build().result_obj],

    [
    #  mae_block1.L7_ch128_mae_s001.build().result_obj,   ### L7_ch002 做不起來
     mae_block1.L7_ch064_mae_s001.build().result_obj,
     mae_block1.L7_ch032_mae_s001.build().result_obj,
     mae_block1.L7_ch016_mae_s001.build().result_obj,
     mae_block1.L7_ch008_mae_s001.build().result_obj,
     mae_block1.L7_ch004_mae_s001.build().result_obj,
    #  mae_block1.L7_ch002_mae_s001.build().result_obj,   ### L7_ch002 做不起來
     mae_block1.L7_ch001_mae_s001.build().result_obj
    ],

    [mae_block1.L6_ch128_mae_s001.build().result_obj,
     mae_block1.L6_ch064_mae_s001.build().result_obj,
     mae_block1.L6_ch032_mae_s001.build().result_obj,
     mae_block1.L6_ch016_mae_s001.build().result_obj,
     mae_block1.L6_ch008_mae_s001.build().result_obj,
     mae_block1.L6_ch004_mae_s001.build().result_obj,
     mae_block1.L6_ch002_mae_s001.build().result_obj,
    #  mae_block1.L6_ch001_mae_s001.build().result_obj,  ### L6_ch001 做不起來
     ],


    [mae_block1.L5_ch032_mae_s001.build().result_obj,
     mae_block1.L5_ch016_mae_s001.build().result_obj,
     mae_block1.L5_ch008_mae_s001.build().result_obj,
     mae_block1.L5_ch004_mae_s001.build().result_obj,
     mae_block1.L5_ch002_mae_s001.build().result_obj,
    #  mae_block1.L5_ch001_mae_s001.build().result_obj,  ### L5_ch001 做不起來
     ],


    [mae_block1.L4_ch064_mae_s001.build().result_obj,
     mae_block1.L4_ch032_mae_s001.build().result_obj,
     mae_block1.L4_ch016_mae_s001.build().result_obj,
     mae_block1.L4_ch008_mae_s001.build().result_obj,
     mae_block1.L4_ch004_mae_s001.build().result_obj,
     mae_block1.L4_ch002_mae_s001.build().result_obj,
    #  mae_block1.L4_ch001_mae_s001.build().result_obj,  ### L4_ch001 做不起來
     ],

]

# ### L2345_E_relu
# mae_block1_L2345_E_relu = [
#     [mae_block1.L2_ch128_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch064_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch008_mae_s001_E_relu.build().result_obj],
#     [mae_block1.L3_ch128_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch064_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch008_mae_s001_E_relu.build().result_obj],
#     [mae_block1.L4_ch064_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch008_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch004_mae_s001_E_relu.build().result_obj],
#     [mae_block1.L5_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch008_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch004_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch002_mae_s001_E_relu.build().result_obj],
# ]
# ### L2345_E_lrelu vs E_relu
# mae_block1_L2345__E_lrelu_vs_E_relu = [
#     [mae_block1.L2_ch128_mae_s001.build().result_obj,
#      mae_block1.L2_ch064_mae_s001.build().result_obj,
#      mae_block1.L2_ch032_mae_s001.build().result_obj,
#      mae_block1.L2_ch016_mae_s001.build().result_obj,
#      mae_block1.L2_ch008_mae_s001.build().result_obj],
#     [mae_block1.L2_ch128_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch064_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L2_ch008_mae_s001_E_relu.build().result_obj],

#     [mae_block1.L3_ch128_mae_s001.build().result_obj,
#      mae_block1.L3_ch064_mae_s001.build().result_obj,
#      mae_block1.L3_ch032_mae_s001.build().result_obj,
#      mae_block1.L3_ch016_mae_s001.build().result_obj,
#      mae_block1.L3_ch008_mae_s001.build().result_obj],
#     [mae_block1.L3_ch128_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch064_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L3_ch008_mae_s001_E_relu.build().result_obj],

#     [mae_block1.L4_ch064_mae_s001.build().result_obj,
#      mae_block1.L4_ch032_mae_s001.build().result_obj,
#      mae_block1.L4_ch016_mae_s001.build().result_obj,
#      mae_block1.L4_ch008_mae_s001.build().result_obj,
#      mae_block1.L4_ch004_mae_s001.build().result_obj],
#     [mae_block1.L4_ch064_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch008_mae_s001_E_relu.build().result_obj,
#      mae_block1.L4_ch004_mae_s001_E_relu.build().result_obj],

#     [mae_block1.L5_ch032_mae_s001.build().result_obj,
#      mae_block1.L5_ch016_mae_s001.build().result_obj,
#      mae_block1.L5_ch008_mae_s001.build().result_obj,
#      mae_block1.L5_ch004_mae_s001.build().result_obj,
#      mae_block1.L5_ch002_mae_s001.build().result_obj],
#     [mae_block1.L5_ch032_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch016_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch008_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch004_mae_s001_E_relu.build().result_obj,
#      mae_block1.L5_ch002_mae_s001_E_relu.build().result_obj],
# ]


