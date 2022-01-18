import Exps_7_flow_unet.I_w_Mgt_to_C               .mae_s001.a_normal.step10_a as I_w_Mgt_to_C
import Exps_7_flow_unet.I_w_Mgt_to_Cx_Cy           .mae_s001.a_normal.step10_a as I_w_Mgt_to_Cx_Cy
import Exps_7_flow_unet.I_w_Mgt_to_Cx_Cy_bottle_div.mae_s001.a_normal.step10_a as I_w_Mgt_to_Cx_Cy_bottle_div
import Exps_8_multi_unet.comb_multi_try.I_w_Mgt_to_Cx_Cy                           .a_normal.step10_a as I_w_Mgt_to_Cx_Cy_2unet

I_w_Mgt_to_C_compare_L5 = [
    [
        I_w_Mgt_to_C.L5_ch128.build().result_obj,
        I_w_Mgt_to_C.L5_ch064.build().result_obj,
        I_w_Mgt_to_C.L5_ch032.build().result_obj,
        I_w_Mgt_to_C.L5_ch016.build().result_obj,
        I_w_Mgt_to_C.L5_ch008.build().result_obj,
        I_w_Mgt_to_C.L5_ch004.build().result_obj,
        I_w_Mgt_to_C.L5_ch002.build().result_obj,
        I_w_Mgt_to_C.L5_ch001.build().result_obj,
    ],
    [
        I_w_Mgt_to_Cx_Cy.L5_ch128.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch064.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch032.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch016.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch008.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch004.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch002.build().result_obj,
        I_w_Mgt_to_Cx_Cy.L5_ch001.build().result_obj,
    ],
    [
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch128.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch064.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch032.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch016.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch008.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch004.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch002.build().result_obj,
        I_w_Mgt_to_Cx_Cy_bottle_div.L5_ch001.build().result_obj,
    ],
    [
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch128_and_I_to_Cy_L5_ch128_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch064_and_I_to_Cy_L5_ch064_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch032_and_I_to_Cy_L5_ch032_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch016_and_I_to_Cy_L5_ch016_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch008_and_I_to_Cy_L5_ch008_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch004_and_I_to_Cy_L5_ch004_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch002_and_I_to_Cy_L5_ch002_ep060.build().result_obj,
        I_w_Mgt_to_Cx_Cy_2unet.I_to_Cx_L5_ch001_and_I_to_Cy_L5_ch001_ep060.build().result_obj,
    ],
]
