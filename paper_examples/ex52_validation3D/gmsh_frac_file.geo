Geometry.Tolerance = 0.0001;
// Define points
p0 = newp; Point(p0) = {0.5, 0.25, 0.75, 0.04 };
p1 = newp; Point(p1) = {0.5, 0.25, 0.25, 0.04 };
p2 = newp; Point(p2) = {0.5, 0.75, 0.25, 0.04 };
p3 = newp; Point(p3) = {0.5, 0.75, 0.75, 0.04 };
p4 = newp; Point(p4) = {0.0, 0.25, 0.25, 0.04 };
p5 = newp; Point(p5) = {0.0, 0.75, 0.25, 0.04 };
p6 = newp; Point(p6) = {1.0, 0.25, 0.25, 0.04 };
p7 = newp; Point(p7) = {1.0, 0.75, 0.25, 0.04 };
p8 = newp; Point(p8) = {0.0, 0.25, 0.75, 0.04 };
p9 = newp; Point(p9) = {0.0, 0.75, 0.75, 0.04 };
p10 = newp; Point(p10) = {1.0, 0.25, 0.75, 0.04 };
p11 = newp; Point(p11) = {1.0, 0.75, 0.75, 0.04 };
p12 = newp; Point(p12) = {0.0, 1.0, 0.25, 0.04 };
p13 = newp; Point(p13) = {0.5, 1.0, 0.25, 0.04 };
p14 = newp; Point(p14) = {1.0, 1.0, 0.25, 0.04 };
p15 = newp; Point(p15) = {0.0, 1.0, 0.75, 0.04 };
p16 = newp; Point(p16) = {0.5, 1.0, 0.75, 0.04 };
p17 = newp; Point(p17) = {1.0, 1.0, 0.75, 0.04 };
p18 = newp; Point(p18) = {0.0, 0.0, 0.25, 0.04 };
p19 = newp; Point(p19) = {0.5, 0.0, 0.25, 0.04 };
p20 = newp; Point(p20) = {1.0, 0.0, 0.25, 0.04 };
p21 = newp; Point(p21) = {0.0, 0.0, 0.75, 0.04 };
p22 = newp; Point(p22) = {0.5, 0.0, 0.75, 0.04 };
p23 = newp; Point(p23) = {1.0, 0.0, 0.75, 0.04 };
p24 = newp; Point(p24) = {0.0, 0.75, 0.0, 0.04 };
p25 = newp; Point(p25) = {0.5, 0.75, 0.0, 0.04 };
p26 = newp; Point(p26) = {1.0, 0.75, 0.0, 0.04 };
p27 = newp; Point(p27) = {0.0, 0.75, 1.0, 0.04 };
p28 = newp; Point(p28) = {0.5, 0.75, 1.0, 0.04 };
p29 = newp; Point(p29) = {1.0, 0.75, 1.0, 0.04 };
p30 = newp; Point(p30) = {0.0, 0.25, 0.0, 0.04 };
p31 = newp; Point(p31) = {0.5, 0.25, 0.0, 0.04 };
p32 = newp; Point(p32) = {1.0, 0.25, 0.0, 0.04 };
p33 = newp; Point(p33) = {0.0, 0.25, 1.0, 0.04 };
p34 = newp; Point(p34) = {0.5, 0.25, 1.0, 0.04 };
p35 = newp; Point(p35) = {1.0, 0.25, 1.0, 0.04 };
p36 = newp; Point(p36) = {0.0, 0.0, 0.0, 0.04 };
p37 = newp; Point(p37) = {0.0, 0.0, 1.0, 0.04 };
p38 = newp; Point(p38) = {0.0, 1.0, 1.0, 0.04 };
p39 = newp; Point(p39) = {0.0, 1.0, 0.0, 0.04 };
p40 = newp; Point(p40) = {1.0, 0.0, 0.0, 0.04 };
p41 = newp; Point(p41) = {1.0, 0.0, 1.0, 0.04 };
p42 = newp; Point(p42) = {1.0, 1.0, 1.0, 0.04 };
p43 = newp; Point(p43) = {1.0, 1.0, 0.0, 0.04 };
// End of point specification

// Define lines 
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Physical Line("FRACTURE_TIP_0") = {frac_line_0};

frac_line_1 = newl; Line(frac_line_1) = {p0, p3};
Physical Line("FRACTURE_TIP_1") = {frac_line_1};

frac_line_2 = newl; Line(frac_line_2) = {p0, p8};
Physical Line("FRACTURE_TIP_2") = {frac_line_2};

frac_line_3 = newl; Line(frac_line_3) = {p0, p10};
Physical Line("FRACTURE_TIP_3") = {frac_line_3};

frac_line_4 = newl; Line(frac_line_4) = {p0, p22};
Physical Line("FRACTURE_TIP_4") = {frac_line_4};

frac_line_5 = newl; Line(frac_line_5) = {p0, p34};
Physical Line("FRACTURE_TIP_5") = {frac_line_5};

frac_line_6 = newl; Line(frac_line_6) = {p1, p2};
Physical Line("FRACTURE_TIP_6") = {frac_line_6};

frac_line_7 = newl; Line(frac_line_7) = {p1, p4};
Physical Line("FRACTURE_TIP_7") = {frac_line_7};

frac_line_8 = newl; Line(frac_line_8) = {p1, p6};
Physical Line("FRACTURE_TIP_8") = {frac_line_8};

frac_line_9 = newl; Line(frac_line_9) = {p1, p19};
Physical Line("FRACTURE_TIP_9") = {frac_line_9};

frac_line_10 = newl; Line(frac_line_10) = {p1, p31};
Physical Line("FRACTURE_TIP_10") = {frac_line_10};

frac_line_11 = newl; Line(frac_line_11) = {p2, p3};
Physical Line("FRACTURE_TIP_11") = {frac_line_11};

frac_line_12 = newl; Line(frac_line_12) = {p2, p5};
Physical Line("FRACTURE_TIP_12") = {frac_line_12};

frac_line_13 = newl; Line(frac_line_13) = {p2, p7};
Physical Line("FRACTURE_TIP_13") = {frac_line_13};

frac_line_14 = newl; Line(frac_line_14) = {p2, p13};
Physical Line("FRACTURE_TIP_14") = {frac_line_14};

frac_line_15 = newl; Line(frac_line_15) = {p2, p25};
Physical Line("FRACTURE_TIP_15") = {frac_line_15};

frac_line_16 = newl; Line(frac_line_16) = {p3, p9};
Physical Line("FRACTURE_TIP_16") = {frac_line_16};

frac_line_17 = newl; Line(frac_line_17) = {p3, p11};
Physical Line("FRACTURE_TIP_17") = {frac_line_17};

frac_line_18 = newl; Line(frac_line_18) = {p3, p16};
Physical Line("FRACTURE_TIP_18") = {frac_line_18};

frac_line_19 = newl; Line(frac_line_19) = {p3, p28};
Physical Line("FRACTURE_TIP_19") = {frac_line_19};

frac_line_20 = newl; Line(frac_line_20) = {p4, p5};
Physical Line("FRACTURE_BOUNDARY_LINE_20") = {frac_line_20};

frac_line_21 = newl; Line(frac_line_21) = {p4, p8};
Physical Line("FRACTURE_BOUNDARY_LINE_21") = {frac_line_21};

frac_line_22 = newl; Line(frac_line_22) = {p4, p18};
Physical Line("FRACTURE_BOUNDARY_LINE_22") = {frac_line_22};

frac_line_23 = newl; Line(frac_line_23) = {p4, p30};
Physical Line("FRACTURE_BOUNDARY_LINE_23") = {frac_line_23};

frac_line_24 = newl; Line(frac_line_24) = {p5, p9};
Physical Line("FRACTURE_BOUNDARY_LINE_24") = {frac_line_24};

frac_line_25 = newl; Line(frac_line_25) = {p5, p12};
Physical Line("FRACTURE_BOUNDARY_LINE_25") = {frac_line_25};

frac_line_26 = newl; Line(frac_line_26) = {p5, p24};
Physical Line("FRACTURE_BOUNDARY_LINE_26") = {frac_line_26};

frac_line_27 = newl; Line(frac_line_27) = {p6, p7};
Physical Line("FRACTURE_BOUNDARY_LINE_27") = {frac_line_27};

frac_line_28 = newl; Line(frac_line_28) = {p6, p10};
Physical Line("FRACTURE_BOUNDARY_LINE_28") = {frac_line_28};

frac_line_29 = newl; Line(frac_line_29) = {p6, p20};
Physical Line("FRACTURE_BOUNDARY_LINE_29") = {frac_line_29};

frac_line_30 = newl; Line(frac_line_30) = {p6, p32};
Physical Line("FRACTURE_BOUNDARY_LINE_30") = {frac_line_30};

frac_line_31 = newl; Line(frac_line_31) = {p7, p11};
Physical Line("FRACTURE_BOUNDARY_LINE_31") = {frac_line_31};

frac_line_32 = newl; Line(frac_line_32) = {p7, p14};
Physical Line("FRACTURE_BOUNDARY_LINE_32") = {frac_line_32};

frac_line_33 = newl; Line(frac_line_33) = {p7, p26};
Physical Line("FRACTURE_BOUNDARY_LINE_33") = {frac_line_33};

frac_line_34 = newl; Line(frac_line_34) = {p8, p9};
Physical Line("FRACTURE_BOUNDARY_LINE_34") = {frac_line_34};

frac_line_35 = newl; Line(frac_line_35) = {p8, p21};
Physical Line("FRACTURE_BOUNDARY_LINE_35") = {frac_line_35};

frac_line_36 = newl; Line(frac_line_36) = {p8, p33};
Physical Line("FRACTURE_BOUNDARY_LINE_36") = {frac_line_36};

frac_line_37 = newl; Line(frac_line_37) = {p9, p15};
Physical Line("FRACTURE_BOUNDARY_LINE_37") = {frac_line_37};

frac_line_38 = newl; Line(frac_line_38) = {p9, p27};
Physical Line("FRACTURE_BOUNDARY_LINE_38") = {frac_line_38};

frac_line_39 = newl; Line(frac_line_39) = {p10, p11};
Physical Line("FRACTURE_BOUNDARY_LINE_39") = {frac_line_39};

frac_line_40 = newl; Line(frac_line_40) = {p10, p23};
Physical Line("FRACTURE_BOUNDARY_LINE_40") = {frac_line_40};

frac_line_41 = newl; Line(frac_line_41) = {p10, p35};
Physical Line("FRACTURE_BOUNDARY_LINE_41") = {frac_line_41};

frac_line_42 = newl; Line(frac_line_42) = {p11, p17};
Physical Line("FRACTURE_BOUNDARY_LINE_42") = {frac_line_42};

frac_line_43 = newl; Line(frac_line_43) = {p11, p29};
Physical Line("FRACTURE_BOUNDARY_LINE_43") = {frac_line_43};

frac_line_44 = newl; Line(frac_line_44) = {p12, p13};
Physical Line("FRACTURE_BOUNDARY_LINE_44") = {frac_line_44};

frac_line_45 = newl; Line(frac_line_45) = {p12, p15};
Physical Line("DOMAIN_BOUNDARY_45") = {frac_line_45};

frac_line_46 = newl; Line(frac_line_46) = {p12, p39};
Physical Line("DOMAIN_BOUNDARY_46") = {frac_line_46};

frac_line_47 = newl; Line(frac_line_47) = {p13, p14};
Physical Line("FRACTURE_BOUNDARY_LINE_47") = {frac_line_47};

frac_line_48 = newl; Line(frac_line_48) = {p14, p17};
Physical Line("DOMAIN_BOUNDARY_48") = {frac_line_48};

frac_line_49 = newl; Line(frac_line_49) = {p14, p43};
Physical Line("DOMAIN_BOUNDARY_49") = {frac_line_49};

frac_line_50 = newl; Line(frac_line_50) = {p15, p16};
Physical Line("FRACTURE_BOUNDARY_LINE_50") = {frac_line_50};

frac_line_51 = newl; Line(frac_line_51) = {p15, p38};
Physical Line("DOMAIN_BOUNDARY_51") = {frac_line_51};

frac_line_52 = newl; Line(frac_line_52) = {p16, p17};
Physical Line("FRACTURE_BOUNDARY_LINE_52") = {frac_line_52};

frac_line_53 = newl; Line(frac_line_53) = {p17, p42};
Physical Line("DOMAIN_BOUNDARY_53") = {frac_line_53};

frac_line_54 = newl; Line(frac_line_54) = {p18, p19};
Physical Line("FRACTURE_BOUNDARY_LINE_54") = {frac_line_54};

frac_line_55 = newl; Line(frac_line_55) = {p18, p21};
Physical Line("DOMAIN_BOUNDARY_55") = {frac_line_55};

frac_line_56 = newl; Line(frac_line_56) = {p18, p36};
Physical Line("DOMAIN_BOUNDARY_56") = {frac_line_56};

frac_line_57 = newl; Line(frac_line_57) = {p19, p20};
Physical Line("FRACTURE_BOUNDARY_LINE_57") = {frac_line_57};

frac_line_58 = newl; Line(frac_line_58) = {p20, p23};
Physical Line("DOMAIN_BOUNDARY_58") = {frac_line_58};

frac_line_59 = newl; Line(frac_line_59) = {p20, p40};
Physical Line("DOMAIN_BOUNDARY_59") = {frac_line_59};

frac_line_60 = newl; Line(frac_line_60) = {p21, p22};
Physical Line("FRACTURE_BOUNDARY_LINE_60") = {frac_line_60};

frac_line_61 = newl; Line(frac_line_61) = {p21, p37};
Physical Line("DOMAIN_BOUNDARY_61") = {frac_line_61};

frac_line_62 = newl; Line(frac_line_62) = {p22, p23};
Physical Line("FRACTURE_BOUNDARY_LINE_62") = {frac_line_62};

frac_line_63 = newl; Line(frac_line_63) = {p23, p41};
Physical Line("DOMAIN_BOUNDARY_63") = {frac_line_63};

frac_line_64 = newl; Line(frac_line_64) = {p24, p25};
Physical Line("FRACTURE_BOUNDARY_LINE_64") = {frac_line_64};

frac_line_65 = newl; Line(frac_line_65) = {p24, p30};
Physical Line("DOMAIN_BOUNDARY_65") = {frac_line_65};

frac_line_66 = newl; Line(frac_line_66) = {p24, p39};
Physical Line("DOMAIN_BOUNDARY_66") = {frac_line_66};

frac_line_67 = newl; Line(frac_line_67) = {p25, p26};
Physical Line("FRACTURE_BOUNDARY_LINE_67") = {frac_line_67};

frac_line_68 = newl; Line(frac_line_68) = {p26, p32};
Physical Line("DOMAIN_BOUNDARY_68") = {frac_line_68};

frac_line_69 = newl; Line(frac_line_69) = {p26, p43};
Physical Line("DOMAIN_BOUNDARY_69") = {frac_line_69};

frac_line_70 = newl; Line(frac_line_70) = {p27, p28};
Physical Line("FRACTURE_BOUNDARY_LINE_70") = {frac_line_70};

frac_line_71 = newl; Line(frac_line_71) = {p27, p33};
Physical Line("DOMAIN_BOUNDARY_71") = {frac_line_71};

frac_line_72 = newl; Line(frac_line_72) = {p27, p38};
Physical Line("DOMAIN_BOUNDARY_72") = {frac_line_72};

frac_line_73 = newl; Line(frac_line_73) = {p28, p29};
Physical Line("FRACTURE_BOUNDARY_LINE_73") = {frac_line_73};

frac_line_74 = newl; Line(frac_line_74) = {p29, p35};
Physical Line("DOMAIN_BOUNDARY_74") = {frac_line_74};

frac_line_75 = newl; Line(frac_line_75) = {p29, p42};
Physical Line("DOMAIN_BOUNDARY_75") = {frac_line_75};

frac_line_76 = newl; Line(frac_line_76) = {p30, p31};
Physical Line("FRACTURE_BOUNDARY_LINE_76") = {frac_line_76};

frac_line_77 = newl; Line(frac_line_77) = {p30, p36};
Physical Line("DOMAIN_BOUNDARY_77") = {frac_line_77};

frac_line_78 = newl; Line(frac_line_78) = {p31, p32};
Physical Line("FRACTURE_BOUNDARY_LINE_78") = {frac_line_78};

frac_line_79 = newl; Line(frac_line_79) = {p32, p40};
Physical Line("DOMAIN_BOUNDARY_79") = {frac_line_79};

frac_line_80 = newl; Line(frac_line_80) = {p33, p34};
Physical Line("FRACTURE_BOUNDARY_LINE_80") = {frac_line_80};

frac_line_81 = newl; Line(frac_line_81) = {p33, p37};
Physical Line("DOMAIN_BOUNDARY_81") = {frac_line_81};

frac_line_82 = newl; Line(frac_line_82) = {p34, p35};
Physical Line("FRACTURE_BOUNDARY_LINE_82") = {frac_line_82};

frac_line_83 = newl; Line(frac_line_83) = {p35, p41};
Physical Line("DOMAIN_BOUNDARY_83") = {frac_line_83};

frac_line_84 = newl; Line(frac_line_84) = {p36, p40};
Physical Line("DOMAIN_BOUNDARY_84") = {frac_line_84};

frac_line_85 = newl; Line(frac_line_85) = {p37, p41};
Physical Line("DOMAIN_BOUNDARY_85") = {frac_line_85};

frac_line_86 = newl; Line(frac_line_86) = {p38, p42};
Physical Line("DOMAIN_BOUNDARY_86") = {frac_line_86};

frac_line_87 = newl; Line(frac_line_87) = {p39, p43};
Physical Line("DOMAIN_BOUNDARY_87") = {frac_line_87};

// End of line specification 

// Start domain specification
// Start boundary surface specification
frac_loop_25 = newll; 
Line Loop(frac_loop_25) = { frac_line_45, frac_line_51, -frac_line_72, frac_line_71, frac_line_81, -frac_line_61, -frac_line_55, frac_line_56, -frac_line_77, -frac_line_65, frac_line_66, -frac_line_46};
boundary_surface_25 = news; Plane Surface(boundary_surface_25) = {frac_loop_25};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_25") = {boundary_surface_25};
Line{frac_line_20} In Surface{boundary_surface_25};
Line{frac_line_21} In Surface{boundary_surface_25};
Line{frac_line_22} In Surface{boundary_surface_25};
Line{frac_line_23} In Surface{boundary_surface_25};
Line{frac_line_24} In Surface{boundary_surface_25};
Line{frac_line_25} In Surface{boundary_surface_25};
Line{frac_line_26} In Surface{boundary_surface_25};
Line{frac_line_34} In Surface{boundary_surface_25};
Line{frac_line_35} In Surface{boundary_surface_25};
Line{frac_line_36} In Surface{boundary_surface_25};
Line{frac_line_37} In Surface{boundary_surface_25};
Line{frac_line_38} In Surface{boundary_surface_25};

frac_loop_26 = newll; 
Line Loop(frac_loop_26) = { frac_line_48, frac_line_53, -frac_line_75, frac_line_74, frac_line_83, -frac_line_63, -frac_line_58, frac_line_59, -frac_line_79, -frac_line_68, frac_line_69, -frac_line_49};
boundary_surface_26 = news; Plane Surface(boundary_surface_26) = {frac_loop_26};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_26") = {boundary_surface_26};
Line{frac_line_27} In Surface{boundary_surface_26};
Line{frac_line_28} In Surface{boundary_surface_26};
Line{frac_line_29} In Surface{boundary_surface_26};
Line{frac_line_30} In Surface{boundary_surface_26};
Line{frac_line_31} In Surface{boundary_surface_26};
Line{frac_line_32} In Surface{boundary_surface_26};
Line{frac_line_33} In Surface{boundary_surface_26};
Line{frac_line_39} In Surface{boundary_surface_26};
Line{frac_line_40} In Surface{boundary_surface_26};
Line{frac_line_41} In Surface{boundary_surface_26};
Line{frac_line_42} In Surface{boundary_surface_26};
Line{frac_line_43} In Surface{boundary_surface_26};

frac_loop_27 = newll; 
Line Loop(frac_loop_27) = { frac_line_55, frac_line_61, frac_line_85, -frac_line_63, -frac_line_58, frac_line_59, -frac_line_84, -frac_line_56};
boundary_surface_27 = news; Plane Surface(boundary_surface_27) = {frac_loop_27};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_27") = {boundary_surface_27};
Line{frac_line_54} In Surface{boundary_surface_27};
Line{frac_line_57} In Surface{boundary_surface_27};
Line{frac_line_60} In Surface{boundary_surface_27};
Line{frac_line_62} In Surface{boundary_surface_27};

frac_loop_28 = newll; 
Line Loop(frac_loop_28) = { frac_line_45, frac_line_51, frac_line_86, -frac_line_53, -frac_line_48, frac_line_49, -frac_line_87, -frac_line_46};
boundary_surface_28 = news; Plane Surface(boundary_surface_28) = {frac_loop_28};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_28") = {boundary_surface_28};
Line{frac_line_44} In Surface{boundary_surface_28};
Line{frac_line_47} In Surface{boundary_surface_28};
Line{frac_line_50} In Surface{boundary_surface_28};
Line{frac_line_52} In Surface{boundary_surface_28};

frac_loop_29 = newll; 
Line Loop(frac_loop_29) = { frac_line_65, frac_line_77, frac_line_84, -frac_line_79, -frac_line_68, frac_line_69, -frac_line_87, -frac_line_66};
boundary_surface_29 = news; Plane Surface(boundary_surface_29) = {frac_loop_29};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_29") = {boundary_surface_29};
Line{frac_line_64} In Surface{boundary_surface_29};
Line{frac_line_67} In Surface{boundary_surface_29};
Line{frac_line_76} In Surface{boundary_surface_29};
Line{frac_line_78} In Surface{boundary_surface_29};

frac_loop_30 = newll; 
Line Loop(frac_loop_30) = { frac_line_71, frac_line_81, frac_line_85, -frac_line_83, -frac_line_74, frac_line_75, -frac_line_86, -frac_line_72};
boundary_surface_30 = news; Plane Surface(boundary_surface_30) = {frac_loop_30};
Physical Surface("DOMAIN_BOUNDARY_SURFACE_30") = {boundary_surface_30};
Line{frac_line_70} In Surface{boundary_surface_30};
Line{frac_line_73} In Surface{boundary_surface_30};
Line{frac_line_80} In Surface{boundary_surface_30};
Line{frac_line_82} In Surface{boundary_surface_30};

domain_loop = newsl;
Surface Loop(domain_loop) = { boundary_surface_25, boundary_surface_26, boundary_surface_27, boundary_surface_28, boundary_surface_29, boundary_surface_30};
Volume(1) = {domain_loop};
Physical Volume("DOMAIN") = {1};
// End of domain specification

// Start fracture specification
frac_loop_0 = newll; 
Line Loop(frac_loop_0) = { frac_line_0, frac_line_6, frac_line_11, -frac_line_1};
fracture_0 = news; Plane Surface(fracture_0) = {frac_loop_0};
Physical Surface("FRACTURE_0") = {fracture_0};
Surface{fracture_0} In Volume{1};


frac_loop_1 = newll; 
Line Loop(frac_loop_1) = { frac_line_6, frac_line_12, -frac_line_20, -frac_line_7};
auxiliary_surface_1 = news; Plane Surface(auxiliary_surface_1) = {frac_loop_1};
Physical Surface("AUXILIARY_1") = {auxiliary_surface_1};
Surface{auxiliary_surface_1} In Volume{1};


frac_loop_2 = newll; 
Line Loop(frac_loop_2) = { frac_line_6, frac_line_13, -frac_line_27, -frac_line_8};
auxiliary_surface_2 = news; Plane Surface(auxiliary_surface_2) = {frac_loop_2};
Physical Surface("AUXILIARY_2") = {auxiliary_surface_2};
Surface{auxiliary_surface_2} In Volume{1};


frac_loop_3 = newll; 
Line Loop(frac_loop_3) = { frac_line_1, frac_line_16, -frac_line_34, -frac_line_2};
auxiliary_surface_3 = news; Plane Surface(auxiliary_surface_3) = {frac_loop_3};
Physical Surface("AUXILIARY_3") = {auxiliary_surface_3};
Surface{auxiliary_surface_3} In Volume{1};


frac_loop_4 = newll; 
Line Loop(frac_loop_4) = { frac_line_1, frac_line_17, -frac_line_39, -frac_line_3};
auxiliary_surface_4 = news; Plane Surface(auxiliary_surface_4) = {frac_loop_4};
Physical Surface("AUXILIARY_4") = {auxiliary_surface_4};
Surface{auxiliary_surface_4} In Volume{1};


frac_loop_5 = newll; 
Line Loop(frac_loop_5) = { frac_line_0, frac_line_7, frac_line_21, -frac_line_2};
auxiliary_surface_5 = news; Plane Surface(auxiliary_surface_5) = {frac_loop_5};
Physical Surface("AUXILIARY_5") = {auxiliary_surface_5};
Surface{auxiliary_surface_5} In Volume{1};


frac_loop_6 = newll; 
Line Loop(frac_loop_6) = { frac_line_0, frac_line_8, frac_line_28, -frac_line_3};
auxiliary_surface_6 = news; Plane Surface(auxiliary_surface_6) = {frac_loop_6};
Physical Surface("AUXILIARY_6") = {auxiliary_surface_6};
Surface{auxiliary_surface_6} In Volume{1};


frac_loop_7 = newll; 
Line Loop(frac_loop_7) = { frac_line_11, frac_line_16, -frac_line_24, -frac_line_12};
auxiliary_surface_7 = news; Plane Surface(auxiliary_surface_7) = {frac_loop_7};
Physical Surface("AUXILIARY_7") = {auxiliary_surface_7};
Surface{auxiliary_surface_7} In Volume{1};


frac_loop_8 = newll; 
Line Loop(frac_loop_8) = { frac_line_11, frac_line_17, -frac_line_31, -frac_line_13};
auxiliary_surface_8 = news; Plane Surface(auxiliary_surface_8) = {frac_loop_8};
Physical Surface("AUXILIARY_8") = {auxiliary_surface_8};
Surface{auxiliary_surface_8} In Volume{1};


frac_loop_9 = newll; 
Line Loop(frac_loop_9) = { frac_line_12, frac_line_25, frac_line_44, -frac_line_14};
auxiliary_surface_9 = news; Plane Surface(auxiliary_surface_9) = {frac_loop_9};
Physical Surface("AUXILIARY_9") = {auxiliary_surface_9};
Surface{auxiliary_surface_9} In Volume{1};


frac_loop_10 = newll; 
Line Loop(frac_loop_10) = { frac_line_13, frac_line_32, -frac_line_47, -frac_line_14};
auxiliary_surface_10 = news; Plane Surface(auxiliary_surface_10) = {frac_loop_10};
Physical Surface("AUXILIARY_10") = {auxiliary_surface_10};
Surface{auxiliary_surface_10} In Volume{1};


frac_loop_11 = newll; 
Line Loop(frac_loop_11) = { frac_line_16, frac_line_37, frac_line_50, -frac_line_18};
auxiliary_surface_11 = news; Plane Surface(auxiliary_surface_11) = {frac_loop_11};
Physical Surface("AUXILIARY_11") = {auxiliary_surface_11};
Surface{auxiliary_surface_11} In Volume{1};


frac_loop_12 = newll; 
Line Loop(frac_loop_12) = { frac_line_17, frac_line_42, -frac_line_52, -frac_line_18};
auxiliary_surface_12 = news; Plane Surface(auxiliary_surface_12) = {frac_loop_12};
Physical Surface("AUXILIARY_12") = {auxiliary_surface_12};
Surface{auxiliary_surface_12} In Volume{1};


frac_loop_13 = newll; 
Line Loop(frac_loop_13) = { frac_line_7, frac_line_22, frac_line_54, -frac_line_9};
auxiliary_surface_13 = news; Plane Surface(auxiliary_surface_13) = {frac_loop_13};
Physical Surface("AUXILIARY_13") = {auxiliary_surface_13};
Surface{auxiliary_surface_13} In Volume{1};


frac_loop_14 = newll; 
Line Loop(frac_loop_14) = { frac_line_8, frac_line_29, -frac_line_57, -frac_line_9};
auxiliary_surface_14 = news; Plane Surface(auxiliary_surface_14) = {frac_loop_14};
Physical Surface("AUXILIARY_14") = {auxiliary_surface_14};
Surface{auxiliary_surface_14} In Volume{1};


frac_loop_15 = newll; 
Line Loop(frac_loop_15) = { frac_line_2, frac_line_35, frac_line_60, -frac_line_4};
auxiliary_surface_15 = news; Plane Surface(auxiliary_surface_15) = {frac_loop_15};
Physical Surface("AUXILIARY_15") = {auxiliary_surface_15};
Surface{auxiliary_surface_15} In Volume{1};


frac_loop_16 = newll; 
Line Loop(frac_loop_16) = { frac_line_3, frac_line_40, -frac_line_62, -frac_line_4};
auxiliary_surface_16 = news; Plane Surface(auxiliary_surface_16) = {frac_loop_16};
Physical Surface("AUXILIARY_16") = {auxiliary_surface_16};
Surface{auxiliary_surface_16} In Volume{1};


frac_loop_17 = newll; 
Line Loop(frac_loop_17) = { frac_line_12, frac_line_26, frac_line_64, -frac_line_15};
auxiliary_surface_17 = news; Plane Surface(auxiliary_surface_17) = {frac_loop_17};
Physical Surface("AUXILIARY_17") = {auxiliary_surface_17};
Surface{auxiliary_surface_17} In Volume{1};


frac_loop_18 = newll; 
Line Loop(frac_loop_18) = { frac_line_13, frac_line_33, -frac_line_67, -frac_line_15};
auxiliary_surface_18 = news; Plane Surface(auxiliary_surface_18) = {frac_loop_18};
Physical Surface("AUXILIARY_18") = {auxiliary_surface_18};
Surface{auxiliary_surface_18} In Volume{1};


frac_loop_19 = newll; 
Line Loop(frac_loop_19) = { frac_line_16, frac_line_38, frac_line_70, -frac_line_19};
auxiliary_surface_19 = news; Plane Surface(auxiliary_surface_19) = {frac_loop_19};
Physical Surface("AUXILIARY_19") = {auxiliary_surface_19};
Surface{auxiliary_surface_19} In Volume{1};


frac_loop_20 = newll; 
Line Loop(frac_loop_20) = { frac_line_17, frac_line_43, -frac_line_73, -frac_line_19};
auxiliary_surface_20 = news; Plane Surface(auxiliary_surface_20) = {frac_loop_20};
Physical Surface("AUXILIARY_20") = {auxiliary_surface_20};
Surface{auxiliary_surface_20} In Volume{1};


frac_loop_21 = newll; 
Line Loop(frac_loop_21) = { frac_line_7, frac_line_23, frac_line_76, -frac_line_10};
auxiliary_surface_21 = news; Plane Surface(auxiliary_surface_21) = {frac_loop_21};
Physical Surface("AUXILIARY_21") = {auxiliary_surface_21};
Surface{auxiliary_surface_21} In Volume{1};


frac_loop_22 = newll; 
Line Loop(frac_loop_22) = { frac_line_8, frac_line_30, -frac_line_78, -frac_line_10};
auxiliary_surface_22 = news; Plane Surface(auxiliary_surface_22) = {frac_loop_22};
Physical Surface("AUXILIARY_22") = {auxiliary_surface_22};
Surface{auxiliary_surface_22} In Volume{1};


frac_loop_23 = newll; 
Line Loop(frac_loop_23) = { frac_line_2, frac_line_36, frac_line_80, -frac_line_5};
auxiliary_surface_23 = news; Plane Surface(auxiliary_surface_23) = {frac_loop_23};
Physical Surface("AUXILIARY_23") = {auxiliary_surface_23};
Surface{auxiliary_surface_23} In Volume{1};


frac_loop_24 = newll; 
Line Loop(frac_loop_24) = { frac_line_3, frac_line_41, -frac_line_82, -frac_line_5};
auxiliary_surface_24 = news; Plane Surface(auxiliary_surface_24) = {frac_loop_24};
Physical Surface("AUXILIARY_24") = {auxiliary_surface_24};
Surface{auxiliary_surface_24} In Volume{1};


// End of fracture specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p36};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p37};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p38};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p39};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {p40};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {p41};
Physical Point("DOMAIN_BOUNDARY_POINT_6") = {p42};
Physical Point("DOMAIN_BOUNDARY_POINT_7") = {p43};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_0") = {p4};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_1") = {p5};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_2") = {p6};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_3") = {p7};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_4") = {p8};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_5") = {p9};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_6") = {p10};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_7") = {p11};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_8") = {p12};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_9") = {p13};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_10") = {p14};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_11") = {p15};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_12") = {p16};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_13") = {p17};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_14") = {p18};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_15") = {p19};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_16") = {p20};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_17") = {p21};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_18") = {p22};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_19") = {p23};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_20") = {p24};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_21") = {p25};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_22") = {p26};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_23") = {p27};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_24") = {p28};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_25") = {p29};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_26") = {p30};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_27") = {p31};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_28") = {p32};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_29") = {p33};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_30") = {p34};
Physical Point("FRACTURE_CONSTRAINT_INTERSECTION_POINT_31") = {p35};
// End of physical point specification

