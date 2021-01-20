
// Define points
p0 = newp; Point(p0) = {0.05, 0.416, 0.0, 0.0125 };
p1 = newp; Point(p1) = {0.15217391304347827, 0.20347826086956522, 0.0, 0.0125 };
p2 = newp; Point(p2) = {0.22, 0.0624, 0.0, 0.0125 };
p3 = newp; Point(p3) = {0.05, 0.275, 0.0, 0.0125 };
p4 = newp; Point(p4) = {0.25, 0.135, 0.0, 0.0125 };
p5 = newp; Point(p5) = {0.15, 0.63, 0.0, 0.0125 };
p6 = newp; Point(p6) = {0.45, 0.09, 0.0, 0.0125 };
p7 = newp; Point(p7) = {0.15, 0.9167, 0.0, 0.0125 };
p8 = newp; Point(p8) = {0.18634062556499725, 0.8561274453082626, 0.0, 0.0125 };
p9 = newp; Point(p9) = {0.4, 0.5, 0.0, 0.0125 };
p10 = newp; Point(p10) = {0.65, 0.8333, 0.0, 0.0125 };
p11 = newp; Point(p11) = {0.6620579744829095, 0.7931108772454312, 0.0, 0.0125 };
p12 = newp; Point(p12) = {0.849723, 0.167625, 0.0, 0.0125 };
p13 = newp; Point(p13) = {0.7, 0.235, 0.0, 0.0125 };
p14 = newp; Point(p14) = {0.8150369245622153, 0.28323338394700304, 0.0, 0.0125 };
p15 = newp; Point(p15) = {0.6, 0.38, 0.0, 0.0125 };
p16 = newp; Point(p16) = {0.85, 0.2675, 0.0, 0.0125 };
p17 = newp; Point(p17) = {0.35, 0.9714, 0.0, 0.0125 };
p18 = newp; Point(p18) = {0.3732601169869882, 0.9581107198281008, 0.0, 0.0125 };
p19 = newp; Point(p19) = {0.8, 0.7143, 0.0, 0.0125 };
p20 = newp; Point(p20) = {0.75, 0.9574, 0.0, 0.0125 };
p21 = newp; Point(p21) = {0.95, 0.8155, 0.0, 0.0125 };
p22 = newp; Point(p22) = {0.15, 0.8363, 0.0, 0.0125 };
p23 = newp; Point(p23) = {0.4, 0.9727, 0.0, 0.0125 };
p24 = newp; Point(p24) = {0.0, 0.0, 0.0, 0.0125 };
p25 = newp; Point(p25) = {1.0, 0.0, 0.0, 0.0125 };
p26 = newp; Point(p26) = {1.0, 1.0, 0.0, 0.0125 };
p27 = newp; Point(p27) = {0.0, 1.0, 0.0, 0.0125 };
// End of point specification

// Start of specification of domain// Define lines that make up the domain boundary
bound_line_0 = newl;
Line(bound_line_0) ={p24, p25};
Physical Line("DOMAIN_BOUNDARY_20") = { bound_line_0 };
bound_line_1 = newl;
Line(bound_line_1) ={p25, p26};
Physical Line("DOMAIN_BOUNDARY_21") = { bound_line_1 };
bound_line_2 = newl;
Line(bound_line_2) ={p26, p27};
Physical Line("DOMAIN_BOUNDARY_22") = { bound_line_2 };
bound_line_3 = newl;
Line(bound_line_3) ={p27, p24};
Physical Line("DOMAIN_BOUNDARY_23") = { bound_line_3 };

// Line loop that makes the domain boundary
Domain_loop = newll;
Line Loop(Domain_loop) = {bound_line_0, bound_line_1, bound_line_2, bound_line_3};
domain_surf = news;
Plane Surface(domain_surf) = {Domain_loop};
Physical Surface("DOMAIN") = {domain_surf};
// End of domain specification

// Start specification of fractures/compartment boundary/auxiliary elements
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Line{frac_line_0} In Surface{domain_surf};
Physical Line("FRACTURE_0") = { frac_line_0 };

frac_line_1 = newl; Line(frac_line_1) = {p1, p2};
Line{frac_line_1} In Surface{domain_surf};
Physical Line("FRACTURE_1") = { frac_line_1 };

frac_line_2 = newl; Line(frac_line_2) = {p1, p3};
Line{frac_line_2} In Surface{domain_surf};
Physical Line("FRACTURE_2") = { frac_line_2 };

frac_line_3 = newl; Line(frac_line_3) = {p1, p4};
Line{frac_line_3} In Surface{domain_surf};
Physical Line("FRACTURE_3") = { frac_line_3 };

frac_line_4 = newl; Line(frac_line_4) = {p5, p6};
Line{frac_line_4} In Surface{domain_surf};
Physical Line("FRACTURE_4") = { frac_line_4 };

frac_line_5 = newl; Line(frac_line_5) = {p7, p8};
Line{frac_line_5} In Surface{domain_surf};
Physical Line("FRACTURE_5") = { frac_line_5 };

frac_line_6 = newl; Line(frac_line_6) = {p8, p9};
Line{frac_line_6} In Surface{domain_surf};
Physical Line("FRACTURE_6") = { frac_line_6 };

frac_line_7 = newl; Line(frac_line_7) = {p10, p11};
Line{frac_line_7} In Surface{domain_surf};
Physical Line("FRACTURE_7") = { frac_line_7 };

frac_line_8 = newl; Line(frac_line_8) = {p12, p13};
Line{frac_line_8} In Surface{domain_surf};
Physical Line("FRACTURE_8") = { frac_line_8 };

frac_line_9 = newl; Line(frac_line_9) = {p12, p14};
Line{frac_line_9} In Surface{domain_surf};
Physical Line("FRACTURE_9") = { frac_line_9 };

frac_line_10 = newl; Line(frac_line_10) = {p14, p15};
Line{frac_line_10} In Surface{domain_surf};
Physical Line("FRACTURE_10") = { frac_line_10 };

frac_line_11 = newl; Line(frac_line_11) = {p14, p16};
Line{frac_line_11} In Surface{domain_surf};
Physical Line("FRACTURE_11") = { frac_line_11 };

frac_line_12 = newl; Line(frac_line_12) = {p17, p18};
Line{frac_line_12} In Surface{domain_surf};
Physical Line("FRACTURE_12") = { frac_line_12 };

frac_line_13 = newl; Line(frac_line_13) = {p11, p19};
Line{frac_line_13} In Surface{domain_surf};
Physical Line("FRACTURE_13") = { frac_line_13 };

frac_line_14 = newl; Line(frac_line_14) = {p20, p21};
Line{frac_line_14} In Surface{domain_surf};
Physical Line("FRACTURE_14") = { frac_line_14 };

frac_line_15 = newl; Line(frac_line_15) = {p8, p22};
Line{frac_line_15} In Surface{domain_surf};
Physical Line("FRACTURE_15") = { frac_line_15 };

frac_line_16 = newl; Line(frac_line_16) = {p18, p23};
Line{frac_line_16} In Surface{domain_surf};
Physical Line("FRACTURE_16") = { frac_line_16 };

frac_line_17 = newl; Line(frac_line_17) = {p8, p18};
Line{frac_line_17} In Surface{domain_surf};
Physical Line("FRACTURE_17") = { frac_line_17 };

frac_line_18 = newl; Line(frac_line_18) = {p11, p14};
Line{frac_line_18} In Surface{domain_surf};
Physical Line("FRACTURE_18") = { frac_line_18 };

frac_line_19 = newl; Line(frac_line_19) = {p11, p18};
Line{frac_line_19} In Surface{domain_surf};
Physical Line("FRACTURE_19") = { frac_line_19 };

// End of /compartment boundary/auxiliary elements specification

// Start physical point specification
Physical Point("FRACTURE_POINT_0") = {p1};
Physical Point("FRACTURE_POINT_1") = {p8};
Physical Point("FRACTURE_POINT_2") = {p11};
Physical Point("FRACTURE_POINT_3") = {p12};
Physical Point("FRACTURE_POINT_4") = {p14};
Physical Point("FRACTURE_POINT_5") = {p18};
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p24};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p25};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p26};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p27};
// End of physical point specification

