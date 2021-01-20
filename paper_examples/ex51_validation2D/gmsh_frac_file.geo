
// Define points
p0 = newp; Point(p0) = {0.5, 0.25, 0.0, 0.1 };
p1 = newp; Point(p1) = {0.5, 0.75, 0.0, 0.1 };
p2 = newp; Point(p2) = {0.0, 0.75, 0.0, 0.1 };
p3 = newp; Point(p3) = {1.0, 0.75, 0.0, 0.1 };
p4 = newp; Point(p4) = {0.0, 0.25, 0.0, 0.1 };
p5 = newp; Point(p5) = {1.0, 0.25, 0.0, 0.1 };
p6 = newp; Point(p6) = {0.0, 0.0, 0.0, 0.1 };
p7 = newp; Point(p7) = {1.0, 0.0, 0.0, 0.1 };
p8 = newp; Point(p8) = {1.0, 1.0, 0.0, 0.1 };
p9 = newp; Point(p9) = {0.0, 1.0, 0.0, 0.1 };
// End of point specification

// Start of specification of domain// Define lines that make up the domain boundary
bound_line_0 = newl;
Line(bound_line_0) ={p6, p7};
Physical Line("DOMAIN_BOUNDARY_3") = { bound_line_0 };
bound_line_1 = newl;
Line(bound_line_1) ={p7, p5};
bound_line_2 = newl;
Line(bound_line_2) ={p5, p3};
bound_line_3 = newl;
Line(bound_line_3) ={p3, p8};
Physical Line("DOMAIN_BOUNDARY_4") = { bound_line_1, bound_line_2, bound_line_3 };
bound_line_4 = newl;
Line(bound_line_4) ={p8, p9};
Physical Line("DOMAIN_BOUNDARY_5") = { bound_line_4 };
bound_line_5 = newl;
Line(bound_line_5) ={p9, p2};
bound_line_6 = newl;
Line(bound_line_6) ={p2, p4};
bound_line_7 = newl;
Line(bound_line_7) ={p4, p6};
Physical Line("DOMAIN_BOUNDARY_6") = { bound_line_5, bound_line_6, bound_line_7 };

// Line loop that makes the domain boundary
Domain_loop = newll;
Line Loop(Domain_loop) = {bound_line_0, bound_line_1, bound_line_2, bound_line_3, bound_line_4, bound_line_5, bound_line_6, bound_line_7};
domain_surf = news;
Plane Surface(domain_surf) = {Domain_loop};
Physical Surface("DOMAIN") = {domain_surf};
// End of domain specification

// Start specification of fractures/compartment boundary/auxiliary elements
frac_line_0 = newl; Line(frac_line_0) = {p0, p1};
Line{frac_line_0} In Surface{domain_surf};
Physical Line("FRACTURE_0") = { frac_line_0 };

seg_line_1 = newl; Line(seg_line_1) = {p1, p2};
Line{seg_line_1} In Surface{domain_surf};
seg_line_2 = newl; Line(seg_line_2) = {p1, p3};
Line{seg_line_2} In Surface{domain_surf};
Physical Line("AUXILIARY_1") = { seg_line_1, seg_line_2 };

seg_line_3 = newl; Line(seg_line_3) = {p0, p4};
Line{seg_line_3} In Surface{domain_surf};
seg_line_4 = newl; Line(seg_line_4) = {p0, p5};
Line{seg_line_4} In Surface{domain_surf};
Physical Line("AUXILIARY_2") = { seg_line_3, seg_line_4 };

// End of /compartment boundary/auxiliary elements specification

// Start physical point specification
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p6};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p7};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p8};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p9};
// End of physical point specification

