
// Define points
p0 = newp; Point(p0) = {0.3, 0.2, 0.0, 0.05 };
p1 = newp; Point(p1) = {0.7, 0.8, 0.0, 0.05 };
p2 = newp; Point(p2) = {0.8, 0.2, 0.0, 0.05 };
p3 = newp; Point(p3) = {0.2, 0.8, 0.0, 0.05 };
p4 = newp; Point(p4) = {0.0, 0.0, 0.0, 0.1 };
p5 = newp; Point(p5) = {1.0, 0.0, 0.0, 0.1 };
p6 = newp; Point(p6) = {1.0, 1.0, 0.0, 0.1 };
p7 = newp; Point(p7) = {0.0, 1.0, 0.0, 0.1 };
p8 = newp; Point(p8) = {0.5, 0.5, 0.0, 0.05 };
// End of point specification

// Start of specification of domain// Define lines that make up the domain boundary
bound_line_0 = newl;
Line(bound_line_0) ={p4, p5};
Physical Line("DOMAIN_BOUNDARY_2") = { bound_line_0 };
bound_line_1 = newl;
Line(bound_line_1) ={p5, p6};
Physical Line("DOMAIN_BOUNDARY_3") = { bound_line_1 };
bound_line_2 = newl;
Line(bound_line_2) ={p6, p7};
Physical Line("DOMAIN_BOUNDARY_4") = { bound_line_2 };
bound_line_3 = newl;
Line(bound_line_3) ={p7, p4};
Physical Line("DOMAIN_BOUNDARY_5") = { bound_line_3 };

// Line loop that makes the domain boundary
Domain_loop = newll;
Line Loop(Domain_loop) = {bound_line_0, bound_line_1, bound_line_2, bound_line_3};
domain_surf = news;
Plane Surface(domain_surf) = {Domain_loop};
Physical Surface("DOMAIN") = {domain_surf};
// End of domain specification

// Start specification of fractures/compartment boundary/auxiliary elements
frac_line_0 = newl; Line(frac_line_0) = {p0, p8};
Line{frac_line_0} In Surface{domain_surf};
frac_line_1 = newl; Line(frac_line_1) = {p1, p8};
Line{frac_line_1} In Surface{domain_surf};
Physical Line("FRACTURE_0") = { frac_line_0, frac_line_1 };

frac_line_2 = newl; Line(frac_line_2) = {p2, p8};
Line{frac_line_2} In Surface{domain_surf};
frac_line_3 = newl; Line(frac_line_3) = {p3, p8};
Line{frac_line_3} In Surface{domain_surf};
Physical Line("FRACTURE_1") = { frac_line_2, frac_line_3 };

// End of /compartment boundary/auxiliary elements specification

// Start physical point specification
Physical Point("FRACTURE_POINT_0") = {p8};
Physical Point("DOMAIN_BOUNDARY_POINT_0") = {p4};
Physical Point("DOMAIN_BOUNDARY_POINT_1") = {p5};
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {p6};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {p7};
// End of physical point specification

