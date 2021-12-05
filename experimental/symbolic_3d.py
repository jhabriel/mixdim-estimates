import sympy as sym


#%% Atomic symbols
x, y, z = sym.symbols("x y z")
alpha = sym.symbols("alpha")
beta1, beta2 = sym.symbols("beta1, beta2")
gamma1, gamma2 = sym.symbols("gamma1, gamma2")
n = sym.symbols("n")
d1, d2, d3, d4, d5, d6, d7, d8, d9 = sym.symbols("d1 d2 d3 d4 d5 d6 d7 d8 d9")
omega = sym.symbols("omega")

# alpha and its derivatives
alpha_exp = x - 0.5
dalpha_dx = sym.Integer(1)
dalpha_dy = sym.Integer(0)
dalpha_dz = sym.Integer(0)

# beta1 and its derivatives
beta1_exp = y - 0.25
dbeta1_dx = sym.Integer(0)
dbeta1_dy = sym.Integer(1)
dbeta1_dz = sym.Integer(0)

# beta2 and its derivatives
beta2_exp = y - 0.75
dbeta2_dx = sym.Integer(0)
dbeta2_dy = sym.Integer(1)
dbeta2_dz = sym.Integer(0)

# gamma1 and its derivatives
gamma1_exp = z - 0.25
dgamma1_dx = sym.Integer(0)
dgamma1_dy = sym.Integer(0)
dgamma1_dz = sym.Integer(1)

# gamma2 and its derivatives
gamma2_exp = z - 0.75
dgamma2_dx = sym.Integer(0)
dgamma2_dy = sym.Integer(0)
dgamma2_dz = sym.Integer(1)

# omega and its derivatives
omega_exp = beta1 ** 2 * beta2 ** 2 * gamma1 ** 2 * gamma2 ** 2
domega_dbeta1 = sym.diff(omega_exp, beta1)
domega_dbeta2 = sym.diff(omega_exp, beta2)
domega_dgamma1 = sym.diff(omega_exp, gamma1)
domega_dgamma2 = sym.diff(omega_exp, gamma2)
domega_dx = (domega_dbeta1 * dbeta1_dx
             + domega_dbeta2 * dbeta2_dx
             + domega_dgamma1 * dgamma1_dx
             + domega_dgamma2 * dgamma2_dx)
domega_dy = (domega_dbeta1 * dbeta1_dy
             + domega_dbeta2 * dbeta2_dy
             + domega_dgamma1 * dgamma1_dy
             + domega_dgamma2 * dgamma2_dy)
domega_dz = (domega_dbeta1 * dbeta1_dz
             + domega_dbeta2 * dbeta2_dz
             + domega_dgamma1 * dgamma1_dz
             + domega_dgamma2 * dgamma2_dz)

# d1 and its derivatives
d1_exp = (alpha ** 2 + beta1 ** 2 + gamma1 ** 2) ** 0.5
dd1_dalpha = sym.diff(d1_exp, alpha)
dd1_dbeta1 = sym.diff(d1_exp, beta1)
dd1_dgamma1 = sym.diff(d1_exp, gamma1)
dd1_dx = dd1_dalpha * dalpha_dx + dd1_dbeta1 * dbeta1_dx + dd1_dgamma1 * dgamma1_dx
dd1_dy = dd1_dalpha * dalpha_dy + dd1_dbeta1 * dbeta1_dy + dd1_dgamma1 * dgamma1_dy
dd1_dz = dd1_dalpha * dalpha_dz + dd1_dbeta1 * dbeta1_dz + dd1_dgamma1 * dgamma1_dz

# d2 and its derivatives
d2_exp = (alpha ** 2 + beta1 ** 2) ** 0.5
dd2_dalpha = sym.diff(d2_exp, alpha)
dd2_dbeta1 = sym.diff(d2_exp, beta1)
dd2_dx = dd2_dalpha * dalpha_dx + dd2_dbeta1 * dbeta1_dx
dd2_dy = dd2_dalpha * dalpha_dy + dd2_dbeta1 * dbeta1_dy
dd2_dz = dd2_dalpha * dalpha_dz + dd2_dbeta1 * dbeta1_dz

# d3 and its derivatives
d3_exp = (alpha ** 2 + beta1 ** 2 + gamma2 ** 2) ** 0.5
dd3_dalpha = sym.diff(d3_exp, alpha)
dd3_dbeta1 = sym.diff(d3_exp, beta1)
dd3_dgamma2 = sym.diff(d3_exp, gamma2)
dd3_dx = dd3_dalpha * dalpha_dx + dd3_dbeta1 * dbeta1_dx + dd3_dgamma2 * dgamma2_dx
dd3_dy = dd3_dalpha * dalpha_dy + dd3_dbeta1 * dbeta1_dy + dd3_dgamma2 * dgamma2_dy
dd3_dz = dd3_dalpha * dalpha_dz + dd3_dbeta1 * dbeta1_dz + dd3_dgamma2 * dgamma2_dz

# d4 and its derivatives
d4_exp = (alpha ** 2 + gamma1 ** 2) ** 0.5
dd4_dalpha = sym.diff(d4_exp, alpha)
dd4_dgamma1 = sym.diff(d4_exp, gamma1)
dd4_dx = dd4_dalpha * dalpha_dx + dd4_dgamma1 * dgamma1_dx
dd4_dy = dd4_dalpha * dalpha_dy + dd4_dgamma1 * dgamma1_dy
dd4_dz = dd4_dalpha * dalpha_dz + dd4_dgamma1 * dgamma1_dz

# d5 and its derivatives
d5_exp = (alpha ** 2) ** 0.5
dd5_dalpha = sym.diff(d5_exp, alpha)
dd5_dx = dd5_dalpha * dalpha_dx
dd5_dy = dd5_dalpha * dalpha_dy
dd5_dz = dd5_dalpha * dalpha_dz

# d6 and its derivatives
d6_exp = (alpha ** 2 + gamma2 ** 2) ** 0.5
dd6_dalpha = sym.diff(d6_exp, alpha)
dd6_dgamma2 = sym.diff(d6_exp, gamma2)
dd6_dx = dd6_dalpha * dalpha_dx + dd6_dgamma2 * dgamma2_dx
dd6_dy = dd6_dalpha * dalpha_dy + dd6_dgamma2 * dgamma2_dy
dd6_dz = dd6_dalpha * dalpha_dz + dd6_dgamma2 * dgamma2_dz

# d7 and its derivatives
d7_exp = (alpha ** 2 + beta2 ** 2 + gamma1 ** 2) ** 0.5
dd7_dalpha = sym.diff(d7_exp, alpha)
dd7_dbeta2 = sym.diff(d7_exp, beta2)
dd7_dgamma1 = sym.diff(d7_exp, gamma1)
dd7_dx = dd7_dalpha * dalpha_dx + dd7_dbeta2 * dbeta2_dx + dd7_dgamma1 * dgamma1_dx
dd7_dy = dd7_dalpha * dalpha_dy + dd7_dbeta2 * dbeta2_dy + dd7_dgamma1 * dgamma1_dy
dd7_dz = dd7_dalpha * dalpha_dz + dd7_dbeta2 * dbeta2_dz + dd7_dgamma1 * dgamma1_dz

# d8 and its derivatives
d8_exp = (alpha ** 2 + beta2 ** 2) ** 0.5
dd8_dalpha = sym.diff(d8_exp, alpha)
dd8_dbeta2 = sym.diff(d8_exp, beta2)
dd8_dx = dd8_dalpha * dalpha_dx + dd8_dbeta2 * dbeta2_dx
dd8_dy = dd8_dalpha * dalpha_dy + dd8_dbeta2 * dbeta2_dy
dd8_dz = dd8_dalpha * dalpha_dz + dd8_dbeta2 * dbeta2_dz

# d9 and its derivatives
d9_exp = (alpha ** 2 + beta2 ** 2 + gamma2 ** 2) ** 0.5
dd9_dalpha = sym.diff(d9_exp, alpha)
dd9_dbeta2 = sym.diff(d9_exp, beta2)
dd9_dgamma2 = sym.diff(d9_exp, gamma2)
dd9_dx = dd9_dalpha * dalpha_dx + dd9_dbeta2 * dbeta2_dx + dd9_dgamma2 * dgamma2_dx
dd9_dy = dd9_dalpha * dalpha_dy + dd9_dbeta2 * dbeta2_dy + dd9_dgamma2 * dgamma2_dy
dd9_dz = dd9_dalpha * dalpha_dz + dd9_dbeta2 * dbeta2_dz + dd9_dgamma2 * dgamma2_dz

# p2_1 and its derivatives
p1_exp = d1 ** (n + 1)
dp1_dd1 = sym.diff(p1_exp, d1)
dp1_dx = dp1_dd1 * dd1_dx
dp1_dy = dp1_dd1 * dd1_dy
dp1_dz = dp1_dd1 * dd1_dz

# p2_2 and its derivatives
p2_exp = d2 ** (n + 1)
dp2_dd2 = sym.diff(p2_exp, d2)
dp2_dx = dp2_dd2 * dd2_dx
dp2_dy = dp2_dd2 * dd2_dy
dp2_dz = dp2_dd2 * dd2_dz

# p2_3 and its derivatives
p3_exp = d3 ** (n + 1)
dp3_dd3 = sym.diff(p3_exp, d3)
dp3_dx = dp3_dd3 * dd3_dx
dp3_dy = dp3_dd3 * dd3_dy
dp3_dz = dp3_dd3 * dd3_dz

# p2_4 and its derivatives
p4_exp = d4 ** (n + 1)
dp4_dd4 = sym.diff(p4_exp, d4)
dp4_dx = dp4_dd4 * dd4_dx
dp4_dy = dp4_dd4 * dd4_dy
dp4_dz = dp4_dd4 * dd4_dz

# p2_5 and its derivatives
p5_exp = d5 ** (n + 1) + omega * d5
dp5_dd5 = sym.diff(p5_exp, d5)
dp5_domega = sym.diff(p5_exp, omega)
dp5_dx = dp5_dd5 * dd5_dx + dp5_domega * domega_dx
dp5_dy = dp5_dd5 * dd5_dy + dp5_domega * domega_dy
dp5_dz = dp5_dd5 * dd5_dz + dp5_domega * domega_dz

# p2_6 and its derivatives
p6_exp = d6 ** (n + 1)
dp6_dd6 = sym.diff(p6_exp, d6)
dp6_dx = dp6_dd6 * dd6_dx
dp6_dy = dp6_dd6 * dd6_dy
dp6_dz = dp6_dd6 * dd6_dz

# p2_7 and its derivatives
p7_exp = d7 ** (n + 1)
dp7_dd7 = sym.diff(p7_exp, d7)
dp7_dx = dp7_dd7 * dd7_dx
dp7_dy = dp7_dd7 * dd7_dy
dp7_dz = dp7_dd7 * dd7_dz

# p2_8 and its derivatives
p8_exp = d8 ** (n + 1)
dp8_dd8 = sym.diff(p8_exp, d8)
dp8_dx = dp8_dd8 * dd8_dx
dp8_dy = dp8_dd8 * dd8_dy
dp8_dz = dp8_dd8 * dd8_dz

# p2_9 and its derivatives
p9_exp = d9 ** (n + 1)
dp9_dd9 = sym.diff(p9_exp, d9)
dp9_dx = dp9_dd9 * dd9_dx
dp9_dy = dp9_dd9 * dd9_dy
dp9_dz = dp9_dd9 * dd9_dz

# u2_1 and its derivatives
u1x_exp = -d1 ** (n - 1) * (n+1) * alpha
du1x_alpha = sym.diff(u1x_exp, alpha)
du1x_dd1 = sym.diff(u1x_exp, d1)
du1x_dx = du1x_alpha * dalpha_dx + du1x_dd1 * dd1_dx

u1y_exp = -d1 ** (n - 1) * (n + 1) * beta1
du1y_beta1 = sym.diff(u1y_exp, beta1)
du1y_dd1 = sym.diff(u1y_exp, d1)
du1y_dy = du1y_beta1 * dbeta1_dy + du1y_dd1 * dd1_dy

u1z_exp = -d1 ** (n - 1) * (n + 1) * gamma1
du1z_gamma1 = sym.diff(u1z_exp, gamma1)
du1z_dd1 = sym.diff(u1z_exp, d1)
du1z_dz = du1z_gamma1 * dgamma1_dz + du1z_dd1 * dd1_dz

# u2_2 and its derivatives
u2x_exp = -d2 ** (n - 1) * (n + 1) * alpha
du2x_alpha = sym.diff(u2x_exp, alpha)
du2x_dd2 = sym.diff(u2x_exp, d2)
du2x_dx = du2x_alpha * dalpha_dx + du2x_dd2 * dd2_dx

u2y_exp = -d2 ** (n - 1) * (n + 1) * beta1
du2y_beta1 = sym.diff(u2y_exp, beta1)
du2y_dd2 = sym.diff(u2y_exp, d2)
du2y_dy = du2y_beta1 * dbeta1_dy + du2y_dd2 * dd2_dy

# u2_3 and its derivatives
u3x_exp = -d3 ** (n - 1) * (n + 1) * alpha
du3x_alpha = sym.diff(u3x_exp, alpha)
du3x_dd3 = sym.diff(u3x_exp, d3)
du3x_dx = du3x_alpha * dalpha_dx + du3x_dd3 * dd3_dx

u3y_exp = -d3 ** (n - 1) * (n + 1) * beta1
du3y_beta1 = sym.diff(u3y_exp, beta1)
du3y_dd3 = sym.diff(u3y_exp, d3)
du3y_dy = du3y_beta1 * dbeta1_dy + du3y_dd3 * dd3_dy

u3z_exp = -d3 ** (n - 1) * (n + 1) * gamma2
du3z_gamma2 = sym.diff(u3z_exp, gamma2)
du3z_dd3 = sym.diff(u3z_exp, d3)
du3z_dz = du3z_gamma2 * dgamma2_dz + du3z_dd3 * dd3_dz

# u2_4 and its derivatives
u4x_exp = -d4 ** (n - 1) * (n + 1) * alpha
du4x_alpha = sym.diff(u4x_exp, alpha)
du4x_dd4 = sym.diff(u4x_exp, d4)
du4x_dx = du4x_alpha * dalpha_dx + du4x_dd4 * dd4_dx

u4z_exp = -d4 ** (n - 1) * (n + 1) * gamma1
du4z_gamma1 = sym.diff(u4z_exp, gamma1)
du4z_dd4 = sym.diff(u4z_exp, d4)
du4x_dz = du4z_gamma1 * dgamma1_dz + du4z_dd4 * dd4_dz

# u2_5 and its derivatives
u5x_exp = -d5 * alpha ** (-1) * (omega * d5 ** n * (n+1))
du5x_dalpha = sym.diff(u5x_exp, alpha)
du5x_domega = sym.diff(u5x_exp, omega)
du5x_dd5 = sym.diff(u5x_exp, d5)
du5x_dx = (du5x_dalpha * dalpha_dx
           + du5x_domega * domega_dx
           + du5x_dd5 * dd5_dx)

u5y_exp = -d5 * (2 * beta1 ** 2 * beta2 * gamma1 ** 2 * gamma2 ** 2
                 + 2 * beta1 * beta2 ** 2 * gamma1 ** 2 * gamma2 ** 2)
du5y_dbeta1 = sym.diff(u5y_exp, beta1)
du5y_dbeta2 = sym.diff(u5y_exp, beta2)
du5y_dgamma1 = sym.diff(u5y_exp, gamma1)
du5y_dgamma2 = sym.diff(u5y_exp, gamma2)
du5y_dd5 = sym.diff(u5y_exp, d5)
du5y_dy = (du5y_dbeta1 * dbeta1_dy
           + du5y_dbeta2 * dbeta2_dy
           + du5y_dgamma1 * dgamma1_dy
           + du5y_dgamma2 * dgamma2_dy
           + du5y_dd5 * dd5_dy)

u5z_exp = -d5 * (2 * beta1 ** 2 * beta2 ** 2 * gamma1 ** 2 * gamma2
                 + 2 * beta1 ** 2 * beta2 ** 2 * gamma1 * gamma2 ** 2)
du5z_dbeta1 = sym.diff(u5z_exp, beta1)
du5z_dbeta2 = sym.diff(u5z_exp, beta2)
du5z_dgamma1 = sym.diff(u5z_exp, gamma1)
du5z_dgamma2 = sym.diff(u5z_exp, gamma2)
du5z_dd5 = sym.diff(u5z_exp, d5)
du5z_dz = (du5z_dbeta1 * dbeta1_dz
           + du5z_dbeta2 * dbeta2_dz
           + du5z_dgamma1 * dgamma1_dz
           + du5z_dgamma2 * dgamma2_dz
           + du5z_dd5 * dd5_dz)


# f2_1
f1 = sym.simplify(du1x_dx + du1y_dy + du1z_dz)
f2 = sym.simplify(du2x_dx + du2y_dy)
f3 = sym.simplify(du3x_dx + du3y_dy + du3z_dz)
f4 = sym.simplify(du4x_dx + du4x_dz)
f5 = sym.simplify(du5x_dx + du5y_dy + du5z_dz)
print(sym.latex(f1))
print(sym.latex(f2))
print(sym.latex(f3))
print(sym.latex(f4))
print(sym.latex(f5))

#
ufracy_exp = 2 * gamma1 ** 2 + gamma2 ** 2 * (
        beta1 * beta2 ** 2 + beta1 ** 2 * beta2)
dufracy_dbeta1 = sym.diff(ufracy_exp, beta1)
dufracy_dbeta2 = sym.diff(ufracy_exp, beta2)
dufrac_dy = dufracy_dbeta1 * dbeta1_dy + dufracy_dbeta2 * dbeta2_dy

ufracz_exp = 2 * beta1 ** 2 + beta2 ** 2 * (
        gamma1 * gamma2 ** 2 + gamma1 ** 2 * gamma2)
dufracz_dgamma1 = sym.diff(ufracz_exp, gamma1)
dufracz_dgamma2 = sym.diff(ufracz_exp, gamma2)
dufrac_dz = dufracz_dgamma1 * dgamma1_dz + dufracz_dgamma2 * dgamma2_dz

ffrac = sym.simplify(dufrac_dy + dufrac_dz)
print(sym.latex(ffrac))


#
#
# dbot_exp = (alpha ** 2 + beta1 ** 2) ** 0.5
# ddbot_dalpha = sym.diff(dbot_exp, alpha)
# ddbot_dbeta1 = sym.diff(dbot_exp, beta1)
# ddbot_dx = ddbot_dalpha * dalpha_dx + ddbot_dbeta1 * dbeta1_dx
# ddbot_dy = ddbot_dalpha * dalpha_dy + ddbot_dbeta1 * dbeta1_dy
#
# dmid_exp = (alpha ** 2) ** 0.5
# ddmid_dalpha = sym.diff(dmid_exp, alpha)
# ddmid_dx = ddmid_dalpha * dalpha_dx
# ddmid_dy = ddmid_dalpha * dalpha_dy
#
# dtop_exp = (alpha ** 2 + beta2 ** 2) ** 0.5
# ddtop_dalpha = sym.diff(dtop_exp, alpha)
# ddtop_dbeta2 = sym.diff(dtop_exp, beta2)
# ddtop_dx = ddtop_dalpha * dalpha_dx + ddtop_dbeta2 * dbeta2_dx
# ddtop_dy = ddtop_dalpha * dalpha_dy + ddtop_dbeta2 * dbeta2_dy
#
# # Derivatives of the pressure
# pbot = dbot ** (n + 1)
# dpbot_ddbot = sym.diff(pbot, dbot)
# dpbot_dx = sym.simplify(dpbot_ddbot * ddbot_dx)
# dpbot_dy = sym.simplify(dpbot_ddbot * ddbot_dy)
#
# pmid = dmid ** (n + 1) + omega * dmid
# dpmid_ddmid = sym.diff(pmid, dmid)
# dpmid_domega = sym.diff(pmid, omega)
# dpmid_dx = dpmid_ddmid * ddmid_dx + dpmid_domega * domega_dx
# dpmid_dy = dpmid_ddmid * ddmid_dy + dpmid_domega * domega_dy
#
# ptop = dtop ** (n + 1)
# dptop_ddtop = sym.diff(ptop, dtop)
# dptop_dx = sym.simplify(dptop_ddtop * ddtop_dx)
# dptop_dy = sym.simplify(dptop_ddtop * ddtop_dy)
#
# ubotx = - alpha * (n + 1) * dbot ** (n - 1)
# dubotx_dalpha = sym.diff(ubotx, alpha)
# dubotx_ddbot = sym.diff(ubotx, dbot)
# dubotx_dx = dubotx_dalpha * dalpha_dx + dubotx_ddbot * ddbot_dx
#
# uboty = - beta1 * (n + 1) * dbot ** (n - 1)
# duboty_dbeta1 = sym.diff(uboty, beta1)
# duboty_ddbot = sym.diff(uboty, dbot)
# duboty_dy = duboty_dbeta1 * dbeta1_dy + duboty_ddbot * ddbot_dy
#
# umidx = - (alpha ** 2) ** 0.5 * alpha ** (-1) * (omega + (n + 1) * dmid ** n)
# dumidx_dalpha = sym.diff(umidx, alpha)
# dumidx_domega = sym.diff(umidx, omega)
# dumidx_ddmid = sym.diff(umidx, dmid)
# dumidx_dx = dumidx_dalpha * dalpha_dx + dumidx_domega * domega_dx + dumidx_ddmid * ddmid_dx
#
# umidy = - dmid * (2 * beta1 ** 2 * beta2 + 2 * beta1 * beta2 ** 2)
# dumidy_ddmid = sym.diff(umidy, dmid)
# dumidy_dbeta1 = sym.diff(umidy, beta1)
# dumidy_dbeta2 = sym.diff(umidy, beta2)
# dumidy_dy = dumidy_ddmid * ddmid_dy + dumidy_dbeta1 * dbeta1_dy + dumidy_dbeta2 * dbeta2_dy
#
# utopx = -alpha * (n + 1) * dtop ** (n - 1)
# dutopx_dalpha = sym.diff(utopx, alpha)
# dutopx_ddtop = sym.diff(utopx, dtop)
# dutopx_dx = dutopx_dalpha * dalpha_dx + dutopx_ddtop * ddtop_dx
#
# utopy = - beta2 * (n + 1) * dtop ** (n - 1)
# dutopy_dbeta2 = sym.diff(utopy, beta2)
# dutopy_ddtop = sym.diff(utopy, dtop)
# dutopy_dy = dutopy_dbeta2 * dbeta2_dy + dutopy_ddtop * ddtop_dy
#
# fbot = sym.simplify(dubotx_dx + duboty_dy)
# fmid = sym.simplify(dumidx_dx + dumidy_dy)
# ftop = sym.simplify(dutopx_dx + dutopy_dy)



#
#
# dbot_exp = (alpha ** 2 + beta1 ** 2) ** 0.5
# ddbot_dalpha = sym.diff(dbot_exp, alpha)
# ddbot_dbeta1 = sym.diff(dbot_exp, beta1)
# ddbot_dx = ddbot_dalpha * dalpha_dx + ddbot_dbeta1 * dbeta1_dx
# ddbot_dy = ddbot_dalpha * dalpha_dy + ddbot_dbeta1 * dbeta1_dy
#
# dmid_exp = (alpha ** 2) ** 0.5
# ddmid_dalpha = sym.diff(dmid_exp, alpha)
# ddmid_dx = ddmid_dalpha * dalpha_dx
# ddmid_dy = ddmid_dalpha * dalpha_dy
#
# dtop_exp = (alpha ** 2 + beta2 ** 2) ** 0.5
# ddtop_dalpha = sym.diff(dtop_exp, alpha)
# ddtop_dbeta2 = sym.diff(dtop_exp, beta2)
# ddtop_dx = ddtop_dalpha * dalpha_dx + ddtop_dbeta2 * dbeta2_dx
# ddtop_dy = ddtop_dalpha * dalpha_dy + ddtop_dbeta2 * dbeta2_dy
#
# # Derivatives of the pressure
# pbot = dbot ** (n + 1)
# dpbot_ddbot = sym.diff(pbot, dbot)
# dpbot_dx = sym.simplify(dpbot_ddbot * ddbot_dx)
# dpbot_dy = sym.simplify(dpbot_ddbot * ddbot_dy)
#
# pmid = dmid ** (n + 1) + omega * dmid
# dpmid_ddmid = sym.diff(pmid, dmid)
# dpmid_domega = sym.diff(pmid, omega)
# dpmid_dx = dpmid_ddmid * ddmid_dx + dpmid_domega * domega_dx
# dpmid_dy = dpmid_ddmid * ddmid_dy + dpmid_domega * domega_dy
#
# ptop = dtop ** (n + 1)
# dptop_ddtop = sym.diff(ptop, dtop)
# dptop_dx = sym.simplify(dptop_ddtop * ddtop_dx)
# dptop_dy = sym.simplify(dptop_ddtop * ddtop_dy)
#
# ubotx = - alpha * (n + 1) * dbot ** (n - 1)
# dubotx_dalpha = sym.diff(ubotx, alpha)
# dubotx_ddbot = sym.diff(ubotx, dbot)
# dubotx_dx = dubotx_dalpha * dalpha_dx + dubotx_ddbot * ddbot_dx
#
# uboty = - beta1 * (n + 1) * dbot ** (n - 1)
# duboty_dbeta1 = sym.diff(uboty, beta1)
# duboty_ddbot = sym.diff(uboty, dbot)
# duboty_dy = duboty_dbeta1 * dbeta1_dy + duboty_ddbot * ddbot_dy
#
# umidx = - (alpha ** 2) ** 0.5 * alpha ** (-1) * (omega + (n + 1) * dmid ** n)
# dumidx_dalpha = sym.diff(umidx, alpha)
# dumidx_domega = sym.diff(umidx, omega)
# dumidx_ddmid = sym.diff(umidx, dmid)
# dumidx_dx = dumidx_dalpha * dalpha_dx + dumidx_domega * domega_dx + dumidx_ddmid * ddmid_dx
#
# umidy = - dmid * (2 * beta1 ** 2 * beta2 + 2 * beta1 * beta2 ** 2)
# dumidy_ddmid = sym.diff(umidy, dmid)
# dumidy_dbeta1 = sym.diff(umidy, beta1)
# dumidy_dbeta2 = sym.diff(umidy, beta2)
# dumidy_dy = dumidy_ddmid * ddmid_dy + dumidy_dbeta1 * dbeta1_dy + dumidy_dbeta2 * dbeta2_dy
#
# utopx = -alpha * (n + 1) * dtop ** (n - 1)
# dutopx_dalpha = sym.diff(utopx, alpha)
# dutopx_ddtop = sym.diff(utopx, dtop)
# dutopx_dx = dutopx_dalpha * dalpha_dx + dutopx_ddtop * ddtop_dx
#
# utopy = - beta2 * (n + 1) * dtop ** (n - 1)
# dutopy_dbeta2 = sym.diff(utopy, beta2)
# dutopy_ddtop = sym.diff(utopy, dtop)
# dutopy_dy = dutopy_dbeta2 * dbeta2_dy + dutopy_ddtop * ddtop_dy
#
# fbot = sym.simplify(dubotx_dx + duboty_dy)
# fmid = sym.simplify(dumidx_dx + dumidy_dy)
# ftop = sym.simplify(dutopx_dx + dutopy_dy)