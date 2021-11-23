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

# p1 and its derivatives
p1_exp = d1 ** (n + 1)
dp1_dd1 = sym.diff(p1_exp, d1)
dp1_dx = dp1_dd1 * dd1_dx
dp1_dy = dp1_dd1 * dd1_dy
dp1_dz = dp1_dd1 * dd1_dz

# p2 and its derivatives
p2_exp = d2 ** (n + 1)
dp2_dd2 = sym.diff(p2_exp, d2)
dp2_dx = dp2_dd2 * dd2_dx
dp2_dy = dp2_dd2 * dd2_dy
dp2_dz = dp2_dd2 * dd2_dz

# p3 and its derivatives
p3_exp = d3 ** (n + 1)
dp3_dd3 = sym.diff(p3_exp, d3)
dp3_dx = dp3_dd3 * dd3_dx
dp3_dy = dp3_dd3 * dd3_dy
dp3_dz = dp3_dd3 * dd3_dz

# p4 and its derivatives
p4_exp = d4 ** (n + 1)
dp4_dd4 = sym.diff(p4_exp, d4)
dp4_dx = dp4_dd4 * dd4_dx
dp4_dy = dp4_dd4 * dd4_dy
dp4_dz = dp4_dd4 * dd4_dz

# p5 and its derivatives
p5_exp = d5 ** (n + 1) + omega * d5
dp5_dd5 = sym.diff(p5_exp, d5)
dp5_domega = sym.diff(p5_exp, omega)
dp5_dx = dp5_dd5 * dd5_dx + dp5_domega * domega_dx
dp5_dy = dp5_dd5 * dd5_dy + dp5_domega * domega_dy
dp5_dz = dp5_dd5 * dd5_dz + dp5_domega * domega_dz

# p6 and its derivatives
p6_exp = d6 ** (n + 1)
dp6_dd6 = sym.diff(p6_exp, d6)
dp6_dx = dp6_dd6 * dd6_dx
dp6_dy = dp6_dd6 * dd6_dy
dp6_dz = dp6_dd6 * dd6_dz

# p7 and its derivatives
p7_exp = d7 ** (n + 1)
dp7_dd7 = sym.diff(p7_exp, d7)
dp7_dx = dp7_dd7 * dd7_dx
dp7_dy = dp7_dd7 * dd7_dy
dp7_dz = dp7_dd7 * dd7_dz

# p8 and its derivatives
p8_exp = d8 ** (n + 1)
dp8_dd8 = sym.diff(p8_exp, d8)
dp8_dx = dp8_dd8 * dd8_dx
dp8_dy = dp8_dd8 * dd8_dy
dp8_dz = dp8_dd8 * dd8_dz

# p9 and its derivatives
p9_exp = d9 ** (n + 1)
dp9_dd9 = sym.diff(p9_exp, d9)
dp9_dx = dp9_dd9 * dd9_dx
dp9_dy = dp9_dd9 * dd9_dy
dp9_dz = dp9_dd9 * dd9_dz

# u1
u1x = -dp1_dx
u1y = -dp1_dy
u1z = -dp1_dz
u1 = sym.Array([u1x, u1y, u1z])
print(sym.latex(u1))

# u2
u2x = -dp2_dx
u2y = -dp2_dy
u2z = -dp2_dz
u2 = sym.Array([u2x, u2y, u2z])
print(sym.latex(u2))

# u3
u3x = -dp3_dx
u3y = -dp3_dy
u3z = -dp3_dz
u3 = sym.Array([u3x, u3y, u3z])
print(sym.latex(u3))

# u4
u4x = -dp4_dx
u4y = -dp4_dy
u4z = -dp4_dz
u4 = sym.Array([u4x, u4y, u4z])
print(sym.latex(u4))

# u5
u5x = -dp5_dx
u5y = -dp5_dy
u5z = -dp5_dz
u5 = sym.Array([u5x, u5y, u5z])
print(sym.latex(u5))


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