import numpy
from opt_einsum import contract as einsum
#from numpy import einsum
from numpy import outer
import scipy.linalg


"""
General utility functions
"""

def _vec_to_rotmat(vec, nocc=0, nvir=0):
    assert nocc > 0 and nvir > 0, f"The number of occupied and virtual orbitals must both be positive)"

    nocc = int(nocc)
    nvir = int(nvir)
    nocc_param = int(nocc * (nocc - 1) // 2)
    nvir_param = int(nvir * (nvir - 1) // 2)
    assert len(vec) == nocc_param + \
               nvir_param, f"Invalid Vector length, received {len(vec)}, required {nocc_param + nvir_param}"

    occ_triu_idx = numpy.triu_indices(nocc, k=1)
    vir_triu_idx = numpy.triu_indices(nvir, k=1)
    occ_mat = numpy.zeros((nocc, nocc))
    vir_mat = numpy.zeros((nvir, nvir))
    occ_mat[occ_triu_idx] = vec[:nocc_param]
    vir_mat[vir_triu_idx] = vec[nocc_param:]

    mat = numpy.zeros((nocc+nvir, nocc+nvir))
    mat[:nocc,:nocc] = occ_mat
    mat[nocc:,nocc:] = vir_mat
    mat -= mat.T
    return mat

def _extract_eri_delta(mp, mo_coeff, mo_energy, n_occ, n_vir, chemist=False):
    # e_ijab = e_i + e_j - e_a - e_b
    e_ijab = mo_energy[:n_occ,None,None,None] + mo_energy[None,:n_occ,None,None] - \
        mo_energy[None,None,n_occ:,None] - mo_energy[None,None,None,n_occ:]

    D_oovv = -1/e_ijab

    eris = mp.ao2mo(mo_coeff).ovov
    v_oovv = None
    if isinstance(eris, numpy.ndarray) and eris.ndim == 4:
        v_oovv = eris
    else:
        v_oovv = numpy.zeros((n_occ, n_occ, n_vir, n_vir))
        for i in range(n_occ):
            gi = numpy.asarray(eris[i*n_vir:(i+1)*n_vir])
            v_oovv[i] = gi.reshape(n_vir, n_occ, n_vir).transpose(1,0,2)

    if chemist:
        return einsum('ijab->iajb', v_oovv), D_oovv
    return v_oovv, einsum('ijab->abij', D_oovv)

def _wrap_gradient(g_oo, g_vv, n_occ, n_vir):
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    n_param = n_occ_p + n_vir_p

    occ_idx = numpy.array(numpy.triu_indices(n_occ, k=1))
    vir_idx = numpy.array(numpy.triu_indices(n_vir, k=1))
    grad = numpy.zeros(n_param)

    grad_o = g_oo[occ_idx[0], occ_idx[1]]
    grad_v = g_vv[vir_idx[0], vir_idx[1]]
    grad[:n_occ_p] = grad_o
    grad[n_occ_p:] = grad_v

    return grad

def _wrap_hessian(h_oooo, h_oovv, h_vvvv, n_occ, n_vir):
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    n_param = n_occ_p + n_vir_p

    h_oo = numpy.zeros((n_occ_p, n_occ_p))
    h_ov = numpy.zeros((n_occ_p, n_vir_p))
    h_vv = numpy.zeros((n_vir_p, n_vir_p))
    hess = numpy.zeros((n_param, n_param))
    occ_idx = numpy.array(numpy.triu_indices(n_occ, k=1))
    vir_idx = numpy.array(numpy.triu_indices(n_vir, k=1))

    for hi, oidx0 in enumerate(occ_idx.T):
        for hj, oidx1 in enumerate(occ_idx.T):
            h_oo[hi,hj] = h_oooo[oidx0[0],oidx0[1],oidx1[0],oidx1[1]]
        for ha, vidx1 in enumerate(vir_idx.T):
            h_ov[hi,ha] = h_oovv[oidx0[0],oidx0[1],vidx1[0],vidx1[1]]

    for ha, vidx0 in enumerate(vir_idx.T):
        for hb, vidx1 in enumerate(vir_idx.T):
            h_vv[ha,hb] = h_vvvv[vidx0[0],vidx0[1],vidx1[0],vidx1[1]]

    hess[:n_occ_p,:n_occ_p] = h_oo
    hess[n_occ_p:,n_occ_p:] = h_vv
    hess[:n_occ_p,n_occ_p:] = h_ov
    hess[n_occ_p:,:n_occ_p] = h_ov.T

    return hess

"""
Numerical functions
"""

def _crit_grad_num(ordmp, mo_coeff, mo_energy, mo_occ, epsilon=10**-3, n_occ=0, n_vir=0):
    if n_occ <= 0 or n_vir <= 0:
        n_occ = numpy.sum(mo_occ) // 2
        n_vir = len(mo_occ) - n_occ
    n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

    grad = numpy.zeros(n_param)
    for i in range(n_param):
        vp = numpy.zeros(n_param)
        vm = numpy.zeros(n_param)
        vp[i] += epsilon
        vm[i] -= epsilon
        grad[i] = (ordmp.criterion(vp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir) -
                   ordmp.criterion(vm, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)) / (2 * epsilon)

    return grad

def _crit_hess_num(ordmp, mo_coeff, mo_energy, mo_occ, epsilon=10**-8, n_occ=0, n_vir=0):
    if n_occ <= 0 or n_vir <= 0:
        n_occ = numpy.sum(mo_occ) // 2
        n_vir = len(mo_occ) - n_occ
    n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

    rel_indices = numpy.array(numpy.triu_indices(n_param, k=0))
    hess = numpy.zeros((n_param, n_param))
    for p in rel_indices.T:
        i, j = p[0], p[1]
        #print(f"Calculating Hessian: {i}, {j}")
        vpp = numpy.zeros(n_param)
        vpm = numpy.zeros(n_param)
        vmp = numpy.zeros(n_param)
        vmm = numpy.zeros(n_param)
        vpp[i] += epsilon
        vpp[j] += epsilon
        vpm[i] += epsilon
        vpm[j] -= epsilon
        vmp[i] -= epsilon
        vmp[j] += epsilon
        vmm[i] -= epsilon
        vmm[j] -= epsilon

        c_pp = ordmp.criterion(vpp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        c_pm = ordmp.criterion(vpm, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        c_mp = ordmp.criterion(vmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        c_mm = ordmp.criterion(vmm, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        hess[i,j] = (c_pp + c_mm - c_pm - c_mp) / (4 * epsilon**2)

    return hess + hess.T - numpy.diag(numpy.diag(hess))





"""
Criterion 1: Sum over all spatial ERIs in the oovv block
"""
def _eri_sum(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    if vec is not None:
        if not numpy.allclose(vec, numpy.zeros(vec.shape)):
            rmat = _vec_to_rotmat(vec, n_occ, n_vir)
            mo_coeff = mo_coeff @ scipy.linalg.expm(rmat)

    v_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)[0]
    return einsum('ijab->', v_oovv)

def _eri_sum_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    g_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)[0]

    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)

    # ====================================================================================================
    # Occupied Gradient:
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij}) to:
    itmd0 = (+ 1 * einsum('ac,bd,im,jk,ln,mncd->ij', d_vv, d_vv, d_oo, d_oo, d_oo, g_oovv)   # N^10: O^6V^4 / N^4: O^2V^2
    + 1 * einsum('ac,bd,in,jl,km,mncd->ij', d_vv, d_vv, d_oo, d_oo, d_oo, g_oovv))  # N^10: O^6V^4 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ij->ji', itmd0)
    # ====================================================================================================
    # Virtual Gradient:
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab}) to:
    itmd1 = (+ 1 * einsum('ae,bc,df,ik,jl,klef->ab', d_vv, d_vv, d_vv, d_oo, d_oo, g_oovv)   # N^10: O^4V^6 / N^4: O^2V^2
    + 1 * einsum('af,bd,ce,ik,jl,klef->ab', d_vv, d_vv, d_vv, d_oo, d_oo, g_oovv))  # N^10: O^4V^6 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ab->ba', itmd1)

    return _wrap_gradient(itmd0, itmd1, n_occ, n_vir)

def _eri_sum_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)[0]

    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)

    # ====================================================================================================
    # Occupied-Occupied Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl} + P_{ik}P_{jl} + P_{il}P_{jk}
    # - P_{ij}P_{ik}P_{jl} - P_{ij}P_{il}P_{jk}) to:
    itmd0 = (+ 1 * einsum('im,kn,jlab->ijkl', d_oo, d_oo, v_oovv)   # N^8: O^6V^2 / N^4: O^2V^2
    + 0.5 * einsum('il,jm,knab->ijkl', d_oo, d_oo, v_oovv)   # N^8: O^6V^2 / N^4: O^2V^2
    + 0.5 * einsum('il,jn,mkab->ijkl', d_oo, d_oo, v_oovv))  # N^8: O^6V^2 / N^4: O^2V^2

    h_oooo = itmd0 - einsum('ijkl->jikl',
    itmd0) - einsum('ijkl->ijlk',
    itmd0) + einsum('ijkl->jilk',
    itmd0) + einsum('ijkl->klij',
    itmd0) + einsum('ijkl->lkji',
    itmd0) - einsum('ijkl->lkij',
    itmd0) - einsum('ijkl->klji',
     itmd0)
    del itmd0

    # ====================================================================================================
    # Occupied-Virtual Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{cd} - P_{ij} + P_{ij}P_{cd}) to:
    itmd2 = (+ 1 * einsum('ac,ik,jldb->ijcd', d_vv, d_oo, v_oovv)   # N^8: O^4V^4 / N^4: O^2V^2
    + 1 * einsum('ac,il,kjdb->ijcd', d_vv, d_oo, v_oovv)   # N^8: O^4V^4 / N^4: O^2V^2
    + 1 * einsum('bc,ik,jlad->ijcd', d_vv, d_oo, v_oovv)   # N^8: O^4V^4 / N^4: O^2V^2
    + 1 * einsum('bc,il,kjad->ijcd', d_vv, d_oo, v_oovv))  # N^8: O^4V^4 / N^4: O^2V^2

    h_oovv = itmd2 - einsum('ijcd->ijdc', itmd2) - einsum('ijcd->jicd', itmd2) + einsum('ijcd->jidc', itmd2)
    del itmd2
    # ====================================================================================================
    # Virtual-Virtual Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd} + P_{ac}P_{bd} + P_{ad}P_{bc}
    # - P_{ab}P_{ac}P_{bd} - P_{ab}P_{ad}P_{bc}) to:
    itmd3 = (+ 1 * einsum('ae,cf,ijbd->abcd', d_vv, d_vv, v_oovv)   # N^8: O^2V^6 / N^4: V^4
    + 0.5 * einsum('ad,be,ijcf->abcd', d_vv, d_vv, v_oovv)   # N^8: O^2V^6 / N^4: V^4
    + 0.5 * einsum('ad,bf,ijec->abcd', d_vv, d_vv, v_oovv))  # N^8: O^2V^6 / N^4: V^4

    h_vvvv = itmd3 - einsum('abcd->bacd',
    itmd3) - einsum('abcd->abdc',
    itmd3) + einsum('abcd->badc',
    itmd3) + einsum('abcd->cdab',
    itmd3) + einsum('abcd->dcba',
    itmd3) - einsum('abcd->dcab',
    itmd3) - einsum('abcd->cdba',
     itmd3)
    del itmd3

    return _wrap_hessian(h_oooo, h_oovv, h_vvvv, n_occ, n_vir)


"""
Criterion 2: Maximum correlation energy
"""
def _max_energy(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    if vec is not None:
        if not numpy.allclose(vec, numpy.zeros(vec.shape)):
            rmat = _vec_to_rotmat(vec, n_occ, n_vir)
            mo_coeff = mo_coeff @ scipy.linalg.expm(rmat)
    e_corr = ordmp._mp2.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff)[0]
    return e_corr

def _max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_ovov, D_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir, chemist=True)

    # ================================================================================
    # GRADIENT Block oo
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij}) to:
    itmd0 = (- 4 * einsum('jkab,iakb,jakb->ij', D_oovv, v_ovov, v_ovov)   # N^5: O^3V^2 / N^4: O^2V^2
    - 4 * einsum('jkab,jbka,kaib->ij', D_oovv, v_ovov, v_ovov)   # N^5: O^3V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,iakb,jbka->ij', D_oovv, v_ovov, v_ovov)   # N^5: O^3V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,iakb,kajb->ij', D_oovv, v_ovov, v_ovov))  # N^5: O^3V^2 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ij->ji', itmd0)

    # ================================================================================
    # GRADIENT Block vv
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab}) to:
    itmd1 = (- 4 * einsum('ijbc,iajc,ibjc->ab', D_oovv, v_ovov, v_ovov)   # N^5: O^2V^3 / N^4: O^2V^2
    - 4 * einsum('ijbc,icja,icjb->ab', D_oovv, v_ovov, v_ovov)   # N^5: O^2V^3 / N^4: O^2V^2
    - 2 * einsum('ijac,iajc,icjb->ab', D_oovv, v_ovov, v_ovov)   # N^5: O^2V^3 / N^4: O^2V^2
    - 2 * einsum('ijac,ibjc,icja->ab', D_oovv, v_ovov, v_ovov))  # N^5: O^2V^3 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ab->ba', itmd1)

    return _wrap_gradient(itmd0, itmd1, n_occ, n_vir)

def _max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_ovov, D_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir, chemist=True)

    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)

    # ================================================================================
    # HESSIAN Block oooo
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl} + P_{ik}P_{jl} + P_{il}P_{jk} - P_{ij}P_{ik}P_{jl} - P_{ij}P_{il}P_{jk}) to:
    itmd0 = (- 4 * einsum('ilab,ibka,jbla->ijkl', D_oovv, v_ovov, v_ovov)   # N^6: O^4V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,ialb,jbka->ijkl', D_oovv, v_ovov, v_ovov)   # N^6: O^4V^2 / N^4: O^2V^2
    - 2 * einsum('ilab,ialb,jbka->ijkl', D_oovv, v_ovov, v_ovov)   # N^6: O^4V^2 / N^4: O^2V^2
    + 1 * einsum('ik,jmab,jamb,lbma->ijkl', d_oo, D_oovv, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 1 * einsum('ik,jmab,jamb,malb->ijkl', d_oo, D_oovv, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 2 * einsum('il,jmab,jamb,mbka->ijkl', d_oo, D_oovv, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 2 * einsum('il,jmab,jbma,kbma->ijkl', d_oo, D_oovv, v_ovov, v_ovov))  # N^7: O^5V^2 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ijkl->jikl', itmd0) - einsum('ijkl->ijlk', itmd0) + einsum('ijkl->jilk', itmd0) + einsum('ijkl->klij', itmd0) + einsum('ijkl->lkji', itmd0) - einsum('ijkl->lkij', itmd0) - einsum('ijkl->klji', itmd0)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl}) to:
    itmd1 = (- 4 * einsum('il,imab,jbma,kbma->ijkl', d_oo, D_oovv, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 4 * einsum('il,imab,mbja,mbka->ijkl', d_oo, D_oovv, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * einsum('ik,imab,jamb,lbma->ijkl', d_oo, D_oovv, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * einsum('ik,imab,majb,mbla->ijkl', d_oo, D_oovv, v_ovov, v_ovov))  # N^7: O^5V^2 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ijkl->jikl', itmd1) - einsum('ijkl->ijlk', itmd1) + einsum('ijkl->jilk', itmd1)

    # ================================================================================
    # HESSIAN Block oovv
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{ab} + P_{ij}P_{ab}) to:
    itmd2 = (- 4 * einsum('ikbc,iakc,jbkc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ibkc,jakc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ibkc,kcja->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,icka,jckb->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ickb,jcka->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ickb,kajc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,kaic,kbjc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,kcia,kcjb->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,iakc,jckb->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,iakc,kbjc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ibkc,jcka->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,icka,jbkc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,icka,kcjb->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ickb,jakc->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,kajc,kcib->ijab', D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,kbic,kcja->ijab', D_oovv, v_ovov, v_ovov))  # N^6: O^3V^3 / N^4: O^2V^2

    itmd2 = itmd2 - einsum('ijab->jiab', itmd2) - einsum('ijab->ijba', itmd2) + einsum('ijab->jiba', itmd2)

    # ================================================================================
    # HESSIAN Block vvvv
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd} + P_{ac}P_{bd} + P_{ad}P_{bc} - P_{ab}P_{ac}P_{bd} - P_{ab}P_{ad}P_{bc}) to:
    itmd3 = (- 4 * einsum('ijad,iajc,ibjd->abcd', D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    - 2 * einsum('ijad,iajd,ibjc->abcd', D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    + 1 * einsum('ac,ijbe,ibje,iejd->abcd', d_vv, D_oovv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    + 1 * einsum('ac,ijbe,idje,iejb->abcd', d_vv, D_oovv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    + 2 * einsum('ad,ijbe,ibje,icje->abcd', d_vv, D_oovv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    + 2 * einsum('ad,ijbe,iejb,iejc->abcd', d_vv, D_oovv, v_ovov, v_ovov))  # N^7: O^2V^5 / N^4: V^4

    itmd3 = itmd3 - einsum('abcd->bacd', itmd3) - einsum('abcd->abdc', itmd3) + einsum('abcd->badc', itmd3) + einsum('abcd->cdab', itmd3) + einsum('abcd->dcba', itmd3) - einsum('abcd->dcab', itmd3) - einsum('abcd->cdba', itmd3)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd}) to:
    itmd4 = (- 2 * einsum('ijac,iajd,icjb->abcd', D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    - 2 * einsum('ijac,ibjc,idja->abcd', D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    - 4 * einsum('ad,ijae,ibje,icje->abcd', d_vv, D_oovv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    - 4 * einsum('ad,ijae,iejb,iejc->abcd', d_vv, D_oovv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    - 2 * einsum('ac,ijae,ibje,iejd->abcd', d_vv, D_oovv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    - 2 * einsum('ac,ijae,idje,iejb->abcd', d_vv, D_oovv, v_ovov, v_ovov))  # N^7: O^2V^5 / N^4: V^4

    itmd4 = itmd4 - einsum('abcd->bacd', itmd4) - einsum('abcd->abdc', itmd4) + einsum('abcd->badc', itmd4)

    h_oooo = itmd0 + itmd1
    h_oovv = itmd2
    h_vvvv = itmd3 + itmd4

    return -_wrap_hessian(h_oooo, h_oovv, h_vvvv, n_occ, n_vir)

"""
Criterion 2a: Minimum correlation energy
"""
def _min_energy(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    return -_max_energy(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

def _min_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    return -_max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

def _min_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    return -_max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)


"""
Criterion 3: Maximum norm
"""

def _max_norm(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    if vec is not None:
        if not numpy.allclose(vec, numpy.zeros(vec.shape)):
            rmat = _vec_to_rotmat(vec, n_occ, n_vir)
            mo_coeff = mo_coeff @ scipy.linalg.expm(rmat)

    v_ovov, D_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir, chemist=True)

    renorm = 1.0
    if hasattr(ordmp, "renormalisation"):
        if isinstance(ordmp.renormalisation, (int, float)):
            renorm = ordmp.renormalisation

    # ================================================================================
    # EXPRESSION
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply 1 to:
    e = (+ 1
    + 2 * einsum('ijab,ijab,ibja,ibja->', D_oovv, D_oovv, v_ovov, v_ovov) * renorm   # N^4: O^2V^2 / N^4: O^2V^2
    - 1 * einsum('ijab,ijab,iajb,ibja->', D_oovv, D_oovv, v_ovov, v_ovov) * renorm)  # N^4: O^2V^2 / N^4: O^2V^2
   
    return -e

def _max_norm_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_ovov, D_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir, chemist=True)

    renorm = 1.0
    if hasattr(ordmp, "renormalisation"):
        if isinstance(ordmp.renormalisation, (int, float)):
            renorm = ordmp.renormalisation

    # ================================================================================
    # GRADIENT Block oo
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij}) to:
    itmd0 = (- 4 * einsum('jkab,jkab,ibka,jbka->ij', D_oovv, D_oovv, v_ovov, v_ovov)   # N^5: O^3V^2 / N^4: O^2V^2
    - 4 * einsum('jkab,jkab,jakb,kbia->ij', D_oovv, D_oovv, v_ovov, v_ovov)   # N^5: O^3V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,ikab,iakb,jbka->ij', D_oovv, D_oovv, v_ovov, v_ovov)   # N^5: O^3V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,ikab,iakb,kajb->ij', D_oovv, D_oovv, v_ovov, v_ovov))  # N^5: O^3V^2 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ij->ji', itmd0)

    # ================================================================================
    # GRADIENT Block vv
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab}) to:
    itmd1 = (- 4 * einsum('ijbc,ijbc,iajc,ibjc->ab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^5: O^2V^3 / N^4: O^2V^2
    - 4 * einsum('ijbc,ijbc,icja,icjb->ab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^5: O^2V^3 / N^4: O^2V^2
    - 2 * einsum('ijac,ijac,iajc,icjb->ab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^5: O^2V^3 / N^4: O^2V^2
    - 2 * einsum('ijac,ijac,ibjc,icja->ab', D_oovv, D_oovv, v_ovov, v_ovov))  # N^5: O^2V^3 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ab->ba', itmd1)

    return _wrap_gradient(itmd0, itmd1, n_occ, n_vir) * renorm

def _max_norm_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_ovov, D_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir, chemist=True)
   
    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)

    renorm = 1.0
    if hasattr(ordmp, "renormalisation"):
        if isinstance(ordmp.renormalisation, (int, float)):
            renorm = ordmp.renormalisation

    # ================================================================================
    # HESSIAN Block oooo
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl} + P_{ik}P_{jl} + P_{il}P_{jk} - P_{ij}P_{ik}P_{jl} - P_{ij}P_{il}P_{jk}) to:
    itmd0 = (- 4 * einsum('ilab,ilab,ibka,jbla->ijkl', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^4V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,ikab,ialb,jbka->ijkl', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^4V^2 / N^4: O^2V^2
    - 2 * einsum('ilab,ilab,ialb,jbka->ijkl', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^4V^2 / N^4: O^2V^2
    + 1 * einsum('imab,imab,jl,iamb,kbma->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 1 * einsum('imab,imab,jl,iamb,makb->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 2 * einsum('imab,imab,jk,iamb,mbla->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 2 * einsum('imab,imab,jk,ibma,lbma->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov))  # N^7: O^5V^2 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ijkl->jikl', itmd0) - einsum('ijkl->ijlk', itmd0) + einsum('ijkl->jilk', itmd0) + einsum('ijkl->klij', itmd0) + einsum('ijkl->lkji', itmd0) - einsum('ijkl->lkij', itmd0) - einsum('ijkl->klji', itmd0)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl}) to:
    itmd1 = (- 4 * einsum('imab,imab,il,jbma,kbma->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 4 * einsum('imab,imab,il,mbja,mbka->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * einsum('imab,imab,ik,jamb,lbma->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * einsum('imab,imab,ik,majb,mbla->ijkl', D_oovv, D_oovv, d_oo, v_ovov, v_ovov))  # N^7: O^5V^2 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ijkl->jikl', itmd1) - einsum('ijkl->ijlk', itmd1) + einsum('ijkl->jilk', itmd1)

    h_oooo = itmd0 + itmd1
    del itmd0, itmd1

    # ================================================================================
    # HESSIAN Block oovv
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{ab} + P_{ij}P_{ab}) to:
    itmd2 = (- 4 * einsum('ikbc,ikbc,iakc,jbkc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,ibkc,jakc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,ibkc,kcja->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,icka,jckb->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,ickb,jcka->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,ickb,kajc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,kaic,kbjc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikbc,ikbc,kcia,kcjb->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,iakc,jckb->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,iakc,kbjc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,ibkc,jcka->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,icka,jbkc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,icka,kcjb->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,ickb,jakc->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,kajc,kcib->ijab', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,ikac,kbic,kcja->ijab', D_oovv, D_oovv, v_ovov, v_ovov))  # N^6: O^3V^3 / N^4: O^2V^2

    itmd2 = itmd2 - einsum('ijab->jiab', itmd2) - einsum('ijab->ijba', itmd2) + einsum('ijab->jiba', itmd2)

    # ================================================================================
    # HESSIAN Block vvvv
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd} + P_{ac}P_{bd} + P_{ad}P_{bc} - P_{ab}P_{ac}P_{bd} - P_{ab}P_{ad}P_{bc}) to:
    itmd3 = (- 4 * einsum('ijad,ijad,iajc,ibjd->abcd', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    - 2 * einsum('ijad,ijad,iajd,ibjc->abcd', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    + 1 * einsum('ijae,ijae,bd,iaje,iejc->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    + 1 * einsum('ijae,ijae,bd,icje,ieja->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    + 2 * einsum('ijae,ijae,bc,iaje,idje->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    + 2 * einsum('ijae,ijae,bc,ieja,iejd->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov))  # N^7: O^2V^5 / N^4: V^4

    itmd3 = itmd3 - einsum('abcd->bacd', itmd3) - einsum('abcd->abdc', itmd3) + einsum('abcd->badc', itmd3) + einsum('abcd->cdab', itmd3) + einsum('abcd->dcba', itmd3) - einsum('abcd->dcab', itmd3) - einsum('abcd->cdba', itmd3)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd}) to:
    itmd4 = (- 2 * einsum('ijac,ijac,iajd,icjb->abcd', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    - 2 * einsum('ijac,ijac,ibjc,idja->abcd', D_oovv, D_oovv, v_ovov, v_ovov)   # N^6: O^2V^4 / N^4: V^4
    - 4 * einsum('ijae,ijae,ad,ibje,icje->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    - 4 * einsum('ijae,ijae,ad,iejb,iejc->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    - 2 * einsum('ijae,ijae,ac,ibje,iejd->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov)   # N^7: O^2V^5 / N^4: V^4
    - 2 * einsum('ijae,ijae,ac,idje,iejb->abcd', D_oovv, D_oovv, d_vv, v_ovov, v_ovov))  # N^7: O^2V^5 / N^4: V^4

    itmd4 = itmd4 - einsum('abcd->bacd', itmd4) - einsum('abcd->abdc', itmd4) + einsum('abcd->badc', itmd4)

    h_vvvv = itmd3 + itmd4
    del itmd3, itmd4

    return -_wrap_hessian(h_oooo, itmd2, h_vvvv, n_occ, n_vir) * renorm

"""
Criterion 4: Variational Priniciple Implementation 1
"""

def _hylleraas_2(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    energy = _max_energy(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = _max_norm(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    return energy / norm

def _hylleraas_2_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    zeros = numpy.zeros(n_occ_p + n_vir_p)

    energy = _max_energy(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = _max_norm(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    e_grad = _max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    n_grad = _max_norm_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    
    g = (e_grad * norm - n_grad * energy) / norm**2
    return g

def _hylleraas_2_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    zeros = numpy.zeros(n_occ_p + n_vir_p)

    energy = _max_energy(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = _max_norm(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    e_grad = _max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    n_grad = _max_norm_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    e_hess = _max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    n_hess = _max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

    e1_n2 = outer(e_grad, n_grad)
    n1_n2 = outer(n_grad, n_grad)

    h1 = (e_hess * norm**3 - norm**2 * (e1_n2 + e1_n2.T) - n_hess * energy * norm**2 
          + 2 * norm * n1_n2 * energy)

    return h1 / norm**4


"""
Criterion 4a: Renormalised Variational Principle: Indirect Implementation
"""

def _hylleraas_renorm(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    if not hasattr(ordmp, renormalisation):
        raise Exception("Renormalisation required as ordmp.renormalisation")
    if not isinstance(ordmp.renormalisation, (int, float)):
        raise Exception("Renormalisation must be a number")
    energy = _max_energy(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = _max_norm(ordmp, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = (norm - 1) * ordmp.renormalisation + 1
    return energy / norm

def _hylleraas_renorm_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    if not hasattr(ordmp, renormalisation):
        raise Exception("Renormalisation required as ordmp.renormalisation")
    if not isinstance(ordmp.renormalisation, (int, float)):
        raise Exception("Renormalisation must be a number")
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    zeros = numpy.zeros(n_occ_p + n_vir_p)
    
    energy = _max_energy(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = _max_norm(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    e_grad = _max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    n_grad = _max_norm_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    
    n_grad *= ordmp.renormalisation
    norm = (norm - 1) * ordmp.renormalisation + 1

    g = (e_grad * norm - n_grad * energy) / norm**2
    return g

def _hylleraas_renorm_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    if not hasattr(ordmp, renormalisation):
        raise Exception("Renormalisation required as ordmp.renormalisation")
    if not isinstance(ordmp.renormalisation, (int, float)):
        raise Exception("Renormalisation must be a number")
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    zeros = numpy.zeros(n_occ_p + n_vir_p)

    energy = _max_energy(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    norm = _max_norm(ordmp, zeros, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    e_grad = _max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    n_grad = _max_norm_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    e_hess = _max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
    n_hess = _max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

    norm = (norm - 1) * ordmp.renormalisation + 1
    n_grad *= ordmp.renormalisation
    n_hess *= ordmp.renormalisation

    e1_n2 = outer(e_grad, n_grad)
    n1_n2 = outer(n_grad, n_grad)

    h1 = (e_hess * norm**3 - norm**2 * (e1_n2 + e1_n2.T) - n_hess * energy * norm**2 
          + 2 * norm * n1_n2 * energy)

    return h1 / norm**4




