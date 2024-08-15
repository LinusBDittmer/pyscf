#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
REG2MP2
'''


import numpy
from opt_einsum import contract as einsum
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import __config__
from pyscf.mp.mp2 import _ChemistsERIs, get_nocc, get_nmo, _mem_usage

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)
PARAMETERS = 15

def energy(t2_1, v_ovov):
    # Apply 1 to:
    return einsum('ijab,iajb->', t2_1, v_ovov)  # N^4: O^2V^2 / N^4: O^2V^2

def H0_oo(t2_1, v_oooo, v_ovov, v_oovv, f_oo, f_vv, params):
    A = params
    d_oo = numpy.eye(f_oo.shape[1]) 
    nr = 0.5 / d_oo.shape[0]
    
    # Occupied-Occupied H0:

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply 1 to:
    itmd0 = (+ 1 * A[1] * einsum('ikab,jkab->ij', t2_1, v_oovv)   # N^7: O^5V^2 / N^4: O^2V^2
    - 1 * A[0] * einsum('ikab,jbka->ij', t2_1, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * A[1] * einsum('ikab,jbka->ij', t2_1, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    + 3 * A[0] * einsum('ikab,jakb->ij', t2_1, v_ovov)   # N^7: O^5V^2 / N^4: O^2V^2
    - 6 * A[5] * einsum('klac,klbc,ijab->ij', t2_1, t2_1, v_oovv)   # N^9: O^6V^3 / N^4: O^2V^2
    - 3 * A[3] * einsum('ilab,jkab,kl->ij', t2_1, t2_1, f_oo)   # N^8: O^6V^2 / N^4: O^2V^2
    - 3 * A[4] * einsum('jkac,ikbc,ab->ij', t2_1, t2_1, f_vv)   # N^8: O^5V^3 / N^4: O^2V^2
    - 3 * A[6] * einsum('klab,kmab,imjl->ij', t2_1, t2_1, v_oooo)   # N^9: O^7V^2 / N^4: O^2V^2
    + 3 * A[2] * einsum('ikab,klab,jl->ij', t2_1, t2_1, f_oo)   # N^8: O^6V^2 / N^4: O^2V^2
    + 3 * A[5] * einsum('klac,klbc,ibja->ij', t2_1, t2_1, v_ovov)   # N^9: O^6V^3 / N^4: O^2V^2
    + 6 * A[6] * einsum('klab,kmab,ijlm->ij', t2_1, t2_1, v_oooo)   # N^9: O^7V^2 / N^4: O^2V^2
    - 6 * A[0] * nr * einsum('klab,ij,kalb->ij', t2_1, d_oo, v_ovov)   # N^8: O^6V^2 / N^4: O^2V^2
    - 2 * A[1] * nr * einsum('klab,ij,klab->ij', t2_1, d_oo, v_oovv)   # N^8: O^6V^2 / N^4: O^2V^2
    + 2 * A[0] * nr * einsum('klab,ij,kbla->ij', t2_1, d_oo, v_ovov)   # N^8: O^6V^2 / N^4: O^2V^2
    + 4 * A[1] * nr * einsum('klab,ij,kbla->ij', t2_1, d_oo, v_ovov)   # N^8: O^6V^2 / N^4: O^2V^2
    - 24 * A[6] * nr * einsum('lmab,lnab,ij,kkmn->ij', t2_1, t2_1, d_oo, v_oooo)   # N^10: O^8V^2 / N^4: O^2V^2
    - 6 * A[5] * nr * einsum('klac,klbc,ij,mamb->ij', t2_1, t2_1, d_oo, v_ovov)   # N^10: O^7V^3 / N^4: O^2V^2
    + 6 * A[4] * nr * einsum('klac,klbc,ab,ij->ij', t2_1, t2_1, f_vv, d_oo)   # N^9: O^6V^3 / N^4: O^2V^2
    + 12 * A[2] * nr * einsum('klab,kmab,lm,ij->ij', t2_1, t2_1, f_oo, d_oo)   # N^9: O^7V^2 / N^4: O^2V^2
    + 12 * A[3] * nr * einsum('klab,kmab,lm,ij->ij', t2_1, t2_1, f_oo, d_oo)   # N^9: O^7V^2 / N^4: O^2V^2
    + 12 * A[5] * nr * einsum('klac,klbc,ij,mmab->ij', t2_1, t2_1, d_oo, v_oovv)   # N^10: O^7V^3 / N^4: O^2V^2
    + 12 * A[6] * nr * einsum('lmab,lnab,ij,kmkn->ij', t2_1, t2_1, d_oo, v_oooo)   # N^10: O^8V^2 / N^4: O^2V^2
    + 1 * f_oo)  # N^4: O^4 / N^2: O^2

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij}) to:
    itmd1 = - 3 * A[7] * einsum('klac,klbc,iajb->ij', t2_1, t2_1, v_ovov)   # N^9: O^6V^3 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ij->ji', itmd1)

    h0_oo = itmd0 + itmd1
    h0_oo = 0.5 * (h0_oo + h0_oo.conj().T).real

    return h0_oo 

def H0_vv(t2_1, v_oooo, v_ovov, v_oovv, f_oo, f_vv, params):
    A = params
    
    # Virtual-Virtual H0:

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply 1 to:
    itmd0 = (- 1 * A[8] * einsum('ijac,icjb->ab', t2_1, v_ovov)   # N^7: O^2V^5 / N^4: O^2V^2
    - 1 * A[9] * einsum('ijac,ijbc->ab', t2_1, v_oovv)   # N^7: O^2V^5 / N^4: O^2V^2
    + 2 * A[9] * einsum('ijac,ibjc->ab', t2_1, v_ovov)   # N^7: O^2V^5 / N^4: O^2V^2
    + 3 * A[8] * einsum('ijac,ibjc->ab', t2_1, v_ovov)   # N^7: O^2V^5 / N^4: O^2V^2
    - 6 * A[13] * einsum('ijcd,ikcd,jkab->ab', t2_1, t2_1, v_oovv)   # N^9: O^3V^6 / N^4: O^2V^2
    - 3 * A[10] * einsum('ijac,ikbc,jk->ab', t2_1, t2_1, f_oo)   # N^8: O^3V^5 / N^4: O^2V^2
    - 3 * A[11] * einsum('ijac,ijbd,cd->ab', t2_1, t2_1, f_vv)   # N^8: O^2V^6 / N^4: O^2V^2
    + 3 * A[12] * einsum('ijac,ijcd,bd->ab', t2_1, t2_1, f_vv)   # N^8: O^2V^6 / N^4: O^2V^2
    + 3 * A[13] * einsum('ijcd,ikcd,jbka->ab', t2_1, t2_1, v_ovov)   # N^9: O^3V^6 / N^4: O^2V^2
    + 1 * f_vv)  # N^4: V^4 / N^2: V^2

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab}) to:
    itmd1 = - 3 * A[14] * einsum('ijcd,ikcd,jbka->ab', t2_1, t2_1, v_ovov)   # N^9: O^3V^6 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ab->ba', itmd1)

    h0_vv = itmd0 + itmd1
    h0_vv = 0.5 * (h0_vv + h0_vv.conj().T).real

    return h0_vv

def make_eris(mp, mo_coeff, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mp, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((co,cv,co,cv)).reshape(nocc*nvir,nocc*nvir)
            eris.oooo = ao2mofn((co,co,co,co)).reshape(nocc*nocc,nocc*nocc)
            eris.oovv = ao2mofn((co,co,cv,cv)).reshape(nocc*nocc,nvir*nvir)
            #eris.vvvv = ao2mofn((cv,cv,cv,cv)).reshape(nvir*nvir,nvir*nvir)
            #eris.ovvv = ao2mofn((co,cv,cv,cv)).reshape(nocc*nvir,nvir*nvir)
            #eris.ooov = ao2mofn((co,co,co,cv)).reshape(nocc*nocc,nocc*nvir)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv), compact=False)
            eris.oooo = ao2mo.general(mp._scf._eri, (co,co,co,co), compact=False)
            eris.oovv = ao2mo.general(mp._scf._eri, (co,co,cv,cv), compact=False)
            #eris.vvvv = ao2mo.general(mp._scf._eri, (cv,cv,cv,cv), compact=False)
            #eris.ovvv = ao2mo.general(mp._scf._eri, (co,cv,cv,cv), compact=False)
            #eris.ooov = ao2mo.general(mp._scf._eri, (co,co,co,cv), compact=False)

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                 '3-tensor integrals.\n'
                 'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))
        eris.oooo = mp._scf.with_df.ao2mo((co,co,co,co))
        eris.vvvv = mp._scf.with_df.ao2mo((cv,cv,cv,cv))

    else:
        raise NotImplementedError('Outcore ERI transformation not implemented')

    log.timer('Integral transformation', *time0)
    return eris

def eri_transform(mp, mo_coeff, nocc, nvir):
    eris = make_eris(mp, mo_coeff)
    v_ovov = numpy.empty((nocc,nvir,nocc,nvir), dtype=eris.ovov.dtype)
    v_oooo = numpy.empty((nocc,nocc,nocc,nocc), dtype=eris.oooo.dtype)
    v_oovv = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.oovv.dtype)

    if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
        v_ovov = eris.ovov
    else:
        v_ovov = numpy.asarray(eris.ovov).reshape(nocc,nvir,nocc,nvir)
    if isinstance(eris.oooo, numpy.ndarray) and eris.oooo.ndim == 4:
        v_oooo = eris.oooo
    else:
        v_oooo = numpy.asarray(eris.oooo).reshape(nocc,nocc,nocc,nocc)
    if isinstance(eris.oovv, numpy.ndarray) and eris.oovv.ndim == 4:
        v_oovv = eris.oovv
    else:
        v_oovv = numpy.asarray(eris.oovv).reshape(nocc,nocc,nvir,nvir)

    return v_oooo, v_ovov, v_oovv

def make_diag_t2(v_ovov, mo_energy, nocc, nvir):
    t2 = numpy.empty((nocc, nocc, nvir, nvir), dtype=v_ovov.dtype)
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    for i in range(nocc):
        gi = 2 * v_ovov[i].transpose(1, 0, 2) - einsum('ajb->jba', v_ovov[i])
        t2[i] = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])

    return t2

def eigh(mat):
    mat = numpy.nan_to_num(mat)
    e, c = numpy.linalg.eigh(mat)
    idx = numpy.argmax(abs(c.real), axis=0)
    c[:,c[idx,numpy.arange(len(e))].real<0] *= -1
    return e, c

def kernel(mp, mo_energy=None, mo_coeff=None, t2_guess=None, parameters=None):
    nocc = mp.nocc
    nvir = mp.nmo - nocc

    if mo_coeff is None:
        mo_coeff = mp.mo_coeff
    if mo_energy is None:
        mo_energy = mp._scf.mo_energy
    if parameters is None:
        parameters = mp.parameters

    fock = numpy.diag(mo_energy)
    v_oooo, v_ovov, v_oovv = eri_transform(mp, mo_coeff, nocc, nvir)
    if t2_guess is None:
        t2 = make_diag_t2(v_ovov, mo_energy, nocc, nvir)
    elif t2_guess.shape == (nocc, nocc, nvir, nvir):
        t2 = t2_guess
    else:
        t2 = t2_guess.reshape(nocc, nocc, nvir, nvir)

    # Iterative solution for the t amplitudes
    t2_old = numpy.copy(t2)
    converged = False
    cycle = 0
    e_corr_old = 0.0
    for cycle in range(mp.max_cycle):
        if converged:
            break
        # Generate R_oo and R_vv
        h0_oo = H0_oo(t2_old, v_oooo, v_ovov, v_oovv, fock[:nocc,:nocc], 
                      fock[nocc:,nocc:], parameters)
        h0_vv = H0_vv(t2_old, v_oooo, v_ovov, v_oovv, fock[:nocc,:nocc], 
                      fock[nocc:,nocc:], parameters)
        
        # Diagonalise H0 in orbital space
        mo_energy_occ, u_occ = eigh(h0_oo)
        mo_energy_vir, u_vir = eigh(h0_vv)

        # Rotate orbitals
        u_mat = numpy.zeros((nocc+nvir, nocc+nvir))
        u_mat[:nocc,:nocc] = u_occ
        u_mat[nocc:,nocc:] = u_vir
        mo_energy = numpy.zeros(nocc+nvir, dtype=mo_energy_occ.dtype)
        mo_energy[:nocc] = mo_energy_occ
        mo_energy[nocc:] = mo_energy_vir
        mo_coeff = mo_coeff @ u_mat

        # Transform ERIs
        v_oooo, v_ovov, v_oovv = eri_transform(mp, mo_coeff, nocc, nvir)
        # Generate new t2 amps
        t2 = make_diag_t2(v_ovov, mo_energy, nocc, nvir)

        # Transform Fock matrix
        fock = u_mat.conj().T @ fock @ u_mat

        # Check convergence
        evec = t2 - t2_old
        error_t = numpy.linalg.norm(evec)
        e_corr = energy(t2, v_ovov)
        error_e = numpy.abs(e_corr - e_corr_old)
        
        t2_old = numpy.copy(t2)
        e_corr_old = e_corr

        print(f"Reg2MP2 Iteration {cycle}: T2-Error: {error_t}, Ec-Error: {error_e}")
        print(f"HOMO-LUMO gap: {mo_energy[nocc] - mo_energy[nocc-1]}")
        #if error_t < 10**-(0.5*mp.convergence) and error_e < 10**-mp.convergence:
        #    converged = True
        if error_e < 10**-mp.convergence:
            converged = True

    if converged:
        print(f"Reg2-MP2 converged within {cycle+1} cycles")
    else:
        print(f"Reg2-MP2 did not converge within {cycle+1} cycles")

    return e_corr, t2, mo_coeff, mo_energy, converged

class Reg2MP2(lib.StreamObject):

    def __init__(self, mf, parameters=None, mo_coeff=None, mo_occ=None):

        if mo_coeff is None: mo_coeff = numpy.array(mf.mo_coeff)
        if mo_occ is None: mo_occ = numpy.array(mf.mo_occ)

        self.mol = mf.mol
        self._scf = mf
        if parameters is None: parameters = numpy.zeros(PARAMETERS)
        assert len(parameters) == PARAMETERS
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        
        self.convergence = 5
        self.max_cycle = 250

        self.frozen = None

        self.parameters = parameters
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_hf = None
        self.e_corr = None
        self.t2 = None
        self.converged = None
    
    @property
    def nocc(self):
        return get_nocc(self)
    
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return get_nmo(self)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    @property
    def emp2(self):
        return self.e_corr

    def get_e_hf(mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
        vhf = mp._scf.get_veff(mp._scf.mol, dm)
        return mp._scf.energy_tot(dm=dm, vhf=vhf)

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        self.e_corr, self.t2, self.mo_coeff, self.mo_energy, self.converged = kernel(self, mo_energy, mo_coeff, parameters=self.parameters)

        return self.e_corr, self.t2, self.converged

    def num_param_gradient(self, epsilon=10**-10):
        gradient = numpy.zeros(self.parameters.shape)
        for i in range(PARAMETERS):
            param0 = numpy.array(self.parameters)
            param1 = numpy.array(self.parameters)
            param0[i] -= epsilon
            param1[i] += epsilon
            e0, _, _, _, _ = kernel(self, None, None, parameters=param0)
            e1, _, _, _, _ = kernel(self, None, None, parameters=param1)
            gradient[i] = (e1 - e0) / (2 * epsilon)
        return gradient

    def param_gradient(self, pm4=False):
        gradient = numpy.zeros(self.parameters.shape)

        if self.t2 is None:
            self.kernel()
        
        v_oooo, v_ovov, v_oovv = eri_transform(self, self.mo_coeff, self.nocc, self.nmo-self.nocc)
        fock_matrix = self.mo_coeff.T @ self._scf.get_fock() @ self.mo_coeff
        f_oo = fock_matrix[:self.nocc,:self.nocc]
        f_vv = fock_matrix[self.nocc:,self.nocc:]
        A = numpy.ones(self.parameters.shape)
        t2_1 = self.t2
        d_oo = numpy.eye(f_oo.shape[1])
        nr = 0.5 / d_oo.shape[0]
        no = self.nocc
        nv = self.nmo - no
        mo_energy = self.mo_energy
        delta = -(mo_energy[:no,None,None,None] + mo_energy[None,:no,None,None]  
                - mo_energy[None,None,no:,None] - mo_energy[None,None,None,no:])
        delta = 1 / delta

        def fill_gradient_occ(w, idx):
            w = 0.5 * (w + w.conj().T).real
            wt = einsum('ikab,jk->ijab', t2_1, w)
            # wt -= einsum('ijab->jiab', wt)
            gradient[idx] = 2 * einsum('ijab,ijab,iajb->', delta, wt, v_ovov)

        def fill_gradient_vir(w, idx):
            w = 0.5 * (w + w.conj().T).real
            wt = einsum('ijbc,ac->ijab', t2_1, w)
            # wt -= einsum('ijab->ijba', wt)
            gradient[idx] = -8 * einsum('ijab,ijab,iajb->', delta, wt, v_ovov)

        # A0
        w = (- 1 * A[0] * einsum('ikab,jbka->ij', t2_1, v_ovov)
            + 3 * A[0] * einsum('ikab,jakb->ij', t2_1, v_ovov)
            - 6 * A[0] * nr * einsum('klab,ij,kalb->ij', t2_1, d_oo, v_ovov)
            + 2 * A[0] * nr * einsum('klab,ij,kbla->ij', t2_1, d_oo, v_ovov))
        fill_gradient_occ(w, 0)

        # A1
        w = (+ 1 * A[1] * einsum('ikab,jkab->ij', t2_1, v_oovv)
            - 2 * A[1] * einsum('ikab,jbka->ij', t2_1, v_ovov)
            - 2 * A[1] * nr * einsum('klab,ij,klab->ij', t2_1, d_oo, v_oovv)
            + 4 * A[1] * nr * einsum('klab,ij,kbla->ij', t2_1, d_oo, v_ovov))
        fill_gradient_occ(w, 1)

        # A2
        w = (+ 3 * A[2] * einsum('ikab,klab,jl->ij', t2_1, t2_1, f_oo)
            + 12 * A[2] * nr * einsum('klab,kmab,lm,ij->ij', t2_1, t2_1, f_oo, d_oo))
        fill_gradient_occ(w, 2)

        # A3
        w = (- 3 * A[3] * einsum('ilab,jkab,kl->ij', t2_1, t2_1, f_oo)
            + 12 * A[3] * nr * einsum('klab,kmab,lm,ij->ij', t2_1, t2_1, f_oo, d_oo))
        fill_gradient_occ(w, 3)

        # A4
        w = (- 3 * A[4] * einsum('jkac,ikbc,ab->ij', t2_1, t2_1, f_vv)
            + 6 * A[4] * nr * einsum('klac,klbc,ab,ij->ij', t2_1, t2_1, f_vv, d_oo))
        fill_gradient_occ(w, 4)

        # A5
        w = (- 6 * A[5] * einsum('klac,klbc,ijab->ij', t2_1, t2_1, v_oovv)
            + 3 * A[5] * einsum('klac,klbc,ibja->ij', t2_1, t2_1, v_ovov)
            - 6 * A[5] * nr * einsum('klac,klbc,ij,mamb->ij', t2_1, t2_1, d_oo, v_ovov)
            + 12 * A[5] * nr * einsum('klac,klbc,ij,mmab->ij', t2_1, t2_1, d_oo, v_oovv))
        fill_gradient_occ(w, 5)

        # A6
        w = (- 3 * A[6] * einsum('klab,kmab,imjl->ij', t2_1, t2_1, v_oooo)
            + 6 * A[6] * einsum('klab,kmab,ijlm->ij', t2_1, t2_1, v_oooo)
            - 24 * A[6] * nr * einsum('lmab,lnab,ij,kkmn->ij', t2_1, t2_1, d_oo, v_oooo)
            + 12 * A[6] * nr * einsum('lmab,lnab,ij,kmkn->ij', t2_1, t2_1, d_oo, v_oooo))
        fill_gradient_occ(w, 6)

        # A7
        w = (- 3 * A[7] * einsum('klac,klbc,iajb->ij', t2_1, t2_1, v_ovov))
        w = w - w.T
        fill_gradient_occ(w, 7)

        # VIRTUAL
        # A8
        w = (- 1 * A[8] * einsum('ijac,icjb->ab', t2_1, v_ovov)
            + 3 * A[8] * einsum('ijac,ibjc->ab', t2_1, v_ovov))
        fill_gradient_vir(w, 8)

        # A9
        w = (- 1 * A[9] * einsum('ijac,ijbc->ab', t2_1, v_oovv)
            + 2 * A[9] * einsum('ijac,ibjc->ab', t2_1, v_ovov))
        fill_gradient_vir(w, 9)

        # A10
        w = (- 3 * A[10] * einsum('ijac,ikbc,jk->ab', t2_1, t2_1, f_oo))
        fill_gradient_vir(w, 10)

        # A11
        w = (- 3 * A[11] * einsum('ijac,ijbd,cd->ab', t2_1, t2_1, f_vv))
        fill_gradient_vir(w, 11)

        # A12
        w = (+ 3 * A[12] * einsum('ijac,ijcd,bd->ab', t2_1, t2_1, f_vv))
        fill_gradient_vir(w, 12)

        # A13
        w = (- 6 * A[13] * einsum('ijcd,ikcd,jkab->ab', t2_1, t2_1, v_oovv)
            + 3 * A[13] * einsum('ijcd,ikcd,jbka->ab', t2_1, t2_1, v_ovov))
        fill_gradient_vir(w, 13)

        # A14
        w = (- 3 * A[14] * einsum('ijcd,ikcd,jbka->ab', t2_1, t2_1, v_ovov))
        w = w - w.T
        fill_gradient_vir(w, 14)

        if pm4:
            # Only A0 and A8 remain
            gradient[1:7] = 0
            gradient[9:] = 0

        return gradient


if __name__ == '__main__':
    from pyscf import scf, mp
    numpy.set_printoptions(linewidth=250, precision=5, suppress=True)
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    mo = numpy.array(mf.mo_energy.copy())

    reg2mp2 = Reg2MP2(mf)
    #reg2mp2.parameters[0] = 1.0
    reg2mp2.kernel()

    param1 = reg2mp2.param_gradient()
    param2 = reg2mp2.num_param_gradient()

    print(param1)
    print(param2)
    print(param2 / param1)

