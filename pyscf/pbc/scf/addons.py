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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
import scipy.special
import scipy.optimize
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.lib import logger
from pyscf.scf import addons as mol_addons
from pyscf.pbc.lib.kpts import KPoints
from pyscf import __config__

SMEARING_METHOD = mol_addons.SMEARING_METHOD


def project_mo_nr2nr(cell1, mo1, cell2, kpts=None):
    r''' Project orbital coefficients

    .. math::

        |\psi1> = |AO1> C1

        |\psi2> = P |\psi1> = |AO2>S^{-1}<AO2| AO1> C1 = |AO2> C2

        C2 = S^{-1}<AO2|AO1> C1
    '''
    s22 = cell2.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    s21 = pbcgto.intor_cross('int1e_ovlp', cell2, cell1, kpts=kpts)
    if kpts is None or numpy.shape(kpts) == (3,):  # A single k-point
        return scipy.linalg.solve(s22, s21.dot(mo1), assume_a='pos')
    else:
        assert (len(kpts) == len(mo1))
        return [scipy.linalg.solve(s22[k], s21[k].dot(mo1[k]), sym_pos=True)
                for k, kpt in enumerate(kpts)]


def smearing_(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Fermi-Dirac or Gaussian smearing'''
    from pyscf.scf import uhf
    from pyscf.scf import ghf
    from pyscf.pbc.scf import khf
    mf_class = mf.__class__
    is_uhf = isinstance(mf, uhf.UHF)
    is_ghf = isinstance(mf, ghf.GHF)
    is_rhf = (not is_uhf) and (not is_ghf)
    is_khf = isinstance(mf, khf.KSCF)

    if fix_spin and not is_uhf:
        raise KeyError("fix_spin only supports UHF.")
    if fix_spin and mu0 is not None:
        raise KeyError("fix_spin does not support fix mu0")

    def fermi_smearing_occ(m, mo_energy_kpts, sigma):
        occ = numpy.zeros_like(mo_energy_kpts)
        de = (mo_energy_kpts - m) / sigma
        occ[de<40] = 1./(numpy.exp(de[de<40])+1.)
        return occ
    def gaussian_smearing_occ(m, mo_energy_kpts, sigma):
        return 0.5 * scipy.special.erfc((mo_energy_kpts - m) / sigma)

    def partition_occ(mo_occ, mo_energy_kpts):
        mo_occ_kpts = []
        p1 = 0
        for e in mo_energy_kpts:
            p0, p1 = p1, p1 + e.size
            occ = mo_occ[p0:p1]
            mo_occ_kpts.append(occ)
        return mo_occ_kpts

    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        kpts = getattr(mf, 'kpts', None)
        if isinstance(kpts, KPoints):
            mo_energy_kpts = kpts.transform_mo_energy(mo_energy_kpts)

        #mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
            return mo_occ_kpts

        if is_khf:
            if isinstance(kpts, KPoints):
                nkpts = kpts.nkpts
            else:
                nkpts = len(kpts)
        else:
            nkpts = 1
        if isinstance(mf.mol, pbcgto.Cell):
            nelectron = mf.mol.tot_electrons(nkpts)
        else:
            nelectron = mf.mol.tot_electrons()
        if is_uhf:
            nocc = nelectron
            if fix_spin:
                nocc = mf.nelec
                mo_es = []
                mo_es.append(numpy.hstack(mo_energy_kpts[0]))
                mo_es.append(numpy.hstack(mo_energy_kpts[1]))
            else:
                mo_es = numpy.append(numpy.hstack(mo_energy_kpts[0]),
                                     numpy.hstack(mo_energy_kpts[1]))
        elif is_ghf:
            nocc = nelectron
            mo_es = numpy.hstack(mo_energy_kpts)
        else:
            nocc = nelectron // 2
            mo_es = numpy.hstack(mo_energy_kpts)

        if mf.smearing_method.lower() == 'fermi':  # Fermi-Dirac smearing
            f_occ = fermi_smearing_occ
        else:  # Gaussian smearing
            f_occ = gaussian_smearing_occ

        if fix_spin:
            mo_energy = []
            mo_energy.append(numpy.sort(mo_es[0].ravel()))
            mo_energy.append(numpy.sort(mo_es[1].ravel()))
        else:
            mo_energy = numpy.sort(mo_es.ravel())

        sigma = mf.sigma
        if fix_spin:
            fermi = [mo_energy[0][nocc[0]-1], mo_energy[1][nocc[1]-1]]
        else:
            fermi = mo_energy[nocc-1]
        if mu0 is None:
            def nelec_cost_fn(m, _mo_es, _nelectron):
                mo_occ_kpts = f_occ(m, _mo_es, sigma)
                if is_rhf:
                    mo_occ_kpts *= 2
                return (mo_occ_kpts.sum() - _nelectron)**2
            if fix_spin:
                mu = []
                mo_occs = []
                res = scipy.optimize.minimize(nelec_cost_fn, fermi[0], args=(mo_es[0], nocc[0]), method='Powell',
                                              options={'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 10000})
                mu.append(res.x)
                mo_occs.append(f_occ(mu[0], mo_es[0], sigma))
                res = scipy.optimize.minimize(nelec_cost_fn, fermi[1], args=(mo_es[1], nocc[1]), method='Powell',
                                              options={'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 10000})
                mu.append(res.x)
                mo_occs.append(f_occ(mu[1], mo_es[1], sigma))
                f = copy.copy(mo_occs)
            else:
                res = scipy.optimize.minimize(nelec_cost_fn, fermi, args=(mo_es, nelectron), method='Powell',
                                              options={'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 10000})
                mu = res.x
                mo_occs = f = f_occ(mu, mo_es, sigma)
        else:
            # If mu0 is given, fix mu instead of electron number. XXX -Chong Sun
            mu = mu0
            mo_occs = f = f_occ(mu, mo_es, sigma)

        # See https://www.vasp.at/vasp-workshop/slides/k-points.pdf
        if mf.smearing_method.lower() == 'fermi':
            if fix_spin:
                f[0] = f[0][(f[0]>0) & (f[0]<1)]
                mf.entropy = -(f[0]*numpy.log(f[0]) + (1-f[0])*numpy.log(1-f[0])).sum() / nkpts
                f[1] = f[1][(f[1]>0) & (f[1]<1)]
                mf.entropy += -(f[1]*numpy.log(f[1]) + (1-f[1])*numpy.log(1-f[1])).sum() / nkpts
            else:
                f = f[(f>0) & (f<1)]
                mf.entropy = -(f*numpy.log(f) + (1-f)*numpy.log(1-f)).sum() / nkpts
        else:
            if fix_spin:
                mf.entropy = (numpy.exp(-((mo_es[0]-mu[0])/mf.sigma)**2).sum()
                              / (2*numpy.sqrt(numpy.pi)) / nkpts)
                mf.entropy += (numpy.exp(-((mo_es[1]-mu[1])/mf.sigma)**2).sum()
                               / (2*numpy.sqrt(numpy.pi)) / nkpts)
            else:
                mf.entropy = (numpy.exp(-((mo_es-mu)/mf.sigma)**2).sum()
                              / (2*numpy.sqrt(numpy.pi)) / nkpts)
        if is_rhf:
            mo_occs *= 2
            mf.entropy *= 2

        # DO NOT use numpy.array for mo_occ_kpts and mo_energy_kpts, they may
        # have different dimensions for different k-points
        if is_uhf:
            if is_khf:
                if fix_spin:
                    mo_occ_kpts =(partition_occ(mo_occs[0], mo_energy_kpts[0]),
                                  partition_occ(mo_occs[1], mo_energy_kpts[1]))
                else:
                    nao_tot = mo_occs.size//2
                    mo_occ_kpts =(partition_occ(mo_occs[:nao_tot], mo_energy_kpts[0]),
                                  partition_occ(mo_occs[nao_tot:], mo_energy_kpts[1]))
            else:
                if numpy.isscalar(self.mu0):
                    mu_a = mu_b = self.mu0
                elif len(self.mu0) == 2:
                    mu_a, mu_b = self.mu0
                else:
                    raise TypeError(f'Unsupported mu0: {self.mu0}')
                occa = f_occ(mu_a, mo_es[0], sigma)
                occb = f_occ(mu_b, mo_es[1], sigma)
            mu = [mu_a, mu_b]
            mo_occs = [occa, occb]
            self.entropy  = self._get_entropy(mo_es[0], mo_occs[0], mu[0])
            self.entropy += self._get_entropy(mo_es[1], mo_occs[1], mu[1])
            self.entropy /= nkpts

            fermi = (mol_addons._get_fermi(mo_es[0], nocc[0]),
                     mol_addons._get_fermi(mo_es[1], nocc[1]))
            logger.debug(self, '    Alpha-spin Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi[0], mo_occs[0].sum(), nocc[0])
            logger.debug(self, '    Beta-spin  Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi[1], mo_occs[1].sum(), nocc[1])
            logger.info(self, '    sigma = %g  Optimized mu_alpha = %.12g  entropy = %.12g',
                        sigma, mu[0], self.entropy)
            logger.info(self, '    sigma = %g  Optimized mu_beta  = %.12g  entropy = %.12g',
                        sigma, mu[1], self.entropy)

            mo_occ_kpts =(_partition_occ(mo_occs[0], mo_energy_kpts[0]),
                          _partition_occ(mo_occs[1], mo_energy_kpts[1]))
            tools.print_mo_energy_occ_kpts(self, mo_energy_kpts, mo_occ_kpts, True)
            if is_rohf:
                mo_occ_kpts = numpy.array(mo_occ_kpts, dtype=numpy.float64).sum(axis=0)
        else:
            nocc = nelectron = self.mol.tot_electrons(nkpts)
            if is_uhf:
                mo_es_a = numpy.hstack(mo_energy_kpts[0])
                mo_es_b = numpy.hstack(mo_energy_kpts[1])
                mo_es = numpy.append(mo_es_a, mo_es_b)
            else:
                mo_es = numpy.hstack(mo_energy_kpts)
            if is_rhf:
                nocc = (nelectron + 1) // 2

        if fix_spin:
            logger.debug(mf, '    Alpha-spin Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi[0], mo_occs[0].sum(), nocc[0])
            logger.debug(mf, '    Beta-spin  Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi[1], mo_occs[1].sum(), nocc[1])
            logger.info(mf, '    sigma = %g  Optimized mu_alpha = %.12g  entropy = %.12g',
                        mf.sigma, mu[0], mf.entropy)
            logger.info(mf, '    sigma = %g  Optimized mu_beta  = %.12g  entropy = %.12g',
                        mf.sigma, mu[1], mf.entropy)
        else:
            logger.debug(mf, '    Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi, mo_occs.sum(), nelectron)
            logger.info(mf, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                        mf.sigma, mu, mf.entropy)

        kpts = getattr(mf, 'kpts', None)
        if isinstance(kpts, KPoints):
            if is_uhf:
                mo_occ_kpts = (kpts.check_mo_occ_symmetry(mo_occ_kpts[0]),
                               kpts.check_mo_occ_symmetry(mo_occ_kpts[1]))
            else:
                mo_occ_kpts = kpts.check_mo_occ_symmetry(mo_occ_kpts)

        if is_khf:
            tools.print_mo_energy_occ_kpts(mf,mo_energy_kpts,mo_occ_kpts,is_uhf)
        else:
            tools.print_mo_energy_occ(mf,mo_energy_kpts,mo_occ_kpts,is_uhf)
        return mo_occ_kpts

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            return super().get_grad(mo_coeff_kpts, mo_occ_kpts, fock)

        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore() + self.get_veff(self.mol, dm1)
        if self.istype('KUHF'):
            ga = _get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = _get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return numpy.hstack((ga,gb))
        else: # rhf and ghf
            return _get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Fermi-Dirac or Gaussian smearing'''
    from pyscf.pbc.scf import khf
    if not isinstance(mf, khf.KSCF):
        return mol_addons.smearing(mf, sigma, method, mu0, fix_spin)

    if isinstance(mf, mol_addons._SmearingSCF):
        mf.sigma = sigma
        mf.smearing_method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    return lib.set_class(_SmearingKSCF(mf, sigma, method, mu0, fix_spin),
                         (_SmearingKSCF, mf.__class__))

def smearing_(mf, *args, **kwargs):
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf

def canonical_occ_(mf, nelec=None):
    '''Label the occupancies for each orbital for sampled k-points.
    This is for KUHF objects.
    Each k-point has a fixed number of up and down electrons in this,
    which results in a finite size error for metallic systems
    but can accelerate convergence.
    '''
    from pyscf.pbc.scf import kuhf
    assert (isinstance(mf, kuhf.KUHF))

    def get_occ(mo_energy_kpts=None, mo_coeff=None):
        if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

        if nelec is None:
            cell_nelec = mf.cell.nelec
        else:
            cell_nelec = nelec

        homo=[-1e8,-1e8]
        lumo=[1e8,1e8]
        mo_occ_kpts = [[], []]
        for s in [0,1]:
            for k, mo_energy in enumerate(mo_energy_kpts[s]):
                e_idx = numpy.argsort(mo_energy)
                e_sort = mo_energy[e_idx]
                n = cell_nelec[s]
                mo_occ = numpy.zeros_like(mo_energy)
                mo_occ[e_idx[:n]] = 1
                homo[s] = max(homo[s], e_sort[n-1])
                lumo[s] = min(lumo[s], e_sort[n])
                mo_occ_kpts[s].append(mo_occ)

        for nm,s in zip(['alpha','beta'],[0,1]):
            logger.info(mf, nm+' HOMO = %.12g  LUMO = %.12g', homo[s], lumo[s])
            if homo[s] > lumo[s]:
                logger.warn(mf, "WARNING! HOMO is greater than LUMO! "
                            "This may lead to incorrect canonical occupation.")

        return mo_occ_kpts

    mf.get_occ = get_occ
    return mf
canonical_occ = canonical_occ_


def convert_to_uhf(mf, out=None):
    '''Convert the given mean-field object to the corresponding unrestricted
    HF/KS object
    '''
    from pyscf.pbc import scf
    from pyscf.pbc import dft

    if out is None:
        if isinstance(mf, (scf.uhf.UHF, scf.kuhf.KUHF)):
            return mf.copy()
        else:
            unknown_cls = [scf.kghf.KGHF]
            for i, cls in enumerate(mf.__class__.__mro__):
                if cls in unknown_cls:
                    raise NotImplementedError(
                        "No conversion from %s to uhf object" % cls)

            known_cls = {dft.krks.KRKS  : dft.kuks.KUKS,
                         dft.krks_ksymm.KRKS  : dft.kuks_ksymm.KUKS,
                         dft.kroks.KROKS: dft.kuks.KUKS,
                         scf.khf.KRHF   : scf.kuhf.KUHF,
                         scf.khf_ksymm.KRHF : scf.kuhf_ksymm.KUHF,
                         scf.krohf.KROHF: scf.kuhf.KUHF,
                         dft.rks.RKS    : dft.uks.UKS  ,
                         dft.roks.ROKS  : dft.uks.UKS  ,
                         scf.hf.RHF     : scf.uhf.UHF  ,
                         scf.rohf.ROHF  : scf.uhf.UHF  ,}
            # .with_df should never be removed or changed during the conversion.
            # It is needed to compute JK matrix in all pbc SCF objects
            out = mol_addons._object_without_soscf(mf, known_cls, False)
    else:
        assert (isinstance(out, (scf.uhf.UHF, scf.kuhf.KUHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert (isinstance(out, scf.khf.KSCF))
        else:
            assert (not isinstance(out, scf.khf.KSCF))

    if out is not None:
        out = mol_addons._update_mf_without_soscf(mf, out, False)

    return mol_addons._update_mo_to_uhf_(mf, out)

def convert_to_rhf(mf, out=None):
    '''Convert the given mean-field object to the corresponding restricted
    HF/KS object
    '''
    from pyscf.pbc import scf
    from pyscf.pbc import dft

    if getattr(mf, 'nelec', None) is None:
        nelec = mf.cell.nelec
    else:
        nelec = mf.nelec

    if out is not None:
        assert (isinstance(out, (scf.hf.RHF, scf.khf.KRHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert (isinstance(out, scf.khf.KSCF))
        else:
            assert (not isinstance(out, scf.khf.KSCF))

    elif nelec[0] != nelec[1] and isinstance(mf, (scf.rohf.ROHF, scf.krohf.KROHF)):
        if getattr(mf, '_scf', None):
            return mol_addons._update_mf_without_soscf(mf, mf._scf.copy(), False)
        else:
            return mf.copy()

    else:
        if isinstance(mf, (scf.hf.RHF, scf.khf.KRHF)):
            return mf.copy()
        else:
            if isinstance(mf, (scf.ghf.GHF, scf.kghf.KGHF)):
                raise NotImplementedError(
                    f'No conversion from {mf.__class__} to rhf object')

            if nelec[0] == nelec[1]:
                known_cls = {dft.kuks.KUKS : dft.krks.KRKS,
                             dft.kuks_ksymm.KUKS : dft.krks_ksymm.KRKS,
                             scf.kuhf.KUHF : scf.khf.KRHF ,
                             scf.kuhf_ksymm.KUHF : scf.khf_ksymm.KRHF,
                             dft.uks.UKS   : dft.rks.RKS  ,
                             scf.uhf.UHF   : scf.hf.RHF   ,
                             dft.kroks.KROKS : dft.krks.KRKS,
                             scf.krohf.KROHF : scf.khf.KRHF ,
                             dft.roks.ROKS   : dft.rks.RKS  ,
                             scf.rohf.ROHF   : scf.hf.RHF   }
            else:
                known_cls = {
                    dft.kuks.KUKS : dft.kroks.KROKS,
                    dft.uks.UKS   : dft.roks.ROKS  ,
                    scf.kuhf.KUHF : scf.krohf.KROHF,
                    scf.uhf.UHF   : scf.rohf.ROHF  ,
                }
            # .with_df should never be removed or changed during the conversion.
            # It is needed to compute JK matrix in all pbc SCF objects
            out = mol_addons._object_without_soscf(mf, known_cls, False)

    return mol_addons._update_mo_to_rhf_(mf, out)

def convert_to_ghf(mf, out=None):
    '''Convert the given mean-field object to the generalized HF/KS object

    Args:
        mf : SCF object

    Returns:
        An generalized SCF object
    '''
    from pyscf.pbc import scf

    if out is not None:
        assert (isinstance(out, (scf.ghf.GHF, scf.kghf.KGHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert (isinstance(out, scf.khf.KSCF))
        else:
            assert (not isinstance(out, scf.khf.KSCF))

    if isinstance(mf, (scf.ghf.GHF, scf.kghf.KGHF)):
        if out is None:
            return mf.copy()
        else:
            out.__dict__.update(mf.__dict__)
            return out

    elif isinstance(mf, scf.khf.KSCF):

        def update_mo_(mf, mf1):
            mf1.__dict__.update(mf.__dict__)
            if mf.mo_energy is not None:
                mf1.mo_energy = []
                mf1.mo_occ = []
                mf1.mo_coeff = []
                if hasattr(mf.kpts, 'nkpts_ibz'):
                    nkpts = mf.kpts.nkpts_ibz
                else:
                    nkpts = len(mf.kpts)
                is_rhf = isinstance(mf, scf.hf.RHF)
                for k in range(nkpts):
                    if is_rhf:
                        mo_a = mo_b = mf.mo_coeff[k]
                        ea = getattr(mf.mo_energy[k], 'mo_ea', mf.mo_energy[k])
                        eb = getattr(mf.mo_energy[k], 'mo_eb', mf.mo_energy[k])
                        occa = mf.mo_occ[k] > 0
                        occb = mf.mo_occ[k] == 2
                        orbspin = mol_addons.get_ghf_orbspin(ea, mf.mo_occ[k], True)
                    else:
                        mo_a = mf.mo_coeff[0][k]
                        mo_b = mf.mo_coeff[1][k]
                        ea = mf.mo_energy[0][k]
                        eb = mf.mo_energy[1][k]
                        occa = mf.mo_occ[0][k]
                        occb = mf.mo_occ[1][k]
                        orbspin = mol_addons.get_ghf_orbspin((ea, eb), (occa, occb), False)

                    nao, nmo = mo_a.shape

                    mo_energy = numpy.empty(nmo*2)
                    mo_energy[orbspin==0] = ea
                    mo_energy[orbspin==1] = eb
                    mo_occ = numpy.empty(nmo*2)
                    mo_occ[orbspin==0] = occa
                    mo_occ[orbspin==1] = occb

                    mo_coeff = numpy.zeros((nao*2,nmo*2), dtype=mo_a.dtype)
                    mo_coeff[:nao,orbspin==0] = mo_a
                    mo_coeff[nao:,orbspin==1] = mo_b
                    mo_coeff = lib.tag_array(mo_coeff, orbspin=orbspin)

                    mf1.mo_energy.append(mo_energy)
                    mf1.mo_occ.append(mo_occ)
                    mf1.mo_coeff.append(mo_coeff)

            return mf1

        if out is None:
            out = scf.kghf.KGHF(mf.cell)
        return update_mo_(mf, out)

    else:
        if out is None:
            out = scf.ghf.GHF(mf.cell)
        out = mol_addons.convert_to_ghf(mf, out, remove_df=False)
        # Manually update .with_df because this attribute may not be passed to the
        # output object correctly in molecular convert function
        out.with_df = mf.with_df
        return out

def convert_to_kscf(mf, out=None):
    '''Convert gamma point SCF object to k-point SCF object
    '''
    from pyscf.pbc import scf, dft
    if not isinstance(mf, scf.khf.KSCF):
        assert isinstance(mf, scf.hf.SCF)
        known_cls = {
            dft.uks.UKS   : dft.kuks.KUKS  ,
            dft.roks.ROKS : dft.kroks.KROKS,
            dft.rks.RKS   : dft.krks.KRKS  ,
            dft.gks.GKS   : dft.kgks.KGKS  ,
            scf.uhf.UHF   : scf.kuhf.KUHF  ,
            scf.rohf.ROHF : scf.krohf.KROHF,
            scf.hf.RHF    : scf.khf.KRHF   ,
            scf.ghf.GHF   : scf.kghf.KGHF  ,
        }
        mf = mol_addons._object_without_soscf(mf, known_cls, False)
        if mf.mo_energy is not None:
            if mf.istype('UHF'):
                mf.mo_occ = mf.mo_occ[:,numpy.newaxis]
                mf.mo_coeff = mf.mo_coeff[:,numpy.newaxis]
                mf.mo_energy = mf.mo_energy[:,numpy.newaxis]
            else:
                mf.mo_occ = mf.mo_occ[numpy.newaxis]
                mf.mo_coeff = mf.mo_coeff[numpy.newaxis]
                mf.mo_energy = mf.mo_energy[numpy.newaxis]

    if out is None:
        return mf

del (SMEARING_METHOD)

convert_to_khf = convert_to_kscf

def mo_energy_with_exxdiv_none(mf, mo_coeff=None):
    ''' compute mo energy from the diagonal of fock matrix with exxdiv=None
    '''
    cell.basis = 'ccpvdz'
    cell.a = numpy.eye(3) * 4
    cell.mesh = [17] * 3
    cell.verbose = 3
    cell.build()
    nks = [2,1,1]
    mf = pscf.KUHF(cell, cell.make_kpts(nks)).density_fit()
    mf = smearing_(mf, .1)
    mf.kernel()
    print(mf.e_tot - -5.56769351866668)
    mf = smearing_(mf, .1, mu0=0.351195741757)
    mf.kernel()
    print(mf.e_tot - -5.56769351866668)
    mf = smearing_(mf, .1, method='gauss')
    mf.kernel()
    print(mf.e_tot - -5.56785857886738)
