import copy
from pyscf import gto
from pyscf.lib import logger
from pyscf import lib
from pyscf import mp

import numpy
import scipy.linalg

from pyscf.mp.ordmp2_crit import (_vec_to_rotmat, _crit_grad_num, _crit_hess_num,
                          _eri_sum, _eri_sum_grad, _eri_sum_hess,
                          _max_energy, _max_energy_grad, _max_energy_hess,
                          _min_energy, _min_energy_grad, _min_energy_hess,
                          _max_norm, _max_norm_grad, _max_norm_hess,
                          _hylleraas_2, _hylleraas_2_grad, _hylleraas_2_hess,
                          _hylleraas_renorm, _hylleraas_renorm_grad, _hylleraas_renorm_hess)
from pyscf.mp.ordmp2_solver import (TRAH_Minimiser, NewTRAH_Minimiser, NR_RootSolver, AdamMinimiser)

_criteria = {
        "eri_sum": _eri_sum,
        "max_energy": _max_energy,
        "min_energy": _min_energy,
        "max_norm": _max_norm,
        "hylleraas_2": _hylleraas_2,
        "hylleraas_renorm": _hylleraas_renorm
        }

_crit_gradients = {
        "eri_sum": _eri_sum_grad,
        "max_energy": _max_energy_grad,
        "min_energy": _min_energy_grad,
        "max_norm": _max_norm_grad,
        "hylleraas_2": _hylleraas_2_grad,
        "hylleraas_renorm": _hylleraas_renorm_grad
        }

_crit_hessians = {
        "eri_sum": _eri_sum_hess,
        "max_energy": _max_energy_hess,
        "min_energy": _min_energy_hess,
        "max_norm": _max_norm_hess,
        "hylleraas_2": _hylleraas_2_hess,
        "hylleraas_renorm": _hylleraas_renorm_hess
        }

class ORDMP2(lib.StreamObject):
    """
    Orbital rotation dependent Möller-Plesset Theory that utilises an MP backend.
    """

    def __init__(self, mf, optimality='max_energy', opt_type='min', stepsize = 0.5,
                 max_cycles = 500, convergence = 4):

        self._mf = mf
        self._mp2 = mp.MP2(mf)
        self._mp2.verbose = 0
        self.u_mat = None
        self.opt = optimality
        self.opt_type = 'min'
        self.mo_coeff = None
        self.max_cycles = max_cycles
        self.convergence = convergence
        self.renormalisation = None

        self.solver = TRAH_Minimiser(self, stepsize=stepsize)

    def switch_solver(self, solver, **kwargs):
        avail_solvers = {"trah": TRAH_Minimiser, 
                         "ntrah": NewTRAH_Minimiser,
                         "adam": AdamMinimiser, 
                         "nr": NR_RootSolver}
        if solver not in avail_solvers:
            raise NotImplementedError(f"The solver {solver} is not implemented. "
                                     + f"Only {list(avail_solvers.keys())} are available.")
        self.solver = avail_solvers[solver](self, **kwargs)

    def make_rdm1(self, t2=None, eris=None, ao_repr=False):
        return self._mp2.make_rdm1(t2, eris, ao_repr)

    def make_rdm2(self, t2=None, eris=None, ao_repr=False):
        return self._mp2.make_rdm2(t2, eris, ao_repr)

    @property
    def e_corr(self):
        return self._mp2.e_corr

    @property
    def t2(self):
        return self._mp2.t2

    @property
    def e_tot(self):
        return self._mp2.e_tot

    @property
    def n_param(self):
        return int((self.n_occ * (self.n_occ - 1) + self.n_vir * (self.n_vir - 1)) // 2)

    @property
    def n_occ(self):
        return int(numpy.sum(self._mf.mo_occ) // 2)

    @property
    def n_vir(self):
        return int(len(self._mf.mo_occ) - self.n_occ)

    def criterion(self, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        if n_occ <= 0 or n_vir <= 0:
            n_occ = numpy.sum(mo_occ) // 2
            n_vir = len(mo_occ) - n_occ
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        if vec is None:
            vec = numpy.zeros(n_param)

        if self.opt in _criteria:
            return _criteria[self.opt](self, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        else:
            raise NotImplementedError(f"The optimality criterion {opt} is not available")

    def crit_grad(self, mo_coeff, mo_energy, mo_occ, n_occ=0, n_vir=0, force_numeric=False, epsilon=10**-5):
        if force_numeric:
            return _crit_grad_num(self, mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)

        if n_occ <= 0 or n_vir <= 0:
            n_occ = int(numpy.sum(mo_occ) // 2)
            n_vir = int(len(mo_occ) - n_occ)

        if self.opt in _crit_gradients:
            return _crit_gradients[self.opt](self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        return _crit_grad_num(self, mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)

    def crit_hess(self, mo_coeff, mo_energy, mo_occ, n_occ=0, n_vir=0, force_numeric=False, epsilon=10**-5):
        if force_numeric:
            return _crit_hess_num(self, mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)
        if n_occ <= 0 or n_vir <= 0:
            n_occ = int(numpy.sum(mo_occ) // 2)
            n_vir = int(len(mo_occ) - n_occ)

        if self.opt in _crit_hessians:
            return _crit_hessians[self.opt](self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        return _crit_hess_num(self, mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)

    def check_convergence(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, stepvec, checksize=0.1):
        if self.opt_type == 'min':
            grad = self.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
            hess = self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
            u, s, vt = scipy.linalg.svd(hess)
            s2 = numpy.abs(s)
            s2 /= numpy.amax(s2)
            m_hess = numpy.dot(u, numpy.dot(numpy.diag(s2), vt))
            m_grad = numpy.dot(grad, numpy.dot(m_hess, grad))
            print(f"Modified Gradient norm: {m_grad}")
            print(f"Full Gradient norm: {numpy.linalg.norm(grad)}")

            if numpy.linalg.norm(grad) > 10**-self.convergence:
                return False, stepvec
            
            n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)
            svecs, svals, _ = scipy.linalg.svd(self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir))
            svecs = numpy.einsum('ij,j->ij', svecs, svals * checksize)
            svecs = numpy.concatenate((svecs, -svecs))
            current_val = self.criterion(None, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

            print(f"Checking Manually for Saddlepoint...")

            for cv in svecs:
                c = self.criterion(cv, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
                if c - current_val < -10**-self.convergence:
                    print(f"Found to be Saddlepoint, Dist: {c - current_val}.")
                    return False, cv
        elif self.opt_type == 'root':
            val = self.criterion(None, mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
            if abs(val) >= 10**-self.convergence:
                return False, stepvec
        return True, None

    @property
    def n_param(self):
        return int((self.n_occ * (self.n_occ - 1) + self.n_vir * (self.n_vir - 1)) // 2)

    @property
    def n_occ(self):
        return int(numpy.sum(self._mf.mo_occ) // 2)

    @property
    def n_vir(self):
        return int(len(self._mf.mo_occ) - self.n_occ)

    def test_grad_hess(self):
        mo_coeff = copy.copy(self._mf.mo_coeff)
        mo_energy = self._mf.mo_energy
        mo_occ = self._mf.mo_occ

        n_occ = int(numpy.sum(mo_occ) // 2)
        n_vir = int(len(mo_occ) - n_occ)
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        numpy.set_printoptions(linewidth=500, precision=3, suppress=True)
        self._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy, with_t2=True)
        print("Calculating anayltical Gradient")
        grad_anal = self.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ, n_vir) * 100
        print("Calculating numerical Gradient")
        grad_num = self.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, force_numeric=True) * 100
        print("Calculating anayltical Hessian")
        hess_anal = self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir) * 100
        print("Calculating numerical Hessian")
        hess_num = self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, force_numeric=True) * 100

        print(f"Exact MP2 energy: {self._mp2.e_corr}")

        print("Analytical Gradient")
        print(grad_anal)
        print("Numerical Gradient")
        print(grad_num)
        print("Division")
        print((grad_num / grad_anal) * (abs(grad_anal) > 0.001))
        print("Difference")
        print(grad_num - grad_anal)
        print("Difference Norm")
        print(numpy.linalg.norm(grad_num - grad_anal))
        print("\n\n\n")

        print("Analytical Hessian")
        print(hess_anal)
        print("Numerical Hessian")
        print(hess_num)
        print("Division")
        print((hess_num / hess_anal) * (abs(hess_anal) > 0.001))
        print("Difference")
        print(hess_num - hess_anal)
        print("Difference Norm")
        print(numpy.linalg.norm(hess_num - hess_anal))
        quit()

    def kernel(self, optimality='max_energy'):
        mo_coeff = copy.copy(self._mf.mo_coeff)
        mo_energy = self._mf.mo_energy
        mo_occ = self._mf.mo_occ

        log = logger.new_logger(self)

        n_occ = int(numpy.sum(mo_occ) // 2)
        n_vir = int(len(mo_occ) - n_occ)
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        print(f"Number of Orbitals: {n_occ} occupied, {n_vir} virtual.")
        print(f"Number of Parameters: {n_param}")

        #self.test_grad_hess()

        for cycle in range(self.max_cycles):
            print(f"ORD-MP2 iteration: {cycle}")
            stepvec = self.solver.next_step(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, cycle)
            converged, stepvec = self.check_convergence(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, stepvec)

            if converged:
                print(f"ORD-MP2 converged within {cycle} iterations.")
                break

            rot_coeff = _vec_to_rotmat(stepvec, n_occ, n_vir)
            mo_coeff = mo_coeff @ scipy.linalg.expm(rot_coeff)

            print(f"Value: {self.criterion(numpy.zeros(stepvec.shape), mo_coeff, mo_energy, mo_occ, n_occ, n_vir)}")

        if not converged:
            print(f"ORD-MP2 did not converge within {self.max_cycles} iterations.")

        self.mo_coeff = mo_coeff
        ec = self._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy, with_t2=True)
        return ec

if __name__ == '__main__':
    from pyscf import gto, scf, mp, cc
    mol = gto.M(atom='Li 0.0 0.0 0.0; H 0 0 1.0', basis='sto-3g', verbose=4)
    mf = scf.RHF(mol)
    mf.kernel()

    ordmp = ORDMP2(mf, optimality='hylleraas_2')
    ordmp.switch_solver("ntrah")
    #ordmp.test_grad_hess()

    mp2 = mp.MP2(mf)
    mp2.kernel()

    ec = ordmp.kernel()
    ccsd = cc.CCSD(mf)
    print(f"Starting CCSD calc")
    ccsd.kernel()
    ccsd.ccsd_t()

    dm = ordmp.make_rdm1()
    dip_mom = mf.dip_moment(dm=dm)
    e_scf = mf.energy_tot()
    print(f"ORD-MP2 energy: {ec[0]}")
    print(f"CCSD difference: {ccsd.e_corr - ec[0]}")
    print(f"MP2-CCSD difference: {ccsd.e_tot - mp2.e_tot}")
    print(f"ORD-MP2 dip mom: {dip_mom}")
    print(f"SCF energy: {e_scf}")

