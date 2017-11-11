from __future__ import print_function, division
import sys, numpy as np
from numpy import dot, zeros, einsum, pi, log
from pyscf.nao import tddft_iter
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l

class gw(tddft_iter):

  def __init__(self, **kw):
    from pyscf.nao.m_log_mesh import log_mesh
    """ Constructor G0W0 class """
    # how to exclude from the input the dtype and xc_code ?
    tddft_iter.__init__(self, dtype=np.float64, xc_code='RPA', **kw)
    self.xc_code = 'G0W0'
    self.niter_max_ev = kw['niter_max_ev'] if 'niter_max_ev' in kw else 5
    self.nocc_0t = nocc_0t = self.nelectron // (3 - self.nspin)
    self.nocc = kw['nocc'] if 'nocc' in kw else min(6,nocc_0t)
    self.nvrt = kw['nvrt'] if 'nvrt' in kw else min(6,self.norbs-nocc_0t)
    self.start_st = self.nocc_0t-self.nocc
    self.finish_st = self.nocc_0t+self.nvrt
    self.nn = range(self.start_st, self.finish_st) # list of states to correct?
    self.nff_ia = kw['nff_ia'] if 'nff_ia' in kw else 32
    self.tol_ia = kw['tol_ia'] if 'tol_ia' in kw else 1e-3
    (wmin_def,wmax_def,tmax_def) = self.get_wmin_wmax_tmax_ia_def(self.tol_ia)
    self.wmin_ia = kw['wmin_ia'] if 'wmin_ia' in kw else wmin_def
    self.wmax_ia = kw['wmax_ia'] if 'wmax_ia' in kw else wmax_def
    self.tmax_ia = kw['tmax_ia'] if 'tmax_ia' in kw else tmax_def
    self.ww_ia,self.tt_ia = log_mesh(self.nff_ia, self.wmin_ia, self.wmax_ia, self.tmax_ia)
    self.dw_ia = self.ww_ia*(log(self.ww_ia[-1])-log(self.ww_ia[0]))/(len(self.ww_ia)-1)
    self.dw_excl = self.ww_ia[0]
    
    assert self.cc_da.shape[1]==self.nprod
    self.kernel_sq = pack2den_l(self.kernel)
    self.v_dab_ds = self.pb.get_dp_vertex_doubly_sparse(axis=2)
    self.snmw2sf = self.sf_gw_corr()
    
  def get_wmin_wmax_tmax_ia_def(self, tol):
    from numpy import log, exp, sqrt, where, amin, amax
    """ 
      This is a default choice of the wmin and wmax parameters for a log grid along 
      imaginary axis. The default choice is based on the eigenvalues. 
    """
    E = self.ksn2e[0,0,:]
    E_fermi = self.fermi_energy
    E_homo = amax(E[where(E<=E_fermi)])
    E_gap  = amin(E[where(E>=E_fermi)]) - E_homo  
    E_maxdiff = amax(E) - amin(E)
    d = amin(abs(E_homo-E)[where(abs(E_homo-E)>1e-4)])
    wmin_def = sqrt(tol * (d**3) * (E_gap**3)/(d**2+E_gap**2))
    wmax_def = (E_maxdiff**2/tol)**(0.250)
    tmax_def = -log(tol)/ (E_gap)
    tmin_def = -100*log(1.0-tol)/E_maxdiff
    return wmin_def, wmax_def, tmax_def

  def rf0_cmplx_ref(self, ww):
    """ Full matrix response in the basis of atom-centered product functions """
    rf0 = np.zeros((len(ww), self.nprod, self.nprod), dtype=self.dtypeComplex)
    v_arr = self.pb.get_dp_vertex_array()    
    
    zvxx_a = zeros((len(ww), self.nprod), dtype=self.dtypeComplex)
    for n,(en,fn) in enumerate(zip(self.ksn2e[0,0,0:self.nfermi], self.ksn2f[0, 0, 0:self.nfermi])):
      vx = dot(v_arr, self.xocc[n,:])
      for m,(em,fm) in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        if (fn - fm)<0 : break
        vxx_a = dot(vx, self.xvrt[m,:]) * self.cc_da
        for iw,comega in enumerate(ww):
          zvxx_a[iw,:] = vxx_a * (fn - fm) * ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
        rf0 = rf0 + einsum('wa,b->wab', zvxx_a, vxx_a)

    return rf0
  
  rf0 = rf0_cmplx_ref
  
  def si_c(self, ww=None):
    from numpy.linalg import solve
    """ 
    This computes the correlation part of the screened interaction W_c
    by solving <self.nprod> linear equations (1-K chi0) W = K chi0 K
    scr_inter[w,p,q], where w in ww, p and q in 0..self.nprod 
    """
    ww = 1j*self.ww_ia if ww is None else ww
    rf0 = si0 = self.rf0(ww)
    for iw,w in enumerate(ww):
      k_c = dot(self.kernel_sq, rf0[iw,:,:])
      b = dot(k_c, self.kernel_sq)
      k_c = np.eye(self.nprod)-k_c
      si0[iw,:,:] = solve(k_c, b)

    return si0

  def sf_gw_corr(self):
    """ 
    This computes a spectral function of the GW correction.
    sf[spin,n,m,w] = X^n V_mu X^m W_mu_nu X^n V_nu X^m,
    where n runs from s...f, m runs from 0...norbs, w runs from 0...nff_ia, spin=0...1 or 2.
    """
    snmw2sf = zeros((self.nspin, len(self.nn), self.norbs, self.nff_ia), dtype=self.dtype)
    wpq2si0 = self.si_c().real
    v_pab = self.pb.get_ac_vertex_array()
    
    for s in range(self.nspin):
      xna = self.mo_coeff[0,s,self.nn,:,0]
      xmb = self.mo_coeff[0,s,:,:,0]
      nmp2xvx = einsum('na,pab,mb->nmp', xna, v_pab, xmb)
      for iw,si0 in enumerate(wpq2si0):
        snmw2sf[s,:,:,iw] = einsum('nmp,pq,nmq->nm', nmp2xvx, si0, nmp2xvx)
    return snmw2sf

  def gw_corr_int(self, sn2w):
    """ This computes an integral part of the GW correction at energies sn2e[spin,len(self.nn)] """
    sn2int = np.zeros_like(sn2w, dtype=self.dtype)
    for s,ww in enumerate(sn2w):
      for n,w in enumerate(ww):
        for m in range(self.norbs):
          if abs(w-self.ksn2e[0,s,m])<self.dw_excl : continue
          sn2int[s,n] -= ((self.dw_ia*self.snmw2sf[s,n,m,:] / (w + 1j*self.ww_ia-self.ksn2e[0,s,m])).sum()/pi).real
    return sn2int

  def gw_corr_res(self, sn2w):
    """ This computes a residue part of the GW correction at energies sn2w[spin,len(self.nn)] """
    sn2res = np.zeros_like(sn2w, dtype=self.dtype)
    for s,ww in enumerate(sn2w):
      for n,w in enumerate(ww):
        lsos = self.lsofs_inside_contour(s,w)
        
    return sn2res
  
  def lsofs_inside_contour(self, s, w):
    """ 
      Computes number of states the eigen energies of which are located inside an integration contour.
      The integration contour depends on w 
    """ 
    nGamma_pos = 0
    nGamma_neg = 0
    E = self.ksn2e[0,s,:]
    for i,e in enumerate(E):
      zr = e - w
      zi = -np.sign(e-self.fermi_energy)
      if zr>=self.dw_excl and zi>0 : nGamma_pos = nGamma_pos + 1
      if abs(zr)<self.dw_excl and zi>0 : nGamma_pos = nGamma_pos + 1
    
      if zr<=-self.dw_excl and zi<0 : nGamma_neg = nGamma_neg + 1
      if abs(zr)<self.dw_excl and zi<0 : nGamma_neg = nGamma_neg + 1
      print(s,w, i,zr,zi, nGamma_pos, nGamma_neg)
    
    return nGamma_pos

  
  def correct_ev(self):
    """ This computes the corrections to the eigenvalues """
    sn2eval_gw = np.copy(self.ksn2e[0,:,self.nn]).T
    
    gw_corr_int = self.gw_corr_int(sn2eval_gw)
    print(gw_corr_int)
    gw_corr_res = self.gw_corr_res(sn2eval_gw)
    print(gw_corr_res)
    return 0
    
  
