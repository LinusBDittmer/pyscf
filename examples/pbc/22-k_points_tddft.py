#!/usr/bin/env python
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#

import numpy as np
from pyscf.pbc import gto, scf, tdscf
from pyscf import scf as molscf
from pyscf import lib


'''
TDSCF with k-point sampling
'''
atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
a = '''
1.7850000000 1.7850000000 0.0000000000
0.0000000000 1.7850000000 1.7850000000
1.7850000000 0.0000000000 1.7850000000
'''
basis = '''
C  S
    9.031436   -1.960629e-02
    3.821255   -1.291762e-01
    0.473725    5.822572e-01
C  P
    4.353457    8.730943e-02
    1.266307    2.797034e-01
    0.398715    5.024424e-01
''' # a trimmed DZ basis for fast test
pseudo = 'gth-hf-rev'
cell = gto.M(atom=atom, basis=basis, a=a, pseudo=pseudo).set(verbose=3)
kmesh = [2,1,1]
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)
mf = scf.KRHF(cell, kpts).rs_density_fit().run()

log = lib.logger.new_logger(mf)

''' k-point TDSCF solutions can have non-zero momentum transfer between particle and hole.
    This can be controlled by `td.kshift_lst`. By default, kshift_lst = [0] and only the
    zero-momentum transfer solution (i.e., 'vertical' in k-space) will be solved, as
    demonstrated in the example below.
'''
td = mf.TDA().set(nstates=5).run()
log.note('RHF-TDA:')
for kshift,es in zip(td.kshift_lst,td.e):
    log.note('kshift = %d  Eex = %s', kshift, ' '.join([f'{e:.3f}' for e in es*27.2114]))

''' If GDF/RSDF is used as the density fitting method (as in this example), solutions
    with non-zero particle-hole momentum-transfer solution is also possible. The example
    below demonstrates how to calculate solutions with all possible kshift.

    NOTE: if FFTDF is used, pyscf will set `kshift_lst` to [0].
'''
td = mf.TDHF().set(nstates=5, kshift_lst=list(range(nkpts))).run()
log.note('RHF-TDHF:')
for kshift,es in zip(td.kshift_lst,td.e):
    log.note('kshift = %d  Eex = %s', kshift, ' '.join([f'{e:.3f}' for e in es*27.2114]))


''' TDHF at a single k-point compared to molecular TDSCF
'''
atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build()
mf = scf.KRHF(cell, cell.make_kpts([2,2,2]))
mf.run()

td = tdscf.KTDA(mf)
td.nstates = 5
td.verbose = 5
print(td.kernel()[0] * 27.2114)

td = tdscf.KTDDFT(mf)
td.nstates = 5
td.verbose = 5
print(td.kernel()[0] * 27.2114)

mf = scf.RHF(cell)
mf.kernel()
td = tdscf.TDA(mf)
td.kernel()

#
# Gamma-point RKS
#
ks = scf.RKS(cell)
ks.run()

td = tdscf.KTDDFT(ks)
td.nstates = 5
td.verbose = 5
print(td.kernel()[0] * 27.2114)
print(td.oscillator_strength())


# TODO:
#kpt = cell.get_abs_kpts([0.25, 0.25, 0.25])
#mf = scf.RHF(cell, kpt=kpt)
#mf.kernel()
#td = tdscf.TDA(mf)
#td.kernel()
