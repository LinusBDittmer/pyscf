'''

File for ORDMP2 solvers

'''

import numpy
from scipy.linalg import expm, pinv
from scipy.optimize import minimize

class ORDMP2_Solver:

    def __init__(self, ordmp):
        self.ordmp = ordmp

    def next_step(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        """
        TO BE OVERRIDDEN
        """
        return None

class TRAH_Minimiser(ORDMP2_Solver):

    def __init__(self, ordmp, level_shift = 1.25, stepsize = 5.0):
        super().__init__(ordmp)
        self.level_shift = level_shift
        self.stepsize = stepsize

    def next_step(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, step):
        self.ordmp._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy, with_t2=True)
        print(f"Calculating Gradient")
        grad = self.ordmp.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
        print(f"Calculating Hessian")
        hess = self.ordmp.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)

        eigvals = numpy.linalg.eigvalsh(hess)

        hess_shifted = hess - numpy.eye(grad.shape[0]) * (self.level_shift * eigvals[0])
        print(f"Lowest Hessian Eigenvalue: {eigvals[0]}")
        print(f"Gradient Length: {numpy.linalg.norm(grad)}")
        step = numpy.linalg.solve(hess_shifted, -grad)

        stepnorm = numpy.linalg.norm(step)
        if stepnorm > self.stepsize:
            step *= self.stepsize / stepnorm

        print(f"Stepsize: {stepnorm}")

        return step

class NewTRAH_Minimiser(ORDMP2_Solver):

    def __init__(self, ordmp, stepsize = 0.5, initial_trust_radius=0.2, max_trust_radius=1000.0,
                 lower_trust_ratio=0.25, upper_trust_ratio=0.75, lower_trust_fac=0.25,
                 upper_trust_fac=2.0, accept_step_ratio=0.02):
        super().__init__(ordmp)
        self.stepsize = stepsize
        self.initial_trust_radius = initial_trust_radius
        self.trust_radius = initial_trust_radius
        self.max_trust_radius = max_trust_radius
        self.lower_trust_ratio = lower_trust_ratio
        self.upper_trust_ratio = upper_trust_ratio
        self.lower_trust_fac = lower_trust_fac
        self.upper_trust_fac = upper_trust_fac
        self.accept_step_ratio = accept_step_ratio

    def solve_local(self, val, grad, hess):
        # Adapted from Scipy trust-ncg
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_trustregion_ncg.py

        p_origin = numpy.zeros_like(grad)
        g_mag = numpy.linalg.norm(grad)
        tolerance = min(0.5, numpy.sqrt(g_mag)) * g_mag

        if g_mag < tolerance:
            return p_origin, False, val

        def get_boundaries_intersections(z, d):
            a = numpy.dot(d, d)
            b = 2 * numpy.dot(z, d)
            c = numpy.dot(z, z) - self.trust_radius**2
            sqrt_discriminant = numpy.sqrt(b*b - 4 * a * c)

            ta = (-b - sqrt_discriminant) / (2 * a)
            tb = (-b + sqrt_discriminant) / (2 * a)
            return sorted([ta, tb])

        def eval_quadratic(p):
            return val + numpy.dot(grad, p) + 0.5 * numpy.dot(p, numpy.dot(hess, p))

        z = p_origin
        r = grad
        d = -r

        while True:
            Bd = numpy.dot(hess, d)
            dBd = numpy.dot(d, Bd)
            if dBd <= 0:
                ta, tb = get_boundaries_intersections(z, d)
                pa = z + ta * d
                pb = z + tb * d
                epa = eval_quadratic(pa)
                epb = eval_quadratic(pb)
                if epa < epb:
                    p_boundary = pa
                    pval = epa
                else:
                    p_boundary = pb
                    pval = epb
                hits_boundary = True
                return p_boundary, hits_boundary, pval
            r_squared = numpy.dot(r, r)
            alpha = r_squared / dBd
            z_next = z + alpha * d
            if numpy.linalg.norm(z_next) >= self.trust_radius:
                ta, tb = get_boundaries_intersections(z, d)
                p_boundary = z + tb * d
                pval = eval_quadratic(p_boundary)
                hits_boundary = True
                return p_boundary, hits_boundary, pval
            r_next = r + alpha * Bd
            r_next_squared = numpy.dot(r_next, r_next)
            if numpy.sqrt(r_next_squared) < tolerance:
                hits_boundary = False
                return z_next, hits_boundary, eval_quadratic(z_next)
            beta_next = r_next_squared / r_squared
            d_next = -r_next + beta_next * d

            z = z_next
            r = r_next
            d = d_next

        return numpy.zeros_like(grad), False, val

    def eff_val(self, x, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        return self.ordmp.criterion(x, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

    def eff_grad(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        return self.ordmp.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

    def eff_hess(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        return self.ordmp.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

    def next_step(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, step):
        # Adapted from SciPy trust-ncg
        # https://github.com/scipy/scipy/blob/main/scipy/optimize/_trustregion.py

        self.ordmp._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy, with_t2=True)
        grad = self.eff_grad(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        hess = self.eff_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        val = self.eff_val(None, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        x_proposed, hits_boundary, predicted_val = self.solve_local(val, grad, hess)
        actual_reduction = val-self.eff_val(x_proposed, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        predicted_reduction = val - predicted_val
        rho = actual_reduction / predicted_reduction
        if rho <= 0:
            print(f"Warning: Negative or zero reduction in Solver! Calculated: {round(rho, 3)}")
        if rho < self.lower_trust_ratio:
            self.trust_radius *= self.lower_trust_fac
        elif rho > self.upper_trust_ratio and hits_boundary:
            self.trust_radius = min(self.trust_radius*self.upper_trust_fac, self.max_trust_radius)

        print(f"X Proposed: {x_proposed}")
        print(f"Predicted Reduction: {predicted_reduction}")
        print(f"Actual Reduction: {actual_reduction}")
        print(f"Hits Boundary: {hits_boundary}")
        print(f"Current Trust Radius: {self.trust_radius}")

        if rho >= self.accept_step_ratio:
            return x_proposed
        return numpy.zeros_like(x_proposed)

class NewTRAH_Minimiser_Constrained(NewTRAH_Minimiser):

    def __init__(self, ordmp, stepsize = 0.5, initial_trust_radius=0.2, max_trust_radius=1000.0,
                 lower_trust_ratio=0.25, upper_trust_ratio=0.75, lower_trust_fac=0.25,
                 upper_trust_fac=2.0, accept_step_ratio=0.02):
        super().__init__(ordmp, stespize, initial_trust_radius, max_trust_radius, lower_trust_ratio,
                         upper_trust_ratio, lower_trust_fac, upper_trust_fac, accept_step_ratio)
        self.eq_vals = []
        self.eq_grads = []
        self.eq_hesss = []
        self.eq_multipliers = []

        self.ineq_vals = []

    def add_constraint(self, shape, value, gradient, hessian, ctype='eq'):
        if ctype not in ['eq', 'ineq']:
            raise ValueError(f"Invalid constraint type {ctype}. Only eq, ineq allowed.")
       
        if ctype == 'eq':
            self.eq_vals.append(value)
            self.eq_grads.append(gradient)
            self.eq_hesss.append(hessian)
            self.eq_multiplier.append(numpy.zeros(shape))
        else:
            self.ineq_vals.append(value)

    def eff_val(x, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)
        if x is None:
            n_param_eff = n_param
            for m in self.eq_multipliers:
                n_param_eff += m.shape[0]
            x = numpy.zeros(n_param_eff)
        else:
            n_param_eff = len(x)
        prim_val = self.ordmp.criterion(x[:n_param], mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        for i, mult in enumerate(self.eq_multipliers):
            prim_val -= numpy.dot(mult, self.eq_vals[i](x, mo_coeff, mo_energy, mo_occ, n_occ, 
                                                        n_vir))
        return prim_val

    def eff_grad(mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)
        n_param_eff = n_param
        for m in self.eq_multipliers:
            n_param_eff += m.shape[0]
        grad = numpy.zeros(n_param_eff)
        grad[:n_param] = self.ordmp.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        l_offset = 0
        for i, mult in enumerate(self.eq_multipliers):
            grad[:n_param] -= numpy.einsum('pq,p', self.eq_grads[i](mo_coeff, mo_energy, mo_occ, n_occ, n_vir), mult)
            grad[n_param+l_offset:n_param+l_offset+mult.shape[0]] = - self.eq_vals[i](None, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            l_offset += mult.shape[0]
        return grad

    def eff_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)
        n_param_eff = n_param
        for m in self.eq_multipliers:
            n_param_eff += m.shape[0]
        hess = numpy.zeros((n_param_eff, n_param_eff))
        hess[:n_param,:n_param] = self.ordmp.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        l_offset = 0
        for i, mult in enumerate(self.eq_multipliers):
            hess[:n_param,:n_param] -= numpy.einsum('pqr,p', self.eq_hesss[i](mo_coeff, mo_energy, mo_occ, n_occ, n_vir), mult)
            coupl = -self.eq_grads[i](mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            hess[n_param:,l_offset:l_offset+mult.shape[0]] = coupl
            hess[l_offset:l_offset+mult.shape[0],n_param:] = coupl.T
        return hess

    def next_step(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, step):
        prop = super().next_step(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, step)


class NR_RootSolver(ORDMP2_Solver):

    def __init__(self, ordmp, stepsize = 5.0):
        super().__init__(ordmp)
        self.stepsize = stepsize

    def next_step(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, step):
        print(f"Calculating Value")
        val = self.ordmp.criterion(None, mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
        print(f"Calculating Gradient")
        grad = self.ordmp.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
        step = - grad / vec
        
        stepnorm = numpy.linalg.norm(step)
        if stepnorm > self.stepsize:
            step *= self.stepsize

        print(f"Stepsize: {stepnorm}")

        return step

class AdamMinimiser(ORDMP2_Solver):

    def __init__(self, ordmp, stepsize = 5.0, decay1 = 0.9, decay2 = 0.999, renorm = 10**-8):
        super().__init__(ordmp)
        self.stepsize = stepsize
        self.decay1 = decay1
        self.decay2 = decay2
        self.renorm = renorm

        self.last_mom1 = None
        self.last_mom2 = None

    def next_step(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, cycle):
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)
        if self.last_mom1 is None or self.last_mom2 is None:
            self.last_mom1 = numpy.zeros(n_param)
            self.last_mom2 = numpy.zeros(n_param)
        print(f"Calculating Gradient")
        grad = self.ordmp.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
        mom1 = self.last_mom1 * self.decay1 + (1 - self.decay1) * grad
        mom2 = self.last_mom2 * self.decay2 + (1 - self.decay2) * numpy.power(grad, 2)
        self.last_mom1 = mom1
        self.last_mom2 = mom2
        bmom1 = mom1 / (1 - self.decay1**(cycle+1))
        bmom2 = mom2 / (1 - self.decay2**(cycle+1))
        
        step = -self.stepsize * bmom1 / (numpy.sqrt(bmom2) + self.renorm)

        print(f"Stepsize: {numpy.linalg.norm(step)}")
        return step


