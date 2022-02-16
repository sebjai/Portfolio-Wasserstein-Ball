import numpy as np 

import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from statsmodels.distributions.empirical_distribution import ECDF

from scipy.stats import norm
# from scipy import interpolate
from scipy.special import lambertw
# from scipy import integrate
from scipy import optimize

from IPython.display import display, Math

import copy


class SIR_CEV_Model():
    
    def __init__(self, x0, S0, pi, T, params):
        
        self.S0 = S0
        self.x0 = x0
        self.pi = pi
        self.T = T
        
        self.params = params
        

    def GetParams(self, measure="P"):
    
        x0 = self.x0
        S0 = self.S0
        pi = self.pi
        T = self.T
        
        if measure == "P":

            mu = self.params["P"]["mu"]
            sigma = self.params["P"]["sigma"]
            beta = self.params["P"]["beta"]
            rho = self.params["P"]["rho"]
            r0 = self.params["P"]["r0"]
            kappa = self.params["P"]["kappa"]
            theta_r = self.params["P"]["theta_r"]
            sigma_r = self.params["P"]["sigma_r"] 
            
        elif measure == "Q":
            
            mu = self.params["P"]["mu"]
            sigma = self.params["P"]["sigma"]
            beta = self.params["P"]["beta"]
            rho = self.params["P"]["rho"]
            r0 = self.params["P"]["r0"]
            kappa = self.params["Q"]["kappa"]
            theta_r = self.params["Q"]["theta_r"]
            sigma_r = self.params["Q"]["sigma_r"] 

        return x0, S0, mu, sigma, beta, rho, pi, r0, kappa, theta_r, sigma_r, T        
        
    
    def SimAndPlot(self, Ndt, Nsims, measure="P"):
        
        t, S, X, Z, W, r = self.Sim(Ndt, Nsims, measure)
        
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        
        
        plt.figure(figsize=(10,5))

        for i in range(S.shape[2]):
            
            ax = plt.subplot(1,S.shape[2],i+1)
            
            plt.fill_between(t,np.quantile(S[:,:,i],0.1,axis=0).T,np.quantile(S[:,:,i],0.9,axis=0).T, alpha=0.2, color='y')
            plt.plot(t,S[:100,:,i].T, linewidth=0.3)
            plt.plot(t,S[0,:,i].T, linewidth=2)
            plt.plot(t,np.quantile(S[:,:,i],[0.1,0.5,0.9],axis=0).T,'-k',linewidth=0.2 )
            plt.xlabel( '$t$')
            plt.ylabel( '$S_t^{' + str(i+1) +'}$')           
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))        

        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10,5))
        
        ax = plt.subplot(1,2,1)
        plt.fill_between(t,np.quantile(X,0.1,axis=0).T,np.quantile(X,0.9,axis=0).T, alpha=0.2, color='y')
        plt.plot(t,X[:100,:].T, linewidth=0.3)
        plt.plot(t,X[0,:].T, linewidth=2)
        plt.plot(t,np.quantile(X,[0.1,0.5,0.9],axis=0).T,'-k',linewidth=0.2 )
        plt.xlabel( '$t$')
        plt.ylabel( '$X_t^{\delta}$')        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))        
        
        ax = plt.subplot(1,2,2)
        plt.fill_between(t,np.quantile(Z,0.1,axis=0).T,np.quantile(Z,0.9,axis=0).T, alpha=0.2, color='y')
        plt.plot(t,Z[:100,:].T, linewidth=0.3)
        plt.plot(t,Z[0,:].T, linewidth=2)
        plt.plot(t,np.quantile(Z,[0.1,0.5,0.9],axis=0).T,'-k',linewidth=0.2 )
        plt.xlabel( '$t$')
        plt.ylabel( '$Z_t$') 
        plt.ylim(0,2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))        
        
        
        plt.tight_layout()
        plt.show()
        
        return t, S, X, Z, W, r 
    
    def BondPrice(self, r, tau):
        
        kappa = self.params["Q"]["kappa"]
        theta_r = self.params["Q"]["theta_r"]
        sigma_r = self.params["Q"]["sigma_r"]
        
        B = (1 -np.exp(-kappa*tau))/kappa
        
        A = np.exp( (theta_r-sigma_r**2/(2*kappa) )*(B - tau) - (sigma_r*B)**2/(4*kappa) )
        
        return A * np.exp(-B*r)
    
    
    
    def Sim(self, Ndt, Nsims, measure="P") :
        
        # import pdb; pdb.set_trace()
        
        x0, S0, mu, sigma, beta, rho, pi, r0, kappa, theta_r, sigma_r, T  = self.GetParams(measure)
        
        t = np.linspace(0, T, Ndt+1)
        dt = t[1] - t[0]
        sqrt_dt = np.sqrt(dt)
        
        r = np.zeros((Nsims, Ndt+1))
        r[:,0] = r0        

        # used to change from P->Q
        #
        # (a0 - a1 * r)/sigma_r is the drift correction for IR
        #
        a0 = self.params["P"]["kappa"]*self.params["P"]["theta_r"]-self.params["Q"]["kappa"]*self.params["Q"]["theta_r"]
        a1 = self.params["P"]["kappa"]-self.params["Q"]["kappa"]
        
        sigma_r_eff = sigma_r*np.sqrt((1-np.exp(-2*kappa*dt))/(2*kappa*dt))

        
        # equity assets let last asset be the bond
        S = np.zeros( (Nsims, Ndt+1, len(S0)+1) )
        S[:, 0, :len(S0)] = S0
        S[:,0,-1] = self.BondPrice(r0, T)
        
        X = np.zeros((Nsims, Ndt+1))
        X[:,0] = x0
        
        Z = np.zeros((Nsims, Ndt+1))
        Z[:,0] = 1

        W = np.zeros((Nsims,len(S0)+1))
        
        rho_inv = np.linalg.inv(rho)
        
        mpr = np.zeros((Nsims,len(S0)+1))
        
        for i in range(Ndt) :
            
            
            # compute the market-price-of-risk
            eff_vol = sigma * S[:,i,:len(S0)]**beta
            mpr[:,:len(S0)] = ((self.params["P"]["mu"].reshape(1,-1) -r[:,i].reshape(-1,1) )/(eff_vol))
            mpr[:,-1] = (a0 - a1 * r[:,i])/sigma_r
            
            mpr = mpr @ rho_inv            
            
            
            # determine Brownian motions...
            dW = sqrt_dt*np.random.multivariate_normal(np.zeros(len(S0)+1), rho, Nsims)
            if measure == "Q":
                dW -= (mpr @ rho) * dt
            W += dW
            
            # update interest rate
            r[:,i+1] = theta_r + (r[:,i]-theta_r)*np.exp(-kappa*dt) + sigma_r_eff*dW[:,-1]

            # update risky assets (excluding the bond)
            S[:, i+1,:len(S0)] = S[:,i,:len(S0)] * np.exp( (mu - 0.5 * eff_vol**2)*dt + eff_vol * dW[:,:len(S0)])
            
            # update the bond
            S[:, i+1,-1] = self.BondPrice(r[:,i+1], T-t[i+1])
            
            # update the SDF
            Lambda = np.sum(mpr * (rho @ mpr.T).T, axis=1)
            Z[:,i+1] = Z[:,i]* np.exp( -(r[:,i]+0.5*Lambda)*dt - np.sum(dW* mpr, axis=1 ) )

            # update the portfolio value
            
            # compute effective volatilities with portfolio positions
            B = (1-np.exp(-kappa*(T-t[i])))/kappa
            pi_eff_vol = pi*np.concatenate((eff_vol, -sigma_r*B*np.ones((Nsims,1))), axis=1 )
            
            # convexity correction term from Ito's lemma
            upsilon =  np.sum(pi_eff_vol * (rho @ pi_eff_vol.T).T, axis=1)
            
            bond_drift = r[:,i] - B*(a0-a1*r[:,i])
            mu_all = np.concatenate((mu * np.ones((Nsims,1)), bond_drift.reshape(-1,1)), axis=1)
            
            X[:,i+1] = X[:,i]* np.exp( ( r[:,i] + np.sum(pi * (mu_all-r[:,i].reshape(-1,1)), axis=1) \
                                        -0.5* upsilon)*dt + np.sum(pi_eff_vol * dW, axis=1) )
            
        return t, S, X, Z, W, r
    
    
    def GenerateUniforms(self, X,Y): 
        
        np.seterr(all='ignore')
        
        h = lambda  X : 1.06 * np.std(X) * (len(X))**(-1/5)
        
        f = lambda x, X, h : np.sum(  norm.pdf((x.reshape(1,-1)-X.reshape(-1,1))/h)/h, axis=0 ) / len(X)         
    
        ecdf_X = ECDF(X)
        ecdf_Y = ECDF(Y)
        
        # generate uniforms        
        U_X = ecdf_X(X)
        U_Y = ecdf_Y(Y)
                    
        # generate  G_{X|Y}(x|y) := P(X \le x | Y = y) using kernel densities
        G = lambda x, y, X, Y, hx, hy : np.sum(  norm.cdf((x.reshape(1,-1)-X.reshape(-1,1))/hx) * norm.pdf((y.reshape(1,-1)-Y.reshape(-1,1))/hy)/hy, axis=0 ) / len(X) / f(y, Y, hy)
        
        U_X_tilde = G(X, Y, X, Y, h(X), h(Y))
        
        ecdf_U_X_tilde = ECDF(U_X_tilde)
        U_X_tilde = ecdf_U_X_tilde(U_X_tilde)
        
        
        return U_X, U_Y, U_X_tilde
    
    def integrate(self, f, u):
    
        return np.sum(0.5*(f[:-1]+f[1:])*np.diff(u))
    
    
    def Create_a_b_grid(self, a, b, N):
        
        eps=0.002    
        
        u_eps = 10**(np.linspace(-10, np.log(eps)/np.log(10),30))-1e-11
        u_eps_flip = np.flip(copy.deepcopy(u_eps)) 
    
        u1 = a + u_eps
        u2 = np.linspace(a + eps, b - eps, N)
        u3 = b - u_eps_flip
        
        return np.concatenate((u1,u2, u3))    
        
    
    def Sim_Uniforms_Xi(self, params, Ndt, Nsims, u_grid, measure="P", copula="Gaussian"):
        
        display(Math(r"Simulating\quad X, Z..."))
        _, _, X, Z, W,_ = self.Sim(Ndt, Nsims, measure)

        X = X[:,-1]
        Z = Z[:,-1] 
        Z *= self.BondPrice(self.params["P"]["r0"], self.T) / np.mean(Z)
        
        c = (np.mean(X*Z) - self.x0)/self.BondPrice(self.params["P"]["r0"], self.T) 
        
        X -= c
        
        # estimate empirical distribution function of X and its inverse
        
        # bandwidth
        h = lambda  X : 1.06 * np.std(X) * (len(X))**(-1/5)

        min_X = np.min(X)
        min_X -= np.abs(min_X)*0.01
        max_X = np.max(X)
        max_X += np.abs(max_X)*0.01
        
        ECDF_X = ECDF(X)
        
        self.invF_P = np.zeros(len(u_grid))
        for i in range(len(u_grid)):
            
            self.invF_P[i] = optimize.root_scalar(lambda x : (ECDF_X(x) - u_grid[i]), \
                                  method='Brentq', bracket=[min_X,max_X]).root
                
        X = np.log( X.reshape(-1) )
        Z = np.log( Z.reshape(-1) )

        display(Math(r'Generating\quad U_X, \tilde{U}_Z...'))
        U_Z, U_X, U_Z_X = self.GenerateUniforms(Z, X)        
        
        # generate V
        display(Math(r"generating \quad V..."))
        
        if copula == "Gaussian":
            print("Gaussian")
            
            p = params["p"]
            V = norm.cdf(  p * norm.ppf(U_X) + np.sqrt(1-p**2)* norm.ppf( 1 - U_Z_X ) )
            
        elif copula == "CoIn":
            print("CoIn")
            
            if params["us"] > 0:
                V = (1-U_Z_X) * params["us"] * (U_X <= params["us"]) + U_X * (U_X > params["us"]) * (1- U_Z_X > 0)
            else:
                V = U_X
                    
            
        elif copula == "Gumbel":
            
            print("Gumbel")
            a = params["zeta"]
            
            psi_inv = ( - np.log(U_X) )**a
            psi_dot_psi_inv = -psi_inv**(1/a-1) /a * U_X
            
            y = (1-U_Z_X) * psi_dot_psi_inv 
            psi_dot_inv = ((a-1) * lambertw(  (-a* y)**(1/(1-a)) / (a-1) ) )**a
          
            V = np.real(np.exp(-(psi_dot_inv - psi_inv )**(1/a) ))
            
        elif copula == "All":

            ecdf_Z = ECDF(np.exp(Z))
            
            V = 1 - ecdf_Z(np.exp(Z))


        ecdf_V = ECDF(V)
        V = ecdf_V(V)
        
        
        # compute Q-density
        
        display(Math(r"generating \quad \xi(u)..."))
        
        # transform V to Y = log(V/(1-V)) then compute Y's Q-density, 
        # then transform back to V
        
        mask = ~((V==0) | (V==1))
        Y = np.log(V[mask]/(1-V[mask]))
        
        # w = \varsigma_T / E[ \varsigma_T ]
        w = (np.exp(Z) / np.mean(np.exp(Z)))[mask]
        

        
        # KDE estimator for *any* input data X, and sampling points x
        f = lambda x, X, h : np.sum(  norm.pdf((x.reshape(1,-1)-X.reshape(-1,1))/h)/h, axis=0 ) / len(X) 
        
        # Q-KDE estimator for *any* input data Y, and sampling points y, needs dQ/dP weights w
        f_Q = lambda  y, Y, w, h  : np.sum( w.reshape(-1,1) * norm.pdf( (y.reshape(1,-1) - Y.reshape(-1,1))/h )/h, axis=0 ) / len(Y) 
        
        Y_qtl = np.quantile(Y,[0.01,0.99])
        
        h_Y = h(Y[(Y>=Y_qtl[0]) & (Y<=Y_qtl[1])])/4

        f_Y = lambda y : f(y, Y, h_Y)
        f_Y_Q = lambda y : f_Q(y, Y, w, h_Y)
        
        f_V = lambda v : f_Y(np.log(v/(1-v))) / (v*(1-v))
        f_V_Q = lambda v : f_Y_Q(np.log(v/(1-v))) / (v*(1-v))

   
        y = np.linspace(-10,6,50)
        
        v = u_grid

        # evaluate the P and Q-pdfs of V and normalize to integrate to one
        f_V_eval = f_V(v)
        f_V_eval /= self.integrate(f_V_eval, v)
        
        f_V_Q_eval = f_V_Q(v)
        f_V_Q_eval /= self.integrate(f_V_Q_eval, v)
        
        self.u = v
        self.xi = f_V_Q_eval * self.BondPrice(self.params["P"]["r0"], self.T) 
        
        return np.exp(X), np.exp(Z), U_X, U_Z, U_Z_X, V, v, f_V_eval, f_V_Q_eval    
        
