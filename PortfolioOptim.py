import numpy as np 

from SIR_CEV_Model import SIR_CEV_Model

from sklearn.isotonic import IsotonicRegression

from scipy import optimize

import matplotlib.pyplot as plt

class PortfolioOptim:
    
    def __init__(self, MarketModel : SIR_CEV_Model, epsilon, gamma):
        
        self.ir = IsotonicRegression()
        
        self.epsilon = epsilon    
        self.MarketModel = MarketModel
        self.x0 = self.MarketModel.x0
        self.u = self.MarketModel.u
        self.gamma = gamma
                
        self.xi = self.MarketModel.xi

    def __ell_iso__(self, l1, l2):
        
        lam1 = np.exp(l1)
        lam2 = np.exp(l2)
        
        ell = self.MarketModel.invF_P + 1/(2*lam1)*(self.gamma - lam2 * self.xi)
                
        return ell, self.ir.fit_transform(self.u, ell )

    def integrate(self, f, u):
        
        return np.sum(0.5*(f[:-1]+f[1:])*np.diff(u))

    def WassersteinDistance(self):
    
        return np.sqrt(self.integrate((self.gs-self.MarketModel.invF_P)**2, self.u))
    
    def Cost(self):
        
        return self.integrate( self.gs* self.xi, self.u)
    
    def RiskMeasure(self):
                    
        return -self.integrate( self.gs* self.gamma, self.u)    
    
    
    def Plot_ell_iso(self, ell):
        
        
        fig = plt.figure()
        plt.plot(self.u, ell, linewidth=0.5, color='g', label=r"$\ell$")
        plt.plot(self.u, self.gs, label = r"$g^*$", color='r')
        plt.plot(self.u, self.MarketModel.invF_P, linestyle='--', color='b', label=r"$F^{-1}_{X_T^\delta}$")
        plt.legend(fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.ylim( 0, 3 )
        
        plt.show()    
        
        return fig
    
    def Optimise(self):
                
        def WD_error(l1, l2):
            
            # perform isotonic projection with these parameters 
            ell, self.gs = self.__ell_iso__(l1, l2 )
            
            error = np.abs( self.WassersteinDistance() - self.epsilon)
            
            return error
                
        def Bdgt_error(l2):
            
            sol_opt_l1 = optimize.minimize(lambda x : WD_error(x,l2), self.l1, method='Nelder-Mead')
            
            self.l1 = sol_opt_l1.x
            
            ell, self.gs = self.__ell_iso__(self.l1, l2)            
            
            return np.abs(self.Cost()- self.x0)
        

        KeepSearching = True
        
        lambda0 = np.zeros(2)
        
        while KeepSearching:
            
            lambda0[1] = np.random.uniform(-1,1)
            self.l1 = np.random.uniform(-1,1)
            
            sol_opt_l2 = optimize.minimize(Bdgt_error, lambda0[1], method='Nelder-Mead')
            l2 = sol_opt_l2.x
            
            sol_opt_l1 = optimize.minimize(lambda x : WD_error(x, l2), self.l1, method='Nelder-Mead')
            
            l1 = sol_opt_l1.x

            ell, self.gs = self.__ell_iso__(l1, l2)            
            
            if (np.abs(self.WassersteinDistance()-self.epsilon) < 1e-3) and (np.abs(self.Cost()-self.x0) < 1e-3) :
                KeepSearching = False
        
            if np.exp(l2) < 1e-10:
                KeepSearching = False
                
        if np.exp(l2) < 1e-10:
            
            l2[0] = -np.inf
            sol_opt_l1 = optimize.minimize(lambda x : WD_error(x, l2), self.l1, method='Nelder-Mead')
            
            l1 = sol_opt_l1.x
            
            ell, self.gs = self.__ell_iso__(l1, l2)            
            
        fig = self.Plot_ell_iso(ell)             
            
        
        return np.exp(np.array([l1[0],l2[0]])), self.WassersteinDistance(), self.Cost(), self.RiskMeasure(), fig