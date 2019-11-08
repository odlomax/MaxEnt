import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

# define a very small number
tiny=np.finfo(1.).eps

# define default intial value of x
x_default=1.

def psf_to_linear_map(psf,x_shape,y_shape):
    
    """
    
    Function:
        create linear map from vector x to y via psf
        
    Arguments
    ---------
        
    psf[:,:]: float
            point-spread function of observation
            
    x_shape[2]: int
        shape of model image
            
    y_shape[2]: int
            shape of observed image
            
    Result
    ------
    
    A: csr_matrix
        x_size by y_size linear map of psf
    
    """
    
    # set matrix dimensions
    m=np.prod(x_shape)
    n=np.prod(y_shape)
    
    # perform some broadcasting Kung Fu to get indices of A
        
    # make grid of indices [jx,jy] for psf
    jx_min=-psf.shape[0]//2+1
    jx_max=psf.shape[0]//2+1
    jy_min=-psf.shape[1]//2+1
    jy_max=psf.shape[1]//2+1
    jx,jy=np.meshgrid(np.arange(jx_min,jx_max),np.arange(jy_min,jy_max),indexing="ij")
    jx=jx.flatten()
    jy=jy.flatten()
    
    # make n grids, off indices, offset by pix position
    k=np.arange(n)
    y_off=np.mod(k,y_shape[1])
    x_off=((k-y_off)//y_shape[1])
    jy=jy[np.newaxis,:]+y_off[:,np.newaxis]
    jx=jx[np.newaxis,:]+x_off[:,np.newaxis]
    
    # flatten out into indices of A
    j=(jx*y_shape[1]+jy).flatten()
    i=(np.arange(m)[:,np.newaxis]+np.zeros(psf.size,np.int)[np.newaxis,:]).flatten()
    
    # format psf data
    data=(np.zeros(m)[:,np.newaxis]+psf.flatten()[np.newaxis,:]).flatten()
    
    # remove out of bounds entries
    in_bounds=np.logical_and(j>=0,j<n)
    data=data[in_bounds]
    i=i[in_bounds]
    j=j[in_bounds]
    
    # set matrix
    A=csr_matrix((data,[i,j]),shape=(m,n))
    
    return A


def ln(x):
    
        """
        
        Function:
            ln(x) doesn't blow up if x is slightly negative!
            
        Arguments
        ---------
        
        x[:]: float
            x
            
        Result
        ------
        
        ln_x[:]: float
            natural log of x=max(tiny,x)
        
        """
        
        #return np.log(x)
        return np.log(np.where(x>0.,x,tiny))

class estimator:
    
    """
    
    MaxEnt image class
    
    """
    
    def __init__(self,y,sigma_y,psf):
        
        """
        
        Subroutine: initialise MaxEnt image x
        
        Arguments
        ---------
        
        y[:,:]: float
            observed image
            
        sigma_y[:,:]: float
            noise map of image
            
        psf[:,:]: float
            point-spread function of observation
        
        """
        
        # set shape of model to shape of observation
        x_shape=y.shape
        y_shape=y.shape
        
        self.y=y.flatten()
        
        # get psf linear map
        self.A=psf_to_linear_map(psf,x_shape,y_shape)
        
        # set noise matrix
        self.R_inv=diags(1./sigma_y.flatten()**2,0)
        
        # precompute some matrices
        self.ATR_inv=(self.A.T*self.R_inv).tocsr()        
        
        # define secant iteration
        def secant_it(log_alpha):
            
            nonlocal self
            nonlocal i
            
            i+=1
            self.alpha=np.exp(log_alpha)
            print()
            print("iteration:",i)
            print("log alpha:",log_alpha)
            
            # perform conjugate gradient minimisation
            start_time=time.time()
            res=minimize(self.__J,self.x,method="trust-ncg",jac=self.__grad_J,hessp=self.__hessp_J)
            
            # next iteration will be faster if we use this result as initial state
            self.x=res.x
            
            # get reduced chi2
            log_chi2=np.log(self.__chi2(self.x))
            log_red_chi2=log_chi2-np.log(self.x.size)
            print("log chi^2_nu:",log_red_chi2)
            print("time taken (s):",time.time()-start_time)
            
            self.log_alpha.append(log_alpha)
            self.log_chi2.append(log_chi2)
            
            return log_red_chi2
        
        # set iteration count
        i=0
        
        # initialise model vector
        self.x=np.full(self.y.size,x_default)
        
        # initialse alpha
        # start with a high value and relax it to optimum
        log_alpha_0=np.log((self.y**2*self.R_inv.data).sum())
        log_alpha_1=log_alpha_0-1.
        alpha_tol=np.log(1.+1./np.sqrt(self.x.size))
        
        # keep track of log_alpha and log_chi2 values
        self.log_alpha=[]
        self.log_chi2=[]
        
        # perform secant iterations
        print("solving for x")
        root=newton(secant_it,log_alpha_0,x1=log_alpha_1,tol=alpha_tol)
        self.alpha=np.exp(root)
        
        # sort lists
        i=np.argsort(self.log_alpha)
        self.log_alpha=np.array(self.log_alpha)[i]
        self.log_chi2=np.array(self.log_chi2)[i]
        
        # get uncertainties
        self.sigma_x=np.sqrt(2./self.__hess_diag_J(self.x))
        
        return
    
    def relax_fit(self,h=0.01,alpha_tol=0.01):
        
        """
        
        Subroutine:
            Sets alpha to the "kink" in the ln_chi2(ln_alpha) curve
            Useful if the classical solutions has overfitted to noise
            
        Arguments
        ---------
        
        h: float
            finite difference
            
        tol: float
            fractional tolerance on alpha
        
        mat_it: int
            max number of iterations
        
        """
        
        # define curvature function
        def neg_curvature(log_alpha):
            
            nonlocal self
            nonlocal i
            
            # get finite differences
            i+=1
            print()
            print("iteration:",i)
            print("log alpha:",log_alpha)
            
            # perform conjugate gradient minimisation
            start_time=time.time()
            
            self.alpha=np.exp(log_alpha-h)
            res=minimize(self.__J,self.x,method="trust-ncg",jac=self.__grad_J,hessp=self.__hessp_J)
            f_0=np.log(self.__chi2(res.x))
            
            self.alpha=np.exp(log_alpha+h)
            res=minimize(self.__J,res.x,method="trust-ncg",jac=self.__grad_J,hessp=self.__hessp_J)
            f_2=np.log(self.__chi2(res.x))
            
            self.alpha=np.exp(log_alpha)
            res=minimize(self.__J,res.x,method="trust-ncg",jac=self.__grad_J,hessp=self.__hessp_J)
            f_1=np.log(self.__chi2(res.x))

            
            # calculate curvature
            dfdx=(f_2-f_0)/(2*h)
            d2fdx2=(f_2-2*f_1+f_0)/h**2
            kappa=d2fdx2/(1+dfdx**2)**(1.5)
            
            print("log chi^2_nu:",f_1-np.log(self.x.size))
            print("curvature:",kappa)
            print("time taken (s):",time.time()-start_time)
            
            self.log_alpha.append(log_alpha)
            self.log_chi2.append(f_1)
            self.kappa.append(kappa)
            self.x=res.x
            
            return -kappa
        
        # set iteration count
        i=0
        j=np.max([1,self.log_alpha.size-3])
        log_alpha_0=self.log_alpha[0]
        log_alpha_1=self.log_alpha[j]
        
        # keep track of log_alpha, log_chi2, and kappa
        self.log_alpha=[]
        self.log_chi2=[]
        self.kappa=[]
        
        # set x_tol relative DoF optimal alpha
        xtol=alpha_tol/np.log(self.alpha)
        
        # find "kink" in curve
        print("relaxing fit")
        minimum=minimize_scalar(neg_curvature,(log_alpha_0,log_alpha_1),
                            method="brent",options={"xtol":xtol})
        self.alpha=np.exp(minimum.x)
        print("applying optimal fit")
        res=minimize(self.__J,self.x,method="trust-ncg",jac=self.__grad_J,hessp=self.__hessp_J)
        self.x=res.x
        
        
        # sort lists
        i=np.argsort(self.log_alpha)
        self.log_alpha=np.array(self.log_alpha)[i]
        self.log_chi2=np.array(self.log_chi2)[i]
        self.kappa=np.array(self.kappa)[i]
        
        # get uncertainties
        self.sigma_x=np.sqrt(2./self.__hess_diag_J(self.x))
        
        return
    
    
    # define loss function
    # entropy
    def __S(self,x):
        
        sum_x=x.sum()
        f=x/sum_x
        result=-(f*ln(f)).sum()
        
        return result

    # chi2
    def __chi2(self,x):
        
        Ax_y=self.A*x-self.y
        result=(csc_matrix(Ax_y)*self.R_inv*Ax_y)[0]
        
        return result
    
    # total
    def __J(self,x):
        
        result=-self.alpha*self.__S(x)+0.5*self.__chi2(x)
        
        return result
    
    
    # grad loss function        
    def __grad_J(self,x):
        
        sum_x=x.sum()
        f=x/sum_x
        log_f=ln(f)
        grad_S=(((x*log_f).sum()-x*log_f)-(sum_x-x)*log_f)/sum_x**2
        
        Ax_y=self.A*x-self.y
        grad_chi2=self.ATR_inv*Ax_y
        
        result=-self.alpha*grad_S+grad_chi2
        
        return result
    
    # Hessian of loss function times vector
    def __hessp_J(self,x,p):
        
        sum_x=x.sum()
        f=x/sum_x
        log_f=ln(f)
        x_log_f=x*log_f
        sum_x_log_f=(x*log_f).sum()
        # set diagonal component
        diag=p*(sum_x**2-sum_x*x-x**2
                +2*x*(sum_x_log_f-x_log_f)
                -2*(sum_x-x)*x_log_f)/(-x*sum_x**3)
        # set off-diagonal component
        # independent part
        off_diag=p.sum()*(sum_x-2*sum_x_log_f)
        # set column-dependent part
        temp_term=(sum_x-2*x)*log_f+2*x_log_f
        off_diag+=(p*(temp_term)).sum()
        # set row-dependent part
        off_diag+=p.sum()*temp_term
        # remove erroneous diagonal part
        off_diag-=p*(sum_x-2*sum_x_log_f
                     +2*(temp_term))
        # normalize
        off_diag/=sum_x**3
        hessp_S=diag+off_diag
        
        
        Ap=self.A*p
        hessp_chi2=self.ATR_inv*Ap
        
        result=-self.alpha*hessp_S+hessp_chi2
        
        return result
    
    # get diagonal elements of loss function Hessian
    def __hess_diag_J(self,x):
        
        # S Hessian
        sum_x=x.sum()
        f=x/sum_x
        log_f=ln(f)
        x_log_f=x*log_f
        sum_x_log_f=(x*log_f).sum()
        hess_diag_S=(sum_x**2-sum_x*x-x**2
                +2*x*(sum_x_log_f-x_log_f)
                -2*(sum_x-x)*x_log_f)/(-x*sum_x**3)
        
        # chi2 Hessian
        hess_diag_chi2=np.array((self.ATR_inv.multiply(self.A)).sum(axis=0)).flatten()
        
        result=-self.alpha*hess_diag_S+hess_diag_chi2
        
        return result
    
    