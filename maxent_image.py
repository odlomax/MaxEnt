import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from scipy.optimize import newton
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

# define a very small number
tiny=np.sqrt(np.finfo(1.).tiny)

# define default intial value of x
x_default=1.

# definie minimisation method
min_method="Newton-CG"

def psf_to_linear_map(psf,y_shape,x_pad=False):
    
    """ 
    
    Function:
        create linear map from vector x to y via psf
        
    Arguments
    ---------
        
    psf[:,:]: float
            point-spread function of observation
            
    y_shape[2]: int
        shape of observed image
            
    x_pad: boolean
            pad shape of model image to account for PSF
            
    Result
    ------
    
    A: csr_matrix
        x_size by y_size linear map of psf
    
    """
    
    # define model index bounds
    if x_pad:
        
        delta_i=psf.shape[0]//2
        delta_j=psf.shape[1]//2
        
    else:
        
        delta_i=0
        delta_j=0
    
    xi_min=0-delta_i
    xj_min=0-delta_j
    xi_max=y_shape[0]+delta_i
    xj_max=y_shape[1]+delta_j
    
    # define psf index bounds
    zi_min=np.ceil(-psf.shape[0]/2.).astype(np.int)
    zi_max=np.ceil(psf.shape[0]/2.).astype(np.int)
    zj_min=np.ceil(-psf.shape[1]/2.).astype(np.int)
    zj_max=np.ceil(psf.shape[1]/2.).astype(np.int)
    
    # define pixel grids for  observation and psf
    xi,xj=np.meshgrid(np.arange(xi_min,xi_max),np.arange(xj_min,xj_max),indexing="ij")
    zi,zj=np.meshgrid(np.arange(zi_min,zi_max),np.arange(zj_min,zj_max),indexing="ij")
    
    # make a map of y indices for every x index
    yi=(xi[...,np.newaxis,np.newaxis]+zi[np.newaxis,np.newaxis,...]).flatten()
    yj=(xj[...,np.newaxis,np.newaxis]+zj[np.newaxis,np.newaxis,...]).flatten()
    
    # locate within-bounds indices
    in_bounds=np.all([yi>=0,yi<y_shape[0],yj>=0,yj<y_shape[1]],0)
    
    # set number of rows
    m=y_shape[0]*y_shape[1]
    
    # set number of columns
    n=(xi_max-xi_min)*(xj_max-xj_min)
    
    # set max number of entries per column
    p=(zi_max-zi_min)*(zj_max-zj_min)
    
    # set i indices of map
    i=yi*y_shape[1]+yj
    i=i[in_bounds]
    
    # set j indices of map
    j=(np.arange(n)[:,np.newaxis]+np.zeros(p,np.int)[np.newaxis,:]).flatten()
    j=j[in_bounds]
    
    # set data of map
    data=(psf.flatten()[np.newaxis,:]+np.zeros(n)[:,np.newaxis]).flatten()
    data=data[in_bounds]
    
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
        
        #return ln(x)
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
        
        self.y_shape=y.shape
        self.x_shape=tuple(np.array(self.y_shape)+np.array(psf.shape)-1)
        self.y=y.flatten()
        
        # get psf linear map
        self.A=psf_to_linear_map(psf,self.y_shape,True)
        
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
            res=minimize(self.__J,self.x,method=min_method,
                         jac=self.__grad_J,hessp=self.__hessp_J)
            
            # next iteration will be faster if we use this result as initial state
            self.x=res.x
            
            # get reduced chi2
            log_chi2=ln(self.__chi2(self.x))
            log_red_chi2=log_chi2-ln(self.x.size)
            print("log chi^2_nu:",log_red_chi2)
            print("time taken (s):",time.time()-start_time)
            
            self.log_alpha.append(log_alpha)
            self.log_chi2.append(log_chi2)
            
            return log_red_chi2
        
        # set iteration count
        i=0
        
        # initialise model vector
        self.x=np.full(self.A.shape[1],x_default)
        
        # initialse alpha
        # start with a high value and relax it to optimum
        log_alpha_0=ln(0.5*self.__chi2(self.x)/self.__S(self.x))+1.
        log_alpha_1=log_alpha_0-1
        alpha_tol=ln(1.+1./np.sqrt(self.x.size))
        
        # keep track of log_alpha and log_chi2 values
        self.log_alpha=[]
        self.log_chi2=[]
        
        # perform secant iterations
        print()
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
            
            # check if solution has been calculated before
            if log_alpha in self.log_alpha:
                
                j=self.log_alpha.index(log_alpha)
                kappa=self.kappa[j]
                
            else:
                
                # get finite differences
                i+=1
                print()
                print("iteration:",i)
                print("log alpha:",log_alpha)
            
                # perform conjugate gradient minimisation
                start_time=time.time()
                
                self.alpha=np.exp(log_alpha-h)
                res=minimize(self.__J,self.x,method=min_method,
                             jac=self.__grad_J,hessp=self.__hessp_J)
                f_0=ln(self.__chi2(res.x))
                
                self.alpha=np.exp(log_alpha+h)
                res=minimize(self.__J,res.x,method=min_method,
                             jac=self.__grad_J,hessp=self.__hessp_J)
                f_2=ln(self.__chi2(res.x))
                
                self.alpha=np.exp(log_alpha)
                res=minimize(self.__J,res.x,method=min_method,
                             jac=self.__grad_J,hessp=self.__hessp_J)
                f_1=ln(self.__chi2(res.x))
    
                
                # calculate curvature
                dfdx=(f_2-f_0)/(2*h)
                d2fdx2=(f_2-2*f_1+f_0)/h**2
                kappa=d2fdx2/(1+dfdx**2)**(1.5)
                
                self.log_alpha.append(log_alpha)
                self.log_chi2.append(f_1)
                self.kappa.append(kappa)
                self.x=res.x
            
                print("log chi^2_nu:",f_1-ln(self.x.size))
                print("curvature:",kappa)
                print("time taken (s):",time.time()-start_time)
            
            return -kappa
        
        # set iteration count
        i=0
        log_alpha_0=self.log_alpha[0]
        log_alpha_1=self.log_alpha[self.log_alpha.size//2]
        
        # keep track of log_alpha, log_chi2, and kappa
        self.log_alpha=[]
        self.log_chi2=[]
        self.kappa=[]
        
        # set x_tol relative DoF optimal alpha
        xtol=alpha_tol/ln(self.alpha)
        
        # find "kink" in curve
        print()
        print("relaxing fit")
        minimum=minimize_scalar(neg_curvature,(log_alpha_0,log_alpha_1),
                            method="brent",options={"xtol":xtol})
        self.alpha=np.exp(minimum.x)
        print()
        print("applying optimal fit")
        res=minimize(self.__J,self.x,method=min_method,
                     jac=self.__grad_J,hessp=self.__hessp_J)
        self.x=res.x
        
        
        # sort lists
        i=np.argsort(self.log_alpha)
        self.log_alpha=np.array(self.log_alpha)[i]
        self.log_chi2=np.array(self.log_chi2)[i]
        self.kappa=np.array(self.kappa)[i]
        
        # get uncertainties
        self.sigma_x=np.sqrt(2./self.__hess_diag_J(self.x))
        
        return
    
    def x_pad_image(self):
        
        """
        
        Function:
            return padded x vector formatted as image
        
        """
        
        result=self.x.reshape(self.x_shape)
    
        return result
    
    def x_image(self):
        
        """
        
        Function:
            return x vector formatted as image with (padding removed)
        
        """
        
        delta=tuple((np.array(self.x_shape)-np.array(self.y_shape))//2)
        result=self.x.reshape(self.x_shape)
        result=result[delta[0]:-delta[0],delta[1]:-delta[1]]
    
        return result
    
    def sigma_x_image(self):
        
        """
        
        Function:
            return x vector formatted as image with (padding removed)
        
        """
        
        delta=tuple((np.array(self.x_shape)-np.array(self.y_shape))//2)
        result=self.sigma_x.reshape(self.x_shape)
        result=result[delta[0]:-delta[0],delta[1]:-delta[1]]
    
        return result
    
    
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
        hess_diag_chi2=np.array((self.ATR_inv.T.multiply(self.A)).sum(axis=0)).flatten()
        
        result=-self.alpha*hess_diag_S+hess_diag_chi2
        
        return result
    
    