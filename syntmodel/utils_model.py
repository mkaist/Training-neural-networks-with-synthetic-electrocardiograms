import numpy as np

class Prms():
    '''
    Class used to control signal properties where randomize can be called
    when a signal is created to make a unique sample and scale when 
    default limits in __init__ are scaled changing subsequent randomization.
    '''
    
    def __init__(self,
                 pa = 0.1, qa = -0.08, ra = 1.0, sa = -0.08, ta = 0.3,
                 pb = 0.15, qb = 0.1, rb = 0.1, sb = 0.1, tb = 0.5,
                 pd = -0.16, qd = -0.03, rd = 0.0, sd = 0.03, td = 0.35,
                 pm = 1.0, qm = 1.0, rm = 1.0, sm = 1.0, tm = 3.0,
                 mu = 1.0, a = 1.2, std = 0.5, b = 0.075, bc = 0.1, bf = 1/3.6,
                 alpha = 1.0, c = 0.4, wn = 0.002,
                 
                 #NOTE: not all variables are allowed to be randomized
                 pa1=0.05, pa2=0.2,
                 qa1=-0.05, qa2=-0.2,
                 ra1=0.8, ra2=1.2,
                 sa1=-0.05, sa2=-0.2,
                 ta1=0.1, ta2=0.6,
                 
                 pb1=0.065, pb2=0.085,
                 qb1=0.03, qb2=0.08,
                 rb1=0.06, rb2=0.085,
                 sb1=0.03, sb2=0.08,
                 tb1=0.085, tb2=0.21,
                 
                 pd1=-0.12, pd2=-0.18,
                 qd1=-0.03, qd2=-0.05,                 
                 sd1=0.03, sd2=0.05,
                 td1=0.2, td2=0.25,
                 
                 tm1=1.0, tm2=3.0,
                 
                 std1=0.45, std2=0.55,
                 mu1=0.75, mu2=1.0,
                 
                 alpha1 = 0, alpha2= 0.67,
                 c1= 0.0, c2 = 4,
                 wn1= 0.0, wn2= 0.17):         

        self.pa = pa
        self.qa = qa
        self.ra = ra
        self.sa = sa
        self.ta = ta
        self.pb = pb
        self.qb = qb
        self.rb = rb
        self.sb = sb
        self.tb = tb
        self.pd = pd
        self.qd = qd
        self.rd = rd
        self.sd = sd
        self.td = td
        self.pm = pm
        self.qm = qm
        self.rm = rm
        self.sm = sm
        self.tm = tm
        self.mu = mu
        self.a  = a
        self.std= std
        self.b = b
        self.bc = bc
        self.bf = bf
        self.alpha = alpha
        self.c = c
        self.wn = wn
        
        self.pa1, self.pa2 = pa1, pa2
        self.qa1, self.qa2 = qa1, qa2
        self.ra1, self.ra2 = ra1, ra2
        self.sa1, self.sa2 = sa1, sa2
        self.ta1, self.ta2 = ta1, ta2

        self.pb1, self.pb2 = pb1, pb2
        self.qb1, self.qb2 = qb1, qb2
        self.rb1, self.rb2 = rb1, rb2
        self.sb1, self.sb2 = sb1, sb2
        self.tb1, self.tb2 = tb1, tb2

        self.pd1, self.pd2 = pd1, pd2  
        self.qd1, self.qd2 = qd1, qd2
        self.sd1, self.sd2 = sd1, sd2
        self.td1, self.td2 = td1, td2

        self.tm1, self.tm2 = tm1, tm2

        self.mu1, self.mu2 = mu1, mu2
        self.std1, self.std2 = std1, std2

        self.alpha1, self.alpha2 = alpha1, alpha2
        self.c1, self.c2 = c1, c2
        self.wn1, self.wn2 = wn1, wn2  
    
    def randomize(self):
        
        x = lambda l1, l2: np.random.uniform(l1, l2)
                    
        self.pa = x(self.pa1, self.pa2)
        self.qa = x(self.qa1, self.qa2)
        self.ra = x(self.ra1, self.ra2)
        self.sa = x(self.sa1, self.sa2)
        self.ta = x(self.ta1, self.ta2)
        
        self.pb = x(self.pb1, self.pb2)
        self.qb = x(self.qb1, self.qb2)
        self.rb = x(self.rb1, self.rb2)
        self.sb = x(self.sb1, self.sb2)
        self.tb = x(self.tb1, self.tb2)
        
        self.pd = x(self.pd1, self.pd2)
        self.qd = x(self.qd1, self.qd2)
        self.sd = x(self.sd1, self.sd2)
        self.td = x(self.td1, self.td2)
        
        self.tm = x(self.tm1, self.tm2)
        
        self.mu = x(self.mu1, self.mu2)
        self.std = x(self.std1, self.std2)
        
        self.alpha = x(self.alpha1, self.alpha2)
        self.c = x(self.c1, self.c2)
        self.wn = x(self.wn1, self.wn2)
        
    def scale_limits(self, scale_waveform, scale_fiducialpoint, scale_rr, scale_noise):
        
        self.__init__() 
        
        def x(r1, r2, f):
            
            s = np.abs(r1 + r2)            
            fr1, fr2 = np.abs(r1/s), np.abs(r2/s)
            
            d = np.abs( (r2 - r1)*(f-1))   
            r1d, r2d = d*fr1, d*fr2            
            
            if f > 1:
                return (r1 - r1d, r2 + r2d)  
            else:
                return (r1 + r1d, r2 - r2d)        
        
        self.pa1, self.pa2 = x(self.pa1, self.pa2,scale_waveform)        
        self.qa1, self.qa2 = x(self.qa1, self.qa2,scale_waveform)        
        self.sa1, self.sa2 = x(self.sa1, self.sa2,scale_waveform) 
        self.ta1, self.ta2 = x(self.ta1, self.ta2,scale_waveform)         
        
        self.pb1, self.pb2 = x(self.pb1, self.pb2,scale_waveform) 
        self.qb1, self.qb2 = x(self.qb1, self.qb2,scale_waveform)   
        self.sb1, self.sb2 = x(self.sb1, self.sb2,scale_waveform) 
        self.tb1, self.tb2 = x(self.tb1, self.tb2,scale_waveform) 
        
        self.pd1, self.pd2 = x(self.pd1, self.pd2,scale_fiducialpoint) 
        self.qd1, self.qd2 = x(self.qd1, self.qd2,scale_fiducialpoint)     
        self.sd1, self.sd2 = x(self.sd1, self.sd2,scale_fiducialpoint) 
        self.td1, self.td2 = x(self.td1, self.td2,scale_fiducialpoint) 

        self.tm1, self.tm2 = x(self.tm1, self.tm2,scale_waveform) 

        self.mu1, self.mu2 = x(self.mu1, self.mu2, scale_rr)
        self.std1, self.std2 =  x(self.std1, self.std2,scale_rr)
        
        self.alpha1, self.alpha2 = x(self.alpha1, self.alpha2, scale_noise)
        self.c1, self.c2 = x(self.c1, self.c2, scale_noise)
        self.wn1, self.wn2 = x(self.wn1, self.wn2, scale_noise)   
    
  