#-*-coding:utf-8-*-
import numpy as np
import sympy as sp
import numpy.linalg as LA
import random
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from functools import partial
from tkinter import filedialog

#determinant D(p,q,f)
def D(p,q,f,epsilon,xi,w):
    tau,mu,eta = 1-2*epsilon-xi,1-epsilon-xi,epsilon+xi
    
    Matrix= [[w*(tau*p[1]*q[1]+epsilon*p[1]*q[2]+epsilon*p[2]*q[1]+xi*p[2]*q[2])-1+(1-w)*p[0]*q[0],
              w*(mu*p[1]+eta*p[2])-1+(1-w)*p[0],
              w*(mu*q[1]+eta*q[2])-1+(1-w)*q[0],
              f[0]],
    
             [w*(epsilon*p[1]*q[3]+xi*p[1]*q[4]+tau*p[2]*q[3]+epsilon*p[2]*q[4])+(1-w)*p[0]*q[0],
              w*(eta*p[1]+mu*p[2])-1+(1-w)*p[0],
              w*(mu*q[3]+eta*q[4])  +(1-w)*q[0],
              f[1]],
              
             [w*(epsilon*p[3]*q[1]+tau*p[3]*q[2]+xi*p[4]*q[1]+epsilon*p[4]*q[2])+(1-w)*p[0]*q[0],
              w*(mu*p[3]+eta*p[4])  +(1-w)*p[0],w*(eta*q[1]+mu*q[2])-1+(1-w)*q[0],
              f[2]],
              
             [w*(xi*p[3]*q[3]+epsilon*p[3]*q[4]+epsilon*p[4]*q[3]+tau*p[4]*q[4])+(1-w)*p[0]*q[0],
              w*(eta*p[3]+mu*p[4])+(1-w)*p[0],
              w*(eta*q[3]+mu*q[4])+(1-w)*q[0],
              f[3]]
            ]
    return LA.det(Matrix)

def PD(l,p,q,f,epsilon,xi,w):
    tau,mu,eta = 1-2*epsilon-xi,1-epsilon-xi,epsilon+xi
    
    if l==0:
        pD1 = [[(1-w)*p[0], (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, 1-w, f[0]],
               [(1-w)*p[0], (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, 1-w, f[1]],
               [(1-w)*p[0], (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   1-w, f[2]],
               [(1-w)*p[0], (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   1-w, f[3]]
              ]
        pD2 = [[(1-w)*p[0], (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, w*(eta*q[2]+mu*q[1])-1, f[0]],
               [(1-w)*p[0], (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, w*(eta*q[4]+mu*q[3]),   f[1]],
               [(1-w)*p[0], (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   w*(eta*q[1]+mu*q[2])-1, f[2]],
               [(1-w)*p[0], (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   w*(eta*q[3]+mu*q[4]),   f[3]]
              ]
        pD3 = [[w*(epsilon*p[1]*q[2]+epsilon*p[2]*q[1]+tau*p[1]*q[1]+xi*p[2]*q[2])-1, (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, 1-w, f[0]],
               [w*(epsilon*p[1]*q[3]+epsilon*p[2]*q[4]+tau*p[2]*q[3]+xi*p[1]*q[4]),   (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, 1-w, f[1]],
               [w*(epsilon*p[3]*q[1]+epsilon*p[4]*q[2]+tau*p[3]*q[2]+xi*p[4]*q[1]),   (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   1-w, f[2]],
               [w*(epsilon*p[3]*q[4]+epsilon*p[4]*q[3]+tau*p[4]*q[4]+xi*p[3]*q[3]),   (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   1-w, f[3]]
              ]
        return 2*q[l]*LA.det(pD1) +LA.det(pD2) +LA.det(pD3)
    elif l==1:
        pD1 = [[w*(epsilon*p[2]+tau*p[1]), (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, w*mu,  f[0]],
               [0,                         (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, 0,     f[1]],
               [w*(epsilon*p[3]+xi*p[4]),  (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   w*eta, f[2]],
               [0,                         (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   0,     f[3]]
              ]
        pD2 = [[w*(epsilon*p[2]+tau*p[1]), (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, (1-w)*q[0]+w*eta*q[2]-1,           f[0]],
               [0,                         (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, (1-w)*q[0]+w*(eta*q[4]+mu*q[3]),   f[1]],
               [w*(epsilon*p[3]+xi*p[4]),  (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   (1-w)*q[0]+w*mu*q[2]-1,            f[2]],
               [0,                         (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   (1-w)*q[0]+w*(eta*q[3]+mu*q[4]),   f[3]]
              ]
        pD3 = [[(1-w)*p[0]*q[0]+w*(epsilon*p[1]*q[2]+xi*p[2]*q[2])-1,                               (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, w*mu,  f[0]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[1]*q[3]+epsilon*p[2]*q[4]+tau*p[2]*q[3]+xi*p[1]*q[4]), (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, 0,     f[1]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[4]*q[2]+tau*p[3]*q[2]),                                (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   w*eta, f[2]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[3]*q[4]+epsilon*p[4]*q[3]+tau*p[4]*q[4]+xi*p[3]*q[3]), (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   0,     f[3]]
              ]
        return 2*q[l]*LA.det(pD1) +LA.det(pD2) +LA.det(pD3)
    elif l==2:
        pD1 = [[w*(epsilon*p[1]+xi*p[2]),  (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, w*eta, f[0]],
               [0,                         (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, 0,     f[1]],
               [w*(epsilon*p[4]+tau*p[3]), (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   w*mu , f[2]],
               [0,                         (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   0,     f[3]]
              ]
        pD2 = [[w*(epsilon*p[1]+xi*p[2]),  (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, (1-w)*q[0]+w*mu*q[1]-1,            f[0]],
               [0,                         (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, (1-w)*q[0]+w*(eta*q[4]+mu*q[3]),   f[1]],
               [w*(epsilon*p[4]+tau*p[3]), (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   (1-w)*q[0]+w*eta*q[1]-1,           f[2]],
               [0,                         (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   (1-w)*q[0]+w*(eta*q[3]+mu*q[4]),   f[3]]
              ]
        pD3 = [[(1-w)*p[0]*q[0]+w*(epsilon*p[2]*q[1]+tau*p[1]*q[1])-1,                              (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, w*eta, f[0]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[1]*q[3]+epsilon*p[2]*q[4]+tau*p[2]*q[3]+xi*p[1]*q[4]), (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, 0,     f[1]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[3]*q[1]+xi*p[4]*q[1]),                                 (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   w*mu,  f[2]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[3]*q[4]+epsilon*p[4]*q[3]+tau*p[4]*q[4]+xi*p[3]*q[3]), (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   0,     f[3]]
              ]
        return 2*q[l]*LA.det(pD1) +LA.det(pD2) +LA.det(pD3)

    elif l==3:
        pD1 = [[0,                         (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, 0,     f[0]],
               [w*(epsilon*p[1]+tau*p[2]), (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, w*mu,  f[1]],
               [0,                         (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   0,     f[2]],
               [w*(epsilon*p[4]+xi*p[3]),  (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   w*eta, f[3]]
              ]
        pD2 = [[0,                         (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, (1-w)*q[0]+w*(eta*q[2]+mu*q[1])-1, f[0]],
               [w*(epsilon*p[1]+tau*p[2]), (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, (1-w)*q[0]+w*eta*q[4],             f[1]],
               [0,                         (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   (1-w)*q[0]+w*(eta*q[1]+mu*q[2])-1, f[2]],
               [w*(epsilon*p[4]+xi*p[3]),  (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   (1-w)*q[0]+w*mu*q[4],              f[3]]
              ]
        pD3 = [[(1-w)*p[0]*q[0]+w*(epsilon*p[1]*q[2]+epsilon*p[2]*q[1]+tau*p[1]*q[1]+xi*p[2]*q[2])-1, (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, 0,     f[0]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[2]*q[4]+xi*p[1]*q[4]),                                   (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, w*mu,  f[1]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[3]*q[1]+epsilon*p[4]*q[2]+tau*p[3]*q[2]+xi*p[4]*q[1]),   (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   0,     f[2]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[3]*q[4]+tau*p[4]*q[4]),                                  (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   w*eta, f[3]]
              ]
        return 2*q[l]*LA.det(pD1) +LA.det(pD2) +LA.det(pD3)
    elif l==4:
        pD1 = [[0,                         (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, 0,     f[0]],
               [w*(epsilon*p[2]+xi*p[1]),  (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, w*eta, f[1]],
               [0,                         (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   0,     f[2]],
               [w*(epsilon*p[3]+tau*p[4]), (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   w*mu,  f[3]]
              ]
        pD2 = [[0,                         (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, (1-w)*q[0]+w*(eta*q[2]+mu*q[1])-1, f[0]],
               [w*(epsilon*p[2]+xi*p[1]),  (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, (1-w)*q[0]+w*mu*q[3],              f[1]],
               [0,                         (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   (1-w)*q[0]+w*(eta*q[1]+mu*q[2])-1, f[2]],
               [w*(epsilon*p[3]+tau*p[4]), (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   (1-w)*q[0]+w*eta*q[3],             f[3]]
              ]
        pD3 = [[(1-w)*p[0]*q[0]+w*(epsilon*p[1]*q[2]+epsilon*p[2]*q[1]+tau*p[1]*q[1]+xi*p[2]*q[2])-1, (1-w)*p[0]+w*(eta*p[2]+mu*p[1])-1, 0,     f[0]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[1]*q[3]+tau*p[2]*q[3]),                                  (1-w)*p[0]+w*(eta*p[1]+mu*p[2])-1, w*eta, f[1]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[3]*q[1]+epsilon*p[4]*q[2]+tau*p[3]*q[2]+xi*p[4]*q[1]),   (1-w)*p[0]+w*(eta*p[4]+mu*p[3]),   0,     f[2]],
               [(1-w)*p[0]*q[0]+w*(epsilon*p[4]*q[3]+xi*p[3]*q[3]),                                   (1-w)*p[0]+w*(eta*p[3]+mu*p[4]),   w*mu,  f[3]]
              ]

        return 2*q[l]*LA.det(pD1) +LA.det(pD2) +LA.det(pD3)


def Calculation_Determinant(p,q_list,epsilon,xi,Sx,Sy,w):
    lx,ly = [],[]
    for i in range(len(q_list)):
        q=q_list[i]
        v1=[1,1,1,1]
        vdot1=D(p,q,v1,epsilon,xi,w)
        sy = D(p,q,Sx,epsilon,xi,w)/vdot1
        sx = D(p,q,Sy,epsilon,xi,w)/vdot1
        
        lx.append(sx)
        ly.append(sy)
    return lx,ly

#Inverse Matrix, Hilbe et al.,2015,GEB
def Calculation_Inverse(p,q_list,epsilon,xi,Sx,Sy,w):
    lx,ly=[],[]
    tau= 1-2*epsilon-xi
    for i in range(len(q_list)):
        q=q_list[i]

        M = np.array([
         [tau*p[1]*q[1]        +epsilon*p[1]*q[2]        +epsilon*p[2]*q[1]        +xi*p[2]*q[2],
          tau*p[1]*(1-q[1])    +epsilon*p[1]*(1-q[2])    +epsilon*p[2]*(1-q[1])    +xi*p[2]*(1-q[2]),
          tau*(1-p[1])*q[1]    +epsilon*(1-p[1])*q[2]    +epsilon*(1-p[2])*q[1]    +xi*(1-p[2])*q[2],
          tau*(1-p[1])*(1-q[1])+epsilon*(1-p[1])*(1-q[2])+epsilon*(1-p[2])*(1-q[1])+xi*(1-p[2])*(1-q[2])],
          
         [epsilon*p[1]*q[3]        +xi*p[1]*q[4]        +tau*p[2]*q[3]        +epsilon*p[2]*q[4],
          epsilon*p[1]*(1-q[3])    +xi*p[1]*(1-q[4])    +tau*p[2]*(1-q[3])    +epsilon*p[2]*(1-q[4]),
          epsilon*(1-p[1])*q[3]    +xi*(1-p[1])*q[4]    +tau*(1-p[2])*q[3]    +epsilon*(1-p[2])*q[4],
          epsilon*(1-p[1])*(1-q[3])+xi*(1-p[1])*(1-q[4])+tau*(1-p[2])*(1-q[3])+epsilon*(1-p[2])*(1-q[4])],
          
         [epsilon*p[3]*q[1]        +tau*p[3]*q[2]        +xi*p[4]*q[1]        +epsilon*p[4]*q[2],
          epsilon*p[3]*(1-q[1])    +tau*p[3]*(1-q[2])    +xi*p[4]*(1-q[1])    +epsilon*p[4]*(1-q[2]),
          epsilon*(1-p[3])*q[1]    +tau*(1-p[3])*q[2]    +xi*(1-p[4])*q[1]    +epsilon*(1-p[4])*q[2],
          epsilon*(1-p[3])*(1-q[1])+tau*(1-p[3])*(1-q[2])+xi*(1-p[4])*(1-q[1])+epsilon*(1-p[4])*(1-q[2])],
          
         [xi*p[3]*q[3]        +epsilon*p[3]*q[4]        +epsilon*p[4]*q[3]        +tau*p[4]*q[4],
          xi*p[3]*(1-q[3])    +epsilon*p[3]*(1-q[4])    +epsilon*p[4]*(1-q[3])    +tau*p[4]*(1-q[4]),
          xi*(1-p[3])*q[3]    +epsilon*(1-p[3])*q[4]    +epsilon*(1-p[4])*q[3]    +tau*(1-p[4])*q[4],
          xi*(1-p[3])*(1-q[3])+epsilon*(1-p[3])*(1-q[4])+epsilon*(1-p[4])*(1-q[3])+tau*(1-p[4])*(1-q[4])]
         ])
        
        IwM=np.eye(4)-w*M
        try:
            inverseIwM=np.linalg.inv(IwM)
        except Exception as error:
            txt.delete(0, tkinter.END)
            txt.insert(tkinter.END,error.__str__())
            print("It can not be executed on w=1")
        
        v0=np.array([[p[0]*q[0],p[0]*(1-q[0]),(1-p[0])*q[0],(1-p[0])*(1-q[0])]])
        u=(1-w)*np.dot(v0,inverseIwM)
        sx=np.dot(u,Sx)
        sy=np.dot(u,Sy)
        lx.append(sx)
        ly.append(sy)
    return ly,lx


"""
def calc_D(p,q,Sx,epsilon,xi,w):
    sp.var('p0_ p1_ p2_ p3_ p4_')
    sp.var('q0_ q1_ q2_ q3_ q4_')
    sp.var('f1_ f2_ f3_ f4_ epsilon_ tau_ mu_ eta_ xi_ w_')
    
    D = sp.Matrix([
        [w_*(tau_*p1_*q1_+epsilon_*p1_*q2_+epsilon_*p2_*q1_+xi_*p2_*q2_)-1+(1-w_)*p0_*q0_,
         w_*(mu_*p1_+eta_*p2_)-1+(1-w_)*p0_,
         w_*(mu_*q1_+eta_*q2_)-1+(1-w_)*q0_,
         f1_
        ],
        [w_*(epsilon_*p1_*q3_+xi_*p1_*q4_+tau_*p2_*q3_+epsilon_*p2_*q4_)+(1-w_)*p0_*q0_,
         w_*(eta_*p1_+mu_*p2_)-1+(1-w_)*p0_,
         w_*(mu_*q3_+eta_*q4_)  +(1-w_)*q0_,
         f3_
        ],
        [w_*(epsilon_*p3_*q1_+tau_*p3_*q2_+xi_*p4_*q1_+epsilon_*p4_*q2_)+(1-w_)*p0_*q0_,
         w_*(mu_*p3_+eta_*p4_)  +(1-w_)*p0_,
         w_*(eta_*q1_+mu_*q2_)-1+(1-w_)*q0_,
         f2_
        ],
        [w_*(xi_*p3_*q3_+epsilon_*p3_*q4_+epsilon_*p4_*q3_+tau_*p4_*q4_)+(1-w_)*p0_*q0_,
         w_*(eta_*p3_+mu_*p4_)+(1-w_)*p0_,
         w_*(eta_*q3_+mu_*q4_)+(1-w_)*q0_,
         f4_
        ]
    ])
    
    D_ = D.subs(list(zip([p0_,p1_,p2_,p3_,p4_],p)))
    D_ = D_.subs(list(zip([q0_,q1_,q2_,q3_,q4_],q)))
    D_ = D_.subs(list(zip([f1_,f2_,f3_,f4_],Sx)))
    D_ = D_.subs([(tau_,1-2*epsilon_-xi_),(mu_,1-epsilon_-xi_),(eta_,epsilon_+xi_),
                 (epsilon_,epsilon),(xi_,xi),(w_,w)])
    
    return D_.det()


def calc_pD(l,p,q,Sx,epsilon,xi,w):
    sp.var('p0_ p1_ p2_ p3_ p4_')
    sp.var('q0_ q1_ q2_ q3_ q4_')
    sp.var('f1_ f2_ f3_ f4_ epsilon_ tau_ mu_ eta_ xi_ w_')

    qv = (q0_,q1_,q2_,q3_,q4_)
    qd = q.copy()
    qd[l] = 0.

    subs_list = [(p0_, p[0]), (p1_, p[1]), (p2_, p[2]), (p3_, p[3]), (p4_, p[4]),
                 (q0_, qd[0]), (q1_, qd[1]), (q2_, qd[2]), (q3_, qd[3]), (q4_, qd[4]),
                 (f1_, Sx[0]), (f2_, Sx[1]), (f3_, Sx[2]), (f4_, Sx[3]),
                 (tau_,1-2*epsilon_-xi_),(mu_,1-epsilon_-xi_),(eta_,epsilon_+xi_),
                 (epsilon_,epsilon),(xi_,xi),(w_,w)
                ]

    D = sp.Matrix([
        [w_*(tau_*p1_*q1_+epsilon_*p1_*q2_+epsilon_*p2_*q1_+xi_*p2_*q2_)-1+(1-w_)*p0_*q0_,
         w_*(mu_*p1_+eta_*p2_)-1+(1-w_)*p0_,
         w_*(mu_*q1_+eta_*q2_)-1+(1-w_)*q0_,
         f1_
        ],
        [w_*(epsilon_*p1_*q3_+xi_*p1_*q4_+tau_*p2_*q3_+epsilon_*p2_*q4_)+(1-w_)*p0_*q0_,
         w_*(eta_*p1_+mu_*p2_)-1+(1-w_)*p0_,
         w_*(mu_*q3_+eta_*q4_)  +(1-w_)*q0_,
         f3_
        ],
        [w_*(epsilon_*p3_*q1_+tau_*p3_*q2_+xi_*p4_*q1_+epsilon_*p4_*q2_)+(1-w_)*p0_*q0_,
         w_*(mu_*p3_+eta_*p4_)  +(1-w_)*p0_,
         w_*(eta_*q1_+mu_*q2_)-1+(1-w_)*q0_,
         f2_
        ],
        [w_*(xi_*p3_*q3_+epsilon_*p3_*q4_+epsilon_*p4_*q3_+tau_*p4_*q4_)+(1-w_)*p0_*q0_,
         w_*(eta_*p3_+mu_*p4_)+(1-w_)*p0_,
         w_*(eta_*q3_+mu_*q4_)+(1-w_)*q0_,
         f4_
        ]
    ])

    pD = sp.Matrix([
      [sp.diff(D[0,0],qv[l]), sp.diff(D[0,1],qv[l]), sp.diff(D[0,2],qv[l]), sp.diff(D[0,3],qv[l])],
      [sp.diff(D[1,0],qv[l]), sp.diff(D[1,1],qv[l]), sp.diff(D[1,2],qv[l]), sp.diff(D[1,3],qv[l])],
      [sp.diff(D[2,0],qv[l]), sp.diff(D[2,1],qv[l]), sp.diff(D[2,2],qv[l]), sp.diff(D[2,3],qv[l])],
      [sp.diff(D[3,0],qv[l]), sp.diff(D[3,1],qv[l]), sp.diff(D[3,2],qv[l]), sp.diff(D[3,3],qv[l])]
     ])

    pD1 = sp.Matrix([
      [pD[0,0], D[0,1], pD[0,2], D[0,3]],
      [pD[1,0], D[1,1], pD[1,2], D[1,3]],
      [pD[2,0], D[2,1], pD[2,2], D[2,3]],
      [pD[3,0], D[3,1], pD[3,2], D[3,3]]
    ]).subs(subs_list)

    pD2 = sp.Matrix([
      [pD[0,0], D[0,1], D[0,2], D[0,3]],
      [pD[1,0], D[1,1], D[1,2], D[1,3]],
      [pD[2,0], D[2,1], D[2,2], D[2,3]],
      [pD[3,0], D[3,1], D[3,2], D[3,3]]
    ]).subs(subs_list)

    pD3 = sp.Matrix([
      [D[0,0], D[0,1], pD[0,2], D[0,3]],
      [D[1,0], D[1,1], pD[1,2], D[1,3]],
      [D[2,0], D[2,1], pD[2,2], D[2,3]],
      [D[3,0], D[3,1], pD[3,2], D[3,3]]
    ]).subs(subs_list)

    return 2*q[l]*pD1.det() +pD2.det() +pD3.det()
"""


def Calc_Partial_Derivative(l, p, q_list, epsilon, xi, Sx, w):
    one = [1.,1.,1.,1.]
    rlist = []
    for q in q_list:
        rlist.append(D(p,q,one,epsilon,xi,w) * PD(l,p,q,Sx,epsilon,xi,w) 
                    -PD(l,p,q,one,epsilon,xi,w) * D(p,q,Sx,epsilon,xi,w))

    return rlist


def Select_Method_Calculation(p,q_list,epsilon,xi,Sx,Sy,w,option):
    if option==0:
        lx,ly=Calculation_Determinant(p,q_list,epsilon,xi,Sx,Sy,w)
    elif option==1:
        if w==1:#Cannot Calculate, return wrong value when w=1
            lx,ly=100,100   
        else:
            lx,ly=Calculation_Inverse(p,q_list,epsilon,xi,Sx,Sy,w)
    return lx,ly

def Quit():
    global root
    root.quit()
    root.destroy()

def change_q(canvas, ax):
    global q_list
    q_list=[[random.random(),random.random(),random.random(),random.random(),random.random()] for i in range(1000)]
    txt.delete(0, tkinter.END)
    txt.insert(tkinter.END,"changed opponents")
    Select_DrawCanvas(canvas, ax, colors = "gray")

def change_5310(canvas, ax):
    global T,R,P,S
    if T==5:
        T,R,P,S=1.5,1,0,-0.5
    else:
        T,R,P,S=5,3,1,0
    Select_DrawCanvas(canvas, ax, colors = "gray")

def save_fig():
    filepath = filedialog.askdirectory(initialdir = dir)
    path=filepath+'\\fig.png'
    print('save_Image')
    print(path)
    plt.savefig(path)

def change_way_cal(canvas, ax):
    global option
    if option==1:
        option=0
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END,"The method of calcuculation is DETERMINANT")
    else:
        option=1
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END,"The method of calcuculation is INVERSE MATRIX")
    
    DrawCanvas(canvas, ax, colors = "gray")

def switch_view(canvas, ax, switch_N_to_P):
    selected_opt = listbox.get(listbox.curselection()[0])
    global draw_coord

    if switch_N_to_P and draw_coord:
        draw_coord = False
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END, f"Partial derivative view ({selected_opt})")

    elif switch_N_to_P and not(draw_coord):
        draw_coord = True
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END, f"Normal view")

    else:
        draw_coord = False
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END, f"Partial derivative view ({selected_opt})")

    Select_DrawCanvas(canvas, ax, colors = "gray")


def Select_DrawCanvas(canvas, ax, colors='gray'):
    global draw_coord

    if draw_coord:
        DrawCanvas_coord(canvas, ax, colors)
    else:
        DrawCanvas_adapting_path(canvas, ax, colors)

    
def DrawCanvas_coord(canvas, ax, colors = "gray"):
    #draw_coord = True
    #txt.delete(0, tkinter.END)
    #txt.insert(tkinter.END, f"Normal view")
    ax.cla()
    
    w=round(1-scale5.get()/100,2)
    epsilon=round(scale6.get()/1000,3)
    xi=round(scale7.get()/1000,3)
    #option=scale8.get()
    
    i=1000#stride
    #p=(p0,p1,p2,p3,p4)
    p=[scale0.get()/i,scale1.get()/i,scale2.get()/i,scale3.get()/i,scale4.get()/i]
    

    RE = R*(1-epsilon-xi)+S*(epsilon+xi)
    SE = S*(1-epsilon-xi)+R*(epsilon+xi)
    TE = T*(1-epsilon-xi)+P*(epsilon+xi)
    PE = P*(1-epsilon-xi)+T*(epsilon+xi)
    
    plt.title(r"(T,R,P,S)=("+str(T)+","+str(R)+","+str(P)+","+str(S)+")\n"\
              +r"$(w,\epsilon,\xi)=($"+str(w)+","+str(epsilon)+","+str(xi)+r"$)$",
              fontsize=15)
    
    plt.ylabel(f"Payoff of ({p[1]},{p[2]},{p[3]},{p[4]}),$p_0=${p[0]}",fontsize=15)
    plt.xlabel("Payoff of Opponent",fontsize=15)
    
    plt.grid()
    plt.xlim([S-0.2,T+0.2])
    plt.ylim([S-0.2,T+0.2])
    if R==3:
        xy_list=[[P-0.65,P-0.35],[R+0.1, R+0.1],[T-0.5, S+0.9],[S+1, T-0.5]]
    else:
        xy_list=[[P-0.35,P-0.15],[R+0.05, R+0.05],[T-0.5, S-0.105],[S+0.4, T-0.105]]
    
    plt.text(xy_list[0][0],xy_list[0][1], r'$(P,P)$',fontsize=15)
    plt.text(xy_list[1][0],xy_list[1][1], r'$(R,R)$',fontsize=15)
    plt.text(xy_list[2][0],xy_list[2][1], r'$(T,S)$',fontsize=15)
    plt.text(xy_list[3][0],xy_list[3][1], r'$(S,T)$',fontsize=15)
    plt.plot([R,S,P,T,R],[R,T,P,S,R],'r',color="black",markersize=3,alpha=0.5)
    plt.plot([R,S,P,T,R],[R,T,P,S,R],'o',color="black",markersize=3)
    plt.plot([RE,SE,PE,TE,RE],[RE,TE,PE,SE,RE],'r',color="black",markersize=3,alpha=0.5)
    plt.plot([RE,SE,PE,TE,RE],[RE,TE,PE,SE,RE],'o',color="black",markersize=3)
    plt.plot([R,P],[R,P],'r',linestyle="dashed",color="black",markersize=3,alpha=0.5)
    
    Sx,Sy = [RE,SE,TE,PE],[RE,TE,SE,PE]#expected stage payoff vector
    y,x = Select_Method_Calculation(p,q_list,epsilon,xi,Sx,Sy,w,option)
    
    x0,y0=Select_Method_Calculation(p,[[0,0,0,0,0]],epsilon,xi,Sx,Sy,w,option)
    x1,y1=Select_Method_Calculation(p,[[1,1,1,1,1]],epsilon,xi,Sx,Sy,w,option)
    plt.plot(y,x,'go',markersize=3,alpha=0.8)
    plt.plot(x0,y0,'ro',markersize=5,alpha=0.8)
    plt.plot(x1,y1,'bo',markersize=5,alpha=0.8)
    #plt.rcParams["font.size"] = 15
    
    canvas.draw()

def DrawCanvas_adapting_path(canvas, ax, colors = "gray"):
    ax.cla()
    
    w=round(1-scale5.get()/100,2)
    epsilon=round(scale6.get()/1000,3)
    xi=round(scale7.get()/1000,3)
    #option=scale8.get()
    
    i=1000#stride
    #p=(p0,p1,p2,p3,p4)
    p=[scale0.get()/i,scale1.get()/i,scale2.get()/i,scale3.get()/i,scale4.get()/i]
    

    RE = R*(1-epsilon-xi)+S*(epsilon+xi)
    SE = S*(1-epsilon-xi)+R*(epsilon+xi)
    TE = T*(1-epsilon-xi)+P*(epsilon+xi)
    PE = P*(1-epsilon-xi)+T*(epsilon+xi)
    
    plt.title(r"(T,R,P,S)=("+str(T)+","+str(R)+","+str(P)+","+str(S)+")\n"\
              +r"$(w,\epsilon,\xi)=($"+str(w)+","+str(epsilon)+","+str(xi)+r"$)$",
              fontsize=15)
    
    plt.ylabel(f"Payoff of ({p[1]},{p[2]},{p[3]},{p[4]}),$p_0=${p[0]}",fontsize=15)
    plt.xlabel("Payoff of Opponent",fontsize=15)
    
    plt.grid()
    plt.xlim([S-0.2,T+0.2])
    plt.ylim([S-0.2,T+0.2])
    if R==3:
        xy_list=[[P-0.65,P-0.35],[R+0.1, R+0.1],[T-0.5, S+0.9],[S+1, T-0.5]]
    else:
        xy_list=[[P-0.35,P-0.15],[R+0.05, R+0.05],[T-0.5, S-0.105],[S+0.4, T-0.105]]
    
    plt.text(xy_list[0][0],xy_list[0][1], r'$(P,P)$',fontsize=15)
    plt.text(xy_list[1][0],xy_list[1][1], r'$(R,R)$',fontsize=15)
    plt.text(xy_list[2][0],xy_list[2][1], r'$(T,S)$',fontsize=15)
    plt.text(xy_list[3][0],xy_list[3][1], r'$(S,T)$',fontsize=15)
    plt.plot([R,S,P,T,R],[R,T,P,S,R],'r',color="black",markersize=3,alpha=0.5)
    plt.plot([R,S,P,T,R],[R,T,P,S,R],'o',color="black",markersize=3)
    plt.plot([RE,SE,PE,TE,RE],[RE,TE,PE,SE,RE],'r',color="black",markersize=3,alpha=0.5)
    plt.plot([RE,SE,PE,TE,RE],[RE,TE,PE,SE,RE],'o',color="black",markersize=3)
    plt.plot([R,P],[R,P],'r',linestyle="dashed",color="black",markersize=3,alpha=0.5)
    
    l = listbox.curselection()[0]
    Sx,Sy = [RE,SE,TE,PE],[RE,TE,SE,PE]#expected stage payoff vector

    q_list.append([0,0,0,0,0]); q_list.append([1,1,1,1,1]);
    y,x = Select_Method_Calculation(p,q_list,epsilon,xi,Sx,Sy,w,option)
    
    pds = Calc_Partial_Derivative(l,p,q_list,epsilon,xi,Sx,w)
    plt.scatter(y, x, s=20, c=pds, alpha=1, linewidths=0.1, edgecolors='k', cmap='bwr_r', vmin=-1., vmax=1., zorder=2)
    #plt.scatter(y,x,s=3,c='k',alpha=0.8)
    #plt.rcParams["font.size"] = 15
    
    canvas.draw()

draw_coord = True

if __name__ == "__main__":
    try:
        #generate GUI
        root = tkinter.Tk()
        root.geometry("700x550")
        root.title("GUI- vs 1,000+2 Strategies Under Discounting and Observation Errors in RPD game")
        
        #generate graph
        fig,ax1 = plt.subplots(figsize=(6,6), dpi=70)
        fig.gca().set_aspect('equal', adjustable='box')
        
        #generate Canvas
        Canvas = FigureCanvasTkAgg(fig, master=root)
        Canvas.get_tk_widget().grid(row=0, column=0, rowspan=1000)
        T,R,P,S=1.5,1,0,-0.5
        option=0
        q_list=[[random.random(),random.random(),random.random(),random.random(),random.random()] for i in range(1000)]
        
        ReDrawButton = tkinter.Button(text="Other Opponent", width=15, command=partial(change_q, Canvas, ax1))
        ReDrawButton.grid(row=12, column=1, columnspan=1)
        TRPSButton = tkinter.Button(text="TRPS 5310", width=15, command=partial(change_5310, Canvas, ax1))
        TRPSButton.grid(row=14, column=1, columnspan=1)
        SaveButton = tkinter.Button(text="Save Fig", width=15, command=save_fig)
        SaveButton.grid(row=16, column=1, columnspan=1)
        SaveButton = tkinter.Button(text="Method", width=15, command=partial(change_way_cal,Canvas, ax1))
        SaveButton.grid(row=10, column=1, columnspan=1)
        SwViewButton = tkinter.Button(text="Switch Normal/AP", width=15, command=partial(switch_view, Canvas, ax1, True))
        SwViewButton.grid(row=18, column=1, columnspan=1)
        QuitButton = tkinter.Button(text="Quit", width=15, command=Quit)
        QuitButton.grid(row=20, column=1, columnspan=1)
        

        scale0 = tkinter.Scale(root, label='p0', orient='h', from_=0, to=1000, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale0.grid(row=2, column=3, columnspan=1)
        scale1 = tkinter.Scale(root, label='p1', orient='h', from_=0, to=1000, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale1.grid(row=3, column=3, columnspan=1)
        scale2 = tkinter.Scale(root, label='p2',orient='h', from_=0.0, to=1000, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale2.grid(row=4, column=3, columnspan=1)
        scale3 = tkinter.Scale(root, label='p3',orient='h', from_=0, to=1000, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale3.grid(row=5, column=3, columnspan=1)
        scale4 = tkinter.Scale(root, label='p4',orient='h', from_=0, to=1000, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale4.grid(row=6, column=3, columnspan=1)
        
        scale5 = tkinter.Scale(root, label='discount rate',orient='h', from_=0, to=100, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale5.grid(row=2, column=1, columnspan=1)
        scale6 = tkinter.Scale(root, label='epsilon',orient='h', from_=0, to=300, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale6.grid(row=3, column=1, columnspan=1)
        scale7 = tkinter.Scale(root, label='xi', orient='h', from_=0, to=300, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale7.grid(row=4, column=1, columnspan=1)

        # for partial derivative view 
        view_var = tkinter.StringVar()
        listbox = tkinter.Listbox(root, height=5, width=10)
        for line in ["q0", "q1","q2","q3", "q4"]:
            listbox.insert(tkinter.END, line)
        listbox.select_set(1)
        listbox.grid(row=6, column=1)
        listbox.bind('<<ListboxSelect>>', lambda event: switch_view(Canvas, ax1, False))

        
        lbl = tkinter.Label(text='Message:')
        lbl.place(x=10, y=460)
        txt = tkinter.Entry(width=50)
        txt.place(x=10, y=480)
        txt.insert(tkinter.END,"Welcome!")
        
        Select_DrawCanvas(Canvas,ax1)
        root.mainloop()
    except Exception as error:
        #import traceback
        #traceback.print_exc()
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END,error.__str__())
