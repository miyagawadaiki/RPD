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
    """
    else:
        s = (PD(0,p,q,f,epsilon,xi,w)**2
            + PD(1,p,q,f,epsilon,xi,w)**2
            + PD(2,p,q,f,epsilon,xi,w)**2
            + PD(3,p,q,f,epsilon,xi,w)**2
            + PD(4,p,q,f,epsilon,xi,w)**2)
        return .8 if s < 0.00001 else 0.
    """


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



def Calc_Partial_Derivative(l, p, q_list, epsilon, xi, Sx, w):
    one = [1.,1.,1.,1.]
    rlist = []
    if l < 5:
        for q in q_list:
            rlist.append(D(p,q,one,epsilon,xi,w) * PD(l,p,q,Sx,epsilon,xi,w) 
                        -PD(l,p,q,one,epsilon,xi,w) * D(p,q,Sx,epsilon,xi,w))
    else :
        for q in q_list:
            s = 0.
            for i in range(5):
                s += (D(p,q,one,epsilon,xi,w) * PD(i,p,q,Sx,epsilon,xi,w) -PD(i,p,q,one,epsilon,xi,w) * D(p,q,Sx,epsilon,xi,w))**2
            rlist.append(10. if s < 1E-10 else 0.)

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

def change_q(canvas, ax, flag):
    global q_list, corner_case_flag
    if flag:
        corner_case_flag = True
        q_list=[[float(format(i,'05b')[0]),float(format(i,'05b')[1]),float(format(i,'05b')[2]),float(format(i,'05b')[3]),float(format(i,'05b')[4])] for i in range(32)]
    else:
        corner_case_flag = False
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
        #T,R,P,S=4.1,4,2,-1
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
    
    Select_DrawCanvas(canvas, ax, colors = "gray")

def switch_view(canvas, ax, switch_N_to_P):
    l = listbox.curselection()[0]
    selected_opt = listbox.get(l)
    global draw_coord, pd_Sx_flag
    ss = "sX" if pd_Sx_flag else "sY"
    partial = r"d"

    if switch_N_to_P and draw_coord:
        draw_coord = False
        txt.delete(0, tkinter.END)
        if l < 5:
            txt.insert(tkinter.END, r"Partial derivative view ({0}{1}/{0}{2})".format(partial, ss, selected_opt))
        else:
            txt.insert(tkinter.END, r"Partial derivative view (suboptimal)")

    elif switch_N_to_P and not(draw_coord):
        draw_coord = True
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END, f"Normal view")

    else:
        draw_coord = False
        txt.delete(0, tkinter.END)
        if l < 5:
            txt.insert(tkinter.END, r"Partial derivative view ({0}{1}/{0}{2})".format(partial, ss, selected_opt))
        else:
            txt.insert(tkinter.END, r"Partial derivative view (suboptimal)")

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
    #mx = scale8.get()/100.
    #mx = np.exp(scale8.get()/1000.) -1.
    #mx = np.log(scale8.get()/1000.+1.)
    mx = (scale8.get()/1000.)**3
    
    if pd_Sx_flag:
        pds = Calc_Partial_Derivative(l,p,q_list,epsilon,xi,Sx,w)
    else:
        pds = Calc_Partial_Derivative(l,p,q_list,epsilon,xi,Sy,w)
    if l < 5:
        plt.scatter(y, x, s=20, c=pds, alpha=.5, linewidths=0.1, edgecolors='k', cmap='bwr_r', vmin=-mx, vmax=mx, zorder=2)
    else:
        plt.scatter(y, x, s=20, c=pds, alpha=.5, linewidths=0.2, edgecolors='k', cmap='Reds', zorder=2)
    #plt.rcParams["font.size"] = 15
    
    canvas.draw()


def adjust_zd(l,scale_l):
    w=round(1-scale5.get()/100,2)
    epsilon=round(scale6.get()/1000,3)
    xi=round(scale7.get()/1000,3)
    i=1000#stride
    p=[scale0.get()/i,scale1.get()/i,scale2.get()/i,scale3.get()/i,scale4.get()/i]
    RE = R*(1-epsilon-xi)+S*(epsilon+xi)
    SE = S*(1-epsilon-xi)+R*(epsilon+xi)
    TE = T*(1-epsilon-xi)+P*(epsilon+xi)
    PE = P*(1-epsilon-xi)+T*(epsilon+xi)


    sp.var('p0_ p1_ p2_ p3_ p4_ delta_ R_ T_ S_ P_')
    
    pv = [p0_,p1_,p2_,p3_,p4_]
    pl = pv[l]
    
    subs_list = [(pv[i], p[i]) for i in range(5) if i != l]
    subs_list.extend([(R_,RE),(T_, TE),(S_,SE),(P_,PE),(delta_,w)])
    
    eq = (delta_*p2_ +delta_*p3_ -2*delta_*p4_ -1)*(R_ -P_) -(-1 +delta_*p1_ -delta_*p4_)*(T_+S_-2*P_)
    #eq = 1 +2*delta_*p4_ -(1 -delta_*p1_ +delta_*p4_)*(T_+S_) -delta_*p2_ -delta_*p3_
    #display(sp.solve(eq.subs(subs_list), pl))
    #return sp.solve(eq.subs(subs_list), pl)
    s = sp.solve(eq.subs(subs_list), pl)[0]

    if 0. <= s <= 1.:
        p[l] = s
        scale_l.set(int(round(s*1000)))
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END, f"Adjusted p{l}: p=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},{p[3]:.3f},{p[4]:.3f}]")

        Select_DrawCanvas(Canvas, ax1)
    else:
        txt.delete(0, tkinter.END)
        txt.insert(tkinter.END, f"Error: there's no suitable value in [0,1]")








def _set_pd_Sx_flag(event):
    global pd_Sx_flag

    pd_Sx_flag = not(pd_Sx_flag)
    switch_view(Canvas, ax1, draw_coord)
    Select_DrawCanvas(Canvas, ax1)

"""
def _set_corner_case_flag(event):
    global corner_case_flag

    corner_case_flag = not(corner_case_flag)
    change_q(Canvas, ax1)
    Select_DrawCanvas(Canvas, ax1)
"""


draw_coord = True
pd_Sx_flag = True
corner_case_flag = False

if __name__ == "__main__":
    try:
        #generate GUI
        root = tkinter.Tk()
        root.geometry("750x550")
        #root.geometry("700x550")
        root.title("GUI- vs 1,000+2 Strategies Under Discounting and Observation Errors in RPD game")
        
        #generate graph
        fig,ax1 = plt.subplots(figsize=(6,6), dpi=70)
        fig.gca().set_aspect('equal', adjustable='box')
        
        #generate Canvas
        Canvas = FigureCanvasTkAgg(fig, master=root)
        Canvas.get_tk_widget().grid(row=0, column=0, rowspan=1000)
        #T,R,P,S=1.1,1,0,-1.
        T,R,P,S=1.5,1,0,-0.5
        option=0
        q_list=[[random.random(),random.random(),random.random(),random.random(),random.random()] for i in range(1000)]
        
        ReDrawButton = tkinter.Button(text="Other Opponent", width=15, command=partial(change_q, Canvas, ax1, False))
        ReDrawButton.grid(row=8, column=1, columnspan=1)
        TRPSButton = tkinter.Button(text="TRPS 5310", width=15, command=partial(change_5310, Canvas, ax1))
        TRPSButton.grid(row=9, column=1, columnspan=1)
        SaveButton = tkinter.Button(text="Save Fig", width=15, command=save_fig)
        SaveButton.grid(row=10, column=1, columnspan=1)
        SaveButton = tkinter.Button(text="Method", width=15, command=partial(change_way_cal,Canvas, ax1))
        SaveButton.grid(row=7, column=1, columnspan=1)
        SwViewButton = tkinter.Button(text="Switch Normal/AP", width=15, command=partial(switch_view, Canvas, ax1, True))
        SwViewButton.grid(row=11, column=1, columnspan=1)
        QuitButton = tkinter.Button(text="Quit", width=15, command=Quit)
        QuitButton.grid(row=12, column=1, columnspan=1)
        

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

        # for adjust to zd strategy
        AdjustButton = tkinter.Button(text='zd', width=2, command=partial(adjust_zd, 1, scale1))
        AdjustButton.grid(row=3, column=4, columnspan=1)
        AdjustButton = tkinter.Button(text='zd', width=2, command=partial(adjust_zd, 2, scale2))
        AdjustButton.grid(row=4, column=4, columnspan=1)
        AdjustButton = tkinter.Button(text='zd', width=2, command=partial(adjust_zd, 3, scale3))
        AdjustButton.grid(row=5, column=4, columnspan=1)
        AdjustButton = tkinter.Button(text='zd', width=2, command=partial(adjust_zd, 4, scale4))
        AdjustButton.grid(row=6, column=4, columnspan=1)
        

        # for partial derivative view 
        view_var = tkinter.StringVar()
        listbox = tkinter.Listbox(root, height=6, width=8)
        for line in ["q0", "q1","q2","q3", "q4", "suboptimal"]:
            listbox.insert(tkinter.END, line)
        listbox.select_set(1)
        listbox.grid(row=8, column=3, rowspan=4, columnspan=1)
        listbox.bind('<<ListboxSelect>>', lambda event: switch_view(Canvas, ax1, False))

        root.bind("<KeyPress-p>", lambda event: switch_view(Canvas, ax1, True))
        root.bind("<Shift-KeyPress-P>", _set_pd_Sx_flag)
        root.bind("<Shift-KeyPress-C>", lambda event: change_q(Canvas, ax1, not(corner_case_flag)))
        #root.bind("<Shift-KeyPress-C>", _set_corner_case_flag)

        scale8 = tkinter.Scale(root, label='PD max', orient='h', from_=0, to=200, command=partial(Select_DrawCanvas, Canvas, ax1))
        scale8.set(10)
        scale8.grid(row=12, column=3, columnspan=1)

        
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
