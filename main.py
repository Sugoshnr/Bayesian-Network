import numpy as np
import scipy.stats
import xlrd
import math
#import matplotlib.pyplot as plt
#import pandas as pd
#from pandas.tools.plotting import scatter_matrix

def prod(a,b):
    prod_sum=0;
    for i in range(49):
        prod_sum+=a[i]*b[i];
    return prod_sum;

def sig1(beta,x,y):
    sigma=0;
    z=[]
    for i in range(49):
        z.append(math.pow(((float(beta[0])+float(beta[1])*x[i])-y[i]),2))
        sigma+=z[i];#**2;
    sigma=sigma/49;
    return sigma,z;

def sig2(beta,x1,x2,y):
    sigma=0;
    z=[]
    for i in range(49):
        z.append(float((beta[0]+beta[1]*x1[i]+beta[2]*x2[i])-y[i]));
        sigma+=z[i]**2;
    sigma=sigma/49;
    return sigma,z;


def sig3(beta,x1,x2,x3,y):
    sigma=0;
    z=[]
    for i in range(49):
        z.append(float((beta[0]+beta[1]*x1[i]+beta[2]*x2[i]+beta[3]*x3[i])-y[i]));
        sigma+=z[i]**2;
    sigma=sigma/49;
    return sigma,z;

def logpdf1(beta,x,sigma,y):
    pdf=0;
    for i in range(49):
        logterm=math.log(2*math.pi*sigma)
        logterm*=-1/2
        betaterm=((beta[0]+beta[1]*x[i]-y[i])**2)/(-2*sigma)
        pdf=pdf+logterm+betaterm
    return pdf

def logpdf2(beta,x1,x2,sigma,y):
    pdf=0;
    for i in range(49):
        logterm=math.log(2*math.pi*sigma)
        logterm*=-1/2
        betaterm=(((beta[0]+beta[1]*x1[i]+beta[2]*x2[i])-y[i])**2)/(-2*sigma)
        pdf=pdf+logterm+betaterm
    return pdf

def logpdf3(beta,x1,x2,x3,sigma,y):
    pdf=0;
    for i in range(49):
        logterm=math.log(2*math.pi*sigma)
        logterm*=-1/2
        betaterm=(((beta[0]+beta[1]*x1[i]+beta[2]*x2[i]+beta[3]*x3[i])-y[i])**2)/(-2*sigma)
        pdf=pdf+logterm+betaterm
    return pdf
        
        

def cond_prob1(list_y,list_x):
    x0=np.ones(49);
    x1=np.array(list_x)
    a=np.zeros((2,2))
    row=0;col=0;
    for i in x0,x1:
        for j in x0,x1:
            a[row][col]=(prod(j,i));
            col=col+1;
        row=row+1;
        col=0;
    y1=np.array(list_y);
    y00=prod(y1,x0);
    y10=prod(y1,x1);
    Y=[y00],[y10];
    Y=np.asmatrix(Y)
    #A=np.asarray(a)
    A=np.asmatrix(a)
    #print A
    A_inv = np.linalg.inv(A)
    beta=A_inv*Y
    #print "--",type(beta),float(beta[0][0])
    sigma,X=sig1(beta,x1,y1)
    logterm=math.log(2*math.pi*sigma)
    #print Y
    #log_pdf=logpdf1(beta,x1,sigma,y1)
    log_pdf=(-24.5)*((logterm+1))
    return log_pdf



def cond_prob2(list_y,list_x1,list_x2):
    x0=np.ones(49);
    x1=np.array(list_x1)
    x2=np.array(list_x2)
    a=np.zeros((3,3))
    row=0;col=0;
    for i in x0,x1,x2:
        for j in x0,x1,x2:
            a[row][col]=(prod(j,i));
            col=col+1;
        row=row+1;
        col=0;
    y1=np.array(list_y);
    y00=prod(y1,x0);
    y10=prod(y1,x1);
    y20=prod(y1,x2);
    Y=[y00],[y10],[y20];
    Y=np.asmatrix(Y)
    A=np.asmatrix(a)
    A_inv = np.linalg.inv(A)
    beta=A_inv*Y
    sigma,X=sig2(beta,x1,x2,y1)
    logterm=math.log(2*math.pi*sigma)
    #log_pdf=logpdf1(beta,x1,sigma,y1)
    log_pdf=(-24.5)*((logterm+1))
    return log_pdf




def cond_prob3(list_y,list_x1,list_x2,list_x3):
    x0=np.ones(49);
    x1=np.array(list_x1)
    x2=np.array(list_x2)
    x3=np.array(list_x3)
    a=np.zeros((4,4))
    row=0;col=0;
    for i in x0,x1,x2,x3:
        for j in x0,x1,x2,x3:
            a[row][col]=(prod(i,j));
            col=col+1;
        row=row+1;
        col=0;
    y1=np.array(list_y);
    y00=prod(y1,x0);
    y10=prod(y1,x1);
    y20=prod(y1,x2);
    y30=prod(y1,x3)
    Y=[y00],[y10],[y20],[y30];
    Y=np.asmatrix(Y)
    A=np.asmatrix(a)
    A_inv = np.linalg.inv(A)
    beta=A_inv*Y
    sigma,X=sig3(beta,x1,x2,x3,y1)
    logterm=math.log(2*math.pi*sigma)
    #log_pdf=logpdf3(beta,x1,x2,x3,sigma,y1)
    log_pdf=(-24.5)*((logterm+1))
    return log_pdf






data=xlrd.open_workbook('university data.xlsx')
data_sheet=data.sheet_by_index(0)
list1=list();list2=list();list3=list();list4=list();
for i in range(data_sheet.nrows-2):
    list1.append(data_sheet.cell(i+1,2).value);
    list2.append(data_sheet.cell(i+1,3).value);
    list3.append(data_sheet.cell(i+1,4).value);
    list4.append(data_sheet.cell(i+1,5).value);
#list11=np.array(list1);list22=np.array(list2);list33=np.array(list3);list44=np.array(list4);

#Mean-Variance-Standard_Devation
mu1=np.mean(list1);mu2=np.mean(list2);mu3=np.mean(list3);mu4=np.mean(list4);
var1=np.var(list1)*49/48;var2=np.var(list2)*49/48;var3=np.var(list3)*49/48;var4=np.var(list4)*49/48;
sigma1=math.sqrt(var1);sigma2=math.sqrt(var2);sigma3=math.sqrt(var3);sigma4=math.sqrt(var4);


print ('UBitName = sugoshna')
print ('personNumber = 50207357')
print ('mu1 = %.3f\nmu2 = %.3f\nmu3 = %.3f\nmu4 = %.3f'%(mu1,mu2,mu3,mu4))
print ('var1 = %.3f\nvar2 = %.3f\nvar3 = %.3f\nvar4 = %.3f'%(var1,var2,var3,var4))
print ('sigma1 = %.3f\nsigma2 = %.3f\nsigma3 = %.3f\nsigma4 = %.3f'%(sigma1,sigma2,sigma3,sigma4))

#Covariance and Correlation matrices
mat= np.vstack([list1,list2,list3,list4])
cov= np.cov(mat);
cor= np.corrcoef(mat);
cov_list=cov.tolist()
for i in range(4):
    for j in range(4):
        cov_list[i][j]=round(cov_list[i][j],3);
        cor[i][j]=round(cor[i][j],3);
print ('covarianceMat = ')
cov_all=np.asmatrix(np.asarray([cov_list[0],cov_list[1],cov_list[2],cov_list[3]]))
print cov_all
print ('correlationMat = ')
print (cor)
#plt.matshow(cor);
#plt.scatter(list3,list4)
#plt.show();


pdf_1=scipy.stats.norm.pdf(list1,loc=mu1,scale=sigma1)
pdf_2=scipy.stats.norm.pdf(list2,loc=mu2,scale=sigma2)
pdf_3=scipy.stats.norm.pdf(list3,loc=mu3,scale=sigma3)
pdf_4=scipy.stats.norm.pdf(list4,loc=mu4,scale=sigma4)

prob_prod=[];
log_sum=[];
log_pdf_1=0
log_pdf_2=0
log_pdf_3=0
log_pdf_4=0

for i in range(49):
    prob_prod.append(pdf_1[i]*pdf_2[i]*pdf_3[i]*pdf_4[i]);
    log_pdf_1+=math.log(pdf_1[i])
    log_pdf_2+=math.log(pdf_2[i])
    log_pdf_3+=math.log(pdf_3[i])
    log_pdf_4+=math.log(pdf_4[i])
    log_sum.append(math.log(prob_prod[i]));    

print ('logLikelihood = %.3f'%np.sum(log_sum))
print log_pdf_1,log_pdf_2,log_pdf_3,log_pdf_4
BNGraph=[[0,1,0,1],[0,0,0,0],[0,0,0,0],[0,0,1,0]]
print ('BNgraph = ')
print (np.asmatrix(BNGraph))


##
##print log_pdf_1,log_pdf_2,log_pdf_3,log_pdf_4
##
##print "-----1----"
##
##for i in [list2,list3,list4]:
##    print cond_prob1(list1,i)
##
##print "-----2----"
##
##for i,j in [[list2,list3],[list3,list4],[list4,list2]]:
##    print cond_prob2(list1,i,j)
##
##print "-----3----"
##
##print cond_prob3(list1,list2,list3,list4)
##

#print ('\nBNlogLikelihood = %.3f'%(float(cond_prob3(list1,list2,list3,list4))+log_pdf_2+log_pdf_3+log_pdf_4))
print ('BNlogLikelihood = %.3f'%(float(log_pdf_1+cond_prob1(list2,list1)+cond_prob1(list3,list4)+cond_prob1(list4,list1))))
print ("%.3f %.3f %.3f %.3f"%(float(log_pdf_1),float(cond_prob1(list2,list1)),float(cond_prob1(list3,list4)),float(cond_prob1(list4,list1))))

#print log_pdf_1,log_pdf_2,log_pdf_3,log_pdf_4

## GRAPH

##z=[list1,list2,list3,list4]
###print z
##z1=list(zip(*z))
##df = pd.DataFrame(z1, columns=['CS Score (USNews)', 'Research Overhead %', 'Admin Base Pay$', 'Tuition(out-state)$'])
##plt.figure(1)
##plt.subplot(331)
##plt.plot(list1,list2)
##plt.xlabel('CS Score')
##plt.ylabel('Research Overhead')
##
##plt.subplot(334)
##plt.plot(list1,list3)
##plt.xlabel('CS Score')
##plt.ylabel('Admin Base Pay $')
##
##plt.subplot(337)
##plt.plot(list1,list4)
##plt.xlabel('CS Score')
##plt.ylabel('Tuition $')
##
##plt.subplot(333)
##plt.plot(list2,list3)
##plt.xlabel('Research Overhead')
##plt.ylabel('Admin Base pay $')
##
##plt.subplot(336)
##plt.plot(list2,list4)
##plt.xlabel('Research Overhead')
##plt.ylabel('Tuition')
##
##plt.subplot(339)
##plt.plot(np.asarray(list4)/1000,list3)
##plt.xlabel('Tuition in 1000s')
##plt.ylabel('Admin Base Pay $')
##
##scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='hist')
##
##plt.show()
##


















