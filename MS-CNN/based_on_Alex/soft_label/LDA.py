#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 23:28:11 2019

@author: d
"""

import numpy as np
import scipy.special as ss

mid_corups=np.load('mid_corups.npy')
hig_corups=np.load('hig_corups.npy')

Nd,M,V=mid_corups.shape[1],mid_corups.shape[0],800
K=20# for the AID, K=10 
alpha=1

beta=np.zeros((K,V))# parameters for the high-level word
pi=np.zeros((K,V))#parameters for the middle-level word,correspond the 'eta' in the paper 

r=np.random.rand()
for k in range(K):
    for v in range(V):
        beta[k,v]=1/V*(1+0.1*np.random.rand())
        pi[k,v]=1/V*(1+0.1*np.random.rand())
        
    beta[k,:]=beta[k,:]/np.sum(beta[k,:])
    pi[k,:]=pi[k,:]/np.sum(pi[k,:])


    
est_iter=0
est_max=30
theta=np.zeros((M,K))

all_gamma=np.zeros((M,K))
while est_iter<est_max :
    est_iter+=1
    ss_topic_word_beta=np.zeros((K,V))
    ss_topic_beta=np.zeros((K))
    
    ss_topic_word_pi=np.zeros((K,V))
    ss_topic_pi=np.zeros((K))
    
    for m in range(M):
        mid_word=mid_corups[m,:]
        hig_word=hig_corups[m,:]
        
        gamma0=alpha+Nd/K
        dig0=ss.psi(gamma0)
        gamma=np.ones((K))*gamma0
        dig=np.ones((K))*dig0
        inf_iter=0
        inf_not_converged=True
        
        phi0=np.ones((1,K))*1/K
        phi=np.ones((1,K))*1/K
        for i in range(Nd-1):
            phi=np.vstack((phi,phi0))
        
        while inf_not_converged:
            inf_iter+=1
            gamma_old=gamma.copy()            
            ########compute phi###########
            for n in range(Nd):               
                phisum=0.0
                for k in range(K):
                    phi[n,k]=beta[k,int(hig_word[n])] \
                            *pi[k,int(mid_word[n])] \
                            *np.exp(dig[k])
                    phisum+=phi[n,k]
                phi[n,:]=phi[n,:]/phisum
                
            ########compute gamma############
            for k in range(K):
                gamma[k]=alpha
                for n in range(Nd):
                    gamma[k]+=phi[n,k]
                dig[k]=ss.psi(gamma[k])
            
            #########compute converge###########
                
            c_gamma=np.linalg.norm(gamma-gamma_old)/np.linalg.norm(gamma_old)
            
            if c_gamma<5e-3:
                inf_not_converged=False
                all_gamma[m,:]=gamma
                
        if m%100==0:
            print('in iter image {}'.format(m))
            
        for k in range(K):
            for n in range(Nd):                
                ss_topic_word_beta[k,int(hig_word[n])]+=phi[n,k]
                ss_topic_beta[k]+=phi[n,k]
                
                ss_topic_word_pi[k,int(mid_word[n])]+=phi[n,k]
                ss_topic_pi[k]+=phi[n,k]
    
    for k in range(K):
        beta[k,:]=ss_topic_word_beta[k,:]/ss_topic_beta[k]
        pi[k,:]=ss_topic_word_pi[k,:]/ss_topic_pi[k]
    
    print('out iter {}'.format(est_iter))

for m in range(M):
    all_gamma[m,:]=all_gamma[m,:]/np.sum(all_gamma[m,:]);

np.save('sample_feature.npy',all_gamma)
