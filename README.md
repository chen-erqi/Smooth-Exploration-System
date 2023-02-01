# Smooth-Exploration-System
all the source code about the paper “Smooth Exploration System: an ease-of-use and specialized module for improving exploration of whale optimization algorithm”

Numba is used to accelerate the main loop in line 21 of SWOA.py. 
Experimental configuration is max_iter=300000, search_num=30, lb=-100, ub=100, dim=30.
benchamrk function is F=x1*x1+...+xn*xn.

With Numba, run time is 5s.
Pure python, run time is 1h.
