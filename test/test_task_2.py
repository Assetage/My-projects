# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:55:00 2021

@author: Asset Yespolov
"""
# below is the sample json string which is used as a input variable to the
# function cumulative_interest_paid.
# the mandatory parameters are loan amount, annual interest and number of years,
# while number of compounding periods per year is optional parameter with
# a default value of 1. 
sample_json_string = '{"Loan amount": 50000, "Annual interest": 0.03, \
                       "Number of years": 3, "Compounding periods per year": 2}'

# the cumulative interest paid function is defined below
def cumulative_interest_paid(strng):
    # due to the limitations on using 3rd party libraries, the json library
    # could not be used. Instead I suggest using eval method which reads
    # string as a dictionary.
    a = eval(strng)
    # here is the implementation of an optionality of compounding periods per
    # year parameter. If it is given, then the value from string is taken.
    # otherwise the default number of 1 is taken.
    try:        
        k = a["Compounding periods per year"]
    except:
        k = 1
    # declare the variables corresponding to the input parameters
    P = a["Loan amount"]
    i = a["Annual interest"]
    n = a["Number of years"]
    # calculate the cumulative interest paid
    interest_paid = P*(1+i/k)**(n*k) - P
    return interest_paid
# the use of the implemented function is described below
# as a result we get the cumulative interest paid on loan for the given
# parameters.      
result = cumulative_interest_paid(sample_json_string)