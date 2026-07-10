"""Algorithm 1 The main SmoteR algorithm.
function SmoteR(D, tE, o, u, k)
// D - A data set
// tE - The threshold for relevance of the target variable values
// %o,%u - Percentages of over- and under-sampling
// k - The number of neighbours used in case generation
rareL ← {hx, yi ∈ D : φ(y) > tE ∧ y < y˜} // ˜y is the median of the target Y
newCasesL ← genSynthCases(rareL, %o, k) // generate synthetic cases for rareL
rareH ← {hx, yi ∈ D : φ(y) > tE ∧ y > y˜}
newCasesH ← genSynthCases(rareH, %o, k) // generate synthetic cases for rareH
newCases ← newCasesL S
newCasesH
nrNorm ←%u of |newCases|
normCases ←sample of nrNorm cases ∈ D\{rareL S
rareH} // under-sampling
return newCases S
normCases
end function"""
import pandas as pd
import numpy as np

def phi_func(y):
    return y
def check_rare(y,y_median,threshold):
    phi_y = phi_func(y)

    if (phi_y > threshold) and y < y_median:
        return True
    else:
        return False
    
def genSynthCases(rare_sample, percentage_over_sampling, num_neighbours):
    pass

def SmoteR(D: pd.DataFrame, tE: float, o: float, u: float, k: int):
    y_median = y.median()
    all_new_cases = []

    for x,y in D:
        if check_rare(y,y_median,tE):
            new_cases = genSynthCases(y, percentage_over_sampling = o, num_neighbours= k)
            all_new_cases.append(new_cases)
