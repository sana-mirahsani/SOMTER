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

def find_control_points(y_train: pd.Series):

    Q1 = np.quantile(y_train, 0.25)
    Q2 = np.quantile(y_train, 0.50)  # median
    Q3 = np.quantile(y_train, 0.75)

    print("Q1:", Q1)
    print("Q2:", Q2)
    print("Q3:", Q3)

    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    print("lower_fence:",lower_fence)
    print("upper_fence:", upper_fence)

    return (Q1, Q2, Q3, lower_fence, upper_fence)

def SmoteR(D: pd.DataFrame, tE: float, o: float, u: float, k: int):
    pass