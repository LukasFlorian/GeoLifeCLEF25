# Excerpt

**Paper:**  
"Combining Present-Only and Present-Absent Data with
Pseudo-Label Generation for Species Distribution
Modeling - Notebook for the LifeCLEF Lab at CLEF 2024"

**Authors:**  
Yi-Chia Chen, Tai Peng, Wei-Hua Li and Chu-Song Chen

## Challenge conditions:

### Objective
- Predict species occurrence in a given area $\rightarrow$ Species Distribution Modeling (SDM)
- Around 10k different species

### Data
- Training data split into PA (Presence-Absence) and PO (Presence-only) data
- 

**PO-Data:**
- Label 1 indicates species occurrence
- Label 0 does not imply absence
- 5 Million PO data points
- 

**PA-Data:**
- Label 1 indicates occurrence
- Label 0 indicates absence
- 90k PA data points


50% -> 4.0
100% -> 1.0
Linear interpolation

$Grade = RoundUp\left( \begin{cases}
    1.0 + \frac{3}{50}\cdot (100-Percentage) & if\ Percentage \geq 50 \\
    4.0 + \frac{1}{15}\cdot (50 - Percentage) & if\ 35 < Percentage < 50 \\
    5.0 & else
\end{cases}\right)$