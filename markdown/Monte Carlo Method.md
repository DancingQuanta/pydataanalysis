```python
#Initial guess based on measured values.
def z0(Ra, Rb):
    return 2 * np.log(2) / (np.pi * (Ra + Rb))

def y_i(Ra, Rb, zi):
    test1 = np.pi * Ra * zi
    test2 = np.pi * Rb * zi
    return (np.exp(-test1)) + (np.exp(-test2))

#Current iteration of calculation
def z_i(Ra, Rb, zi1, yi):
    Coeff1 = (1.0 - yi) / np.pi

    test1 = np.pi * Ra * zi1
    test2 = np.pi * Rb * zi1
    test3 = Ra * np.exp(-test1)
    test4 = Rb * np.exp(-test2)
    return (zi1 - (Coeff1/(test3 + test4)))

#Solver and tolerance check.
def VDP_solver(_tol,_z0,Ra,Rb):
    zi1 = _z0
    counter = 0
    while True:
        yi = y_i(Ra,Rb, float(zi1))

        zi =  z_i(Ra,Rb,zi1,yi)
        if (zi-zi1)/zi < tol:
            return 1.0/zi
        else:
            zi1 = zi
        counter += 1


# Average and standard deviation of a list
def stats(lst):
    _mean = sum(lst) / len(lst)
    variance = sum([((x - _mean) ** 2) for x in lst]) / len(lst)
    _stdDev= variance ** 0.5

    return _mean, _stdDev


# Variable setup already done in the previous cell
Ra
Rb
#Ra=90.82e6
#Rb=90.23e6

# Tolerance
tol = 1.0e-4

# Monte Carlo simulation number
N = 100000

# Resistance uncertainty
Ra_perc = 5
Rb_perc = 5

# Filename formatting
filename= "MonteCarlo-SheetResRa_{:.2e}Rb_{:.2e}_StandardDev_a_{:.2f}_StandardDev_a_{:.2f}.txt"

#empty array to fill with Resistance values with some statistical fluctuations.
res_MC = []


# Run main funtion
for i in range(N):
    Ra_MC = Ra + (Ra_perc/100)*(Ra)*np.random.normal()
    Rb_MC = Rb + (Rb_perc/100)*(Rb)*np.random.normal()

    # Initial value
    val_initMC = z0(Ra_MC,Rb_MC)

    #---void main---#

    res_MC.append(VDP_solver(tol,val_initMC,Ra_MC,Rb_MC))


# Compute statistical variation in data
Stats = stats(res_MC)
Average = Stats[0]
SD      = Stats[1]
#print(res_MC)
#res_MC.ndim()
len(res_MC)

Rs = Average
```
