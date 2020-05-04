import numpy as np
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.stats
from scipy.stats import norm
import json
import math
import matplotlib.pyplot as plt
import nashpy as nash

class epidemic:
    def __init__(self, base_rate, recover_rate, init_s, init_i, max_std):
        self.base_rate = base_rate
        self.recover_rate = recover_rate
        self.init_s = init_s
        self.init_i = init_i
        self.max_std = max_std

    # measure the effect caused by keeping a specific social distancing
    def effect_sd(self, sd, mean, std):
        if sd <= 0:
            return norm.pdf((sd-mean)/std)
        elif (sd < 10) and (sd > 0):
            return norm.pdf((sd-mean)/std)/(1+sd**2)
        else:
            return norm.pdf((sd-mean)/std)/(1+10**2)

    # the real infection rate under the effect of social distancing
    # base_rate is the infection rate when there is no social distancing 
    def infection_rate(self, mean, std):
        discount_factor,_ = integrate.quad(self.effect_sd, mean-3*std, 
                                           mean+3*std, args = (mean, std))
        return discount_factor*self.base_rate

    def SIR(self, mean, std):
        real_infection_rate = self.infection_rate(mean,std)
        s_record = list()
        s_record.append(self.init_s)
        i_record = list()
        i_record.append(self.init_i)
        r_record = list()
        r_record.append(0)
        while (i_record[-1] > 0.1*self.init_i) and (len(i_record)<365): # not all recovered
            updated_s = s_record[-1] - real_infection_rate*s_record[-1]*i_record[-1]
            updated_i = i_record[-1] + real_infection_rate*s_record[-1]*i_record[-1] - self.recover_rate*i_record[-1]
            updated_r = r_record[-1] + self.recover_rate*i_record[-1]
            s_record.append(updated_s)
            i_record.append(updated_i)
            r_record.append(updated_r)
        return s_record,i_record,r_record

    def utility_public_fcn(self, sd, mean, std):
        u1 = ((10/(sd**2)-(1/sd)**4))**0.25*norm.pdf((sd-mean)/std)
        return u1
    
    def utility_public(self, T, mean, std, i_record):
        u = 0
        for t in range(T):
            u2,_ = integrate.quad(self.utility_public_fcn, 1, 10, args = (mean, std))
            u2 -= 0.05*std**2+1.826*i_record[t]
            u += u2
        return u/T

    def utility_gov(self, T, mean, i_record):
        total_rate = 1
        for t in range(T):
            u = (100-1.8-90*i_record[t]+27.66/((2.8+mean)**2))/100
            total_rate *= u
        avg_rate = math.pow(total_rate,1/T)
        return (avg_rate-1)*100

    # gov is player 1, public is player 2
    def get_utility_pair_sir(self, mean, std):
        s_record,i_record,r_record = self.SIR(mean, std)
        utils_pub = self.utility_public(len(i_record), mean, std, i_record)
        utils_gov = self.utility_gov(len(i_record), mean, i_record)
        return round(utils_gov,3), round(utils_pub,3), s_record, i_record, r_record

def str_generator(L):
    string = ""
    for l in L:
        pair = str(l[0]) + "," + str(l[1]) + " "
        string += pair
    return string[:-1]

base_rate = 0.143
recover_rate = 0.33
init_s = 1
init_i = 0.00001
max_std = 5

game = epidemic(base_rate, recover_rate, init_s, init_i, max_std)
# s,i,r = game.SIR(2,3)
# plt.plot(i)
# plt.plot(s)
# plt.plot(r)
# plt.show()
payoff = list()
player1_payoff = list()
player2_payoff = list()
matrix_file = "weak.txt"
sir_file = "weak.json"
sir_record = dict() # {(mean,std):{"S":[s_record],"I":[i_record],"R":[r_record]}}
with open(matrix_file, "a") as f:
    for mean in range(1,11):
        result = list()
        player1 = list()
        player2 = list()
        for std in range(1,max_std+1):
            utils_gov, utils_pub, s_record, i_record, r_record = game.get_utility_pair_sir(mean, std)
            sir_record[str(mean)+"_"+str(std)] = {"S":s_record,"I":i_record,"R":r_record}
            result.append([utils_gov, utils_pub])
            player1.append(utils_gov)
            player2.append(utils_pub)
            print((mean,std))
            print(len(i_record))
        payoff.append(result)
        player1_payoff.append(player1)
        player2_payoff.append(player2)
        string = str_generator(result)
        f.write(string)
        f.write("\n")
    payoff = np.array(payoff)
    player1_payoff = np.array(player1_payoff)
    player2_payoff = np.array(player2_payoff)
    print(payoff)
f.close()

with open(sir_file,"w") as f:
    json.dump(sir_record, f)
    

create_game = nash.Game(player1_payoff, player2_payoff)
eqs = create_game.support_enumeration()
print(list(eqs))

