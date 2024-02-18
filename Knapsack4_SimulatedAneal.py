import copy

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt



def Write_Log(txt):
    _ = 1
    # with open('./log.txt','a+') as f:
    #     f.write(txt)


class State():
    def __init__(self, item_cost, item_weight, capacity, knapsack_list):
        # self.attribute = attribute
        self.cost = item_cost
        self.weight = item_weight
        self.idx = knapsack_list
        self.capacity = capacity
        self.valid, self.totalweight = self.CheckValid(self.weight, self.idx)
        self.maxcost = self.MaxCost(self.cost, self.idx)


    def CheckValid(self, itemweight, knapsack_list):
        weight = 0
        for i in range(len(itemweight)):
            if knapsack_list[i] == 1:
                weight = weight + itemweight[i]

        valid = False
        if weight > self.capacity:
            valid = False
        else:
            valid = True

        return valid, weight

    def MaxCost(self, item_cost, knapsack_list):
        cost = 0
        for i in range(len(item_cost)):
            if knapsack_list[i] == 1:
                cost = cost + item_cost[i]

        return cost


def Flip(idx, item):
    if item[idx] == 0:
        item[idx] = 1
    elif item[idx] == 1:
        item[idx] = 0

    return item


class SimulatedAnealingProblem():
    def __init__(self, knapsack, iter_limit=50, temperature=100, ter_thres=0.1):
        self.knapsack_cost = knapsack['cost']  # problem
        self.knapsack_weight = knapsack['weight']
        self.knapsack_capacity = knapsack['capacity']
        self.curr_state = self.Init_State(self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity)
        self.op_cost = self.curr_state.maxcost  # best solution ever
        self.local_max = [self.curr_state.maxcost]
        self.op_state = self.curr_state

        self.temp = temperature #algorithm parameters
        self.ter_thres = ter_thres
        self.cool_coef = 0.995

        self.iter_limit = iter_limit # loop, debug
        self.iteration = 0
        self.visit = 0
        self.view = 0


    def Init_State(self, cost, weight, capacity):
        size = len(cost)
        valid = False
        total_weight = 0
        total_cost = 0
        knapsack_list = []
        idx = 0
        while (1):
            knapsack_list = []
            knapsack_list = np.random.randint(2, size=(size))
            init_state = State(cost, weight, capacity, knapsack_list)
            valid = init_state.valid

            if valid == True:
                break
            idx = idx + 1
            if (idx >= 100000):
                knapsack_list = np.zeros(size, dtype=np.int32)
                init_state = State(cost, weight, capacity, knapsack_list)
                break

        Write_Log(f'Init_state = {init_state.idx} \n')
        return init_state

    def SA(self):

        while (self.Frozen(self.temp) != True):

            while (self.iteration < self.iter_limit):
                self.iteration = self.iteration + 1

                new_state_list = self.LocalOptimalCandidate(self.curr_state, self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity)

                # new_state_list.sort(key = lambda a: a.maxcost, reverse=True)
                rnd = np.random.randint(0, len(new_state_list))
                new_state = new_state_list[rnd]

                delta_cost = self.curr_state.maxcost - new_state.maxcost

                if new_state.valid == True:
                    if delta_cost < 0:
                        self.curr_state = new_state
                    elif self.Accept(delta_cost, self.temp):
                        self.curr_state = new_state


                self.local_max.append(self.curr_state.maxcost)

            self.temp = self.Cold(self.temp)
            self.iteration = 0

        return self.op_state

    def Frozen(self, temp):
        if temp < 0.1:
            return True
        else:
            return False
    def Accept(self, cost, temp):
        exponential = np.exp(-cost / float(temp))
        rand = np.random.random()
        if rand < exponential:
            return True
        else:
            return False


    def Cold(self, temp):
        return temp * self.cool_coef

    def LocalOptimalCandidate(self, curr_state, knapsack_cost, knapsack_weight, knapsack_capacity):
        new_state_tmp = []

        for i in range(len(knapsack_cost)):
            new_knapsack = Flip(i, copy.deepcopy(curr_state.idx))
            new_state = State(knapsack_cost, knapsack_weight, knapsack_capacity, new_knapsack)
            if new_state.valid == True:
                new_state_tmp.append(new_state)
                self.visit = self.visit + 1


        return new_state_tmp



    def flipidx(self, flip):

        idx = -1
        for i in range(len(flip)):
            if flip[i] == 1:
                idx = i
                break

        return idx


instance_path = 'C:/Users/user/Downloads/ZR/'
item_num_total = [20]

tabu_size_list = [10, 15, 20]

loop_T = []
time_elapse_T = []
visit_node_brT = []
visit_node_tabuT = []


result_T = []

error_T = []
average_error = []
for item_num in item_num_total:
    instance = pd.read_csv(instance_path + f'ZR{item_num}_inst.dat', sep=' ', header=None)
    solution = pd.read_csv(instance_path + f'ZK{item_num}_sol.dat', sep=' ', header=None)

    row = 50

    index_from = 0
    time_elapse = []
    loop = []
    visit_node_br = []
    visit_node_tabu = []
    time_different = []
    result_ = []
    error = []
    for index in range(row):
        id = instance.loc[index + index_from][0]
        M = instance.loc[index + index_from][2]
        B_thres = instance.loc[index + index_from][3]  # lower_bound

        offset = 0
        answer_id = solution.loc[index + index_from][0]
        while(abs(id) > answer_id):
            offset = offset + 1
            answer_id = solution.loc[index + index_from + offset][0]
        answer = solution.loc[index + index_from + offset][2]


        item_weight = []
        item_cost = []
        knapsack_ = dict()
        for i in range(item_num):
            item_weight.append(instance.loc[index + index_from][4 + 2 * i])
            item_cost.append(instance.loc[index + index_from][4 + 2 * i + 1])
        knapsack_['capacity'] = M
        knapsack_['cost'] = item_cost
        knapsack_['weight'] = item_weight

        iter_limit = 50
        error_element = []
        for temp in tabu_size_list:

            problem = SimulatedAnealingProblem(knapsack_, iter_limit=iter_limit, temperature=temp)
            thestate = problem.SA()
            result = thestate.maxcost



            Write_Log(f'error = {(answer-result) / answer} \n \n \n')
            x_label = [i for i in range(len(problem.local_max))]

            plt.plot(x_label, problem.local_max,label=f'Tabu Search , size={temp}')
            error_element.append((answer-result) / answer)
        the_answser = [answer for i in range(iter_limit)]
        x_label = [i for i in range(iter_limit)]
        plt.plot(x_label, the_answser, color='k', linestyle='--', label='Optimum')
        plt.xlabel('Iteration')
        plt.ylabel('Knapsack Value')
        plt.title(f'Tabu Search Solution over ItemNum {item_num}: {index} ')
        plt.savefig(f'./plot/{index}.png')



        result_.append([answer, result])
        error.append(error_element)
        visit_node_tabu.append(problem.visit)
        if ((answer-result) / answer) < 0.0 :# index % 10 == 0:
            weight = problem.op_state.totalweight
            plt.legend()
            plt.show()
        plt.close()
        # if ((answer-result) / answer) < 0:
        #     weight = 0
        #     for i in range(len(item_weight)):
        #         if thestate.idx[i] == 1:
        #             weight = weight+item_weight[i]
        #     print(f'capacity = {M} but weight = {weight} score = {answer} and cost={thestate.maxcost}')

    result_T.append(result_)
    error_T.append(error)
    visit_node_tabuT.append(visit_node_tabu)
print('finish')
colormap = ['r', 'g', 'y', 'c','m']
for idx in range (len(item_num_total)):
    x_label = [ np.log10(i) for i in visit_node_tabuT[idx]]

    list_tmp = [error_T[idx][i][-1] for i in range(row)]
    plt.scatter(x_label ,list_tmp, s= 10)

    for idx_j in range(len(tabu_size_list)):
        error_abs = [abs(error_T[idx][i][idx_j]) for i in range(row)]
        error = sum(error_abs) / len(error_abs)
        x_label_ = sum(x_label) / len(error_abs)
        plt.scatter(x_label_, error, color=colormap[idx_j], label=f'tabulist size={tabu_size_list[idx_j]} Avg.={error}')
    # plt.scatter(x_label_, error, color='r', label=f'Average')
    plt.legend()
    plt.xlabel('Visited State Number(log)')
    plt.ylabel=('Relative Error ε')
    plt.title('Time Complexity versus Relative Error')
    plt.show()
