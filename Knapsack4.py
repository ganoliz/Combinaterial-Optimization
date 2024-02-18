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


class TabuProblem():
    def __init__(self, knapsack, iter_limit=1000, tabu_size=100, ):
        self.knapsack_cost = knapsack['cost']
        self.knapsack_weight = knapsack['weight']
        self.knapsack_capacity = knapsack['capacity']
        self.curr_state = self.Init_State(self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity)
        self.op_cost = self.curr_state.maxcost  # best solution ever
        self.local_max = [self.curr_state.maxcost]
        self.op_state = self.curr_state
        self.tabu_list = []
        self.tabu_size = tabu_size
        self.iter_limit = iter_limit
        self.iteration = 0
        self.visit = 0
        self.view = 0
        self.aspiration=[0 for i in range(len(knapsack['cost']))]
        Write_Log(f'cost : {self.knapsack_cost}, weight : {self.knapsack_weight}, capacity : {self.knapsack_capacity} \n')


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
            # total_weight = 0
            # total_cost = 0
            # for i in range(size):
            #     valid = True
            #     if knapsack_list[i] == 1:
            #         total_weight = total_weight + weight[i]
            #         total_cost = total_cost + cost[i]
            #         if total_weight > capacity:
            #             valid = False
            #             break
        Write_Log(f'Init_state = {init_state.idx} \n')
        return init_state

    def TabuSearch(self):

        while (self.iteration < self.iter_limit):
            self.iteration = self.iteration + 1

            self.aspiration = [i+1 for i in self.aspiration]
            best_fit = -1
            new_state_list = []
            new_state_tmp = self.LocalOptimalCandidate(self.curr_state, self.knapsack_cost, self.knapsack_weight,
                                                       self.knapsack_capacity)

            new_state_list = sorted(new_state_tmp, key=lambda x: x.maxcost, reverse=True) if len(new_state_tmp) > 1 else new_state_tmp # new_state_tmp.sort( key=lambda x: x.maxcost, reverse=True) error!
            # Write_Log(f'new_state_list = {new_state_list} \n')
            for state in new_state_list:

                if state.maxcost > self.op_cost and self.TabuCheck(self.tabu_list, state, self.curr_state) == True: # override tabu if it is best ever

                    best_fit = state.maxcost
                    #self.tabu_list = self.Update_Tabu(self.tabu_list, state, self.curr_state)
                    flip = np.asarray(self.curr_state.idx) ^ np.asarray(state.idx)
                    flip = flip.tolist()
                    idx_ = self.flipidx(flip)
                    self.aspiration[idx_] = 0

                    tabu_state = self.tabu_list.remove(flip)
                    self.tabu_list.append(tabu_state)
                    self.curr_state = state
                    Write_Log(f'In{self.iteration} run :tabu override to {self.curr_state.idx} \n')
                    break

                elif self.TabuCheck(self.tabu_list, state, self.curr_state) == False:
                    best_fit = state.maxcost
                    self.tabu_list = self.Update_Tabu(self.tabu_list, state, self.curr_state)

                    flip = np.asarray(self.curr_state.idx) ^ np.asarray(state.idx)
                    flip = flip.tolist()
                    idx_ = self.flipidx(flip)
                    self.aspiration[idx_] = 0

                    self.curr_state = state
                    Write_Log(f'In{self.iteration} run : move to {self.curr_state.idx} \n')
                    break


            if best_fit == -1:


                tmp_list = sorted(self.aspiration, reverse=True)
                for max_value in tmp_list:
                    idx_ = self.aspiration.index(max_value)
                    next_state_idx = np.asarray(self.curr_state.idx)
                    next_state_idx[idx_] = next_state_idx[idx_] ^ 1
                    next_state_idx = next_state_idx.tolist()
                    state = State(self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity, next_state_idx)
                    if state.valid == True:
                        break
                if state.valid == True:
                    self.aspiration[idx_] = 0
                    self.tabu_list = self.Update_Tabu(self.tabu_list, state, self.curr_state)
                    self.curr_state = state
                else:
                    return self.op_state
                # tabu_state = self.tabu_list.pop()
                # state = State(self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity, tabu_state)
                #
                # if state.valid == True:
                #     self.curr_state = state
                #     self.tabu_list = self.Update_Tabu(self.tabu_list, state, self.curr_state)


                #break

            if (self.curr_state.maxcost >= self.op_cost):
                self.op_cost = self.curr_state.maxcost
                self.op_state = copy.deepcopy(self.curr_state)

            if self.view == 1:
                self.local_max.append(self.op_cost)
            else:
                self.local_max.append(self.curr_state.maxcost)

        return self.op_state

    def LocalOptimalCandidate(self, curr_state, knapsack_cost, knapsack_weight, knapsack_capacity):
        new_state_tmp = []

        for i in range(len(knapsack_cost)):
            new_knapsack = Flip(i, copy.deepcopy(curr_state.idx))
            new_state = State(knapsack_cost, knapsack_weight, knapsack_capacity, new_knapsack)
            if new_state.valid == True:
                new_state_tmp.append(new_state)
                self.visit = self.visit + 1


        return new_state_tmp

    def Update_Tabu(self, tabu_list, new_state, curr_state):

        tabu_state = []
        num = len(new_state.idx)

        for j in range(num):
            flip = (new_state.idx[j]) ^ (curr_state.idx[j])
            if flip == 1:
                tabu_state.append(1)
            else:
                tabu_state.append(0)

        tabu_list.append(tabu_state)

        while (len(tabu_list) > self.tabu_size):
            tabu_list.pop()

        return tabu_list

    def flipidx(self, flip):

        idx = -1
        for i in range(len(flip)):
            if flip[i] == 1:
                idx = i
                break

        return idx

    def TabuCheck(self, tabu_list, state, current_state):

        a = state.idx
        b = current_state.idx

        flip = np.asarray(a) ^ np.asarray(b)
        flip = flip.tolist()
        if tabu_list.count(flip) == 0:
            return False
        else:
            return True


        # num = len(state.cost)
        #
        # for i in range(len(tabu_list)):
        #     for j in range(num):
        #         flip = (state.idx[j]) ^ (current_state.idx[j])
        #         if flip == 1 and flip == tabu_list[i][j]:
        #             Write_Log(f'tabu_list = {tabu_list} \n')
        #             return True

        # return False

instance_path = 'C:/Users/user/Downloads/ZR/'
item_num_total = [20]

tabu_size_list = [1, 5, 25, 35, 45]

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

        iter_limit = 300
        error_element = []
        for tabu_size in tabu_size_list:

            problem = TabuProblem(knapsack_, iter_limit=iter_limit, tabu_size=tabu_size)
            thestate = problem.TabuSearch()
            result = thestate.maxcost



            Write_Log(f'error = {(answer-result) / answer} \n \n \n')
            x_label = [i for i in range(len(problem.local_max))]

            plt.plot(x_label, problem.local_max,label=f'Tabu Search , size={tabu_size}')
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
