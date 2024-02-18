import copy

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import random


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

        if self.valid != True:
            cost = cost - max(self.cost)

        return cost


def Flip(idx, item):
    if item[idx] == 0:
        item[idx] = 1
    elif item[idx] == 1:
        item[idx] = 0

    return item


class GeneAlgorithmProblem():
    def __init__(self, knapsack, iter_limit=50, population = 50,elite = 3, ):
        self.knapsack_cost = knapsack['cost']  # problem
        self.knapsack_weight = knapsack['weight']
        self.knapsack_capacity = knapsack['capacity']



        self.tourament = 3 #algorithm parameters
        self.population = population
        self.selection_type = 0
        self.elite = elite
        self.mutation_rate = 0.01

        self.generation = self.Init_State(self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity, self.population)
        self.fit_value, self.fit_avg = self.CountFit(self.generation)

        # self.op_cost = self.curr_state.maxcost  # best solution ever
        # self.local_max = [self.curr_state.maxcost]
        # self.op_state = self.curr_state

        _the = max(self.generation, key=lambda a:a.maxcost)
        self.local_max = [_the.maxcost]
        self.op_state = _the
        self.iter_limit = iter_limit # loop, debug
        self.iteration = 0
        self.visit = 0
        self.view = 0


    def Init_State(self, cost, weight, capacity, population):
        size = len(cost)
        valid = False
        total_weight = 0
        total_cost = 0
        init_generation = []
        idx = 0
        for i in range(population):
            while(1):
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

            init_generation.append(copy.deepcopy(init_state))


        return init_generation

    def CountFit(self, generation):

        cost_avg = 0
        valid_state = 0
        for state in generation:
            # if state.valid == True:
            cost_avg = cost_avg + state.maxcost
            valid_state = valid_state + 1

        if valid_state == 0:
            valid_state = 1
        cost_avg = cost_avg / valid_state

        fit_list = []

        for state in generation:
            fit_list.append(state.maxcost / cost_avg)
            # if state.valid == True:
            #     fit_list.append(state.maxcost / cost_avg)
            # else:
            #     fit_list.append(-1)

        return fit_list, cost_avg



    def GA(self):

        while (self.iteration < self.iter_limit):
            self.iteration = self.iteration + 1

            fitness, cost_avg = self.CountFit(self.generation)
            selection, select_fit = self.Selection(self.generation, fitness, selection_type=self.selection_type)
            self.generation = self.Generate(selection, self.population)
            local_state = max(self.generation, key=lambda a: a.maxcost)
            # local_state = sorted(self.generation, key=lambda a: a.maxcost, reverse=True)
            # for state in local_state:
            #     if state.valid == True:
            self.local_max.append(local_state.maxcost)

            if local_state.maxcost > self.op_state.maxcost and local_state.valid == True:
                self.op_state = local_state
            self.visit = self.visit + self.population
        sortlist = sorted(self.generation, key= lambda a : a.maxcost, reverse=True)

        for state in sortlist:
            if state.valid == True:
                break

        return state


    # def Fitness(self, state_list):
    #     cost_avg = 0
    #     valid_state = 0
    #     for state in state_list:
    #         if state.valid == True:
    #             cost_avg = cost_avg + state.maxcost
    #             valid_state = valid_state + 1
    #
    #     if valid_state == 0:
    #         valid_state = 1
    #     cost_avg = cost_avg / valid_state
    #
    #     evaluation_list = []
    #
    #     for state in state_list:
    #         if state.valid == True:
    #             evaluation_list.append(state.maxcost / cost_avg)
    #         else:
    #             evaluation_list.append(-1)
    #
    #     return evaluation_list, cost_avg

    def Generate(self, state_list, population,):

        new_generation_idx = []
        new_generation = []
        for _ in range(population - self.elite):

            a = 0
            b = 0

            while (a == b):
                a = np.random.randint(0, len(state_list))
                b = np.random.randint(0, len(state_list))

            crossover_rate = random.random()
            if crossover_rate < 0.9:
                child = self.Crossover(state_list[a], state_list[b])

                mutation_rate = random.random()
                if mutation_rate < self.mutation_rate:
                    child = self.Mutation(child)

            else:
                child = copy.deepcopy(state_list[a].idx)

            new_generation_idx.append(child)

        the_list = sorted(state_list, key=lambda a:a.maxcost, reverse=True)

        for i in range(self.elite):
            new_generation_idx.append(copy.deepcopy(the_list[i].idx))

        for idx in new_generation_idx:
            state = State(self.knapsack_cost, self.knapsack_weight, self.knapsack_capacity, idx)
            new_generation.append(copy.deepcopy(state))


        return new_generation

    def Crossover(self, parent1_state, parent2_state):
        parent1 = parent1_state.idx
        parent2 = parent2_state.idx
        mash = np.random.randint(1, size=len(parent1))
        child = []
        for i in range(len(mash)):
            child.append(-1)
            if mash[i] == 0:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        return child

    def Mutation(self, state):

        state_idx = state
        flip = np.random.randint(0, len(state_idx))
        state_idx[flip] = state_idx[flip] ^ 1

        return state




    def Selection(self,state_list, fitness, selection_type = 0):

        if selection_type == 0:
            selection_fit = self.LinearScaling(fitness)

            sum_fit = sum(selection_fit)

            selection = []

            for i in range(self.population):
                a = random.uniform(0, sum_fit)
                idx = 0
                cal_percent = 0
                for percent in selection_fit:

                    cal_percent = cal_percent + percent
                    if a < cal_percent:
                        selection.append(copy.deepcopy(state_list[idx]))
                        break
                    idx = idx + 1

        elif selection_type == 1:
            selection_fit = self.Ranking(fitness)
        elif selection_type == 2:
            selection = self.Tournament(fitness)
        elif selection_type == 3:
            return -1


        return selection, selection_fit,

    def LinearScaling(self, idvl, f1 = 1, f2 = 2,):
        value = [i for i in idvl]
        mini = min(value)
        maxi = max(value)

        dif = maxi - mini if maxi-mini != 0 else 1
        fit_function = []
        for i in idvl:
            fit = f1 + (i - mini) * (f2 - f1) / (dif)
            fit_function.append(fit)

        return fit_function

    def Ranking(self,idvl, f1 = 10, ):
        f2 = f1 + len(idvl) - 1

        fit_function = []

        idvl = idvl.sort(reverse = False)
        for i in idvl:
            idx = idvl.index(i)
            fit_function.append(f1 + idx)

        return fit_function

    def Tournament(self, idvl, selection_number, tour_size = 5):

        selection = []
        for i in range(selection_number):
            candidate_idx = np.random.randint(0, len(idvl), size=tour_size)
            tmp = [idvl[int(cand)] for cand in candidate_idx]
            winner = max(tmp, key= lambda a: a.maxcost)
            selection.append(winner)

        return selection

    def Truncate(self, ):
        return -1

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

generation_size_list = [40, 50, 60]

loop_T = []
time_elapse_T = []
visit_node_brT = []
visit_node_gaT = []


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
    visit_node_ga = []
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
        for temp in generation_size_list:

            problem = GeneAlgorithmProblem(knapsack_, iter_limit=temp, )
            thestate = problem.GA()
            result = thestate.maxcost



            Write_Log(f'error = {(answer-result) / answer} \n \n \n')
            x_label = [i for i in range(len(problem.local_max))]

            plt.plot(x_label, problem.local_max,label=f'Generation Algorithm ,generation size={temp}')
            error_element.append((answer-result) / answer)
        the_answser = [answer for i in range(len(x_label))]
        # x_label = [i for i in range(iter_limit)]
        plt.plot(x_label, the_answser, color='k', linestyle='--', label='Optimum')
        plt.xlabel('Iteration')
        plt.ylabel('Knapsack Value')
        plt.title(f'Generation Algorithm Solution over ItemNum {item_num}: {index} ')
        plt.savefig(f'./plot/{index}.png')



        result_.append([answer, result])
        error.append(error_element)
        visit_node_ga.append(problem.visit)
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
    visit_node_gaT.append(visit_node_ga)
print('finish')
colormap = ['r', 'g', 'y', 'c','m']
for idx in range (len(item_num_total)):
    x_label = [np.log10(i) for i in visit_node_gaT[idx]]

    list_tmp = [error_T[idx][i][-1] for i in range(row)]
    plt.scatter(x_label ,list_tmp, s= 10)

    for idx_j in range(len(generation_size_list)):
        error_abs = [abs(error_T[idx][i][idx_j]) for i in range(row)]
        error = sum(error_abs) / len(error_abs)
        x_label_ = sum(x_label) / len(error_abs)
        plt.scatter(x_label_, error, color=colormap[idx_j], label=f'Generation size={generation_size_list[idx_j]} Avg.={error}')
    # plt.scatter(x_label_, error, color='r', label=f'Average')
    plt.legend()
    plt.xlabel('Visited State Number(log)')
    plt.ylabel=('Relative Error ε')
    plt.title('Time Complexity versus Relative Error')
    plt.show()
