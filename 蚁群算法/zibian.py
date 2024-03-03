import sys
import numpy as np
import random
import math
import copy
import pandas as pd

#sys.stdin=open('input.txt',"r")
sys.stdout = open("output.txt", "w")
customer_number = 35                                  #客户数量
period = 5                                            #周期数量
vehicle_capacity = 600                               #车辆载货能力
vehicle_no = 1                                        #车辆数量
cij = 10                                              #ij之间单位距离车辆运输费用
bij = 1                                               #ij之间单位转运费用
f = 5                                                 #单位货物缺货成本
h = 1                                                 #单位库存持有成本
# 仓库和顾客的初始库存数量如何定义，每个周期仓库获得一定数量的ai如何定义？

tau_0=0                                             #初始信息素浓度，假设为0
Q=1                                                 #信息素增量，假设为1
m=25                                                #蚂蚁的数量
rho=0.1                                             #信息素挥发系数，假设为0.1
alpha:int=1                                         #信息素启发因子
beta:int=5                                          #路径启发因子
gen_number = 200                                    #最大迭代次数

warehouse = {"x":114.291383,"y":30.574599,"dt":0,"L":1000,"h":0}     #定义仓库的初始参数，坐标和初始库存
customer_list = []
# 读取客户数据
def read_customer_data(filename):
    df = pd.read_excel(filename)
    customer_list = []
    for index, row in df.iterrows():
        customer = {
            "x": row["x"],
            "y": row["y"],
            "h": row["h"],
            "dt": row["dt"],
            "L": row["L"],
            "k": row["k"],
            "a": row["a"],
            "Lt": 0, # 期末库存初始设置为0
            "AL": 0  # 初始平均库存设置为0
        }
        customer_list.append((index + 1, customer))
    return customer_list
def run_simulation(period,customer_list):
    if period != 1:  # From the second period onwards, only update the demand 'dt'
        for i, customer in customer_list:
            customer["dt"] = int(np.random.normal(40, 4))  # Random demand for new period

    for index, (i, customer) in enumerate(customer_list[1:]):  # 解包顾客信息和序号
        print(f'{i} : {customer}')


    replen_list1, replen_list2, replen_list3 = [], [], []
    for index, (i, customer) in enumerate(customer_list, start=1):
        diff = customer["L"] - customer["dt"]
        if diff <= 0:
            replen_list1.append((i, customer))
        elif diff < customer["a"]:
            replen_list2.append((i, customer))
        else:
            replen_list3.append((i, customer))
    print("replen_list1")

    for index, (i, customer) in enumerate(replen_list1):
        print(f'{i} : {customer}')

    print("replen_list2")
    for index, (i, customer) in enumerate(replen_list2):
        print(f'{i} : {customer}')

    print("replen_list3")
    for index, (i, customer) in enumerate(replen_list3):
        print(f'{i} : {customer}')

    #拆分
    replen_list1_1, replen_list1_2, replen_list1_3, transin_list1 = [], [], [], []
    if replen_list1:
        total_sum = sum([customer["dt"] - customer["L"] + customer["a"] for _, customer in replen_list1])
        if total_sum == vehicle_capacity:
            replen_list1_1 = copy.deepcopy(replen_list1)
        elif total_sum < vehicle_capacity:
            replen_list1_2 = copy.deepcopy(replen_list1)
        else:
            replen_list1_sorted = sorted(replen_list1, key=lambda x: x[1]["dt"] - x[1]["L"] + x[1]["a"], reverse=True)
            current_sum = 0
            for index, (i, customer) in enumerate(replen_list1_sorted):
                if current_sum + customer["dt"] - customer["L"] + customer["a"] <= vehicle_capacity:
                    current_sum += customer["dt"] - customer["L"] + customer["a"]
                    replen_list1_3.append((i, copy.deepcopy(customer)))
                else:
                    transin_list1.append((i, copy.deepcopy(customer)))
            print("replen_list1_3")
            for index, (i, customer) in enumerate(replen_list1_3):
                print(f'{i} : {customer}')
            print("transin_list1")
            for index, (i, customer) in enumerate(transin_list1):
                print(f'{i} : {customer}')
    print("replen_list1_1")
    for index, (i, customer) in enumerate(replen_list1_1):
        print(f'{i} : {customer}')
    print("replen_list1_2")
    for index, (i, customer) in enumerate(replen_list1_2):
        print(f'{i} : {customer}')

    #继续拆分
    replen_list2_1, replen_list2_2, transin_list2 = [], [], []
    if replen_list1_2 and replen_list2:
        total_sum = sum([customer["a"] - customer["L"] + customer["dt"] for _, customer in replen_list2])
        remain_capacity = vehicle_capacity - sum([customer["dt"] - customer["L"] + customer["a"] for _, customer in replen_list1_2])
        if remain_capacity >= total_sum:
            replen_list2_1 = copy.deepcopy(replen_list1_2 + replen_list2)
        else:
            replen_list2_sorted = sorted(replen_list2, key=lambda x: x[1]["a"] - x[1]["L"] + x[1]["dt"], reverse=True)
            current_sum = 0
            for index, (i, customer) in enumerate(replen_list2_sorted):
                if current_sum + customer["a"] - customer["L"] + customer["dt"] <= remain_capacity:
                    current_sum += customer["a"] - customer["L"] + customer["dt"]
                    replen_list2_2.append((i, copy.deepcopy(customer)))
                else:
                    transin_list2.append((i, copy.deepcopy(customer)))
            replen_list2_2 = copy.deepcopy(replen_list2_2 + replen_list1_2)
        print("replen_list2_1")
        for index, (i, customer) in enumerate(replen_list2_1):
            print(f'{i} : {customer}')
        print("replen_list2_2")
        for index, (i, customer) in enumerate(replen_list2_2):
            print(f'{i} : {customer}')
        print("transin_list2")
        for index, (i, customer) in enumerate(transin_list2):
            print(f'{i} : {customer}')


    #计算运输量以及更新期末库存

    relenish = 0
    inventory_cost1, inventory_cost2, inventory_cost3, inventory_cost4, inventory_cost5, \
    inventory_cost6, inventory_cost7, inventory_cost8, inventory_cost9, inventory_cost10, \
    inventory_cost11, inventory_cost12, inventory_cost13, inventory_cost14, inventory_cost15, \
    inventory_cost16, inventory_cost17, inventory_cost18, inventory_cost19, inventory_cost20, \
    inventory_cost21, inventory_cost22, inventory_cost23 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    if replen_list1_1:
        for index, (i, customer) in enumerate(replen_list1_1):
            customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
            relenish += customer["dt"] - customer["L"] + customer["a"]
            customer["AL"] = (customer["Lt"] + customer["L"])/2
            inventory_cost1 += customer["AL"] * customer["h"]
    if replen_list1_3:
        for index, (i, customer) in enumerate(replen_list1_3):
            customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
            relenish += customer["dt"] - customer["L"] + customer["a"]
            customer["AL"] = (customer["Lt"] + customer["L"]) / 2
            inventory_cost2 += customer["AL"] * customer["h"]
    if replen_list2_1:
        for index, (i, customer) in enumerate(replen_list2_1):
            relenish_1 = 0
            relenish_2 = 0
            if customer in replen_list1:
                customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
                relenish_1 = customer["dt"] - customer["L"] + customer["a"]
                customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                inventory_cost3 += customer["AL"] * customer["h"]
            else:
                customer["Lt"] = customer["L"] + (customer["a"] - customer["L"] + customer["dt"]) - customer["dt"]
                relenish_2 = customer["a"] - customer["L"] + customer["dt"]
                customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                inventory_cost4 += customer["AL"] * customer["h"]
            relenish += relenish_1 + relenish_2

    if replen_list2_2:
        for index, (i, customer) in enumerate(replen_list2_2):
            relenish_1 = 0
            relenish_2 = 0
            if customer in replen_list1_2:
                customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
                relenish_1 = customer["dt"] - customer["L"] + customer["a"]
                customer["AL"] = (customer["Lt"] + customer["L"]) / 2
            else:
                customer["Lt"] = customer["L"] + (customer["a"] - customer["L"] + customer["dt"]) - customer["dt"]
                relenish_2 = customer["a"] - customer["L"] + customer["dt"]
                customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                inventory_cost5 += customer["AL"] * customer["h"]
        relenish += relenish_1 + relenish_2

    ####计算中转量
    transdone_list1, transdone_list2, transdone_list3, transdone_list4 = [], [], [], []
    stockout_list1, stockout_list2, stockout_list3, stockout_list4 = [], [], [], []
    stock_num = 0
    trans_replenish = 0
    if replen_list3:
        if transin_list1:
            sum1 = sum([customer["L"] - customer["dt"] - customer["a"] for _, customer in replen_list3])
            sum2 = sum([customer["dt"] - customer["L"] + customer["a"] for _, customer in transin_list1])
            difference = sum1 - sum2

            if difference == 0:
                for index, (i, customer) in enumerate(replen_list3):
                    customer["Lt"] = customer["L"] - (customer["L"] - customer["dt"] - customer["a"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost8 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(transin_list1):
                    customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost9 = customer["AL"] * customer["h"]
                trans_replenish = sum1

            # 计算顾客需要的中转量
            if difference < 0:
                transin_list1_sorted = sorted(transin_list1, key=lambda x: x[1]["dt"] - x[1]["L"] + x[1]["a"], reverse=True)
                current_sum = 0
                for index, customer in transin_list1_sorted:
                    if current_sum + customer["dt"] - customer["L"] + customer["a"] <= sum1:
                        current_sum += customer["dt"] - customer["L"] + customer["a"]
                        transdone_list1.append(copy.deepcopy(customer))
                    else:
                        stockout_list1.append(copy.deepcopy(customer))

                # 计算tansdone_list1中(dt-L+a)总量作为余量的初始量
                total_remain = sum([customer["dt"] - customer["L"] + customer["a"] for _, customer in transdone_list1])
                # 将replen_list3的顾客按照(L-dt-a)降序排列
                replen_list3_sorted = sorted(replen_list3, key=lambda x: x[1]["L"] - x[1]["dt"] - x[1]["a"], reverse=True)
                for index, (i, customer) in enumerate(replen_list3_sorted):
                    if customer["L"] - customer["dt"] - customer["a"] <= total_remain:
                        # 如果(L-dt-a)小于等于余量，则更新顾客的库存Lt和余量
                        customer["Lt"] = customer["L"] - (customer["L"] - customer["dt"] - customer["a"]) - customer["dt"]
                        total_remain -= customer["L"] - customer["dt"] - customer["a"]
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2

                    else:
                        # 如果(L-dt-a)大于余量，则更新顾客的库存Lt，并减去余量
                        customer["Lt"] = customer["L"] - total_remain - customer["dt"]
                        total_remain = 0
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost10 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(transdone_list1):
                    customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost11 = customer["AL"] * customer["h"]
                for index, (i, customer) in enumerate(stockout_list1):
                    customer["Lt"] = customer["L"] - customer["dt"]
                    if customer["Lt"] < 0:
                        customer["Lt"] = 0
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost12 = customer["AL"] * customer["h"]
                    stock_num += abs(customer["L"] - customer["dt"])
                trans_replenish = sum([customer["dt"] - customer["L"] + customer["a"] for _, customer in transin_list1])

            if difference > 0:
                current_sum = 0
                replen_list3_sorted = sorted(replen_list3, key=lambda x: x[1]["L"] - x[1]["dt"] - x[1]["a"],reverse=True)
                for index, customer in replen_list3_sorted:
                    if current_sum + customer["L"] - customer["dt"] - customer["a"] <= sum2:
                        current_sum += customer["L"] - customer["dt"] - customer["a"]
                        transdone_list2.append(copy.deepcopy(customer))
                    else:
                        stockout_list2.append(copy.deepcopy(customer))

                # 计算tansdone_list2中(L-dt-a)总量作为余量的初始量
                total_remain = sum([customer["L"] - customer["dt"] - customer["a"] for _, customer in transdone_list2])
                transin_list1_sorted = sorted(transin_list1, key=lambda x: x[1]["dt"] - x[1]["L"] + x[1]["a"],
                                              reverse=True)
                for index, (i, customer) in enumerate(transin_list1_sorted):
                    if customer["dt"] - customer["L"] + customer["a"] <= total_remain:
                        # 如果(dt-L+a)小于等于余量，则更新顾客的库存Lt和余量
                        customer["Lt"] = customer["L"] + (customer["dt"] - customer["L"] + customer["a"]) - customer["dt"]
                        total_remain -= customer["dt"] - customer["L"] + customer["a"]
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    else:
                        # 如果(dt-L+a)大于余量，则更新顾客的库存Lt，并减去余量
                        customer["Lt"] = customer["L"] + total_remain - customer["dt"]
                        total_remain = 0
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost13 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(transdone_list2):
                    customer["Lt"] = customer["L"] - (customer["L"] - customer["dt"] - customer["a"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost14 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(stockout_list2):
                    customer["Lt"] = customer["L"] - customer["dt"]
                    if customer["Lt"] < 0:
                        customer["Lt"] = 0
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost15 = customer["AL"] * customer["h"]
                    trans_replenish = sum([customer["L"] - customer["dt"] - customer["a"] for _, customer in transdone_list2])

        if transin_list2:
            sum1 = sum([customer["L"] - customer["dt"] - customer["a"] for _, customer in replen_list3])
            sum3 = sum([customer["a"] - customer["L"] + customer["dt"] for _, customer in transin_list2])
            difference1 = sum1 - sum3
            if difference1 == 0 :
                for index, (i, customer) in enumerate(replen_list3):
                    customer["Lt"] = customer["L"] - (customer["L"] - customer["dt"] - customer["a"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost16 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(transin_list2):
                    customer["Lt"] = customer["L"] + (customer["a"] - customer["L"] + customer["dt"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost17 = customer["AL"] * customer["h"]
                trans_replenish = sum3

            if difference1 < 0:
                transin_list2_sorted = sorted(transin_list2, key=lambda x: x[1]["a"] - x[1]["L"] + x[1]["dt"], reverse=True)
                current_sum = 0
                for index, customer in transin_list2_sorted:
                    if current_sum + customer["a"] - customer["L"] + customer["dt"] <= sum1:
                        current_sum += customer["a"] - customer["L"] + customer["dt"]
                        transdone_list3.append(copy.deepcopy(customer))
                    else:
                        stockout_list3.append(copy.deepcopy(customer))

                total_remain = sum([customer["a"] - customer["L"] + customer["dt"] for _, customer in transdone_list3])
                replen_list3_sorted = sorted(replen_list3, key=lambda x: x[1]["L"] - x[1]["dt"] - x[1]["a"],
                                             reverse=True)
                for index, customer in replen_list3_sorted:
                    if customer["L"] - customer["dt"] - customer["a"] <= total_remain:
                        # 如果(L-dt-a)小于等于余量，则更新顾客的库存Lt和余量
                        customer["Lt"] = customer["L"] - (customer["L"] - customer["dt"] - customer["a"]) - customer["dt"]
                        total_remain -= customer["L"] - customer["dt"] - customer["a"]
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    else:
                        # 如果(L-dt-a)大于余量，则更新顾客的库存Lt，并减去余量
                        customer["Lt"] = customer["L"] - total_remain - customer["dt"]
                        total_remain = 0
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost18 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(transdone_list3):
                    customer["Lt"] = customer["L"] + (customer["a"] - customer["L"] + customer["dt"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost19 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(stockout_list3):
                    customer["Lt"] = customer["L"] - customer["dt"]
                    if customer["Lt"] < 0:
                        customer["Lt"] = 0
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost20 = customer["AL"] * customer["h"]
                    trans_replenish = sum([customer["a"] - customer["L"] + customer["dt"] for _, customer in transin_list3])

            if difference1 > 0:
                current_sum = 0
                replen_list3_sorted = sorted(replen_list3, key=lambda x: x[1]["L"] - x[1]["dt"] - x[1]["a"],
                                             reverse=True)
                for index, customer in replen_list3_sorted:
                    if current_sum + customer["L"] - customer["dt"] - customer["a"] <= sum3:
                        current_sum += customer["L"] - customer["dt"] - customer["a"]
                        transdone_list4.append(copy.deepcopy(customer))
                    else:
                        stockout_list4.append(copy.deepcopy(customer))
                # 计算tansdone_list4中(L-dt-a)总量作为余量的初始量
                total_remain = sum([customer["L"] - customer["dt"] - customer["a"] for _, customer in transdone_list4])
                for index, (i, customer) in enumerate(transin_list2_sorted):
                    if customer["a"] - customer["L"] + customer["dt"] <= total_remain:
                        # 如果(a-L+dt)小于等于余量，则更新顾客的库存Lt和余量
                        customer["Lt"] = customer["L"] + (customer["a"] - customer["L"] + customer["dt"]) - customer["dt"]
                        total_remain -= customer["a"] - customer["L"] + customer["dt"]
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    else:
                        # 如果(a-L+dt)大于余量，则更新顾客的库存Lt，并减去余量
                        customer["Lt"] = customer["L"] + total_remain - customer["dt"]
                        total_remain = 0
                        customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost21 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(transdone_list4):
                    customer["Lt"] = customer["L"] - (customer["L"] - customer["dt"] - customer["a"]) - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost22 = customer["AL"] * customer["h"]

                for index, (i, customer) in enumerate(stockout_list4):
                    customer["Lt"] = customer["L"] - customer["dt"]
                    if customer["Lt"] < 0:
                        customer["Lt"] = 0
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost23 = customer["AL"] * customer["h"]
                trans_replenish = sum([customer["L"] - customer["dt"] - customer["a"] for _, customer in transdone_list4])

            else:  # 如果transin_list2为空集
                for index, (i, customer) in enumerate(replen_list3):
                    customer["Lt"]=customer["L"]-customer["dt"]

    else:  # 如果replen_list3为空集
        if transin_list1:
            for index, (i, customer) in enumerate(transin_list1):
                customer["Lt"] = customer["L"] - customer["dt"]
                if customer["Lt"] <0:
                    customer["Lt"] = 0
                customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                inventory_cost6 += customer["AL"] * customer["h"]
                stock_num += abs(customer["L"] - customer["dt"])
        else:  ###考虑replen_list3是空集但transin_list1或2不是空集的情况的更新
            if transin_list2 :
                for index, (i, customer) in enumerate(transin_list2):
                    customer["Lt"] = customer["L"] - customer["dt"]
                    customer["AL"] = (customer["Lt"] + customer["L"]) / 2
                    inventory_cost7 += customer["AL"] * customer["h"]


    # 合并拆分的列表
    replen_list = []
    if replen_list1_1:
        replen_list += copy.deepcopy(replen_list1_1)
    if replen_list1_3:
        replen_list += copy.deepcopy(replen_list1_3)
    if replen_list2_1:
        replen_list += copy.deepcopy(replen_list2_1)
    if replen_list2_2:
        replen_list += copy.deepcopy(replen_list2_2)
    replen_list = [dict(zip(["id", "location"], t)) for t in replen_list]
    replen_list.insert(0, {"id": 0, "location": copy.deepcopy(warehouse)})  # 将元组改为字典，并添加 "id" 键

    # 从 0 开始再标序号
    f_customer_number = len(replen_list) - 1  # 减去仓库字典
    print("replen_list")
    for i in range(f_customer_number + 1):
        sys.stdout.write("\n%d: " % i)
        print(replen_list[i])

    ########
    def distance_between_two_node(x1, x2, y1, y2):
        return math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))


    """Calculating Distance from one Customer TO all Other Customer"""
    distance_matrix = []
    for i in range(f_customer_number + 1):
        distance_array = []
        for j in range(f_customer_number + 1):
            distance_array.append(
                    distance_between_two_node(
                        replen_list[i]["location"]["x"],  # 索引 "x" 改为 "location" 中的 "x"
                        replen_list[j]["location"]["x"],
                        replen_list[i]["location"]["y"],  # 索引 "y" 改为 "location" 中的 "y"
                        replen_list[j]["location"]["y"],
                    )
                )
        distance_matrix.append(distance_array)

    distance = copy.deepcopy(distance_matrix)

    """Printing distance Matrix"""
    print("\nDisatance Matrix")
    for i in range(f_customer_number + 1):
        sys.stdout.write("\n%d: " % i)
        print(distance_matrix[i])




    ####
    def mini(dist_array):
        minimum_dist = sys.maxsize
        for i in range(1, len(dist_array)):
            if dist_array[i] == 0:
                continue
            else:
                minimum_dist = min(minimum_dist, dist_array[i])
        return [minimum_dist, dist_array.index(minimum_dist)]


    # Printing some result generated so far
    min_dist = []
    """This is later going to be the total distance we need to travel if we don't have any Time Period and Demand Constraint"""
    ideally_total_distance = 0
    ideally_initial_path = []
    k = 0

    """Getting meinimum distant node for each customer and print with index"""
    print("\nMinimum distant Node for each customer : ")
    for i in range(f_customer_number):
        sys.stdout.write("\n%d: " % k)
        ideally_initial_path.append(k)
        # return a array with two eleemnt-->min_diatance and index as path
        min_dist = mini(distance[k])
        k = min_dist[1]
        ideally_total_distance += min_dist[0]

        print(min_dist[0], min_dist[1])

        for j in range(f_customer_number + 1):
            distance[j][min_dist[1]] = 0

    """Lastly we have to reach to depo so we append this last node index and distance between (0,0) to the node"""
    ideally_initial_path.append(k)
    ideally_total_distance += int(distance_between_two_node(0, replen_list[k]["location"]["x"], 0, replen_list[k]["location"]["y"]))

    """Pheromone density(initial)"""                            #初始信息速浓度
    tau_0 = 1.0 / ideally_total_distance

    print("General Initial Path : {}".format(ideally_initial_path))
    print("General Total Distance : {}".format(ideally_total_distance))
    print("Tau_INITIAL : {}".format(tau_0))

    ####
    """Generate Pheromone Matrix"""             #生成信息素浓度矩阵
    pheromone_matrix = []
    for i in range(f_customer_number + 1):
        pheromone = []
        for j in range(f_customer_number + 1):
            if distance_matrix[i][j] == 0:
                pheromone.append(0.0)
            else:
                pheromone.append(tau_0)
        pheromone_matrix.append(pheromone)

    """Some printing"""
    print("\nDisatance Matrix")
    for i in range(f_customer_number + 1):
        sys.stdout.write("\n%d: " % i)
        print(distance_matrix[i])
    print("\nPheromone Matrix")
    for i in range(f_customer_number + 1):
        sys.stdout.write("\n%d: " % i)
        print(pheromone_matrix[i])

    ####
    """Probability function to choose the best route"""     #概率公式，用于选择最佳路径
    """State Transition Probability"""
    def probability(route):
        r = []                                                #禁忌表，用来存储已访问顾客
        customer_instance = []                                #用来存储未访问顾客
        for i in range(1, f_customer_number + 1):
            r.append(i)

        for i in r:
            if i not in route:
                customer_instance.append(i)

        sigma = 0                                           #用来表示状态转移概率的总和——就是轮盘赌算法中的累计概率
        rev_route = route[-1]                               #表示当前路线的最后一个城市

        """ When q<q0 then according to Biased Roulette Method with state transition probability is
            max(sigma) where sigma=((tau_ij)^alpha)/((c_ij)^beta)
        """
        for i in customer_instance:
            if rev_route != i:                              #!=表示不等于，当前路线的最后一个顾客不在未访问列表
                try:

                    sigma += (pheromone_matrix[rev_route][i] ** alpha) * (
                    (1.0 / distance_matrix[rev_route][i]) ** beta)
                except ZeroDivisionError:  # 忽略分母为0
                    continue
        p_max = 0.0                                       #表示未访问城市中概率最大的城市
        for i in customer_instance:
            p_ij = 0.0

            # If the path is not visited then
            if rev_route != i:
                try:
                    p_ij = (pheromone_matrix[rev_route][i] ** alpha) * (
                    (1.0 / distance_matrix[rev_route][i]) ** beta)
                except ZeroDivisionError:  # Ignore if the deominator is 0
                    continue
                if p_ij > p_max:
                    p_max = p_ij
                    try:
                        b = i
                    except:
                        print("Can be taken")
        return [p_max, b]                         #返回未访问城市中概率最大的城市，以及其索引


    #####
    """Claculate the path distance for the given route"""


    def path_distance(route, glob=False):
        dist = 0
        for i in range(len(route) - 1):
            dist += distance_matrix[route[i]][route[i + 1]]
        return dist


    """Function to Update the pheromone after each iteration"""


    def update_pheromone(route, glob=False):
        print(route)
        dist = path_distance(route)

        """For loop only for if we want ot update it locally"""     #局部更新信息素
        for i in range(len(route) - 1):
            pheromone_local[route[i]][route[i + 1]] += Q / dist

        """This one for global update"""     #全局更新信息素
        if glob:
            for i in range(f_customer_number):
                for j in range(f_customer_number):
                    pheromone_matrix[i][j] += (
                    rho * pheromone_matrix[i][j] + pheromone_local[i][j])
        return dist

    """Run the process for certain number of generation and filter best route and best distance in each iteartion"""
    ###
    global best_route, best_distance
    best_route = []
    best_distance = ideally_total_distance
    track_min_distance = []  # For Ploting the graph

    for n in range(gen_number):
        print("\nIteration No: {}".format(n))

        """Start with random root"""
        ant_route = []
        for i in range(m):
            ant = []
            ant.append(random.randint(1, f_customer_number))            #随机生成1-num之间的整数
            ant_route.append(ant)                                       #生成蚂蚁的初始位置

        """Calculate the Probabilty for each route and extract the max"""   #计算每条路由的概率并提取最大值
        for i in range(m):
            a = []
            for j in range(f_customer_number - 1):
                p1 = probability(ant_route[i])
                # Get the index of maximum probability route
                ant_route[i].append(p1[1])
        """Get the minimum path distane for maximum probabilty route"""
        di = []
        for i in range(m):
            di.append(path_distance(ant_route[i]))
        # sys.stdout.write("Minimum Distance: ")
        print("Minimum Distance : {}".format(min(di)))
        track_min_distance.append(min(di))
        ant = di.index(min(di))

        """Copy the Pheromone Matrix to manipulate"""
        pheromone_local = copy.deepcopy(pheromone_matrix)

        """Finally Get the best route and best distance"""
        if update_pheromone(ant_route[ant], True) < best_distance:
            best_distance = update_pheromone(ant_route[ant], True)
            best_route = ant_route[i]
        final_route = [replen_list[i]["id"] for i in best_route]
        print(update_pheromone(ant_route[ant], True))

    print("********\nBest Distance : {}".format(best_distance))
    print("Best Routes")
    print(best_route)
    print(final_route)
    def update_customers(customer_list, *args):
        # 创建一个辅助字典，用于存储最新的顾客信息
        customer_dict = {id: customer for id, customer in customer_list}

        # 遍历所有传入的列表，更新顾客字典中的信息
        for lst in args:
            for id, customer in lst:
                customer_dict[id] = customer

        # 使用辅助字典更新原始的 customer_list
        updated_customer_list = [(id, customer_dict[id]) for id in sorted(customer_dict)]

        return updated_customer_list

    customer_list = update_customers(customer_list, replen_list1_1, replen_list1_2, replen_list1_3, replen_list2_1,
                                     replen_list2_2, transin_list1, transin_list2, transdone_list1, transdone_list2,
                                     transdone_list3, transdone_list4, stockout_list1, stockout_list2, stockout_list3,
                                     stockout_list4)

    ####计算各种成本
    inventory_cost = inventory_cost1 + inventory_cost2 + inventory_cost3 + inventory_cost4 + inventory_cost5 \
                    + inventory_cost6 + inventory_cost7 + inventory_cost8 + inventory_cost9 + inventory_cost10\
                    + inventory_cost11 + inventory_cost12 + inventory_cost13 + inventory_cost14 + inventory_cost15\
                    + inventory_cost16 + inventory_cost17 + inventory_cost18 + inventory_cost19 + inventory_cost20 \
                    + inventory_cost21 + inventory_cost22 + inventory_cost23
    delivery_cost = cij*best_distance
    stockout_cost = f*stock_num
    transship_cost = trans_replenish*bij
    total_cost = inventory_cost+delivery_cost+stockout_cost+transship_cost

    return best_route, inventory_cost, delivery_cost, stockout_cost, transship_cost,total_cost




# 主程序部分
if __name__ == "__main__":
    sys.stdout = open("output.txt", "w")

    # 加载第一个周期的客户数据
    customer_list = read_customer_data('customer_data.xlsx')
    # 存储每个周期的最佳路径和成本
    period_details = {}

    # 循环4个周期
    for period in range(1, 5):
        print("\nPeriod:", period)
        best_route, inventory_cost, delivery_cost, stockout_cost, transship_cost, total_cost = run_simulation(period, customer_list)
        # 存储每个周期的最佳路径和成本细节
        period_details[period] = {
            "Best Route": best_route,
            "Inventory Cost": inventory_cost,
            "Delivery Cost": delivery_cost,
            "Stockout Cost": stockout_cost,
            "Transship Cost": transship_cost,
            "Total Cost": total_cost,
            "Customer List":customer_list
        }
    for i, customer in customer_list:
        customer["L"] = customer["Lt"]  # 更新期初库存为期末库存

    # 打印每个周期的最佳路径和成本
    print("\nSummary per period:")
    for period, details in period_details.items():
        print(f"Period {period} Summary:")
        print(f"  Best Route: {details['Best Route']}")
        print(f"  Inventory Cost: {details['Inventory Cost']}")
        print(f"  Delivery Cost: {details['Delivery Cost']}")
        print(f"  Stockout Cost: {details['Stockout Cost']}")
        print(f"  Transship Cost: {details['Transship Cost']}")
        print(f"  Total Cost: {details['Total Cost']}")
        print()

    # 关闭输出文件
    sys.stdout.close()
