import math
import gurobipy as gp
import numpy as np
import sys
from gurobipy import GRB

class task():
    def __init__(self,name,wcec,rel,deadline):
        self.name = name
        self.wcec = wcec
        self.rel = rel
        self.deadline = deadline


#REL = 0.9999623

#Deadline = 216.547

# L = [0.801, 0.8291, 0.8553, 0.8797, 0.9027, 1.0]

# L = [0.15, 0.40, 0.60, 0.80, 1.0]

# V = {0.801:0.85, 0.8291:0.90, 0.8553:0.95, 0.8797:1.00, 0.9027:1.05, 1.0:1.1}

# V = {0.15: 0.15, 0.40: 0.40, 0.60: 0.60, 0.80: 0.80, 1.0: 1.0}

# CEFF = {0.801:7.3249, 0.8291:8.6126, 0.8553:10.238, 0.8797:12.315, 0.9027:14.998, 1.0:18.497}

# CEFF = {0.15: 1.0, 0.40: 1.0, 0.60: 1.0, 0.80: 1.0, 1.0: 1.0}

# M = [1, 2, 3, 4, 5, 6, 7, 8]

def read_task(taskfile):
    f = open(taskfile, 'r+')
    tasks = []
    line = f.readline()
    rel = 0.99
    d = 200
    while line:
        str = line.split(' ')
        task1 = task(int(str[0]) - 1, float(str[1]), float(rel), float(d))
        tasks.append(task1)
        line = f.readline()
    t_num = tasks.__len__()
    for i in range(t_num):
        t1 = tasks[i]
        nid = int(t1.name) + t_num
        t2 = task(nid, t1.wcec, float(rel), float(d))
        tasks.append(t2)
    return tasks

def read_three_task(taskfile):
    f = open(taskfile, 'r+')
    tasks = []
    line = f.readline()
    rel = 0.99
    d = 200
    while line:
        str = line.split(' ')
        task1 = task(int(str[0]) - 1, float(str[1]), float(rel), float(d))
        tasks.append(task1)
        line = f.readline()
    t_num = tasks.__len__()
    for i in range(t_num):
        t1 = tasks[i]
        nid = int(t1.name) + t_num
        t2 = task(nid, t1.wcec, float(rel), float(d))
        tasks.append(t2)
    for i in range(t_num):
        t1 = tasks[i]
        nid = int(t1.name) + 2 * t_num
        t2 = task(nid, t1.wcec, float(rel), float(d))
        tasks.append(t2)
    return tasks



def read_edge(edgefile,tasks):
    f = open(edgefile, 'r+')
    node_num = int(tasks.__len__() / 2)
    o = np.zeros((2*node_num, 2*node_num))
    line = f.readline()
    while line:
        str = line.split(' ')
        o[int(str[0]) - 1][int(str[1]) - 1] = 1
        o[int(str[0]) - 1][int(str[1]) - 1 + node_num] = 1
        o[int(str[0]) - 1 + node_num][int(str[1]) - 1] = 1
        o[int(str[0]) + node_num - 1][int(str[1]) + node_num - 1] = 1
        line = f.readline()
    return o

def read_three_edge(edgefile,tasks):
    f = open(edgefile, 'r+')
    node_num = int(tasks.__len__() / 3)
    o = np.zeros((3*node_num, 3*node_num))
    line = f.readline()
    while line:
        str = line.split(' ')
        o[int(str[0]) - 1][int(str[1]) - 1] = 1
        o[int(str[0]) - 1][int(str[1]) - 1 + node_num] = 1
        o[int(str[0]) - 1][int(str[1]) - 1 + 2 * node_num] = 1
        o[int(str[0]) - 1 + node_num][int(str[1]) - 1] = 1
        o[int(str[0]) + node_num - 1][int(str[1]) + node_num - 1] = 1
        o[int(str[0]) + node_num - 1][int(str[1]) + 2 * node_num - 1] = 1
        o[int(str[0]) - 1 + 2 * node_num][int(str[1]) - 1] = 1
        o[int(str[0]) + 2 * node_num - 1][int(str[1]) + node_num - 1] = 1
        o[int(str[0]) + 2 * node_num - 1][int(str[1]) + 2 * node_num - 1] = 1
        line = f.readline()
    return o


def slove_scheduling(tasks,L,V,CEFF,M,o,d, st):

    t_num = int(tasks.__len__() / 2)

    N = [i for i in range(int(tasks.__len__() / 2))]

    N1 = [t.name for t in tasks]

    print(N1)

    Rel = {t.name: t.rel for t in tasks}

    print(Rel)

    D = {t.name: t.deadline for t in tasks}

    WCEC = {t.name: t.wcec for t in tasks}

    print(WCEC)

    #for i in range(10, 20):
    #    WCEC[i] = WCEC[i - 10]

    P = {l: CEFF[l]*V[l]*V[l]*l for l in L}

    e = gp.Env()

    e.setParam('TimeLimit', 300)

    ms = gp.Model("schdule0", env=e)

    s = ms.addVars(L, N1, vtype=GRB.BINARY, name="s")

    q = ms.addVars(M, N1, vtype=GRB.BINARY, name="q")

    w = ms.addVars(N1, N1, vtype=GRB.BINARY, name="w")

    h = ms.addVars(N1, N1, M, vtype=GRB.BINARY, name="h")

    g = ms.addVars(N1, N1, L, L, vtype=GRB.BINARY, name="g")

    b = ms.addVars(N1, L, vtype=GRB.BINARY, name="b")

    ts = ms.addVars(N1, name="ts")

    te = ms.addVars(N1, name="te")

    #realrel = m.addVars(N1, name="realrel")

    # te = {i: gp.quicksum(s[l, i]*WCEC[i]/l for l in L) for i in N1}

    r = {i: {l: math.exp(-0.000001 * math.exp(4 * (1 - l) / (1 - L[0]))*(WCEC[i] / l)) for l in L} for i in N1}

    # print(r[0][0.801])

    sigma = ms.addVars(N1, vtype=GRB.BINARY, name="sigma")

    realrel = {n: gp.quicksum(s[l, n]*r[n][l] for l in L) for n in N1}


    # Temporal constraints (1)
    ms.addConstrs((s.sum('*', n) == sigma[n] for n in N1), name="sfre")

    # Temporal constraints (4)
    ms.addConstrs((q.sum('*', n) == sigma[n] for n in N1), name="qconstr")

    #m.addConstrs((realrel[n] == gp.quicksum(s[l, n] * r[n][l] for l in L) for n in N1))

    # Temporal constraints (2)
    ms.addConstrs((0.000000001 - (1 + 0.000000001) * sigma[t_num + i] <= realrel[i] - Rel[i] for i in N),
                 name="reliable1")

    ms.addConstrs((realrel[i] - Rel[i] <= 1 - sigma[t_num + i] for i in N),
                 name="reliable2")

    ms.addConstrs((sigma[i] >= sigma[t_num + i] for i in N), name="wuzhiweikkk")

    ms.addConstrs((g[i, t_num + i, l1, l2] <= s[l1, i] for i in N for l1 in L for l2 in L), name="rellinear1")
    ms.addConstrs((g[i, t_num + i, l1, l2] <= s[l2, t_num + i] for i in N for l1 in L for l2 in L), name="rellinear2")
    ms.addConstrs((s[l1, i] + s[l2, t_num + i] - g[i, t_num + i, l1, l2] <= 1 for i in N for l1 in L for l2 in L), name="rellinear3")

    R = {i: (1 - sigma[t_num + i]) * realrel[i] + sigma[t_num + i] * (realrel[i] + realrel[t_num + i]) - sigma[t_num + i] *gp.quicksum(g[i, t_num + i, l1, l2]*r[i][l1]*r[i+t_num][l2] for l1 in L for l2 in L) for i in N}

    #R = {i: (1 - sigma[t_num + i]) * realrel[i] + sigma[t_num + i] * (1 - (1 - realrel[t_num + i]) * (1 - realrel[i])) for i in N}

    # Temporal constraints (3)
    ms.addConstrs((R[i] >= Rel[i] for i in N), name="rrr")

    # Temporal constraints (5)
    ms.addConstrs((q[m, i] + q[m, i + t_num] <= 1 for m in M for i in N), name="qdiff")

    ms.addConstrs((te[i] == ts[i] + gp.quicksum(s[l, i] * WCEC[i] / l for l in L) for i in N1), name="tefind")

    # Temporal constraints (6)
    ms.addConstrs((te[i] <= ts[j] + (2 - q[m, i] - q[m, j])*d + (1-w[i, j])*d for m in M for i in N1 for j in N1 if i != j), name="no-overlapping")

    #ms.addConstrs((w[i, j] + w[j, i] == gp.quicksum(q[m, i] * q[m, j] for m in M) for i in N1 for j in N1 if i != j), name="no-overlapping2")



    #ms.addConstrs((h[i, j, m] == q[m, i] * q[m, j] for i in N1 for j in N1 for m in M if i != j), name="h1")

    #ms.addConstrs((h[i, i, m] == 0 for i in N1 for m in M), name="h2")

    ms.addConstrs((w[i, j] + w[j, i] == h.sum(i, j, '*') for i in N1 for j in N1 if i != j), name="hc1")

    ms.addConstrs((h[i, j, m] <= q[m, i] for i in N1 for j in N1 for m in M if i != j), name="hc2")

    ms.addConstrs((h[i, j, m] <= q[m, j] for i in N1 for j in N1 for m in M if i != j), name="hc3")

    ms.addConstrs((q[m, i] + q[m, j] - h[i, j, m] <= 1 for i in N1 for j in N1 for m in M if i != j), name="hc4")

    ms.addConstrs((ts[j] + (1 - o[i][j]) * d >= ts[i] + o[i][j] * gp.quicksum(s[l, i] * WCEC[i] / l for l in L) for i in N1 for j in N1 if i != j), name="task-depend")

    ms.addConstrs((te[i] <= d for i in N1), name="deadline")

    ms.setObjective(gp.quicksum(s[l, n]*WCEC[n]/l*P[l] for l in L for n in N1), GRB.MINIMIZE)

    ms.optimize()

    #print(q[1,0].X)
    fo = open(st, "w+")

    oe = 0.0

    for i in N1:
        for j in M:
            for l in L:
                if q[j, i].X > 0.5 and s[l, i].X > 0.5:
                    print("task number:{},q:{},l:{},ts:{},te:{}".format(i,j,l,ts[i].X,te[i].X))
                    fo.write("{} {} {} {:.3f} {:.3f}\n".format(i,j,l,ts[i].X,te[i].X))
                    oe += (te[i].X - ts[i].X) * l * l * l

    print("ori-energy:{}".format(oe))
    #for n in N1:
    #    for l in L:
    #        if s[l, n].X > 0.5:
    #            print("task number: {},l:{},ts:{},te:{}".format(n, l, ts[n].X, te[n].X))




def slove_three_scheduling(tasks,L,V,CEFF,M,o,d, st):
    t_num = int(tasks.__len__() / 3)

    N = [i for i in range(int(tasks.__len__() / 3))]

    N1 = [t.name for t in tasks]

    print(N1)

    Rel = {t.name: t.rel for t in tasks}

    print(Rel)

    D = {t.name: t.deadline for t in tasks}

    WCEC = {t.name: t.wcec for t in tasks}

    print(WCEC)

    # for i in range(10, 20):
    #    WCEC[i] = WCEC[i - 10]

    P = {l: CEFF[l] * V[l] * V[l] * l for l in L}

    e = gp.Env()

    e.setParam('TimeLimit', 300)

    ms = gp.Model("schdule0", env=e)

    s = ms.addVars(L, N1, vtype=GRB.BINARY, name="s")

    q = ms.addVars(M, N1, vtype=GRB.BINARY, name="q")

    w = ms.addVars(N1, N1, vtype=GRB.BINARY, name="w")

    h = ms.addVars(N1, N1, M, vtype=GRB.BINARY, name="h")

    g = ms.addVars(N1, N1, L, L, vtype=GRB.BINARY, name="g")

    g2 = ms.addVars(N1, L, L, L, vtype=GRB.BINARY, name="g2")

    b = ms.addVars(N1, L, vtype=GRB.BINARY, name="b")

    ts = ms.addVars(N1, name="ts")

    te = ms.addVars(N1, name="te")

    r = {i: {l: math.exp(-0.000001 * math.exp(4 * (1 - l) / (1 - L[0])) * (WCEC[i] / l)) for l in L} for i in N1}

    sigma = ms.addVars(N1, vtype=GRB.BINARY, name="sigma")

    realrel = {n: gp.quicksum(s[l, n] * r[n][l] for l in L) for n in N1}

    # Temporal constraints (1): Frequency assignment
    ms.addConstrs((s.sum('*', n) == sigma[n] for n in N1), name="sfre")

    # Temporal constraints (4)
    ms.addConstrs((q.sum('*', n) == sigma[n] for n in N1), name="qconstr")

    ms.addConstrs((sigma[i] >= sigma[t_num + i] for i in N), name="wuzhiweikkk1")

    ms.addConstrs((sigma[t_num + i] >= sigma[2 * t_num + i] for i in N), name="wuzhiweikkk2")

    # Temporal constraints (2):

    ms.addConstrs((0.000000001 - (1 + 0.000000001) * sigma[t_num + i] <= realrel[i] - Rel[i] for i in N),
                  name="reliable01")

    ms.addConstrs((realrel[i] - Rel[i] <= 1 - sigma[t_num + i] for i in N),
                  name="reliable02")

    ms.addConstrs((g[i, t_num + i, l1, l2] <= s[l1, i] for i in N for l1 in L for l2 in L), name="rellinear1")
    ms.addConstrs((g[i, t_num + i, l1, l2] <= s[l2, t_num + i] for i in N for l1 in L for l2 in L), name="rellinear2")
    ms.addConstrs((s[l1, i] + s[l2, t_num + i] - g[i, t_num + i, l1, l2] <= 1 for i in N for l1 in L for l2 in L),
                  name="rellinear3")

    R = {i: (1 - sigma[t_num + i]) * realrel[i] + sigma[t_num + i] * (realrel[i] + realrel[t_num + i]) - sigma[
        t_num + i] * gp.quicksum(g[i, t_num + i, l1, l2] * r[i][l1] * r[i + t_num][l2] for l1 in L for l2 in L) for i in
         N}

    ms.addConstrs((0.000000001 - (1 + 0.000000001) * sigma[2 * t_num + i] <= R[i] - Rel[i] for i in N),
                  name="reliable11")

    ms.addConstrs((R[i] - Rel[i] <= 1 - sigma[2 * t_num + i] for i in N),
                  name="reliable12")

    ms.addConstrs((g[i, 2 * t_num + i, l1, l2] <= s[l1, i] for i in N for l1 in L for l2 in L), name="rellinear4")
    ms.addConstrs((g[i, 2 * t_num + i, l1, l2] <= s[l2, 2 * t_num + i] for i in N for l1 in L for l2 in L), name="rellinear5")
    ms.addConstrs((s[l1, i] + s[l2, 2 * t_num + i] - g[i, 2 * t_num + i, l1, l2] <= 1 for i in N for l1 in L for l2 in L),
                  name="rellinear6")

    ms.addConstrs((g[t_num + i, 2 * t_num + i, l1, l2] <= s[l1, t_num + i] for i in N for l1 in L for l2 in L), name="rellinear7")
    ms.addConstrs((g[t_num + i, 2 * t_num + i, l1, l2] <= s[l2, 2 * t_num + i] for i in N for l1 in L for l2 in L),
                  name="rellinear8")
    ms.addConstrs(
        (s[l1, t_num + i] + s[l2, 2 * t_num + i] - g[t_num + i, 2 * t_num + i, l1, l2] <= 1 for i in N for l1 in L for l2 in L),
        name="rellinear9")

    ms.addConstrs((g2[i, l1, l2, l3] <= s[l1, i] for i in N for l1 in L for l2 in L for l3 in L), name="rellinear10")
    ms.addConstrs(
        (g2[i, l1, l2, l3] <= s[l2, t_num + i] for i in N for l1 in L for l2 in L for l3 in L),
        name="rellinear10")
    ms.addConstrs(
        (g2[i, l1, l2, l3] <= s[l3, 2 * t_num + i] for i in N for l1 in L for l2 in L for l3 in L),
        name="rellinear11")
    ms.addConstrs(
        (g2[i, l1, l2, l3] >= s[l1, i] + s[l2, t_num + i] + s[l3, 2 * t_num + i] - 2 for i in N for l1 in L for l2 in L for l3 in L),
        name="rellinear12")

    ss = ms.addVars(N1, vtype=GRB.BINARY, name="ss")

    ms.addConstrs((ss[i] <= sigma[t_num + i] for i in N), name="sigmamulti1")

    ms.addConstrs((ss[i] <= sigma[2 * t_num + i] for i in N), name="sigmamulti2")

    ms.addConstrs((ss[i] >= sigma[t_num + i] + sigma[2 * t_num + i] - 1 for i in N), name="sigmamulti3")

    R2 = {i: realrel[i] - sigma[t_num + i] * realrel[i] - sigma[2 * t_num + i] * realrel[i] + ss[i] * realrel[i]
              + sigma[t_num + i] * (realrel[i] + realrel[t_num + i]) - ss[i] * (realrel[i] + realrel[t_num + i])
              - sigma[t_num + i] * gp.quicksum(g[i, t_num + i, l1, l2] * r[i][l1] * r[i + t_num][l2] for l1 in L for l2 in L)
              + ss[i] * gp.quicksum(g[i, t_num + i, l1, l2] * r[i][l1] * r[i + t_num][l2] for l1 in L for l2 in L)
              + sigma[2 * t_num + i] * ( realrel[i] + realrel[t_num + i] + realrel[2 * t_num + i] )
              - sigma[2 * t_num + i] * gp.quicksum(g[i, t_num + i, l1, l2] * r[i][l1] * r[i+t_num][l2] for l1 in L for l2 in L)
              - sigma[2 * t_num + i] * gp.quicksum(g[i, 2 * t_num + i, l1, l2] * r[i][l1] * r[i + 2 * t_num][l2] for l1 in L for l2 in L)
              - sigma[2 * t_num + i] * gp.quicksum(g[t_num + i, 2 * t_num + i, l2, l3]*r[t_num + i][l2]*r[i+ 2 * t_num][l3] for l2 in L for l3 in L)
              + sigma[2 * t_num + i] * gp.quicksum(g2[i, l1, l2, l3] * r[i][l1] * r[t_num + i][l2] * r[2 * t_num + i][l3] for l1 in L for l2 in L for l3 in L ) for i in N}



    # Temporal constraints (3)
    ms.addConstrs((R2[i] >= Rel[i] for i in N), name="rrr")

    # Temporal constraints (5)
    ms.addConstrs((q[m, i] + q[m, i + t_num] <= 1 for m in M for i in N), name="qdiff1")

    ms.addConstrs((q[m, i] + q[m, i + 2 * t_num] <= 1 for m in M for i in N), name="qdiff2")

    ms.addConstrs((q[m, i + t_num] + q[m, i + 2 * t_num] <= 1 for m in M for i in N), name="qdiff3")

    ms.addConstrs((te[i] == ts[i] + gp.quicksum(s[l, i] * WCEC[i] / l for l in L) for i in N1), name="tefind")

    # Temporal constraints (6)
    ms.addConstrs(
        (te[i] <= ts[j] + (2 - q[m, i] - q[m, j]) * d + (1 - w[i, j]) * d for m in M for i in N1 for j in N1 if i != j),
        name="no-overlapping")

    ms.addConstrs((w[i, j] + w[j, i] == h.sum(i, j, '*') for i in N1 for j in N1 if i != j), name="hc1")

    ms.addConstrs((h[i, j, m] <= q[m, i] for i in N1 for j in N1 for m in M if i != j), name="hc2")

    ms.addConstrs((h[i, j, m] <= q[m, j] for i in N1 for j in N1 for m in M if i != j), name="hc3")

    ms.addConstrs((q[m, i] + q[m, j] - h[i, j, m] <= 1 for i in N1 for j in N1 for m in M if i != j), name="hc4")

    ms.addConstrs(
        (ts[j] + (1 - o[i][j]) * d >= ts[i] + o[i][j] * gp.quicksum(s[l, i] * WCEC[i] / l for l in L) for i in N1 for j
         in N1 if i != j), name="task-depend")

    ms.addConstrs((te[i] <= d for i in N1), name="deadline")

    ms.setObjective(gp.quicksum(s[l, n] * WCEC[n] / l * P[l] for l in L for n in N1), GRB.MINIMIZE)

    ms.optimize()

    # print(q[1,0].X)
    fo = open(st, "w+")

    oe = 0.0


    for i in N1:
        for j in M:
            for l in L:
                if q[j, i].X > 0.5 and s[l, i].X > 0.5:
                    print("task number:{},q:{},l:{},ts:{},te:{}".format(i, j, l, ts[i].X, te[i].X))
                    fo.write("{} {} {} {:.3f} {:.3f}\n".format(i, j, l, ts[i].X, te[i].X))
                    oe += (te[i].X - ts[i].X) * l * l * l

    print("ori-energy:{}".format(oe))


def GS_time(filename, tasks):
    t_num = int(tasks.__len__() / 2)
    f = open("gurobi-result.txt","r+")
    line = f.readline()
    s = {}
    while line:
        str = line.split(' ')
        s[int(str[0])] = str
        line = f.readline()
    e = 0.0
    num = 0
    for str in s:
        num = num + 1
    for i in range(t_num):
        if s.get(i + t_num) == None:
            #print("no other replica id:{}".format(i))
            l1 = float(s[i][2])
            wcec = float(s[i][4]) - float(s[i][3])
            e += wcec * l1 * l1 * l1
        elif float(s[i][4]) < float(s[i+t_num][4]):
            l1 = float(s[i][2])
            wcec = float(s[i][4]) - float(s[i][3])
            l2 = float(s[i+t_num][2])
            addi = max(0.0, float(s[i][4]) - float(s[i+t_num][3]))
            e += wcec * l1 * l1 * l1 + addi * l2 * l2 * l2
        else:
            l1 = float(s[i+t_num][2])
            wcec = float(s[i+t_num][4]) - float(s[i+t_num][3])
            l2 = float(s[i][2])
            addi = max(0.0, float(s[i+t_num][4]) - float(s[i][3]))
            e += wcec * l1 * l1 * l1 + addi * l2 * l2 * l2

    print("energy cost:{}".format(e))
    f1 = open(filename, 'a')
    f1.write("{}\n".format(e))
    f1.close()
    return e





if __name__ == "__main__":
    # Base model
    #print(o[0][1])
    nodename = sys.argv[1]
    edgename = sys.argv[2]
    tasks = read_task(nodename)
    o = read_edge(edgename, tasks)
    #tasks = read_three_task("node_Seismology_workflow_50.txt")
    #o = read_three_edge("edge_Seismology_workflow_50.txt", tasks)
    M = [1, 2, 3, 4, 5, 6, 7, 8]
    #L = [0.211, 0.250, 0.281, 0.316, 0.375, 0.421, 0.50, 0.562, 0.633, 0.750, 0.844, 1.0]
    #V = {0.211: 0.211, 0.250: 0.250, 0.281: 0.281, 0.316: 0.316, 0.375: 0.375, 0.421: 0.421, 0.50: 0.50, 0.562: 0.562, 0.633: 0.633, 0.750: 0.750, 0.844: 0.844, 1.0: 1.0}
    #CEFF = {0.211: 1.0, 0.250: 1.0, 0.281: 1.0, 0.316: 1.0, 0.375: 1.0, 0.421: 1.0, 0.50: 1.0, 0.562: 1.0, 0.633: 1.0, 0.750: 1.0, 0.844: 1.0, 1.0: 1.0}
    #L = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #V = {0.1: 0.1, 0.2: 0.2, 0.3: 0.3, 0.4: 0.4, 0.5: 0.5, 0.6: 0.6, 0.7: 0.7, 0.8: 0.8, 0.9: 0.9, 1.0: 1.0}
    #CEFF = {0.1: 1.0, 0.2: 1.0, 0.3: 1.0, 0.4: 1.0, 0.5: 1.0, 0.6: 1.0, 0.7: 1.0, 0.8: 1.0, 0.9: 1.0, 1.0: 1.0}
    #L = [0.21, 0.32, 0.46, 0.57, 0.71, 0.86, 1.0]
    #V = {0.21: 0.21, 0.32: 0.32, 0.46: 0.46, 0.57: 0.57, 0.71: 0.71, 0.86: 0.86, 1.0: 1.0}
    #CEFF = {0.21: 1.0, 0.32: 1.0, 0.46: 1.0, 0.57: 1.0, 0.71: 1.0, 0.86: 1.0, 1.0: 1.0}

    L = [0.15, 0.40, 0.60, 0.80, 1.0]
    V = {0.15: 0.15, 0.40: 0.40, 0.60: 0.60, 0.80: 0.80, 1.0: 1.0}
    CEFF = {0.15: 1.0, 0.40: 1.0, 0.60: 1.0, 0.80: 1.0, 1.0: 1.0}


    f = open("output_out_zmilp_f1_" + nodename)
    line = f.readline()
    t_num = int(tasks.__len__() / 2)
    r_num = 0
    d_num = 0
    line_num = 0
    while line:
        str = line.split(' ')
        line_num = line_num + 1
        r_num = int((line_num - 1) / 5)
        d_num = line_num - (r_num * 5)
        print("r_num:{},d_num:{}".format(r_num,d_num))
        r = str[1]
        d = str[2]
        if line_num <= 0:
            line = f.readline()
            continue
        for t in tasks:
            t.rel = float(r)**(1/t_num)
            t.deadline = float(d)
        scheduling_text = "MILP_f0_node_" + nodename[:-4] + "_{}_{}.txt".format(r_num, d_num)
        print(scheduling_text)
        slove_three_scheduling(tasks, L, V, CEFF, M, o, float(d), scheduling_text)
        #GS_time("task_milp_result_f2_Genome_10.txt", tasks)
        line = f.readline()
    #print(2*tasks.__len__())

