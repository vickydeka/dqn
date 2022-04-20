import time
import random
import numpy as np
from memory import *
from DDPG import Agent

class Task(object):   
    def __init__(self, jobID, index, CPU, RAM, disk, status):
        self.parent = []
        self.child = []
        self.jobID = jobID
        self.index = index
        self.CPU = CPU
        self.RAM = RAM
        self.disk = disk
        self.status = status  #-1: rejected, 0: finished, 1: ready, 2: running
        self.runtime = random.randint(1, 10)/1000.0
        self.ddl = time.time() + self.runtime + random.randint(1, 1000) * 100
        self.endtime = 0
 

class DAG(object):
    """
    Transform job queue to task ready queue
    """
    def __init__(self, fname, num_task):
        self.fname = fname
        self.num_task = num_task
        self.job = []
        self.task = []
    
    def readfile(self):
        """
        Read the input job file
        All task are initialized to ready status
        """
        num_task = 0
        with open(self.fname, 'r') as f:
            task = []            
            for line in f:
                if line[0] == 'J':
                    if len(task) != 0:
                        self.job.append(task)
                        task = []
                else:
                    info = list(line.strip(' ').split())
                    task.append(Task(info[0], info[1], float(info[2]), float(info[3]), info[4], 1))
                    num_task += 1
                if num_task == self.num_task: 
                    break
            if len(task) != 0:
                self.job.append(task)
    
    def checkRing(self, parent, child): 
        """
        Check whether there is a loop between parent and child
        Return True if has loop
        """
        if parent.index == child.index:
            return True
        if len(child.child) == 0:
            return False
        for c in child.child:
            if self.checkRing(parent, c):
                return True
        return False
    
    
    def buildDAG(self):
        """
        Randomly build dependencies between tasks within each job
        """
        import random
        for job in self.job:           
            for task in job:
                i = random.randint(-len(job), len(job) - 1)
                if i < 0:
                    continue
                parent = job[i]
                if self.checkRing(parent, task) == False:
                    task.parent.append(parent)
                    parent.child.append(task)
    
    def rejTask(self, task):
        """
        If one task is rejected
        Then all tasks that depended on this task will be rejected
        """
        task.status = -1
        for c in task.child:
            self.rejTask(c)
    
    def hasParent(self, task):
        """
        When a task are finished
        Remove it from the parent for all child tasks
        """
        for c in task.parent:
            if c.status == 1:  #still has parent
                return True
        return False
    
    def updateStatus(self, task):
        if task.status == -1:
            self.rejTask(task)
    

    def initTask(self):
        """
        run readfile and buildDAG functions
        """
        self.readfile()
        self.buildDAG()
    
    def taskQueue(self): 
        """
        Build the task ready queue
        Just put the one whose status is 1 
        and whose parent are all finished
        """
        for job in self.job:
            for task in job:
                if task.status == 1 and self.hasParent(task) == False:
                    self.task.append(task)


    def printTask(self):
        """
        Print tasks which are in task queue info
        """
        for j in self.task:
            print(j.jobID, ",", j.index, ",", j.status, ",", len(j.parent))     


class environment(object):

    def __init__(self, scale, fname, num_task, num_server):
        self.scale = scale
        self.fname = fname
        self.task = []
        self.dag = DAG(self.fname, num_task)
        self.VMNum = 5
        self.rej = 0
        self.num_task = num_task
        self.severNum = num_server
        if self.scale == 'small':
#             self.severNum = 200
            self.controllerNum = 10
        elif self.scale == 'large':
#             self.severNum = 4000
            self.controllerNum = int(self.severNum / 50)
        # self.init_severs()
        self.remaincontroller = []
        self.controllerResources = []
        self.severs = [[1,1]for _ in range(self.severNum)]
        self.VMtask = []
        self.totalcost = 0
#         print("Total Number of tasks: {0}".format(num_task))

    def init_severs(self, severNum):
        VM = [[[1.0/self.VMNum, 1.0/self.VMNum]for _ in range(self.VMNum)]for _ in range(severNum)]
#         VM = [[[1.0 , 1.0 ] for _ in range(self.VMNum)] for _ in range(severNum)]
        self.VMtask.append([[[]for _ in range(self.VMNum)]for _ in range(severNum)])
        return VM
    
    def generateQueue(self):
        self.dag.taskQueue()
        self.task = self.dag.task

    def setcontroller(self):
        self.controllerOri = []
        m = self.severNum
        n = self.controllerNum
        f = int(self.severNum / self.controllerNum)
        for _ in range(self.controllerNum):
#             f = random.randint(0,int(2*m/n))
#             f = random.randint(1, int(2 * m / n))
            self.remaincontroller.append(self.init_severs(f))
            self.controllerResources.append([f, f])
            self.controllerOri.append(f)
            m -= f
            n -= 1

        self.controllerOri.append(m)
        self.pwrPre = [0]*self.severNum #power usage pre sever
        self.pwrPcontroller = [0]*self.controllerNum #power usage per controller


    def elecPrice(self, t, pwr):
        """
        The energy cost on time t
        """
        threshold = 1.5
        if pwr < threshold:
            p = 5.91 #dynamic price
        else:
            p = 8.27
        return pwr * p

    def getPwr(self, r, c):
        # eq.2
        if r < c:
            pwrS = 1
        else:
            pwrS = 0
        alpha = 0.5 #alpha
        beta = 10 #beta
        Ur = (c-r)/c # eq.1
        if Ur < 0.7:
            pwrDy = alpha * Ur
        else:
            pwrDy = 0.7 * alpha + (Ur - 0.7)**2 * beta
        return pwrDy+pwrS

    def rewardFcn1(self):
        """
        Implement the reward function for each fog controller
        For stage 1: choose the fog controller
        """
        # eq.5
        pwrCcontroller = []
        for i in range(self.controllerNum):
            pwrc = self.getPwr(self.controllerResources[i][0], self.controllerOri[i])
            pwrCcontroller.append(pwrc)
        pwr = sum(pwrCcontroller) - sum(self.pwrPcontroller)
        self.pwrPcontroller = pwrCcontroller
        return self.elecPrice(1, pwr)


    def rewardFcn2(self):
        """
        Implement the reward function for each fog node
        For stage 2: choose the fog node
        """
        # eq.6
        pwrCur = []
        for f in self.remaincontroller:
            for s in f:
                sremain = 0
                for v in s:
                    sremain += v[0]
#                 print(sremain)    
                pwrc = self.getPwr(sremain, 1.0)
                if pwrc < 0:
                    print("here", sremain)
                pwrCur.append(pwrc)
#                 print("pwrc", pwrc)
        pwr = sum(pwrCur) - sum(self.pwrPre)
        self.totalcost += sum(pwrCur)
#         print("sum(pwrCur)", sum(pwrCur), "sum(self.pwrPre)", sum(self.pwrPre))
        self.pwrPre = pwrCur
#         print("pwr",pwr)
#         print("r2", self.elecPrice(1,pwr))
        return self.elecPrice(1, pwr)

    def release(self):
        """
        Randomly release resources from each VM
        And set the corresponding task as finished
        """
        rancontroller = random.randint(0, self.controllerNum-1)
        ranSer = random.randint(0, self.controllerOri[rancontroller]-1)
        ranVM = random.randint(0, self.VMNum-1)
        if self.VMtask[rancontroller][ranSer][ranVM]:
            random.shuffle(self.VMtask[rancontroller][ranSer][ranVM])
            t = self.VMtask[rancontroller][ranSer][ranVM].pop()
            t.status = 0
            self.remaincontroller[rancontroller][ranSer][ranVM][0] += float(t.CPU)
            self.remaincontroller[rancontroller][ranSer][ranVM][1] += float(t.RAM)

    def releaseByTime(self, controller_i, server_i, vm_j):
        curtime = time.time()
        for t in self.VMtask[controller_i][server_i][vm_j]:
            if t.endtime < curtime:
                t.status = 0
                self.remaincontroller[controller_i][server_i][vm_j][0] += float(t.CPU)
                self.remaincontroller[controller_i][server_i][vm_j][1] += float(t.RAM)
                self.controllerResources[controller_i][0] += float(t.CPU)
                self.controllerResources[controller_i][1] += float(t.RAM)
                self.VMtask[controller_i][server_i][vm_j].remove(t)

    def training(self):
        """
        Read the task file by readfile()
        Set the variables
        Pass tasks to agents in real time
        Get the corresponding reward value
        Reject task when R_cpu ≥ C_cpu or R_ram < C_ram
        """
        #send one tesk to dqn and calculate reward
        self.dag.initTask()
        self.generateQueue()
        time_start=time.time()
        print(self.controllerNum, end=' ')
        print(self.severNum, end=' ')
        print(self.num_task, end=' ')
        self.trainDDPG_v1()
        time_end=time.time()
        timecost = round(time_end-time_start, 3)
        print(timecost, end=' ')
        print(round(self.totalcost, 3), end=' ')
        print()

    def checkRej(self, controller_i, server_i, vm_j, task):
        """
        Check whether this task should be rejected in ith sever, jth VM
        Reject task when current time + task's runtime > task's ddl
        """
        import time
        if task.CPU > 1/self.VMNum or task.RAM > 1/self.VMNum:
            self.rej += 1 
            return -1
        remain_cpu = self.remaincontroller[controller_i][server_i][vm_j][0] - float(task.CPU)
        remain_ram = self.remaincontroller[controller_i][server_i][vm_j][1] - float(task.RAM)
        curtime = time.time()
        if curtime + task.runtime <= task.ddl:
            if remain_cpu >= 0 and remain_ram >=0:
                return 0  # do not reject
            else:
                return 1  # reject temporarily because cpu or ram
        else:
            self.rej += 1 
            return -1  #reject because ddl

    def UpdateServerState(self, tempServercontroller, tempSever, vm_numb, task):
        self.remaincontroller[tempServercontroller][tempSever][vm_numb][0] -= float(task.CPU)
        self.remaincontroller[tempServercontroller][tempSever][vm_numb][1] -= float(task.RAM)
        self.controllerResources[tempServercontroller][0]  -= float(task.CPU)
        self.controllerResources[tempServercontroller][1] -= float(task.RAM)
        return self.custom_reshape(self.remaincontroller)

    def custom_reshape(self, a):
        result = []
        for farNum in a:
            for serNum in farNum:
                result.append(serNum)
        c = np.array(result)
        d = c.reshape(2 * self.VMNum * self.severNum)
        return d

    def trainDDPG_v1(self):
        rej = 0
        self.setcontroller()
        energy = 0
        input_stage2 = input_stage1 = self.custom_reshape(self.remaincontroller)
        Agent_stage1 = Agent(actor_learning_rate=0.0001, input_dims=len(input_stage1),
                             n_actions=self.controllerNum)
        stage1_current_state = input_stage1
        # input_stage2 = np.array(self.remaincontroller[0]).reshape(2*self.VMNum*int(self.severNum/self.controllerNum))
        Agent_stage2 = Agent(actor_learning_rate=0.0001, input_dims=len(input_stage2),
                             n_actions=int(self.severNum/self.controllerNum))
        stage2_current_state = input_stage2
        acc = 0
        while len(self.task) != 0:
#             print(len(self.task))
            while len(self.task) != 0:
                for t in self.task:
                    if t.status == -1: #rejected
                        self.dag.updateStatus(t)
                        self.task.remove(t)
                    elif t.status == 1:   #ready 
                        f = stage1_action = Agent_stage1.get_action(stage1_current_state)
                        s = stage2_action = Agent_stage2.get_action(stage2_current_state)
                        vm = random.randint(0,self.VMNum-1)
                        # self.releaseByTime(f, s, vm)  # release by time
                        rej = self.checkRej(f, s, vm, t)
                        if rej == -1:  #rejected due to ddl
                            t.status = -1
                        # if not reject:
                        elif rej == 0:
                            t.endtime = time.time() + t.runtime
                            stage1_next_state = stage2_next_state = self.UpdateServerState(f, s, vm, t)
#                             print(self.remaincontroller)
                            reward_stage2 = self.rewardFcn2()
                            energy += reward_stage2
                            # Agent_stage2.learn(stage2_current_state, stage2_action, reward_stage2, stage2_next_state)
                            Agent_stage2.update(reward_stage2)
                            stage2_current_state = stage2_next_state
                            reward_stage1 = self.rewardFcn1()
                            # Agent_stage1.learn(stage1_current_state, stage1_action, reward_stage1, stage1_next_state)
                            Agent_stage1.update(reward_stage1)
                            stage1_current_state = stage1_next_state
                            self.VMtask[f][s][vm].append(t)
                            t.status = 2
#                             self.dag.updateStatus(t)
                            self.task.remove(t)
                            acc += 1
#                         else:
#                             t.status = -1
#                             rej += 1
            self.generateQueue()
        # print("total number of tasks: {0}, rejected tasks: {1}".format(len(self.task), rej))
        print(round(1 - acc/self.num_task, 3), end= ' ')

p1 = environment('small', 'output_5000.txt', 5000, 300)
p1.training()