import gurobipy as gp
import numpy as np
import pandas as pd
import argparse

def do_enrol_matrix(instance):
    enrol_matrix = pd.read_table("instances/" + instance + ".stu", sep=" ", names=["STUDENTE", "ESAME"])
    enrol_matrix = np.array(pd.crosstab(enrol_matrix['STUDENTE'], enrol_matrix['ESAME']).astype(bool).astype(int))    
    n_students = np.shape(enrol_matrix)[0]
    n_exams = np.shape(enrol_matrix)[1]
    return enrol_matrix, n_students, n_exams 

def do_conflict_matrix(enrol_matrix):
    n_exams = np.shape(enrol_matrix)[1]
    conflict_matrix = np.zeros((n_exams, n_exams))
    for exam_1 in range(n_exams):
        for exam_2 in range(exam_1 + 1, n_exams):
            conflict_matrix[exam_1,exam_2] = np.sum([stud[exam_1]*stud[exam_2] for stud in enrol_matrix])
    return conflict_matrix

def create_model(model_name='ExamTimetabling', time_limit=50, pre_solve=-1, mip_gap=1e-4, threads=4):
    model = gp.Model(model_name)
    model.setParam('TimeLimit', time_limit)
    model.setParam("Presolve", pre_solve)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Threads", threads)
    model.setParam("NodefileStart", 0.5)
    return model

def do_variables(n_exams, n_timeslots, model):
    x = {}
    y = {}
    z = {}
    for timeslot in range(n_timeslots):
        z[timeslot] =  model.addVar(vtype=gp.GRB.BINARY, name=f'z[{timeslot}]')
        for exam in range(n_exams):
            x[exam, timeslot] = model.addVar(vtype=gp.GRB.BINARY, name=f'x[{exam},{timeslot}]')
            for exam_2 in range(exam+1, n_exams):
                y[exam, exam_2, timeslot] = model.addVar(vtype=gp.GRB.INTEGER, name=f'extra_var[{exam},{exam_2},{timeslot}]')
    return x, y, z

def do_obj_function(measure, n_exams, n_students, n_timeslots, conflict_matrix, x):
    obj_function = 0
    
    if measure == "penalty":
        for exam_1 in range(n_exams):
            for exam_2 in range(exam_1+1,n_exams):
                if conflict_matrix[exam_1,exam_2] > 0:
                    for timeslot_1 in range(n_timeslots):
                        for timeslot_2 in range(max(0, timeslot_1 - 5), min(timeslot_1 + 6, n_timeslots)):
                            obj_function += 2**(5 - abs(timeslot_1 - timeslot_2))*conflict_matrix[exam_1,exam_2]/n_students*x[exam_1, timeslot_1]*x[exam_2, timeslot_2] 
    
    elif measure == "b2b":
        for timeslot in range(n_timeslots-1):
            for exam_1 in range(n_exams):
                for exam_2 in range(exam_1+1, n_exams):
                    obj_function += conflict_matrix[exam_1, exam_2]*(x[exam_2, timeslot+1]*x[exam_1, timeslot] + x[exam_2, timeslot]*x[exam_1, timeslot+1])
                                
    else:
        print("This measure is not implemented. Type 'penalty' or 'b2b_students'.")

    return obj_function 

def add_constraints(model_type, n_exams, n_timeslots, conflict_matrix, model):

    if model_type == "base":
        # CONSTRAINT 0: Each exam is scheduled exactly once
        for exam in range(n_exams):
            model.addConstr(gp.quicksum(x[exam, timeslot] for timeslot in range(n_timeslots)) == 1)

        # CONSTRAINT 1: Conflicting exams cannot be scheduled in the same time-slot
        for exam_1 in range(n_exams):
            for exam_2 in range(exam_1+1, n_exams):
                if conflict_matrix[exam_1,exam_2] > 0:
                    for timeslot in range(n_timeslots):
                        model.addConstr(x[exam_1, timeslot] + x[exam_2, timeslot] <= 1)
    
    elif model_type == "advanced":
        # CONSTRAINT 0: Each exam is scheduled exactly once
        for exam in range(n_exams):
            model.addConstr(gp.quicksum(x[exam, timeslot] for timeslot in range(n_timeslots)) == 1)

        # CONSTRAINT 1: At most 3 consecutive time slots can have conflicting exams
        for timeslot in range(n_timeslots):
            model.addConstr(gp.quicksum(x[exam_1, timeslot]*x[exam_2, timeslot] for exam_1 in range(n_exams) for exam_2 in range(exam_1+1, n_exams) if conflict_matrix[exam_1, exam_2]>0) <= 1000*z[timeslot])
            model.addConstr(z[timeslot] <= gp.quicksum(x[exam_1, timeslot]*x[exam_2, timeslot] for exam_1 in range(n_exams) for exam_2 in range(exam_1+1, n_exams) if conflict_matrix[exam_1, exam_2]>0))
        
        for timeslot in range(n_timeslots-3):    
            for exam in range(n_exams):
                model.addConstr(5-gp.quicksum(z[timeslot+i] for i in range(3)) >= 3*x[exam, timeslot+3])
                
        for timeslot in range(1, n_timeslots-3):    
            for exam in range(n_exams):
                model.addConstr(5-gp.quicksum(z[timeslot+i] for i in range(3)) >= 3*x[exam, timeslot-1])

        # CONSTRAINT 2: If two consecutive time slots contain conflicting exams, then no conflicting exam can be scheduled in the next 3 time slots
        for timeslot in range(n_timeslots-4):
            for exam_1 in range(n_exams):
                for exam_2 in range(exam_1+1, n_exams):
                    if conflict_matrix[exam_1, exam_2]>0:
                        model.addConstr(y[exam_1, exam_2, timeslot] == x[exam_1, timeslot]*x[exam_2, timeslot+1] + x[exam_2, timeslot]*x[exam_1, timeslot+1])
                        model.addConstr(gp.quicksum(x[exam, t]*y[exam_1, exam_2, timeslot] for t in range(timeslot+2, timeslot+5) for exam in range(n_exams) if exam!=exam_1 and exam!=exam_2 and (conflict_matrix[exam, exam_1]>0 or conflict_matrix[exam, exam_2])>0) == 0)

        # CONSTRAINT 3: Change the constraints that impose that no conflicting exams can be scheduled in the same time slot. 
        #               Instead, impose that at most 3 conflicting pairs can be scheduled in the same time slot.
        for timeslot in range(n_timeslots):
            model.addConstr(gp.quicksum(x[exam_1, timeslot] * x[exam_2, timeslot] for exam_1 in range(n_exams) for exam_2 in range(exam_1+1, n_exams) if conflict_matrix[exam_1, exam_2] > 0) <= 3)

    else:
        print("This type of model is not valid. Type 'base' or 'advanced'.")
                            
    return 0

def write_solution(folder, instance, model, x, n_exams, n_timeslots):
    file = open("solutions/" + folder + "/" + instance + ".sol", "w")  # 'w' for write mode

    # Print the solution
    if model.status == gp.GRB.INFEASIBLE:
        file.write("INFEASIBLE")
    elif model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        for exam in range(n_exams):
            for timeslot in range(n_timeslots):
                if x[exam, timeslot].x > 0.5:
                    file.write(f'{str(exam+1).zfill(4)} {timeslot+1}\n')                
        
    file.close()
    
    return 0
############################################

if __name__ == "__main__":

    # Instantiate the parser
    parser = argparse.ArgumentParser(description="This util can be used to check the feasibility of a solution you have produced for a certain instance. Please, provide the solution file and the two input data files required (the .stu and the .slo files).", epilog="""-------------------""")

    # Required positional arguments
    parser.add_argument("instance", type=str,
                        help="[string] required argument - name of the instance ('test' or 'instanceXX').")
    parser.add_argument("measure", type=str,
                        help="[string] required argument - name of the objective function ('penalty' or 'b2b_students').")
    parser.add_argument("model_type", type=str,
                        help="[string] required argument - type of the model ('base' or 'advanced').")
    args = parser.parse_args()
    
    instance = args.instance
    measure = args.measure
    model_type = args.model_type

    enrol_matrix, n_students, n_exams = do_enrol_matrix(instance)
    conflict_matrix = do_conflict_matrix(enrol_matrix)
    n_timeslots = int(list(pd.read_table("instances/" + instance + ".slo").columns)[0])

    model = create_model(model_name='ExamTimetabling', time_limit=1000, pre_solve=-1, mip_gap=1e-4, threads=4) # Create the model
    x, y, z = do_variables(n_exams, n_timeslots, model) # Create decision variables
    obj_function = do_obj_function(measure, n_exams, n_students, n_timeslots, conflict_matrix, x) # Set objective function 
    model.setObjective(obj_function, gp.GRB.MINIMIZE)
    add_constraints(model_type, n_exams, n_timeslots, conflict_matrix, model) # Add constraints
    model.optimize() # Optimize the model

    write_solution(model_type + "_" + measure, instance, model, x, n_exams, n_timeslots)

####################################################
