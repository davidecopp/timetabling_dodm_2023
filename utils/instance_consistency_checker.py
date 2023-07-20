#!/usr/bin/env python3
from sys import stderr
import os
import argparse
from pathlib import Path

""" general supporting data (the various programs will resort only on parts of these):
    num_timeslots = tmax (int)
    examIDs (set)
    studIDs (set)
    studs_enrolled_to_exam (Dict<Set>)
    num_studs_enrolled_to_exam (Dict<int>)
    timeslot_of_exam (Dict<int>)     [NOT NEEDED]
    exams_at_timeslot (List<List>)   [NOT NEEDED]
"""

def check_consistency():
    for e in examIDs:
        if len(studs_enrolled_to_exam[e]) != num_studs_enrolled_to_exam[e]:
            print(f"Error: inconsistency detected! According to file {FILE_EXM} the number of students enrolled to exam {e} is {num_studs_enrolled_to_exam[e]}. However, the students enrolled according to file:\n    {FILE_STU}\n are actually these:\n{studs_enrolled_to_exam[e]}")
            exit(1)


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description="This util can be used to check the consistency of the instance, as based on comparing the information in the .exm and in the .stu input data files.", epilog="""-------------------""")

    # Required positional arguments
    parser.add_argument("slo_filename", type=str,
                        help="[string] second required positional argument - the filename of the file containing the length of the examination period.")
    parser.add_argument("stu_filename", type=str,
                        help="[string] third required positional argument - the filename of the file containing the list of the enrollments (i.e., ID_student - ID_exam pairs).")
    parser.add_argument("exm_filename", type=str,
                        help="[string] fourth required positional argument - the filename of the file summarizing the total number of student's subscriptions to each exam.")

    args = parser.parse_args()

    FILE_SLO = os.path.join(os.getcwd(), args.slo_filename)
    if not os.path.exists(FILE_SLO):
        if slo_filename[-4] != ".":
            if not os.path.exists(FILE_SLO+".slo"):
                print(f"Error: I could not find the file:\n    {FILE_SLO}\nThis file is needed (launch this script with the -h flag to know more about its usage!).")
                exit(1)
            else:
               FILE_SLO = FILE_SLO+".slo"
    if not os.path.isfile(FILE_SLO):
        print(f"Error: the following is not the path of a file:\n    {FILE_SLO}\nthis seems to be the path of a directory!).")
        exit(1)

    FILE_STU = os.path.join(os.getcwd(), args.stu_filename)
    if not os.path.exists(FILE_STU):
        if stu_filename[-4] != ".":
            if not os.path.exists(FILE_STU+".stu"):
                print(f"Error: I could not find the file:\n    {FILE_STU}\nThis file is needed (launch this script with the -h flag to know more about its usage!).")
                exit(1)
            else:
               FILE_STU = FILE_STU+".stu"
    if not os.path.isfile(FILE_STU):
        print(f"Error: the following is not the path of a file:\n    {FILE_STU}\nthis seems to be the path of a directory!).")
        exit(1)

    FILE_EXM = os.path.join(os.getcwd(), args.exm_filename)
    if not os.path.exists(FILE_EXM):
        if exm_filename[-4] != ".":
            if not os.path.exists(FILE_EXM+".exm"):
                print(f"Error: I could not find the file:\n    {FILE_EXM}\nThis file is needed (launch this script with the -h flag to know more about its usage!).")
                exit(1)
            else:
               FILE_EXM = FILE_EXM+".exm"
    if not os.path.isfile(FILE_EXM):
        print(f"Error: the following is not the path of a file:\n    {FILE_EXM}\nthis seems to be the path of a directory!).")
        exit(1)


    # BEGIN: READING THE CONTENT OF THE REQUIRED FILES:
    # required filenames: FILE_SLO, FILE_STU, FILE_EXM

    examIDs = set({})
    studIDs = set({})
    
    with open(FILE_SLO) as file_slo:
        num_timeslots = tmax = int(file_slo.readline())

    with open(FILE_STU) as file_stu:
        stu_lines = file_stu.readlines()
        #print(f"{stu_lines=}", file=stderr)
        studs_enrolled_to_exam = {}
        for line in stu_lines:
            ID_stud, ID_exam = line.strip().split()
            ID_exam = int(ID_exam)
            examIDs.add(ID_exam)
            studIDs.add(ID_stud)
            if ID_exam in studs_enrolled_to_exam:
                if ID_stud in studs_enrolled_to_exam[ID_exam]:
                    print(f"WARNING: double registration of student {ID_stud} to exam {ID_exam}!", file=stderr)
                else:
                    studs_enrolled_to_exam[ID_exam].add(ID_stud)
            else:
                studs_enrolled_to_exam[ID_exam] = {ID_stud}
    print(f"{examIDs=}", file=stderr)
    print(f"{studIDs=}", file=stderr)
    print(f"{studs_enrolled_to_exam=}", file=stderr)
    num_exams = len(studs_enrolled_to_exam)

    with open(FILE_EXM) as file_exm:
        exm_lines = file_exm.readlines()
        #print(f"{exm_lines=}", file=stderr)
        num_studs_enrolled_to_exam = {}
        for line in exm_lines:
            ID_exam, num_enrolls = map(int, line.strip().split())
            if ID_exam in num_studs_enrolled_to_exam:
                print(f"WARNING: double declaration of the number of enrollments to exam {ID_exam}!", file=stderr)
            num_studs_enrolled_to_exam[ID_exam] = num_enrolls
        print(f"{num_studs_enrolled_to_exam=}", file=stderr)

    # END: READING THE CONTENT OF THE REQUIRED FILES
    
    # BEGIN OF THE SMALL LOGIC SECTION:

    check_consistency()
    print("OK, the instance appears to be consistent (in that the data within file FILE_EXM{} and file {FILE_STU} do not come in contradiction)!")    
