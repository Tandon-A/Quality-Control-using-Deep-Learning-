"""
Python helper file to prepare train.txt. One for full network and one for each class. Using DAGM 2007 database.  
"""

import os 

#change path variable to location where data is stored 
path = "D:\\Abhishek_Tandon\\sop_scm\\data\\data\\"

#Open training files 
f = open("D:\\Abhishek_Tandon\\sop_scm\\main_data.txt","a")
fsub1 = open("D:\\Abhishek_Tandon\\sop_scm\\sub1.txt","a")
fsub2 = open("D:\\Abhishek_Tandon\\sop_scm\\sub2.txt","a")
fsub3 = open("D:\\Abhishek_Tandon\\sop_scm\\sub3.txt","a")
fsub4 = open("D:\\Abhishek_Tandon\\sop_scm\\sub4.txt","a")
fsub5 = open("D:\\Abhishek_Tandon\\sop_scm\\sub5.txt","a")
fsub6 = open("D:\\Abhishek_Tandon\\sop_scm\\sub6.txt","a")



files = []
files = os.listdir(path+"Class1\\")
for i in files:
    f.write(path+"Class1\\" + i)
    f.write(" 0\n")
    fsub1.write(path+"Class1\\" + i)
    fsub1.write(" 0\n")
    
files = os.listdir(path+"Class1_def\\")
for i in files:
    f.write(path+"Class1_def\\" + i)
	#in the main_data.txt each defective class will also be given the label of that class
    f.write(" 0\n") 
	#in the separate training files for each class, defective images will be labelled as 1. 
    fsub1.write(path+"Class1_def\\" + i)
    fsub1.write(" 1\n")    
    
    
files = os.listdir(path+"Class2\\")
for i in files:
    f.write(path+"Class2\\" + i)
    f.write(" 1\n")
    fsub2.write(path+"Class2\\" + i)
    fsub2.write(" 0\n") 
    
files = os.listdir(path+"Class2_def\\")
for i in files:
    f.write(path+"Class2_def\\" + i)
    f.write(" 1\n")
    fsub2.write(path+"Class2_def\\" + i)
    fsub2.write(" 1\n")
    
    
files = os.listdir(path+"Class3\\")
for i in files:
    f.write(path+"Class3\\" + i)
    f.write(" 2\n")
    fsub3.write(path+"Class3\\" + i)
    fsub3.write(" 0\n") 
    
files = os.listdir(path+"Class3_def\\")
for i in files:
    f.write(path+"Class3_def\\" + i)
    f.write(" 2\n")   
    fsub3.write(path+"Class3_def\\" + i)
    fsub3.write(" 1\n")    
    
    
files = os.listdir(path+"Class4\\")
for i in files:
    f.write(path+"Class4\\" + i)
    f.write(" 3\n")
    fsub4.write(path+"Class4\\" + i)
    fsub4.write(" 0\n")
    
files = os.listdir(path+"Class4_def\\")
for i in files:
    f.write(path+"Class4_def\\" + i)
    f.write(" 3\n")
    fsub4.write(path+"Class4_def\\" + i)
    fsub4.write(" 1\n")
    
    
files = os.listdir(path+"Class5\\")
for i in files:
    f.write(path+"Class5\\" + i)
    f.write(" 4\n")
    fsub5.write(path+"Class5\\" + i)
    fsub5.write(" 0\n")
    
files = os.listdir(path+"Class5_def\\")
for i in files:
    f.write(path+"Class5_def\\" + i)
    f.write(" 4\n")
    fsub5.write(path+"Class5_def\\" + i)
    fsub5.write(" 1\n")
    
    
files = os.listdir(path+"Class6\\")
for i in files:
    f.write(path+"Class6\\" + i)
    f.write(" 5\n")
    fsub6.write(path+"Class6\\" + i)
    fsub6.write(" 0\n")
    
files = os.listdir(path+"Class6_def\\")
for i in files:
    f.write(path+"Class6_def\\" + i)
    f.write(" 5\n")
    fsub6.write(path+"Class6_def\\" + i)
    fsub6.write(" 1\n")    
    
f.close()
fsub1.close()
fsub2.close()
fsub3.close()
fsub4.close()
fsub5.close()
fsub6.close()


