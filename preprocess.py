#!/usr/bin/python
import numpy

A="1,0,0,0"
T="0,1,0,0"
C="0,0,1,0"
G="0,0,0,1"
fi = open("test.csv", "r")
fo = open("test_processed.csv","w")
seq=fi.read()
seq=seq.strip().split('\n')
seq=seq[1:]
for line in seq:
	#print line
	line=line.strip()
	line=line.split(",")
	#print line	
	#label=line[2]
	sequence=line[1]
	hot_enc=[]
	for i in sequence:
		if i=="A":
			hot_enc.append(A)
		if i=="T":
			hot_enc.append(T)
		if i=="C":
			hot_enc.append(C)
		if i=="G":
			hot_enc.append(G)
		'''
	if label=="1":
		label="0,1"
	else:
		label="1,0"
		'''
	#hot_enc.append(label)
	hot_enc=",".join(hot_enc)+"\n"
	print hot_enc
	fo.write(hot_enc)
fo.close()
