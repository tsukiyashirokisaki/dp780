file=open("train.sh","w")
feature=["Orient","BC","BS","MAD","Error","Bands","Phase"]
ret=[]
def choose(sel,feature,num):
	global ret
	if num==0:
		ret.append(sel)
		return
	elif num==len(feature):
		ret.append(sel+feature)
		return 
	elif len(feature)>num:
		choose(sel,feature[1:],num)
		choose(sel+[feature[0]],feature[1:],num-1)
	else:
		return
for i in range(1,len(feature)+1):
	choose([],feature,i)
for i,ele in enumerate(ret):
	file.write("python classify3.py %s &\n"%("_".join(ele)))
	file.write("python classify5.py %s &\n"%("_".join(ele)))
	if i%3==2:
		file.write("wait\n")

file.close()



