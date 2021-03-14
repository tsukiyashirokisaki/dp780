file=open("train.sh","w")
# feature=["Orient","BC","BS","MAD","Error","Bands","Phase"]
feature=["Phase","MAD","BC","BS","Bands","Error","Quaternion"]
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
temp=open("job_temp.sh","r")
file.write(temp.read()+"\n")
for i,ele in enumerate(ret):
	file.write("python3 train.py %s\n"%("_".join(ele)))	
file.write("sbatch_post.sh")
file.close()



