import numpy as np
import random
import os
import time


class cluster_data(object):
	def __init__(self,num_clust):
		
		self.num_clust=num_clust
		self.centroids=[]
		self.assignments={}
		self.assignments1={}
		self.assignments2={}
		
	def k_means_clust(self,data,num_iter,progress=False):
		
		self.centroids=random.sample(data,self.num_clust)
		
		
		for n in range(num_iter):
			if progress:
				print 'iteration '+str(n+1)
	        #assign data points to clusters
		        self.assignments={}
			for ind,i in enumerate(data):
				min_dist=float('inf')
				closest_clust=None
				for c_ind,j in enumerate(self.centroids):
					if self.LB_Keogh(i,j,5)<min_dist:
						cur_dist=self.DTWDistance(i,j)
						if cur_dist<min_dist:
							min_dist=cur_dist
							closest_clust=c_ind
				self.assignments1[ind]=closest_clust
							
				if closest_clust in self.assignments:
					self.assignments[closest_clust].append(ind)
				else:
					self.assignments[closest_clust]=[ind]
					
			        t1=time.asctime(time.localtime(time.time()))
				print "time",t1
                        
			a=0
			for key1 in self.assignments:
                                if self.assignments.get(key1) == self.assignments2.get(key1):
                                   a=a+1
                        if a==13:
                                return
                                   
			#recalculate centroids of clusters	
			for key in self.assignments:
                            clust_sum=0
                            for k in self.assignments[key]:
                                clust_sum=clust_sum+data[k]
                            self.centroids[key] = clust_sum/len(self.assignments[key])
                            
                        self.assignments2=self.assignments
                            

        def writePrediction(self):
                file = open('prediction2.txt', 'w')
                file.write("Asset,Cluster")
                file.write("\n")
                print self.assignments1
                for i in self.assignments1:
                        line="X"+str(i+1)+","+str(self.assignments1[i]+1)
                        file.write(line+"\n")                

	def DTWDistance(self,s1,s2):
		
		DTW={}
    
		for i in range(len(s1)):
		        DTW[(i, -1)] = float('inf')
		for i in range(len(s2)):
		        DTW[(-1, i)] = float('inf')
		
		DTW[(-1, -1)] = 0
	
		for i in range(len(s1)):
			
			for j in range(len(s2)):
				dist= (s1[i]-s2[j])**2
				DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
			
		return np.sqrt(DTW[len(s1)-1, len(s2)-1])
	   
	def LB_Keogh(self,s1,s2,r):
		
		LB_sum=0
		for ind,i in enumerate(s1):
	        
			lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
			upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
	        
			if i>upper_bound:
				LB_sum=LB_sum+(i-upper_bound)**2
			elif i<lower_bound:
				LB_sum=LB_sum+(i-lower_bound)**2
	    
		return np.sqrt(LB_sum)  


def main():
    
   cluster = cluster_data(13)
   train = np.genfromtxt('test.csv', delimiter=',')
   data=np.column_stack(train)
   data1=data[1:101,1:1362]
   cluster.k_means_clust(data1,4,True)
   cluster.writePrediction()
   

if __name__ == '__main__':
    main()
