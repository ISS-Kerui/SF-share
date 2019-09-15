import math
from sklearn import datasets
from collections import defaultdict
import random
from random import uniform
def points_avg(points):
	if points == []:
		return None
	dimensions = len(points[0])
	new_center = []
	for dimension in range(dimensions):
		dim_sum = 0
		for p in points:
			dim_sum += p[dimension]
		new_center.append(dim_sum / float(len(points)))
	return new_center

def update_centroids(dataset, assigments):
	new_means = defaultdict(list)
	centroids = []
	for assigment, point in zip(assigments,dataset):
		new_means[assigment].append(point)
	for points in new_means.values():
		centroids.append(points_avg(points))
	return centroids

def euler_distance(point1, point2):
	distance = 0.0
	for a, b in zip(point1, point2):
		distance += math.pow(a-b, 2)

	return math.sqrt(distance)


def get_closest_dist(point, centroids):
	min_dist = math.inf 
	min_index = -1
	for i, centroid in enumerate(centroids):
		dist = euler_distance(point,centroid)
		if dist < min_dist:
			min_dist = dist
			min_index = i
	return min_dist,min_index

def kmeansInit(dataset, k):
	cluster_centers = []
	cluster_centers.append(random.choice(dataset))
	d = [0 for _ in range(len(dataset))]
	for _ in range(1,k):
		total = 0.0
		for i, point in enumerate(dataset):
			d[i],_ = get_closest_dist(point,cluster_centers)
			total += d[i]
		total *= random.random()
		for i ,di in enumerate(d):
			total -= di
			if total > 0:
				continue
			cluster_centers.append(dataset[i])
			break
	return cluster_centers

def randomInit(datasets, k):
	dimensions = len(datasets[0])
	min_max = defaultdict(int)
	centers = []

	for point in datasets:
		for i in range(dimensions):
			val = point[i]
			min_key = 'min_%d'%(i)
			max_key = 'max_%d'%(i)
			if min_key not in min_max or val < min_max[min_key]:
				min_max[min_key] = val
			if max_key not in min_max or val > min_max[max_key]:
				min_max[max_key] = val

	for _k in range(k):
		rand_point = []
		for i in range(dimensions):
			#print (min_max)
			min_val = min_max['min_%d'%(i)]
			max_val = min_max['max_%d'%(i)]
			rand_point.append(uniform(min_val,max_val))
		centers.append(rand_point)
	return centers
	
def assign_points(data_points,centroids):
	assigments = []
	for point in data_points:	
		min_dist, min_index = get_closest_dist(point,centroids)
		assigments.append(min_index)
	return assigments

def k_means(dataset, k):
	centroids =  kmeansInit(dataset,k) #kmeans++
	#centroids = randomInit(dataset,k) #kmeans
	assigments = assign_points(dataset,centroids)
	old_assignments = None
	times = 0
	while assigments != old_assignments:
		times += 1
		print ('time is :',times)
		new_centers = update_centroids(dataset,assigments)
		old_assignments = assigments
		assigments = assign_points(dataset,new_centers)
	return (assigments,dataset)


if __name__ == '__main__':
	iris = datasets.load_iris()
	print (k_means(iris.data,4))
