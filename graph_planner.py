import heapq
import argparse
from heapq import heappush, heappop
from IPython import embed
import numpy as np
from keras.models import load_model

def parse_arguments():
	parser = argparse.ArgumentParser(prefix_chars='@')
	parser.add_argument("@@model_path",type=str, default='model/model.hdf5')
	parser.add_argument("@@bins_a", type=int)
	parser.add_argument("@@bins_b", type=int)
	parser.add_argument("@@bins_action", type=int)
	parser.add_argument("@@range_a", type=float,nargs=2)
	parser.add_argument("@@range_b", type=float,nargs=2)
	parser.add_argument("@@range_action",type=float,nargs=2)
	parser.add_argument("@@start",type=float,nargs=2)
	parser.add_argument("@@end",type=float,nargs=2)




	
	args = parser.parse_args()
	return args


def build_graph(bins_a, bins_b, bins_action, range_a, range_b, range_action, model):
	states = []
	values_a = []
	values_b = []
	adj_list = []
	action_list = []

	for i in range(bins_a):
		values_a.append((range_a[0]+ i*(range_a[1] - range_a[0])/bins_a,range_a[0]+ (i+1)*(range_a[1] - range_a[0])/bins_a))

	for i in range(bins_b):
		values_b.append((range_b[0]+ i*(range_b[1] - range_b[0])/bins_b,range_b[0]+ (i+1)*(range_b[1] - range_b[0])/bins_b))

	for x in range(bins_a):
		for y in range(bins_b):
			states.append((values_a[x],values_b[y]))
			adj_list.append(set())

	for i in range(bins_action):
		action_list.append((range_action[0]+ i*(range_action[1] - range_action[0])/bins_action,range_action[0]+ (i + 1)*(range_action[1] - range_action[0])/bins_action ))

	for i in range(bins_a*bins_b):
		for action in action_list:
			theta = np.mean(states[i][0])
			cos = np.cos(theta)
			sin = np.sin(theta)
			theta_dot = np.mean(states[i][1])
			act = np.mean(action)
			x = [[cos,sin,theta_dot,act,theta]]
			s = model.predict(np.array(x))
			new_theta = s[0][3]*(range_a[1] - range_a[0]) + range_a[0]
			new_theta_dot = s[0][2]*(range_b[1] - range_b[0]) + range_b[0]
			new_theta = min(max(s[0][3],range_a[0]),range_a[1])
			new_theta_dot = min(max(s[0][2],range_b[0]),range_b[1])
			ii = min(np.floor((theta - range_a[0])*bins_a/(range_a[1] - range_a[0])).astype(int),bins_a - 1)
			jj = min(np.floor((theta_dot - range_b[0])*bins_b/(range_b[1] - range_b[0])).astype(int),bins_b - 1)
			ii_new = min(np.floor((new_theta - range_a[0])*bins_a/(range_a[1] - range_a[0])).astype(int),bins_a - 1)
			jj_new = min(np.floor((new_theta_dot - range_b[0])*bins_b/(range_b[1] - range_b[0])).astype(int),bins_b - 1)
			lst = [j for j,v in enumerate(adj_list[ii*bins_b + jj]) if v[1] == ii_new*bins_b + jj_new]
			if (not lst):
				adj_list[ii*bins_b + jj].add((-new_theta**2 - new_theta_dot**2 - 0.001*(act**2), ii_new*bins_b + jj_new, act))

	return adj_list


def a_star(adj_list, start_state, end_state):
	open_list = []
	cost = dict()
	parent = dict()
	cost[start_state] = 0
	parent[start_state] = None
	heappush(open_list,(0,(0,start_state,0)))
	while(open_list):
		curr = heappop(open_list)
		if(curr[1][1] == end_state):
			break
		for neighbor in adj_list[curr[1][1]]:
			cost_nb = cost[curr[1][1]] + neighbor[0] + 1
			if neighbor not in cost or cost_nb < cost[neighbor[1]]:
				cost[neighbor[1]] = cost_nb
				heappush(open_list,(cost_nb, neighbor))
				parent[neighbor] = cost

	path = []
	while(curr):
		path.append(curr[1][2])
		curr = parent[curr]
	curr.reverse()

	return curr






def main():
	args = parse_arguments()
	model = load_model(args.model_path)

	adj_list = build_graph(args.bins_a, args.bins_b, args.bins_action, args.range_a, args.range_b,args.range_action, model)

	start_ii = min(np.floor((args.start[0] - args.range_a[0])*args.bins_a/(args.range_a[1] - args.range_a[0])).astype(int), args.bins_a - 1)
	start_jj = min(np.floor((args.start[1] - args.range_b[0])*args.bins_b/(args.range_b[1] - args.range_b[0])).astype(int), args.bins_b - 1)
	end_ii = min(np.floor((args.end[0] - args.range_a[0])*args.bins_a/(args.range_a[1] - args.range_a[0])).astype(int), args.bins_a - 1)
	end_jj = min(np.floor((args.end[1] - args.range_b[0])*args.bins_b/(args.range_b[1] - args.range_b[0])).astype(int), args.bins_b - 1)

	start_state = start_ii*args.bins_b + start_jj
	end_state = end_ii*args.bins_b + end_jj
	actions = a_star(adj_list, start_state, end_state)
	for action in actions:
		print(action)



if __name__ == "__main__":
	main()



