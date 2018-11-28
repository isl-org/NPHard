CXX = g++
CXXFLAGS = -Wall -O3 -std=c++11 -I./lib/mis -I./extern/KaHIP/lib -I./extern/KaHIP/lib/tools -I./lib/mis/evolutionary -I./lib -I./extern/KaHIP/lib/data_structure -I./lib/mis/initial_mis -I./extern/KaHIP/lib/partition -I./lib/mis/kernel -I./app -I./lib/mis/ils -fPIC -MMD

libreduce.so: modified.o bara.o ils.o local_search.o greedy_mis.o initial_mis.o population_mis.o random_mis.o random_functions.o greedy_vertex.o operation_log.o mis_permutation.o candidate_list.o bucket_array.o
	$(CXX) -shared $(CXXFLAGS) -I./extern/KaHIP/interface -o libreduce.so reduce_lib.cpp modified.o bara.o ils.o local_search.o greedy_mis.o initial_mis.o population_mis.o random_functions.o random_mis.o greedy_vertex.o operation_log.o mis_permutation.o candidate_list.o bucket_array.o

modified.o: ./lib/mis/kernel/modified.cpp
	$(CXX) $(CXXFLAGS) -c -o modified.o ./lib/mis/kernel/modified.cpp

bara.o: ./lib/mis/kernel/branch_and_reduce_algorithm.cpp
	$(CXX) $(CXXFLAGS) -c -o bara.o ./lib/mis/kernel/branch_and_reduce_algorithm.cpp

ils.o: ./lib/mis/ils/ils.cpp
	$(CXX) $(CXXFLAGS) -I./extern/KaHIP/interface -c -o ils.o ./lib/mis/ils/ils.cpp

local_search.o: ./lib/mis/ils/local_search.cpp
	$(CXX) $(CXXFLAGS) -c -o local_search.o ./lib/mis/ils/local_search.cpp

greedy_mis.o: ./lib/mis/initial_mis/greedy_mis.cpp
	$(CXX) $(CXXFLAGS) -c -o greedy_mis.o ./lib/mis/initial_mis/greedy_mis.cpp

initial_mis.o: ./lib/mis/initial_mis/initial_mis.cpp
	$(CXX) $(CXXFLAGS) -c -o initial_mis.o ./lib/mis/initial_mis/initial_mis.cpp

population_mis.o: ./lib/mis/evolutionary/population_mis.cpp
	$(CXX) $(CXXFLAGS) -I./extern/KaHIP/interface -c -o population_mis.o ./lib/mis/evolutionary/population_mis.cpp

random_functions.o: ./extern/KaHIP/lib/tools/random_functions.cpp
	$(CXX) $(CXXFLAGS) -c -o random_functions.o ./extern/KaHIP/lib/tools/random_functions.cpp

random_mis.o: ./lib/mis/initial_mis/random_mis.cpp
	$(CXX) $(CXXFLAGS) -c -o random_mis.o ./lib/mis/initial_mis/random_mis.cpp

greedy_vertex.o: ./lib/mis/initial_mis/greedy_vertex.cpp
	$(CXX) $(CXXFLAGS) -c -o greedy_vertex.o ./lib/mis/initial_mis/greedy_vertex.cpp

operation_log.o: ./lib/data_structure/operation_log.cpp
	$(CXX) $(CXXFLAGS) -c -o operation_log.o ./lib/data_structure/operation_log.cpp

mis_permutation.o: ./lib/data_structure/mis_permutation.cpp
	$(CXX) $(CXXFLAGS) -c -o mis_permutation.o ./lib/data_structure/mis_permutation.cpp

candidate_list.o: ./lib/data_structure/candidate_list.cpp
	$(CXX) $(CXXFLAGS) -c -o candidate_list.o ./lib/data_structure/candidate_list.cpp

bucket_array.o: ./lib/data_structure/priority_queues/bucket_array.cpp
	$(CXX) $(CXXFLAGS) -c -o bucket_array.o ./lib/data_structure/priority_queues/bucket_array.cpp
