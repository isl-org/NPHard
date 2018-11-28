#include "reduce_lib.h"
#include "branch_and_reduce_algorithm.h"
#include "data_structure/graph_access.h"
#include "mis_config.h"
#include "population_mis.h"
#include "configuration_mis.h"
#include "ils.h"

#include <algorithm>
/*
int Reduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* reduced_node)
{
	std::vector<std::vector<int>> adj(num_nodes);
	//std::cout << num_nodes << " " << num_edges << std::endl;
	for (int i=0;i<num_edges;++i)
	{
		// std::cout << edges_from[i] << " " << edges_to[i] << std::endl;
		adj[edges_from[i]].push_back(edges_to[i]);
	}

	//full_reducer = std::unique_ptr<branch_and_reduce_algorithm>(new branch_and_reduce_algorithm(adj, adj.size()));
	branch_and_reduce_algorithm* full_reducer = new branch_and_reduce_algorithm(adj, adj.size());

    // perform reduction
    full_reducer->reduce_graph();

	for (int i=0;i<num_nodes;++i)
		reduced_node[i]=full_reducer->x[i];
	return full_reducer->get_current_is_size();
}
*/

int Reduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* reduced_node, int& new_num_nodes, int& new_num_edges, int* reduced_xadj, int* reduced_adjncy, int* reduced_mapping, int* reduced_reverse_mapping)
{
	std::vector<std::vector<int>> adj(num_nodes);
	//std::cout << num_nodes << " " << num_edges << std::endl;
	for (int i=0;i<num_edges;++i)
	{
		// std::cout << edges_from[i] << " " << edges_to[i] << std::endl;
		adj[edges_from[i]].push_back(edges_to[i]);
	}

	//full_reducer = std::unique_ptr<branch_and_reduce_algorithm>(new branch_and_reduce_algorithm(adj, adj.size()));
	branch_and_reduce_algorithm* full_reducer = new branch_and_reduce_algorithm(adj, adj.size());

    // perform reduction
    full_reducer->reduce_graph();
	
	////////////////////////////////// new adj and mapping //////////////////////////////
	std::vector<int> x(full_reducer->x);
	std::vector<std::vector<int>> new_adj(full_reducer->adj);
	// Number of nodes
    unsigned int const node_count = full_reducer->number_of_nodes_remaining();
    // Number of edges
    int m = 0;
	// Nodes -> Range
    std::vector<NodeID> mapping(new_adj.size(), UINT_MAX);
	std::vector<NodeID> reverse_mapping(new_adj.size(), 0);
    // Get number of edges and reorder nodes
    unsigned int node_counter = 0;
    for (NodeID node = 0; node < new_adj.size(); ++node) if (x[node] < 0) {
        for (int const neighbor : new_adj[node]) if (x[neighbor] < 0) m++;
        mapping[node] = node_counter;
        reverse_mapping[node_counter] = node;
        node_counter++;
    }
    // Create the adjacency array
    std::vector<int> xadj(node_count + 1);
    std::vector<int> adjncy(m);
    unsigned int adjncy_counter = 0;
    for (unsigned int i = 0; i < node_count; ++i) {
        xadj[i] = adjncy_counter;
        for (int const neighbor : new_adj[reverse_mapping[i]]) {
            if (mapping[neighbor] == i) continue;
            if (mapping[neighbor] == UINT_MAX) continue;
            adjncy[adjncy_counter++] = mapping[neighbor];
        }
        std::sort(std::begin(adjncy) + xadj[i], std::begin(adjncy) + adjncy_counter);
    }
    xadj[node_count] = adjncy_counter;
	////////////////////////////////// new adj and mapping //////////////////////////////

	// return values

	new_num_nodes = node_count;
	new_num_edges = m;

	for (unsigned int i=0;i<(node_count+1);++i)
		reduced_xadj[i]=xadj[i];

	for (int i=0;i<m;++i)
		reduced_adjncy[i]=adjncy[i];

	for (int i=0;i<num_nodes;++i)
	{
		reduced_mapping[i]=mapping[i];
		reduced_reverse_mapping[i]=reverse_mapping[i];
		reduced_node[i]=x[i];
	}

	return full_reducer->get_current_is_size();
}


void LocalSearch(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* init_mis, int* fin_mis)
{

	// Configurations
	MISConfig mis_config;
    configuration_mis cfg;
    cfg.standard(mis_config);

	// Build the adjacency matrix
	std::vector<std::vector<int>> adj(num_nodes);
	// std::cout << num_nodes << " " << num_edges << std::endl;
	for (int i=0;i<num_edges;++i)
	{
		// std::cout << edges_from[i] << " " << edges_to[i] << std::endl;
		adj[edges_from[i]].push_back(edges_to[i]);
	}

	// Build the adjacency array
    std::vector<int> xadj(num_nodes + 1);
    std::vector<int> adjncy(num_edges);
    unsigned int adjncy_counter = 0;
    for (unsigned int i = 0; i < num_nodes; ++i) {
        xadj[i] = adjncy_counter;
        for (int const neighbor : adj[i]) {
            if (neighbor == i) continue;
            if (neighbor == UINT_MAX) continue;
            adjncy[adjncy_counter++] = neighbor;
        }
        std::sort(std::begin(adjncy) + xadj[i], std::begin(adjncy) + adjncy_counter);
    }
    xadj[num_nodes] = adjncy_counter;

	// Construct graph_access data structure
	graph_access G;
	G.build_from_metis(num_nodes, &xadj[0], &adjncy[0]);

	// Initial indepedent set
	forall_nodes(G, node) {
        G.setPartitionIndex(node, init_mis[node]);
		// std::cout << init_mis[node] << std::endl;
    } endfor

	// Apply ILS
	mis_config.ils_iterations = std::min(G.number_of_nodes(), mis_config.ils_iterations);
	//mis_config.ils_iterations = 1;

    ils iterate;
    iterate.perform_ils(mis_config, G, mis_config.ils_iterations);

	// Create individuum for final independent set
    individuum_mis final_mis;
	population_mis island;
	island.init(mis_config, G);
    NodeID *solution = new NodeID[G.number_of_nodes()];
    final_mis.solution_size = island.create_solution(G, solution);
    final_mis.solution = solution;
    island.set_mis_for_individuum(mis_config, G, final_mis);
    forall_nodes(G, node) {
        init_mis[node] = final_mis.solution[node];
		fin_mis[node] = final_mis.solution[node];
    } endfor

	delete[] solution;
    solution = NULL;
}

int test()
{
	std::cout << "Hello" << std::endl;
	return 1;
}
