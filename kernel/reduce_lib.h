#ifndef REDUCE_LIB_H
#define REDUCE_LIB_H

extern "C" int Reduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* reduced_node, int& new_num_nodes, int& new_num_edges, int* reduced_xadj, int* reduced_adjncy, int* reduced_mapping, int* reduced_reverse_mapping);

extern "C" void LocalSearch(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* init_mis, int* fin_mis);

//extern "C" int Reduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* reduced_node);

extern "C" int test();

#endif
