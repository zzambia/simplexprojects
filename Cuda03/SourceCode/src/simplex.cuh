/*
 * simplex.cuh
 *
 *  Created on: 11-Apr-2015
 *      Author: bilzcuda
 */

#ifndef SIMPLEX_CUH_
#define SIMPLEX_CUH_
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>
#include <malloc.h>
#include <vector>
#include <list>
#include "math/matrix.h"
#include <climits>


//interface is   Simplex(A<B<C,N_S,No_O,No_C)


struct block_lp{
	math::matrix<float> block_obj_coeff;
};
struct block_lp_result{
	std::vector<float> results;
};



class Simplex {
private:
	unsigned int number_of_LPs;	//total number of LPs to be solved per instance
	math::matrix<double> orig_CoefficientMatrix;
	std::vector<double> BoundValue;
	//unsigned int number_constraints; //can be obtained from orig_CoefficientMatrix.size1();
	math::matrix<float> C;
	unsigned int number_of_Constraint;
	bool single_bound_flag_result;
public:
	//math::matrix<float> A, C;
	//std::vector<float> B;
	float *bound_result, *d_bound_result,  *d_obj_coeff, *obj_coeff;
	int *Sel, *G_Sel;
	float a;
	int M, N, i, j;
	float *MAT, *G_MAT, *G_R, *R, *N_MAT;

	int NB, f, c, No_c;
	__host__ Simplex(unsigned int N_S);
//get status of particular simplex
	__host__ int getStatus(int n); //get status of particular simplex

//get the No of simplex the object is ruuning on GPU
	__host__ int getNo_OF_Simplx(); //get the No of simplex the object is ruuning on GPU

//get the result of all simplex
	__host__ std::vector<float> getResultAll();

//get the result of all simplex

	__host__ float getResult(int n);	// get result of particular simplex

	__host__ std::vector<int> getStatusAll();	//get the status of all simplex

	__host__ void setConstratint(math::matrix<double> A1, std::vector<double> B1);							//setting constraints of simplex
	//__host__ void ComputeLP(math::matrix<float> &C1,unsigned int number_of_streams);
	__host__ void ComputeLP(math::matrix<float> &C1);

	/*
		 * Computes the entire list of LPs by diving into different blocks
		 */
	std::vector<float> bulkSolver(math::matrix<float> &List_of_ObjValue);


	~Simplex() {
	}

// Compute the LP. argument is Objective function(S)
};

#endif /* SIMPLEX_CUH_ */
