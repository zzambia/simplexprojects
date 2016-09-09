/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include "simplex.cuh"
#include "math/glpk_lp_solver/glpk_lp_solver.h"
#include <vector>
#include "math/matrix.h"
#include <climits>
#include <iostream>
#include "boost/timer/timer.hpp"
#include "sys/time.h"

int main(int argc, char *argv[]) {
	unsigned int LP_size = 1, avg, dimension, art, stream;
	math::matrix<double> A;
	math::matrix<float> C, newC;
	std::vector<double> b;
	std::vector<float> result;
	std::vector<int> status_val;
	double Final_Time;

	if (argc > 1) {
		if (argc != 5) {		//1(ApplicationName) + 4 (Input Arguments)
			std::cout << "\nInsufficient Number of Arguments!!!\n";
			std::cout << "Correct Usages/Syntax:\n";
			std::cout
					<< "./ProjName --'Dimension'--'Average'--'Batch-size' ---'Artificial-Variables'!!\n";
			std::cout
					<< "argument 1) Dimension -- select the dimension of LP to be solved\n";
			std::cout
					<< "argument 2) Average -- select the number of runs for average readings\n";
			std::cout
					<< "argument 3) Batch-size -- select the number of LPs to be solved\n";
			std::cout
					<< "argument 4) Artificial-Variables -- Number of negative on RHS of the LP constraints\n";
			//std::cout<< "argument 5) Streams -- 0 for no streaming, 1 or 10 or 'n' to select the number of streams \n";
			return 0;
		} else {
			unsigned int num;
			dimension = atoi(argv[1]);
			num = atoi(argv[2]);
			avg = num;
			LP_size = atoi(argv[3]);
			art = atoi(argv[4]);
			//stream = atoi(argv[5]);
		}
	}
	glpk_lp_solver lp;
	unsigned int status = 0, res1 = 0, res2 = 0;
	//**** Creating Random Lp's Any x Any ****
	int sign = -1, sum1;

	while (status != 5) {
		A.resize(dimension, dimension);
		b.resize(dimension);
		for (unsigned int j = 0; j < dimension; j++) {
			for (unsigned int k = 0; k < dimension; k++) {
				A(j, k) = rand() % (k + 10) + 1;
			}

			if (j < art) {
				res1 = (rand() % 50);
				res2 = (res1 + 1);
				sum1 = res2 * sign;
				b[j] = (double) sum1;
				A(j, j) = A(j, j) * sign;
			} else
				b[j] = (rand() % 50) + 1;
		}
		//** Setting current Lp to GLPK
		lp.setMin_Or_Max(2);
		lp.setConstraints(A, b, 1);
		status = lp.TestConstraints();
	}
	/*
	 * Experimental Note:
	 *  We generate randomly the first LP problem (the matrix A and vector b) and then generate
	 *  randomly the objective function(s). For the ease generation process we take all the
	 *  LP problems to be the same LP generated above. However, for each of this LPs we generate
	 *  randomly different objective functions.
	 *
	 *  Also, to record the average time we ignore the first reading taken in GPU as it does not
	 *  reflect the correct computation time because for the first GPU call it also include an
	 *  extra overhead of GPU initialization time.
	 */


	double sum = 0.0;
	double wall_clock, user_clock, system_clock, return_Time;
	boost::timer::cpu_timer tt1, tt2;	//tt1 -- Variable declaration

	C.resize(LP_size, dimension);	//objective function
	for (unsigned int i = 0; i < C.size1(); i++) {
		for (unsigned int j = 0; j < C.size2(); j++) {
			C(i, j) = rand() % (j + 1) + 1;
		}
	}

	std::cout << "\n*****GPU RESULT*****\n";

	for (unsigned int i = 0; i <= avg; i++) {
		Simplex s(C.size1());
		s.setConstratint(A, b);	//setting constraints also recorded like in GLPK for independent LP
		tt1.start();
		s.ComputeLP(C);
		tt1.stop();
		if (i == 0) {
			wall_clock = tt1.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
			return_Time = wall_clock / (double) 1000; //convert milliseconds to seconds
			std::cout << "Iter = " << i << " Time = " << return_Time
					<< std::endl;
		} else {
			wall_clock = tt1.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
			return_Time = wall_clock / (double) 1000; //convert milliseconds to seconds
			std::cout << "Iter = " << i << " Time = " << return_Time
					<< std::endl;
			sum = sum + return_Time; //convert nanoseconds to milliseconds
		}
		result = s.getResultAll();
	}
	Final_Time = sum / avg;
	std::cout << "\nNumber of Simplex Solved = " << C.size1() << std::endl;
	std::cout << "\nBoost Time taken:Wall  (in Seconds):: GPU= "
			<< (double) Final_Time << std::endl;
	std::cout << "\n**Answer_Of_All_Simplex**\n";

	//Computation for CPU ie GLPK
	//	sum=0.0;
	std::cout << "\n*****GLPK RESULT*****\n";
	std::vector<double> dir(dimension);

	//***** MODEL SELECTION *****
	double res = 0.0;
	double batchTime = 0.0, AvgBatchTime = 0.0;
	std::vector<double> resul(C.size1());
	for (int i = 1; i <= avg; i++) {
		//batchTime = 0.0;
		tt1.start();
		for (int i = 0; i < C.size1(); i++) {
			glpk_lp_solver mylp;
			mylp.setMin_Or_Max(2);
			for (int j = 0; j < dimension; j++) {
				dir[j] = C(i, j);
			}
			mylp.setConstraints(A, b, 1); //this function actually determines independent LP in GLPK
			res = mylp.Compute_LLP(dir); //We consider every dir an independent LP problem
			resul[i] = res;
		}
		tt1.stop();
		wall_clock = tt1.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
		return_Time = wall_clock / (double) 1000; //convert milliseconds to seconds
		batchTime = return_Time; //convert nanoseconds to milliseconds
		AvgBatchTime = AvgBatchTime + batchTime;
	}
	Final_Time = AvgBatchTime / avg;
	std::cout << "\nNumber of Simplex Solved = " << C.size1() << std::endl;
	std::cout << "\nBoost Time taken:Wall  (in Seconds):: GLPK= "
			<< (double) Final_Time << std::endl;

	int max = 5;	//Verifying results of only first 5 LPs.
	if (LP_size < max)
		max = LP_size;

	std::cout << "\nVERIFICATION FOR CORRECTNESS\n";
	for (int i = 0; i < max; i++) {
		std::cout << "GLPK: " << resul[i] << " || GPU: " << result[i] << std::endl;
	}

	return 0;
}

