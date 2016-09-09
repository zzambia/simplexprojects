//============================================================================
// Name        : GLPK_MULTI.cpp
// Author      : Amit Gurung
// Version     :
// Copyright   : NIT Meghalaya
// Description : Hello World in C++, Ansi-style
//============================================================================
//#include "/usr/include/boost/timer.hpp"
#include <boost/timer/timer.hpp>
#include <iostream>
#include "glpk_lp_solver.h"
#include "matrix.h"
#include <omp.h>
#include <random>
#include <array>
#include <iterator>
#include <algorithm>
#include "memory_usages.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
#include "sys/vtimes.h"
#include <stdlib.h>

static clock_t lastCPU, lastSysCPU, lastUserCPU;

static int numProcessors;
void init_cpu_usage() {
	FILE* file;
	struct tms timeSample;
	char line[128];

	lastCPU = times(&timeSample);
	lastSysCPU = timeSample.tms_stime;
	lastUserCPU = timeSample.tms_utime;

	file = fopen("/proc/cpuinfo", "r");
	numProcessors = 0;
	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "processor", 9) == 0)
			numProcessors++;
	}
	fclose(file);
	//std::cout<<"Number of Processors = " <<numProcessors<<std::endl;
}

double getCurrent_ProcessCPU_usage() {
	struct tms timeSample;
	clock_t now;
	double percent;

	now = times(&timeSample);

	if (now <= lastCPU || timeSample.tms_stime < lastSysCPU
			|| timeSample.tms_utime < lastUserCPU) {
		//Overflow detection. Just skip this value.
		percent = -1.0;
	} else {
		percent = (timeSample.tms_stime - lastSysCPU)
				+ (timeSample.tms_utime - lastUserCPU);
		percent /= (now - lastCPU);
		percent /= numProcessors;
		percent *= 100;
	}
	lastCPU = now;
	lastSysCPU = timeSample.tms_stime;
	lastUserCPU = timeSample.tms_utime;

	return percent;
}

using namespace std;
int main(int argc, char *argv[]) {

	unsigned int LP_size = 1, avg, dimension, art, rg;
	// ****** Tester for Helicopter Begins  ******************************
	math::matrix<double> A;
	math::matrix<float> C, newC;
	std::vector<double> b;
	std::vector<float> result;
	std::vector<int> status_val;
	//variables
	int col, row;
	double Final_Time;
	if (argc > 1) {
		if (argc != 5) {		//1(ApplicationName) + 4 (Input Arguments)
			std::cout << "\nInsufficient Number of Arguments!!!\n";
			std::cout << "Correct Usages/Syntax:\n";
			std::cout
					<< "./ProjName --'Dimension'--'Average'--'NO_OF_LP' ---'Artificial-Variables' !!\n";
			std::cout
					<< "argument 1) Dimension -- select the dimension of LP to be solved\n";
			std::cout
					<< "argument 2) Average -- select the number of runs for average readings\n";
			std::cout
					<< "argument 3) Batch-size -- select the number of LPs to be solved\n";
			std::cout
					<< "argument 4) Artificial-Variables -- Number of negative on RHS of the LP constraints\n";
			return 0;
		} else {
			unsigned int num;
			dimension = atoi(argv[1]);
			num = atoi(argv[2]);
			avg = num;
			LP_size = atoi(argv[3]);
			art = atoi(argv[4]);
			//rg = atoi(argv[5]);
		}
	}

	unsigned int status = 0, res1 = 0, res2 = 0;
	int sign = -1, sum1;
	glpk_lp_solver lp1, lp;
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
				b[j] = (rand() % (j + 1) + (10 + j));
		}
		//** Setting current Lp to GLPK
		lp.setMin_Or_Max(2);
		lp.setConstraints(A, b, 1);
		status = lp.TestConstraints(); //std::cout << "Status = " << status << "\n";
	}
	/*
	 * Experimental Note:
	 *  We generate randomly the first LP problem (the matrix A and vector b) and then generate
	 *  randomly the objective function(s). For the ease generation process we take all the
	 *  LP problems to be the same LP generated above. However, for each of this LPs we generate
	 *  randomly different objective functions.
	 *
	 *  Also, to record the average time we ignore the first reading taken in our Supercomputer as it does not
	 *  reflect the correct computation time, we suspect the reason could be similar to GPU initialization time.
	 */

	C.resize(LP_size, dimension);
	unsigned int i;
	for (int i = 0; i < LP_size; i++) {
		for (unsigned int j = 0; j < C.size2(); j++) {
			C(i, j) = rand() % (j + 1) + 1;
		}
	}

	//Computation for CPU ie GLPK
	//cout<<"No of threads: "<<omp_get_max_threads();
	std::vector<double> dir(dimension);
	std::vector<std::vector<double> > dir_all(LP_size);
	for (int i = 0; i < LP_size; i++) {
		dir_all[i].resize(dimension);
	}

	unsigned int l, j;
	double res_seq[LP_size], res_omp[LP_size];

	for (l = 0; l < LP_size; l++) {
		for (j = 0; j < dimension; j++) {
			dir[j] = C(l, j);
		}
		dir_all[l] = dir;
	}

	std::cout << std::fixed; //to assign precision on the std::output stream
	std::cout.precision(17); //cout << setprecision(17);

	double sum = 0.0, cpu = 0.0, sum_cpu = 0.0, avg_cpu = 0.0;
	long memory;
	sum = 0.0;
	double Avg_wall_clock = 0.0;

	boost::timer::cpu_timer tt1;
	for (int a = 0; a <= avg; a++) {
		init_cpu_usage();
		tt1.start();
		//#pragma omp parallel for
		for (int l = 0; l < LP_size; l++) {
			glpk_lp_solver lp4;
			lp4.setMin_Or_Max(2);
			lp4.setConstraints(A, b, 1);
			res_seq[l] = lp4.Compute_LLP(dir_all[l]);
		}
		tt1.stop();
		cpu = getCurrent_ProcessCPU_usage();
		//std::cout<<"CPU Utilization % : " << cpu <<std::endl;

		double wall_clock, user_clock, system_clock;
		wall_clock = tt1.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
		user_clock = tt1.elapsed().user / 1000000;
		system_clock = tt1.elapsed().system / 1000000;

		if (a != 0) {
			Avg_wall_clock = Avg_wall_clock + wall_clock;

			memory = getCurrentProcess_PhysicalMemoryUsed();
			sum_cpu = sum_cpu + cpu;
		} else {
			std::cout << "Iteration No : " << a << " SEQ: Time"
				<< wall_clock / (double) 1000 << " seconds\n";
		}
	}

	std::cout
			<< " ********************** Sequential GLKP *********************\n";
	avg_cpu = sum_cpu / (double) avg;
	Avg_wall_clock = Avg_wall_clock / avg;
	double return_Time = Avg_wall_clock / (double) 1000;
	std::cout << "SEQ: Time to solve " << LP_size << " LPs = " << return_Time
			<< " seconds\n";
	std::cout << "SEQ: memory taken : " << memory << " KB\n";
	std::cout << "SEQ: CPU utilization(%) : " << avg_cpu << "\n";
	std::cout
			<< " ********************** Sequential GLKP *********************\n\n";

	avg_cpu = 0.0, sum_cpu = 0.0;
	Avg_wall_clock = 0.0;
	boost::timer::cpu_timer tt2;

	for (int a = 0; a <= avg; a++) {
		init_cpu_usage();
		tt2.start();
		//omp_set_dynamic(0);
		std::cout << "Number of LPs in parallel = " << LP_size << std::endl;
#pragma omp parallel for
		for (int l = 0; l < LP_size; l++) {
			glpk_lp_solver lp3;
			lp3.setMin_Or_Max(2);
			lp3.setConstraints(A, b, 1);
			res_omp[l] = lp3.Compute_LLP(dir_all[l]);
		}
		tt2.stop();
		cpu = getCurrent_ProcessCPU_usage();

		double wall_clock, user_clock, system_clock;
		wall_clock = tt2.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
		user_clock = tt2.elapsed().user / 1000000;
		system_clock = tt2.elapsed().system / 1000000;

		if (a != 0) {
			//glpk_lp_solver::free_environment_glpk_lp_solver();
			memory = getCurrentProcess_PhysicalMemoryUsed();
			std::cout << "Iteration No : " << a << " OMP: Time"
					<< wall_clock / (double) 1000 << " seconds\n";
			Avg_wall_clock = Avg_wall_clock + wall_clock;
			sum_cpu = sum_cpu + cpu;
		} else {
			std::cout << "Iteration No : " << a << " OMP: Time"
					<< wall_clock / (double) 1000 << " seconds\n";
		}
	}
	cout<< "\n\n *************** Summary on Thread Safe GLPK ***********************\n";

	avg_cpu = sum_cpu / avg;
	Avg_wall_clock = Avg_wall_clock / avg;
	double return_Time2 = Avg_wall_clock / (double) 1000;
	std::cout << "OMP: Time to solve " << LP_size << " LPs = " << return_Time2
			<< " seconds\n";
	std::cout << "OMP: memory taken : " << memory << "\n";
	std::cout << "OMP: CPU Utilization(%) : " << avg_cpu << "\n";
	cout << " *************** Thread Safe GLPK ***********************\n\n";

	int max = 5;	//Verifying results of only first 5 LPs.
	if (LP_size < max)
		max = LP_size;

	cout << "\nVERIFICATION FOR CORRECTNESS\n";
	for (int l = 0; l < max; l++) {
		cout << "SEQ: " << res_seq[l] << " || OMP: " << res_omp[l] << endl;
	}
	return 0;
}
