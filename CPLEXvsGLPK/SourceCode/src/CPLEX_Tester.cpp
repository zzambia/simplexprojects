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
#include "parseBenchmark.h"

using namespace std;

void runCPLEX(unsigned int bno){
	//std::cout<<"Runing CPLEX for benchmark number = "<<bno<<std::endl;
	if (bno == 1) {
		system("cplex<netlibMPS/ADLITTLE;");
		//system ("/opt/ibm/ILOG/CPLEX_Studio_Community127/cplex/bin/x86-64_linux/./cplex<mycplex >f1.txt;");
	} else if (bno == 2) {
		//system("cplex<netlibMPS/AFIRO >f1.txt;");
		system("cplex<netlibMPS/AFIRO;");
	} else if (bno == 3) {
		system("cplex<netlibMPS/BEACONFD;");
	} else if (bno == 4) {
		system("cplex<netlibMPS/BLEND;");
	} else if (bno == 5) {
		system("cplex<netlibMPS/BOEING2;");
	} else if (bno == 6) {
		system("cplex<netlibMPS/BRANDY;");
	} else if (bno == 7) {
		system("cplex<netlibMPS/E226;");
	} else if (bno == 8) {
		system("cplex<netlibMPS/ISRAEL;");
	} else if (bno == 9) {
		system("cplex<netlibMPS/KB2;");
	} else if (bno == 10) {
		system("cplex<netlibMPS/RECIPELP;");
	} else if (bno == 11) {
		system("cplex<netlibMPS/SC105;");
	} else if (bno == 12) {
		system("cplex<netlibMPS/SC205;");
	} else if (bno == 13) {
		system("cplex<netlibMPS/SC50A;");
	} else if (bno == 14) {
		system("cplex<netlibMPS/SC50B;");
	} else if (bno == 15) {
		system("cplex<netlibMPS/SCAGR7;");
	} else if (bno == 16) {
		system("cplex<netlibMPS/SHARE1B;");
	} else if (bno == 17) {
		system("cplex<netlibMPS/SHARE2B;");
	} else if (bno == 18) {
		system("cplex<netlibMPS/STOCFOR1;");
	} else if (bno == 19) {
		system("cplex<netlibMPS/VTP-BASE;");
	}
}

int main(int argc, char *argv[]) {
	unsigned int LP_size = 1, avg, benchmarkNo;
	unsigned int MaxMinFlag=1; //1 for Min and 2 for Max
	math::matrix<double> A;
	math::matrix<float> C, newC;
	std::vector<double> b, c;
	std::vector<float> result;
	std::vector<int> status_val;
	double Final_Time;
	std::string afile, bfile, cfile;



	if (argc > 1) {
		//if (argc != 6) { //1(ApplicationName) + 5 (Input Arguments)
		if (argc != 4) { //1(ApplicationName) + 3 (Input Arguments)
			std::cout << "\nInsufficient Number of Arguments!!!\n";
			std::cout << "Correct Usages/Syntax:\n";
			std::cout << "./ProjName   'Benchmark-NO'   'Average'  'Batch-Size'\n";
			std::cout << "Argument 1) Benchmark-NO -- The Benchmark No as in the file parseBenchmark.cpp\n";
			std::cout << "Argument 2) Average -- Number of Average reading to be taken\n";
			std::cout << "Argument 3) Batch-size -- Number of LPs to be solved \n";
			return 0;
		}
		benchmarkNo = atoi(argv[1]);
		selectBenchmark(benchmarkNo, afile, bfile, cfile);//Selecting the required benchmark files having Matrix A, vector b and c.
		parseLP(afile.c_str(), bfile.c_str(), cfile.c_str(), A, b, c, MaxMinFlag);//parse the selected Benchmark files to convert into Matrix A along with vector b and c.

		avg = atoi(argv[2]);
		LP_size = atoi(argv[3]);

		/*
		 * Experimental Note:
		 *  To record the average time we ignore the first reading taken in GPU as it does not
		 *  reflect the correct computation time because for the first GPU call it also include an
		 *  extra overhead of GPU initialization time.
		 *
		 *  Note: The NetLib benchmarks are taken as
		 *  		Minimize cx
		 *  		sub to   Ax <= b
		 *  		For All  x>=0
		 *
		 *  so, we convert it as cx to -cx and display the result as -1 * (the computed result)
		 *  		-1 * (Maximize -cx)
		 *  		sub to   Ax <= b
		 *  		For All  x>=0
		 *
		 */


		double sum = 0.0;
		double wall_clock, return_Time;
		boost::timer::cpu_timer tt1, tt2;	//tt1 -- Variable declaration

		//Since our GPU LP Solver consider Maximize by default so we multiply -1 to the objective function
		for (unsigned int j = 0; j < c.size(); j++) {
			if (MaxMinFlag == 1)
				c[j] = -1 * c[j]; //Given problem is Minimize so converting to work on Maximization
		}

		std::cout << std::fixed; //to assign precision on the std::output stream
		std::cout.precision(17); //cout << setprecision(17);

		// *******************************************************************************************************************
		std::cout << "\n*****GLPK RESULT*****\n";
		double res = 0.0;
		double batchTime = 0.0, AvgBatchTime = 0.0;
		std::vector<double> resul(LP_size);
		for (int i = 1; i <= avg; i++) {
			tt1.start();
			for (int i = 0; i < LP_size; i++) {
				glpk_lp_solver mylp;
				mylp.setMin_Or_Max(2);	//2 for Maximize and 1 for Minimize
				mylp.setConstraints(A, b, 1); //this function actually determines independent LP in GLPK
				res = mylp.Compute_LLP(c); //We consider every dir an independent LP problem
				resul[i] = res;
			}
			tt1.stop();
			wall_clock = tt1.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
			return_Time = wall_clock / (double) 1000; //convert milliseconds to seconds
			batchTime = return_Time; //convert nanoseconds to milliseconds
			AvgBatchTime = AvgBatchTime + batchTime;
		}
		Final_Time = AvgBatchTime / avg;
		//std::cout << "\nNumber of Simplex Solved = " << C.size1() << std::endl;
	//	std::cout << "\nBoost Time taken:Wall  (in Seconds):: GLPK= " << (double) Final_Time << std::endl;
		// *******************************************************************************************************************


		// *******************************************************************************************************************
				std::cout << "\n*****CPLEX RESULT*****\n";
				double wall_clock2, return_Time2;
				double batchTime2 = 0.0, AvgBatchTime2 = 0.0;
				double Final_Time2;
				for (int i = 1; i <= avg; i++) {
					tt2.start();
					for (int i = 0; i < LP_size; i++) {
						runCPLEX(benchmarkNo);	//Running the CPLEX solver from a script file
					}
					tt2.stop();
					wall_clock2 = tt2.elapsed().wall / 1000000; //convert nanoseconds to milliseconds
					return_Time2 = wall_clock2 / (double) 1000; //convert milliseconds to seconds
					batchTime2 = return_Time2; //convert nanoseconds to milliseconds
					AvgBatchTime2 = AvgBatchTime2 + batchTime2;
				}
				Final_Time2 = AvgBatchTime2 / avg;
				//std::cout << "\nNumber of Simplex Solved = " << C.size1() << std::endl;

	std::cout << "\nBoost Time taken:Wall  (in Seconds):: CPLEX= " << (double) Final_Time2 << std::endl;
	// *******************************************************************************************************************

	std::cout << "\nBoost Time taken:Wall  (in Seconds):: GLPK= " << (double) Final_Time << std::endl;
	// *******************************************************************************************************************


// *******************************************************************************************************************
		int max = 2;	//Verifying results of only first 2 LPs.
		if (LP_size < max)
			max = LP_size;

		std::cout << "\nVERIFICATION FOR CORRECTNESS\n";
		for (int i = 0; i < max; i++) {
			if (MaxMinFlag == 1)
				std::cout << "GLPK: " << -1 * resul[i] << std::endl;
			else
				std::cout << "GLPK: " << resul[i] << std::endl;
		}
	}
	return 0;

}

