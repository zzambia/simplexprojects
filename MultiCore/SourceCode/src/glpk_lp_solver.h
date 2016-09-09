/*
 * lp_solver.h
 *
 *  Created on: 23-Apr-2014
 *      Author: gurung
 */

#ifndef GLPK_SOLVER_H_
#define GLPK_SOLVER_H_

#include <iostream>
#include <vector>
#include <glpk.h>
#include "matrix.h"
#include <boost/shared_ptr.hpp>

/* solution status:
 *
 #define GLP_UNDEF          1  		// solution is undefined
 #define GLP_FEAS           2  		// solution is feasible
 #define GLP_INFEAS         3  		// solution is infeasible
 #define GLP_NOFEAS         4  		// no feasible solution exists
 #define GLP_OPT            5  		// solution is optimal
 #define GLP_UNBND          6  		// solution is unbounded
 */

class glpk_lp_solver {
public:
	typedef boost::shared_ptr<glpk_lp_solver> glpk_ptr;

	void setDefalultObject(); // same as calling a constructor
	/*
	 glpk_lp_solver(std::vector<std::vector <double> > coeff_constraints, std::vector <double> bounds,
	 std::vector <int> bound_signs);
	 void setConstraints(std::vector<std::vector <double> > coeff_constraints, std::vector <double> bounds,
	 int bound_signs);
	 */
	glpk_lp_solver(math::matrix<double> coeff_constraints,
			std::vector<double> bounds, std::vector<int> bound_signs);
	void setConstraints(math::matrix<double> coeff_constraints,
			std::vector<double> bounds, int bound_signs);	//Before calling setConstraints(), the member
															//function setMin_or_Max() must be called.
	void setMin_Or_Max(int Min_Or_Max);
	int getMin_Or_Max();
	void setIteration_Limit(int limits);
	void setInitial_SimplexControlParameters();
	unsigned int getStatus();

	/*double Compute_LLP(std::vector<std::vector<double> > coeff_function, std::vector<std::vector <double> > coeff_constraints,
	 std::vector <double> bounds,std::vector <int> bound_signs);
	 double Compute_LLP(std::vector<std::vector<double> > coeff_function);
	 void setConstraints(std::vector<std::vector <double> > coeff_constraints, std::vector <double> bounds,
	 std::vector <int> bound_signs);
	 *
	 * Executes the simplex method with the given function.
	 */
	double Compute_LLP(std::vector<double> coeff_function) ;

	/*
	 * Executes the simplex method with the objective function set to zero. That is assigning all coefficient to zero
	 * This results in checking if the specified constraints has feasible solution or solution is infeasible or has no feasible soultion
	 * This application can be used to test the intersection of polytope with gaurd or with another polytope.
	 */
	unsigned int TestConstraints();


	// ******* Functions to be Interfaced later to the common LP_Solver class **********
	void displayMaxVariables();
	void free_glpk_lp_solver();
	//   ******************************************************************************

	glpk_lp_solver();
	~glpk_lp_solver();
private:

	glp_prob *mylp;
	glp_smcp param; // simplex parameter structure
	//char *lp_name;
	int Min_Or_Max;		//1 for Min and 2 for Max
	/* optimization direction flag:
	 #define GLP_MIN            1  // minimization
	 #define GLP_MAX            2  // maximization
	 */

	unsigned int dimension;				//dimension of the system
	unsigned int number_of_constraints;	//rows in glpk or facets of a polytope
	std::vector<double> Maximizing_Variables;// values of the variable that maximizes the result
	double result;				//Maximize or Minimize value

};

#endif /* GLPK_SOLVER_H_ */
