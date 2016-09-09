#ifndef LP_SOLVER_CPP_
#define LP_SOLVER_CPP_

/*
 * lp_solver.cpp
 *
 *  Created on: 23-Apr-2014
 *      Author: gurung
 */

#include "math/glpk_lp_solver/glpk_lp_solver.h"
#include "math/matrix.h"

using namespace std;

unsigned int glpk_lp_solver::lp_count = 0;		//initialize the static variable

glpk_lp_solver::~glpk_lp_solver() {
	//cout<<"\nLP problem died Here \n";
	lp_count--;
		glp_delete_prob(mylp);
	//	glp_free(mylp);
	//	if (lp_count == 0) {
	//		glp_free_env();
	//	}
}
void glpk_lp_solver::free_glpk_lp_solver() {
	lp_count--;
	glp_delete_prob(mylp);
	//glp_free(mylp);
	//	if (lp_count <= 0) {
	//		glp_free_env();
	//	}
}
void glpk_lp_solver::free_environment_glpk_lp_solver() {
	glp_free_env();
}
glpk_lp_solver::glpk_lp_solver() {
	//cout<<"\nLP problem Created Empty Here :::: No GLPK OBJECT CREATED SO FAR\n";
	this->setDefalultObject();
}

void glpk_lp_solver::setDefalultObject() {		//if this is a virtual member functions, then we cannot call from constructor
	mylp = glp_create_prob();
//	cout<<"\nLP problem Created Here\n";
	glp_init_smcp(&param);
	Min_Or_Max = GLP_MAX;	//Setting the GLP_MAX as the default value
	lp_count++;
	dimension = 0;
	number_of_constraints = 0;
	result = 0.0;
}

void glpk_lp_solver::displayMaxVariables() {
	for (int i = 0; i < dimension; i++)
		cout << "\t x" << i + 1 << " = " << Maximizing_Variables[i];
}

glpk_lp_solver::glpk_lp_solver(math::matrix<double> coeff_constraints,
		std::vector<double> bounds, std::vector<int> bound_signs) {
	lp_count++;
	//cout << "One Object created : Count = " << lp_count;
	Min_Or_Max = GLP_MAX;	//Setting the GLP_MAX as the default value
	mylp = glp_create_prob();
	glp_init_smcp(&param);

	//lp_name = "Sample";
	dimension = coeff_constraints.size2();		//.at(0).size();
	//dimension = coeff_function.size();
	number_of_constraints = bounds.size();
	result = 0.0;
}

void glpk_lp_solver::setMin_Or_Max(int Min_Or_Max) {
	/* optimization direction flag:
	 #define GLP_MIN            1  // minimization
	 #define GLP_MAX            2  // maximization
	 */
	if (Min_Or_Max == 1)
		this->Min_Or_Max = GLP_MIN;
	else
		this->Min_Or_Max = GLP_MAX;
}

int glpk_lp_solver::getMin_Or_Max() {
	return Min_Or_Max;
}

/*
 *
 * InEqualitySign :	The in equalities sign of the bound values 'b'. Possible values are
 * 					0 :	for  Ax = b (b Equals to)		FOR FUTURE USE IF ANY
 * 					1 :	for  Ax <= b (b is Greater Than or Equals to)
 * 					2 :	for  Ax >= b (b is Less Than or Equals to)
 *
 * We have not the bound values as 0 for now. At present the values are either 1 or 2.
 */

void glpk_lp_solver::setConstraints(math::matrix<double> coeff_constraints,
		std::vector<double> bounds, int bound_signs) {//here bound_sign is an Integer value

	dimension = coeff_constraints.size2();	//       .at(0).size();
	number_of_constraints = bounds.size();
	result = 0.0;
	glp_set_prob_name(mylp, "Sample");	// eg "sample"
	glp_set_obj_dir(mylp, Min_Or_Max);//Made change here 3rd June 2014			// GLP_MAX);
	//std::cout<<"\nMin or max = "<<Min_Or_Max;
	glp_add_rows(mylp, number_of_constraints);

	for (int i = 0; i < number_of_constraints; i++) {
		glp_set_row_name(mylp, i + 1, "p");
		if (bound_signs == 1)		//Ax<=b
			glp_set_row_bnds(mylp, i + 1, GLP_UP, 0.0, bounds[i]);
		else
			//Ax>=b
			glp_set_row_bnds(mylp, i + 1, GLP_LO, bounds[i], 0.0);
	}

	glp_add_cols(mylp, dimension);
	for (int i = 0; i < dimension; i++) {
		glp_set_col_name(mylp, i + 1, "xi");
		//glp_set_col_bnds(mylp, i + 1, GLP_FR, 0.0, 0.0); //Free Variables
		glp_set_col_bnds(mylp, i + 1, GLP_LO, 0.0, 0.0); // x>=0 where x is variables
		//	glp_set_obj_coef(mylp, i+1, coeff_function[i]);
	}

	int prod = number_of_constraints * dimension;
	int count = 1, ia[prod], ja[prod];
	double ar[prod];
	for (int i = 0; i < number_of_constraints; i++) {
		for (int j = 0; j < dimension; j++) {
			ia[count] = i + 1, ja[count] = j + 1, ar[count] = coeff_constraints(
					i, j);
			//cout<<"ia="<<ia[count]<<"\t ja="<< ja[count]<<"\t ar="<<ar[count]<<endl;
			count++;
		}
	}
	count--;

	glp_load_matrix(mylp, count, ia, ja, ar);

}

void glpk_lp_solver::setIteration_Limit(int limits) {
	param.it_lim = limits;
}
void glpk_lp_solver::setInitial_SimplexControlParameters() {
	//param = NULL;
	glp_init_smcp(&param);
}
unsigned int glpk_lp_solver::getStatus() {
	/*   solution status:
	 The routine glp_get_status reports the generic status of the current basic solution for
	 the specified problem object as follows:

	 #define GLP_UNDEF          1  // solution is undefined
	 #define GLP_FEAS           2  // solution is feasible
	 #define GLP_INFEAS         3  // solution is infeasible
	 #define GLP_NOFEAS         4  // no feasible solution exists
	 #define GLP_OPT            5  // solution is optimal
	 #define GLP_UNBND          6  // solution is unbounded
	 */
	unsigned int val = glp_get_status(mylp);
	return val;
}

unsigned int glpk_lp_solver::TestConstraints() {

	for (int i = 0; i < dimension; i++) {
		glp_set_obj_coef(mylp, i + 1, 0.0);	//assigning objective coefficients to zero
	}
	glp_term_out(GLP_OFF);
	glp_simplex(mylp, &param);
	/*result = glp_get_obj_val(mylp);
	 Maximizing_Variables.resize(dimension, 0.0);
	 for (int i = 0; i < dimension; i++)
	 Maximizing_Variables[i] = glp_get_col_prim(mylp, i + 1);
	 return result;
	 */
	return getStatus();
}

double glpk_lp_solver::Compute_LLP(std::vector<double> coeff_function) {//Here argument is a Vector

	for (int i = 0; i < dimension; i++) {
		glp_set_obj_coef(mylp, i + 1, coeff_function[i]);
	}
	glp_term_out(GLP_OFF);
	// initiliase the simplex parameters
	/*glp_smcp param;
	 glp_init_smcp(&param);
	 param.msg_lev = GLP_MSG_ERR; */
	/*param.msg_lev = GLP_MSG_OFF;*/
	glp_simplex(mylp, &param);
	//glp_simplex(mylp, NULL);
	double result = glp_get_obj_val(mylp);
	//std::cout<<"result   glpk="<<result;
	Maximizing_Variables.resize(dimension, 0.0);
	for (int i = 0; i < dimension; i++)
		Maximizing_Variables[i] = glp_get_col_prim(mylp, i + 1);

	return result;
}

/*
 * void glpk_lp_solver::setConstraints(
 std::vector<std::vector<double> > coeff_constraints,
 std::vector<double> bounds, std::vector<int> bound_signs) {	//here bound_sign is an Integer Array/Vector

 dimension = coeff_constraints.at(0).size();
 number_of_constraints = bounds.size();
 result = 0.0;
 glp_set_prob_name(mylp, "Sample");	// eg "sample"
 glp_set_obj_dir(mylp, Min_Or_Max);		//Made change here 3rd June 2014
 glp_add_rows(mylp, number_of_constraints);

 for (int i = 0; i < number_of_constraints; i++) {
 glp_set_row_name(mylp, i + 1, "p");
 if (bound_signs[i] == 1)		//Ax<=b
 glp_set_row_bnds(mylp, i + 1, GLP_UP, 0.0, bounds[i]);
 else
 //Ax>=b
 glp_set_row_bnds(mylp, i + 1, GLP_LO, bounds[i], 0.0);
 }

 glp_add_cols(mylp, dimension);
 for (int i = 0; i < dimension; i++) {
 glp_set_col_name(mylp, i + 1, "xi");
 glp_set_col_bnds(mylp, i + 1, GLP_LO, 0.0, 0.0);
 //	glp_set_obj_coef(mylp, i+1, coeff_function[i]);
 }

 int prod = number_of_constraints * dimension;
 int count = 1, ia[prod], ja[prod];
 double ar[prod];
 for (int i = 0; i < number_of_constraints; i++) {
 for (int j = 0; j < dimension; j++) {
 ia[count] = i + 1, ja[count] = j + 1, ar[count] =
 coeff_constraints[i][j];
 //cout<<"ia="<<ia[count]<<"\t ja="<< ja[count]<<"\t ar="<<ar[count]<<endl;
 count++;
 }
 }
 count--;
 //cout<<"\n count = "<<count;
 glp_load_matrix(mylp, count, ia, ja, ar);
 //	cout<<"Result Hey= "<<result<<endl;

 //	cout<<"Address of mylp before simplex call " << mylp << endl;

 }

 double glpk_lp_solver::Compute_LLP(
 std::vector<std::vector<double> > coeff_function,
 std::vector<std::vector<double> > coeff_constraints,
 std::vector<double> bounds, std::vector<int> bound_signs) {

 setConstraints(coeff_constraints, bounds, bound_signs);

 int ct=0, cpeak;
 unsigned int total, tpeak;
 glp_mem_usage(&ct, &cpeak, &total, &tpeak);
 std::cout << "blocks of mylp before simplex= " << ct << endl;

 for (int i = 0; i < dimension; i++) {
 glp_set_obj_coef(mylp, i + 1, coeff_function[i][0]);
 }

 // output msg OFF
 //param.msg_lev = GLP_MSG_OFF;
 glp_term_out(GLP_OFF);
 glp_simplex(mylp, &param);
 //	glp_simplex(mylp, NULL);
 //	cout<<"\nx ="<<x<<endl;
 result = glp_get_obj_val(mylp);
 //	cout<< "Result = " << result << endl;

 Maximizing_Variables.resize(dimension, 0.0);
 for (int i = 0; i < dimension; i++)
 Maximizing_Variables[i] = glp_get_col_prim(mylp, i + 1);
 //	std::cout << "Reached end";
 //	std::cout.flush();

 return result;
 }

 double glpk_lp_solver::Compute_LLP(
 std::vector<std::vector<double> > coeff_function) { 	//Here argument is a Matrix of size (m x 1)

 for (int i = 0; i < dimension; i++) {
 glp_set_obj_coef(mylp, i + 1, coeff_function[i][0]);
 }
 glp_term_out(GLP_OFF);
 // initiliase the simplex parameters
 glp_smcp param;
 glp_init_smcp(&param);
 param.msg_lev = GLP_MSG_ERR;

 param.msg_lev = GLP_MSG_OFF;
 glp_simplex(mylp, &param);
 //glp_simplex(mylp, NULL);
 result = glp_get_obj_val(mylp);
 Maximizing_Variables.resize(dimension, 0.0);
 for (int i = 0; i < dimension; i++)
 Maximizing_Variables[i] = glp_get_col_prim(mylp, i + 1);

 return result;
 }*/

#endif /* LP_SOLVER_CPP_ */
