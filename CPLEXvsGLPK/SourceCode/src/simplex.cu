#include "simplex.cuh"
#include<omp.h>
#include "iostream"
#include <math.h>


//1st Method : Most Negative Value approach
__global__ void mykernel(float *S_MAT, int S_row, int S_col, float *Result, int S_N, float *R_data, int *R_index) {
	//int index = threadIdx.x + (blockIdx.x * blockDim.x);
	int index = blockIdx.x;
	if (index < S_N) {
		int tid;
		int i; // used for for index
		unsigned int temp_index;
		unsigned int temp_index1;
		int base = index * S_row * S_col;
		int R_base = index * blockDim.x;  // blockDim.x = 96
		__shared__ bool c;
		__shared__ int rm;
		__shared__ int row;	//pivotRow
//		__shared__ int pivotCol;//pivotCol this can remove global variable S_Sel

		int col = 1;
		__shared__ int remember[1024]; //Found a column which is negative but theta/Min has no positive value
		//Debug ------ Initializing ------------
		remember[threadIdx.x]= 7777; //Since our maximum value might be 1024 for very large LP. But depending upon dim, that many remember[] will be assigned
		//-------------------
		//int Last_row = S_row - 1;//Amit now this should be 2nd last row
		int Last_row = S_row - 2;//Amit now this should be 2nd last row

		__shared__ float col1[1024];	//pivotColumn
		/*************/
		if (threadIdx.x == 0) {
//			printf("sizeof(float) =%d ",sizeof(float));
//			printf("\nsizeof(int) =%d \n", sizeof(int));
			c = false;
			rm = 0;
			row = -1;		//pivotRow
//			pivotCol = -1;
		}
		__syncthreads();
		while (!c) {
			__syncthreads();
			//   ***************** Get_Pivot function begins  *****************
			// ******** First Reduction Begins **********
			//using reduction to compute min and newpivotcol
			__shared__ int notEntered;
			__shared__ float minValue;
			__shared__ int newpivotcol;
			if (threadIdx.x == 0) {
				row = -1;		//pivotRow
				minValue = 0;
				newpivotcol = -1;
				notEntered = 1;
				c = true;
			/*	printf("Printing Remembered\n");
				for (int tt=0;tt<1024;tt++)
					printf("Remember[%d] = %d \n",tt, remember[tt]);
				printf("\n");*/
			}
			__syncthreads();	//making sure newpivotcol is initialised to -1
			// Since keeping limit only upto (S_col - 1) which is not equal to BLOCK_SIZE creates problem
			// in using syncthreads() inside Reduction for-loop so use all threads(all R_data)
			//int data_size = (S_col - 1) - 2;
			int data_size = blockDim.x;
			tid = threadIdx.x;
			if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)) { //find minimum from last row leaving last column
				//tid = threadIdx.x - 2;//here my tid should be from 0 to (evenThreadId - 1)
				//int j = threadIdx.x;//is the actual column/index number less than (S_col - 1)
				//printf("Data_size = %d ", data_size);
				temp_index = Last_row + tid * S_row + base;	//avoiding re-computation
				R_data[tid + R_base] = S_MAT[temp_index];	//	g_data[i];
				R_index[tid + R_base] = tid;//tid; should be the real index of the data
			} else {
				R_data[tid + R_base] = INT_MAX;	//	g_data[i];
				R_index[tid + R_base] = tid;	//tid;
			}
			__syncthreads();//here will have all values in shared memory from 0 to BLOCK_SIZE
			//Debug----
			/*if (threadIdx.x==0){
				printf("\n Data and Index");
				for (int x=0;x<blockDim.x;x++){
					printf("(%f, %d), ", R_data[R_base + x], R_index[R_base + x]);
				}
				printf("\n");
			}
			__syncthreads();*/
			//----- Verified correct copy of Data and Index

			tid = threadIdx.x;
			for (i = (data_size / 2); i > 0;) {
				if (tid < i) {
					//	if ((R_data[tid] >= R_data[tid + s]) && ((R_data[tid + s] < 0) && (R_data[tid] < 0))){
					//(R_data[tid + R_base] < 0) && (R_data[tid + R_base + i] < 0)&&
					if (R_data[tid + R_base] > R_data[tid + R_base + i]) { //is right-side value small?
						//if (R_data[tid + R_base + i] == -0.000000)
							 //R_data[tid + R_base + i] = 0.0;
						//if ((R_data[tid + R_base + i] <= -0.000001) || (R_data[tid + R_base + i] < 0) ) {	//only if the value on the right-side is -ive
						if (R_data[tid + R_base + i] <= -0.000001) {	//only if the value on the right-side is -ive
							R_data[tid + R_base] = R_data[tid + R_base + i];//put the smaller value to left-side
							R_index[tid + R_base] = R_index[tid + R_base + i];

							//notEntered = false;  //race condition avoided
							//notEntered = 0;  //race condition avoided
							int local_notEntered;
							local_notEntered = *(volatile int*) &notEntered;
							atomicCAS(&notEntered, local_notEntered, 0);
						}
					}
				}
				/*if (tid == 0)
				 printf("Data_size = %d ", i);*/
				__syncthreads();
				i >>= 1;
				if ((i != 1) && (i % 2) != 0) {	//if s is odd
					i = i + 1;
				}
			}
			// if (notEntered == false && tid == 2) { // tid==0 is always true if minValue is still -1 then what?
			if (threadIdx.x == 0) { // tid==0 is always true if minValue is still -1 then what?
			//	printf("\n Min Value = %f NewPivotCol = %d ", minValue,newpivotcol);
				if (notEntered == false) {
					minValue = R_data[R_base];
					newpivotcol = R_index[R_base];
					printf("\n Min Value = %f NewPivotCol = %d ", minValue,newpivotcol);
				}
			}
			__syncthreads(); //waiting for all threads to have same newpivotcol value

			//Debug----------
			/*if (threadIdx.x==0){
				printf("\n");
				for (int x=0;x<blockDim.x;x++){
					printf("(Data = %f Index = %d), ", R_data[R_base + x], R_index[R_base + x]);
				}
				printf("\n");
			}
			__syncthreads();*/
			//--------------
			//		}
			//		__syncthreads();	//here we have min and newpivotcol

			//Note to return minValue with Index (index of the simplex tableau and not the index of variable which will be -2)
			// ********* First Reduction Ends *************
			//  ******** Second Reduction Begins **********
			if (newpivotcol == -1) {//All Threads will follow the Same path so no issue with divergence
				//return -2;
				row = -2; //No pivot column found so Optimal solution reached
				c=true; //can terminate
			} else { //if pivot column found then Find pivot row

				// ********** Second Reduction Process ******
				__shared__ float row_min;
				__shared__ int row_num;
				__shared__ int notEntered2;
				if (threadIdx.x == 0) {
					row_min = INT_MAX;
					row_num = -1;
					notEntered2 = 1;
				}
				__syncthreads();
				// Since keeping limit only upto Last_row which is not equal to block_size creates problem
				// in using syncthreads() inside Reduction for-loop so use all threads(all R_data
				int k1;
				if (threadIdx.x >= 0 && threadIdx.x < Last_row) {
					k1 = threadIdx.x;	//here k1 =0 to Last_row only
					//for (int k1 = 0; k1 < Last_row; k1++) {	//Last_row = (S_row - 1)
					int temp_index2 = newpivotcol * S_row + k1 + base;
					temp_index1 = k1 + (S_col - 1) * S_row + base; //avoiding re-computation
					//if ((S_MAT[temp_index2] > 0) && (S_MAT[temp_index1] > 0)) {
					//Although Simplex algorithm says exclude zero quotient but case study show it takes zero as minimum positive ratio
					//if (S_MAT[temp_index2] > 0 && S_MAT[temp_index2] < INT_MAX) {
					if ((S_MAT[temp_index2] > 0) && (S_MAT[temp_index1] >= 0)) {
						//printf("\nS_MAT[temp_index2]= %f and S_MAT[temp_index1]= %f R_index= %d",S_MAT[temp_index2], S_MAT[temp_index1], k1);
						R_data[k1 + R_base] = (float)S_MAT[temp_index1] / (float)S_MAT[temp_index2]; //b_i / S_MAT[pivotcol]
						//-------------------------------------------------
						//Since there exists some feasible value which may be in the 1st location
						int local_notEntered2;
						local_notEntered2 = *(volatile int*) &notEntered2;
						atomicCAS(&notEntered2, local_notEntered2, 0);
						//-------------------------------------------------
						R_index[k1 + R_base] = k1;
					} else {
						R_data[k1 + R_base] = INT_MAX; //to make the array size equal
						R_index[k1 + R_base] = k1; //to make the array size equal
					}
				} else { //remaining threads above Last_row(including) upto Block_Size
					k1 = threadIdx.x;
					R_data[k1 + R_base] = INT_MAX; //to make the array size equal
					R_index[k1 + R_base] = k1; //to make the array size equal
				}
				__syncthreads(); //Verified All data and index stored correctly with index as threadIdx.x
				//Debugging ---------------------------------------
/*				if(threadIdx.x==0){
					printf("Printing Last Row\n");
					for (int tt=0;tt<20;tt++)
						printf("%f ",R_data[tt + R_base]);
					printf("\nPrinting Done\n");
				}*/
				//----------------------------------------------------
				//Now find the minValue and its index from R_data and R_index using Reduction
				//int data_size = Last_row;
				int data_size2 = blockDim.x; //Now it is Block_Size
				// ***** Second Reduction on R_data and R_index ****
				//	if (threadIdx.x >= 0 && threadIdx.x < Last_row) {	//Now for all threads
				tid = threadIdx.x;
				for (int s = (data_size2 / 2); s > 0;) {
					if (tid < s) {
						int indexValue2 = tid + R_base;
						if (R_data[indexValue2] > R_data[indexValue2 + s]) { //changed >= to > ToDo:: Accordingly Fix other bug
							R_data[indexValue2] = R_data[indexValue2 + s];	//For arranging in ascending order we better swap the value instead of only replacing
							R_index[indexValue2] = R_index[indexValue2 + s];
							//notEntered2 = false;

							//notEntered2 = 0;
/*	Fixing					int local_notEntered2;
							local_notEntered2 = *(volatile int*) &notEntered2;
							atomicCAS(&notEntered2, local_notEntered2, 0);*/
						}
					}
					__syncthreads();	//This creates unpredictable behaviour
					s >>= 1;
					if ((s != 1) && (s % 2) != 0) {	//if s is odd
						s = s + 1;
					}
				}
				//if (notEntered2 == false && tid == 0) {
				if (tid == 0) {
					if (notEntered2 == false) {	//if at least once swaped the flag 'notEntered2' will be equal to 0
						row_min = R_data[R_base];
						row_num = R_index[R_base];
						printf("\nR_Data = %f pivotRow = %d", R_data[R_base], R_index[R_base]);
						//printf("\nR_Data next = %f R_Index next = %d", R_data[R_base + 1], R_index[R_base + 1]);
					}
				}
				__syncthreads(); // Looks like this can be skipped
				//	}
				//	__syncthreads();	//here we have Row_min and newpivotRow
				// ********** Second Reduction on R_data and R_index ******
				if (threadIdx.x == 0) {
//					pivotCol = newpivotcol;
					if (row_min == INT_MAX) {
						//if (notEntered2 == true) {
						//return -1;
						//printf("%f ", R_data[R_base]);
						row = -1;
					}
					if ((row_min != INT_MAX) && (row_num != -1)) {
						//}else {
						//return row_num;
						//printf("%f %d ", row_min, row_num);
						row = row_num;
					}
				}
				__syncthreads(); // Looks like this can be skipped
			} //end of else of newpivotcol == -1
			__syncthreads(); // Looks like this can be skipped but here we have row synchronized
			//  ******** Second Reduction Ends **********
			//   ***************** Get_Pivot function ends  *****************
//*******************************************************************************************************
//			col = pivotCol;
			col = newpivotcol;
			//printf("Row= %d col = %d\n",row,col);
			if (row > -1) {	//some candidate leaving variable or pivot row have been found
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int i1 = 2; i1 < S_col; i1++) {		//Data Parallel section 1
					/*if (tid == remember[tid - 2]) { //Before actual process convert back the Original column value of last row remembered in the previous iteration
						temp_index = Last_row + (tid * S_row) + base; //avoiding re-computation
						S_MAT[temp_index] = -1 * S_MAT[temp_index]; //replacing back to original
					}*/

					if (remember[(tid - 2)] != 7777) { //Before actual process convert back the Original column value of last row remembered in the previous iteration
						printf("\nPreviously remembered column is %d ", remember[(tid - 2)]);
						//on the 2nd last row and on the remembered column replace back the original value
						temp_index = Last_row + (remember[(tid - 2)] * S_row) + base; //remember[(tid - 2)] gives the remembered column number
						S_MAT[temp_index] = -1 * S_MAT[temp_index]; //replacing back to original
						remember[(tid - 2)] = 7777; //reset back to default value for next iterations
					}
				}		//Data Parallel section 1 done
				__syncthreads();
//*******************************************************************************************************
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data Parallel section 2
					col1[tid] = S_MAT[(tid + col * S_row) + base];//keeping the old pivotcol coeff
				}	//Data Parallel section 2 done
				__syncthreads();
//*******************************************************************************************************
				unsigned int temp_row_base = row + base;
				//S_MAT[temp_row_base + 1 * S_row] = S_MAT[temp_row_base + col * S_row]; //column 1 replaced with objective coefficient
				S_MAT[temp_row_base + 1 * S_row] = S_MAT[(S_row -1) + col * S_row + base]; //column 1 replaced by the new Last row containing objective coefficient
				//S_MAT[temp_row_base] = col - 1;
				S_MAT[temp_row_base] = col - 1; //replacing entering variable index in leaving variable index (1-based indexing)
//*******************************************************************************************************
/*
				//Debugging after one iteration -----------------------
				if (threadIdx.x==0){
					printf("\n ----- Iteration Before update operation----- \n");
					for (int r=0;r<S_row;r++){
						for (int cl=0;cl<S_col;cl++){
							printf("%f  ",S_MAT[r+cl*S_row+base]);
						}
						printf("\n");
					}
					printf("\n");
				}
				//----------------------------------------------------- Correct replacement
*/
				tid = threadIdx.x;
				//Debug------------
				/*if (threadIdx.x==0)
					printf("\ncol1[row]=%f\n", col1[row]);*/
				//-----------------
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int j = 2; j < S_col; j++){		//Data Parallel section 3
					unsigned int row_base = row + base;	//avoiding re-computation
					temp_index = row_base + (tid * S_row);//avoiding re-computation
					S_MAT[temp_index] = S_MAT[temp_index] / col1[row];//updating pivot row by dividing, current pivot row value by (pivot element)
				}		//Data Parallel section 3 done
				__syncthreads();

				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < (S_row - 1)) { //updating all rows
					//for (int i = 0; i < S_row; i++) {	//Data parallel section 4
					for (i = 2; i < S_col; i++) {
						if (tid != row) {
							temp_index1 = i * S_row + base;
							temp_index = tid + temp_index1;
							float zeroTemp; 
							zeroTemp = col1[tid] * S_MAT[row + temp_index1];
							S_MAT[temp_index] = S_MAT[temp_index] - zeroTemp;
						} else {
							break;
						}
					}
				}	//Data Parallel section 4 done
				__syncthreads();

				//if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)){
				//tid = threadIdx.x;
				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {
						if (S_MAT[(Last_row + i * S_row) + base] < 0) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();

				//Debugging after one iteration -----------------------
				/*if (threadIdx.x==0){
					printf("\n ----- Iteration ----- \n");
					for (int r=0;r<S_row;r++){
						for (int cl=0;cl<S_col;cl++){
							printf("%f  ",S_MAT[r+cl*S_row+base]);
						}
						printf("\n");
					}
					printf("\n");
				}*/
				//-----------------------------------------------------

			} else if (row == -1) { //No candidate leaving row have been found so remember this pivot column (and try next iteration although not mentioned in Algorithm)
				//This is actually the situation of UNBOUNDEDNESS but we are trying next iteration to see if the other candidate pivot column might give feasible pivot row
				//as it happened in the case of Helicopter model.
				//ToDo::Note we should have terminated this process when all columns have been tested. Otherwise for unbounded LPs our algorithm will not terminate.
				if (threadIdx.x == 0) {
					c = true;
					remember[rm] = col; //remember this particular column and do not select this column in the next iterations
					printf("\n Remembered col = %d\n",col);
					rm++;
				}
				__syncthreads();
				//In the last row the value with the pivot column, col,
				temp_index = Last_row + (col * S_row) + base; //if col==-1 than problem for base==0 i.e. temp_index==-1
				S_MAT[temp_index] = -1 * S_MAT[temp_index];	//remembering by making positive so that this column will not be selected in the next iteration
				//if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)){
				// tid = threadIdx.x;
				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {		//Data parallel 5
						if ((S_MAT[(Last_row + i * S_row) + base] < 0)) { //check if any negative in the last row (S_row -1)
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();
			}/*
			//Debugging ------------------- Functioning without this as well
			else if (row == -2){
				//no entering variable (pivot column) found
				c=true;
			}
			// -------------------------------------*/
		} //end of while
		__syncthreads();
		
		if (threadIdx.x == 0) {
			Result[index] = S_MAT[(Last_row + (S_col - 1) * S_row) + base];
			//printf("\nResult Inside Kernel: %f \n",Result[index]);
		}
	}
}

__host__ Simplex::Simplex(unsigned int N_S) {
	number_of_LPs = N_S;
	M = 0;
	N = 0;
	c = 0;
	No_c = 0;
	R = (float*) calloc(N_S,sizeof(float));
	}

//get status of particular simplex
__host__ int Simplex::getStatus(int n) {
	int s;
	for (int i = 0; i < C.size1(); i++) {
		if (i == (n - 1)) {
			if (R[i] == -1) {
				s = 6;	// 6 = Simplex Is Unbounded
			} else if (R[i] > 0) {
				s = 2;	// 2= Simplex has feasible and Optimal solution
			}
		}
	}
	return s;

}	//get status of particular simplex

//get the No of simplex the object is ruuning on GPU
__host__ int Simplex::getNo_OF_Simplx() {
	return C.size1();
}	//get the No of simplex the object is ruuning on GPU

//get the result of all simplex
__host__ std::vector<float> Simplex::getResultAll() {

	std::vector<float> Res(C.size1());
	for (int i = 0; i < C.size1(); i++) {
		Res[i] = R[i];
	}
	return Res;
}

//get the result of all simplex

__host__ float Simplex::getResult(int n) {
	// get result of particular simplex
	float r;
	for (int i = 0; i < C.size1(); i++) {
		if (i == (n - 1)) {
			r = R[i];
		}
	}
	return r;
}	// get result of particular simplex

__host__ std::vector<int> Simplex::getStatusAll() {

	std::vector<int> Status(C.size1());
	for (int i = 0; i < C.size1(); i++) {
		if (R[i] == -1)
			Status[i] = 6;
		else
			Status[i] = 2;
	}
	return Status;
}	//get the status of all simplex

__host__ void Simplex::setConstratint(math::matrix<double> A, std::vector<double> B) {
	int N_S = number_of_LPs;
	orig_CoefficientMatrix = A;
	BoundValue = B;
	int No_O = A.size2();
	int No_C = A.size1();
	//M = No_C + 1;
	M = No_C + 2;//Extra row for coefficient of objective function
	N = No_O + 3 + No_C;//original variables + slack + 3 extra(index,pivot-col,b_i); artificial is not included now/here
	c = 1 + No_O;
//	MAT_COPY = (float *) calloc(N_S * M * N, sizeof(float));
	MAT = (float *) calloc(N_S * M * N, sizeof(float));

	/*
	 * Simplex tableau Re-Structure Amit :: Note The variables are implemented as 1-based indexing
	 * row-size= (m-constraints + 1-row Z and Optimal Solution value + 1-row for copy of coefficient of Objective function) = m+2
	 * column-size = (n-original-variables + m-slack-variables + a-artificial-variables+ 3 (1 for index, 1 for coefficient of the basic variables, 1 for bounds b_i's)
	 * column 0: index of basic/slack variables
	 * column 1: coefficient of basic variables
	 * column 2 to n: coefficients of variables (non-basic) --implemented as 2 to No_O+2 where No_O is the size of original variables
	 * column (n+1) to (n+m): includes slack variables --implemented as (No_O+2 +1) to (No_O+2+No_C) where No_C is the number of constraints comprising slack variables
	 * column (n+m+1) to (n+m+a): includes artificial variables --implemented after (No_O+2+No_C + 1) to <(N-1) where N is the total size of columns
	 * column last column (N-1) : bounds b_i
	 *
	 * row 0 through m: contains the coefficient for Simplex method algorithm
	 * row (m+1), the 2nd last row: contains the values of the operations (Cj - Zj) of each iterations starting from column 2 through (last - 1) and the last column contains the
	 * value of the optimal solution of each iterations.
	 * Last row (m+2): contains a copy of the coefficients of the objective function required for Simplex Algorithm. Column 2 through (N-1) is used to store these values
	 * ** NB: In phase-I : Artificial variables contains -1 and all values are 0. But in phase-II artificial variables are eliminated and the original coefficients are replaced
	 * 		  with original variables having their respective values and 0 for slack variables.
	 *
	 */
#pragma omp parallel for
	for (int s = 0; s < N_S; s++) {
		unsigned int some=M * N * s;
		for (int i = 0; i < (M - 2); i++) {
			for (int j = 0; j < N; j++) {
				if (j == 0) {	//index of basic/slack variables
					MAT[(int) ((i + j * M) + some)] = c + i;
				} else if (j > 1) { //excluding 'column 1' from loop
					if (j < (No_O + 2)) {	// coefficients of the variables (a.k.a. non-basic)
						MAT[(int) ((i + j * M) + some)] = (float) A(i,j - 2);
					} else if (j == (N - 1)) { //last column stores the bounds b_i
						MAT[(int) ((i + j * M) + some)] = (float) B[i];
					} else if (j < (N - 1)) { //includes slack variables
						MAT[(int) ((i + (No_O + 2 + i) * M) + some)] = 1;
					}
				}
			}
		}
	}

	/*  //Debugging
		printf("\n");
	for (int s = 0; s < N_S; s++) {
		unsigned int some=M * N * s;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				std::cout<<MAT[(int)(i+ j*M +some)]<<"  ";
			}
			printf("\n");
		}
		printf("\n");
	}*/

}	

__host__ void Simplex::ComputeLP(math::matrix<float> &C1) {

	cudaError_t err;
	unsigned int threads_per_block;	//Maximum threads depends on CC 1.x =512 2.x and > = 1024
	unsigned int number_of_blocks;//depends on our requirements (better to be much more than the number of SMs)
	int device;
	cudaDeviceProp props;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&props, device);
	int No_C = orig_CoefficientMatrix.size1();
	C = math::matrix<float>(C1);
	int N_S = C.size1();
	int No_O = C.size2();
	M = No_C + 2, N = No_O + 3 + No_C;// M is now + 2 instead of +1

	int N_C = No_C;
	c = 1 + No_O;
	//int s;
#pragma omp parallel for
	for (int s = 0; s < N_S; s++) {
		unsigned int some = M * N * s; //base address for each LP
		for (int j = 2; j < (No_O+2); j++) { //Amit::Infact can be < (No_O+2) for Optimization
			//if (j < 2 + No_O) { //assigning objective coefficients of variables only
				//MAT[(int) (((M-1) + j * M) + some)] = -C(s, j - 2); //Last row (M-1)
			MAT[(int) (((M-2) + j * M) + some)] = C(s, j - 2); //Last row (M-1)  Amit::removed negative Now modified to M-2 the 2nd last row
			//Now keep a copy of coefficients of the objective function
			MAT[(int) (((M-1) + j * M) + some)] = C(s, j - 2); //slack is already zero as initialized
			//}
		}
	}

	//Debugging ---------------------------------------
	/*printf("\n******************* MAT tableau *******************\n");
	for (int s = 0; s < N_S; s++) {
		unsigned int some=M * N * s;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				std::cout<<MAT[(int)(i+ j*M +some)]<<"  ";
			}
			printf("\n");
		}
		printf("\n");
	}*/
	// ---------------------------------------
	std::vector<int> rem;
	for (int i = 0; i < N_C; i++) {
		//std::cout<<BoundValue[i]<<"\n";
		if (BoundValue[i] < 0) {
			rem.push_back(i);
		}
	}

//	std::cout<<"Number of Artificial Variables = "<< rem.size()<<"\n";
	int nc = N + rem.size();
	
	threads_per_block = 32 * (nc / 32) + 32; //if count equal 0 than nc=N so works for all Model
	if (threads_per_block > props.maxThreadsPerBlock) //Assuming maximum threads supported by CC is 1024
		threads_per_block = props.maxThreadsPerBlock;

	int *R_index;	//reduction data
	float *R_data;	//reduction index
	err = cudaMalloc((void **) &R_data, C1.size1() * threads_per_block * sizeof(float));//C1.size1() * 96 being the maximum threads
	err = cudaMalloc((void **) &R_index, C1.size1() * threads_per_block * sizeof(int));//C1.size1() being the number of LPs
	err = cudaMalloc((void **) &G_R, N_S * sizeof(float));//Doing it here for the First Time

	//printf("CUDA malloc R_index: %s\n", cudaGetErrorString(err));
	//std::cout << "Number of threads per block = " << threads_per_block << "\n";

	if (rem.size() > 0) {
		//std::cout << "Simplex -Non-Basic Feasible Solution\n";
		N_MAT = (float *) calloc(N_S * M * nc, sizeof(float)); //initialized to zero (this tableau include artificial variables)

	/*
	 *Copied only RHS of all constraints to N_MAT from MAT
	 * This is done to copy b_i column from MAT into last column of N_MAT
	 */
#pragma omp parallel for
	for (int i = 0; i < N_S; i++) {
		int base = i * M * N;//base of every LP in MAT tableau
		int basen = i * M * nc;//base of every LP in N_MAT tableau (this include artificial variables)
		for (int j = 0; j < M; j++) {	//from every row/constraints
			//base=i*M*N;
			N_MAT[j + ((nc - 1) * M) + basen] = MAT[j + ((N - 1) * M) + base]; // N_MAT[lastCol] = MAT[lastCol]
		}
		for (int j = 2; j < (nc-1); j++) {	//from every column
			if (j<((No_O + 3 + No_C)-1))
				//N_MAT[(int) ((M-1) + j * M + some)] = MAT[(int) ((M - 1) + j * M + base)];//original and slack variables
				;//for Phase-I //original and slack variables will remain 0//N_MAT[(int) ((M-1) + j * M + some)] = 0;
			else
				N_MAT[(int) ((M-1) + j * M + basen)] = -1; //artificial variable
		}
	}

	//Debugging
/*	printf("\n******************* N_MAT tableau *******************\n");
	for (int s = 0; s < N_S; s++) {
		int basen = s * M * nc;//base of every LP in N_MAT tableau (this include artificial variables)
		for (int i = 0; i < M; i++) {	//from every row/constraints
			for (int j = 0; j < nc; j++) {	//from every column
				std::cout<<N_MAT[(i + (j * M) + basen)]<<" ";
			}
			printf("\n");
		}
		printf("\n");
	}*/
	// Verified upto here CPU code is fine


	//Creating Artificial Variables
#pragma omp parallel for
	for (int k = 0; k < N_S; k++) {
		bool once=false;
		int artif=0, ch;
		int base = k * M * N;//base of every LP in MAT tableau
		int basen = k * M * nc;//base of every LP in N_MAT tableau (this include artificial variables)
		for ( int i = 0; i < (M-1); i++) { //for every row including 2nd last row of the tableau :: leave last row
			ch = 0;
			for ( int j = 0; j < nc; j++) {  //for every column of the MAT and N_MAT tableau
				if (MAT[i + ((N - 1) * M) + base] < 0) { //this indicates the negative b_i from the MAT tableau
					if ((j >= (N - 1)) && (j < (nc - 1))) { //this indicate all columns that represent artificial variables
						if (!ch) {
							float v = N_MAT[(unsigned int)((i-1) + (j * M) + basen)]; //why (i-1)?
							if((once)&&(v==1)){ // computing v is meaningless since once is false and not made true anywhere so this will always be false
									N_MAT[(i + (j+1) * M) + basen] = 1;//so this block will never be executed ToDo:: can be skipped
							} else {
								N_MAT[(i + j * M) + basen] = 1;
							}
							ch = 1;	//this will allow populating 1, diagonally in artificial variables
						}
					} else if (j == (nc - 1)) { //this indicate the last column of N_MAT tableau which is b_i's
						N_MAT[(i + j * M) + basen] = -1 * N_MAT[(i + j * M) + basen]; //negating b_i's
					} else if (j == 1) { //the extra temporary working column
						N_MAT[(i + j * M) + basen] = -1; //why populate -1 only for negative b_i's in the extra column? May be used as coefficient of artificial variables
					} else if (j == 0) { //first index column
						//NOTE: Binayak used table index as variable indexing so it is 1-based Indexing
						//ToDo:: Amit detected Bugged here in index computation for Artificial variables
						//N_MAT[((i + j * M)) + basen] = (N + i)-2;//computes the index of artificial variables as n+m+a where size of vars, slacks and artificial are n,m and a respectively
						N_MAT[((i + j * M)) + basen] = (N + artif)-2;//increase index only when found artificial variable and not for every row i.
					//	std::cout<<" artif = " <<artif;
						artif++;//increase for next artificial variable found
						//std::cout<<" (N + i)-2 = " <<(N + i)-2;
					} else if (j > 1) { //negated all variables and slacks (only non-basic excluding artificial)
						N_MAT[(i + j * M) + basen] = -1 * (MAT[(i + j * M) + base]);
					}
				} else if ((i != (M - 2)) && (j < (N - 1))) { //except last row and last column of MAT i.e. b_i's
					N_MAT[(i + j * M) + basen] = MAT[(i + j * M) + base];//copy into N_MAT as it is
				} else if (i == (M - 2)) {
					if ((j >= (N - 1)) && (j < (nc - 1))) {
						N_MAT[(i + j * M) + basen] = -1; //ALL artificial variable coefficient is assigned -1
					}
				}
			}
		}
	}

	//Debugging
/*
	printf("\n******************* N_MAT tableau *******************\n");
	for (int s = 0; s < N_S; s++) {
		int basen = s * M * nc;//base of every LP in N_MAT tableau (this include artificial variables)
		for (int i = 0; i < M; i++) {	//from every row/constraints
			for (int j = 0; j < nc; j++) {	//from every column
				std::cout<<N_MAT[i + (j * M) + basen]<<" ";
			}
			printf("\n");
		}
		printf("\n");
	}
*/


//Creation of Last Row or Z-Value(Zj-Cj)
#pragma omp parallel for
	for (int k = 0; k < N_S; k++) {
		//int sum = 0;
		//base = k * M * N;
		int basen = k * M * nc;
		for (int k1 = 2; k1 < nc; k1++) {//for all columns upto b_i from column 2
			float sum = 0.0; //reset for every column k1 (objective function value)
			for (int j = 0; j < (M - 2); j++) { //for all rows except the 2nd last row for which this computation is performed and also last row
				sum = sum + (N_MAT[(j + k1 * M) + basen] * N_MAT[(j + 1 * M) + basen]); // column 1 currently contains -1, the coefficient of artificial variables
			}
			//std::cout << sum << "-"	<< N_MAT[((M - 1) + k1 * M) + basen];
			N_MAT[((M - 2) + k1 * M) + basen] = sum - N_MAT[((M - 2) + k1 * M) + basen]; //formula Zj - Cj
		}
	}
	//cudaEvent_t start, stop;
	//Debugging ----------------------------
/*	std::cout << "\nSimplex AFTER CREATION OF Z before sending to GPU\n";
	for (int k = 0; k < N_S; k++) {
		int basen = k * M * nc;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < nc; j++) {
				std::cout << N_MAT[(i + j * M) + basen] << "  ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}*/
	// ----------------------------

//		std::cout << "Before Kernel Called 1\n";
	cudaMalloc((void **) &G_MAT, (N_S * M * nc * sizeof(float)));
	//printf("CUDA malloc G_MAT: %s\n", cudaGetErrorString(err));
	cudaMemcpy(G_MAT, N_MAT, (N_S * M * nc * sizeof(float)), cudaMemcpyHostToDevice); //copy N_MAT from host to device in G_MAT
	//printf("CUDA malloc N_MAT: %s\n", cudaGetErrorString(err));
	//	cudaMemcpy(G_Sel, Sel, sizeof(int), cudaMemcpyHostToDevice);
	//printf("CUDA malloc G_Sel: %s\n", cudaGetErrorString(err));

	//mykernel<<<N_S, threads_per_block>>>(G_MAT, M, nc, G_R, N_S, G_Sel, R_data, R_index);
	mykernel<<<N_S, threads_per_block>>>(G_MAT, M, nc, G_R, N_S, R_data,R_index);
	cudaDeviceSynchronize();
	cudaMemcpy(R, G_R, N_S * sizeof(float), cudaMemcpyDeviceToHost); //copy the reduced result in arrary R
	cudaMemcpy(N_MAT, G_MAT, (N_S * M * nc * sizeof(float)),cudaMemcpyDeviceToHost); //copy the current status of the G_MAT from device to N_MAT
	//for (int k = 0; k < N_S; k
/*	std::cout<< "\n***********Auxiliary SIMPLEX from GPU*************\n";
	for (int k = 0; k < N_S; k++) {
		// base=k*M*N;
		int basen = k * M * nc;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < nc; j++) {
				std::cout << N_MAT[(i + j * M) + basen] << "  ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}*/
	//	std::cout << "Result for Artificial\n";
#pragma omp parallel for
		for (int i = 0; i < N_S; i++) {
			int base = i * M * N;
			int basen = i * M * nc;
			for (int j = 0; j < M; j++) { //for every row
				for (int k = 0; k < N; k++) { //for each column in MAT
					if (N_MAT[j + 0*M + basen] == (k + 1)) { //column 0 i.e. (row,0) has index of variables starting from (1 to n+m+a) so k starts from (1 to N, N=n+m+a+3)
						//So this condition will be met every value of k. if variable indexing is correctly assigned (Note Artificial is in-correct)
						N_MAT[j + 1*M + basen] = MAT[(M - 1) + (2 + k) * M + base]; //in column 1 of N_MAT replacing original problem's objective coefficients
					}
				}
			}
		}
/*	std::cout<< "\n***********Auxiliary SIMPLEX from GPU After modification: N_MAT *************\n";
	for (int k = 0; k < N_S; k++) {
		int basen = k * M * nc;
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < nc; j++) {
				std::cout << N_MAT[(i + j * M) + basen] << "  ";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}*/

//std::cout<<"\nResult = "<<R[0]<<"\n";
#pragma omp parallel for
		for (int s = 0; s < N_S; s++) {
			if ((roundf(R[s]/10000)*10000) == 0) {
				//std::cout<<"\nInside ** IF*** Result = "<<R[0]<<"\n";
				//int sum = 0;
				int base = s * M * N;
				int basen = s * M * nc;
				for (int i = 0; i < N; i++) { //for each column i
					float sum = 0;
					for (int j = 0; j < (M - 1); j++) { //for every row j except the new last row of obj. coefficients
						if ((j < (M - 2))) { //except the last row
							if (i != (N - 1)) { //except the last column ie b_i's
								//std::cout<<N_MAT[(j+(i*M))+basen]<<"*"<<N_MAT[(j+(1*M))+basen]<<std::endl;
								sum = sum + (N_MAT[(j + (i * M)) + basen] * N_MAT[(j + (1 * M)) + basen]); //sum = sum + N_MAT[row,1] * N_MAT[row,i]
								MAT[(j + (i * M)) + base] = N_MAT[(j + (i * M)) + basen]; //copy data from N_MAT to MAT for every column i, as row by row (row is j)
							} else if (i == N - 1) { //for the last column ie b_i's
								sum = sum + (N_MAT[(j + (nc - 1) * M) + basen] * N_MAT[(j + (1 * M)) + basen]); //sum is for objective value column, the product of b_i's and coefficient's in column 1
								MAT[(j + (i * M)) + base] = N_MAT[(j + (nc - 1) * M) + basen]; //copy data of b_i's from N_MAT to MAT
							}
						}
						//if (j == (M - 1)) { //for the last row
						if (j == (M - 2)) { //for the 2nd last row
							if (i > 1) { // for all column from variables to slack variables excluding artificial variables
								//std::cout<<sum<<" And "<<MAT[(j+(i*M))+base]<<std::endl;
								//MAT[(j + (i * M)) + base] = MAT[(j + (i * M)) + base] + (-1) * sum; // Zj = Zj - Cj
								MAT[(j + (i * M)) + base] = sum + (-1 * MAT[(j + (i * M)) + base]); // Zj = Zj - Cj  Amit:: Corrected
							}
						}
					}
				}
			} else
				std::cout<<"The problem is Infeasible !!!\n";
		}
		cudaFree (G_MAT);
		//		cudaFree(G_R);
		//cudaFree(G_Sel);
		//cudaDeviceSynchronize();
		//		cudaMalloc((void **) &G_R, N_S * sizeof(float));
		cudaMalloc((void **) &G_MAT, (N_S * M * N * sizeof(float)));
		// printf("CUDA malloc G_MAT: %s\n", cudaGetErrorString(err));
		cudaMemcpy(G_MAT, MAT, (N_S * M * N * sizeof(float)), cudaMemcpyHostToDevice); //Now copy MAT to device G_MAT
		//printf("CUDA malloc N_MAT: %s\n", cudaGetErrorString(err));
		//	cudaMemcpy(G_Sel, Sel, sizeof(int), cudaMemcpyHostToDevice);
		//printf("CUDA malloc G_Sel: %s\n", cudaGetErrorString(err));
//std::cout<<"Kernel Called 2\n";
	//Debugging
/*
		std::cout<< "\n***********Before SIMPLEX sent to GPU: MAT *************\n";
		for (int k = 0; k < N_S; k++) {
			int basen = k * M * N;
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < N; j++) {
					std::cout << MAT[(int)(i + (j * M) + basen)] << "  ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
*/

		mykernel<<<N_S, threads_per_block>>> (G_MAT, M, N, G_R, N_S, R_data, R_index);
		cudaDeviceSynchronize();
		cudaMemcpy(R, G_R, N_S * sizeof(float), cudaMemcpyDeviceToHost); //store the result in arrary R

		cudaMemcpy(MAT, G_MAT, (N_S * M * N * sizeof(float)), cudaMemcpyDeviceToHost);
		std::cout<< "\n***********Final SIMPLEX from GPU*************\n";
		for (int k = 0; k < N_S; k++) {
			int basen = k * M * N;
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < N; j++) {
					std::cout << MAT[(int)(i + (j * M) + basen)] << "  ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}

		//std::cout
		// << "***********Final SIMPLEX from GPU*************\n Time took:\n";*/
	}
	//cudaFree(G_MAT);
	//cudaFree(G_Sel);
	//cudaFree(G_R);
	cudaFree(R_index);	//Only to synchronize with the cudamemcpy
	//cudaFree(R_data);	//Only to synchronize with the cudamemcpy
}

