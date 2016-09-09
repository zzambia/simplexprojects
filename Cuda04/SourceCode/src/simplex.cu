#include "simplex.cuh"
#include<omp.h>
#include "iostream"

//LPC with Stream
//Method : To determine Pivot Column i.e., the entering variable we use the most negative value approach
//Race Condition completely avoided
//Both Reduction implemented- One for finding Pivot column and the other for finding Pivot Row
//Implemented Reduction. But without Streams.


//__global__ void mykernel(float *S_MAT, int S_row, int S_col, float *Result, int S_N, int *S_Sel, float *R_data, int *R_index) {
//1st Method : Most Negative Value approach
__global__ void mykernelCurrent(float *S_MAT, int S_row, int S_col, float *Result,int S_N, float *R_data, int *R_index,int offset_res) {
	int index = offset_res + blockIdx.x;
	if (index < (offset_res + S_N)) {
		int tid;
		int i; // used for for index
		unsigned int temp_index;
		unsigned int temp_index1;
		int base = index * S_row * S_col;
		int R_base = index * blockDim.x;  // blockDim.x = 96
		__shared__ bool c;
		__shared__ int rm;
		__shared__ int row;	//pivotRow
		__shared__ int pivotCol;//pivotCol this can remove global variable S_Sel

		int col = 1;
		__shared__ int remember[1024]; //Found a column which is negative but theta/Min has no positive value
		__shared__ float col1[1024];	//pivotColumn
		/*************/
		if (threadIdx.x == 0) {
			c = false;
			rm = 0;
			row = -1;		//pivotRow
			pivotCol = -1;
		}
		__syncthreads();
		while (!c) {
			__syncthreads();
			int Last_row = S_row - 1;
			__syncthreads();
			//   ***************** Get_Pivot function begins  *****************
			// ******** First Reduction Begins **********
			//using reduction to compute min and newpivotcol
			__shared__ int notEntered;
//not used			__shared__ float minValue;
			__shared__ int newpivotcol;
			if (threadIdx.x == 0) {
//not used				minValue = 0;
				newpivotcol = -1;
				notEntered = 1;
				c = true;
			}
			__syncthreads();	//making sure newpivotcol is initialised to -1
			// Since keeping limit only upto (S_col - 1) which is not equal to BLOCK_SIZE creates problem
			// in using syncthreads() inside Reduction for-loop so use all threads(all R_data)
			//int data_size = (S_col - 1) - 2;
			int data_size = blockDim.x;
			tid = threadIdx.x;
			if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)) {
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

			tid = threadIdx.x;
			for (i = (data_size / 2); i > 0;) {
				if (tid < i) {
					//	if ((R_data[tid] >= R_data[tid + s]) && ((R_data[tid + s] < 0) && (R_data[tid] < 0))){
					//(R_data[tid + R_base] < 0) && (R_data[tid + R_base + i] < 0)&&
					if (R_data[tid + R_base] > R_data[tid + R_base + i]) { //is right-side value small?
						if (R_data[tid + R_base + i] <= -0.000001) {	//only if the value on the right-side is -ive
							R_data[tid + R_base] = R_data[tid + R_base + i];//put the smaller value to left-side
							R_index[tid + R_base] = R_index[tid + R_base + i];

							//notEntered = false;  //race condition avoided
						//	notEntered = 0;  //race condition avoided
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
				if (notEntered == false) {
//not used					minValue = R_data[R_base];
					newpivotcol = R_index[R_base];
					//printf("\nminValue = %f newpivotcol = %d ", minValue,newpivotcol);
				}
			}
			__syncthreads(); //waiting for all threads to have same newpivotcol value
			if (newpivotcol == -1) {//All Threads will follow the Same path so no issue with divergence
				//return -2;
				row = -2;
			} else {
				// ********** Second Reduction Process ******
				//in order to avoid global memory transfer:: Using the same R_data and R_index global memory
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
				//if (threadIdx.x >= 0 && threadIdx.x < Last_row) {
				if (threadIdx.x < Last_row) {
					k1 = threadIdx.x;	//here k1 =0 to Last_row only
					//for (int k1 = 0; k1 < Last_row; k1++) {	//Last_row = (S_row - 1)
					int temp_index2 = newpivotcol * S_row + k1 + base;
					temp_index1 = k1 + (S_col - 1) * S_row + base; //avoiding re-computation
					if ((S_MAT[temp_index2] > 0) && (S_MAT[temp_index1] > 0)) {
						R_data[k1 + R_base] = S_MAT[temp_index1]
								/ S_MAT[temp_index2];
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
				__syncthreads(); //here have all values from 0 to BLOCK_SIZE
				//Now find the minValue and its index from R_data and R_index using Reduction
				//int data_size = Last_row;
				int data_size2 = blockDim.x; //Now it is Block_Size
				// ***** Second Reduction on R_data and R_index ****
				//	if (threadIdx.x >= 0 && threadIdx.x < Last_row) {	//Now for all threads
				tid = threadIdx.x;
				for (int s = (data_size2 / 2); s > 0;) {
					if (tid < s) {
						int indexValue2 = tid + R_base;
						if (R_data[indexValue2] >= R_data[indexValue2 + s]) {
							R_data[indexValue2] = R_data[indexValue2 + s];
							R_index[indexValue2] = R_index[indexValue2 + s];
							//notEntered2 = false;
							//notEntered2 = 0;//check using atomic
							int local_notEntered2;
							local_notEntered2 = *(volatile int*) &notEntered2;
							atomicCAS(&notEntered2, local_notEntered2, 0);

						}
					}
//					if (tid == 0)
//						printf("Data_size = %d ", s);
					__syncthreads();	//This creates unpredictable behaviour
					s >>= 1;
					if ((s != 1) && (s % 2) != 0) {	//if s is odd
						s = s + 1;
					}
				}
				//if (notEntered2 == false && tid == 0) {
				if (tid == 0) {
					if (notEntered2 == false) {
						row_min = R_data[R_base];
						row_num = R_index[R_base];
						//printf("R_Data = %f R_Index = %d", R_data[R_base], R_index[R_base]);
					}
				}
				__syncthreads(); // Looks like this can be skipped
				//	}
				//	__syncthreads();	//here we have Row_min and newpivotRow
				// ********** Second Reduction on R_data and R_index ******
				if (threadIdx.x == 0) {
					pivotCol = newpivotcol;
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

			//col = S_Sel[index];
			//col = *S_Sel;
			col = pivotCol;
			if (row > -1) {
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int i1 = 2; i1 < S_col; i1++) {		//Data Parallel section 1
					if (tid == remember[tid - 2]) {
						temp_index = (S_row - 1) + (tid * S_row) + base; //avoiding re-computation
						S_MAT[temp_index] = -1 * S_MAT[temp_index]; //replacing back to original
					}
				}		//Data Parallel section 1 done
				__syncthreads();
				tid = threadIdx.x;
				//if (threadIdx.x >= 0 && threadIdx.x < S_row) {
				if (threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data Parallel section 2
					col1[tid] = S_MAT[(tid + col * S_row) + base];//keeping the old pivotcol coeff
				}	//Data Parallel section 2 done
				__syncthreads();

				unsigned int temp_row_base = row + base;//avoiding re-computation
				S_MAT[temp_row_base + S_row] =
						S_MAT[temp_row_base + col * S_row];
				//S_MAT[temp_row_base] = col - 1;
				S_MAT[row + base] = col - 1;//now temp_row_base is not required
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int j = 2; j < S_col; j++){		//Data Parallel section 3
					unsigned int row_base = row + base;	//avoiding re-computation
					temp_index = row_base + (tid * S_row);//avoiding re-computation
					S_MAT[temp_index] = S_MAT[temp_index] / col1[row];//S_MAT[row_base + S_row];
					//S_MAT[temp_index] = S_MAT[temp_index] / S_MAT[row_base + S_row];
				}		//Data Parallel section 3 done
				__syncthreads();
				//printf("Row here = %d",row);
				tid = threadIdx.x;
				//if (threadIdx.x >= 0 && threadIdx.x < S_row) {
				if (threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data parallel section 4
					/*for (i = 2; i < S_col; i++) {
						if (tid != row) {
							temp_index1 = i * S_row + base;
							temp_index = tid + temp_index1;
							S_MAT[temp_index] = S_MAT[temp_index]
									- (col1[tid] * S_MAT[row + temp_index1]);
						} else {
							break;
						}
					}*/


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
						if (S_MAT[((S_row - 1) + i * S_row) + base] < 0) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();

			} else if (row == -1) {
				if (threadIdx.x == 0) {
					c = true;
					remember[rm] = col;
					rm++;
				}
				__syncthreads();

				temp_index = (S_row - 1) + (col * S_row) + base; //if col==-1 than problem for base==0 i.e. temp_index==-1
				S_MAT[temp_index] = -1 * S_MAT[temp_index];	//remembering by making positive
				//if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)){
				// tid = threadIdx.x;
				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {		//Data parallel 5
						if ((S_MAT[((S_row - 1) + i * S_row) + base] < 0)) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();
			}
		} //end of while
		__syncthreads();
		if (threadIdx.x == 0) {
			//printf("Value = %f ",S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base]);
			Result[index] = S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base];
			//printf("blockIdx.x = %d   Result[index] = %f ",index,Result[index]);
		}
	}
}



//1st Method : Most Negative Value approach (//Works even for Large arguments)
__global__ void mykernelWorks(float *S_MAT, int S_row, int S_col, float *Result,
		int S_N, float *R_data, int *R_index) {
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
		__shared__ int pivotCol;//pivotCol this can remove global variable S_Sel

		int col = 1;
		__shared__ int remember[1024]; //Found a column which is negative but theta/Min has no positive value
		__shared__ float col1[1024];	//pivotColumn
		/*************/
		if (threadIdx.x == 0) {
			c = false;
			rm = 0;
			row = -1;		//pivotRow
			pivotCol = -1;
		}
		__syncthreads();
		while (!c) {
			__syncthreads();
			int Last_row = S_row - 1;
			//   ***************** Get_Pivot function begins  *****************
			// ******** First Reduction Begins **********
			//using reduction to compute min and newpivotcol
			__shared__ int notEntered;
			__shared__ float minValue;
			__shared__ int newpivotcol;
			if (threadIdx.x == 0) {
				minValue = 0;
				newpivotcol = -1;
				notEntered = 1;
				c = true;
			}
			__syncthreads();	//making sure newpivotcol is initialised to -1
			// Since keeping limit only upto (S_col - 1) which is not equal to BLOCK_SIZE creates problem
			// in using syncthreads() inside Reduction for-loop so use all threads(all R_data)
			//int data_size = (S_col - 1) - 2;
			int data_size = blockDim.x;
			tid = threadIdx.x;
			if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)) {
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

			tid = threadIdx.x;
			for (i = (data_size / 2); i > 0;) {
				if (tid < i) {
					//	if ((R_data[tid] >= R_data[tid + s]) && ((R_data[tid + s] < 0) && (R_data[tid] < 0))){
					//(R_data[tid + R_base] < 0) && (R_data[tid + R_base + i] < 0)&&
					if (R_data[tid + R_base] > R_data[tid + R_base + i]) { //is right-side value small?

						//if (R_data[tid + R_base + i] == -0.000000)
							 //R_data[tid + R_base + i] = 0.0;
						//This modification was required for large batch-size(1500) for LP dim 300 and above
						if (R_data[tid + R_base + i] <= -0.000001 ) {	//only if the value on the right-side is -ive
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
				if (notEntered == false) {
					minValue = R_data[R_base];
					newpivotcol = R_index[R_base];
					//printf("\nminValue = %f newpivotcol = %d ", minValue,newpivotcol);
				}
			}
			__syncthreads(); //waiting for all threads to have same newpivotcol value
			if (newpivotcol == -1) {//All Threads will follow the Same path so no issue with divergence
				//return -2;
				row = -2;
			} else {
				// ********** Second Reduction Process ******
				//in order to avoid global memory transfer:: Using the same R_data and R_index global memory
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
					//This modification was required for large batch-size(5000) for LP dim 300 and above
					if ((S_MAT[temp_index2] >= -0.000001) && (S_MAT[temp_index1] >= -0.000001)) {
						R_data[k1 + R_base] = S_MAT[temp_index1]/ S_MAT[temp_index2];
						//R_data[k1 + R_base] = roundf(((S_MAT[temp_index1]/ S_MAT[temp_index2])/1000000)*1000000);
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
				__syncthreads(); //here have all values from 0 to BLOCK_SIZE
				//Now find the minValue and its index from R_data and R_index using Reduction
				//int data_size = Last_row;
				int data_size2 = blockDim.x; //Now it is Block_Size
				// ***** Second Reduction on R_data and R_index ****
				//	if (threadIdx.x >= 0 && threadIdx.x < Last_row) {	//Now for all threads
				tid = threadIdx.x;
				for (int s = (data_size2 / 2); s > 0;) {
					if (tid < s) {
						int indexValue2 = tid + R_base;
						if (R_data[indexValue2] >= R_data[indexValue2 + s]) {
							R_data[indexValue2] = R_data[indexValue2 + s];
							R_index[indexValue2] = R_index[indexValue2 + s];
							//notEntered2 = false;

							//notEntered2 = 0;
							int local_notEntered2;
							local_notEntered2 = *(volatile int*) &notEntered2;
							atomicCAS(&notEntered2, local_notEntered2, 0);
						}
					}
//					if (tid == 0)
//						printf("Data_size = %d ", s);
					__syncthreads();	//This creates unpredictable behaviour
					s >>= 1;
					if ((s != 1) && (s % 2) != 0) {	//if s is odd
						s = s + 1;
					}
				}
				//if (notEntered2 == false && tid == 0) {
				if (tid == 0) {
					if (notEntered2 == false) {
						row_min = R_data[R_base];
						row_num = R_index[R_base];
						//printf("R_Data = %f R_Index = %d", R_data[R_base], R_index[R_base]);
					}
				}
				__syncthreads(); // Looks like this can be skipped
				//	}
				//	__syncthreads();	//here we have Row_min and newpivotRow
				// ********** Second Reduction on R_data and R_index ******
				if (threadIdx.x == 0) {
					pivotCol = newpivotcol;
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

			//col = S_Sel[index];
			//col = *S_Sel;
			col = pivotCol;
			//printf("Row= %d col = %d\n",row,col);
			if (row > -1) {
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int i1 = 2; i1 < S_col; i1++) {		//Data Parallel section 1
					if (tid == remember[tid - 2]) {
						temp_index = (S_row - 1) + (tid * S_row) + base; //avoiding re-computation
						//if (S_MAT[temp_index] == -0.0)
							//S_MAT[temp_index] = -1 * 0.0; //replacing back to original
						//else
							S_MAT[temp_index] = -1 * S_MAT[temp_index]; //replacing back to original
					}
				}		//Data Parallel section 1 done
				__syncthreads();
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data Parallel section 2
					col1[tid] = S_MAT[(tid + col * S_row) + base];//keeping the old pivotcol coeff
				}	//Data Parallel section 2 done
				__syncthreads();

				unsigned int temp_row_base = row + base;//avoiding re-computation
				S_MAT[temp_row_base + S_row] =
						S_MAT[temp_row_base + col * S_row];
				//S_MAT[temp_row_base] = col - 1;
				S_MAT[row + base] = col - 1;//now temp_row_base is not required
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int j = 2; j < S_col; j++){		//Data Parallel section 3
					unsigned int row_base = row + base;	//avoiding re-computation
					temp_index = row_base + (tid * S_row);//avoiding re-computation
					S_MAT[temp_index] = S_MAT[temp_index] / col1[row];//S_MAT[row_base + S_row];
					//S_MAT[temp_index] = roundf(((S_MAT[temp_index] / col1[row])/1000000)*1000000);//S_MAT[row_base + S_row];
					//S_MAT[temp_index] = S_MAT[temp_index] / S_MAT[row_base + S_row];
				}		//Data Parallel section 3 done
				__syncthreads();
				//printf("Row here = %d",row);
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data parallel section 4
					for (i = 2; i < S_col; i++) {
						if (tid != row) {
							temp_index1 = i * S_row + base;
							temp_index = tid + temp_index1;
							float zeroTemp;
							zeroTemp = col1[tid] * S_MAT[row + temp_index1];
							S_MAT[temp_index] = S_MAT[temp_index] - zeroTemp;
							//S_MAT[temp_index] = roundf(((S_MAT[temp_index] - zeroTemp)/1000000)*1000000);
							//if (S_MAT[temp_index] == -0.0)
								//S_MAT[temp_index]= 0.0;

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
						if (S_MAT[((S_row - 1) + i * S_row) + base] < 0) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();

			} else if (row == -1) {
				if (threadIdx.x == 0) {
					c = true;
					remember[rm] = col;
					rm++;
				}
				__syncthreads();

				temp_index = (S_row - 1) + (col * S_row) + base; //if col==-1 than problem for base==0 i.e. temp_index==-1
				//if (S_MAT[temp_index] == -0.0)
					//S_MAT[temp_index] = 0;	//remembering by making positive
				//else
					S_MAT[temp_index] = -1 * S_MAT[temp_index];	//remembering by making positive
				//if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)){
				// tid = threadIdx.x;
				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {		//Data parallel 5
						if ((S_MAT[((S_row - 1) + i * S_row) + base] < 0)) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();
			}

		} //end of while
		__syncthreads();

		if (threadIdx.x == 0) {
			//printf("Value = %f ",S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base]);
			Result[index] = S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base];
			//printf("\nResult Inside Kernel: %f \n",Result[index]);
		}
	}
}

//1st Method : Most Negative Value approach
__global__ void mykernel(float *S_MAT, int S_row, int S_col, float *Result,int S_N, float *R_data, int *R_index,int offset_res) {
	int index = offset_res + blockIdx.x;
	if (index < (offset_res + S_N)) {
		int tid;
		int i; // used for for index
		unsigned int temp_index;
		unsigned int temp_index1;
		int base = index * S_row * S_col;
		int R_base = index * blockDim.x;  // blockDim.x = 96
		__shared__ bool c;
		__shared__ int rm;
		__shared__ int row;	//pivotRow
		__shared__ int pivotCol;//pivotCol this can remove global variable S_Sel

		int col = 1;
		__shared__ int remember[1024]; //Found a column which is negative but theta/Min has no positive value
		__shared__ float col1[1024];	//pivotColumn
		/*************/
		if (threadIdx.x == 0) {
			c = false;
			rm = 0;
			row = -1;		//pivotRow
			pivotCol = -1;
		}
		__syncthreads();
		while (!c) {
			__syncthreads();
			int Last_row = S_row - 1;
			//   ***************** Get_Pivot function begins  *****************
			// ******** First Reduction Begins **********
			//using reduction to compute min and newpivotcol
			__shared__ int notEntered;
			__shared__ float minValue;
			__shared__ int newpivotcol;
			if (threadIdx.x == 0) {
				minValue = 0;
				newpivotcol = -1;
				notEntered = 1;
				c = true;
			}
			__syncthreads();	//making sure newpivotcol is initialised to -1
			// Since keeping limit only upto (S_col - 1) which is not equal to BLOCK_SIZE creates problem
			// in using syncthreads() inside Reduction for-loop so use all threads(all R_data)
			//int data_size = (S_col - 1) - 2;
			int data_size = blockDim.x;
			tid = threadIdx.x;
			if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)) {
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

			tid = threadIdx.x;
			for (i = (data_size / 2); i > 0;) {
				if (tid < i) {
					//	if ((R_data[tid] >= R_data[tid + s]) && ((R_data[tid + s] < 0) && (R_data[tid] < 0))){
					//(R_data[tid + R_base] < 0) && (R_data[tid + R_base + i] < 0)&&
					if (R_data[tid + R_base] > R_data[tid + R_base + i]) { //is right-side value small?

						//if (R_data[tid + R_base + i] == -0.000000)
							 //R_data[tid + R_base + i] = 0.0;
						if (R_data[tid + R_base + i] <= -0.000001 ) {	//only if the value on the right-side is -ive
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
				__syncthreads();
				i >>= 1;
				if ((i != 1) && (i % 2) != 0) {	//if s is odd
					i = i + 1;
				}
			}
			// if (notEntered == false && tid == 2) { // tid==0 is always true if minValue is still -1 then what?
			if (threadIdx.x == 0) { // tid==0 is always true if minValue is still -1 then what?
				if (notEntered == false) {
					minValue = R_data[R_base];
					newpivotcol = R_index[R_base];
					//printf("\nminValue = %f newpivotcol = %d ", minValue,newpivotcol);
				}
			}
			__syncthreads(); //waiting for all threads to have same newpivotcol value
			//		__syncthreads();	//here we have min and newpivotcol
			// ********* First Reduction Ends *************
			//  ******** Second Reduction Begins **********
			if (newpivotcol == -1) {//All Threads will follow the Same path so no issue with divergence
				//return -2;
				row = -2;
			} else {
				// ********** Second Reduction Process ******
				//in order to avoid global memory transfer:: Using the same R_data and R_index global memory
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
					if ((S_MAT[temp_index2] > 0) && (S_MAT[temp_index1] > 0)) {
						R_data[k1 + R_base] = S_MAT[temp_index1]/ S_MAT[temp_index2];
						//R_data[k1 + R_base] = roundf(((S_MAT[temp_index1]/ S_MAT[temp_index2])/1000000)*1000000);
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
				__syncthreads(); //here have all values from 0 to BLOCK_SIZE
				//Now find the minValue and its index from R_data and R_index using Reduction
				//int data_size = Last_row;
				int data_size2 = blockDim.x; //Now it is Block_Size
				// ***** Second Reduction on R_data and R_index ****
				//	if (threadIdx.x >= 0 && threadIdx.x < Last_row) {	//Now for all threads
				tid = threadIdx.x;
				for (int s = (data_size2 / 2); s > 0;) {
					if (tid < s) {
						int indexValue2 = tid + R_base;
						if (R_data[indexValue2] >= R_data[indexValue2 + s]) {
							R_data[indexValue2] = R_data[indexValue2 + s];
							R_index[indexValue2] = R_index[indexValue2 + s];
							//notEntered2 = false;

							//notEntered2 = 0;
							int local_notEntered2;
							local_notEntered2 = *(volatile int*) &notEntered2;
							atomicCAS(&notEntered2, local_notEntered2, 0);
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
					if (notEntered2 == false) {
						row_min = R_data[R_base];
						row_num = R_index[R_base];
						//printf("R_Data = %f R_Index = %d", R_data[R_base], R_index[R_base]);
					}
				}
				__syncthreads(); // Looks like this can be skipped
				//	}
				//	__syncthreads();	//here we have Row_min and newpivotRow
				// ********** Second Reduction on R_data and R_index ******
				if (threadIdx.x == 0) {
					pivotCol = newpivotcol;
					if (row_min == INT_MAX) {
						row = -1;
					}
					if ((row_min != INT_MAX) && (row_num != -1)) {
						row = row_num;
					}
				}
				__syncthreads(); // Looks like this can be skipped
			} //end of else of newpivotcol == -1
			__syncthreads(); // Looks like this can be skipped but here we have row synchronized
			//  ******** Second Reduction Ends **********
			//   ***************** Get_Pivot function ends  *****************
			col = pivotCol;
			if (row > -1) {
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int i1 = 2; i1 < S_col; i1++) {		//Data Parallel section 1
					if (tid == remember[tid - 2]) {
						temp_index = (S_row - 1) + (tid * S_row) + base; //avoiding re-computation
						S_MAT[temp_index] = -1 * S_MAT[temp_index]; //replacing back to original
					}
				}		//Data Parallel section 1 done
				__syncthreads();
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data Parallel section 2
					col1[tid] = S_MAT[(tid + col * S_row) + base];//keeping the old pivotcol coeff
				}	//Data Parallel section 2 done
				__syncthreads();

				unsigned int temp_row_base = row + base;//avoiding re-computation
				S_MAT[temp_row_base + S_row] =
						S_MAT[temp_row_base + col * S_row];
				//S_MAT[temp_row_base] = col - 1;
				S_MAT[row + base] = col - 1;//now temp_row_base is not required
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int j = 2; j < S_col; j++){		//Data Parallel section 3
					unsigned int row_base = row + base;	//avoiding re-computation
					temp_index = row_base + (tid * S_row);//avoiding re-computation
					S_MAT[temp_index] = S_MAT[temp_index] / col1[row];//S_MAT[row_base + S_row];
				}		//Data Parallel section 3 done
				__syncthreads();
				//printf("Row here = %d",row);
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
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
				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {
						if (S_MAT[((S_row - 1) + i * S_row) + base] < 0) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();

			} else if (row == -1) {
				if (threadIdx.x == 0) {
					c = true;
					remember[rm] = col;
					rm++;
				}
				__syncthreads();

				temp_index = (S_row - 1) + (col * S_row) + base; //if col==-1 than problem for base==0 i.e. temp_index==-1
				S_MAT[temp_index] = -1 * S_MAT[temp_index];	//remembering by making positive

				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {		//Data parallel 5
						if ((S_MAT[((S_row - 1) + i * S_row) + base] < 0)) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();
			}

		} //end of while
		__syncthreads();

		if (threadIdx.x == 0) {
			Result[index] = S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base];
		//	printf("\nResult Inside Kernel: %f \n",Result[index]);
		}
	}
}



// 2nd Method:: Random negative value
__global__ void mykernel2(float *S_MAT, int S_row, int S_col, float *Result,
		int S_N, float *R_data, int *R_index) {
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
		__shared__ int pivotCol;//pivotCol this can remove global variable S_Sel

		int col = 1;
		__shared__ int remember[1024]; //Found a column which is negative but theta/Min has no positive value
		__shared__ float col1[1024];	//pivotColumn
		/*************/
		//if (threadIdx.x == 0) {
			c = false;
			rm = 0;
			row = -1;		//pivotRow
			pivotCol = -1;
		//}
		__syncthreads();
		while (!c) {
			//__syncthreads();
			int Last_row = S_row - 1;
			//   ***************** Get_Pivot function begins  *****************
			// ******** First Reduction Begins **********
			//using reduction to compute min and newpivotcol
			__shared__ float minValue;
			__shared__ int newpivotcol;
			if (threadIdx.x == 0) {
				minValue = 0;
				newpivotcol = -1;
				c = true;
			}
			__syncthreads();	//making sure newpivotcol is initialised to -1
			//for (int j = 2; j < S_col - 1; j++) {//only last row but all column
			if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)) {
				int j = threadIdx.x;
				unsigned int temp_index1 = Last_row + j * S_row + base; //avoiding re-computation
				if (S_MAT[temp_index1] < minValue) {
					//minValue = S_MAT[temp_index1];
					newpivotcol = j; //"Any(Random) negative coefficient rule"
					/*
					int local_NewPivotCol;
					local_NewPivotCol = *(volatile int*) &newpivotcol;
					atomicCAS(&newpivotcol, local_NewPivotCol, j);
					*/
					/*
					 http://stackoverflow.com/questions/27616417/cuda-is-there-any-way-to-prevent-other-threads-from-changing-a-shared-or-global
					 if (atomicCAS(&newpivotcol, local_NewPivotCol, j)==local_NewPivotCol){
					 //this thread won the write
					 printf("Thread ID = %d ",threadIdx.x);
					 }*/
					//break;
				}
			}
			__syncthreads(); //here we have min and newpivotcol

			//  ******** Second Reduction Begins **********

			if (newpivotcol == -1) { //All Threads will follow the Same path so no issue with divergence
				//return -2;
				row = -2;
			} else {
				// ********** Second Reduction Process ******
				//in order to avoid global memory transfer:: Using the same R_data and R_index global memory
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
					if ((S_MAT[temp_index2] > 0) && (S_MAT[temp_index1] > 0)) {
						R_data[k1 + R_base] = S_MAT[temp_index1]
								/ S_MAT[temp_index2];
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
				__syncthreads(); //here have all values from 0 to BLOCK_SIZE
				//Now find the minValue and its index from R_data and R_index using Reduction
				//int data_size = Last_row;
				int data_size2 = blockDim.x; //Now it is Block_Size
				/*if (threadIdx.x == 0) {
				 printf("\nR_data \n");
				 for (int x = 0; x < Last_row; x++)
				 printf("%f  ", R_data[x]);
				 printf("\nR_Index \n");
				 for (int x = 0; x < Last_row; x++)
				 printf("%d  ", R_index[x]);
				 printf("Data_size2 = %d ", data_size2);
				 }
				 __syncthreads();*/
				// ***** Second Reduction on R_data and R_index ****
				//	if (threadIdx.x >= 0 && threadIdx.x < Last_row) {	//Now for all threads
				tid = threadIdx.x;
				for (int s = (data_size2 / 2); s > 0;) {
					if (tid < s) {
						int indexValue2 = tid + R_base;
						if (R_data[indexValue2] >= R_data[indexValue2 + s]) {
							R_data[indexValue2] = R_data[indexValue2 + s];
							R_index[indexValue2] = R_index[indexValue2 + s];
							//notEntered2 = false;
							notEntered2 = 0;
							/*int local_notEntered2;
							local_notEntered2 = *(volatile int*) &notEntered2;
							atomicCAS(&notEntered2, local_notEntered2, 0);
							*/
						}
					}
//					if (tid == 0)
//						printf("Data_size = %d ", s);
					__syncthreads();	//This creates unpredictable behaviour
					s >>= 1;
					if ((s != 1) && (s % 2) != 0) {	//if s is odd
						s = s + 1;
					}
				}
				//if (notEntered2 == false && tid == 0) {
				if (tid == 0) {
					if (notEntered2 == false) {
						row_min = R_data[R_base];
						row_num = R_index[R_base];
						//printf("R_Data = %f R_Index = %d", R_data[R_base], R_index[R_base]);
					}
				}
				__syncthreads(); // Looks like this can be skipped
				//	}
				//	__syncthreads();	//here we have Row_min and newpivotRow
				// ********** Second Reduction on R_data and R_index ******
				if (threadIdx.x == 0) {
					pivotCol = newpivotcol;
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

			//col = S_Sel[index];
			//col = *S_Sel;
			col = pivotCol;
			if (row > -1) {
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int i1 = 2; i1 < S_col; i1++) {		//Data Parallel section 1
					if (tid == remember[tid - 2]) {
						temp_index = (S_row - 1) + (tid * S_row) + base; //avoiding re-computation
						S_MAT[temp_index] = -1 * S_MAT[temp_index]; //replacing back to original
					}
				}		//Data Parallel section 1 done
				__syncthreads();
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data Parallel section 2
					col1[tid] = S_MAT[(tid + col * S_row) + base];//keeping the old pivotcol coeff
				}	//Data Parallel section 2 done
				__syncthreads();

				unsigned int temp_row_base = row + base;//avoiding re-computation
				S_MAT[temp_row_base + S_row] =
						S_MAT[temp_row_base + col * S_row];
				//S_MAT[temp_row_base] = col - 1;
				S_MAT[row + base] = col - 1;//now temp_row_base is not required
				tid = threadIdx.x;
				if (threadIdx.x >= 2 && threadIdx.x < S_col) {
					//for (int j = 2; j < S_col; j++){		//Data Parallel section 3
					unsigned int row_base = row + base;	//avoiding re-computation
					temp_index = row_base + (tid * S_row);//avoiding re-computation
					S_MAT[temp_index] = S_MAT[temp_index] / col1[row];//S_MAT[row_base + S_row];
					//S_MAT[temp_index] = S_MAT[temp_index] / S_MAT[row_base + S_row];
				}		//Data Parallel section 3 done
				__syncthreads();
				//printf("Row here = %d",row);
				tid = threadIdx.x;
				if (threadIdx.x >= 0 && threadIdx.x < S_row) {
					//for (int i = 0; i < S_row; i++) {	//Data parallel section 4
					for (i = 2; i < S_col; i++) {
						if (tid != row) {
							temp_index1 = i * S_row + base;
							temp_index = tid + temp_index1;
							S_MAT[temp_index] = S_MAT[temp_index]
									- (col1[tid] * S_MAT[row + temp_index1]);
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
						if (S_MAT[((S_row - 1) + i * S_row) + base] < 0) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();

			} else if (row == -1) {
				if (threadIdx.x == 0) {
					c = true;
					remember[rm] = col;
					rm++;
				}
				__syncthreads();

				temp_index = (S_row - 1) + (col * S_row) + base; //if col==-1 than problem for base==0 i.e. temp_index==-1
				S_MAT[temp_index] = -1 * S_MAT[temp_index];	//remembering by making positive
				//if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)){
				// tid = threadIdx.x;
				if (threadIdx.x == 0) {
					for (i = 2; i < (S_col - 1); i++) {		//Data parallel 5
						if ((S_MAT[((S_row - 1) + i * S_row) + base] < 0)) {
							c = false; // check needed for race condition here.
							break;
						}
					}
				}
				__syncthreads();
			}
		} //end of while
		__syncthreads();
		if (threadIdx.x == 0) {
			//printf("Value = %f ",S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base]);
			Result[index] = S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base];
		}
	}
}


__host__ Simplex::Simplex(unsigned int N_S) {
	number_of_LPs = N_S;
	M = 0;
	N = 0;
	c = 0;
	No_c = 0;
	/*unsigned int memSize = N_S * sizeof(float);
	 R = (float*) malloc(memSize);*/
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
		//std::cout<<"No error here!!!\n";
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

__host__ void Simplex::setConstratint(math::matrix<double> A,
		std::vector<double> B) {
	int N_S = number_of_LPs;
	orig_CoefficientMatrix = A;
	BoundValue = B;
	int No_O = A.size2();
	int No_C = A.size1();
	M = No_C + 1;
	N = No_O + 3 + No_C;
	c = 1 + No_O;
		//MAT = (float *) calloc(N_S * M * N, sizeof(float));
		unsigned int memSize = N_S * M * N * sizeof(float);
		cudaError_t err;
		err = cudaMallocHost(&MAT, memSize);//Pinned memory Syntax:: cudaMallocHost(&h_ptr,bytes);
		//printf("CUDA cudaMallocHost-- MAT: %s\n", cudaGetErrorString(err));
		cudaMemset(MAT, 0, memSize);	//initializing all elements to zero
#pragma omp parallel for
	for (int s = 0; s < N_S; s++) {
		for (int i = 0; i < M - 1; i++) {
			for (int j = 0; j < N; j++) {
				if (j == 0) {
					MAT[(int) ((i + j * M) + (M * N * s))] = c + i;
				}
				else if (j > 1) {
					if (j < (No_O + 2)) {
						MAT[(int) ((i + j * M) + (M * N * s))] = (float) A(i,
								j - 2);
					} else if (j == N - 1) {
						MAT[(int) ((i + j * M) + (M * N * s))] = (float) B[i];
					} else if (j < N - 1) {
						MAT[(int) ((i + (No_O + 2 + i) * M) + (M * N * s))] = 1;
					}
				}
			}
		}
	}
	//std::cout<<"Constraints Setting Over!!!\n";
}	

__host__ void Simplex::ComputeLP(math::matrix<float> &C1,unsigned int number_of_streams ) {

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
	unsigned int memSize = N_S * sizeof(float);
	//R = (float*) malloc(memSize);
	err = cudaMallocHost((void**)&R, memSize);	//PINNED Memory	 //cudaMallocHost((void**)&a, bytes) );      // host pinned
	//printf("CUDA cudaMallocHost-- R: %s\n", cudaGetErrorString(err));
	int No_O = C.size2();
	M = No_C + 1;
	N = No_O + 3 + No_C;
	int N_C = No_C;
	c = 1 + No_O;
	//float sum = 0;
#pragma omp parallel for
	for (int s = 0; s < N_S; s++) {
		for (int i = M - 1; i < M; i++) {
			for (int j = 2; j < N; j++) {
				if (j < 2 + No_O) {
					MAT[(int) ((i + j * M) + (M * N * s))] = -C(s, j - 2);
				}	
			}
		}
	}
	std::vector<int> rem;
	for (int i = 0; i < N_C; i++) {
		//std::cout<<BoundValue[i]<<"\n";
		if (BoundValue[i] < 0) {
			rem.push_back(i);
			
		}
	}

	//std::cout<<"C= "<< rem.size()<<"\n";
	int nc = N + rem.size();
	threads_per_block = 32 * (nc / 32) + 32; //if count equal 0 than nc=N so works for for Model
	if (threads_per_block > props.maxThreadsPerBlock) //Assuming maximum threads supported by CC is 1024
		threads_per_block = props.maxThreadsPerBlock;
	int offset;
	int *R_index;	//reduction data
	float *R_data;	//reduction index
	err = cudaMalloc((void **) &R_data, C1.size1() * threads_per_block * sizeof(float));//C1.size1() * 96 being the maximum threads
	err = cudaMalloc((void **) &R_index,C1.size1() * threads_per_block * sizeof(int));//C1.size1() being the number of LPs
	err = cudaMalloc((void **) &G_R, N_S * sizeof(float));//Doing it here for the First Time
									// eg 	 cudaMalloc((void**)&d_a, bytes) ); // device
//	printf("CUDA malloc R_index: %s\n", cudaGetErrorString(err));
//	std::cout << "Number of threads per block = " << threads_per_block << "\n";
	if (rem.size() > 0) {
		;
	}
	else {
			err = cudaMalloc((void **) &G_MAT, (N_S * M * N * sizeof(float)));
			//printf("CUDA malloc G_MAT : %s\n", cudaGetErrorString(err));
			// **** Begin of Stream Processing *******
			//Using Asynchronous Memory copy:: needs //MAT to be a PINNED memory
			int num_streams = number_of_streams;//number of streams desired to create ::Note check for odd numbers
			int Each_LP_size = M * N;	// * sizeof(float);
			int num_LPs_perStream;
			bool equal_stream = true;
			if (N_S % num_streams == 0) {
				num_LPs_perStream = (N_S / num_streams);
				equal_stream = true;
			} else {
				num_LPs_perStream = (N_S / num_streams);//last stream will not be of the same size
				num_streams = num_streams + 1;//one extra streams.where nos of LPs to be solved will be less;
				equal_stream = false;
			}
			cudaStream_t stream[num_streams];
			cudaError_t result;

			//Creation of Streams
			for (int i = 0; i < num_streams; i++) {
				result = cudaStreamCreate(&stream[i]);
			}

			//err = cudaMemcpy(G_MAT, MAT, (N_S * M * N * sizeof(float)),cudaMemcpyHostToDevice);
			//Stream -- memcopy Host to Device
		//	std::cout << "\nNumber of LPs_perStream = " << num_LPs_perStream << std::endl;
			unsigned int lastBlock_size;
			if (equal_stream == false) {
				lastBlock_size = N_S - (N_S / (num_streams - 1)) * (num_streams - 1);//LAST Stream Size
			//	std::cout << "\nAmit Last Block size (LPs is )= " << lastBlock_size<< std::endl;
			}

			for (int i = 0; i < num_streams; i++) {
				if (equal_stream == false && i == (num_streams - 1)) {//last stream
					int offset = i * Each_LP_size * lastBlock_size;	//for memory copy
					cudaMemcpyAsync(&G_MAT[offset], &MAT[offset], (lastBlock_size * M * N * sizeof(float)), cudaMemcpyHostToDevice, stream[i]);
				} else {
					int offset = i * Each_LP_size * num_LPs_perStream;//for memory copy
					cudaMemcpyAsync(&G_MAT[offset], &MAT[offset], (num_LPs_perStream * M * N * sizeof(float)), cudaMemcpyHostToDevice, stream[i]);
				}
			}
			//mykernel<<<N_S, threads_per_block>>>(G_MAT, M, N, G_R, N_S, G_Sel, R_data, R_index);
			//	mykernel<<<N_S, threads_per_block>>>(G_MAT, M, N, G_R, N_S, R_data, R_index);
			//std::cout << "Before Kernel Call!!!" << std::endl;
			//Stream -- Kernel
			for (int i = 0; i < num_streams; i++) {
				if (equal_stream == false && i == (num_streams - 1)) {//last stream
					int offset_res = i * lastBlock_size;//for result here offset_res is a pointer to the LP number
					//mykernel<<<num_LPs_perStream, 256, 0, stream[i]>>>(G_MAT, M, N, G_R, G_Sel, num_LPs_perStream, offset_res);
					mykernel<<<lastBlock_size, threads_per_block, 0, stream[i]>>>(G_MAT, M, N, G_R, lastBlock_size, R_data, R_index, offset_res);
				} else {
					int offset_res = i * num_LPs_perStream;	//for result here offset_res is a pointer to the LP number
					//mykernel<<<num_LPs_perStream, 256, 0, stream[i]>>>(G_MAT, M, N, G_R, G_Sel, num_LPs_perStream, offset_res);
				//	std::cout<<"Kernel Called!!!\n";
					mykernel<<<num_LPs_perStream, threads_per_block, 0, stream[i]>>>(G_MAT, M, N, G_R, num_LPs_perStream, R_data, R_index, offset_res);
				//	std::cout<<"Kernel Finished!!!\n";
				//	std::cout<<"this kernel\n";
				}
			}
		//	std::cout << "After Kernel Call!!!" << std::endl;
		//	cudaDeviceSynchronize();//removed as hopping that cudaFree will handle it
			//err = cudaMemcpy(R, G_R, N_S * sizeof(float),cudaMemcpyDeviceToHost);

			//Stream -- memcopy Device to Host
			for (int i = 0; i < num_streams; i++) {
				//if (equal_stream == false && i == (num_streams - 1)) {//last stream
					//int offset_res = i * lastBlock_size;//for result here offset_res is a pointer to the LP number
					//cudaMemcpyAsync(&R[offset_res], &G_R[offset_res],(lastBlock_size * sizeof(float)),cudaMemcpyDeviceToHost, stream[i]);
				//} else {
					int offset_res = i * num_LPs_perStream;	//for result here offset_res is a pointer to the LP number
				//	std::cout<<"offset_res = "<<offset_res<<std::endl;
				//	std::cout<<"Memcopy started!!!\n";
					cudaMemcpyAsync(&R[offset_res], &G_R[offset_res],(num_LPs_perStream * sizeof(float)),
							cudaMemcpyDeviceToHost, stream[i]);
				//	std::cout<<"Memcopy Finished!!!\n";
				//	printf("CUDA memcopyAsync G_R : %s\n", cudaGetErrorString(err));
				//}
			}
			//std::cout<<"Memcopy End of Memory Copy!!!\n";
			// **** End of Stream Processing *******
			//printf("CUDA memcpy G_R: %s\n", cudaGetErrorString(err));
			//	std::cout << "Testing: R[0] = " << R[0] << std::endl;

				/*for (int i = 0; i < num_streams; ++i)
				    cudaStreamDestroy(stream[i]);*/
		}
	//std::cout<<"Before cudaFree command 1 !!!\n";
		//cudaFree(R_index);	//Only to synchronise with the cudamemcpy
		//cudaFree(R_data);	//Only to synchronise with the cudamemcpy
		cudaFree(G_MAT); //OK required
	//std::cout<<"Before cudaFree command 2 !!!\n";
		//cudaFreeHost(MAT); //should not be used
		cudaFree(G_R); //OK required
		//cudaFreeHost(R);	//This is needed to avoid Segmentation fault error
	//	std::cout << "N_S after = " << R[1] << std::endl;
	//	std::cout<<"After cudaFree command!!!\n";
}

