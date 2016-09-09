#include "simplex.cuh"
#include<omp.h>
#include "iostream"

//LPC without Stream
//Tested Working for Both Helicopter and Five Dimensional system with Reduction
//Both Reduction implemented- One for finding Pivot column and the other for finding Pivot Row
//Implemented Reduction. But without Streams.

// 1st Method:: Most negative value (//Not Working for Large arguments)
__global__ void mykernelOld(float *S_MAT, int S_row, int S_col, float *Result,
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
		__shared__ float col1[1024];	//1024 is required to function well for threads
		/*************/
		if (threadIdx.x == 0) {
			c = false;
			rm = 0;
			row = -1;		//pivotRow
			pivotCol = -1;			//printf("Before starting While Loop\n");
		}
		__syncthreads();
		while (!c) {
			__syncthreads();
			int Last_row = S_row - 1;
			//__syncthreads();
			//   ***************** Get_Pivot function begins  *****************
			// ******** First Reduction Begins **********
			//using reduction to compute min and newpivotcol
			//__shared__ bool notEntered;
			__shared__ int notEntered;
//not in use	__shared__ float minValue;
			__shared__ int newpivotcol;
			if (threadIdx.x == 0) {
//not in use	minValue = 0;
				newpivotcol = -1;
				//notEntered = true;
				notEntered = 1;
				c = true;
				//printf("Before starting reduction 1\n");
			}
			__syncthreads();	//making sure newpivotcol is initialised to -1
			int data_size = blockDim.x;
			tid = threadIdx.x;
			if (threadIdx.x >= 2 && threadIdx.x < (S_col - 1)) {
				//tid = threadIdx.x - 2;//here my tid should be from 0 to (evenThreadId - 1)
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
					if (R_data[tid + R_base] > R_data[tid + R_base + i]) { //is right-side value small?
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
				__syncthreads();
				i >>= 1;
				if ((i != 1) && (i % 2) != 0) {	//if s is odd
					i = i + 1;
				}
			}
			// if (notEntered == false && tid == 2) { // tid==0 is always true if minValue is still -1 then what?
			if (threadIdx.x == 0) { // tid==0 is always true if minValue is still -1 then what?
				if (notEntered == false) {
//not in use		minValue = R_data[R_base];
					newpivotcol = R_index[R_base];
					//printf("\nminValue = %f newpivotcol = %d ", minValue,newpivotcol);
				}
			}
			__syncthreads(); //waiting for all threads to have same newpivotcol value
			/*if (threadIdx.x == 0) {
				printf("May be Finished reduction 1\n");
			}
			__syncthreads();*/
			if (newpivotcol == -1) {//All Threads will follow the Same path so no issue with divergence
				//return -2;
				row = -2;
			} else {
				// ********** Second Reduction Process ******
				//in order to avoid global memory transfer:: Using the same R_data and R_index global memory
				__shared__ float row_min;
				__shared__ int row_num;
				//__shared__ bool notEntered2;
				__shared__ int notEntered2;
				if (threadIdx.x == 0) {
					row_min = INT_MAX;
					row_num = -1;
					//notEntered2 = true;
					notEntered2 = 1;
				}
				__syncthreads();
				// Since keeping limit only upto Last_row which is not equal to block_size creates problem
				// in using syncthreads() inside Reduction for-loop so use all threads(all R_data
				int k1;
				//if (threadIdx.x >= 0 && threadIdx.x < Last_row) {
				if (threadIdx.x < Last_row) { //because threadID is from 0
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
	//			if (threadIdx.x >= 0 && threadIdx.x < S_row) {
				if (threadIdx.x < S_row) { //because threadID is from 0
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
				if (threadIdx.x < S_row) {	//because threadID is from 0
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
			//printf("Result = %f  ",Result[index]);
		}
	}
}

//1st Method : Most Negative Value approach (//Works even for Large arguments)
__global__ void mykernelSSS(float *S_MAT, int S_row, int S_col, float *Result,
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
__global__ void mykernel(float *S_MAT, int S_row, int S_col, float *Result,
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
		/*if(threadIdx.x==0){
			for(int ik=0;ik<S_row;ik++){
					for(int jk=0;jk<S_col;jk++){
						printf("%f  ",S_MAT[ik+jk*S_row+index]);
					}
					printf("\n");
				}
			printf("\n\n\n\n");
			printf("Row= %d col = %d\n",row,pivotCol);
			printf("\n\n\n\n");

		}
		__syncthreads(); */
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
			//		}
			//		__syncthreads();	//here we have min and newpivotcol
			// ********* First Reduction Ends *************
			//  ******** Second Reduction Begins **********
			/*			if (threadIdx.x == 0) {
			 if (newpivotcol == -1) {
			 //return -2;
			 row = -2;
			 } else {
			 float row_min = INT_MAX;
			 float row_num = -1;
			 //TODO:: this temp_res can be an array of value computed in parallel
			 //TODO:: row_min and row_num can then be computed using reduction
			 for (i = 0; i < S_row - 1; i++) {

			 temp_index = newpivotcol * S_row + i + base; //avoiding re-computation
			 temp_index1 = i + (S_col - 1) * S_row + base; //avoiding re-computation
			 if ((S_MAT[temp_index] > 0)
			 && (S_MAT[temp_index1] > 0)) {
			 float temp_res = S_MAT[temp_index1]
			 / S_MAT[temp_index]; //avoiding re-computation
			 if (temp_res <= row_min) {
			 row_min = temp_res;
			 row_num = i;
			 }
			 }
			 }
			 // *S_Sel = newpivotcol;
			 pivotCol = newpivotcol;
			 //S_Sel[index] = newpivotcol;
			 if (row_min == INT_MAX) {
			 //return -1;
			 row = -1;
			 }
			 if (row_num != -1) {
			 //return row_num;
			 row = row_num;
			 }
			 }
			 } //end of one thread
			 __syncthreads();*/
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
	/*
				if(threadIdx.x==0){
			for(int ik=0;ik<S_row;ik++){
					for(int jk=0;jk<S_col;jk++){
						printf("%f  ",S_MAT[ik+jk*S_row+index]);
					}
					printf("\n");
				}
			printf("\n\n\n\n");
			printf("Row= %d col = %d\n",row,pivotCol);
			printf("\n\n\n\n");

		}
		__syncthreads();
			//printf("Value = %f ",S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base]);

*/			Result[index] = S_MAT[(S_row - 1 + (S_col - 1) * S_row) + base];
			//printf("\nResult Inside Kernel: %f \n",Result[index]);
		}
	}
}


__host__ Simplex::Simplex(unsigned int N_S) {
	number_of_LPs = N_S;
	//i = 0;
	//a = 0.0;
	M = 0;
	N = 0;
	//NB = 0;
	c = 0;
	No_c = 0;
	//f = j = 0;
	//R = (float*) malloc(N_S * sizeof(float));
	R = (float*) calloc(N_S, sizeof(float));

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

__host__ void Simplex::setConstratint(math::matrix<double> A,
		std::vector<double> B) {
	int N_S = number_of_LPs;
	orig_CoefficientMatrix = A;
	BoundValue = B;
//	std::cout<<"Before setConstraints called\n";
//	A = math::matrix<float>(A1);
//	B = std::vector<float>(B1);
	int No_O = A.size2();
	//std::cout << "No of Variable is " << A.size2() << " And no of constraints "	<< A.size1() << std::endl;
	int No_C = A.size1();
	M = No_C + 1;
	N = No_O + 3 + No_C;
	c = 1 + No_O;
	//NB = c;
	//f = 0;

	/*Sel = (int *) malloc(N_S * sizeof(int));
	 R = (float*) malloc(N_S * sizeof(float));*/
	//MAT = (float *) calloc(N_S * M * N, sizeof(float));

	MAT_COPY = (float *) calloc(N_S * M * N, sizeof(float));

	/*cudaMalloc((void **) &G_MAT, (N_S * M * N * sizeof(float)));
	 cudaMalloc((void **) &G_R, N_S * sizeof(float));
	 cudaMalloc((void **) &G_Sel, N_S * sizeof(int));*/
int s;
#pragma omp parallel for
	for (s = 0; s < N_S; s++) {
		unsigned int some = M * N * s;
		for (int i = 0; i < M - 1; i++) {
			for (int j = 0; j < N; j++) {
				if (j == 0) {
					//MAT[(int) ((i + j * M) + (M * N * s))] = c + i;
					//MAT[(int) ((i + j * M) + some)] = c + i;
					MAT_COPY[(int) ((i + j * M) + some)] = c + i;
				} else if (j > 1) {
					if (j < (No_O + 2)) {	//Coefficient of A
						//MAT[(int) ((i + j * M) + (M * N * s))] = (float) A(i, j - 2);
						//MAT[(int) ((i + j * M) + some)] = (float) A(i, j - 2);
						MAT_COPY[(int) ((i + j * M) + some)] = (float) A(i, j - 2);
					} else if (j == N - 1) {//std::cout<<"Enter RHS of coefficient "<< i+1 <<"\n";
						//MAT[(int) ((i + j * M) + (M * N * s))] = (float) B[i];
						//MAT[(int) ((i + j * M) + some)] = (float) B[i];
						MAT_COPY[(int) ((i + j * M) + some)] = (float) B[i];
					} else if (j < N - 1) {
						//MAT[(int) ((i + (No_O + 2 + i) * M) + (M * N * s))] = 1;
						//MAT[(int) ((i + (No_O + 2 + i) * M) + some)] = 1;
						MAT_COPY[(int) ((i + (No_O + 2 + i) * M) + some)] = 1;
					}
				}
			}
		}
	}
//	std::cout<<"setting constraints of simplex Done\n";
}	//setting constraints of simplex

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
	M = No_C + 1, N = No_O + 3 + No_C;
	int N_C = No_C;
	c = 1 + No_O;
	//float sum = 0;
	int s;
#pragma omp parallel for
	for (s = 0; s < N_S; s++) {
		unsigned int some = M * N * s;
		for (int i = M - 1; i < M; i++) {
			for (int j = 2; j < N; j++) {
				if (j < 2 + No_O) {
					//MAT[(int) ((i + j * M) + (M * N * s))] = -C(s, j - 2);
					//MAT[(int) ((i + j * M) + some)] = -C(s, j - 2);
					MAT_COPY[(int) ((i + j * M) + some)] = -C(s, j - 2);
				}
			}
		}
	}
	std::vector<int> rem;
	for (int i = 0; i < N_C; i++) {
		//std::cout << B[i] << "\n";
		if (BoundValue[i] < 0) {
			rem.push_back(i);
			//count++;
			//std::cout<<B[i]<<"\n";
		}
	}

	//std::cout<<"C= "<< rem.size()<<"\n";
	int nc = N + rem.size();
	threads_per_block = 32 * (nc / 32) + 32; //if count equal 0 than nc=N so works for for Model
	if (threads_per_block > props.maxThreadsPerBlock) //Assuming maximum threads supported by CC is 1024
		threads_per_block = props.maxThreadsPerBlock;

	int *R_index;	//reduction data
	float *R_data;	//reduction index
	err = cudaMalloc((void **) &R_data,
			C1.size1() * threads_per_block * sizeof(float));//C1.size1() * 96 being the maximum threads
	err = cudaMalloc((void **) &R_index,
			C1.size1() * threads_per_block * sizeof(int));//C1.size1() being the number of LPs

	err = cudaMalloc((void **) &G_R, N_S * sizeof(float));//Doing it here for the First Time

	//printf("CUDA malloc R_index: %s\n", cudaGetErrorString(err));
	//std::cout << "Number of threads per block = " << threads_per_block << "\n";

	if (rem.size() > 0) {	//Helicopter model has no negative bound so count=0
	;
	} else {
		//cudaEvent_t start, stop;

		err = cudaMalloc((void **) &G_MAT, (N_S * M * N * sizeof(float)));
		//printf("CUDA malloc G_MAT : %s\n", cudaGetErrorString(err));
		//err = cudaMemcpy(G_MAT, MAT, (N_S * M * N * sizeof(float)), cudaMemcpyHostToDevice);
		err = cudaMemcpy(G_MAT, MAT_COPY, (N_S * M * N * sizeof(float)), cudaMemcpyHostToDevice);

	//	printf("CUDA memcpy G_MAT : %s\n", cudaGetErrorString(err));
//
	//	std::cout << "Size copied(bytes) = "<<(long)(N_S * M * N * sizeof(float));
		//err = cudaMemcpy(G_Sel, Sel, sizeof(int), cudaMemcpyHostToDevice);
		//printf("CUDA memcpy G_Sel: %s\n", cudaGetErrorString(err));
		/*std::cout << "Simplex -AFTER CREATION OF Z\n";

		 for (i = 0; i < M; i++) {
		 for (j = 0; j < N; j++) {
		 std::cout << MAT[(i + j * M)] << "(" << (i + j * M) << ")\t";
		 }
		 std::cout << "\n";
		 }*/
	//	std::cout << "Before Kernel Call\n";
		//mykernel<<<N_S, threads_per_block>>>(G_MAT, M, N, G_R, N_S, G_Sel, R_data, R_index);
		mykernel<<<N_S, threads_per_block>>>(G_MAT, M, N, G_R, N_S, R_data, R_index);
	//	std::cout << "After Kernel Called\n";
		//	cudaDeviceSynchronize();		//removed as hopping that cudaFree will handle it
		err = cudaMemcpy(R, G_R, N_S * sizeof(float), cudaMemcpyDeviceToHost);
	//	printf("CUDA memcpy G_R: %s\n", cudaGetErrorString(err));
		//cudaMemcpy(MAT, G_MAT, (N_S * M * N * sizeof(float)), cudaMemcpyDeviceToHost);
		//	std::cout << "N_S = " << N_S << std::endl;
	}
//	std::cout << "before cudaFree \n";
//	cudaFree(G_MAT);
//	cudaFree(G_Sel);
	cudaFree(R_index);	//Only to synchronise with the cudamemcpy

//	std::cout << "After cudaFree \n";
}

//  Computes the entire list of LPs by diving into different blocks :: this interface not use at present
std::vector<float> Simplex::bulkSolver(math::matrix<float> &List_of_ObjValue) {
	unsigned int tot_lp = List_of_ObjValue.size1();
	std::cout << "Total LPs " << tot_lp << std::endl;
	int lp_block_size = 1000;//input how many LPs you want to solve at a time ??????
	unsigned int number_of_blocks;
	if (tot_lp % lp_block_size == 0)
		number_of_blocks = tot_lp / lp_block_size;
	else
		number_of_blocks = (tot_lp / lp_block_size) + 1;
	std::cout << "Total Blocks " << number_of_blocks << std::endl;

	std::list<block_lp> bulk_lps(number_of_blocks);	//list of sub-division of LPs
	struct block_lp myLPList;
	myLPList.block_obj_coeff.resize(lp_block_size, List_of_ObjValue.size2());
	math::matrix<float> block_obj_coeff(lp_block_size,
			List_of_ObjValue.size2());
	unsigned int index = 0;
	for (unsigned int lp_number = 0; lp_number < tot_lp; lp_number++) {
		for (unsigned int i = 0; i < List_of_ObjValue.size2(); i++) {
			myLPList.block_obj_coeff(index, i) = List_of_ObjValue(lp_number, i);
		}
		index++;
		if (index == lp_block_size) {
			index = 0;
			bulk_lps.push_back(myLPList);
		}
	}	//end of all LPs
	std::list<block_lp_result> bulk_result(number_of_blocks);
	struct block_lp_result eachBlock;
	eachBlock.results.resize(lp_block_size);	//last block will be less

	for (std::list<block_lp>::iterator it = bulk_lps.begin();
			it != bulk_lps.end(); it++) {
		ComputeLP((*it).block_obj_coeff);
		eachBlock.results = this->getResultAll();
		bulk_result.push_back(eachBlock);
	}
	std::vector<float> res(tot_lp);
	unsigned int index_res = 0;
	for (std::list<block_lp_result>::iterator it = bulk_result.begin();
			it != bulk_result.end(); it++) {
		unsigned int block_result_size = (*it).results.size();
		for (unsigned int i = 0; i < block_result_size; i++) {
			res[index_res] = (*it).results[i];
			index_res++;
		}
	}
	std::cout << "Result size = " << res.size() << std::endl;
//R = res;
	return res;
}

