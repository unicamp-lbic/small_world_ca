#include <stdlib.h>
#include <time.h>

#define CA_SIZE            %s
#define CA_ITERATIONS      %s
#define CA_REPEAT          %s
#define CONNECTION_RADIUS  %s
#define N_CONNECTIONS      %s
#define N_OBSERVATIONS     %s
#define K_HISTORY          %s
#define N_POSSIBLE_HISTORY %s

__global__ void kernel_calc_diffs(unsigned int majority[CA_REPEAT],
	unsigned int executions[CA_REPEAT][CA_ITERATIONS + 1][CA_SIZE],
	unsigned int sum_diffs[CA_SIZE]) {
	
	int cell = threadIdx.x;
	int errors_local[N_CONNECTIONS];
	int sum_diffs_local;
	int i, repeat, shift;
	
	__shared__ int errors[N_CONNECTIONS];
	__shared__ int flow;
	
	for (i = 0; i < N_CONNECTIONS; i++)
		errors_local[i] = 0;
	
	if (cell == 0)
		for (i = 0; i < N_CONNECTIONS; i++)
			errors[i] = 0;
	
	__syncthreads();
	
	for (repeat = 0; repeat < CA_REPEAT; repeat++)
		for (i = 0; i < CA_ITERATIONS; i++)
			for (shift = -CONNECTION_RADIUS; shift <= CONNECTION_RADIUS;
				shift++)
				
				errors_local[shift + CONNECTION_RADIUS] +=
					abs((float) executions[repeat][i][cell] -
					executions[repeat][i + 1][(cell + shift + CA_SIZE) %%
					CA_SIZE]);
	
	for (i = 0; i < N_CONNECTIONS; i++)
		atomicAdd(&errors[i], errors_local[i]);
	
	__syncthreads();
	
	if (cell == 0) {
		int min_error = errors[0];
		
		flow = -CONNECTION_RADIUS;
		
		for (i = 1; i < N_CONNECTIONS; i++)
			if (errors[i] < min_error) {
				min_error = errors[i];
				flow = i - CONNECTION_RADIUS;
			}
	}
	
	__syncthreads();
	
	for (repeat = 0; repeat < CA_REPEAT; repeat++) {
		sum_diffs_local = 0;
		
		for (i = 0; i < CA_ITERATIONS; i++) {
			int diff = abs((float) executions[repeat][i + 1][cell] -
				executions[repeat][i][(cell - flow + CA_SIZE) %% CA_SIZE]);
		
			sum_diffs_local += diff;
		}
	
		sum_diffs[cell] += sum_diffs_local;
	}
	
}

__device__ int to_binary(unsigned int execution[CA_ITERATIONS + 1][CA_SIZE],
	int cell, int i) {
	
	int k;
	int value = 0;
	
	for (k = 0; k < K_HISTORY; k++)
		value += execution[i - k][cell] << k;
	
	return value;
	
}

__global__ void kernel_probabilities(
	unsigned int execution[CA_REPEAT][CA_ITERATIONS + 1][CA_SIZE],
	float p_joint_table[CA_SIZE][N_POSSIBLE_HISTORY][2],
	float p_prev_table[CA_SIZE][N_POSSIBLE_HISTORY],
	float p_curr_table[CA_SIZE][2]) {
	
	int cell = threadIdx.x;
	int repeat = blockIdx.x;
	int i;
	
	for (i = K_HISTORY; i < CA_ITERATIONS; i++) {
		int past = to_binary(execution[repeat], cell, i - 1);
		int present = execution[repeat][i][cell];
		
		atomicAdd(&p_joint_table[cell][past][present], 1);
		atomicAdd(&p_prev_table[cell][past], 1);
		atomicAdd(&p_curr_table[cell][present], 1);
	}

}

__global__ void kernel_active_storage(
	unsigned int execution[CA_REPEAT][K_HISTORY + 1][CA_SIZE],
	float p_joint_table[CA_SIZE][N_POSSIBLE_HISTORY][2],
	float p_prev_table[CA_SIZE][N_POSSIBLE_HISTORY],
	float p_curr_table[CA_SIZE][2], float active_storage[CA_SIZE]) {
	
	int cell = threadIdx.x;
	int repeat = blockIdx.x;
	int past, present;
	float p1, p2, p3;
	
	past = to_binary(execution[repeat], cell, K_HISTORY - 1);
	present = execution[repeat][K_HISTORY][cell];

	p1 = p_joint_table[cell][past][present] / (float) N_OBSERVATIONS;
	p2 = p_prev_table[cell][past] / (float) N_OBSERVATIONS;
	p3 = p_curr_table[cell][present] / (float) N_OBSERVATIONS;
	
	atomicAdd(&active_storage[cell], log2f(p1 / (p2 * p3)) / N_OBSERVATIONS);

}

__global__ void kernel_entropy_rate(
	float p_joint_table[CA_SIZE][N_POSSIBLE_HISTORY][2],
	float p_prev_table[CA_SIZE][N_POSSIBLE_HISTORY],
	float entropy_rate[CA_SIZE]) {
	
	int cell = threadIdx.x;
	int i, j;
	float aux = 0;
	
	for (i = 0; i < N_POSSIBLE_HISTORY; i++)
		for (j = 0; j < 2; j++)
			if (p_joint_table[cell][i][j] > 0)
				aux -= p_joint_table[cell][i][j] *
					log2f((float) p_joint_table[cell][i][j] /
					p_prev_table[cell][i]) / N_OBSERVATIONS;
	
	entropy_rate[cell] = aux;
	
}

