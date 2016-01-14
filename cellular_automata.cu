#include <stdlib.h>
#include <time.h>

#define CA_SIZE            %s
#define CA_ITERATIONS      %s
#define CA_REPEAT          %s
#define N_CONNECTIONS      %s
#define MAX_EPOCHS         %s

#define CONSTANT_RHO       %s
#define UNIFORM_RHO        %s
#define DECREASING_RHO     %s


__global__ void kernel_simulation(unsigned int *rule,
	unsigned int connections[CA_SIZE][N_CONNECTIONS],
	float random_arr[CA_REPEAT][CA_SIZE],
	int *correct, int epoch, int ic_type, int save_ca,
	unsigned int executions[CA_REPEAT][CA_ITERATIONS + 1][CA_SIZE]) {

	int cell = threadIdx.x;
	int repeat = blockIdx.x;
	float random = random_arr[repeat][cell];
	int i;
	
	__shared__ int initial_majority, final_majority, pre_final_majority;
	__shared__ unsigned char ca[CA_SIZE];
	
	if (cell == 0) {
		initial_majority = 0;
		final_majority = 0;
		pre_final_majority = 0;
	}
	
	switch (ic_type) {
		case CONSTANT_RHO:
			ca[cell] = (random < 0.5 ? 1 : 0);
			break;
		case UNIFORM_RHO:
			ca[cell] = (random < (repeat + 1.0) / (gridDim.x + 1.0) ? 1 : 0);
			break;
		case DECREASING_RHO:
			float progress = float(epoch) / MAX_EPOCHS;
			
			if (progress < 0.5) 
				ca[cell] = ((random < progress + ((repeat + 1.0)
					/ gridDim.x) * (1.0 - 2.0 * progress)) ? 1 : 0);
			else
				ca[cell] = (random < 0.5 ? 1 : 0);
			break;
	}
	
	atomicAdd(&initial_majority, ca[cell]);
	__syncthreads();
	
	for (i = 0; i < CA_ITERATIONS; i++) {
		unsigned int rule_number = 0;
		int count = N_CONNECTIONS - 1;
		
		if (i == CA_ITERATIONS - 1)
			atomicAdd(&pre_final_majority, ca[cell]);
		
		if (save_ca)
			executions[repeat][i][cell] = ca[cell];
		
		for (int j = 0; j < N_CONNECTIONS; j++, count--)
			rule_number |= (unsigned int) (ca[connections[cell][j]] << count);
		
		__syncthreads();
		ca[cell] = rule[rule_number];
		__syncthreads();
	}
	
	if (save_ca)
		executions[repeat][i][cell] = ca[cell];
	
	atomicAdd(&final_majority, ca[cell]);
	__syncthreads();
	
	if (cell == 0)
		if ((((initial_majority > CA_SIZE / 2.) && (final_majority == CA_SIZE))
			|| ((initial_majority <= CA_SIZE / 2.) && (final_majority == 0)))
			&& (pre_final_majority == final_majority))
			
			correct[repeat] = 1;
	
}

