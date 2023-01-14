#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void MatrixInitAl(float* M, int k, int n, int p) {
	for (int i = 0; i < k*n*p; i++) {
		M[i] = (float)rand() / (float)RAND_MAX;
	}
}

void MatrixInitNull(float* M, int k, int n, int p) {
	for (int i = 0; i < k * n * p; i++) {
		M[i] = 0;
	}
}

void MatrixPrint(float* M, int n, int p) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			printf(" %1.3f ", M[p * i + j]); // Permet un affichage avec des distances régulières
		}
		printf("\n");
	}
}


// Convolution 2D

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int n_M, int size_kernel, int n_Mout, int nb_kernel) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp;

	if (row < n_Mout && col < n_Mout) {
		int tot_kernel = size_kernel * size_kernel;
		int tot_Mout = n_Mout * n_Mout;

		for (int k = 0; k < nb_kernel; k++) {
			tmp = 0.0;
			for (int j = 0; j < size_kernel; j++) {
				for (int l = 0; l < size_kernel; l++) {
					tmp += M[(row + j) * n_M + (col + l)] * kernel[l * size_kernel + l + k * tot_kernel];
				}
			}
			Mout[row * n_Mout + col + k * tot_Mout] = tmp;
		}
	}
}


// Subsampling 2D
__global__ void cudaSubsampling(float* M, float* Mout, int n_M, int z_M, int meanpool_size, int n_Mout) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row % meanpool_size == 0 && col % meanpool_size == 0) {
		float tmp;
		int meanpool = meanpool_size * meanpool_size;
		int tot_M = n_M * n_M;
		int tot_Mout = n_Mout * n_Mout;

		for (int k =0; k < z_M; k++) {
			tmp = 0.0;
			for (int j = 0; j < meanpool_size; j++) {
				for (int l = 0; l < meanpool_size; l++) {
					tmp += M[(row + j) * n_M + col + l + z_M * tot_M] / meanpool;
				}
			}
			
			if (row == 0) {
				Mout[row * n_Mout + (col / meanpool_size) + z_M * tot_Mout] = tmp;
			}

			else if (col == 0) {
				Mout[(row / meanpool_size) * n_Mout + col + z_M * tot_Mout] = tmp;
			}

			else {
				Mout[(row / meanpool_size) * n_Mout + (col / meanpool_size) + z_M * tot_Mout] = tmp;
			}
		}
	}
}

// Fonctions d'activation

__device__ float* activation_tanh(float* M, int n, int p, int prof) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < p) {
		int tot_M = n * p;
		for (int z = 0; z < prof; z++) {
			M[row * n + col + z * tot_M] = tanh(M[row * n + col + z * tot_M]);
		}
	}
	return M;
}

__global__ void cudaTanh(float* M, int n, int p, int prof) {
	activation_tanh(M, n, p, prof);
}

// Main

int main(int argc, char* argv[]) {

	//Initialisation CPU
	float *raw_data, *C1_data, *S1_data, *C1_kernel;

	raw_data = (float*)malloc(1*32*32*sizeof(float));
	C1_data = (float*)malloc(6 * 28 * 28 * sizeof(float));
	S1_data = (float*)malloc(6 * 14 * 14 * sizeof(float));
	C1_kernel = (float*)malloc(6 * 5 * 5 * sizeof(float));

	// Initialisation des matrices Layer 1 - Génération des données de test
	MatrixInitAl(raw_data, 1, 32, 32);
	MatrixInitNull(C1_data, 6, 28, 28);
	MatrixInitNull(S1_data, 6, 14, 14);
	MatrixInitAl(C1_kernel, 6, 5, 5);


	// Ajout de matrices à initialiser
	float* C2_data, * S2_data;

	C2_data = (float*)malloc(6 * 10 * 10 * sizeof(float));
	S2_data = (float*)malloc(6 * 5 * 5 * sizeof(float));

	MatrixInitNull(C2_data, 6, 10, 10);
	MatrixInitNull(S2_data, 6, 5, 5);


	//Initialisation GPU
	float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;
	float* d_C2_data, * d_S2_data;

	// Allocation memoire du GPU
	cudaMalloc(&d_raw_data, sizeof(float)*1*32*32);
	cudaMalloc(&d_C1_data, sizeof(float) * 6 * 28 * 28);
	cudaMalloc(&d_S1_data, sizeof(float) * 6 * 14 * 14);
	cudaMalloc(&d_C1_kernel, sizeof(float) * 6 * 5 * 5);

	cudaMalloc(&d_C2_data, sizeof(float) * 6 * 10 * 10);
	cudaMalloc(&d_S2_data, sizeof(float) * 6 * 5 * 5);

	// Copie des données du CPU au GPU
	cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 1 * 32 * 32, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 6 * 28 * 28, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 6 * 14 * 14, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 6 * 5 * 5, cudaMemcpyHostToDevice);

	cudaMemcpy(d_C2_data, C2_data, sizeof(float) * 6 * 10 * 10, cudaMemcpyHostToDevice);
	cudaMemcpy(d_S2_data, S2_data, sizeof(float) * 6 * 5 * 5, cudaMemcpyHostToDevice);

	dim3 block_dim(32, 32);
	dim3 grid_dim(1, 1);

	cudaConv2D << <block_dim, grid_dim >> > (d_raw_data, d_C1_kernel, d_C1_data, 32, 5, 28, 6);

	cudaTanh << <block_dim, grid_dim >> > (d_C1_data, 28, 28, 6);

	cudaSubsampling << <block_dim, grid_dim >> > (d_C1_data, d_S1_data, 28, 6, 2, 14);

	cudaConv2D << <block_dim, grid_dim >> > (d_S1_data, d_C1_kernel, d_C2_data, 14, 15, 10, 16);

	cudaTanh << <block_dim, grid_dim >> > (d_C2_data, 10, 10, 16);

	cudaSubsampling << <block_dim, grid_dim >> > (d_C2_data, d_S2_data, 10, 16, 2, 5);

	// Copie des matrices résultantes du GPU au CPU
	cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 6 * 28 * 28, cudaMemcpyDeviceToHost);
	cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 6 * 14 * 14, cudaMemcpyDeviceToHost);
	cudaMemcpy(C2_data, d_C2_data, sizeof(float) * 6 * 10 * 10, cudaMemcpyDeviceToHost);
	cudaMemcpy(S2_data, d_S2_data, sizeof(float) * 6 * 5 * 5, cudaMemcpyDeviceToHost);

	// Affichage

	printf("Matrice initiale :\n\n");
	MatrixPrint(raw_data, 32, 32);

	printf("Noyau de convolution :\n\n");
	MatrixPrint(C1_kernel, 5, 5);

	printf("Première convolution 2D suivie d'une activation tanh :\n\n");
	MatrixPrint(C1_data,28,28);

	printf("Premier subsampling : \n\n");
	MatrixPrint(S1_data, 14, 14);

	printf("Deuxième convolution 2D avec activation tanh : \n\n");
	MatrixPrint(C2_data, 10, 10);

	printf("Deuxième subsampling : \n\n");
	MatrixPrint(S2_data, 5, 5);

	// Libération de la mémoire
	free(raw_data);
	free(C1_kernel);
	free(C1_data);
	free(S1_data);
	free(C2_data);
	free(S2_data);

	cudaFree(d_raw_data);
	cudaFree(d_C1_kernel);
	cudaFree(d_C1_data);
	cudaFree(d_S1_data);
	cudaFree(d_C2_data);
	cudaFree(d_S2_data);

	return 0;
}