#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void MatrixInit(float* M, int n, int p) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			M[i * n + j] = (float)(float(rand()) / float(RAND_MAX)) * 2.0 - 1.0;
		}
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


void MatrixAdd(float* M1, float* M2, float* Mout, int n, int p) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < p; j++) {
			Mout[p * i + j] = M1[p * i + j] + M2[p * i + j];
		}
	}
}


__global__ void cudaMatrixAdd(float* M1, float* M2, float* Mout, int n) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = col + row * n;
	if (col < n && row < n) {
		Mout[index] = M1[index] + M2[index];
		
	}

}


void MatrixMult(float* M1, float* M2, float* Mout, int n) {

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float c = 0.0;
			for (int k = 0; k < n; k++) {
				c += M1[n * i + k] * M2[j + k * n];
			}
			Mout[n * i + j] = c;
			c = 0.0;
		}
	}
}

__global__ void cudaMatrixMult(float* M1, float* M2, float* Mout, int n) {
	// On implémente chaque thread de la ligne
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// On implémente chaque thread de la colonne
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int tmp = 0;
	
	if ((row < n) && (col < p)) {
		for (int k = 0; k < n; k++) {
			tmp += M1[row * n + k] * M2[k * n + col];
		}
		Mout[row * n + col] = tmp;
	}

}

int main(int argc, char*argv[]) {

	if (argc != 4) {
		return EXIT_FAILURE;
	}
	//Taille de la matrice n*p
	int n = atoi(argv[2]);
	int p = atoi(argv[3]);

	//Taille en bytes de la matrice
	size_t bytes = n * p * sizeof(float);

	// pointeurs du CPU
	float *h_a, *h_b, *h_mul, *h_add;

	// Allocation de mémoire au CPU
	h_a = (float*)malloc(bytes);
	h_b = (float*)malloc(bytes);
	h_mul = (float*)malloc(bytes);
	h_add = (float*)malloc(bytes);
	// Pointeur du GPU
	int *d_a, *d_b, *d_mul, *d_add;

	// Allocation de mémoire au GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_mul, bytes);
	cudaMalloc(&d_add, bytes);

	// Initialisation des matrices
	MatrixInit(h_a, n, p);
	MatrixInit(h_b, n, p);

	// Copie des données du CPU au GPU
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	

	// Threads par blocs
	int BLOCK_SIZE = 16;

	// Blocs dans chaque dimension
	int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Avec le GPU
	if (strcmp(argv[1], "gpu") == 0) {
		//Appel des fonctions/kernels GPU
		printf("Multiplication avec GPU : \n\n");
		cudaMatrixMult << <grid, threads >> > (d_a, d_b, d_mul, n);


		// Copie de la matrice résultante du GPU au CPU
		cudaMemcpy(h_mul, d_mul, bytes, cudaMemcpyDeviceToHost);

		// Affichage de la matrice
		MatrixPrint(h_mul, n, n);

		printf("Addition avec GPU : \n\n");
		cudaMatrixAdd << <grid, threads >> > (d_a, d_b, d_add, n, p);
		cudaMemcpy(h_add, d_add, bytes, cudaMemcpyDeviceToHost);
		MatrixPrint(h_add, n, p);

	}

	// Avec le CPU
	if (strcmp(argv[1], "cpu") == 0) {
		printf("Addition avec CPU : \n\n");
		MatrixAdd(h_a, h_b, h_add, n, p);
		MatrixPrint(h_add, n, p);

		printf("Multiplication avec CPU : \n\n");
		MatrixMult(h_a, h_b, h_mul, n);
		MatrixPrint(h_mul, n, n);
	}
	

	// On libère la mémoire du GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_mul);
	cudaFree(d_add);
	free(h_a);
	free(h_b);
	free(h_mul);
	free(h_add);

	return 0;
}