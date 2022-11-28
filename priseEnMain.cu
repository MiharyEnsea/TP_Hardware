#include <stdio.h>
#include <stdlib.h>

void MatrixInit(float *M, int n, int p);
void MatrixPrint(float *M, int n, int p);
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
void MatrixMult(float *M1, float *M2, float *Mout, int n);

void MatrixInit(float *M, int n, int p) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<p; j++) {
            M[p*i+j] = (float)(rand()/RAND_MAX)*2.0 - 1.0; // (rand()/RAND_MAX) retourne une valeur entre 0 et 1. 
        }
    }
}


void MatrixPrint(float *M, int n, int p) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<p; j++) {
            printf("%1.3f",M[p*i+j]); // Permet un affichage avec des distances régulières
        }
        printf("\n");
    }
}


void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<p; j++) {
            Mout[p*i+j] = M1[p*i+j] + M2[p*i+j] ;
        }
    }
}


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < n && j < p) {
        Mout[p*i+j] = M1[p*i+j] + M2[p*i+j];
    }
    
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    float c = 0.0;
    for (int i = 0; i<n; i++){
        for (int j = 0; j<n ; j++){
            for(int k = 0; k<n; k++){
                c += M1[n*i+k]*M2[n*i+k];
            }
            Mout[n*i+j] = c;
            c = 0.0;
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    float c = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n){
        for(int k = 0; k<n; k++) c += M1[row * n + k] * M2[k * n + col];
    }
    Mout[row * n + col] = c;
}

// Main 

int main() {

    int n = 10; // Nombre de lignes 
    int p = 10; // Nombre de colonnes

    // Creation et affichage d'une matrice 
    float *M;
    cudaMalloc(&M, sizeof(float)*(n*p));

    MatrixInit(M, n, p); // Creation
    MatrixPrint(M, n, p); // Affichage


    return 0;
}