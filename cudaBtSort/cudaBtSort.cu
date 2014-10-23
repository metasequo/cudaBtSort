#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#include <cuda.h>
#include <cutil.h>
#include <cutil_inline.h>
#include <cutil_inline_runtime.h>

// 要素数、ブロックサイズ、ループ回数
#define SIZE 4096
#define BLOCK_SIZE 256
#define LOOP 10000

__global__
static void BtSort(int* inData);

int main(){

	// 変数宣言
	int* targetData;
	int i, j;

	// メモリ確保
	targetData = (int*)malloc(sizeof(int) * SIZE);

	// デバイス側の変数宣言
	int* dTargetData;

	// デバイスメモリ確保
	cutilSafeCall(cudaMalloc((void**)&dTargetData, sizeof(int) * SIZE));
	cutilSafeCall(cudaMemcpy(dTargetData, targetData, sizeof(int) * SIZE, cudaMemcpyHostToDevice));

	// ブロックサイズ、グリッドサイズ設定
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
//	if (SIZE / BLOCK_SIZE < 1)	dim3 grid(1);

	// タイマー変数の宣言、測定開始
	printf("Bitonic sort start in the GPU!\n");
	printf("Element count\t:\t%d\n", SIZE);
	printf("BlockSize.X\t:\t%d\nBlockSize.Y\t:\t%d\nBlockSize.Z\t:\t%d\n", block.x, block.y, block.y);
	printf("GridSize.X\t:\t%d\nGridSize.Y\t:\t%d\nGridSize.X\t:\t%d\n", grid.x, grid.y, grid.z);
	printf("Loop count\t:\t%d\n", LOOP);
	float millseconds = 0.0f, sum = 0.0f, ave = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// メインループ
	for (int k = 0; k < 10; k++){
		sum = 0.0f;
		for (i = 0; i < LOOP; i++){
			// 要素を初期化
			for (j = 0; j < SIZE; j++)
				targetData[j] = (int) ((rand() / ((double) RAND_MAX + 0.1f))* INT_MAX);
				
			// 記録開始、カーネル関数実行
			cudaEventRecord(start, 0);
			BtSort <<<grid, block>>>(dTargetData);
			cudaThreadSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&millseconds, start, stop);
			sum += millseconds;
		}
		printf("Time required\t:\t%f minutes\n", sum / 1000);
		ave += sum;
	}

	// 測定終了
//	printf("Element count\t:\t%d\n", SIZE);
//	printf("BlockSize\t:\t%d\nGridSize\t:\t%d\n", BLOCK_SIZE, SIZE / BLOCK_SIZE);
//	printf("Loop count\t:\t%d\n", LOOP);
	printf("Time average\t:\t%f minutes\n", ave /10000);

	// 結果の領域の確保と、デバイス側からのメモリ転送
	cutilSafeCall(cudaMemcpy(targetData, dTargetData, sizeof(int) * SIZE, cudaMemcpyDeviceToHost));

	// メモリ解放
	free(targetData);
	cutilSafeCall(cudaFree(dTargetData));

	cudaThreadExit();
}

// バイトニックソートするカーネル関数
__global__
static void BtSort(int* inData){
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// 外回りはマージ、内回りは分解交換のループ
	for(unsigned int length = 2; length <= SIZE; length *= 2){
		for(unsigned int mlength = length / 2; mlength > 0; mlength /= 2){
			unsigned int ixj = idx ^ mlength;

			if(ixj > idx){
				int tmp;

				// 昇順か降順か判断して入れ替え
				if((idx & ixj) == 0){
					if(inData[idx] > inData[ixj]){
						tmp = inData[ixj];
						inData[ixj] = inData[idx];
						inData[idx] = tmp;
					}
				}else{
					if(inData[idx] < inData[ixj]){
						tmp = inData[ixj];
						inData[ixj] = inData[idx];
						inData[idx] = tmp;
					}
				}
			}
			__threadfence();	//アクセス可能まで待機
			__syncthreads();	//スレッド同期
		}
	}
}