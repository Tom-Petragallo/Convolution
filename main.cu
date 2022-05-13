#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std;

void blur(string in);
void edges(string in);
void id(string in);
void gray_scale(string in);
template <typename T>
void afficher_matrice(vector<T> M, int N);
template <typename T>
void convolution(vector<T> M, int N, string in, string out);

int main(int argc, char *argv[])
{

	if (argc != 3) {
		cout << "Utilisation : cmd fichier blur|edges|id|gray" << endl;
		return 0;
	}
	string nom = argv[2];

	if (nom == "blur")
		blur(argv[1]);
	else if (nom == "edges")
		edges(argv[1]);
	else if (nom == "id")
		id(argv[1]);
	else if (nom == "gray")
		gray_scale(argv[1]);
	else {
		cout << "non reconnu" << endl;
		return 0;
	}

	cout << "Traitement terminé" << endl;
	return 0;
}





void id(string in) {
	vector<int> M = {0,0,0,0,1,0,0,0,0};
	afficher_matrice<int>(M,3);
	convolution<int>(M,3,in,"out_id.jpg");
}

void blur(string in) {
	vector<float> M = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
	afficher_matrice<float>(M,3);
	convolution<float>(M,3,in,"out_blur.jpg");
}

void edges(string in) {
	vector<int> M = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	afficher_matrice<int>(M,3);
	convolution<int>(M,3,in,"out_edges.jpg");
}

template <typename T>
void afficher_matrice(vector<T> M, int N) {
	for (int i=0; i<N; i++) {
		for (int j = 0; j < N; j++) {
			cout << M[i+j*N] << ",";
		}
		cout << endl;
	}
}





// Fonction reprise des TP
__global__ void grayscale( unsigned char * rgb, unsigned char * g, std::size_t cols, std::size_t rows ) {
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;
	if( i < cols && j < rows ) {
		g[ j * cols + i ] = (
									307 * rgb[ 3 * ( j * cols + i ) ]
									+ 604 * rgb[ 3 * ( j * cols + i ) + 1 ]
									+ 113 * rgb[  3 * ( j * cols + i ) + 2 ]
							) / 1024;
	}
}

void gray_scale(string in) {
	cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
	auto rgb = m_in.data;
	auto rows = m_in.rows;
	auto cols = m_in.cols;

	cout << "Taille image : " << rows << " x " << cols << " px" << endl;

	std::vector< unsigned char > g( rows * cols );
	cv::Mat m_out( rows, cols, CV_8UC1, g.data() );
	unsigned char * rgb_d;
	unsigned char * g_d;

	cudaError_t status;

	status = cudaMalloc( &rgb_d, 3 * rows * cols );
	if (status != cudaSuccess) cout << "Erreur malloc rgb_d" << endl;
	status = cudaMalloc( &g_d, rows * cols );
	if (status != cudaSuccess) cout << "Erreur malloc g_d" << endl;
	status = cudaMemcpy( rgb_d, rgb, 3 * rows * cols, cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD rgb_d" << endl;

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	dim3 t( 32, 32 );
	dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
	grayscale<<< b, t >>>( rgb_d, g_d, cols, rows );

	auto kernelStatus = cudaGetLastError();
	if ( kernelStatus != cudaSuccess )
		cout << "CUDA Error : "<< cudaGetErrorString(kernelStatus) << " " << endl;

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Temps d'éxecution : " << elapsedTime << "s" << endl;

	status = cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
	if (status != cudaSuccess) cout << "Erreur memcpy DtH g_d" << endl;

	cv::imwrite( "out.jpg", m_out );
	cudaFree( rgb_d);
	cudaFree( g_d);
}




template <typename T>
__global__ void convolution_gpu( unsigned char * g, T * M, unsigned char * res, int N, size_t cols, size_t rows ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int min = -(N-1)/2;
	int max = (N-1)/2;
	if( i < cols && j < rows ) {
		T somme = 0;
		for (int x = min; x < max+1; x++) {
			for (int y = min; y < max+1; y++) {
				int i_c = i + x;
				int j_c = j + y;

				if (i + x < 0)
					i_c = 0;
				if (cols <= i + x)
					i_c = cols - 1;
				if (j + y < 0)
					j_c = 0;
				if (rows <= j + y)
					j_c = rows - 1;

				int i_g = j_c * cols + i_c;
				T gray = g[i_g];
				somme += M[x + 1 + (y + 1)*3] * gray;
			}
		}
		if (somme < 0)
			somme = 0;
		res[i + j * cols] = (unsigned char) somme;
	}
}

template <typename T>
void convolution(vector<T> M, int N, string in, string out) {
	cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
	unsigned char* g = m_in.data;
	int cols = m_in.cols;
	int rows = m_in.rows;

	cout << cols << "*" << rows << endl;
	vector< unsigned char > res (cols*rows);

	unsigned char * g_d;
	T * M_d;
	unsigned char * res_d;

	cudaError_t status;

	status = cudaMalloc( &g_d, rows * cols );
	if (status != cudaSuccess) cout << "Erreur malloc g_d" << endl;
	status = cudaMalloc( &res_d, rows * cols );
	if (status != cudaSuccess) cout << "Erreur malloc res_d" << endl;
	status = cudaMalloc( &M_d, M.size()*sizeof(T) );
	if (status != cudaSuccess) cout << "Erreur malloc M_d" << endl;

	status = cudaMemcpy( g_d, g, rows * cols, cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD g_d" << endl;
	status = cudaMemcpy( M_d, M.data(), M.size()*sizeof(T), cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD M_d" << endl;

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	dim3 t( 32, 32 );
	dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
	convolution_gpu<T><<< b, t >>>( g_d, M_d, res_d, N, cols, rows );

	auto kernelStatus = cudaGetLastError();
	if ( kernelStatus != cudaSuccess )
		cout << "CUDA Error : "<< cudaGetErrorString(kernelStatus) << " " << endl;

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Temps d'éxecution : " << elapsedTime << "s" << endl;

	status = cudaMemcpy( res.data(), res_d, rows * cols, cudaMemcpyDeviceToHost );
	if (status != cudaSuccess) cout << "Erreur memcpy DtH res_d" << endl;

	cv::Mat m_out( rows, cols, CV_8UC1, res.data() );
	cv::imwrite(out, m_out);
}

