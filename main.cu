#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std;

//Signature des fonctions
void blur(string in);
void edges(string in);
void id(string in);
void blurRVB(string in);
void edgesRVB(string in);
void idRVB(string in);
void gray_scale(string in);
template <typename T>
void afficher_matrice(vector<T> M, int N);
template <typename T>
void convolution(vector<T> M, int N, int val_per_pixel, string in, string out);

// Pas fonctionnelle
template <typename T>
void convolutions_shared(vector<vector<T>> VMs, vector<int> Ns, int val_per_pixel, string in, string out);


int main(int argc, char *argv[])
{

	if (argc != 3) {
		cout << "Utilisation : cmd fichier filtre" << endl;
		cout << "Filtres disponibles : gris->blur|edges|id ou rvb->gray|idRVB|blurRVB|edgesRVB" << endl;
		return 0;
	}
	string nom = argv[2];
	string in = argv[1];
	if (nom == "blur")
		blur(in);
	else if (nom == "edges")
		edges(in);
	else if (nom == "id")
		id(in);
	else if (nom == "gray")
		gray_scale(in);
	else if (nom == "blurRVB")
		blurRVB(in);
	else if (nom == "edgesRVB")
		edgesRVB(in);
	else if (nom == "idRVB")
		idRVB(in);
	else {
		cout << "Filtre non reconnu" << endl;
		return 0;
	}

	cout << "Traitement terminé" << endl;
	return 0;
}





void id(string in) {
	vector<int> M = {0,0,0,0,1,0,0,0,0};
	afficher_matrice<int>(M,3);
	convolution<int>(M,3,1,in,"out_id.jpg");
}

void blur(string in) {
	vector<float> M = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
	afficher_matrice<float>(M,3);
	convolution<float>(M,3,1,in,"out_blur.jpg");
}

void edges(string in) {
	vector<int> M = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	afficher_matrice<int>(M,3);
	convolution<int>(M,3,1,in,"out_edges.jpg");
}



void idRVB(string in) {
	vector<int> M = {0,0,0,0,1,0,0,0,0};
	afficher_matrice<int>(M,3);
	convolution<int>(M,3,3,in,"out_idRVB.jpg");
}

void blurRVB(string in) {
	vector<float> M = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
	afficher_matrice<float>(M,3);
	convolution<float>(M,3,3,in,"out_blurRVB.jpg");
}

void edgesRVB(string in) {
	vector<int> M = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	afficher_matrice<int>(M,3);
	convolution<int>(M,3,3,in,"out_edgesRVB.jpg");
}

template <typename T>
void afficher_matrice(vector<T> M, int N) {
	for (int j=0; j<N; j++) {
		for (int i=0; i<N; i++) {
			cout << M[i+j*N] << ",";
		}
		cout << endl;
	}
}





// Fonctions reprise des TP
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
	cout << "Temps d'éxecution : " << elapsedTime << "ms" << endl;

	status = cudaMemcpy( g.data(), g_d, rows * cols, cudaMemcpyDeviceToHost );
	if (status != cudaSuccess) cout << "Erreur memcpy DtH g_d" << endl;

	cv::imwrite( "out.jpg", m_out );
	cudaFree( rgb_d);
	cudaFree( g_d);
}






template <typename T>
__global__ void convolution_gpu( unsigned char *rgb, unsigned char * res, T * M, int N, int val_per_pixel, size_t cols, size_t rows ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int min = -(N-1)/2;
	int max = (N-1)/2;
	if( i < cols && j < rows ) {
		T somme_r = 0;
		T somme_g = 0;
		T somme_b = 0;
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
					somme_r += M[x - min + (y - min)*N] * rgb[i_g*val_per_pixel];
				if (val_per_pixel > 1)
					somme_g += M[x - min + (y - min)*N] * rgb[i_g*val_per_pixel+1];
				if (val_per_pixel > 2)
					somme_b += M[x - min + (y - min)*N] * rgb[i_g*val_per_pixel+2];
			}
		}
		if (somme_r < 0)
			somme_r = 0;
		if (somme_g < 0)
			somme_g = 0;
		if (somme_b < 0)
			somme_b = 0;

		res[(i + j * cols)*val_per_pixel] = (unsigned char) somme_r;
		if (val_per_pixel > 1)
			res[(i + j * cols)*val_per_pixel+1] = (unsigned char) somme_g;
		if (val_per_pixel > 2)
			res[(i + j * cols)*val_per_pixel+2] = (unsigned char) somme_b;
	}
}



template <typename T>
void convolution(vector<T> M, int N, int val_per_pixel, string in, string out) {
	cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
	unsigned char* rgb = m_in.data;
	int cols = m_in.cols;
	int rows = m_in.rows;

	cout << cols << "*" << rows << endl;
	vector< unsigned char > res (cols*rows*val_per_pixel);

	unsigned char *rgb_d;
	T * M_d;
	unsigned char * res_d;

	cudaError_t status;

	status = cudaMalloc( &rgb_d, rows * cols * val_per_pixel);
	if (status != cudaSuccess) cout << "Erreur mallocrgb_d" << endl;
	status = cudaMalloc( &res_d, rows * cols * val_per_pixel);
	if (status != cudaSuccess) cout << "Erreur malloc res_d" << endl;
	status = cudaMalloc( &M_d, M.size()*sizeof(T) );
	if (status != cudaSuccess) cout << "Erreur malloc M_d" << endl;

	status = cudaMemcpy(rgb_d, rgb, rows * cols * val_per_pixel, cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtDrgb_d" << endl;
	status = cudaMemcpy(M_d, M.data(), M.size()*sizeof(T), cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD M_d" << endl;

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	dim3 t( 32, 32 );
	dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
	convolution_gpu<T><<< b, t >>>(rgb_d, res_d, M_d, N, val_per_pixel, cols, rows );

	auto kernelStatus = cudaGetLastError();
	if ( kernelStatus != cudaSuccess )
		cout << "CUDA Error : "<< cudaGetErrorString(kernelStatus) << " " << endl;

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Temps d'éxecution : " << elapsedTime << "ms" << endl;

	status = cudaMemcpy( res.data(), res_d, rows * cols * val_per_pixel, cudaMemcpyDeviceToHost );
	if (status != cudaSuccess) cout << "Erreur memcpy DtH res_d" << endl;


	if (val_per_pixel == 1) {
		cv::Mat m_out(rows, cols, CV_8UC1, res.data());
		cv::imwrite(out, m_out);
	}
	else if (val_per_pixel == 2){
		cv::Mat m_out(rows, cols, CV_8UC2, res.data());
		cv::imwrite(out, m_out);
	}
	else if (val_per_pixel == 3){
		cv::Mat m_out(rows, cols, CV_8UC3, res.data());
		cv::imwrite(out, m_out);
	}
	else {
		cout << "Val per pixel non supportée" << endl;
		return;
	}
	cudaFree(rgb_d);
	cudaFree(res_d);
	cudaFree(M_d);
}





template <typename T>
__global__ void convolutions_gpu_shared(unsigned char *rgb, unsigned char * res, T * Ms, int * Ns, int nbM, int val_per_pixel, size_t cols, size_t rows ) {
	auto li = threadIdx.x;
	auto lj = threadIdx.y;

	auto w = blockDim.x;
	auto h = blockDim.y;

	int i = blockIdx.x * (blockDim.x-nbM) + threadIdx.x-nbM;
	int j = blockIdx.y * (blockDim.y-nbM) + threadIdx.y-nbM;

	if (i == 33 && j == 33) {
		printf("%d,%d\n",li,lj);
	}

	extern __shared__ unsigned char sh[];

	if(0 <= i && i < cols-1 && j <= 0 && j < rows-1) {
		sh[(lj*w+li)*val_per_pixel] = rgb[(j*cols+i)*val_per_pixel];
		if (val_per_pixel > 1)
			sh[(lj*w+li)*val_per_pixel+1] = rgb[(j*cols+i)*val_per_pixel+1];
		if (val_per_pixel > 2)
			sh[(lj*w+li)*val_per_pixel+2] = rgb[(j*cols+i)*val_per_pixel+2];
	}

	__syncthreads();

	if (i == 0 && j == 0) {
		for (int y=2; y<h-2; y++) {
			for (int x=2; x<w-2; x++)
				printf("%d,",(int)sh[x+y*w]);
		}
		printf("\n");
	}
	__syncthreads();
	for (int k=0; k<nbM; k++) {

		int N = Ns[k];
		int min = -(N-1)/2;
		int max = (N-1)/2;
		T * M = Ms+(k*N*N);
		if( 0 < li-k && li+k < w-1 && 0 < lj-k && li+k < h-1 ) {
			T somme_r = 0;
			T somme_g = 0;
			T somme_b = 0;
			for (int x = min; x < max+1; x++) {
				for (int y = min; y < max+1; y++) {
					int i_c = li + x;
					int j_c = lj + y;

					if (i <= 0 && i_c < 0)
						i_c = k;
					if (j <= 0 && j_c < k)
						j_c = k;
					if (cols-1 <= i && w-1-k < i_c)
						i_c = w-1-k;
					if (rows-1 <= i && h-1-k < j_c)
						i_c = h-1-k;

					int i_g = j_c * w + i_c;
					somme_r += M[x - min + (y - min)*N] * sh[i_g*val_per_pixel];
					if (val_per_pixel > 1)
						somme_g += M[x - min + (y - min)*N] * sh[i_g*val_per_pixel+1];
					if (val_per_pixel > 2)
						somme_b += M[x - min + (y - min)*N] * sh[i_g*val_per_pixel+2];

				}
			}

			if (somme_r < 0)
				somme_r = 0;
			if (somme_g < 0)
				somme_g = 0;
			if (somme_b < 0)
				somme_b = 0;

			sh[(li + lj * w)*val_per_pixel] = (unsigned char) somme_r;
			if (val_per_pixel > 1)
				sh[(li + lj * w)*val_per_pixel+1] = (unsigned char) somme_g;
			if (val_per_pixel > 2)
				sh[(li + lj * w)*val_per_pixel+2] = (unsigned char) somme_b;
		}

		__syncthreads();
	}


	if (nbM <= li && li < w-2 && nbM <= lj && lj < h-2
			&& 0 <= i && i < cols-1 && 0 <= j && j < rows-1) {

		res[(i + j * cols)*val_per_pixel] = sh[li+lj*w];
		if (val_per_pixel > 1)
			res[(i + j * cols)*val_per_pixel+1] = sh[li+lj*w+1];
		if (val_per_pixel > 2)
			res[(i + j * cols)*val_per_pixel+2] = sh[li+lj*w+2];
	}
}


template <typename T>
void convolutions_shared(vector<vector<T>> VMs, vector<int> Ns, int val_per_pixel, string in, string out) {
	cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
	unsigned char* rgb = m_in.data;
	int cols = m_in.cols;
	int rows = m_in.rows;
	cout << cols << "*" << rows << endl;

	vector<T> Ms = {};
	for (int k=0; k<VMs.size(); k++) {
		for (int l=0; l<Ns[k]*Ns[k]; l++) {
			T val = VMs[k][l];
			Ms.push_back(val);
		}
	}

	vector< unsigned char > res (cols*rows*val_per_pixel);

	for (int i=0; i<cols*rows*val_per_pixel; i++) {
		res[i] = rgb[i];
	}

	unsigned char *rgb_d;
	unsigned char *res_d;
	T * Ms_d;
	int * Ns_d;

	cudaError_t status;

	status = cudaMalloc( &rgb_d, rows * cols * val_per_pixel);
	if (status != cudaSuccess) cout << "Erreur mallocrgb_d" << endl;
	status = cudaMalloc( &res_d, rows * cols * val_per_pixel);
	if (status != cudaSuccess) cout << "Erreur malloc res_d" << endl;
	status = cudaMalloc( &Ms_d, Ms.size()*sizeof(T) );
	if (status != cudaSuccess) cout << "Erreur malloc Ms_d" << endl;
	status = cudaMalloc( &Ns_d, Ns.size()*sizeof(int) );
	if (status != cudaSuccess) cout << "Erreur malloc Ns_d" << endl;

	status = cudaMemcpy(rgb_d, rgb, rows * cols * val_per_pixel, cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD rgb_d" << endl;
	status = cudaMemcpy(Ms_d, Ms.data(), Ms.size()*sizeof(T), cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD Ms_d" << endl;
	status = cudaMemcpy(Ns_d, Ns.data(), Ns.size()*sizeof(int), cudaMemcpyHostToDevice );
	if (status != cudaSuccess) cout << "Erreur memcpy HtD Ns_d" << endl;

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	int nbM = VMs.size();

	dim3 block( 32, 32 );
	//dim3 grid( ( cols - 1) / block.x + 1 , ( rows - 1 ) / block.y + 1 );
	dim3 grid( ( cols - 1) / (block.x-2*(nbM-1)) + 1 , ( rows - 1 ) / (block.y-2*(nbM-1)) + 1 );
	//cout << cols*rows*val_per_pixel << endl;
	convolutions_gpu_shared<T><<< grid, block , (block.x+nbM*2)*(block.y+nbM*2)*val_per_pixel >>>(rgb_d, res_d, Ms_d, Ns_d, VMs.size(), val_per_pixel, cols, rows );

	auto kernelStatus = cudaGetLastError();
	if ( kernelStatus != cudaSuccess )
		cout << "CUDA Error : "<< cudaGetErrorString(kernelStatus) << " " << endl;

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Temps d'éxecution : " << elapsedTime << "ms" << endl;

	status = cudaMemcpy( res.data(), res_d, rows * cols * val_per_pixel, cudaMemcpyDeviceToHost );
	if (status != cudaSuccess) cout << "Erreur memcpy DtH res_d" << endl;

	if (val_per_pixel == 1) {
		cv::Mat m_out(rows, cols, CV_8UC1, res.data());
		cv::imwrite(out, m_out);
	}
	else if (val_per_pixel == 2){
		cv::Mat m_out(rows, cols, CV_8UC2, res.data());
		cv::imwrite(out, m_out);
	}
	else if (val_per_pixel == 3){
		cv::Mat m_out(rows, cols, CV_8UC3, res.data());
		cv::imwrite(out, m_out);
	}
	else {
		cout << "Val per pixel non supportée" << endl;
		return;
	}

	cudaFree(rgb_d);
	cudaFree(res_d);
	cudaFree(Ms_d);
	cudaFree(Ns_d);
}