#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std;

void blur(string in);
void edges(string in);
void id(string in);
void gray_scale(string in);
void embossing(string in);
template <typename T>
void afficher_matrice(vector<T> M, int N);
template <typename T>
void convolutions(vector<vector<T>> Ms, vector<int> N, int val_per_pixel, string in, string out);

int main(int argc, char *argv[])
{

	if (argc != 3) {
		cout << "Utilisation : cmd fichier filtre" << endl;
		cout << "Filtres disponibles : gris->blur|edges|id ou rvb->gray|embossing" << endl;
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
	else if (nom == "embossing")
		embossing(in);
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
	convolutions<int>({M},{3},1,in,"out_id.jpg");
}

void blur(string in) {
	vector<float> M = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
	afficher_matrice<float>(M,3);
	convolutions<float>({M},{3},1,in,"out_blur.jpg");
}

void edges(string in) {
	vector<int> M = {-1,-1,-1,-1,8,-1,-1,-1,-1};
	afficher_matrice<int>(M,3);
	convolutions<int>({M},{3},1,in,"out_edges.jpg");
}

void embossing(string in) {
	vector<int> M0 = {0,1,0,0,0,0,0,-1,0};
	vector<int> M1 = {1,0,0,0,0,0,0,0,-1};
	vector<int> M2 = {0,0,0,1,0,-1,0,0,0};
	vector<int> M3 = {0,0,1,0,0,0,-1,0,0};
	vector<vector<int>> Ms = {M0/*,M1,M2,M3*/};
	vector<int> Ns = {3,3,3,3};
	convolutions<int>(Ms,Ns,3,in,"out_embossing.jpg");
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
__global__ void convolution_gpu( unsigned char *rgb, T * M, unsigned char * res, int N, int val_per_pixel, size_t cols, size_t rows ) {
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
				T red =rgb[i_g*val_per_pixel];
				T green =rgb[i_g*val_per_pixel+1];
				T blue =rgb[i_g*val_per_pixel+2];
				somme_r += M[x + 1 + (y + 1)*val_per_pixel] * red;
				somme_g += M[x + 1 + (y + 1)*val_per_pixel] * green;
				somme_b += M[x + 1 + (y + 1)*val_per_pixel] * blue;
			}
		}
		if (somme_r < 0)
			somme_r = 0;
		if (somme_g < 0)
			somme_g = 0;
		if (somme_b < 0)
			somme_b = 0;
		res[(i + j * cols)*val_per_pixel] = (unsigned char) somme_r;
		res[(i + j * cols)*val_per_pixel+1] = (unsigned char) somme_g;
		res[(i + j * cols)*val_per_pixel+2] = (unsigned char) somme_b;
	}
}






template <typename T>
void convolutions(vector<vector<T>> Ms, vector<int> Ns, int val_per_pixel, string in, string out) {
	cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
	unsigned char* rgb = m_in.data;
	int cols = m_in.cols;
	int rows = m_in.rows;

	cout << cols << "*" << rows << endl;
	vector< unsigned char > res (cols*rows*val_per_pixel);

	for (int i=0; i<cols*rows*val_per_pixel; i++) {
		res[i] = rgb[i];
	}

	unsigned char *rgb_d;
	T * M_d;
	unsigned char * res_d;

	for (int i=0; i<Ms.size(); i++) {
		int N = Ns[i];
		vector<T> M = Ms[i];

		cudaError_t status;

		status = cudaMalloc( &rgb_d, rows * cols * val_per_pixel);
		if (status != cudaSuccess) cout << "Erreur mallocrgb_d" << endl;
		status = cudaMalloc( &res_d, rows * cols * val_per_pixel);
		if (status != cudaSuccess) cout << "Erreur malloc res_d" << endl;
		status = cudaMalloc( &M_d, M.size()*sizeof(T) );
		if (status != cudaSuccess) cout << "Erreur malloc M_d" << endl;

		status = cudaMemcpy(rgb_d, res.data(), rows * cols * val_per_pixel, cudaMemcpyHostToDevice );
		if (status != cudaSuccess) cout << "Erreur memcpy HtDrgb_d" << endl;
		status = cudaMemcpy( M_d, M.data(), M.size()*sizeof(T), cudaMemcpyHostToDevice );
		if (status != cudaSuccess) cout << "Erreur memcpy HtD M_d" << endl;

		float elapsedTime;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		dim3 t( 32, 32 );
		dim3 b( ( cols - 1) / t.x + 1 , ( rows - 1 ) / t.y + 1 );
		convolution_gpu<T><<< b, t >>>(rgb_d, M_d, res_d, N, val_per_pixel, cols, rows );

		auto kernelStatus = cudaGetLastError();
		if ( kernelStatus != cudaSuccess )
			cout << "CUDA Error : "<< cudaGetErrorString(kernelStatus) << " " << endl;

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cout << "Temps d'éxecution : " << elapsedTime << "ms" << endl;

		status = cudaMemcpy( res.data(), res_d, rows * cols * val_per_pixel, cudaMemcpyDeviceToHost );
		if (status != cudaSuccess) cout << "Erreur memcpy DtH res_d" << endl;
	}

	if (val_per_pixel == 3) {
		cv::Mat m_out(rows, cols, CV_8UC3, res.data());
		cv::imwrite(out, m_out);
	}
	else if (val_per_pixel == 1){
		cv::Mat m_out(rows, cols, CV_8UC1, res.data());
		cv::imwrite(out, m_out);
	}
	else if (val_per_pixel == 4){
		cv::Mat m_out(rows, cols, CV_8UC4, res.data());
		cv::imwrite(out, m_out);
	}
	else {
		cout << "Val per pixel non supporté" << endl;
		return;
	}

	cudaFree(rgb_d);
	cudaFree(res_d);
	cudaFree(M_d);
}