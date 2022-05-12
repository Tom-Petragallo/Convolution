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
void afficher_matrice(vector<vector<T>> M);

template <typename T>
void convolution(vector<vector<T>> M, string in, string out);

int main(int argc, char *argv[])
{

	if (argc != 3) {
		cout << "Utilisation : cmd fichier blur|edges|id|gray" << endl;
		return 0;
	}
	string nom = argv[2];

	auto start = chrono::steady_clock::now();
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

	auto end = std::chrono::steady_clock::now();
	chrono::duration<double> elapsed_seconds = end-start;
	cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return 0;
}

void id(string in) {
	vector<vector<int>> M = {{0,0,0},{0,1,0},{0,0,0}};
	afficher_matrice<int>(M);
	convolution<int>(M,in,"out_blur.jpg");
}

void blur(string in) {
	vector<vector<float>> M = {{0.0625,0.125,0.0625},{0.125,0.25,0.125},{0.0625,0.125,0.0625}};
	afficher_matrice<float>(M);
	convolution<float>(M,in,"out_blur.jpg");
}

void edges(string in) {
	vector<vector<int>> M = {{-1,-1,-1},{-1,8,-1},{-1,-1,-1}};
	afficher_matrice<int>(M);
	convolution<int>(M,in,"out_edges.jpg");
}


void gray_scale(string in) {
	cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
	auto rgb = m_in.data;
	vector< unsigned char > g( m_in.rows * m_in.cols );

	for (auto i=0; i<g.size(); i++) {
		auto red = rgb[i*3];
		auto green = rgb[i*3+1];
		auto blue = rgb[i*3+2];
		g[i] = (307*red+604*green+113*blue)/1024;
	}

	cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );
	cv::imwrite( "out.jpg", m_out );
}

template <typename T>
void convolution(vector<vector<T>> M, string in, string out) {
    cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
    unsigned char* g = m_in.data;
    int larg = m_in.cols;
	int haut = m_in.rows;

	cout << larg << "*" << haut << endl;
    vector< unsigned char > res (larg*haut);

    for (int j=0; j<haut; j++) {
		for (int i=0; i<larg; i++) {
			T somme = 0;
			for (int x = -1; x < 2; x++) {
				for (int y = -1; y < 2; y++) {
					int i_c = i + x;
					int j_c = j + y;

					if (i + x < 0)
						i_c = 0;
					if (larg <= i + x)
						i_c = larg - 1;
					if (j + y < 0)
						j_c = 0;
					if (haut <= j + y)
						j_c = haut - 1;

					int i_g = j_c * larg + i_c;
					T grey = g[i_g];
					somme += M[x + 1][y + 1] * grey;

				}
			}
			if (somme < 0)
				somme = 0;
			res[i + j * larg] = (unsigned char) somme;
		}
	}
    cout << "traitement terminé" << endl;
    cv::Mat m_out( haut, larg, CV_8UC1, res.data() );
    cv::imwrite(out, m_out);
}

template <typename T>
void afficher_matrice(vector<vector<T>> M) {
	for (int i=0; i<M.size(); i++) {
		for (int j = 0; j < M[i].size(); j++) {
			cout << M[i][j] << ",";
		}
		cout << endl;
	}
}