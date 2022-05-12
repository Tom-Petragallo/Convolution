#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std;

void convolution(vector<vector<float>> M, int N, string in, string out);
void blur(string in, int taille);
void edges(string in);
void afficher_matrice(vector<vector<float>> M);

int main()
{
	blur("in.jpg",5);
	edges("in.jpg");
	return 0;
}

void blur(string in, int taille) {
	if (taille % 2 != 1) {
		cout << "La taille doit être impaire" << endl;
		return;
	}
	int N = taille;
	vector<vector<float>> M(N);// = {{1/9.,1/9.,1/9.},{1/9.,1/9.,1/9.},{1/9.,1/9.,1/9.}};
	for (int i=0; i<N; i++) {
		vector<float> ligne(N);
		for (int j = 0; j < N; j++) {
			ligne[j]=1./(N*N);
		}
		M[i]=ligne;
	}
	afficher_matrice(M);
	convolution(M,N,in,"out_blur.jpg");
}

void edges(string in) {
	int N = 3;
	vector<vector<float>> M = {{-1.,-1.,-1.},{-1.,8.,-1.},{-1.,-1.,-1.}};
	afficher_matrice(M);
	convolution(M,N,in,"out_edges.jpg");
}


void convolution(vector<vector<float>> M, int N, string in, string out) {
    int larg_moitie = (N-1)/2;

    cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
    auto rgb = m_in.data;
    int larg = m_in.cols;
	int haut = m_in.rows;
    vector< unsigned char > res (larg*haut*3);

    for (int j=0; j<haut; j++) {
		for (int i=0; i<larg; i++) {
			unsigned char somme_r = 0;
			unsigned char somme_g = 0;
			unsigned char somme_b = 0;
			for (int x = -larg_moitie; x < larg_moitie + 1; x++) {
				for (int y = -larg_moitie; y < larg_moitie + 1; y++) {
					int i_c = i+x;
					int j_c = j+y;

					if (i + x < 0)
						i_c = 0;
					if (larg <= i + x)
						i_c = larg-1;
					if (j + y < 0)
						j_c = 0;
					if (haut <= j + y)
						j_c = haut-1;

					int i_rgb = j_c * larg + i_c;
					auto red = rgb[i_rgb * 3];
					auto green = rgb[i_rgb * 3 + 1];
					auto blue = rgb[i_rgb * 3 + 2];
					somme_r += M[x+larg_moitie][y+larg_moitie] * red;
					somme_g += M[x+larg_moitie][y+larg_moitie] * green;
					somme_b += M[x+larg_moitie][y+larg_moitie] * blue;

				}
			}
			res[(i + j * larg) * 3] = somme_r;
			res[(i + j * larg) * 3 + 1] = somme_g;
			res[(i + j * larg) * 3 + 2] = somme_b;
		}
	}
    cout << "traitement terminé" << endl;
    cv::Mat m_out( haut, larg, CV_8UC3, res.data() );
    cv::imwrite(out, m_out);
}

void afficher_matrice(vector<vector<float>> M) {
	for (int i=0; i<M.size(); i++) {
		for (int j = 0; j < M[i].size(); j++) {
			cout << M[i][j] << ",";
		}
		cout << endl;
	}
}