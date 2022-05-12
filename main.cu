#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std;

void blur(string in, string out) {
	vector<vector<float>> M = {{1/9.,1/9.,1/9.},{1/9.,1/9.,1/9.},{1/9.,1/9.,1/9.}};
	//vector<vector<float>> M = {{1}};
	for (int i=0; i<M.size(); i++)
		for (int j=0; j<M[i].size(); j++)
			cout << M[i][j] << ",";
	cout << endl;
    int N = 3;
    int larg_moitie = (N-1)/2;

    cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
    auto rgb = m_in.data;
    int larg = m_in.cols;
	int haut = m_in.rows;
    vector< unsigned char > res (larg*haut*3);

	cout << "a" << endl;
    for (int j=0; j<haut; j++) {
		for (int i=0; i<larg; i++) {
			unsigned char somme_r = 0;
			unsigned char somme_g = 0;
			unsigned char somme_b = 0;
			for (int x = -larg_moitie; x < larg_moitie + 1; x++) {
				for (int y = -larg_moitie; y < larg_moitie + 1; y++) {
					if (0 <= i + x && i + x < larg
							&& 0 <= j + y && j + y < haut ) {

						int i_c = (y + j) * larg + (x + i);
						auto red = rgb[i_c * 3];
						auto green = rgb[i_c * 3 + 1];
						auto blue = rgb[i_c * 3 + 2];
						somme_r += M[x][y] * red;
						somme_g += M[x][y] * green;
						somme_b += M[x][y] * blue;

					}
				}
			}
			res[(i + j * larg) * 3] = somme_r;
			res[(i + j * larg) * 3 + 1] = somme_g;
			res[(i + j * larg) * 3 + 2] = somme_b;

			if (i == 1 && j == 1) {
				cout << (i + j * larg) << endl;
				cout << (int) res[(i + j * larg) * 3] << " ";
				cout << (int) res[(i + j * larg) * 3 + 1] << " ";
				cout << (int) res[(i + j * larg) * 3 + 2] << " ";
				cout << endl;

				cout << (int) rgb[(i + j * larg) * 3] << " ";
				cout << (int) rgb[(i + j * larg) * 3 + 1] << " ";
				cout << (int) rgb[(i + j * larg) * 3 + 2] << " ";
				cout << endl;
			}
		}
	}
    cout << "b" << endl;
    cv::Mat m_out( larg, haut, CV_8UC3, res.data() );
    cv::imwrite(out, m_out);
}

int main()
{
    blur("in.jpg","out.jpg");
    return 0;
}