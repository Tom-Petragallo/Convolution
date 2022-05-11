#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <chrono>

using namespace std;

void blur(string in, string out) {
    vector<float> M = {1/9.,1/9.,1/9.,1/9.,1/9.,1/9.,1/9.,1/9.};
    int N = 3;
    int larg_moitie = (N-1)/2;

    cv::Mat m_in = cv::imread(in, cv::IMREAD_UNCHANGED );
    auto rgb = m_in.data;
    vector< unsigned char > res(m_in.rows*m_in.cols*3);

    for (int i=0; i<m_in.rows*m_in.cols; i++) {
        auto somme_r = 0;
        auto somme_g = 0;
        auto somme_b = 0;
        for (int x=-larg_moitie; x<larg_moitie; x++) {
            for (int y=-larg_moitie; y<larg_moitie; y++) {
                int i_c = i + x - y*m_in.rows;
                auto red = rgb[i_c*3];
                auto green = rgb[i_c*3+1];
                auto blue = rgb[i_c*3+2];
                somme_r += M[x + larg_moitie + (y + larg_moitie) * N] * red;
                somme_g += M[x + larg_moitie + (y + larg_moitie) * N] * green;
                somme_b += M[x + larg_moitie + (y + larg_moitie) * N] * blue;
            }
        }
        res[i*3] = somme_r;
        res[i*3+1] = somme_g;
        res[i*3+2] = somme_b;
    }

    cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, res.data() );
    cv::imwrite(out, m_out);
}

int main()
{
    blur("in.jpg","out.jpg");
    return 0;
}