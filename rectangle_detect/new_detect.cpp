#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// 定义常量 PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

// Zernike 矩亚像素边缘检测类
class ZernikeSubpixelEdge {
private:
    int N; // 模板大小，通常为 7
    std::vector<std::complex<double>> mask_M11;
    std::vector<std::complex<double>> mask_M20;
    std::vector<Point> mask_coords; // 记录掩膜内的相对坐标

public:
    ZernikeSubpixelEdge(int kernel_size = 7) : N(kernel_size) {
        precomputeMasks();
    }

    // 预计算 Zernike 模板 (N x N)
    // 映射到单位圆: x^2 + y^2 <= 1
    void precomputeMasks() {
        int half = N / 2;
        double radius = half;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // 将像素坐标映射到 [-1, 1] 坐标系
                double y = (double)(i - half) / radius; // 注意图像行是 y
                double x = (double)(j - half) / radius; // 图像列是 x

                double r = std::sqrt(x * x + y * y);
                double theta = std::atan2(y, x);

                // 只计算单位圆内的点
                if (r <= 1.0 + 1e-6) {
                    mask_coords.push_back(Point(j - half, i - half));

                    // Zernike 多项式 V_nm(rho, theta) = R_nm(rho) * exp(j * m * theta)
                    // complex<double> j_complex(0, 1);

                    // M11: n=1, m=1
                    // R_11(r) = r
                    // V_11 = r * e^(i*theta) = x + iy
                    std::complex<double> v11(x, y);
                    // 注意：计算矩时使用的是 V的共轭，但在实数图像卷积中，我们通常直接计算投影
                    // 此处为了匹配 Ghosal 算法，构建复数卷积核
                    mask_M11.push_back(std::conj(v11));

                    // M20: n=2, m=0
                    // R_20(r) = 2*r^2 - 1
                    // V_20 = 2*r^2 - 1 (实数)
                    double v20_val = 2 * r * r - 1;
                    std::complex<double> v20(v20_val, 0);
                    mask_M20.push_back(std::conj(v20));
                }
            }
        }
    }

    // 对单个 Canny 边缘点进行亚像素校正
    // 返回值：是否存在有效的亚像素点
    bool refineEdge(const Mat& img, Point p_canny, Point2f& sub_pixel_out) {
        int half = N / 2;

        // 边界检查
        if (p_canny.x < half || p_canny.x >= img.cols - half ||
            p_canny.y < half || p_canny.y >= img.rows - half) {
            return false;
        }

        std::complex<double> M11(0, 0);
        std::complex<double> M20(0, 0);

        // 卷积计算矩
        for (size_t k = 0; k < mask_coords.size(); k++) {
            int px = p_canny.x + mask_coords[k].x;
            int py = p_canny.y + mask_coords[k].y;

            double gray_val = static_cast<double>(img.at<uchar>(py, px));

            M11 += gray_val * mask_M11[k];
            M20 += gray_val * mask_M20[k];
        }

        // Ghosal & Mehrotra 算法核心
        // 计算边缘的角度 theta
        double theta = std::atan2(M11.imag(), M11.real());

        // 旋转后的 M20 和 M11 (M11_prime 是纯实数，代表边缘强度)
        double M11_mag = std::abs(M11);
        double M20_prime = M20.real() * std::cos(2 * theta) + M20.imag() * std::sin(2 * theta);

        // 避免除零或弱边缘
        if (M11_mag < 10.0) return false;

        // 计算距离 l (单位圆坐标系下)
        // l = M20' / M11'
        double l = M20_prime / M11_mag;

        // 只有当 l 在合理范围内（例如 [-0.5, 0.5] 对应的单位圆比例）才认为边缘就在这个像素附近
        // 这里的阈值取决于单位圆映射的比例，保守起见取一个较小值
        // N=7, half=3.5. 1 pixel approx 1/3.5 = 0.28 in unit unit unit.
        // 如果 l 过大，说明真正的边缘离这个 Canny 像素太远，不可信
        if (std::abs(l) > 0.4) return false;

        // 将单位圆坐标系下的距离 l 映射回像素坐标系
        // 距离 correction = l * (N/2)
        double l_pixel = l * (double)(N / 2);

        // 计算亚像素坐标
        // 偏移方向是 theta
        sub_pixel_out.x = p_canny.x + l_pixel * std::cos(theta);
        sub_pixel_out.y = p_canny.y + l_pixel * std::sin(theta);

        return true;
    }
};

int main() {
    // 1. 读取图像
    // 请替换为你的图片路径
    std::string image_path = "./data/40.png";
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Error: Could not load image." << endl;
        // 生成一个测试圆
        src = Mat::zeros(400, 400, CV_8UC1);
        circle(src, Point(200, 200), 100, Scalar(200), -1);
        GaussianBlur(src, src, Size(9, 9), 2); // 模拟模糊
        cout << "Generated test image." << endl;
    }

    // 2. 预处理
    Mat blurred;
    // 使用高斯滤波平滑噪声，这对于 Canny 和 Zernike 都很重要
    GaussianBlur(src, blurred, Size(5, 5), 1.5);

    // 3. Canny 边缘检测 (粗定位)
    Mat edges;
    // 阈值根据具体图像调整，这里使用自动或宽泛的范围
    Canny(blurred, edges, 40, 50, 3, true);

    // 4. Zernike 亚像素细化
    ZernikeSubpixelEdge zernikeDetector(7); // 7x7 窗口
    std::vector<Point2f> subPixelPoints;
    std::vector<Point> coarsePoints;

    // 获取所有非零点（Canny 边缘点）
    findNonZero(edges, coarsePoints);

    cout << "Coarse edge points: " << coarsePoints.size() << endl;

    for (const auto& p : coarsePoints) {
        Point2f sp;
        if (zernikeDetector.refineEdge(src, p, sp)) {
            subPixelPoints.push_back(sp);
        }
    }

    cout << "Refined sub-pixel points: " << subPixelPoints.size() << endl;

    if (subPixelPoints.size() < 10) {
        cerr << "Not enough points to fit ellipse!" << endl;
        return -1;
    }

    // 5. 最小二乘法拟合椭圆
    // fitEllipse 函数内部使用最小二乘法拟合
    RotatedRect fitBox = fitEllipse(subPixelPoints);

    // 6. 结果可视化与输出
    Mat result;
    cvtColor(src, result, COLOR_GRAY2BGR);

    // 绘制拟合的椭圆
    ellipse(result, fitBox, Scalar(0, 0, 255), 1);
    // 绘制圆心
    int cx = cvRound(fitBox.center.x);
    int cy = cvRound(fitBox.center.y);
    result.at<cv::Vec3b>(cy, cx) = cv::Vec3b(0, 0, 255);


    // 输出高精度圆心
    cout << "-----------------------------------" << endl;
    cout << "Sub-pixel Center (x, y): " << fitBox.center.x << ", " << fitBox.center.y << endl;
    cout << "Axis (width, height): " << fitBox.size.width << ", " << fitBox.size.height << endl;
    cout << "Angle: " << fitBox.angle << endl;
    cout << "-----------------------------------" << endl;

    return 0;
}