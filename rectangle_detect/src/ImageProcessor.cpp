#include "ImageProcessor.h"
#include <iostream>
#include <opencv2/opencv.hpp> 
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits> // 确保包含 limits

// 构造函数：初始化一些固定参数
ImageProcessor::ImageProcessor()
    : min_area_(100.0), max_area_(1000.0)
{
    // 预先初始化形态学核，避免在循环中重复创建
    element_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
}

void ImageProcessor::setThresholdParams(double minArea, double maxArea) {
    min_area_ = minArea;
    max_area_ = maxArea;
}

cv::Mat removeSmallRegions(const cv::Mat& binaryImage, int minArea)
{
    if (binaryImage.empty() || binaryImage.type() != CV_8UC1) {
        std::cerr << "错误: 输入必须是单通道8位二值图像 (CV_8UC1)。" << std::endl;
        return cv::Mat();
    }

    cv::Mat labels, stats, centroids;
    // 使用8连通
    int numComponents = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids, 8, CV_32S);

    // keepLabels[i] 表示第 i 个组件是否保留
    // 优化：使用 vector<uchar> 并在栈上操作通常很快
    std::vector<uchar> keepLabels(numComponents, 0);
    for (int i = 1; i < numComponents; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minArea) {
            keepLabels[i] = 255;
        }
    }

    cv::Mat result = cv::Mat::zeros(binaryImage.size(), CV_8UC1);

    int rows = result.rows;
    int cols = result.cols;

    if (result.isContinuous() && labels.isContinuous()) {
        cols *= rows;
        rows = 1;
    }

    // 这是一个内存密集型操作，当前指针遍历方式已经是 OpenCV 中较优的写法
    for (int i = 0; i < rows; ++i) {
        const int* labelPtr = labels.ptr<int>(i);
        uchar* resPtr = result.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            int lbl = labelPtr[j];
            if (lbl > 0) {
                resPtr[j] = keepLabels[lbl];
            }
        }
    }

    return result;
}

cv::Mat fillInternalContours(const cv::Mat& binaryImage)
{
    if (binaryImage.empty() || binaryImage.type() != CV_8UC1) {
        return cv::Mat();
    }

    cv::Mat imageToProcess = binaryImage.clone();

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // 优化建议：如果只关心孔洞填充，RETR_CCOMP 比 RETR_TREE 更轻量，
    // 但为了不改变算法逻辑（TREE 包含完整的树结构），此处保留 RETR_TREE
    cv::findContours(imageToProcess, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        if (hierarchy[i][3] != -1 && contours[i].size() < 100) {
            cv::drawContours(imageToProcess, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
        }
    }

    return imageToProcess;
}


struct PointInfo {
    int grayValue;
    cv::Point2f point;
};

/**
 * @brief 优化后的亮度检测函数
 *
 * 优化点：
 * 1. 接收外部传入的 maskBuffer，避免每次调用都重新分配内存。
 * 2. 只有在需要时才对 buffer 进行清零和绘制。
 */
bool isContourBrighterThanBackground(const cv::Mat& srcGray,
    const std::vector<cv::Point>& contour,
    cv::Mat& maskBuffer, // 新增：复用缓冲区
    int padding = 10,
    double diff_thresh = 10.0)
{
    cv::Rect bbox = cv::boundingRect(contour);

    // 扩大 ROI
    cv::Rect roiRect = (bbox + cv::Size(padding * 2, padding * 2)) - cv::Point(padding, padding);
    roiRect &= cv::Rect(0, 0, srcGray.cols, srcGray.rows);

    if (roiRect.empty()) return false;

    // 引用 ROI 数据 (O(1))
    cv::Mat roiImg = srcGray(roiRect);

    // --- 内存优化核心 ---
    // 重新调整 buffer 大小 (如果尺寸变大会重新分配，尺寸变小或不变则复用)
    maskBuffer.create(roiRect.size(), CV_8UC1);
    maskBuffer.setTo(0); // 快速清零

    // 绘制内部 Mask (offset 设置为 -roiRect.tl() 以匹配 ROI 坐标系)
    // 这里的 vector构造有轻微开销，但 drawContours 接口限制必须传 ArrayOfArrays
    // 使用 std::vector<std::vector<cv::Point>> 的临时包装是标准做法
    // 只有 contour[0] 被绘制
    const cv::Point* pts = contour.data();
    int npts = (int)contour.size();
    // 使用 fillPoly 可能比 drawContours 略快，因为它不需要处理 hierarchy
    cv::fillPoly(maskBuffer, &pts, &npts, 1, cv::Scalar(255), 8, 0, -roiRect.tl());

    // 计算统计量
    // countNonZero 获取内部面积
    int innerArea = cv::countNonZero(maskBuffer);
    if (innerArea == 0) return false;

    // 计算内部均值
    double innerMean = cv::mean(roiImg, maskBuffer)[0];

    // 数学推导计算外部均值
    double totalSum = cv::sum(roiImg)[0];
    double totalArea = (double)roiRect.area();
    double outerArea = totalArea - innerArea;

    if (outerArea <= 0) return false;

    double innerSum = innerMean * innerArea;
    double outerMean = (totalSum - innerSum) / outerArea;

    return (innerMean > outerMean + diff_thresh);
}


void filterPointsByGraySimilarity(const cv::Mat& grayImg,
    const std::vector<cv::Point2f>& centers,
    std::vector<cv::Point2f>& centers_filter)
{
    const int TARGET_COUNT = 14;
    centers_filter.clear();

    if (centers.size() <= TARGET_COUNT) {
        centers_filter = centers;
        return;
    }
    if (grayImg.empty() || grayImg.channels() != 1) {
        return;
    }

    // 优化：一次性分配内存
    std::vector<PointInfo> pointData;
    pointData.reserve(centers.size());

    const int cols = grayImg.cols;
    const int rows = grayImg.rows;
    // 获取原始指针以加速访问（仅在极其频繁调用时有显著差异，此处保持 safe 的 at 或 ptr 即可）
    // 为了极致性能，若保证坐标在范围内，可用 ptr
    for (const auto& pt : centers) {
        int x = cvRound(pt.x);
        int y = cvRound(pt.y);

        if (x >= 0 && x < cols && y >= 0 && y < rows) {
            int val = grayImg.at<uchar>(y, x);
            pointData.push_back({ val, pt });
        }
    }

    if (pointData.size() < TARGET_COUNT) {
        for (const auto& pd : pointData) centers_filter.push_back(pd.point);
        return;
    }

    std::sort(pointData.begin(), pointData.end(),
        [](const PointInfo& a, const PointInfo& b) {
            return a.grayValue < b.grayValue;
        });

    int min_range = std::numeric_limits<int>::max();
    int best_start_idx = 0;

    const int end_idx = static_cast<int>(pointData.size()) - TARGET_COUNT;
    for (int i = 0; i <= end_idx; ++i) {
        int range = pointData[i + TARGET_COUNT - 1].grayValue - pointData[i].grayValue;
        if (range < min_range) {
            min_range = range;
            best_start_idx = i;
        }
    }

    centers_filter.reserve(TARGET_COUNT);
    for (int i = 0; i < TARGET_COUNT; ++i) {
        centers_filter.push_back(pointData[best_start_idx + i].point);
    }
}


static std::vector<cv::Point2f> filterOutliers_NearestNeighbor(
    const std::vector<cv::Point2f>& points, int k)
{
    if (points.size() <= static_cast<size_t>(k)) return points;

    static const int threshold_lut[] = {
        1000, 10000, 40000, 160000, 250000, 250000, 250000,
        360000, 640000, 810000, 810000, 810000, 810000, 1000000
    };

    int dist_threshold = 1000;
    if (k > 0 && k <= 13) {
        dist_threshold = threshold_lut[k];
    }
    long dist_threshold_sq = (long)dist_threshold * dist_threshold;

    std::vector<cv::Point2f> filtered_points;
    filtered_points.reserve(points.size());

    // 优化：将临时向量移出循环，避免每次外层循环都重新分配内存
    std::vector<float> dists_sq;
    dists_sq.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        dists_sq.clear(); // 仅重置 size，不释放 capacity

        // 此处循环可以考虑 SIMD 优化，但编译器通常对这种简单的平方和做了优化
        for (size_t j = 0; j < points.size(); ++j) {
            if (i == j) continue;
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            dists_sq.push_back(dx * dx + dy * dy);
        }

        if (dists_sq.size() < static_cast<size_t>(k)) {
            filtered_points.push_back(points[i]);
            continue;
        }

        std::nth_element(dists_sq.begin(), dists_sq.begin() + k - 1, dists_sq.end());

        if (dists_sq[k - 1] <= dist_threshold_sq) {
            filtered_points.push_back(points[i]);
        }
    }
    return filtered_points;
}



// contour: 轮廓点集
// sample_step: 采样步长
cv::Point2f fitEllipseCenterSampled(const std::vector<cv::Point>& contour, int sample_step) {
    int total_points = static_cast<int>(contour.size());

    // 1. 采样 (Sampling)
    // 如果轮廓太短，无法采样，回退到普通计算或取均值
    if (total_points < 6) {
        return cv::minAreaRect(contour).center;
    }

    std::vector<cv::Point2f> sampled_points;
    sampled_points.reserve(total_points / sample_step + 1);

    for (int i = 0; i < total_points; i += sample_step) {
        sampled_points.push_back(cv::Point2f(static_cast<float>(contour[i].x), static_cast<float>(contour[i].y)));
    }

    // 保证至少有 5 个点才能拟合椭圆参数
    if (sampled_points.size() < 5) {
        return cv::fitEllipse(contour).center; // 样本太少回退到 OpenCV 方法
    }

    // 2. 构建最小二乘矩阵 (Direct Least Squares)
    // 目标方程 (公式 10): A*u^2 + 2*B*u*v + C*v^2 + 2*D*u + 2*E*v + 1 = 0
    // 这是一个线性方程组 M * X = Y
    // X = [A, B, C, D, E]^T
    // Y = [-1, -1, ..., -1]^T

    int m = static_cast<int>(sampled_points.size());
    cv::Mat M(m, 5, CV_32F);
    cv::Mat Y = cv::Mat::ones(m, 1, CV_32F) * -1.0f;

    for (int i = 0; i < m; ++i) {
        float u = sampled_points[i].x;
        float v = sampled_points[i].y;

        M.at<float>(i, 0) = u * u;       // u^2
        M.at<float>(i, 1) = 2 * u * v;   // 2uv
        M.at<float>(i, 2) = v * v;       // v^2
        M.at<float>(i, 3) = 2 * u;       // 2u
        M.at<float>(i, 4) = 2 * v;       // 2v
    }

    // 求解线性方程组 M * X = Y
    cv::Mat X;
    bool solved = cv::solve(M, Y, X, cv::DECOMP_SVD);

    if (!solved) {
        return cv::fitEllipse(contour).center; // 求解失败回退
    }

    float A = X.at<float>(0);
    float B = X.at<float>(1);
    float C = X.at<float>(2);
    float D = X.at<float>(3);
    float E = X.at<float>(4);

    // 3. 计算圆心 (公式 11)
    float denominator = 4 * A * C - B * B;

    // 防止分母为0 (虽然在椭圆拟合中很少见，但为了稳健性)
    if (std::abs(denominator) < 1e-6) {
        return cv::fitEllipse(contour).center;
    }

    float u0 = (B * E - 2 * C * D) / denominator;
    float v0 = (B * D - 2 * A * E) / denominator;

    return cv::Point2f(u0, v0);
}


bool ImageProcessor::extractReflectiveMarkers(const cv::Mat& image, std::vector<cv::Point2f>& centers, cv::Mat& result) {
    centers.clear();
    if (image.empty()) {
        std::cerr << "Error: Input image is empty for marker extraction." << std::endl;
        return false;
    }


    if (image.channels() == 3) {
        cv::cvtColor(image, gray_cache_, cv::COLOR_BGR2GRAY);
    }
    else {
        image.copyTo(gray_cache_); // 确保不修改原图
    }

    cv::Mat res_mean, res_gaussian, res_median, res_bilateral;

    // 2. 高斯模糊
    //cv::GaussianBlur(gray_cache_, blurred_cache_, cv::Size(9, 9), 2, 2);

    // ---------------------------------------------------------
    // 1. 均值滤波 (Mean Filtering / Box Filter)
    // ---------------------------------------------------------
    // 原理：用卷积核内像素的平均值代替中心像素。
    // 特点：最简单，但去噪同时会使图像变得很模糊。
    // Size(5, 5) 是卷积核大小
    blur(gray_cache_, res_mean, cv::Size(5, 5));

    // ---------------------------------------------------------
    // 2. 高斯滤波 (Gaussian Filtering)
    // ---------------------------------------------------------
    // 原理：根据高斯分布加权平均，中心权重高，边缘权重低。
    // 特点：比均值滤波更自然，能有效消除高斯噪声。
    // Size(5, 5) 是核大小, 0 表示标准差由核大小自动计算
    GaussianBlur(gray_cache_, res_gaussian, cv::Size(5, 5), 0);

    // ---------------------------------------------------------
    // 3. 中值滤波 (Median Filtering)
    // ---------------------------------------------------------
    // 原理：用卷积核内像素的中值代替中心像素。
    // 特点：对抗“椒盐噪声”（黑白噪点）极其有效，且能较好保留边缘。
    // 5 是核的大小（必须是奇数）
    medianBlur(gray_cache_, res_median, 5);

    // ---------------------------------------------------------
    // 4. 双边滤波 (Bilateral Filtering)
    // ---------------------------------------------------------
    // 原理：结合空间邻近度和像素值相似度。
    // 特点：著名的“保边去噪”滤波器。表面变光滑，但边缘依然清晰（磨皮算法基础）。
    // 参数说明：(输入, 输出, 邻域直径d, 颜色空间标准差, 坐标空间标准差)
    // d=9: 邻域直径; sigmaColor=75: 颜色差异多大算不同; sigmaSpace=75: 空间距离
    bilateralFilter(gray_cache_, res_bilateral, 5, 75, 75);


    // 3. 二值化
    cv::adaptiveThreshold(blurred_cache_, binary_cache_, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 17, -1);

    // 形态学处理 (getStructuringElement 内部有缓存，不需要 static)
    //cv::erode(binary_cache_, binary_cache_, element_kernel_);

    // 4. 寻找轮廓
    // RETR_EXTERNAL 比 LIST 快一点点，因为不建立层级
    cv::findContours(binary_cache_, contours_cache_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    result = image.clone(); // 这里的深拷贝是必须的，因为要输出结果图

    std::vector<cv::Point2f> centers_find;
    centers_find.reserve(contours_cache_.size());

    // --- 优化：将凸包和拟合曲线的容器移出循环 ---
    std::vector<cv::Point> hull;
    std::vector<cv::Point> approxCurve;

    cv::Mat debug_sampling_map = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat all_contours = cv::Mat::zeros(image.size(), CV_8UC3);

    cv::Mat contours_size = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat contours_circularity = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat contours_numVertices = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat contours_BrighterThanBackground = cv::Mat::zeros(image.size(), CV_8UC3);

    int a = 0;
    int b = 0;
    int c = 0;

    // 5. 遍历轮廓进行筛选
    for (size_t i = 0; i < contours_cache_.size(); i++) {
        const auto& contour = contours_cache_[i]; // 引用别名

        cv::drawContours(all_contours, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);

        // --- 级联筛选 ---
        double area = cv::contourArea(contour);
        // 使用成员变量 min_area_ / max_area_ 代替硬编码 (如果需要保持原逻辑不变，请改回 100/1000)
        if (area < min_area_ || area > max_area_) continue;
        cv::drawContours(contours_size, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);
        a++;

        double perimeter = cv::arcLength(contour, true);
        if (perimeter == 0) continue;

        double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
        if (circularity <= 0.7) continue;
        cv::drawContours(contours_circularity, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);

        cv::convexHull(contour, hull);
        double hullArea = cv::contourArea(hull);
        double solidity = (hullArea > 0) ? (area / hullArea) : 0;
        if (solidity <= 0.92) continue;

        double epsilon = 0.02 * perimeter;
        cv::approxPolyDP(contour, approxCurve, epsilon, true);
        int numVertices = static_cast<int>(approxCurve.size());
        if (numVertices <= 6 || numVertices >= 11) continue;
        cv::drawContours(contours_numVertices, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);
        b++;


        // --- 亮度检测 (关键优化) ---
        // 传入成员变量 mask_buffer_ 进行复用
        if (!isContourBrighterThanBackground(blurred_cache_, contour, mask_buffer_, 2, 5.0)) {
            continue;
        }
        cv::drawContours(contours_BrighterThanBackground, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);
        c++;

        // 1. 针对小轮廓的步长策略
        // ==========================================
        int total = static_cast<int>(contour.size());
        int step = 2; // 默认步长为 2 (最适合 size=40 左右的情况)

        // ==========================================
        // 2. 绘制 1x1 的像素点 (关键！)
        // ==========================================
        // 因为轮廓很小，如果画 2x2 的矩形，点就会粘在一起变成实线。
        // 所以这里必须画 1x1 的点。

        cv::drawContours(debug_sampling_map, contours_cache_, (int)i, cv::Scalar(0, 0, 255), 1);

        //for (size_t k = 0; k < total; k += step) {
        //    cv::Point p = contour[k];

        //    if (p.x >= 0 && p.x < debug_sampling_map.cols &&
        //        p.y >= 0 && p.y < debug_sampling_map.rows) {

        //        // 使用 at 直接修改单个像素，这是最精细的画法
        //        // 在黑色背景上画红色点
        //        debug_sampling_map.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0, 0, 255);
        //    }
        //}


        cv::Point2f center_sampled = fitEllipseCenterSampled(contour, step);

        centers_find.push_back(center_sampled);

        // 绘图
        cv::drawContours(result, contours_cache_, (int)i, cv::Scalar(0, 255, 0), 2);
        cv::circle(result, center_sampled, 3, cv::Scalar(0, 0, 255), -1); // 画出采样拟合的中心
    }

    // 后续筛选逻辑
    size_t found_count = centers_find.size();
    if (found_count < 14) return false;

    //cv::imwrite("output.jpg", result);


    // 如果正好14个，直接返回，避免后续不必要的拷贝和计算
    if (found_count == 14) {
        centers = std::move(centers_find); // 移动语义
        return true;
    }

    std::vector<cv::Point2f> centers_filter_1;
    filterPointsByGraySimilarity(blurred_cache_, centers_find, centers_filter_1);

    for (const auto& c : centers_filter_1) {
        cv::circle(result, c, 10, cv::Scalar(0, 0, 255), -1);
    }

    if (centers_filter_1.size() == 14) {
        centers = std::move(centers_filter_1);
        return true;
    }

    if (centers_filter_1.size() > 14) {
        int k = static_cast<int>(centers_filter_1.size()) - 14;
        std::vector<cv::Point2f> centers_filter_2 = filterOutliers_NearestNeighbor(centers_filter_1, k);

        for (const auto& c : centers_filter_2) {
            cv::circle(result, c, 30, cv::Scalar(0, 0, 255), -1);
        }

        if (centers_filter_2.size() == 14) {
            centers = std::move(centers_filter_2);
            return true;
        }
    }

    return false;
}






//bool ImageProcessor::extractReflectiveMarkers(const cv::Mat& image, std::vector<cv::Point2f>& centers, cv::Mat& result) {
//    centers.clear();
//    if (image.empty()) {
//        std::cerr << "Error: Input image is empty for marker extraction." << std::endl;
//        return false;
//    }
//
//
//    if (image.channels() == 3) {
//        cv::cvtColor(image, gray_cache_, cv::COLOR_BGR2GRAY);
//    }
//    else {
//        image.copyTo(gray_cache_); // 确保不修改原图
//    }
//
//    // 2. 高斯模糊
//    cv::GaussianBlur(gray_cache_, blurred_cache_, cv::Size(9, 9), 2, 2);
//
//    // 3. 二值化
//    cv::adaptiveThreshold(blurred_cache_, binary_cache_, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 17, -1);
//
//    // 形态学处理 (getStructuringElement 内部有缓存，不需要 static)
//    cv::erode(binary_cache_, binary_cache_, element_kernel_);
//
//    // 4. 寻找轮廓
//    // RETR_EXTERNAL 比 LIST 快一点点，因为不建立层级
//    cv::findContours(binary_cache_, contours_cache_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    result = image.clone(); // 这里的深拷贝是必须的，因为要输出结果图
//
//    std::vector<cv::Point2f> centers_find;
//    centers_find.reserve(contours_cache_.size());
//
//    // --- 优化：将凸包和拟合曲线的容器移出循环 ---
//    std::vector<cv::Point> hull;
//    std::vector<cv::Point> approxCurve;
//
//    // 5. 遍历轮廓进行筛选
//    for (size_t i = 0; i < contours_cache_.size(); i++) {
//        const auto& contour = contours_cache_[i]; // 引用别名
//
//        // --- 级联筛选 ---
//        double area = cv::contourArea(contour);
//        // 使用成员变量 min_area_ / max_area_ 代替硬编码 (如果需要保持原逻辑不变，请改回 100/1000)
//        if (area < min_area_ || area > max_area_) continue;
//
//        double perimeter = cv::arcLength(contour, true);
//        if (perimeter == 0) continue;
//
//        double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
//        if (circularity <= 0.7) continue;
//
//        cv::convexHull(contour, hull);
//        double hullArea = cv::contourArea(hull);
//        double solidity = (hullArea > 0) ? (area / hullArea) : 0;
//        if (solidity <= 0.92) continue;
//
//        double epsilon = 0.02 * perimeter;
//        cv::approxPolyDP(contour, approxCurve, epsilon, true);
//        int numVertices = static_cast<int>(approxCurve.size());
//        if (numVertices <= 6 || numVertices >= 11) continue;
//
//        // --- 亮度检测 (关键优化) ---
//        // 传入成员变量 mask_buffer_ 进行复用
//        if (!isContourBrighterThanBackground(blurred_cache_, contour, mask_buffer_, 2, 5.0)) {
//            continue;
//        }
//
//        // --- 提取 ---
//        cv::RotatedRect ellipse = cv::fitEllipse(contour);
//        centers_find.push_back(ellipse.center);
//
//        // 绘图
//        cv::drawContours(result, contours_cache_, (int)i, cv::Scalar(0, 255, 0), 2);
//        cv::circle(result, ellipse.center, 3, cv::Scalar(0, 0, 255), -1);
//    }
//
//    // 后续筛选逻辑
//    size_t found_count = centers_find.size();
//    if (found_count < 14) return false;
//
//    //cv::imwrite("output.jpg", result);
//
//    
//    // 如果正好14个，直接返回，避免后续不必要的拷贝和计算
//    if (found_count == 14) {
//        centers = std::move(centers_find); // 移动语义
//        return true;
//    }
//
//    std::vector<cv::Point2f> centers_filter_1;
//    filterPointsByGraySimilarity(blurred_cache_, centers_find, centers_filter_1);
//
//    for (const auto& c : centers_filter_1) {
//        cv::circle(result, c, 10, cv::Scalar(0, 0, 255), -1);
//    }
//
//    if (centers_filter_1.size() == 14) {
//        centers = std::move(centers_filter_1);
//        return true;
//    }
//
//    if (centers_filter_1.size() > 14) {
//        int k = static_cast<int>(centers_filter_1.size()) - 14;
//        std::vector<cv::Point2f> centers_filter_2 = filterOutliers_NearestNeighbor(centers_filter_1, k);
//
//        for (const auto& c : centers_filter_2) {
//            cv::circle(result, c, 30, cv::Scalar(0, 0, 255), -1);
//        }
//
//        if (centers_filter_2.size() == 14) {
//            centers = std::move(centers_filter_2);
//            return true;
//        }
//    }
//
//    return false;
//}


void ImageProcessor::drawPoints(cv::Mat& image, const std::vector<cv::Point2f>& points, const cv::Scalar& color, int radius, int thickness) const {
    for (const auto& p : points) {
        cv::circle(image, p, radius, color, thickness);
    }
}

void ImageProcessor::drawCircles(cv::Mat& image, const std::vector<cv::Point2f>& centers, float radius, const cv::Scalar& color, int thickness) const {
    int r = static_cast<int>(radius);
    for (const auto& c : centers) {
        cv::circle(image, c, r, color, thickness);
    }
}