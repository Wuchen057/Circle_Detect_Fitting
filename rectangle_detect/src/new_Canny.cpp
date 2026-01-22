#include "ImageProcessor.h"
#include <iostream>
#include <opencv2/opencv.hpp> 
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits> 

// 构造函数：初始化一些固定参数
ImageProcessor::ImageProcessor()
    : min_area_(100.0), max_area_(1000.0)
{
    // 预先初始化形态学核
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
 */
bool isContourBrighterThanBackground(const cv::Mat& srcGray,
    const std::vector<cv::Point>& contour,
    cv::Mat& maskBuffer,
    int padding = 10,
    double diff_thresh = 10.0)
{
    cv::Rect bbox = cv::boundingRect(contour);

    // 扩大 ROI
    cv::Rect roiRect = (bbox + cv::Size(padding * 2, padding * 2)) - cv::Point(padding, padding);
    roiRect &= cv::Rect(0, 0, srcGray.cols, srcGray.rows);

    if (roiRect.empty()) return false;

    // 引用 ROI 数据
    cv::Mat roiImg = srcGray(roiRect);

    // 重新调整 buffer 大小
    maskBuffer.create(roiRect.size(), CV_8UC1);
    maskBuffer.setTo(0);

    // 绘制内部 Mask 
    const cv::Point* pts = contour.data();
    int npts = (int)contour.size();
    cv::fillPoly(maskBuffer, &pts, &npts, 1, cv::Scalar(255), 8, 0, -roiRect.tl());

    // 计算统计量
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

    std::vector<PointInfo> pointData;
    pointData.reserve(centers.size());

    const int cols = grayImg.cols;
    const int rows = grayImg.rows;

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

    std::vector<float> dists_sq;
    dists_sq.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        dists_sq.clear();

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

// 椭圆拟合采样函数
cv::Point2f fitEllipseCenterSampled(const std::vector<cv::Point>& contour, int sample_step) {
    int total_points = static_cast<int>(contour.size());

    if (total_points < 6) {
        return cv::minAreaRect(contour).center;
    }

    std::vector<cv::Point2f> sampled_points;
    sampled_points.reserve(total_points / sample_step + 1);

    for (int i = 0; i < total_points; i += sample_step) {
        sampled_points.push_back(cv::Point2f(static_cast<float>(contour[i].x), static_cast<float>(contour[i].y)));
    }

    if (sampled_points.size() < 5) {
        return cv::fitEllipse(contour).center;
    }

    int m = static_cast<int>(sampled_points.size());
    cv::Mat M(m, 5, CV_32F);
    cv::Mat Y = cv::Mat::ones(m, 1, CV_32F) * -1.0f;

    for (int i = 0; i < m; ++i) {
        float u = sampled_points[i].x;
        float v = sampled_points[i].y;

        M.at<float>(i, 0) = u * u;
        M.at<float>(i, 1) = 2 * u * v;
        M.at<float>(i, 2) = v * v;
        M.at<float>(i, 3) = 2 * u;
        M.at<float>(i, 4) = 2 * v;
    }

    cv::Mat X;
    bool solved = cv::solve(M, Y, X, cv::DECOMP_SVD);

    if (!solved) {
        return cv::fitEllipse(contour).center;
    }

    float A = X.at<float>(0);
    float B = X.at<float>(1);
    float C = X.at<float>(2);
    float D = X.at<float>(3);
    float E = X.at<float>(4);

    float denominator = 4 * A * C - B * B;

    if (std::abs(denominator) < 1e-6) {
        return cv::fitEllipse(contour).center;
    }

    float u0 = (B * E - 2 * C * D) / denominator;
    float v0 = (B * D - 2 * A * E) / denominator;

    return cv::Point2f(u0, v0);
}

// ----------------------------------------------------------------------------------
// 修改后的核心函数：使用 Canny 算子提取轮廓
// ----------------------------------------------------------------------------------
bool ImageProcessor::extractReflectiveMarkers(const cv::Mat& image, std::vector<cv::Point2f>& centers, cv::Mat& result) {
    centers.clear();
    if (image.empty()) {
        std::cerr << "Error: Input image is empty for marker extraction." << std::endl;
        return false;
    }

    // 1. 转灰度
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_cache_, cv::COLOR_BGR2GRAY);
    }
    else {
        image.copyTo(gray_cache_);
    }

    // 2. 高斯滤波 (Canny 算子对噪声非常敏感，强烈建议使用高斯模糊)
    // 之前被注释掉的代码这里需要启用，作为Canny的前置
    cv::GaussianBlur(gray_cache_, blurred_cache_, cv::Size(5, 5), 1.5, 1.5);

    // 3. Canny 边缘检测
    // 参数说明: 
    // threshold1 (50): 低阈值，低于此值的边缘会被抛弃
    // threshold2 (150): 高阈值，高于此值的边缘会被保留
    // 介于两者之间的，如果连接到强边缘则保留
    // 针对反光标记点（高对比度），通常梯度较大，可以适当调整这两个值
    cv::Canny(blurred_cache_, binary_cache_, 40, 50);

    // 4. 形态学闭运算 (Closing)
    // Canny 提取出的边缘有时会不闭合，使用闭运算(先膨胀后腐蚀)来连接断裂的边缘
    // 这样才能保证后续 findContours 能找到闭合的区域用于面积计算
    cv::morphologyEx(binary_cache_, binary_cache_, cv::MORPH_CLOSE, element_kernel_);

    // 5. 寻找轮廓
    cv::findContours(binary_cache_, contours_cache_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 准备结果图
    result = image.clone();

    std::vector<cv::Point2f> centers_find;
    centers_find.reserve(contours_cache_.size());

    // 辅助变量
    std::vector<cv::Point> hull;
    std::vector<cv::Point> approxCurve;

    // 用于调试的绘制图层（可选）
    // cv::Mat debug_layer = cv::Mat::zeros(image.size(), CV_8UC3);

    int a = 0;
    int b = 0;
    int c = 0;

    cv::Mat contours_size = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat contours_circularity = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat contours_numVertices = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::Mat contours_BrighterThanBackground = cv::Mat::zeros(image.size(), CV_8UC3);

    // 6. 遍历轮廓进行筛选
    for (size_t i = 0; i < contours_cache_.size(); i++) {
        const auto& contour = contours_cache_[i];

        // --- 级联筛选 ---

        // 面积筛选
        double area = cv::contourArea(contour);
        if (area < min_area_ || area > max_area_) continue;
        cv::drawContours(contours_size, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);
        a++;

        // 周长筛选
        double perimeter = cv::arcLength(contour, true);
        if (perimeter == 0) continue;

        // 圆度筛选
        double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
        if (circularity <= 0.7) continue;
        cv::drawContours(contours_circularity, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);

        // 凸包实心度筛选
        cv::convexHull(contour, hull);
        double hullArea = cv::contourArea(hull);
        double solidity = (hullArea > 0) ? (area / hullArea) : 0;
        if (solidity <= 0.92) continue;

        // 多边形拟合顶点数筛选
        double epsilon = 0.02 * perimeter;
        cv::approxPolyDP(contour, approxCurve, epsilon, true);
        int numVertices = static_cast<int>(approxCurve.size());
        if (numVertices <= 6 || numVertices >= 11) continue;
        cv::drawContours(contours_numVertices, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);
        b++;

        // --- 亮度检测 ---
        // 注意：这里仍然使用 blurred_cache_ (灰度图) 来判断亮度，而不是 binary_cache_ (边缘图)
        // Canny 得到的 contour 是边缘，fillPoly 会填充这个边缘内部，逻辑依然成立
        if (!isContourBrighterThanBackground(blurred_cache_, contour, mask_buffer_, 2, 5.0)) {
            continue;
        }
        cv::drawContours(contours_BrighterThanBackground, contours_cache_, (int)i, cv::Scalar(255, 255, 255), 1);
        c++;

        // --- 采样拟合中心 ---
        int step = 2; // 采样步长
        cv::Point2f center_sampled = fitEllipseCenterSampled(contour, step);

        centers_find.push_back(center_sampled);

        // 绘图
        cv::drawContours(result, contours_cache_, (int)i, cv::Scalar(0, 255, 0), 2);
        cv::circle(result, center_sampled, 3, cv::Scalar(0, 0, 255), -1);
    }

    // 后续筛选逻辑（保持不变）
    size_t found_count = centers_find.size();
    if (found_count < 14) return false;

    if (found_count == 14) {
        centers = std::move(centers_find);
        return true;
    }

    // 基于灰度相似度筛选
    std::vector<cv::Point2f> centers_filter_1;
    filterPointsByGraySimilarity(blurred_cache_, centers_find, centers_filter_1);

    for (const auto& c : centers_filter_1) {
        cv::circle(result, c, 10, cv::Scalar(0, 0, 255), -1);
    }

    if (centers_filter_1.size() == 14) {
        centers = std::move(centers_filter_1);
        return true;
    }

    // 基于空间距离筛选 (KNN Outlier Removal)
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