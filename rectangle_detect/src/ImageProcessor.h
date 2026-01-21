#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessor {
public:
    ImageProcessor();

    // --- 核心处理函数 ---
    bool extractReflectiveMarkers(const cv::Mat& image, std::vector<cv::Point2f>& centers, cv::Mat& resultImage);


    // --- 辅助绘图函数 ---
    void drawPoints(cv::Mat& image, const std::vector<cv::Point2f>& points, const cv::Scalar& color = cv::Scalar(0, 255, 0), int radius = 5, int thickness = 2) const;
    void drawCircles(cv::Mat& image, const std::vector<cv::Point2f>& centers, float radius, const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 2) const;

    // --- 参数设置 ---
    void setThresholdParams(double minArea, double maxArea);

private:
    // --- 内存缓存池 (优化点：成员变量复用内存) ---
    cv::Mat gray_cache_;
    cv::Mat blurred_cache_; // [新增] 用于存储高斯模糊后的图像
    cv::Mat binary_cache_;
    cv::Mat mask_buffer_;   // [新增] 专用于亮度检测的遮罩缓存，极大减少内存碎片
    
    // [新增] 预先计算好的形态学核，避免每帧重复构造
    cv::Mat element_kernel_; 

    std::vector<std::vector<cv::Point>> contours_cache_;
    // hierarchy_cache_ 如果在 findContours 中使用 RETR_EXTERNAL 则不需要，
    // 但为了保持通用性可以保留，或者根据 cpp 逻辑移除。
    std::vector<cv::Vec4i> hierarchy_cache_; 

    // 算法参数
    double min_area_;
    double max_area_;
};

#endif // IMAGE_PROCESSOR_HPP